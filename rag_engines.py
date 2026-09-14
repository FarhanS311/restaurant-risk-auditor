#!/usr/bin/env python3
"""
rag_engines.py — Independent retrieval engines for the Restaurant Auditor.
Three standalone query functions. Error-as-Information. No routing logic.
"""

from __future__ import annotations

import os
import pickle
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Paths & config ───────────────────────────────────────────────────
SQLITE_PATH = Path("sandbox_data/restaurants.db")
FAISS_PATH  = Path("sandbox_data/reviews.faiss")
META_PATH   = Path("sandbox_data/reviews_meta.pkl")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5
RRF_K       = 60

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "auditorpassword")

# ── Lazy singletons ────────────────────────────────────────────────────
_faiss_index: faiss.Index | None = None
_faiss_meta:  List[Dict]  | None = None
_embed_model: SentenceTransformer | None = None
_bm25_index:  BM25Okapi | None = None
_bm25_corpus: List[List[str]] | None = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _load_faiss_resources() -> tuple[faiss.Index, List[Dict], SentenceTransformer, BM25Okapi]:
    global _faiss_index, _faiss_meta, _embed_model, _bm25_index, _bm25_corpus

    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(FAISS_PATH))
    if _faiss_meta is None:
        with open(META_PATH, "rb") as fh:
            _faiss_meta = pickle.load(fh)
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    if _bm25_index is None:
        _bm25_corpus = [_tokenize(chunk["text"]) for chunk in _faiss_meta]
        _bm25_index = BM25Okapi(_bm25_corpus)

    return _faiss_index, _faiss_meta, _embed_model, _bm25_index


def _rrf_fuse(rank_lists: List[List[int]]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranked in rank_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _resolve_restaurant_name(conn: sqlite3.Connection, restaurant_name: str) -> sqlite3.Row | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM restaurants WHERE name = ? COLLATE NOCASE",
        (restaurant_name,),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM restaurants WHERE name LIKE ? COLLATE NOCASE LIMIT 1",
            (f"%{restaurant_name}%",),
        ).fetchone()
    return row


# ════════════════════════════════════════════════════════════════════
#  ENGINE 1 — SQLite (The Math)
# ════════════════════════════════════════════════════════════════════

def query_sql(restaurant_name: str) -> str:
    """Pull latest inspection stats for restaurant_name from SQLite."""
    try:
        if not SQLITE_PATH.exists():
            return (
                f"[SQLITE_ERROR] Database file not found at '{SQLITE_PATH}'. "
                "Run generate_sandbox.py first."
            )

        with sqlite3.connect(SQLITE_PATH) as conn:
            restaurant = _resolve_restaurant_name(conn, restaurant_name)
            if restaurant is None:
                return (
                    f"[SQLITE_MISS] No restaurant matching '{restaurant_name}' exists "
                    "in the database. Either a brand-new venue with zero inspection "
                    "history, or a name not in the sandbox."
                )

            inspection = conn.execute(
                """
                SELECT * FROM inspections
                WHERE restaurant_id = ?
                ORDER BY inspection_date DESC
                LIMIT 1
                """,
                (restaurant["restaurant_id"],),
            ).fetchone()

        if inspection is None:
            return (
                f"[SQLITE_MISS] Restaurant '{restaurant['name']}' exists but has no "
                "inspection records on file."
            )

        repeat_flag = "yes" if inspection["repeat_violation_flag"] else "no"
        reinspection = (
            "yes" if inspection["passed_reinspection"] == 1
            else "no" if inspection["passed_reinspection"] == 0
            else "n/a"
        )

        return (
            f"[SQLITE_STATS] {restaurant['name']} "
            f"({restaurant['cuisine_type']}, {restaurant['borough']})\n"
            f"  Latest inspection : {inspection['inspection_date']}  |  "
            f"Grade {inspection['grade']}  |  Score {inspection['score']}/100\n"
            f"  Critical violations: {inspection['critical_violations']}  |  "
            f"General: {inspection['general_violations']}  |  "
            f"Repeat: {repeat_flag}\n"
            f"  Reinspection passed : {reinspection}\n"
            f"  Seating capacity    : {restaurant['seating_capacity']}  |  "
            f"Years open: {restaurant['years_in_operation']}"
        )

    except Exception as exc:
        return f"[SQLITE_ERROR] Unhandled exception querying '{restaurant_name}': {exc}"


# ════════════════════════════════════════════════════════════════════
#  ENGINE 2 — FAISS (The Lore) — BM25 + dense + RRF hybrid
# ════════════════════════════════════════════════════════════════════

def query_faiss(restaurant_name: str) -> str:
    """Retrieve review chunks via metadata match + BM25/dense RRF fusion."""
    try:
        if not FAISS_PATH.exists() or not META_PATH.exists():
            return (
                f"[FAISS_ERROR] Index files not found. "
                "Run generate_sandbox.py first."
            )

        index, meta, model, bm25 = _load_faiss_resources()
        query_text = (
            f"{restaurant_name} restaurant health safety reviews "
            "violations food poisoning"
        )
        query_tokens = _tokenize(query_text)

        # Entity-scoped pool: only chunks tagged to this restaurant
        entity_indices = [
            i for i, chunk in enumerate(meta)
            if chunk["restaurant"].lower() == restaurant_name.lower()
        ]

        if not entity_indices:
            return (
                f"[FAISS_MISS] No review corpus entries found for '{restaurant_name}'. "
                "Either a clean reputation or no indexed public commentary yet."
            )

        entity_set = set(entity_indices)

        # BM25 rank within entity pool
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_ranked = sorted(
            entity_indices,
            key=lambda i: bm25_scores[i],
            reverse=True,
        )

        # Dense rank — search full index, keep entity matches only
        query_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
        ).astype(np.float32)
        _, dense_indices = index.search(query_vec, len(meta))
        dense_ranked = [
            int(i) for i in dense_indices[0]
            if i >= 0 and int(i) in entity_set
        ]

        # Metadata list preserves insertion order as baseline rank
        fused_scores = _rrf_fuse([entity_indices, bm25_ranked, dense_ranked])
        ranked_indices = sorted(
            fused_scores.keys(),
            key=lambda i: fused_scores[i],
            reverse=True,
        )[:TOP_K]

        rrf_count = len(ranked_indices)

        blocks: List[str] = []
        for i, idx in enumerate(ranked_indices, 1):
            chunk = meta[idx]
            rating = chunk.get("rating")
            rating_str = str(rating) if rating is not None else "n/a"
            blocks.append(
                f"  [{i}] source={chunk['source']} | type={chunk['type']} | "
                f"rating={rating_str} | rrf={fused_scores[idx]:.4f}\n"
                f"  {chunk['text']}"
            )

        header = (
            f"[FAISS_LORE] {len(ranked_indices)} review chunk(s) for "
            f"'{restaurant_name}' ({len(entity_indices)} entity-tagged, "
            f"ranked via BM25+dense RRF):"
        )
        return header + "\n\n" + "\n\n".join(blocks)

    except Exception as exc:
        return f"[FAISS_ERROR] Unhandled exception querying '{restaurant_name}': {exc}"


# ════════════════════════════════════════════════════════════════════
#  ENGINE 3 — Neo4j (The Web)
# ════════════════════════════════════════════════════════════════════

def _node_type(labels: List[str]) -> str:
    for label in labels:
        if label != "AuditorNode":
            return label
    return "Node"


def query_graph(restaurant_name: str) -> str:
    """Pull ownership, chain, and supplier relationships from Neo4j."""
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

        with driver.session() as session:
            out_rows = session.run(
                """
                MATCH (a:AuditorNode:Restaurant {name: $name})-[r]->(b:AuditorNode)
                RETURN type(r) AS rel, b.name AS target,
                       labels(b) AS target_labels, properties(r) AS props
                ORDER BY type(r)
                """,
                name=restaurant_name,
            ).data()

            in_rows = session.run(
                """
                MATCH (a:AuditorNode)-[r]->(b:AuditorNode:Restaurant {name: $name})
                RETURN type(r) AS rel, a.name AS source,
                       labels(a) AS source_labels, properties(r) AS props
                ORDER BY type(r)
                """,
                name=restaurant_name,
            ).data()

            supplier_risk_rows = session.run(
                """
                MATCH (a:AuditorNode:Restaurant {name: $name})-[:SUPPLIED_BY]->(s:AuditorNode:Supplier)
                WHERE s.recall_count >= 2
                RETURN s.name AS supplier, s.recall_count AS recalls, s.category AS category
                ORDER BY s.recall_count DESC
                """,
                name=restaurant_name,
            ).data()

            shared_owner_rows = session.run(
                """
                MATCH (a:AuditorNode:Restaurant {name: $name})-[:SHARED_OWNER_WITH]->(peer:AuditorNode:Restaurant)
                MATCH (peer)-[:OWNED_BY]->(owner:AuditorNode:Owner)
                RETURN peer.name AS peer_restaurant, owner.name AS owner_name,
                       owner.entity_type AS entity_type
                ORDER BY peer.name
                """,
                name=restaurant_name,
            ).data()

        if not out_rows and not in_rows and not supplier_risk_rows and not shared_owner_rows:
            return (
                f"[NEO4J_MISS] '{restaurant_name}' has zero mapped relationships "
                "in the graph. No ownership links, chain membership, or supplier "
                "connections on file."
            )

        lines: List[str] = [f"[NEO4J_GRAPH] Relationship map for '{restaurant_name}':"]

        if out_rows:
            lines.append("  — Outgoing —")
            for row in out_rows:
                target_type = _node_type(row["target_labels"])
                props = {k: v for k, v in row["props"].items() if v is not None}
                prop_str = "  |  ".join(f"{k}: {v}" for k, v in props.items())
                lines.append(
                    f"    [{row['rel']}] → {row['target']} ({target_type})"
                    + (f"\n        {prop_str}" if prop_str else "")
                )

        if in_rows:
            lines.append("  — Incoming —")
            for row in in_rows:
                source_type = _node_type(row["source_labels"])
                props = {k: v for k, v in row["props"].items() if v is not None}
                prop_str = "  |  ".join(f"{k}: {v}" for k, v in props.items())
                lines.append(
                    f"    [{row['rel']}] ← {row['source']} ({source_type})"
                    + (f"\n        {prop_str}" if prop_str else "")
                )

        if supplier_risk_rows:
            lines.append("  — High-Recall Supplier Links —")
            for row in supplier_risk_rows:
                lines.append(
                    f"    SUPPLIED_BY → {row['supplier']} ({row['category']}) "
                    f"| recall_count: {row['recalls']}"
                )

        if shared_owner_rows:
            lines.append("  — Shared Ownership Contagion —")
            for row in shared_owner_rows:
                lines.append(
                    f"    SHARED_OWNER_WITH → {row['peer_restaurant']} "
                    f"(owned by {row['owner_name']}, {row['entity_type']})"
                )

        return "\n".join(lines)

    except Exception as exc:
        return f"[NEO4J_ERROR] Unhandled exception querying '{restaurant_name}': {exc}"

    finally:
        if driver:
            driver.close()


# ════════════════════════════════════════════════════════════════════
#  PARALLEL STRIKE — fires all three engines simultaneously
# ════════════════════════════════════════════════════════════════════

def parallel_rag_strike(restaurant_name: str) -> Dict[str, str]:
    """
    Submit query_sql, query_faiss, query_graph to a thread pool and collect results.
    Each engine is independent — a failure in one does not block the others.

    Returns:
        {
            "sql_context":   str,
            "faiss_context": str,
            "graph_context": str,
        }
    """
    task_map = {
        "sql_context":   query_sql,
        "faiss_context": query_faiss,
        "graph_context": query_graph,
    }

    results: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_key = {
            pool.submit(fn, restaurant_name): key
            for key, fn in task_map.items()
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = f"[EXECUTOR_ERROR] {key} thread crashed: {exc}"

    return results


def _latency_benchmark(restaurant_name: str) -> None:
    """Prove parallel latency tracks the slowest engine, not the sum."""
    print(f"\n{'═' * 64}")
    print("  LATENCY BENCHMARK")
    print("═" * 64)

    t0 = time.perf_counter()
    query_sql(restaurant_name)
    t_sql = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_faiss(restaurant_name)
    t_faiss = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_graph(restaurant_name)
    t_graph = time.perf_counter() - t0

    slowest = max(t_sql, t_faiss, t_graph)

    t0 = time.perf_counter()
    query_sql(restaurant_name)
    query_faiss(restaurant_name)
    query_graph(restaurant_name)
    t_sequential = time.perf_counter() - t0

    t0 = time.perf_counter()
    parallel_rag_strike(restaurant_name)
    t_parallel = time.perf_counter() - t0

    print(f"  Engine timings : sql={t_sql:.3f}s  faiss={t_faiss:.3f}s  "
          f"graph={t_graph:.3f}s  slowest={slowest:.3f}s")
    print(f"  Sequential     : {t_sequential:.3f}s")
    print(f"  Parallel       : {t_parallel:.3f}s")

    if t_parallel < t_sequential * 0.9:
        print("  PASS — parallel faster than sequential (near slowest engine, not sum)")
    else:
        print("  NOTE — parallel did not beat sequential threshold; engines may be too fast to measure")


def _fault_isolation_demo(restaurant_name: str) -> None:
    """Check SQL+FAISS succeed when graph is down; skip if Neo4j is up."""
    print(f"\n{'═' * 64}")
    print("  FAULT ISOLATION DEMO")
    print("═" * 64)

    ctx = parallel_rag_strike(restaurant_name)
    sql_ok = ctx.get("sql_context", "").startswith("[SQLITE_STATS]")
    faiss_ok = ctx.get("faiss_context", "").startswith("[FAISS_LORE]")
    graph_err = ctx.get("graph_context", "").startswith("[NEO4J_ERROR]")
    graph_ok = ctx.get("graph_context", "").startswith("[NEO4J_GRAPH]")

    if sql_ok and faiss_ok and graph_err:
        print("  FAULT TEST PASSED (Neo4j down — SQL + FAISS returned, graph tagged error)")
    elif sql_ok and faiss_ok and graph_ok:
        print("  FAULT TEST SKIPPED (Neo4j is up — run: docker stop neo4j-auditor)")
    else:
        print("  FAULT TEST UNEXPECTED:")
        print(f"    sql_context   : {ctx.get('sql_context', '')[:80]}...")
        print(f"    faiss_context : {ctx.get('faiss_context', '')[:80]}...")
        print(f"    graph_context : {ctx.get('graph_context', '')[:80]}...")


# ════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_RESTAURANTS = [
        "Moldy Mike's Wing Factory",
        "Farm Table Kitchen",
        "Brasserie Moderne",
        "NonExistent Restaurant",
    ]

    for name in TEST_RESTAURANTS:
        print(f"\n{'═' * 64}")
        print(f"  PARALLEL RAG STRIKE  →  {name}")
        print("═" * 64)
        ctx = parallel_rag_strike(name)
        for key in ("sql_context", "faiss_context", "graph_context"):
            print(f"\n── {key.upper()} ──")
            print(ctx[key])

    _latency_benchmark("Moldy Mike's Wing Factory")
    _fault_isolation_demo("Moldy Mike's Wing Factory")
