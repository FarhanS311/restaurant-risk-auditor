#!/usr/bin/env python3
"""
compiler.py — DSPy synthesizer for the Restaurant Health & Reputation Auditor.

Steps 5–9: Signature, judge metric, BootstrapFewShot compile, compare analysis.

Requires: OPENROUTER_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

from rag_engines import parallel_rag_strike

load_dotenv()

# ── Model config ─────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"

TEACHER_MODEL = os.getenv(
    "TEACHER_MODEL",
    "openrouter/google/gemini-2.0-flash-001",
)
STUDENT_MODEL = os.getenv(
    "STUDENT_MODEL",
    "openrouter/google/gemini-2.0-flash-lite-001",
)

OPTIMIZED_STATE_PATH = Path("optimized_auditor_state.json")

# ── Banned fluff phrases — instant disqualification ──────────────────
AUDITOR_FLUFF: frozenset[str] = frozenset({
    "synergy",
    "stakeholder",
    "paradigm shift",
    "best-in-class",
    "culinary journey",
    "premium experience",
    "brand alignment",
    "value proposition",
    "going forward",
    "circle back",
    "delightful dining experience",
    "world-class ambiance",
})


# ════════════════════════════════════════════════════════════════════
#  STEP 5A — DSPy Signature
# ════════════════════════════════════════════════════════════════════

class RestaurantRiskSignature(dspy.Signature):
    """
    You are a rigorous restaurant health and reputation auditor.
    You have been given three independent intelligence streams about a restaurant:
    structured inspection metrics from SQLite, unstructured review lore from FAISS,
    and an ownership/supplier relationship graph from Neo4j.

    Cross-reference ALL THREE streams before reaching a verdict. Inspection scores
    can look clean while reviews and ownership links tell a different story. Tagged
    MISS or ERROR strings in any stream are signals — reason about what is unknown,
    not only what is present.

    Output ONLY a raw JSON object — no markdown fences, no preamble. Schema:
    {
        "safe_or_risk": "SAFE" or "RISK",
        "risk_score": <float 0.0 to 10.0>,
        "the_receipts": "<one paragraph citing specific evidence from all three sources>"
    }

    risk_score: 0.0 = spotless record. 10.0 = shut-it-down territory.
    """

    sql_context: str = dspy.InputField(desc=(
        "Structured inspection facts from SQLite: latest grade, score 0-100, "
        "critical/general violation counts, repeat violation flag, reinspection status."
    ))
    faiss_context: str = dspy.InputField(desc=(
        "Unstructured review lore from FAISS: customer reviews, whistleblower posts, "
        "and local news snippets about health and reputation."
    ))
    graph_context: str = dspy.InputField(desc=(
        "Relationship map from Neo4j: ownership (OWNED_BY), franchise chains "
        "(PART_OF_CHAIN), supplier links (SUPPLIED_BY), and shared ownership contagion."
    ))

    verdict: str = dspy.OutputField(desc=(
        'Raw JSON: {"safe_or_risk": "SAFE"|"RISK", "risk_score": float, "the_receipts": str}'
    ))


# ════════════════════════════════════════════════════════════════════
#  STEP 5B — DSPy Module
# ════════════════════════════════════════════════════════════════════

class RestaurantRiskAnalyzer(dspy.Module):
    """ChainOfThought wrapper around RestaurantRiskSignature."""

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(RestaurantRiskSignature)

    def forward(
        self,
        sql_context: str,
        faiss_context: str,
        graph_context: str,
    ) -> dspy.Prediction:
        return self.analyze(
            sql_context=sql_context,
            faiss_context=faiss_context,
            graph_context=graph_context,
        )


# ════════════════════════════════════════════════════════════════════
#  JSON helpers
# ════════════════════════════════════════════════════════════════════

def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from raw verdict string; tolerates markdown fences."""
    clean = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip().strip("`")
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _validate_verdict_shape(data: Optional[Dict[str, Any]], min_receipts: int = 1) -> bool:
    """Syntactic validation of parsed verdict dict."""
    if data is None:
        return False

    required_keys = {"safe_or_risk", "risk_score", "the_receipts"}
    if not required_keys.issubset(data.keys()):
        return False

    if data["safe_or_risk"] not in ("SAFE", "RISK"):
        return False

    try:
        score = float(data["risk_score"])
    except (TypeError, ValueError):
        return False
    if not (0.0 <= score <= 10.0):
        return False

    receipts = str(data.get("the_receipts", ""))
    if len(receipts.strip()) < min_receipts:
        return False

    return True


# ════════════════════════════════════════════════════════════════════
#  STEP 6 — Deterministic Judge Metric
# ════════════════════════════════════════════════════════════════════

def restaurant_risk_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[Any] = None,
) -> bool:
    """
    Deterministic pass/fail judge. Returns True only when ALL conditions hold:
      1. verdict field present and non-empty
      2. No AUDITOR_FLUFF phrases in raw output
      3. Valid JSON with required keys
      4. safe_or_risk is SAFE or RISK
      5. risk_score in [0.0, 10.0]
      6. the_receipts > 30 chars
      7. Gold safe_or_risk label matches when provided on example
    """
    raw_verdict: str = getattr(prediction, "verdict", "") or ""
    if not raw_verdict.strip():
        return False

    lower_raw = raw_verdict.lower()
    for fluff in AUDITOR_FLUFF:
        if fluff.lower() in lower_raw:
            return False

    data = _extract_json(raw_verdict)
    if not _validate_verdict_shape(data, min_receipts=31):
        return False

    gold_label: Optional[str] = getattr(example, "safe_or_risk", None)
    if gold_label is not None and data["safe_or_risk"] != gold_label:
        return False

    return True


def test_judge_metric() -> bool:
    """Self-check: good prediction passes, deliberately bad prediction fails."""
    good_prediction = dspy.Prediction(
        verdict=(
            '{"safe_or_risk":"RISK","risk_score":8.5,"the_receipts":'
            '"Grade C inspection with 4 critical violations, food poisoning reviews, '
            'and Apex shell-corp ownership link via supplier BudgetProvisions."}'
        ),
    )
    good_example = dspy.Example(safe_or_risk="RISK")

    bad_prediction = dspy.Prediction(
        verdict=(
            '{"safe_or_risk":"SAFE","risk_score":1.0,"the_receipts":'
            '"A delightful dining experience with synergy across all stakeholders '
            'and premium experience throughout."}'
        ),
    )
    bad_example = dspy.Example(safe_or_risk="SAFE")

    good_ok = restaurant_risk_metric(good_example, good_prediction)
    bad_ok = restaurant_risk_metric(bad_example, bad_prediction)

    if good_ok:
        print("  GOOD prediction : JUDGE TEST PASSED")
    else:
        print("  GOOD prediction : JUDGE TEST FAILED (expected pass)")

    if not bad_ok:
        print("  BAD prediction  : JUDGE TEST PASSED (correctly rejected)")
    else:
        print("  BAD prediction  : JUDGE TEST FAILED (should have been rejected)")

    return good_ok and not bad_ok


def build_lm(model: str) -> dspy.LM:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return dspy.LM(
        model=model,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE,
        max_tokens=2048,
        temperature=0.1,
    )


# ════════════════════════════════════════════════════════════════════
#  STEP 7A — Training Set Builder
# ════════════════════════════════════════════════════════════════════

# Held-out entity for Step 8 compare (NOT in TRAINING_MANIFEST).
# Farm Table Kitchen: grade-A SQL, scandal lore in FAISS, shell-corp graph risk.
HOLDOUT_RESTAURANT = "Farm Table Kitchen"
HOLDOUT_GOLD_LABEL = "RISK"

# (restaurant_name, expected_safe_or_risk)
# 29 total: 14 SAFE, 15 RISK — Farm Table Kitchen held out for --compare

TRAINING_MANIFEST: List[Tuple[str, str]] = [
    # ── SAFE (14) — clean A/B records, low scandal ───────────────────
    ("Brasserie Moderne",         "SAFE"),
    ("Clean Eats Co-op",          "SAFE"),
    ("Baker's Hearth",            "SAFE"),
    ("Luna Gelato Bar",           "SAFE"),
    ("The Quiet Noodle",          "SAFE"),
    ("Hearth & Harvest",          "SAFE"),
    ("Artisan Coffee Roasters",   "SAFE"),
    ("Noble Steakhouse",          "SAFE"),
    ("Verde Salad Co.",           "SAFE"),
    ("The Ramen Lab",             "SAFE"),
    ("Pho Paradise",              "SAFE"),
    ("Mediterranean Mezze",       "SAFE"),
    ("Seoul Bowl House",          "SAFE"),
    ("The Daily Grind Café",      "SAFE"),
    ("El Patio Cantina",          "SAFE"),

    # ── RISK (15) — grade C, scandal lore, or graph contagion ────────
    ("Moldy Mike's Wing Factory", "RISK"),
    ("Harbor Catch & Cook",       "RISK"),
    ("The Greasy Spoon Depot",    "RISK"),
    ("Quick Bite Express #14",    "RISK"),
    ("Dragon Wok Alley",          "RISK"),
    ("Back Alley BBQ Pit",        "RISK"),
    ("Taco Libre Norte",          "RISK"),
    ("Pizza Pit Stop",            "RISK"),
    ("Luigi's Corner Bistro",     "RISK"),
    ("Bangkok Street Kitchen",    "RISK"),
    ("Burger Junction",           "RISK"),
    ("Mama Rosa's Trattoria",     "RISK"),
    ("Sakura Sushi Bar",          "RISK"),
    ("Curry House Express",       "RISK"),
]

TRAINING_NAMES: frozenset[str] = frozenset(name for name, _ in TRAINING_MANIFEST)


def build_training_set(
    manifest: List[Tuple[str, str]],
    max_workers: int = 6,
) -> List[dspy.Example]:
    """Pre-fetch RAG contexts for every restaurant; wrap as dspy.Example objects."""
    print(f"\n[TrainingSet] Fetching RAG contexts for {len(manifest)} restaurants...")

    restaurant_names = [name for name, _ in manifest]
    context_map: Dict[str, Dict[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_name = {
            pool.submit(parallel_rag_strike, name): name
            for name in restaurant_names
        }
        for i, future in enumerate(as_completed(future_to_name), 1):
            name = future_to_name[future]
            try:
                context_map[name] = future.result()
                print(f"  [{i:02d}/{len(manifest)}] ✓  {name}")
            except Exception as exc:
                print(f"  [{i:02d}/{len(manifest)}] ✗  {name}  ({exc})")
                context_map[name] = {
                    "sql_context":   f"[BUILD_ERROR] {exc}",
                    "faiss_context": f"[BUILD_ERROR] {exc}",
                    "graph_context": f"[BUILD_ERROR] {exc}",
                }

    examples: List[dspy.Example] = []
    for name, label in manifest:
        ctx = context_map.get(name, {})
        example = dspy.Example(
            sql_context=ctx.get("sql_context",   "[MISSING]"),
            faiss_context=ctx.get("faiss_context", "[MISSING]"),
            graph_context=ctx.get("graph_context", "[MISSING]"),
            safe_or_risk=label,
        ).with_inputs("sql_context", "faiss_context", "graph_context")
        examples.append(example)

    print(f"[TrainingSet] Built {len(examples)} examples.\n")
    return examples


# ════════════════════════════════════════════════════════════════════
#  STEP 7B — Optimization Loop
# ════════════════════════════════════════════════════════════════════

def run_optimization(
    trainset: List[dspy.Example],
    max_bootstrapped_demos: int = 6,
    max_labeled_demos: int = 4,
) -> RestaurantRiskAnalyzer:
    """BootstrapFewShot with teacher LM traces compiled into student."""
    teacher_lm = build_lm(TEACHER_MODEL)
    student_lm = build_lm(STUDENT_MODEL)

    print(f"[Optimizer] Teacher : {TEACHER_MODEL}")
    print(f"[Optimizer] Student : {STUDENT_MODEL}")
    print(f"[Optimizer] Bootstrapped demos : {max_bootstrapped_demos}")
    print(f"[Optimizer] Labeled demos      : {max_labeled_demos}\n")

    dspy.configure(lm=student_lm)

    optimizer = BootstrapFewShot(
        metric=restaurant_risk_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        teacher_settings={"lm": teacher_lm},
        max_errors=10,
    )

    student_program = RestaurantRiskAnalyzer()

    print("[Optimizer] Compiling... (this makes LLM calls — watch your token budget)")
    compiled_program: RestaurantRiskAnalyzer = optimizer.compile(
        student=student_program,
        trainset=trainset,
    )

    return compiled_program


def save_optimized_state(program: RestaurantRiskAnalyzer, path: Path) -> None:
    program.save(str(path))
    size_kb = path.stat().st_size / 1024
    print(f"\n[Save] Optimized state → {path}  ({size_kb:.1f} KB)")


def load_optimized_state(path: Path) -> RestaurantRiskAnalyzer:
    if not path.exists():
        raise FileNotFoundError(
            f"No optimized state found at '{path}'. Run compiler.py --optimize first."
        )
    program = RestaurantRiskAnalyzer()
    program.load(str(path))
    print(f"[Load] Loaded optimized state from {path}")
    return program


# ════════════════════════════════════════════════════════════════════
#  Inference helpers (shared by analyze + compare)
# ════════════════════════════════════════════════════════════════════

def _run_program_on_contexts(
    program: RestaurantRiskAnalyzer,
    contexts: Dict[str, str],
) -> dspy.Prediction:
    return program(
        sql_context=contexts["sql_context"],
        faiss_context=contexts["faiss_context"],
        graph_context=contexts["graph_context"],
    )


def _prediction_to_result(
    restaurant_name: str,
    contexts: Dict[str, str],
    prediction: dspy.Prediction,
    gold_label: Optional[str] = None,
) -> Dict[str, Any]:
    raw_verdict = prediction.verdict or ""
    data = _extract_json(raw_verdict) or {}

    example = dspy.Example(safe_or_risk=gold_label)
    passed_metric = restaurant_risk_metric(example, prediction)
    quality = score_verdict_quality(prediction)

    return {
        "restaurant":    restaurant_name,
        "safe_or_risk":  data.get("safe_or_risk", "PARSE_ERROR"),
        "risk_score":    data.get("risk_score", -1.0),
        "the_receipts":  data.get("the_receipts", raw_verdict),
        "metric_passed": passed_metric,
        "raw_reasoning": getattr(prediction, "reasoning", "") or "",
        "raw_verdict":   raw_verdict,
        "contexts":      contexts,
        "quality":       quality,
    }


_SQL_CITATION_WORDS = ("inspection", "grade", "score", "violation", "sqlite")
_FAISS_CITATION_WORDS = ("review", "lore", "news", "whistleblower", "faiss", "customer")
_GRAPH_CITATION_WORDS = ("owner", "chain", "supplier", "graph", "neo4j", "shell", "corp")


def _cites_all_streams(receipts: str) -> bool:
    lower = receipts.lower()
    return (
        any(w in lower for w in _SQL_CITATION_WORDS)
        and any(w in lower for w in _FAISS_CITATION_WORDS)
        and any(w in lower for w in _GRAPH_CITATION_WORDS)
    )


def score_verdict_quality(prediction: dspy.Prediction) -> Dict[str, Any]:
    """Deterministic quality checks for compare reports (no gold label)."""
    raw_verdict = getattr(prediction, "verdict", "") or ""
    data = _extract_json(raw_verdict)

    json_parse_ok = data is not None
    has_required_keys = bool(
        data and {"safe_or_risk", "risk_score", "the_receipts"}.issubset(data.keys())
    )
    label_valid = bool(data and data.get("safe_or_risk") in ("SAFE", "RISK"))

    score_in_range = False
    if data is not None:
        try:
            score = float(data["risk_score"])
            score_in_range = 0.0 <= score <= 10.0
        except (TypeError, ValueError, KeyError):
            pass

    receipts = str(data.get("the_receipts", "")) if data else ""
    receipts_len = len(receipts.strip())
    receipts_substantive = receipts_len > 30

    lower_raw = raw_verdict.lower()
    fluff_free = not any(fluff.lower() in lower_raw for fluff in AUDITOR_FLUFF)

    cites_all_streams = _cites_all_streams(receipts) if receipts else False

    structural_checks = [
        json_parse_ok,
        has_required_keys,
        label_valid,
        score_in_range,
        receipts_substantive,
        fluff_free,
        cites_all_streams,
    ]
    structure_score = sum(structural_checks)

    dummy_example = dspy.Example(safe_or_risk=None)
    metric_passed = restaurant_risk_metric(dummy_example, prediction)

    return {
        "json_parse_ok":        json_parse_ok,
        "has_required_keys":    has_required_keys,
        "label_valid":          label_valid,
        "score_in_range":       score_in_range,
        "receipts_len":         receipts_len,
        "receipts_substantive": receipts_substantive,
        "fluff_free":           fluff_free,
        "cites_all_streams":    cites_all_streams,
        "metric_passed":        metric_passed,
        "structure_score":      structure_score,
    }


# ════════════════════════════════════════════════════════════════════
#  STEP 8 — Compare compiled vs uncompiled baseline
# ════════════════════════════════════════════════════════════════════

def compare_compiled_vs_baseline(
    restaurant_name: str,
    gold_label: Optional[str] = HOLDOUT_GOLD_LABEL,
) -> Dict[str, Any]:
    """Run baseline and compiled programs on the same RAG contexts."""
    if restaurant_name in TRAINING_NAMES:
        raise ValueError(
            f"'{restaurant_name}' is in TRAINING_MANIFEST — pick a held-out entity "
            f"(default: {HOLDOUT_RESTAURANT})."
        )
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY required for --compare")

    student_lm = build_lm(STUDENT_MODEL)
    dspy.configure(lm=student_lm)

    print(f"\n[Compare] Firing RAG strike for '{restaurant_name}' (shared contexts)...")
    contexts = parallel_rag_strike(restaurant_name)

    print("[Compare] Running uncompiled baseline...")
    baseline_pred = _run_program_on_contexts(RestaurantRiskAnalyzer(), contexts)

    print("[Compare] Running compiled program...")
    compiled_program = load_optimized_state(OPTIMIZED_STATE_PATH)
    compiled_pred = _run_program_on_contexts(compiled_program, contexts)

    baseline = _prediction_to_result(restaurant_name, contexts, baseline_pred, gold_label)
    compiled = _prediction_to_result(restaurant_name, contexts, compiled_pred, gold_label)

    b_score = baseline["quality"]["structure_score"]
    c_score = compiled["quality"]["structure_score"]
    b_metric = baseline["quality"]["metric_passed"]
    c_metric = compiled["quality"]["metric_passed"]

    return {
        "restaurant": restaurant_name,
        "gold_label": gold_label,
        "baseline": baseline,
        "compiled": compiled,
        "delta": {
            "structure_score": c_score - b_score,
            "metric_passed": int(c_metric) - int(b_metric),
        },
        "compile_wins": c_score >= b_score and c_metric >= b_metric,
    }


def print_compare_report(result: Dict[str, Any]) -> bool:
    """Print side-by-side compare table. Returns True if compile wins."""
    baseline = result["baseline"]
    compiled = result["compiled"]
    bq = baseline["quality"]
    cq = compiled["quality"]

    print(f"\n{'═' * 72}")
    print(f"  COMPILE COMPARE: {result['restaurant']}")
    if result.get("gold_label"):
        print(f"  Gold label (report only): {result['gold_label']}")
    print(f"{'═' * 72}")

    rows = [
        ("json_parse_ok",        bq["json_parse_ok"],        cq["json_parse_ok"]),
        ("has_required_keys",    bq["has_required_keys"],    cq["has_required_keys"]),
        ("label_valid",          bq["label_valid"],          cq["label_valid"]),
        ("score_in_range",       bq["score_in_range"],       cq["score_in_range"]),
        ("receipts_substantive", bq["receipts_substantive"], cq["receipts_substantive"]),
        ("fluff_free",           bq["fluff_free"],           cq["fluff_free"]),
        ("cites_all_streams",    bq["cites_all_streams"],    cq["cites_all_streams"]),
        ("structure_score",      bq["structure_score"],      cq["structure_score"]),
        ("metric_passed",        bq["metric_passed"],        cq["metric_passed"]),
    ]

    print(f"\n  {'Check':<22} {'Baseline':>12} {'Compiled':>12}")
    print(f"  {'-' * 48}")
    for name, b_val, c_val in rows:
        if isinstance(b_val, bool):
            b_str = "PASS" if b_val else "FAIL"
            c_str = "PASS" if c_val else "FAIL"
        else:
            b_str = str(b_val)
            c_str = str(c_val)
        print(f"  {name:<22} {b_str:>12} {c_str:>12}")

    print(f"\n  {'Field':<22} {'Baseline':>12} {'Compiled':>12}")
    print(f"  {'-' * 48}")
    print(f"  {'safe_or_risk':<22} {str(baseline['safe_or_risk']):>12} {str(compiled['safe_or_risk']):>12}")
    print(f"  {'risk_score':<22} {str(baseline['risk_score']):>12} {str(compiled['risk_score']):>12}")

    print(f"\n  BASELINE RECEIPTS (truncated):")
    print(f"  {str(baseline['the_receipts'])[:400]}...")
    print(f"\n  COMPILED RECEIPTS (truncated):")
    print(f"  {str(compiled['the_receipts'])[:400]}...")

    b_metric = "PASS" if bq["metric_passed"] else "FAIL"
    c_metric = "PASS" if cq["metric_passed"] else "FAIL"
    summary = (
        f"COMPILE WINS: structure {bq['structure_score']}/7 → {cq['structure_score']}/7, "
        f"metric {b_metric} → {c_metric}"
    )
    if result["compile_wins"]:
        print(f"\n  {summary}")
    else:
        print(f"\n  COMPILE DID NOT WIN: structure {bq['structure_score']}/7 vs "
              f"{cq['structure_score']}/7, metric {b_metric} vs {c_metric}")

    return result["compile_wins"]


# ════════════════════════════════════════════════════════════════════
#  Public API — single-restaurant analysis
# ════════════════════════════════════════════════════════════════════

def analyze_restaurant(
    restaurant_name: str,
    program: Optional[RestaurantRiskAnalyzer] = None,
) -> Dict[str, Any]:
    """Full pipeline: parallel RAG → DSPy synthesizer → parsed verdict."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY required to run analysis.")

    student_lm = build_lm(STUDENT_MODEL)
    dspy.configure(lm=student_lm)

    if program is None:
        if OPTIMIZED_STATE_PATH.exists():
            program = load_optimized_state(OPTIMIZED_STATE_PATH)
        else:
            print("[Analyze] No compiled state found — running with raw student model.")
            program = RestaurantRiskAnalyzer()

    print(f"\n[Analyze] Firing RAG strike for '{restaurant_name}'...")
    contexts = parallel_rag_strike(restaurant_name)

    print("[Analyze] Running ChainOfThought synthesizer...")
    prediction = _run_program_on_contexts(program, contexts)

    result = _prediction_to_result(restaurant_name, contexts, prediction)
    del result["raw_verdict"]
    del result["quality"]
    return result


def print_verdict(result: Dict[str, Any]) -> None:
    verdict = result["safe_or_risk"]
    metric_tag = "PASS" if result["metric_passed"] else "FAIL"

    print(f"\n{'═' * 64}")
    print(f"  VERDICT FOR: {result['restaurant']}")
    print(f"{'═' * 64}")
    print(f"  Safe/Risk     : {verdict}")
    print(f"  Risk Score    : {result['risk_score']}")
    print(f"  Metric        : {metric_tag}")
    print(f"\n  THE RECEIPTS:")
    print(f"  {result['the_receipts']}")
    if result["raw_reasoning"]:
        print(f"\n  CHAIN OF THOUGHT (truncated):")
        preview = result["raw_reasoning"][:500].replace("\n", " ")
        print(f"  {preview}...")


# ════════════════════════════════════════════════════════════════════
#  STEP 5 — Smoke test (uncompiled module)
# ════════════════════════════════════════════════════════════════════

def run_smoke_test(restaurant_name: str) -> bool:
    """Fires parallel RAG, runs uncompiled module, validates JSON shape."""
    if not OPENROUTER_API_KEY:
        print(
            "\n[ERROR] OPENROUTER_API_KEY is not set in .env\n"
            "Add it like this:\n\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n\n"
            "Get a key at https://openrouter.ai/keys"
        )
        return False

    print(f"\n[Smoke] Student model : {STUDENT_MODEL}")
    dspy.configure(lm=build_lm(STUDENT_MODEL))

    print(f"\n[Smoke] Firing parallel RAG for '{restaurant_name}'...")
    contexts = parallel_rag_strike(restaurant_name)

    print("[Smoke] Running uncompiled ChainOfThought synthesizer...")
    program = RestaurantRiskAnalyzer()
    prediction = program(
        sql_context=contexts["sql_context"],
        faiss_context=contexts["faiss_context"],
        graph_context=contexts["graph_context"],
    )

    raw_verdict = prediction.verdict or ""
    data = _extract_json(raw_verdict)

    reasoning = getattr(prediction, "reasoning", "") or ""
    if reasoning:
        preview = reasoning[:500].replace("\n", " ")
        print(f"\n── REASONING (truncated) ──\n  {preview}...")

    print(f"\n── RAW VERDICT ──\n{raw_verdict}")
    print(f"\n── PARSED ──\n{json.dumps(data, indent=2) if data else 'PARSE_FAILED'}")

    passed = _validate_verdict_shape(data)
    if passed:
        print("\nSMOKE TEST PASSED — syntactically valid structured output")
    else:
        print("\nSMOKE TEST FAILED — verdict missing required fields or invalid types")

    return passed


# ════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restaurant Risk Analyzer — DSPy synthesizer"
    )
    parser.add_argument(
        "--test-judge",
        action="store_true",
        help="Run judge metric self-check (good pass, bad fail) — no API key needed",
    )
    parser.add_argument(
        "--smoke",
        metavar="RESTAURANT_NAME",
        nargs="?",
        const="Moldy Mike's Wing Factory",
        default=None,
        help="Run uncompiled smoke test on a restaurant",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Build training set and compile with BootstrapFewShot",
    )
    parser.add_argument(
        "--analyze",
        metavar="RESTAURANT_NAME",
        help="Analyze one restaurant using compiled (or raw) student model",
    )
    parser.add_argument(
        "--max-bootstrapped",
        type=int,
        default=6,
        help="Max bootstrapped demos for BootstrapFewShot (default: 6)",
    )
    parser.add_argument(
        "--max-labeled",
        type=int,
        default=4,
        help="Max labeled demos for BootstrapFewShot (default: 4)",
    )
    parser.add_argument(
        "--compare",
        metavar="RESTAURANT_NAME",
        nargs="?",
        const=HOLDOUT_RESTAURANT,
        default=None,
        help=f"Compare uncompiled vs compiled on held-out entity (default: {HOLDOUT_RESTAURANT})",
    )
    args = parser.parse_args()

    if not any([
        args.test_judge,
        args.smoke is not None,
        args.optimize,
        args.analyze,
        args.compare is not None,
    ]):
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python compiler.py --test-judge\n"
            '  python compiler.py --smoke "Moldy Mike\'s Wing Factory"\n'
            "  python compiler.py --optimize\n"
            '  python compiler.py --analyze "Moldy Mike\'s Wing Factory"\n'
            "  python compiler.py --compare\n"
        )
        sys.exit(0)

    if args.test_judge:
        print(f"\n{'═' * 64}")
        print("  JUDGE METRIC SELF-CHECK")
        print("═" * 64)
        ok = test_judge_metric()
        if not ok:
            sys.exit(1)

    if args.smoke is not None:
        ok = run_smoke_test(args.smoke)
        if not ok:
            sys.exit(1)

    if args.optimize:
        if not OPENROUTER_API_KEY:
            print("\n[ERROR] OPENROUTER_API_KEY required for --optimize")
            sys.exit(1)
        print("\n[Mode] OPTIMIZATION RUN")
        trainset = build_training_set(TRAINING_MANIFEST)
        compiled = run_optimization(
            trainset,
            max_bootstrapped_demos=args.max_bootstrapped,
            max_labeled_demos=args.max_labeled,
        )
        save_optimized_state(compiled, OPTIMIZED_STATE_PATH)
        print("\n[Validation] Post-compile check on 'Moldy Mike's Wing Factory'...")
        result = analyze_restaurant("Moldy Mike's Wing Factory", program=compiled)
        print_verdict(result)

    if args.analyze:
        if not OPENROUTER_API_KEY:
            print("\n[ERROR] OPENROUTER_API_KEY required for --analyze")
            sys.exit(1)
        result = analyze_restaurant(args.analyze)
        print_verdict(result)

    if args.compare is not None:
        if not OPENROUTER_API_KEY:
            print("\n[ERROR] OPENROUTER_API_KEY required for --compare")
            sys.exit(1)
        if not OPTIMIZED_STATE_PATH.exists():
            print(f"\n[ERROR] No optimized state at '{OPTIMIZED_STATE_PATH}'. Run --optimize first.")
            sys.exit(1)
        try:
            compare_result = compare_compiled_vs_baseline(args.compare)
        except ValueError as exc:
            print(f"\n[ERROR] {exc}")
            sys.exit(1)
        if not print_compare_report(compare_result):
            sys.exit(1)


if __name__ == "__main__":
    main()
