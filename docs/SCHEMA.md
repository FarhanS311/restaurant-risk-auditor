# Restaurant Health & Reputation Auditor — Schema Note

**Step 1 deliverable.** Defines the three-store data model before any generation or retrieval code is written.

---

## Core Entity

**Primary entity: `Restaurant`**

| Concept | Value |
|---|---|
| Stable ID | `restaurant_id` (`r01`–`r30`) |
| Runtime lookup key | `name` (display name, e.g. `"Luigi's Corner Bistro"`) |
| Verdict labels | `SAFE` / `RISK` |

All three retrieval engines query by `restaurant_name` (the display `name` string). `restaurant_id` is duplicated in SQL and Neo4j for traceability but is not the primary retrieval key.

```
restaurant_name
      │
      ├──► SQLite   (inspection facts)
      ├──► FAISS    (review chunks)
      └──► Neo4j    (ownership / chain / supplier graph)
```

---

## Store Responsibility Split

| Store | Role | Belongs here | Must NOT live here |
|---|---|---|---|
| **SQLite** | The Math | Numeric/categorical inspection facts, grades, violation counts, dates | Review prose, ownership edges, supplier recall narratives |
| **FAISS** | The Lore | Free-text reviews, forum posts, whistleblower notes, news snippets | Structured scores, graph edges |
| **Neo4j** | The Web | Ownership, franchise chain membership, supplier links, contagion paths | Inspection score history, review text bodies |

---

## SQLite — Inspection Facts

**Artifact:** `sandbox_data/restaurants.db`

### Table: `restaurants` (30 rows)

| Column | Type | Notes |
|---|---|---|
| `restaurant_id` | TEXT PK | `r01`–`r30` |
| `name` | TEXT NOT NULL UNIQUE | **Join key** for all engines |
| `cuisine_type` | TEXT | e.g. `italian`, `fast_food`, `sushi` |
| `borough` | TEXT | Synthetic city zone (categorical) |
| `seating_capacity` | INTEGER | |
| `years_in_operation` | INTEGER | |

### Table: `inspections` (~60–90 rows, 2–3 per restaurant)

| Column | Type | Notes |
|---|---|---|
| `inspection_id` | TEXT PK | `i001`, … |
| `restaurant_id` | TEXT FK → `restaurants` | |
| `inspection_date` | TEXT | ISO date (`YYYY-MM-DD`) |
| `score` | INTEGER | 0–100, **higher = safer** |
| `grade` | TEXT | `A` / `B` / `C` |
| `critical_violations` | INTEGER | Food-safety critical count |
| `general_violations` | INTEGER | Non-critical count |
| `repeat_violation_flag` | INTEGER | `0` or `1` |
| `passed_reinspection` | INTEGER | `0` or `1`, nullable if no reinspection yet |

**Retrieval contract:** Given `name`, return the latest inspection row joined with restaurant dimension fields. Tagged output: `[SQLITE_STATS]` / `[SQLITE_MISS]` / `[SQLITE_ERROR]`.

**Signal spread targets:**
- ~8 restaurants: low scores, grade C, high critical violations
- ~12 restaurants: middling B grades
- ~10 restaurants: clean A-grade records (some may still be RISK via FAISS or Neo4j)

---

## FAISS — Review Text

**Artifacts:**
- `sandbox_data/reviews.faiss` — `IndexFlatIP`, dim=384, cosine via normalized inner product
- `sandbox_data/reviews_meta.pkl` — list of dicts, index-aligned with FAISS rows
- Embedding model: `all-MiniLM-L6-v2`

**Target scale:** ~50–60 embedded chunks.

### Per-chunk metadata

```python
{
    "restaurant": "Luigi's Corner Bistro",  # matches restaurants.name
    "source": "yelp",                        # yelp | google | health_forum | local_news
    "type": "customer_review",               # customer_review | staff_whistleblower | news_snippet
    "rating": 2,                             # 1–5, nullable for news/whistleblower
    "date": "2024-11-03",
    "text": "…verbatim review text…"
}
```

**Chunk distribution:**
- ~35 chunks for ~20 high-drama restaurants (1–3 chunks each)
- ~15 chunks for middling-reputation restaurants
- ~10 restaurants with **zero FAISS chunks** (intentional `[FAISS_MISS]` signal)

**RISK themes (text only — do not duplicate SQL violation counts):** food poisoning, pest sightings, dirty kitchen, expired ingredients, management retaliation.

**Retrieval contract:** Given `name`, return top-k semantically relevant chunks filtered by `metadata["restaurant"]`. Tagged output: `[FAISS_LORE]` / `[FAISS_MISS]` / `[FAISS_ERROR]`.

---

## Neo4j — Ownership / Chain / Supplier Graph

**Connection:** `bolt://localhost:7687` (Docker)

**Target scale:** ~45–55 nodes, ~25–30 directed edges.

### Node labels

| Label | Count | Properties |
|---|---|---|
| `:AuditorNode:Restaurant` | 30 | `name`, `restaurant_id`, `cuisine_type`, `borough` |
| `:AuditorNode:Owner` | 8–10 | `name`, `owner_id`, `entity_type` (`individual` \| `shell_corp`) |
| `:AuditorNode:Chain` | 5–6 | `name`, `chain_id`, `segment` (`qsr` \| `fast_casual` \| `fine_dining`) |
| `:AuditorNode:Supplier` | 8–10 | `name`, `supplier_id`, `category` (`produce` \| `meat` \| `dairy` \| `seafood`), `recall_count` |

### Relationship types

| Type | Direction | Edge properties | Meaning |
|---|---|---|---|
| `OWNED_BY` | Restaurant → Owner | `since_year`, `ownership_pct` | Legal/control ownership |
| `PART_OF_CHAIN` | Restaurant → Chain | `franchisee` (bool) | Brand/franchise membership |
| `SUPPLIED_BY` | Restaurant → Supplier | `since_year`, `is_primary` (bool) | Ingredient supply link |
| `SHARED_OWNER_WITH` | Restaurant → Restaurant | `owner_name` | Hidden common ownership (contagion risk) |

**Retrieval contract:** Given `name`, traverse 1–2 hops from the matching `:Restaurant` node; return formatted edge list. Tagged output: `[NEO4J_GRAPH]` / `[NEO4J_MISS]` / `[NEO4J_ERROR]`.

**Signal spread targets:**
- ~8 restaurants: linked to `shell_corp` owners or suppliers with `recall_count >= 2`
- ~6 restaurants: dense chain/franchise subgraph
- ~8 restaurants: **no graph edges** (clean isolated nodes)

---

## Cross-Store Linkage

```
restaurants.name          ←→  FAISS metadata["restaurant"]  ←→  (:Restaurant {name})
restaurants.restaurant_id ←→  (:Restaurant {restaurant_id})
```

- No foreign keys across stores; all populated from one `generate_sandbox.py`
- Neo4j `MERGE` keys on `name` (display string)
- FAISS has no chunk IDs beyond list index position

---

## Downstream Verdict Contract

Defined here for alignment with later DSPy synthesis steps (not built in Step 1):

```python
{
    "safe_or_risk": "SAFE" | "RISK",
    "risk_score": float,       # 0.0–10.0
    "the_receipts": str         # cited evidence from all three contexts
}
```
