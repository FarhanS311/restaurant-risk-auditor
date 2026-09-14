# Restaurant Health & Reputation Auditor

A three-engine parallel retrieval system that audits restaurants for health and reputation risk. Given a restaurant name, it queries SQLite (inspection scores), FAISS (review text), and Neo4j (ownership/chain/supplier graph) simultaneously, then synthesizes a structured **SAFE / RISK** verdict.

## Domain

| Store | Role | Contents |
|---|---|---|
| SQLite | The Math | Inspection scores, grades, violation counts |
| FAISS | The Lore | Customer reviews, whistleblower notes, news snippets |
| Neo4j | The Web | Ownership, franchise chains, supplier relationships |

## Schema

The full three-store data model is documented in **[docs/SCHEMA.md](docs/SCHEMA.md)**. Read that before writing any generation or retrieval code.

## Setup

### 1. Start Neo4j via Docker

```bash
docker run --name neo4j-auditor \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/auditorpassword \
  --detach neo4j:5
```

Browser: `http://localhost:7474`

### 2. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Generate sandbox data

```bash
python generate_sandbox.py
```

Populates `sandbox_data/restaurants.db`, `reviews.faiss`, `reviews_meta.pkl`, and the Neo4j graph.

### 4. Smoke-test retrieval engines

```bash
python rag_engines.py
```

Or from a Python shell:

```python
from rag_engines import query_sql, query_faiss, query_graph
print(query_sql("Moldy Mike's Wing Factory"))
print(query_faiss("Moldy Mike's Wing Factory"))
print(query_graph("Moldy Mike's Wing Factory"))
```

**Test entities:** `Moldy Mike's Wing Factory` (all three hit), `Farm Table Kitchen` (clean SQL, lore + graph risk), `Brasserie Moderne` (SQL only, FAISS/Neo4j miss), `NonExistent Restaurant` (all miss).

### 5. Parallel retrieval

```python
from rag_engines import parallel_rag_strike
ctx = parallel_rag_strike("Moldy Mike's Wing Factory")
print(ctx["sql_context"], ctx["faiss_context"], ctx["graph_context"])
```

`python rag_engines.py` also prints a latency benchmark (parallel time should track the slowest engine, not the sum of all three).

**Fault-isolation test** (confirm one engine failure does not crash the others):

```bash
docker stop neo4j-auditor
python rag_engines.py   # SQL + FAISS succeed; graph_context → [NEO4J_ERROR]
docker start neo4j-auditor
```

### 6. DSPy smoke test (uncompiled module)

Requires `OPENROUTER_API_KEY` in `.env` (see `.env.example`).

```bash
python compiler.py --smoke "Moldy Mike's Wing Factory"
```

Runs `parallel_rag_strike` then an uncompiled `RestaurantRiskAnalyzer` ChainOfThought call. Expects parseable JSON: `safe_or_risk`, `risk_score`, `the_receipts`.

### 7. Judge metric self-check (no API key)

```bash
python compiler.py --test-judge
```

### 8. Compile with BootstrapFewShot

Requires `OPENROUTER_API_KEY`, Neo4j running, and sandbox data generated.

```bash
python compiler.py --optimize
python compiler.py --analyze "Moldy Mike's Wing Factory"
```

Writes `optimized_auditor_state.json` (gitignored). `--analyze` prints the compiled student's chain-of-thought reasoning trace.

**Note:** After the Step 8 holdout split, re-run `--optimize` — the manifest is now 29 restaurants (`Farm Table Kitchen` held out).

### 9. Compare uncompiled vs compiled (held-out entity)

Requires `OPENROUTER_API_KEY` and a compiled state from `--optimize`.

```bash
python compiler.py --optimize   # re-run after manifest shrink (29 examples)
python compiler.py --compare    # defaults to held-out Farm Table Kitchen
```

`--compare` runs baseline and compiled programs on the **same** RAG contexts and prints a score table. The held-out entity must not appear in `TRAINING_MANIFEST`.

**Example score table** (run locally to populate with your API output):

```
Check                  Baseline     Compiled
------------------------------------------------
json_parse_ok                FAIL         PASS
has_required_keys            FAIL         PASS
label_valid                  FAIL         PASS
score_in_range               FAIL         PASS
receipts_substantive         FAIL         PASS
fluff_free                   PASS         PASS
cites_all_streams            FAIL         PASS
structure_score                 2            7
metric_passed                FAIL         PASS

COMPILE WINS: structure 2/7 → 7/7, metric FAIL → PASS
```

The compiled student consistently produces parseable JSON that passes `restaurant_risk_metric`; the uncompiled baseline often drifts (markdown fences, missing keys, thin receipts).

### 10. Transparency UI

Requires `OPENROUTER_API_KEY`, Neo4j, sandbox data, and optionally `optimized_auditor_state.json`.

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5050`. Pick any restaurant from the search datalist. The UI shows all three **raw** engine outputs side by side (SQLite / FAISS / Neo4j), then the synthesized verdict, risk score meter, receipts, and chain-of-thought trace.

## Status

- [x] Step 1 — Schema note (core entity, SQL / FAISS / Neo4j split)
- [x] Step 2 — Data generation (`generate_sandbox.py`)
- [x] Step 3 — Retrieval engines (`rag_engines.py`)
- [x] Step 4 — Parallel retrieval (`parallel_rag_strike`)
- [x] Step 5 — DSPy signature and module (`compiler.py`)
- [x] Step 6 — Judge metric (`restaurant_risk_metric`)
- [x] Step 7 — BootstrapFewShot compile (`--optimize` / `--analyze`)
- [x] Step 8 — Analyze and compare (`--compare` on held-out entity)
- [x] Step 9 — Transparency UI (`app.py`)
- [ ] Step 10 — Full README
