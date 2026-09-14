# Restaurant Health & Reputation Auditor

A three-engine parallel retrieval system that audits restaurants for health and reputation risk. Three databases. One DSPy synthesizer. Zero auditor fluff allowed.

Given a restaurant name it fires three parallel RAG retrievers simultaneously, cross-references the math, the lore, and the web, then produces a `SAFE / RISK` verdict with a risk score and cited receipts.

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │          generate_sandbox.py            │
                        │     The Synthetic Restaurant Factory    │
                        └──────────┬──────────────────────────────┘
                                   │ writes to
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      ┌──────────────┐   ┌──────────────────┐  ┌───────────────┐
      │   SQLite     │   │   FAISS Index    │  │   Neo4j DB    │
      │restaurants.db│   │ reviews.faiss +  │  │  (Docker or   │
      │ 30 restaurants│  │ reviews_meta.pkl │  │  hosted Aura) │
      │ 65 inspections│ │  50 review chunks│  │  56 nodes     │
      │  (The Math)  │   │  (The Lore)      │  │  40 edges     │
      └──────┬───────┘   └────────┬─────────┘  │  (The Web)    │
             │                    │             └──────┬────────┘
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │ queried simultaneously by
                        ┌─────────▼─────────────────────┐
                        │        rag_engines.py          │
                        │  query_sql()                   │
                        │  query_faiss()   ◀── ThreadPoolExecutor
                        │  query_graph()                 │
                        │  parallel_rag_strike()         │
                        └─────────────────┬──────────────┘
                                          │ three context strings
                        ┌─────────────────▼──────────────┐
                        │          compiler.py            │
                        │  RestaurantRiskSignature (DSPy) │
                        │  RestaurantRiskAnalyzer (CoT)   │
                        │  restaurant_risk_metric (Judge) │
                        │  BootstrapFewShot optimizer     │
                        │  ──────────────────────────     │
                        │  Teacher: gemini-2.5-flash      │
                        │  Student: gemini-2.5-flash-lite │
                        └─────────────────┬──────────────┘
                                          │ verdict JSON
                        ┌─────────────────▼──────────────┐
                        │            app.py               │
                        │   Flask + Neobrutalist UI       │
                        │   http://localhost:5050          │
                        └────────────────────────────────┘
```

**Join key across all three stores:** restaurant display `name` (e.g. `"Moldy Mike's Wing Factory"`). Full schema: **[docs/SCHEMA.md](docs/SCHEMA.md)**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Orchestration | DSPy 2.5+ via OpenRouter |
| SQL | SQLite 3 (`sandbox_data/restaurants.db`) |
| Vector DB | FAISS (`IndexFlatIP`, cosine similarity) |
| Graph DB | Neo4j 5 (Docker locally, or hosted Aura in production) |
| Hybrid retrieval | BM25 + dense embeddings + RRF fusion (`rank-bm25`) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Teacher LLM | `google/gemini-2.5-flash` via OpenRouter |
| Student LLM | `google/gemini-2.5-flash-lite` via OpenRouter |
| UI | Flask + Vanilla JS + CSS neobrutalism |

---

## Prerequisites

- Python 3.10+
- Docker (recommended for local Neo4j)
- An [OpenRouter](https://openrouter.ai/keys) API key (required for `--smoke`, `--optimize`, `--analyze`, `--compare`, and the UI)

---

## Setup (end-to-end)

Follow these steps in order. Each step has a verification command so you can confirm progress without guessing.

### 1. Clone and create virtual environment

```bash
git clone https://github.com/FarhanS311/restaurant-risk-auditor.git
cd restaurant-risk-auditor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Neo4j via Docker

**Neo4j is required for full graph retrieval.** SQLite and FAISS work without it, but `graph_context` will return `[NEO4J_ERROR]` and the synthesizer loses ownership/supplier intelligence.

```bash
docker run --name neo4j-auditor \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/auditorpassword \
  --detach neo4j:5
```

The Neo4j browser will be available at `http://localhost:7474` (login: `neo4j` / `auditorpassword`).

**If the container already exists** from a previous run:

```bash
docker start neo4j-auditor
```

**Stop Neo4j** (useful for fault-isolation testing):

```bash
docker stop neo4j-auditor
```

### 3. Configure credentials

```bash
cp .env.example .env   # then edit .env
```

`.env` contents:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=auditorpassword

OPENROUTER_API_KEY=sk-or-v1-...
TEACHER_MODEL=openrouter/google/gemini-2.5-flash
STUDENT_MODEL=openrouter/google/gemini-2.5-flash-lite
```

### 4. Generate sandbox data

```bash
python generate_sandbox.py
```

**What it does:**
- **SQLite** — Creates `sandbox_data/restaurants.db` with 30 restaurants and 65 inspection records (grades, scores, violation counts).
- **FAISS** — Embeds 50 review/lore chunks using `all-MiniLM-L6-v2` (local, no API key). Writes `sandbox_data/reviews.faiss` and `reviews_meta.pkl`.
- **Neo4j** — Builds a directed graph with 56 nodes (30 Restaurant + 10 Owner + 6 Chain + 10 Supplier) and 40 edges (`OWNED_BY`, `PART_OF_CHAIN`, `SUPPLIED_BY`, `SHARED_OWNER_WITH`).

**Expected output:**

```
════════════════════════════════════════════════════════════
  SYNTHETIC RESTAURANT FACTORY  —  initializing sandbox
════════════════════════════════════════════════════════════

[ 1/3 ] SQLite — The Math
  [SQLite]  30 restaurants, 65 inspections → sandbox_data/restaurants.db
  [SQLite]  Verified: 30 restaurants, 65 inspections

[ 2/3 ] FAISS  — The Lore
  [FAISS]   Embedding 50 review chunks...
  [FAISS]   50 vectors (dim=384) → sandbox_data/reviews.faiss
  [FAISS]   Verified: index.ntotal=50, metadata entries=50

[ 3/3 ] Neo4j  — The Web
  [Neo4j]   56 nodes, 40 edges written
  [Neo4j]   Browser: http://localhost:7474  (neo4j / auditorpassword)

════════════════════════════════════════════════════════════
  SANDBOX FULLY INITIALIZED
════════════════════════════════════════════════════════════
```

### 5. Verify retrieval engines

```bash
python rag_engines.py
```

Runs parallel RAG for four test restaurants and prints a latency benchmark.

**What each engine returns:**

| Engine | Function | Tagged outputs |
|---|---|---|
| SQLite | `query_sql(name)` | `[SQLITE_STATS]` / `[SQLITE_MISS]` / `[SQLITE_ERROR]` |
| FAISS | `query_faiss(name)` | `[FAISS_LORE]` / `[FAISS_MISS]` / `[FAISS_ERROR]` |
| Neo4j | `query_graph(name)` | `[NEO4J_GRAPH]` / `[NEO4J_MISS]` / `[NEO4J_ERROR]` |

`parallel_rag_strike(name)` submits all three to a `ThreadPoolExecutor`. Any single engine failure returns a tagged error string instead of crashing the pipeline.

**Test entities:**

| Restaurant | SQL | FAISS | Graph |
|---|---|---|---|
| `Moldy Mike's Wing Factory` | Grade C stats | scandal lore | dense ownership graph |
| `Farm Table Kitchen` | Grade A stats | scandal lore | shell-corp link |
| `Brasserie Moderne` | Grade A stats | miss | miss |
| `NonExistent Restaurant` | miss | miss | miss |

**Sample query:**

```python
from rag_engines import parallel_rag_strike
ctx = parallel_rag_strike("Moldy Mike's Wing Factory")
print(ctx["sql_context"])
print(ctx["faiss_context"])
print(ctx["graph_context"])
```

**Fault-isolation test** (confirm one engine failure does not crash the others):

```bash
docker stop neo4j-auditor
python rag_engines.py   # SQL + FAISS succeed; graph_context → [NEO4J_ERROR]
docker start neo4j-auditor
```

### 6. Judge metric self-check (no API key)

```bash
python compiler.py --test-judge
```

Deterministic pass/fail test — no LLM calls.

### 7. DSPy smoke test (uncompiled module)

```bash
python compiler.py --smoke "Moldy Mike's Wing Factory"
```

Runs `parallel_rag_strike` then an uncompiled `RestaurantRiskAnalyzer` ChainOfThought call. Expects parseable JSON: `safe_or_risk`, `risk_score`, `the_receipts`.

### 8. Compile with BootstrapFewShot

```bash
python compiler.py --optimize
python compiler.py --analyze "Moldy Mike's Wing Factory"
```

**What `--optimize` does:**
1. Builds a training set from 29 restaurants (`Farm Table Kitchen` is held out for compare).
2. Configures teacher (`gemini-2.5-flash`) and student (`gemini-2.5-flash-lite`) LMs via OpenRouter.
3. Runs `BootstrapFewShot` — teacher generates chain-of-thought traces; optimizer compiles the best demos into the student.
4. Saves `optimized_auditor_state.json` (~gitignored artifact in project root).

### 9. Compare uncompiled vs compiled (held-out entity)

```bash
python compiler.py --compare
```

Defaults to **Farm Table Kitchen** — grade-A SQL, scandal lore, shell-corp graph risk. Not in the training manifest.

Prints a score table (baseline vs compiled on the same RAG contexts):

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

Run `python compiler.py --compare` locally to populate with your API output. The compiled student consistently produces parseable JSON that passes `restaurant_risk_metric`.

### 10. Launch the transparency UI

```bash
python app.py
```

Open `http://localhost:5050`.

**UI features:**
- **Restaurant search** — Datalist of all 30 restaurants from SQLite.
- **Parallel context panels** — Three side-by-side panels show the **raw** SQLite, FAISS, and Neo4j strings returned simultaneously.
- **Verdict banner** — Full-width green (`SAFE`) or red (`RISK`) with metric pass/fail tag.
- **Risk score meter** — Brutalist progress bar, 0.0–10.0.
- **The Receipts terminal** — Dark terminal-style box with synthesized evidence.
- **Chain of Thought** — Collapsible panel with the full LLM reasoning trace.

---

## Sample Queries

```bash
# Clear RISK — grade C, food poisoning lore, Apex shell-corp graph
python compiler.py --analyze "Moldy Mike's Wing Factory"

# Held-out edge case — clean SQL, lore + graph risk (compare target)
python compiler.py --analyze "Farm Table Kitchen"

# SQL-only hit — intentional FAISS/Neo4j miss
python compiler.py --analyze "Brasserie Moderne"

# Error-as-information — all three engines miss
python compiler.py --analyze "NonExistent Restaurant"

# Compare baseline vs compiled on holdout
python compiler.py --compare
```

---

## Judge Metric Rules

The `restaurant_risk_metric` function returns `True` (pass) only if ALL conditions hold:

1. `verdict` field is present and non-empty
2. Output parses as valid JSON (handles markdown fences from model)
3. JSON contains `safe_or_risk`, `risk_score`, and `the_receipts`
4. `safe_or_risk` is exactly `"SAFE"` or `"RISK"`
5. `risk_score` is a float in `[0.0, 10.0]`
6. `the_receipts` is > 30 characters
7. **Zero auditor fluff words** in the entire output — any of these kill the verdict instantly: `synergy`, `stakeholder`, `paradigm shift`, `best-in-class`, `culinary journey`, `premium experience`, `brand alignment`, `value proposition`, `going forward`, `circle back`, `delightful dining experience`, `world-class ambiance`
8. If a gold label is present on the example, the predicted `safe_or_risk` must match

---

## Deployment Notes

### What runs where

| Component | Local dev | Deployed app |
|---|---|---|
| SQLite + FAISS | Files in `sandbox_data/` (generated once) | Same — commit or regenerate on the server |
| Neo4j | Docker on `localhost:7687` | **Must be reachable** — see below |
| LLM calls | OpenRouter API | OpenRouter API |
| Flask UI | `python app.py` on port 5050 | Any WSGI host / PaaS |

### Neo4j requirement (read this before deploying)

**The graph engine does not work without a live Neo4j instance.**

- **Local development:** Run Neo4j in Docker (see Setup step 2). Set `NEO4J_URI=bolt://localhost:7687` in `.env`.
- **Deployed / remote:** You must either:
  1. **Run Neo4j alongside the app** (Docker on the same VM, Kubernetes sidecar, etc.), or
  2. **Use a hosted graph DB** such as [Neo4j Aura](https://neo4j.com/cloud/aura/) and point `NEO4J_URI` at the Aura bolt URL.

If Neo4j is unreachable, the app still runs but `graph_context` returns `[NEO4J_ERROR]` and verdicts lose ownership/supplier/contagion evidence. SQL and FAISS continue to work independently.

**After pointing at a new Neo4j instance**, re-run `python generate_sandbox.py` to populate the graph (the script clears and rebuilds nodes/edges).

### Minimal deploy checklist

```bash
# On the server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set OPENROUTER_API_KEY + NEO4J_URI

# Start Neo4j (local Docker example)
docker run --name neo4j-auditor \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/auditorpassword \
  --detach neo4j:5

python generate_sandbox.py    # populate all three stores
python compiler.py --optimize # optional but recommended for UI quality
python app.py                 # or: gunicorn -w 1 -b 0.0.0.0:5050 app:app
```

Set `PORT` in `.env` to override the default `5050`.

---

## File Structure

```
restaurant-risk-auditor/
├── generate_sandbox.py      # Populate SQLite, FAISS, Neo4j
├── rag_engines.py           # Three parallel RAG retrieval functions
├── compiler.py              # DSPy module, judge metric, optimizer, compare
├── app.py                   # Flask server for the transparency UI
├── requirements.txt         # Python dependencies
├── .env.example             # Credential template
├── docs/
│   └── SCHEMA.md            # Three-store data model reference
├── templates/
│   └── index.html           # Neobrutalist frontend
└── sandbox_data/            # Auto-generated (gitignored)
    ├── restaurants.db       # SQLite — 30 restaurants, 65 inspections
    ├── reviews.faiss        # FAISS — 50 review vectors, dim=384
    └── reviews_meta.pkl     # FAISS metadata — chunk text + restaurant tag
```

`optimized_auditor_state.json` is written to the project root after `--optimize` and is gitignored.

---

## Neo4j Graph Browser

After running `generate_sandbox.py`, explore the ownership web at `http://localhost:7474`:

```cypher
-- Show entire ecosystem
MATCH (a:AuditorNode)-[r]->(b:AuditorNode) RETURN a, r, b LIMIT 100

-- Restaurants owned by shell corporations
MATCH (r:AuditorNode:Restaurant)-[o:OWNED_BY]->(owner:AuditorNode:Owner)
WHERE owner.entity_type = 'shell_corp'
RETURN r.name, owner.name, o.ownership_pct

-- Supplier recall risk
MATCH (r:AuditorNode:Restaurant)-[s:SUPPLIED_BY]->(sup:AuditorNode:Supplier)
WHERE sup.recall_count >= 2
RETURN r.name, sup.name, sup.recall_count

-- Hidden common ownership (contagion)
MATCH (a:AuditorNode:Restaurant)-[s:SHARED_OWNER_WITH]->(b:AuditorNode:Restaurant)
RETURN a.name, b.name, s.owner_name
```

---

## Project Status

- [x] Step 1 — Schema note (`docs/SCHEMA.md`)
- [x] Step 2 — Data generation (`generate_sandbox.py`)
- [x] Step 3 — Retrieval engines (`rag_engines.py`)
- [x] Step 4 — Parallel retrieval (`parallel_rag_strike`)
- [x] Step 5 — DSPy signature and module (`compiler.py`)
- [x] Step 6 — Judge metric (`restaurant_risk_metric`)
- [x] Step 7 — BootstrapFewShot compile (`--optimize` / `--analyze`)
- [x] Step 8 — Analyze and compare (`--compare` on held-out entity)
- [x] Step 9 — Transparency UI (`app.py`)
- [x] Step 10 — Full README and deployment guide
