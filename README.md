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

## Status

- [x] Step 1 — Schema note (core entity, SQL / FAISS / Neo4j split)
- [ ] Step 2 — Data generation (`generate_sandbox.py`)
- [ ] Step 3 — Retrieval engines (`rag_engines.py`)
- [ ] Step 4 — DSPy synthesis (`compiler.py`)
