#!/usr/bin/env python3
"""
compiler.py — DSPy synthesizer for the Restaurant Health & Reputation Auditor.

Step 5: RestaurantRiskSignature + ChainOfThought module + smoke test.
Steps 6–7 (judge metric, BootstrapFewShot) added in later steps.

Requires: OPENROUTER_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Optional

import dspy
from dotenv import load_dotenv

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
    """
    ChainOfThought wrapper around RestaurantRiskSignature.
    Forces the LLM to reason through each data source before committing
    to a SAFE/RISK verdict.
    """

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
#  JSON helpers (smoke test now; judge metric in Step 6)
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


def _validate_verdict_shape(data: Optional[Dict[str, Any]]) -> bool:
    """Syntactic validation only — full judge metric comes in Step 6."""
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
    if not receipts.strip():
        return False

    return True


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
#  STEP 5 — Smoke test (uncompiled module)
# ════════════════════════════════════════════════════════════════════

def run_smoke_test(restaurant_name: str) -> bool:
    """
    Fires parallel RAG, runs uncompiled RestaurantRiskAnalyzer,
    validates verdict JSON shape. Returns True on pass.
    """
    if not OPENROUTER_API_KEY:
        print(
            "\n[ERROR] OPENROUTER_API_KEY is not set in .env\n"
            "Add it like this:\n\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n\n"
            "Get a key at https://openrouter.ai/keys"
        )
        return False

    print(f"\n[Smoke] Student model : {STUDENT_MODEL}")
    print(f"[Smoke] Teacher model : {TEACHER_MODEL} (configured for Step 7)")
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
        "--smoke",
        metavar="RESTAURANT_NAME",
        nargs="?",
        const="Moldy Mike's Wing Factory",
        default=None,
        help=(
            "Run uncompiled smoke test on a restaurant "
            '(default: "Moldy Mike\'s Wing Factory")'
        ),
    )
    args = parser.parse_args()

    if args.smoke is None:
        parser.print_help()
        print(
            "\nExample:\n"
            '  python compiler.py --smoke "Moldy Mike\'s Wing Factory"\n'
        )
        sys.exit(0)

    ok = run_smoke_test(args.smoke)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
