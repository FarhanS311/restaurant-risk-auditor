#!/usr/bin/env python3
"""
app.py — Neobrutalist transparency UI for the Restaurant Health & Reputation Auditor.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
_program = None

SQLITE_PATH = Path("sandbox_data/restaurants.db")


def _get_program():
    global _program
    if _program is None:
        import dspy
        from compiler import (
            OPTIMIZED_STATE_PATH,
            STUDENT_MODEL,
            RestaurantRiskAnalyzer,
            build_lm,
        )
        dspy.configure(lm=build_lm(STUDENT_MODEL))
        _program = RestaurantRiskAnalyzer()
        if OPTIMIZED_STATE_PATH.exists():
            _program.load(str(OPTIMIZED_STATE_PATH))
    return _program


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/restaurants")
def list_restaurants():
    if not SQLITE_PATH.exists():
        return jsonify([])
    with sqlite3.connect(SQLITE_PATH) as conn:
        rows = conn.execute("""
            SELECT r.name, r.cuisine_type, r.borough,
                   (SELECT grade FROM inspections i
                    WHERE i.restaurant_id = r.restaurant_id
                    ORDER BY inspection_date DESC LIMIT 1) AS latest_grade
            FROM restaurants r
            ORDER BY r.name
        """).fetchall()
    return jsonify([
        {
            "name": r[0],
            "cuisine_type": r[1],
            "borough": r[2],
            "latest_grade": r[3],
        }
        for r in rows
    ])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True, silent=True) or {}
    name = (body.get("restaurant_name") or "").strip()
    if not name:
        return jsonify({"error": "restaurant_name is required"}), 400
    try:
        from compiler import analyze_restaurant
        result = analyze_restaurant(name, program=_get_program())
        result["metric_passed"] = bool(result.get("metric_passed"))
        result["risk_score"] = float(result.get("risk_score") or 0.0)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5050))
    print(f"\n  RESTAURANT RISK AUDITOR  ▶  http://localhost:{port}\n")
    app.run(debug=False, port=port)
