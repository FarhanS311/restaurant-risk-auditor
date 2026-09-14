#!/usr/bin/env python3
"""
generate_sandbox.py — The Synthetic Restaurant Factory
30 fake restaurants. Three databases. One deeply troubled food ecosystem.

Requires: Neo4j running (see .env), no OpenRouter key needed for this step.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

np.random.seed(42)

DB_DIR      = Path("sandbox_data")
SQLITE_PATH = DB_DIR / "restaurants.db"
FAISS_PATH  = DB_DIR / "reviews.faiss"
META_PATH   = DB_DIR / "reviews_meta.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "auditorpassword")


# ════════════════════════════════════════════════════════════════════
#  DOMAIN MODELS
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Restaurant:
    restaurant_id: str
    name: str
    cuisine_type: str
    borough: str
    seating_capacity: int
    years_in_operation: int


@dataclass(frozen=True)
class Owner:
    owner_id: str
    name: str
    entity_type: str  # individual | shell_corp


@dataclass(frozen=True)
class Chain:
    chain_id: str
    name: str
    segment: str  # qsr | fast_casual | fine_dining


@dataclass(frozen=True)
class Supplier:
    supplier_id: str
    name: str
    category: str  # produce | meat | dairy | seafood
    recall_count: int


@dataclass(frozen=True)
class Inspection:
    inspection_id: str
    restaurant_id: str
    inspection_date: str
    score: int
    grade: str
    critical_violations: int
    general_violations: int
    repeat_violation_flag: int
    passed_reinspection: Optional[int]


# ════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA — THE 30 RESTAURANTS
# ════════════════════════════════════════════════════════════════════

RESTAURANTS: List[Restaurant] = [
    Restaurant("r01", "Moldy Mike's Wing Factory",     "fast_food",    "Industrial District", 85,  12),
    Restaurant("r02", "Harbor Catch & Cook",           "seafood",      "Waterfront",          120, 18),
    Restaurant("r03", "The Greasy Spoon Depot",        "diner",        "Downtown",            60,  22),
    Restaurant("r04", "Quick Bite Express #14",        "fast_food",    "Midtown",             40,   8),
    Restaurant("r05", "Dragon Wok Alley",              "chinese",      "Chinatown",           75,  15),
    Restaurant("r06", "Back Alley BBQ Pit",            "bbq",          "Southside",           90,  10),
    Restaurant("r07", "Taco Libre Norte",              "mexican",      "Northgate",           55,   6),
    Restaurant("r08", "Pizza Pit Stop",                "pizza",        "East End",            50,   9),
    Restaurant("r09", "Luigi's Corner Bistro",         "italian",      "Little Italy",        70,  14),
    Restaurant("r10", "Bangkok Street Kitchen",        "thai",         "Midtown",             65,  11),
    Restaurant("r11", "Seoul Bowl House",              "korean",       "Northgate",           80,   7),
    Restaurant("r12", "The Daily Grind Café",          "cafe",         "Downtown",           45,   5),
    Restaurant("r13", "Burger Junction",               "burgers",      "West End",            95,  13),
    Restaurant("r14", "Mama Rosa's Trattoria",         "italian",      "Little Italy",        88,  20),
    Restaurant("r15", "Sakura Sushi Bar",              "sushi",        "Midtown",             72,  16),
    Restaurant("r16", "Pho Paradise",                  "vietnamese",   "Chinatown",           68,   9),
    Restaurant("r17", "Mediterranean Mezze",         "mediterranean","Waterfront",          76,  12),
    Restaurant("r18", "Curry House Express",         "indian",       "Southside",           62,   8),
    Restaurant("r19", "El Patio Cantina",              "mexican",      "West End",            100, 17),
    Restaurant("r20", "The Ramen Lab",                 "japanese",     "East End",            58,   4),
    Restaurant("r21", "Farm Table Kitchen",            "farm_to_table","Waterfront",          55,   6),
    Restaurant("r22", "Brasserie Moderne",             "french",       "Downtown",           64,  19),
    Restaurant("r23", "Noble Steakhouse",              "steakhouse",   "Midtown",            110, 25),
    Restaurant("r24", "Verde Salad Co.",               "healthy",      "Northgate",           42,   3),
    Restaurant("r25", "Baker's Hearth",                "bakery",       "Little Italy",        30,  11),
    Restaurant("r26", "Clean Eats Co-op",              "healthy",      "West End",            38,   7),
    Restaurant("r27", "Artisan Coffee Roasters",       "cafe",         "East End",            28,   5),
    Restaurant("r28", "The Quiet Noodle",              "japanese",     "Chinatown",           52,  14),
    Restaurant("r29", "Hearth & Harvest",              "american",     "Southside",           74,  21),
    Restaurant("r30", "Luna Gelato Bar",               "dessert",      "Downtown",            24,   4),
]

OWNERS: List[Owner] = [
    Owner("o01", "Helena Voss",                  "individual"),
    Owner("o02", "Silver Spoon Holdings LLC",    "shell_corp"),
    Owner("o03", "Marcus Chen",                  "individual"),
    Owner("o04", "Tidewater Food Group LLC",     "shell_corp"),
    Owner("o05", "The Patel Family Trust",       "individual"),
    Owner("o06", "Golden Dragon Restaurant Group","shell_corp"),
    Owner("o07", "Brenda Okonkwo",               "individual"),
    Owner("o08", "Apex Dining Ventures LLC",     "shell_corp"),
    Owner("o09", "Raymond Ortiz",                "individual"),
    Owner("o10", "Midwest Meat Brokers Inc",     "shell_corp"),
]

CHAINS: List[Chain] = [
    Chain("c01", "Quick Bite Express",   "qsr"),
    Chain("c02", "Dragon Dynasty",       "fast_casual"),
    Chain("c03", "Harbor Fresh Markets", "fast_casual"),
    Chain("c04", "Verde Life Group",     "fast_casual"),
    Chain("c05", "Noble Hospitality",    "fine_dining"),
    Chain("c06", "Taco Libre",           "qsr"),
]

SUPPLIERS: List[Supplier] = [
    Supplier("s01", "FreshFields Produce Co.",  "produce",  0),
    Supplier("s02", "Atlantic Seafood Wholesale","seafood", 3),
    Supplier("s03", "Golden Valley Meats",      "meat",     2),
    Supplier("s04", "Dairyland Distribution",   "dairy",    1),
    Supplier("s05", "Pacific Rim Imports",      "produce",  2),
    Supplier("s06", "BudgetProvisions LLC",     "meat",     4),
    Supplier("s07", "Sunrise Farms Cooperative","produce",  0),
    Supplier("s08", "Coastal Catch Seafood",    "seafood",  3),
    Supplier("s09", "ColdChain Logistics",      "dairy",    1),
    Supplier("s10", "Heritage Grain Supply",    "produce",  0),
]


# ════════════════════════════════════════════════════════════════════
#  INSPECTION HISTORY — 2-3 records per restaurant
# ════════════════════════════════════════════════════════════════════

INSPECTIONS: List[Inspection] = [
    # r01 — grade C, repeat violations
    Inspection("i001", "r01", "2024-08-12", 42, "C", 5, 8, 1, 0),
    Inspection("i002", "r01", "2024-10-03", 38, "C", 6, 9, 1, None),
    Inspection("i003", "r01", "2024-11-18", 45, "C", 4, 7, 1, 0),
    # r02
    Inspection("i004", "r02", "2024-07-20", 48, "C", 4, 6, 1, 0),
    Inspection("i005", "r02", "2024-09-15", 52, "C", 3, 5, 1, None),
    Inspection("i006", "r02", "2024-11-01", 50, "C", 4, 6, 1, 0),
    # r03
    Inspection("i007", "r03", "2024-06-10", 44, "C", 5, 7, 1, 0),
    Inspection("i008", "r03", "2024-09-22", 40, "C", 6, 8, 1, None),
    # r04
    Inspection("i009", "r04", "2024-08-05", 46, "C", 4, 6, 1, 0),
    Inspection("i010", "r04", "2024-10-20", 43, "C", 5, 7, 1, 0),
    Inspection("i011", "r04", "2024-12-01", 47, "C", 4, 5, 1, None),
    # r05
    Inspection("i012", "r05", "2024-07-08", 41, "C", 6, 9, 1, 0),
    Inspection("i013", "r05", "2024-10-14", 39, "C", 7, 10, 1, None),
    # r06
    Inspection("i014", "r06", "2024-08-18", 49, "C", 3, 6, 1, 0),
    Inspection("i015", "r06", "2024-11-05", 45, "C", 4, 7, 1, None),
    # r07
    Inspection("i016", "r07", "2024-09-01", 47, "C", 4, 5, 1, 0),
    Inspection("i017", "r07", "2024-11-12", 44, "C", 5, 6, 1, None),
    # r08
    Inspection("i018", "r08", "2024-07-25", 43, "C", 5, 8, 1, 0),
    Inspection("i019", "r08", "2024-10-08", 46, "C", 4, 6, 1, 0),
    # r09 — B grade
    Inspection("i020", "r09", "2024-06-15", 72, "B", 2, 4, 0, 1),
    Inspection("i021", "r09", "2024-09-28", 68, "B", 2, 5, 0, None),
    # r10
    Inspection("i022", "r10", "2024-07-12", 74, "B", 1, 3, 0, 1),
    Inspection("i023", "r10", "2024-10-25", 70, "B", 2, 4, 0, None),
    # r11
    Inspection("i024", "r11", "2024-08-02", 76, "B", 1, 3, 0, 1),
    Inspection("i025", "r11", "2024-11-08", 73, "B", 2, 3, 0, None),
    # r12
    Inspection("i026", "r12", "2024-06-20", 71, "B", 2, 4, 0, 1),
    Inspection("i027", "r12", "2024-09-10", 69, "B", 2, 5, 0, None),
    # r13
    Inspection("i028", "r13", "2024-07-30", 67, "B", 2, 5, 0, 0),
    Inspection("i029", "r13", "2024-10-15", 70, "B", 1, 4, 0, 1),
    # r14
    Inspection("i030", "r14", "2024-05-18", 75, "B", 1, 3, 0, 1),
    Inspection("i031", "r14", "2024-08-22", 72, "B", 2, 4, 0, None),
    Inspection("i032", "r14", "2024-11-14", 74, "B", 1, 3, 0, None),
    # r15
    Inspection("i033", "r15", "2024-06-28", 66, "B", 2, 5, 0, 0),
    Inspection("i034", "r15", "2024-09-18", 69, "B", 2, 4, 0, 1),
    # r16
    Inspection("i035", "r16", "2024-07-05", 73, "B", 1, 4, 0, 1),
    Inspection("i036", "r16", "2024-10-12", 71, "B", 2, 3, 0, None),
    # r17
    Inspection("i037", "r17", "2024-08-08", 68, "B", 2, 5, 0, 1),
    Inspection("i038", "r17", "2024-11-02", 65, "B", 2, 6, 0, None),
    # r18
    Inspection("i039", "r18", "2024-06-25", 70, "B", 2, 4, 0, 1),
    Inspection("i040", "r18", "2024-09-30", 67, "B", 2, 5, 0, None),
    # r19
    Inspection("i041", "r19", "2024-07-18", 64, "B", 3, 5, 0, 0),
    Inspection("i042", "r19", "2024-10-28", 66, "B", 2, 4, 0, 1),
    # r20 — B grade
    Inspection("i043", "r20", "2024-08-14", 78, "B", 1, 2, 0, 1),
    Inspection("i044", "r20", "2024-11-06", 76, "B", 1, 3, 0, None),
    # r21 — A grade but graph risk
    Inspection("i045", "r21", "2024-05-10", 94, "A", 0, 1, 0, 1),
    Inspection("i046", "r21", "2024-08-15", 92, "A", 0, 2, 0, None),
    Inspection("i047", "r21", "2024-11-20", 95, "A", 0, 1, 0, None),
    # r22-r30 — clean A grades
    Inspection("i048", "r22", "2024-06-12", 91, "A", 0, 2, 0, 1),
    Inspection("i049", "r22", "2024-09-20", 93, "A", 0, 1, 0, None),
    Inspection("i050", "r23", "2024-07-02", 96, "A", 0, 1, 0, 1),
    Inspection("i051", "r23", "2024-10-10", 94, "A", 0, 2, 0, None),
    Inspection("i052", "r24", "2024-08-01", 97, "A", 0, 0, 0, 1),
    Inspection("i053", "r24", "2024-11-05", 95, "A", 0, 1, 0, None),
    Inspection("i054", "r25", "2024-06-30", 92, "A", 0, 1, 0, 1),
    Inspection("i055", "r25", "2024-09-25", 90, "A", 0, 2, 0, None),
    Inspection("i056", "r26", "2024-07-22", 98, "A", 0, 0, 0, 1),
    Inspection("i057", "r26", "2024-10-18", 96, "A", 0, 1, 0, None),
    Inspection("i058", "r27", "2024-08-20", 93, "A", 0, 1, 0, 1),
    Inspection("i059", "r27", "2024-11-10", 91, "A", 0, 2, 0, None),
    Inspection("i060", "r28", "2024-05-28", 94, "A", 0, 1, 0, 1),
    Inspection("i061", "r28", "2024-08-30", 92, "A", 0, 2, 0, None),
    Inspection("i062", "r29", "2024-06-08", 95, "A", 0, 1, 0, 1),
    Inspection("i063", "r29", "2024-09-12", 93, "A", 0, 2, 0, None),
    Inspection("i064", "r30", "2024-07-15", 96, "A", 0, 0, 0, 1),
    Inspection("i065", "r30", "2024-10-22", 94, "A", 0, 1, 0, None),
]


# ════════════════════════════════════════════════════════════════════
#  THE LORE — Reviews, whistleblower posts, news snippets
# ════════════════════════════════════════════════════════════════════

REVIEW_CORPUS: List[Dict] = [

    # ── Moldy Mike's Wing Factory ────────────────────────────────────
    {
        "restaurant": "Moldy Mike's Wing Factory", "source": "yelp", "type": "customer_review",
        "rating": 1, "date": "2024-10-15",
        "text": (
            "Ordered the 24-piece bucket and found something that looked like a mouse tail "
            "stuck to the breading. Manager said it was 'a spice blend fiber.' Called the health "
            "department the next morning. They already had an open file on this place."
        ),
    },
    {
        "restaurant": "Moldy Mike's Wing Factory", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-09-28",
        "text": (
            "[Anonymous kitchen staff] Walk-in cooler has been running at 48°F for three weeks. "
            "Manager told us to 'rotate the probe thermometer' during inspections. Sauce bottles "
            "get topped off with expired product from the back. I quit after the second norovirus cluster."
        ),
    },
    {
        "restaurant": "Moldy Mike's Wing Factory", "source": "local_news", "type": "news_snippet",
        "rating": None, "date": "2024-11-02",
        "text": (
            "INDUSTRIAL DISTRICT — Moldy Mike's Wing Factory issued a partial closure notice after "
            "inspectors documented raw chicken stored above ready-to-eat coleslaw. Owner Mike Delgado "
            "told reporters the violation was 'a one-time stacking error.' Records show four similar "
            "citations in 18 months."
        ),
    },

    # ── Harbor Catch & Cook ──────────────────────────────────────────
    {
        "restaurant": "Harbor Catch & Cook", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-10-08",
        "text": (
            "The clam chowder tasted off. Three people at our table were sick within 6 hours. "
            "One went to urgent care. Restaurant blamed 'seasonal allergies.' The oysters on the "
            "half shell smelled like ammonia. Never again."
        ),
    },
    {
        "restaurant": "Harbor Catch & Cook", "source": "yelp", "type": "customer_review",
        "rating": 2, "date": "2024-09-14",
        "text": (
            "Fish tasted days old despite 'caught this morning' chalkboard sign. Bathroom had no hot "
            "water and the hand dryer was broken. Server admitted they share a supplier with two "
            "restaurants that were shut down last year."
        ),
    },
    {
        "restaurant": "Harbor Catch & Cook", "source": "local_news", "type": "news_snippet",
        "rating": None, "date": "2024-08-20",
        "text": (
            "WATERFRONT — Atlantic Seafood Wholesale, primary supplier to Harbor Catch & Cook, "
            "recalled 12,000 lbs of scallops linked to Vibrio contamination. Restaurant continued "
            "serving seafood tower special for 48 hours after the recall notice."
        ),
    },

    # ── The Greasy Spoon Depot ───────────────────────────────────────
    {
        "restaurant": "The Greasy Spoon Depot", "source": "yelp", "type": "customer_review",
        "rating": 1, "date": "2024-11-05",
        "text": (
            "Eggs were served with a side of something crawling. Not a joke — actual insect on the "
            "plate. Waitress scraped it off and brought the same plate back. When I asked for a "
            "refund she said I was 'making a scene for TikTok.'"
        ),
    },
    {
        "restaurant": "The Greasy Spoon Depot", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-10-01",
        "text": (
            "Dishwasher hasn't worked in two months. We rinse plates in a mop sink with cold water "
            "and stack them wet. Grease trap hasn't been pumped since July. Owner says health "
            "inspectors 'never look behind the fryer.'"
        ),
    },

    # ── Quick Bite Express #14 ───────────────────────────────────────
    {
        "restaurant": "Quick Bite Express #14", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-10-22",
        "text": (
            "Found a band-aid in my burger. Not on it — in it. Wrapped in the patty like a "
            "surprise filling. Manager offered a free meal coupon. I don't want a free meal "
            "from the band-aid restaurant."
        ),
    },
    {
        "restaurant": "Quick Bite Express #14", "source": "yelp", "type": "customer_review",
        "rating": 2, "date": "2024-09-08",
        "text": (
            "Every Quick Bite in this city is disgusting but #14 is special. Floors sticky, "
            "ice machine has black mold visible through the lid, and the night shift openly "
            "discusses which violations to hide before the inspector comes."
        ),
    },
    {
        "restaurant": "Quick Bite Express #14", "source": "local_news", "type": "news_snippet",
        "rating": None, "date": "2024-07-15",
        "text": (
            "MIDTOWN — Quick Bite Express franchisee Apex Dining Ventures LLC faces class-action "
            "after 23 customers reported foodborne illness linked to undercooked chicken at location #14. "
            "Corporate HQ declined comment, citing 'ongoing franchise investigation.'"
        ),
    },

    # ── Dragon Wok Alley ─────────────────────────────────────────────
    {
        "restaurant": "Dragon Wok Alley", "source": "yelp", "type": "customer_review",
        "rating": 1, "date": "2024-10-30",
        "text": (
            "Ordered fried rice and bit into something crunchy that wasn't vegetable. Looked like "
            "a cockroach fragment. They remade the dish without apology and charged me for both "
            "because I 'already ate half.'"
        ),
    },
    {
        "restaurant": "Dragon Wok Alley", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-09-12",
        "text": (
            "Produce arrives from Pacific Rim Imports already slimy half the time. We're told to "
            "rinse it in bleach water and serve same day. Wok oil hasn't been changed in two weeks. "
            "Golden Dragon Restaurant Group owns this and three other failing locations."
        ),
    },

    # ── Back Alley BBQ Pit ───────────────────────────────────────────
    {
        "restaurant": "Back Alley BBQ Pit", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-11-01",
        "text": (
            "Brisket was room temperature when served. Coleslaw tasted sour in a bad way. "
            "Saw a rat run along the baseboard near the smoker. Staff didn't react like it was "
            "the first time."
        ),
    },
    {
        "restaurant": "Back Alley BBQ Pit", "source": "yelp", "type": "customer_review",
        "rating": 1, "date": "2024-08-25",
        "text": (
            "Food poisoning after the pulled pork platter. Four of us, same symptoms, same timeline. "
            "Owner posted on Facebook that we were 'competitors spreading lies.' We're a book club."
        ),
    },

    # ── Taco Libre Norte ─────────────────────────────────────────────
    {
        "restaurant": "Taco Libre Norte", "source": "yelp", "type": "customer_review",
        "rating": 2, "date": "2024-10-18",
        "text": (
            "Sour cream was clearly expired — curdled and separated. Told the cashier and she "
            "just removed the container and kept serving from the backup tub without checking it. "
            "Health grade on the wall says C. Believe the wall."
        ),
    },
    {
        "restaurant": "Taco Libre Norte", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-09-05",
        "text": (
            "Walked in and walked out. Kitchen visible from counter — cook handling raw chicken "
            "then grabbing tortillas with same gloves. No hand wash station in sight. "
            "This location is a Taco Libre franchise and it shows."
        ),
    },

    # ── Pizza Pit Stop ───────────────────────────────────────────────
    {
        "restaurant": "Pizza Pit Stop", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-10-12",
        "text": (
            "Pizza arrived with hair baked into the cheese. Called to complain and was told "
            "'that's how you know it's handmade.' Delivery driver smelled like cigarettes and "
            "left the pizza box on the ground while he checked his phone."
        ),
    },
    {
        "restaurant": "Pizza Pit Stop", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-08-30",
        "text": (
            "Expired pepperoni gets the labels peeled off and moved to the walk-in. Dough sits at "
            "room temp for 6+ hours. Owner says pizza oven heat 'kills everything anyway.' "
            "BudgetProvisions LLC delivers gray ground beef and we use it."
        ),
    },

    # ── Luigi's Corner Bistro ─────────────────────────────────────────
    {
        "restaurant": "Luigi's Corner Bistro", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-10-05",
        "text": (
            "Food is decent but the dining room has a persistent smell like old mop water. "
            "Saw a roach near the wine rack. Server moved us but didn't seem surprised. "
            "For the price point I expected better kitchen standards."
        ),
    },
    {
        "restaurant": "Luigi's Corner Bistro", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-09-20",
        "text": (
            "Carbonara was lukewarm and the parmesan tasted off. When I mentioned it, the chef "
            "came out and said I 'don't understand authentic Italian.' Health inspection posted "
            "B grade. The math checks out."
        ),
    },

    # ── Bangkok Street Kitchen ─────────────────────────────────────────
    {
        "restaurant": "Bangkok Street Kitchen", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-10-28",
        "text": (
            "Pad thai was great but I watched a cook wipe their hands on their apron after "
            "handling cash and go straight back to the wok. Bathroom was out of soap. "
            "Middling experience with some red flags."
        ),
    },
    {
        "restaurant": "Bangkok Street Kitchen", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-08-14",
        "text": (
            "Tom yum soup had a piece of plastic in it. Staff apologized but didn't comp the meal. "
            "Neighborhood forum says this place has been on health watch for months."
        ),
    },

    # ── Seoul Bowl House ─────────────────────────────────────────────
    {
        "restaurant": "Seoul Bowl House", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-11-03",
        "text": (
            "Bibimbap was good but side dishes tasted like they'd been sitting out all day. "
            "Kimchi jar had mold on the rim. Not catastrophic but wouldn't bring family here."
        ),
    },
    {
        "restaurant": "Seoul Bowl House", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-09-22",
        "text": (
            "Found a hair in my japchae. Server replaced it quickly but the replacement had "
            "the same issue. Kitchen needs a serious hygiene reset."
        ),
    },

    # ── The Daily Grind Café ───────────────────────────────────────────
    {
        "restaurant": "The Daily Grind Café", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-10-10",
        "text": (
            "Coffee is excellent but the pastry case has items that look days old. Croissant was "
            "stale and the muffin had a fruit fly on it when they opened the case. "
            "Health grade B — feels accurate."
        ),
    },
    {
        "restaurant": "The Daily Grind Café", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-08-08",
        "text": (
            "Milk steamer smells like sour dairy. I asked if it had been cleaned and the barista "
            "said 'probably yesterday.' For a café that charges $7 for a latte, that's not okay."
        ),
    },

    # ── Burger Junction ────────────────────────────────────────────────
    {
        "restaurant": "Burger Junction", "source": "yelp", "type": "customer_review",
        "rating": 2, "date": "2024-10-20",
        "text": (
            "Burger came out pink in the middle despite ordering well-done. Fries were cold and "
            "soggy. Server shrugged and said the kitchen was 'having a night.'"
        ),
    },
    {
        "restaurant": "Burger Junction", "source": "google", "type": "customer_review",
        "rating": 3, "date": "2024-09-01",
        "text": (
            "Decent burgers but the dining area was filthy — ketchup smears on every table, "
            "floors not swept. Restroom had no paper towels. B grade energy all around."
        ),
    },

    # ── Mama Rosa's Trattoria ──────────────────────────────────────────
    {
        "restaurant": "Mama Rosa's Trattoria", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-11-08",
        "text": (
            "Lovely atmosphere but the seafood risotto tasted like it had been reheated twice. "
            "Bread basket had a slice with green mold on the crust. Server removed it without "
            "checking the rest."
        ),
    },
    {
        "restaurant": "Mama Rosa's Trattoria", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-08-18",
        "text": (
            "Tiramisu made my partner sick. Not severe but unmistakable food poisoning symptoms. "
            "Restaurant said they 'never had complaints before' which is clearly not true from "
            "the other reviews."
        ),
    },

    # ── Sakura Sushi Bar ───────────────────────────────────────────────
    {
        "restaurant": "Sakura Sushi Bar", "source": "yelp", "type": "customer_review",
        "rating": 2, "date": "2024-10-25",
        "text": (
            "Salmon sashimi was mushy and had a metallic taste. Chef said it was 'super fresh.' "
            "My stomach disagreed for the next 24 hours. Sushi bar counter had visible dried fish "
            "residue from the previous service."
        ),
    },
    {
        "restaurant": "Sakura Sushi Bar", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-09-15",
        "text": (
            "Tuna gets refrozen after service and served as 'fresh' the next day. Rice sits at "
            "room temperature for hours. Owner says health inspectors 'don't understand sushi culture.'"
        ),
    },

    # ── Pho Paradise ───────────────────────────────────────────────────
    {
        "restaurant": "Pho Paradise", "source": "google", "type": "customer_review",
        "rating": 3, "date": "2024-10-02",
        "text": (
            "Broth was flavorful but the bean sprouts looked wilted and brown. Table hadn't been "
            "wiped — sticky residue everywhere. Average pho, below-average cleanliness."
        ),
    },

    # ── Mediterranean Mezze ────────────────────────────────────────────
    {
        "restaurant": "Mediterranean Mezze", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-09-28",
        "text": (
            "Hummus had a fermented edge to it. Pita bread was stale. Not terrible but the "
            "mezze platter looked like it had been assembled hours before serving."
        ),
    },
    {
        "restaurant": "Mediterranean Mezze", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-08-05",
        "text": (
            "Feta cheese tasted sour past the normal tang. Asked if it was expired and server "
            "said 'that's how Greek cheese tastes.' I've been to Greece. This isn't it."
        ),
    },

    # ── Curry House Express ────────────────────────────────────────────
    {
        "restaurant": "Curry House Express", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-10-15",
        "text": (
            "Chicken tikka was dry and lukewarm. Naan had hard edges like it was reheated from "
            "yesterday. Spice level was good but food safety vibes were off."
        ),
    },
    {
        "restaurant": "Curry House Express", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-09-10",
        "text": (
            "Buffet steam table had dishes with no labels and no temperature monitoring. "
            "Saw a customer sneeze near the tray and staff didn't swap it out."
        ),
    },

    # ── El Patio Cantina ───────────────────────────────────────────────
    {
        "restaurant": "El Patio Cantina", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-11-01",
        "text": (
            "Guacamole was brown when served. Server said it was 'oxidized, not spoiled.' "
            "Margarita glass had lipstick from a previous guest. Patio seating area smelled like "
            "garbage from the alley."
        ),
    },
    {
        "restaurant": "El Patio Cantina", "source": "google", "type": "customer_review",
        "rating": 2, "date": "2024-08-22",
        "text": (
            "Queso dip was cold and congealed. When we asked for it reheated, it came back "
            "microwave-hot on the edges and frozen in the center. B grade restaurant behaving "
            "like a C."
        ),
    },

    # ── The Ramen Lab ──────────────────────────────────────────────────
    {
        "restaurant": "The Ramen Lab", "source": "yelp", "type": "customer_review",
        "rating": 3, "date": "2024-10-08",
        "text": (
            "Broth was excellent but the soft-boiled egg smelled slightly sulfuric. "
            "Nori was chewy like it had been left out. Good concept, execution slipping."
        ),
    },
    {
        "restaurant": "The Ramen Lab", "source": "google", "type": "customer_review",
        "rating": 3, "date": "2024-09-12",
        "text": (
            "New restaurant, still finding its footing. Kitchen was loud and chaotic. "
            "Saw noodles dropped on the counter and put back in the bowl. Hope they tighten up."
        ),
    },

    # ── Farm Table Kitchen (clean SQL, graph risk) ─────────────────────
    {
        "restaurant": "Farm Table Kitchen", "source": "local_news", "type": "news_snippet",
        "rating": None, "date": "2024-10-30",
        "text": (
            "WATERFRONT — Farm Table Kitchen maintains an A health grade, but investigative "
            "reporters linked owner Silver Spoon Holdings LLC to three shuttered restaurants "
            "with repeat violations. Current chef-owner says 'prior entities are legally separate.'"
        ),
    },
    {
        "restaurant": "Farm Table Kitchen", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-09-18",
        "text": (
            "Beautiful farm-to-table branding but half the 'local' produce comes from the same "
            "wholesale supplier as Moldy Mike's Wing Factory. Owner told us to use farm stand "
            "labels on pre-packaged greens. Inspection scores don't show supplier fraud."
        ),
    },
    {
        "restaurant": "Farm Table Kitchen", "source": "yelp", "type": "customer_review",
        "rating": 4, "date": "2024-11-10",
        "text": (
            "Gorgeous plating and lovely atmosphere. Food was genuinely excellent. Giving 4 not 5 "
            "because a friend who works in restaurant law said this ownership group has a "
            "complicated history. The salad was incredible though."
        ),
    },

    # ── Extra high-drama chunks (scale target ~50) ───────────────────
    {
        "restaurant": "Moldy Mike's Wing Factory", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-08-02",
        "text": (
            "Health inspector was eating lunch at the table next to us. That should tell you everything. "
            "She was writing notes the entire meal. Our wings never came — kitchen 'ran out' "
            "after the inspector asked to see the walk-in."
        ),
    },
    {
        "restaurant": "Harbor Catch & Cook", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-07-10",
        "text": (
            "We thaw seafood on sheet pans at room temp because the owner says ice baths 'wash off flavor.' "
            "Tidewater Food Group LLC rotates managers every 6 months so nobody tracks violation history. "
            "Same supplier, same shortcuts, new sign on the door."
        ),
    },
    {
        "restaurant": "Quick Bite Express #14", "source": "health_forum", "type": "staff_whistleblower",
        "rating": None, "date": "2024-11-12",
        "text": (
            "Franchise playbook says discard held food after 4 hours. We hold fries 8+ hours and "
            "sprinkle fresh salt to 'refresh' them. Apex Dining Ventures audits us with a checklist "
            "they fill out before arriving. Real inspector gets the reheated performance."
        ),
    },
    {
        "restaurant": "Dragon Wok Alley", "source": "local_news", "type": "news_snippet",
        "rating": None, "date": "2024-11-15",
        "text": (
            "CHINATOWN — Pacific Rim Imports flagged for listeria on pre-cut vegetables. Dragon Wok Alley "
            "and two sister locations under Golden Dragon Restaurant Group continued lunch service "
            "for six hours before management acknowledged the recall text chain."
        ),
    },
    {
        "restaurant": "The Greasy Spoon Depot", "source": "google", "type": "customer_review",
        "rating": 1, "date": "2024-12-01",
        "text": (
            "Hash browns served with what looked like mouse droppings mixed in. Not a metaphor. "
            "Owner offered 10% off next visit. Health department placard still says grade C "
            "from October and they act like that's a passing score."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════
#  THE WEB — Ownership, chain, and supplier graph edges
# ════════════════════════════════════════════════════════════════════
# Format: (source, target, relationship, attributes_dict)

GRAPH_EDGES: List[Tuple] = [
    # ── OWNED_BY (Restaurant → Owner) ────────────────────────────────
    ("Moldy Mike's Wing Factory",     "Apex Dining Ventures LLC",     "OWNED_BY",        {"since_year": 2018, "ownership_pct": 100}),
    ("Harbor Catch & Cook",           "Tidewater Food Group LLC",     "OWNED_BY",        {"since_year": 2015, "ownership_pct": 100}),
    ("The Greasy Spoon Depot",        "Raymond Ortiz",                "OWNED_BY",        {"since_year": 2002, "ownership_pct": 100}),
    ("Quick Bite Express #14",        "Apex Dining Ventures LLC",     "OWNED_BY",        {"since_year": 2020, "ownership_pct": 100}),
    ("Dragon Wok Alley",              "Golden Dragon Restaurant Group","OWNED_BY",       {"since_year": 2016, "ownership_pct": 100}),
    ("Back Alley BBQ Pit",            "Marcus Chen",                  "OWNED_BY",        {"since_year": 2019, "ownership_pct": 80}),
    ("Taco Libre Norte",              "Apex Dining Ventures LLC",     "OWNED_BY",        {"since_year": 2021, "ownership_pct": 100}),
    ("Pizza Pit Stop",                "Midwest Meat Brokers Inc",     "OWNED_BY",        {"since_year": 2017, "ownership_pct": 100}),
    ("Luigi's Corner Bistro",         "Helena Voss",                  "OWNED_BY",        {"since_year": 2010, "ownership_pct": 100}),
    ("Bangkok Street Kitchen",        "The Patel Family Trust",       "OWNED_BY",        {"since_year": 2014, "ownership_pct": 100}),
    ("Seoul Bowl House",              "Marcus Chen",                  "OWNED_BY",        {"since_year": 2022, "ownership_pct": 60}),
    ("Farm Table Kitchen",            "Silver Spoon Holdings LLC",    "OWNED_BY",        {"since_year": 2023, "ownership_pct": 100}),
    ("Mama Rosa's Trattoria",         "Helena Voss",                  "OWNED_BY",        {"since_year": 2004, "ownership_pct": 100}),
    ("El Patio Cantina",              "Brenda Okonkwo",               "OWNED_BY",        {"since_year": 2012, "ownership_pct": 100}),

    # ── PART_OF_CHAIN (Restaurant → Chain) ───────────────────────────
    ("Quick Bite Express #14",        "Quick Bite Express",           "PART_OF_CHAIN",   {"franchisee": True}),
    ("Taco Libre Norte",              "Taco Libre",                   "PART_OF_CHAIN",   {"franchisee": True}),
    ("Dragon Wok Alley",              "Dragon Dynasty",               "PART_OF_CHAIN",   {"franchisee": True}),
    ("Harbor Catch & Cook",           "Harbor Fresh Markets",         "PART_OF_CHAIN",   {"franchisee": False}),
    ("Farm Table Kitchen",            "Harbor Fresh Markets",         "PART_OF_CHAIN",   {"franchisee": False}),
    ("Noble Steakhouse",              "Noble Hospitality",            "PART_OF_CHAIN",   {"franchisee": False}),
    ("Verde Salad Co.",               "Verde Life Group",             "PART_OF_CHAIN",   {"franchisee": True}),

    # ── SUPPLIED_BY (Restaurant → Supplier) ──────────────────────────
    ("Moldy Mike's Wing Factory",     "BudgetProvisions LLC",         "SUPPLIED_BY",     {"since_year": 2019, "is_primary": True}),
    ("Harbor Catch & Cook",           "Atlantic Seafood Wholesale",   "SUPPLIED_BY",     {"since_year": 2015, "is_primary": True}),
    ("Harbor Catch & Cook",           "Coastal Catch Seafood",        "SUPPLIED_BY",     {"since_year": 2020, "is_primary": False}),
    ("Quick Bite Express #14",        "BudgetProvisions LLC",         "SUPPLIED_BY",     {"since_year": 2020, "is_primary": True}),
    ("Quick Bite Express #14",        "Dairyland Distribution",       "SUPPLIED_BY",     {"since_year": 2020, "is_primary": False}),
    ("Dragon Wok Alley",              "Pacific Rim Imports",          "SUPPLIED_BY",     {"since_year": 2016, "is_primary": True}),
    ("Dragon Wok Alley",              "Golden Valley Meats",          "SUPPLIED_BY",     {"since_year": 2018, "is_primary": False}),
    ("Pizza Pit Stop",                "BudgetProvisions LLC",         "SUPPLIED_BY",     {"since_year": 2017, "is_primary": True}),
    ("Back Alley BBQ Pit",            "Golden Valley Meats",          "SUPPLIED_BY",     {"since_year": 2019, "is_primary": True}),
    ("Farm Table Kitchen",            "FreshFields Produce Co.",      "SUPPLIED_BY",     {"since_year": 2023, "is_primary": True}),
    ("Luigi's Corner Bistro",         "FreshFields Produce Co.",      "SUPPLIED_BY",     {"since_year": 2012, "is_primary": True}),
    ("Sakura Sushi Bar",              "Coastal Catch Seafood",        "SUPPLIED_BY",     {"since_year": 2018, "is_primary": True}),
    ("The Daily Grind Café",          "Dairyland Distribution",       "SUPPLIED_BY",     {"since_year": 2021, "is_primary": True}),
    ("Mama Rosa's Trattoria",         "Heritage Grain Supply",        "SUPPLIED_BY",     {"since_year": 2010, "is_primary": True}),

    # ── SHARED_OWNER_WITH (Restaurant → Restaurant) ──────────────────
    ("Moldy Mike's Wing Factory",     "Quick Bite Express #14",       "SHARED_OWNER_WITH", {"owner_name": "Apex Dining Ventures LLC"}),
    ("Moldy Mike's Wing Factory",     "Taco Libre Norte",             "SHARED_OWNER_WITH", {"owner_name": "Apex Dining Ventures LLC"}),
    ("Farm Table Kitchen",            "Moldy Mike's Wing Factory",    "SHARED_OWNER_WITH", {"owner_name": "Silver Spoon Holdings LLC / Apex Dining Ventures LLC"}),
    ("Dragon Wok Alley",              "Pizza Pit Stop",               "SHARED_OWNER_WITH", {"owner_name": "Golden Dragon Restaurant Group / Midwest Meat Brokers Inc"}),
    ("Harbor Catch & Cook",           "Farm Table Kitchen",           "SHARED_OWNER_WITH", {"owner_name": "Tidewater Food Group LLC / Silver Spoon Holdings LLC"}),
]


# ════════════════════════════════════════════════════════════════════
#  DATABASE MANAGERS
# ════════════════════════════════════════════════════════════════════

class SQLiteManager:
    """Owns the math — structured inspection facts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def setup_and_populate(
        self,
        restaurants: List[Restaurant],
        inspections: List[Inspection],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS inspections")
            conn.execute("DROP TABLE IF EXISTS restaurants")
            conn.execute("""
                CREATE TABLE restaurants (
                    restaurant_id      TEXT PRIMARY KEY,
                    name               TEXT NOT NULL UNIQUE,
                    cuisine_type       TEXT,
                    borough            TEXT,
                    seating_capacity   INTEGER,
                    years_in_operation INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE inspections (
                    inspection_id         TEXT PRIMARY KEY,
                    restaurant_id         TEXT NOT NULL,
                    inspection_date       TEXT,
                    score                 INTEGER,
                    grade                 TEXT,
                    critical_violations   INTEGER,
                    general_violations    INTEGER,
                    repeat_violation_flag INTEGER,
                    passed_reinspection   INTEGER,
                    FOREIGN KEY (restaurant_id) REFERENCES restaurants(restaurant_id)
                )
            """)
            conn.executemany(
                "INSERT INTO restaurants VALUES (?,?,?,?,?,?)",
                [
                    (r.restaurant_id, r.name, r.cuisine_type, r.borough,
                     r.seating_capacity, r.years_in_operation)
                    for r in restaurants
                ],
            )
            conn.executemany(
                "INSERT INTO inspections VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (i.inspection_id, i.restaurant_id, i.inspection_date,
                     i.score, i.grade, i.critical_violations,
                     i.general_violations, i.repeat_violation_flag,
                     i.passed_reinspection)
                    for i in inspections
                ],
            )
            conn.commit()
        print(f"  [SQLite]  {len(restaurants)} restaurants, {len(inspections)} inspections → {self.db_path}")

    def verify(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            r_count = conn.execute("SELECT COUNT(*) FROM restaurants").fetchone()[0]
            i_count = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        print(f"  [SQLite]  Verified: {r_count} restaurants, {i_count} inspections")


class FAISSManager:
    """Owns the lore — unstructured review corpus, vector-indexed."""

    def __init__(self, faiss_path: Path, meta_path: Path) -> None:
        self.faiss_path = faiss_path
        self.meta_path  = meta_path
        print(f"  [FAISS]   Loading embedding model '{EMBED_MODEL}'...")
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed_and_store(self, corpus: List[Dict]) -> None:
        texts = [entry["text"] for entry in corpus]
        print(f"  [FAISS]   Embedding {len(texts)} review chunks...")
        vectors = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32,
        ).astype(np.float32)

        dim   = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        faiss.write_index(index, str(self.faiss_path))
        with open(self.meta_path, "wb") as fh:
            pickle.dump(corpus, fh)

        print(f"  [FAISS]   {index.ntotal} vectors (dim={dim}) → {self.faiss_path}")
        print(f"  [FAISS]   Metadata                          → {self.meta_path}")

    def verify(self) -> None:
        index = faiss.read_index(str(self.faiss_path))
        with open(self.meta_path, "rb") as fh:
            meta = pickle.load(fh)
        print(f"  [FAISS]   Verified: index.ntotal={index.ntotal}, metadata entries={len(meta)}")


class GraphManager:
    """Owns the web — ownership, chain, and supplier graph in Neo4j."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def _wipe_sandbox(self, tx) -> None:
        tx.run("MATCH (n:AuditorNode) DETACH DELETE n")

    def _create_restaurant(self, tx, r: Restaurant) -> None:
        tx.run(
            """
            MERGE (n:AuditorNode:Restaurant {name: $name})
            SET n.restaurant_id = $rid,
                n.cuisine_type  = $cuisine,
                n.borough       = $borough
            """,
            name=r.name, rid=r.restaurant_id,
            cuisine=r.cuisine_type, borough=r.borough,
        )

    def _create_owner(self, tx, o: Owner) -> None:
        tx.run(
            """
            MERGE (n:AuditorNode:Owner {name: $name})
            SET n.owner_id = $oid, n.entity_type = $etype
            """,
            name=o.name, oid=o.owner_id, etype=o.entity_type,
        )

    def _create_chain(self, tx, c: Chain) -> None:
        tx.run(
            """
            MERGE (n:AuditorNode:Chain {name: $name})
            SET n.chain_id = $cid, n.segment = $segment
            """,
            name=c.name, cid=c.chain_id, segment=c.segment,
        )

    def _create_supplier(self, tx, s: Supplier) -> None:
        tx.run(
            """
            MERGE (n:AuditorNode:Supplier {name: $name})
            SET n.supplier_id  = $sid,
                n.category     = $category,
                n.recall_count = $recalls
            """,
            name=s.name, sid=s.supplier_id,
            category=s.category, recalls=s.recall_count,
        )

    def _create_edge(self, tx, src: str, dst: str, rel: str, attrs: Dict) -> None:
        safe_attrs = {k: v for k, v in attrs.items() if isinstance(v, (str, int, float, bool))}
        query = (
            f"MATCH (a:AuditorNode {{name: $src}}), (b:AuditorNode {{name: $dst}}) "
            f"MERGE (a)-[r:{rel}]->(b) "
            f"SET r += $attrs"
        )
        tx.run(query, src=src, dst=dst, attrs=safe_attrs)

    def build_and_save(
        self,
        restaurants: List[Restaurant],
        owners:    List[Owner],
        chains:    List[Chain],
        suppliers: List[Supplier],
        edges:     List[Tuple],
    ) -> None:
        with self.driver.session() as session:
            session.execute_write(self._wipe_sandbox)
            for r in restaurants:
                session.execute_write(self._create_restaurant, r)
            for o in owners:
                session.execute_write(self._create_owner, o)
            for c in chains:
                session.execute_write(self._create_chain, c)
            for s in suppliers:
                session.execute_write(self._create_supplier, s)
            for src, dst, rel, attrs in edges:
                session.execute_write(self._create_edge, src, dst, rel, attrs)

            node_count = session.run(
                "MATCH (n:AuditorNode) RETURN count(n) AS c"
            ).single()["c"]
            edge_count = session.run(
                "MATCH (:AuditorNode)-[r]->(:AuditorNode) RETURN count(r) AS c"
            ).single()["c"]

        print(f"  [Neo4j]   {node_count} nodes, {edge_count} edges written")
        print(f"  [Neo4j]   Browser: http://localhost:7474  (neo4j / {NEO4J_PASSWORD})")

    def verify(self) -> None:
        with self.driver.session() as session:
            rows = session.run(
                "MATCH (:AuditorNode)-[r]->(:AuditorNode) "
                "RETURN type(r) AS rel, count(r) AS cnt "
                "ORDER BY cnt DESC"
            ).data()
        breakdown = {row["rel"]: row["cnt"] for row in rows}
        print(f"  [Neo4j]   Verified: edge breakdown → {breakdown}")


# ════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "═" * 60)
    print("  SYNTHETIC RESTAURANT FACTORY  —  initializing sandbox")
    print("═" * 60 + "\n")

    DB_DIR.mkdir(exist_ok=True)

    # ── 1. SQLite ────────────────────────────────────────────────────
    print("[ 1/3 ] SQLite — The Math")
    sql = SQLiteManager(SQLITE_PATH)
    sql.setup_and_populate(RESTAURANTS, INSPECTIONS)
    sql.verify()

    # ── 2. FAISS ─────────────────────────────────────────────────────
    print("\n[ 2/3 ] FAISS  — The Lore")
    vec = FAISSManager(FAISS_PATH, META_PATH)
    vec.embed_and_store(REVIEW_CORPUS)
    vec.verify()

    # ── 3. Graph ─────────────────────────────────────────────────────
    print("\n[ 3/3 ] Neo4j  — The Web")
    grph = GraphManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        grph.build_and_save(RESTAURANTS, OWNERS, CHAINS, SUPPLIERS, GRAPH_EDGES)
        grph.verify()
    finally:
        grph.close()

    print("\n" + "═" * 60)
    print("  SANDBOX FULLY INITIALIZED")
    print("═" * 60)
    print(f"  SQLite   →  {SQLITE_PATH}")
    print(f"  FAISS    →  {FAISS_PATH}")
    print(f"  Metadata →  {META_PATH}")
    print(f"  Neo4j    →  {NEO4J_URI}  (browser: http://localhost:7474)")
    print()
    print("  Next: run  python rag_engines.py  to verify retrieval.")
    print()


if __name__ == "__main__":
    main()
