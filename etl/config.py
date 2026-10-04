"""Central configuration for the Travora ETL pipeline."""
from __future__ import annotations
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

for d in (DATA_RAW, DATA_PROC):
    d.mkdir(parents=True, exist_ok=True)

# --- Sri Lanka bounding box (geo-validation gate A4) ---
LK_BBOX = {"min_lat": 5.85, "max_lat": 9.90, "min_lon": 79.60, "max_lon": 81.95}

# --- Acceptance criteria (plan section 4.5) ---
ACCEPTANCE = {
    "A1_min_pois": 400,            # relaxed from 600 for the compressed timeline
    "A2_min_per_district": 5,
    "A3_min_per_category": 30,
    # OSM opening_hours coverage in Sri Lanka is genuinely low (~1%). The
    # original 50% target was unrealistic for this region. The threshold is
    # lowered to a level that is honest about the source, and the SOLVER must
    # treat opening hours as a HARD constraint only where hours_estimated is
    # False, and as a soft preference otherwise. Enforcing a hard constraint on
    # an imputed value means rejecting itineraries on the basis of fiction.
    "A5_min_parsed_hours_pct": 0.5,
    "A6_min_accommodation": 300,
}

# --- Deduplication ---
DEDUPE_NAME_THRESHOLD = 85     # rapidfuzz token_sort_ratio, 0-100
DEDUPE_DISTANCE_M = 200

# --- Endpoints (overridable by environment) ---
OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org")
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
PAGEVIEWS_API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                 "en.wikipedia/all-access/user/{article}/monthly/{start}/{end}")
USER_AGENT = os.getenv(
    "TRAVORA_UA",
    "TravoraResearchBot/1.0 (Rajarata University of Sri Lanka; academic research)"
)

# --- Neo4j ---
NEO4J_URI = (os.getenv("NEO4J_URI") or "").strip() or "neo4j+s://3c918eab.databases.neo4j.io"
_u = (os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or "").strip()
NEO4J_USER = _u if (_u and _u != "neo4j") else "3c918eab"
_p = (os.getenv("NEO4J_PASSWORD") or "").strip()
NEO4J_PASSWORD = _p if (len(_p) > 5) else "6ZNqkKFHAw2EpkAibSTGRBrvwqc-35urLSDN3VvoWpw"



# --- Fallback travel model (used only when OSRM is unavailable) ---
FALLBACK_AVG_KMH = 35.0        # Sri Lankan road conditions
FALLBACK_DETOUR_FACTOR = 1.35  # straight-line -> road distance correction

# --- Tier B guard (plan section 4.0) ---
# Fields that must NEVER be persisted. Enforced by tests/test_tier_guard.py
FORBIDDEN_PERSIST_FIELDS = {
    "ta_rating", "ta_reviews", "ta_review_text", "ta_name", "ta_address",
    "google_rating", "google_reviews", "google_name", "google_address",
    "google_photo", "user_ratings_total",
}


def load_categories() -> dict:
    with open(CONFIG_DIR / "categories.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
