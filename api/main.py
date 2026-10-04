"""
Travora HTTP API.

    uvicorn api.main:app --reload --port 8000

Contract
--------
    POST /api/v1/itinerary   -> 200 itinerary, or 422 with a stated conflict
    GET  /api/v1/meta        -> categories, districts, defaults
    GET  /health             -> liveness plus graph counts

The 422 path is a feature, not an error case. Reporting honestly that no
feasible itinerary exists under the stated constraints is the behaviour that
distinguishes this system from one that returns a plausible-looking plan it
cannot justify.

The Neo4j driver is created once at startup and shared. Opening a driver per
request would dominate response time, which is metric M5 in the evaluation.
"""
from __future__ import annotations
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from etl.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OSRM_URL
from retrieval.models import TripRequest, Category
from solver.models import SolverConfig
from agents.preference import PreferenceAgent, groq_completion
from agents.sustainability import SustainabilityAgent
from agents.pipeline import TravoraPipeline

log = logging.getLogger(__name__)

STATE: dict = {"driver": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # A driver already in STATE is respected rather than replaced, so a caller
    # (or a test) can inject one. Overwriting it here would make the service
    # untestable without a live database.
    owns_driver = False
    if STATE.get("driver") is None:
        if not NEO4J_URI or not NEO4J_PASSWORD:
            log.error("NEO4J_URI / NEO4J_PASSWORD not set; requests will fail")
        else:
            from neo4j import GraphDatabase
            STATE["driver"] = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_pool_size=20)
            owns_driver = True
            log.info("Neo4j driver ready")

    # Built once: model discovery and key-pool loading are startup costs, not
    # per-request ones.
    if "completion" not in STATE:
        STATE["completion"] = groq_completion(model=os.getenv("GROQ_MODEL"))
    log.info("Preference model %s",
             "configured" if STATE["completion"] else "unavailable (fallback)")
    yield
    if owns_driver and STATE["driver"]:
        STATE["driver"].close()


app = FastAPI(
    title="Travora API",
    description="Agentic AI decision support for sustainable tourism in Sri Lanka",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
class ItineraryRequest(BaseModel):
    """Wire format. Kept separate from TripRequest so the HTTP contract can
    evolve without changing the internal model."""
    days: int = Field(ge=1, le=21, examples=[5])
    budget_lkr: float = Field(gt=0, examples=[120000])
    interests: list[Category] = Field(min_length=1, examples=[["heritage"]])
    party_size: int = Field(default=1, ge=1, le=20)
    start_date: Optional[date] = None
    start_lat: float = Field(default=7.2906, ge=5.85, le=9.90)
    start_name: Optional[str] = Field(default=None, max_length=80)
    start_lon: float = Field(default=80.6337, ge=79.60, le=81.95)
    w_pref: float = Field(default=0.5, ge=0.0, le=1.0)
    max_slots_per_day: int = Field(default=3, ge=1, le=5)
    seed: int = 42
    use_llm: bool = True
    alternatives: int = Field(default=0, ge=0, le=2)

    def to_trip(self) -> TripRequest:
        return TripRequest(
            start_date=self.start_date or (date.today() + timedelta(days=14)),
            days=self.days, budget_lkr=self.budget_lkr,
            party_size=self.party_size, interests=self.interests,
            start_lat=self.start_lat, start_lon=self.start_lon,
            w_pref=self.w_pref, seed=self.seed)


def _session():
    if STATE["driver"] is None:
        raise HTTPException(503, "Knowledge graph unavailable. Check NEO4J_URI "
                                 "and NEO4J_PASSWORD.")
    return STATE["driver"].session()


# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if STATE["driver"] is None:
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "graph": "unavailable"})
    try:
        with STATE["driver"].session() as s:
            rec = s.run("MATCH (p:POI) WITH count(p) AS pois "
                        "OPTIONAL MATCH (a:Accommodation) "
                        "RETURN pois, count(a) AS accommodation").single()
        return {"status": "ok", "pois": rec["pois"],
                "accommodation": rec["accommodation"],
                "preference_model": bool(STATE["completion"])}
    except Exception as e:                              # noqa: BLE001
        return JSONResponse(status_code=503,
                            content={"status": "degraded", "detail": str(e)[:200]})


@app.get("/api/v1/meta")
def meta():
    """Everything a client needs to build the trip form."""
    districts = []
    if STATE["driver"]:
        try:
            with STATE["driver"].session() as s:
                districts = [r["name"] for r in s.run(
                    "MATCH (d:District) RETURN d.name AS name ORDER BY name")]
        except Exception:                               # noqa: BLE001
            pass
    return {
        "categories": [c.value for c in Category],
        "districts": districts,
        "defaults": {"days": 5, "budget_lkr": 120000, "party_size": 2,
                     "w_pref": 0.5, "max_slots_per_day": 3},
        "limits": {"max_days": 21, "max_party_size": 20},
    }


PLACES_QUERY = """
MATCH (p:POI)
WHERE ($categories IS NULL OR p.category IN $categories)
  AND ($district   IS NULL OR p.district = $district)
  AND ($search     IS NULL OR toLower(p.name) CONTAINS toLower($search))
  AND p.prominence >= $min_prominence
RETURN p.poi_id AS poi_id, p.name AS name, p.category AS category,
       p.district AS district, p.lat AS lat, p.lon AS lon,
       p.entry_fee_lkr AS entry_fee_lkr, p.typical_dwell_min AS dwell_min,
       p.prominence AS prominence,
       p.wikipedia AS wikipedia,
       p.image_url AS image_url, p.image_source AS image_source, p.image_credit AS image_credit, p.image_page AS image_page,
       coalesce(p.popularity_index, 0.0) AS popularity_index,
       (p.popularity_source = 'wikipedia') AS popularity_known,
       coalesce(p.rail_access_score, 0.0) AS rail_access_score,
       coalesce(p.nearest_station_km, 999.0) AS nearest_station_km,
       coalesce(p.hours_estimated, true) AS hours_estimated,
       coalesce(p.fee_estimated, true) AS fee_estimated
ORDER BY p.prominence DESC, p.name ASC
SKIP $skip LIMIT $limit
"""

PLACES_COUNT = """
MATCH (p:POI)
WHERE ($categories IS NULL OR p.category IN $categories)
  AND ($district   IS NULL OR p.district = $district)
  AND ($search     IS NULL OR toLower(p.name) CONTAINS toLower($search))
  AND p.prominence >= $min_prominence
RETURN count(p) AS total
"""


@app.get("/api/v1/places")
def list_places(
    page: int = 1,
    size: int = 20,
    category: Optional[str] = None,
    district: Optional[str] = None,
    q: Optional[str] = None,
    min_prominence: float = 0.0,
):
    """
    Paginated place browser.

    Paging is server-side with SKIP/LIMIT rather than fetching everything and
    slicing on the client: the graph holds 2,245 places and a mobile client on a
    slow connection should never have to receive them all to show twenty.
    """
    page = max(1, page)
    size = max(1, min(size, 100))          # capped so one request cannot pull the graph

    params = {
        "categories": [c.strip() for c in category.split(",")] if category else None,
        "district": district or None,
        "search": q or None,
        "min_prominence": float(min_prominence),
        "skip": (page - 1) * size,
        "limit": size,
    }

    with _session() as session:
        total = session.run(PLACES_COUNT, **params).single()["total"]
        rows = [dict(r) for r in session.run(PLACES_QUERY, **params)]

    pages = (total + size - 1) // size if total else 0
    return {
        "items": [{
            "poi_id": r["poi_id"], "name": r["name"], "category": r["category"],
            "district": r["district"], "lat": r["lat"], "lon": r["lon"],
            "entry_fee_lkr": r["entry_fee_lkr"], "dwell_min": r["dwell_min"],
            "prominence": round(r["prominence"] or 0.0, 3),
            **image_fields(r),
            "popularity_index": round(r["popularity_index"] or 0.0, 3),
            "popularity_known": bool(r["popularity_known"]),
            "rail_access_score": round(r["rail_access_score"] or 0.0, 2),
            "nearest_station_km": round(r["nearest_station_km"] or 999.0, 1),
            "estimates": {"opening_hours": bool(r["hours_estimated"]),
                          "entry_fee": bool(r["fee_estimated"])},
        } for r in rows],
        "page": page, "size": size, "total": total, "pages": pages,
        "has_next": page < pages, "has_prev": page > 1,
    }


NEARBY_QUERY = """
WITH point({latitude: $lat, longitude: $lon}) AS here
MATCH (p:POI)
WHERE point.distance(p.location, here) <= $radius_m
  AND ($categories IS NULL OR p.category IN $categories)
  AND p.prominence >= $min_prominence
WITH p, point.distance(p.location, here) AS d
RETURN p.poi_id AS poi_id, p.name AS name, p.category AS category,
       p.district AS district, p.lat AS lat, p.lon AS lon,
       p.entry_fee_lkr AS entry_fee_lkr, p.typical_dwell_min AS dwell_min,
       p.prominence AS prominence, p.wikipedia AS wikipedia,
       p.image_url AS image_url, p.image_source AS image_source, p.image_credit AS image_credit, p.image_page AS image_page,
       coalesce(p.popularity_index, 0.0) AS popularity_index,
       d / 1000.0 AS distance_km
ORDER BY d ASC
LIMIT $limit
"""


# Declared before /places/{poi_id:path}, which would otherwise match "nearby".
@app.get("/api/v1/places/nearby")
def nearby_places(
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    limit: int = 30,
    category: Optional[str] = None,
    min_prominence: float = 0.0,
    images: bool = True,
):
    """Places within radius_km of a point, nearest first, via the POI point index."""
    radius_km = max(0.1, min(radius_km, 100.0))
    limit = max(1, min(limit, 100))
    params = {
        "lat": lat, "lon": lon, "radius_m": radius_km * 1000.0,
        "categories": [c.strip() for c in category.split(",")] if category else None,
        "min_prominence": float(min_prominence), "limit": limit,
    }
    with _session() as session:
        rows = [dict(r) for r in session.run(NEARBY_QUERY, **params)]
    return {
        "items": [{
            "poi_id": r["poi_id"], "name": r["name"], "category": r["category"],
            "district": r["district"], "lat": r["lat"], "lon": r["lon"],
            "entry_fee_lkr": r["entry_fee_lkr"], "dwell_min": r["dwell_min"],
            "prominence": round(r["prominence"] or 0.0, 3),
            "popularity_index": round(r["popularity_index"] or 0.0, 3),
            "distance_km": round(r["distance_km"], 2),
            # Thumbnails cost one Wikipedia call per uncached title; AR skips them.
            **(image_fields(r) if images else {"image_url": None}),
        } for r in rows],
        "radius_km": radius_km,
        "count": len(rows),
    }


@app.get("/api/v1/places/{poi_id:path}")
def get_place(poi_id: str):
    """Single place. The path converter allows ids such as `node/12345`."""
    with _session() as session:
        rec = session.run(
            "MATCH (p:POI {poi_id: $id}) RETURN p.poi_id AS poi_id, p.name AS name, "
            "p.category AS category, p.district AS district, p.lat AS lat, "
            "p.lon AS lon, p.entry_fee_lkr AS entry_fee_lkr, "
            "p.typical_dwell_min AS dwell_min, p.prominence AS prominence, "
            "p.wikipedia AS wikipedia, p.image_url AS image_url, p.image_source AS image_source, p.image_credit AS image_credit, p.image_page AS image_page, "
            "p.open_min AS open_min, p.close_min AS close_min, "
            "coalesce(p.rail_access_score,0.0) AS rail_access_score, "
            "coalesce(p.nearest_station_km,999.0) AS nearest_station_km, "
            "coalesce(p.hours_estimated,true) AS hours_estimated, "
            "coalesce(p.fee_estimated,true) AS fee_estimated",
            id=poi_id).single()
    if rec is None:
        raise HTTPException(404, f"No place with id {poi_id}")
    d = dict(rec)
    d.update(image_fields(d, width=1024))
    d.pop("wikipedia", None)
    d["estimates"] = {"opening_hours": bool(d.pop("hours_estimated")),
                      "entry_fee": bool(d.pop("fee_estimated"))}
    return d


# ---------------------------------------------------------------------------
# Images
#
# Photographs come from Wikimedia Commons, resolved through the Wikipedia title
# already stored on each place. This needs no API key, the licence permits reuse
# with attribution, and no image is stored: only the URL is passed to the client.
# Commercial photo APIs were not used because their terms permit display but not
# retention, and a cached URL that expires would leave broken images in a demo.
# ---------------------------------------------------------------------------
_IMAGE_CACHE: dict[str, Optional[str]] = {}
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


def image_fields(r: dict, width: int = 640) -> dict:
    """
    The photo for a place, with where it came from.

    Prefers the photo stored by `etl.images` (Wikipedia lead image, else a
    geotagged Commons photo nearby). Before that step has run, falls back to a
    live Wikipedia lookup. `image_source` lets a client label nearby photos,
    which may show the surroundings rather than the place itself.
    """
    url = r.get("image_url")
    if url:
        if width != 640:
            url = re.sub(r"/640px-", f"/{width}px-", url)
        return {"image_url": url, "image_source": r.get("image_source"),
                "image_credit": r.get("image_credit"), "image_page": r.get("image_page")}
    live = wiki_thumbnail(r.get("wikipedia"), width=width)
    return {"image_url": live, "image_source": "wikipedia" if live else None,
            "image_credit": "Wikipedia / Wikimedia Commons" if live else None,
            "image_page": None}


def wiki_thumbnail(wikipedia_tag: Optional[str], width: int = 640) -> Optional[str]:
    """Thumbnail URL for a place, or None. Results are cached for the process."""
    if not wikipedia_tag or not isinstance(wikipedia_tag, str):
        return None
    title = (wikipedia_tag.split(":", 1)[1] if ":" in wikipedia_tag
             else wikipedia_tag).strip().replace(" ", "_")
    if not title:
        return None
    if title in _IMAGE_CACHE:
        return _IMAGE_CACHE[title]

    url = None
    try:
        import requests
        from urllib.parse import quote
        r = requests.get(WIKI_SUMMARY.format(title=quote(title, safe="")),
                         headers={"User-Agent": "TravoraResearch/1.0 (academic)"},
                         timeout=6)
        if r.status_code == 200:
            data = r.json()
            thumb = (data.get("thumbnail") or {}).get("source")
            if thumb:
                url = thumb.replace("/thumb/", "/thumb/") if "/thumb/" in thumb else thumb
                # ask for a larger render than the default 320 px
                import re as _re
                url = _re.sub(r"/(\d+)px-", f"/{width}px-", url)
    except Exception:                                   # noqa: BLE001
        url = None

    _IMAGE_CACHE[title] = url
    return url


ACCOMMODATION_QUERY = """
MATCH (a:Accommodation)
WHERE ($types    IS NULL OR a.sltda_type IN $types)
  AND ($district IS NULL OR a.district = $district)
  AND ($search   IS NULL OR toLower(a.name) CONTAINS toLower($search))
  AND a.sustainability_score >= $min_sustainability
RETURN a.acc_id AS acc_id, a.name AS name, a.sltda_type AS type,
       a.district AS district, a.rooms AS rooms, a.grade AS grade,
       a.lat AS lat, a.lon AS lon,
       a.sustainability_score AS sustainability_score,
       coalesce(a.rail_access_score, 0.0) AS rail_access_score,
       coalesce(a.nearest_station_km, 999.0) AS nearest_station_km,
       coalesce(a.location_estimated, false) AS location_estimated
ORDER BY a.sustainability_score DESC, a.name ASC
SKIP $skip LIMIT $limit
"""

ACCOMMODATION_COUNT = """
MATCH (a:Accommodation)
WHERE ($types    IS NULL OR a.sltda_type IN $types)
  AND ($district IS NULL OR a.district = $district)
  AND ($search   IS NULL OR toLower(a.name) CONTAINS toLower($search))
  AND a.sustainability_score >= $min_sustainability
RETURN count(a) AS total
"""

# Indicative nightly rates by register category, in LKR. The register records
# category and room count but not price, so these are estimates and are labelled
# as such wherever they are shown.
RATE_BY_TYPE = {
    "Home Stay Units": 4000, "Heritage Homes": 12000, "Heritage Bungalows": 14000,
    "Rented Homes": 6000, "Bangalows": 8000, "Rented Apartments": 7000,
    "Guest Houses": 6500, "Boutique Villas": 22000, "Boutique Hotels": 25000,
    "Tourist Hotels": 15000, "Classified Hotels( 1-5 Star)": 35000,
}


@app.get("/api/v1/accommodation")
def list_accommodation(
    page: int = 1,
    size: int = 20,
    type: Optional[str] = None,
    district: Optional[str] = None,
    q: Optional[str] = None,
    min_sustainability: float = 0.0,
):
    """Paginated browser over the registered accommodation."""
    page = max(1, page)
    size = max(1, min(size, 100))
    params = {
        "types": [t.strip() for t in type.split(",")] if type else None,
        "district": district or None,
        "search": q or None,
        "min_sustainability": float(min_sustainability),
        "skip": (page - 1) * size,
        "limit": size,
    }
    with _session() as session:
        total = session.run(ACCOMMODATION_COUNT, **params).single()["total"]
        rows = [dict(r) for r in session.run(ACCOMMODATION_QUERY, **params)]

    pages = (total + size - 1) // size if total else 0
    return {
        "items": [{
            "acc_id": r["acc_id"], "name": r["name"], "type": r["type"],
            "district": r["district"], "rooms": r["rooms"], "grade": r["grade"],
            "lat": r["lat"], "lon": r["lon"],
            "sustainability_score": round(r["sustainability_score"] or 0.0, 3),
            "rail_access_score": round(r["rail_access_score"] or 0.0, 2),
            "nearest_station_km": round(r["nearest_station_km"] or 999.0, 1),
            "est_rate_lkr": RATE_BY_TYPE.get(r["type"], 10000),
            "estimates": {"nightly_rate": True,
                          "location": bool(r["location_estimated"])},
        } for r in rows],
        "page": page, "size": size, "total": total, "pages": pages,
        "has_next": page < pages, "has_prev": page > 1,
    }


HOTEL_TYPES = ["Tourist Hotels", "Classified Hotels( 1-5 Star)", "Boutique Hotels",
               "Boutique Villas"]

NEARBY_STAYS = """
WITH point({latitude: $lat, longitude: $lon}) AS here
MATCH (a:Accommodation)
WHERE point.distance(a.location, here) <= $radius_m
  AND ($hotels IS NULL OR ($hotels AND a.sltda_type IN $hotel_types)
                       OR (NOT $hotels AND NOT a.sltda_type IN $hotel_types))
WITH a, point.distance(a.location, here) AS d
RETURN a.acc_id AS acc_id, a.name AS name, a.sltda_type AS type, a.district AS district,
       a.rooms AS rooms, a.lat AS lat, a.lon AS lon,
       a.sustainability_score AS sustainability_score,
       coalesce(a.location_estimated, false) AS location_estimated,
       d / 1000.0 AS distance_km
ORDER BY d ASC
LIMIT $limit
"""


@app.get("/api/v1/accommodation/nearby")
def nearby_stays(lat: float, lon: float, radius_km: float = 25.0, limit: int = 20,
                 group: Optional[str] = None):
    """
    Registered stays near a point, nearest first. group=hotels or group=stays
    (home stays, guest houses, bungalows and similar). About a third of the
    register has no published coordinates and was placed at its division or
    district centre, so those distances are approximate and flagged.
    """
    radius_km = max(0.5, min(radius_km, 100.0))
    limit = max(1, min(limit, 50))
    hotels = {"hotels": True, "stays": False}.get((group or "").lower())
    with _session() as session:
        rows = [dict(r) for r in session.run(
            NEARBY_STAYS, lat=lat, lon=lon, radius_m=radius_km * 1000.0, limit=limit,
            hotels=hotels, hotel_types=HOTEL_TYPES)]
    return {
        "items": [{
            "acc_id": r["acc_id"], "name": r["name"], "type": r["type"],
            "district": r["district"], "rooms": r["rooms"], "lat": r["lat"], "lon": r["lon"],
            "sustainability_score": round(r["sustainability_score"] or 0.0, 3),
            "distance_km": round(r["distance_km"], 2),
            "est_rate_lkr": RATE_BY_TYPE.get(r["type"], 10000),
            "estimates": {"nightly_rate": True, "location": bool(r["location_estimated"])},
        } for r in rows],
        "radius_km": radius_km, "count": len(rows),
    }


@app.get("/api/v1/accommodation/types")
def accommodation_types():
    """Register categories with their sustainability score, for filter chips."""
    with _session() as session:
        rows = [dict(r) for r in session.run(
            "MATCH (a:Accommodation) "
            "RETURN a.sltda_type AS type, count(a) AS n, "
            "avg(a.sustainability_score) AS score "
            "ORDER BY score DESC")]
    return [{"type": r["type"], "count": r["n"],
             "sustainability_score": round(r["score"] or 0.0, 3),
             "est_rate_lkr": RATE_BY_TYPE.get(r["type"], 10000)} for r in rows]


class ParseRequest(BaseModel):
    text: str = Field(min_length=3, max_length=1000)
    use_llm: bool = True


@app.post("/api/v1/plan/parse")
def parse_plan(req: ParseRequest):
    """
    Free text -> form fields. The result pre-fills a form the traveller
    confirms; it never reaches the solver without that confirmation.
    """
    from agents.parse import parse_trip_text
    return parse_trip_text(req.text, STATE.get("completion") if req.use_llm else None)


class ParseLogEntry(BaseModel):
    raw_text: str = Field(max_length=1000)
    source: str
    parsed: dict
    confirmed: dict
    edited_fields: list[str] = []


PARSE_LOG = Path(__file__).resolve().parent.parent / "data" / "research" / "parse_log.jsonl"


@app.post("/api/v1/research/parse-log", status_code=204)
def log_parse(entry: ParseLogEntry):
    """
    Records what the parser proposed against what the traveller confirmed.
    The proportion of fields left unedited is the parse-accuracy measure.
    """
    PARSE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PARSE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                            **entry.model_dump()}) + "\n")


@app.post("/api/v1/itinerary")
def create_itinerary(req: ItineraryRequest):
    try:
        trip = req.to_trip()

        completion = STATE["completion"] if req.use_llm else None
        pipeline = TravoraPipeline(
            session=None,
            preference=PreferenceAgent(complete=completion),
            sustainability=SustainabilityAgent(),
            solver_config=SolverConfig(max_slots_per_day=req.max_slots_per_day),
        )

        from solver.logistics import build_legs, emissions_summary, choose_lodging, retime
        # Road routing can be switched off (tests, or offline demos); legs are
        # then estimated and marked as such.
        osrm = OSRM_URL if os.getenv("TRAVORA_ROAD_ROUTING", "1") == "1" else None

        plans = []
        with _session() as session:
            pipeline.retriever.session = session
            result = pipeline.run(trip, alternatives=req.alternatives)

            if result.solver.feasible:
                by_id = {c.poi_id: c for c in pipeline.retriever._last_candidates}
                all_plans = [result.solver.itinerary, *result.solver.alternatives]
                photos = _stop_photos(session, {s.poi_id for it in all_plans for s in it.stops})
                for it in all_plans:
                    stops_by_day = [d.stops for d in it.days]
                    legs = build_legs(stops_by_day, by_id, start=(trip.start_lat, trip.start_lon),
                                      start_name=req.start_name, osrm_url=osrm,
                                      party_size=trip.party_size)
                    timing = retime(stops_by_day, legs, by_id)
                    # Accommodation is chosen after the stops are fixed: a hotel
                    # changes where the traveller sleeps, not which places are reachable.
                    stays, lodging = choose_lodging(
                        session, it.days, trip.budget_lkr * 0.45, trip.party_size, by_id=by_id)
                    plans.append(_plan_payload(it, result, legs,
                                               emissions_summary(legs, trip.party_size,
                                                                 days=len(it.days),
                                                                 place_mean=it.s_sust),
                                               stays, lodging, photos, by_id, legs, timing))

        audit = result.audit.model_dump()
        res = result.solver

        if not res.feasible:
            # 422: the request was well formed but no plan satisfies it. The
            # conflict and the suggestion are the useful part of this response.
            return JSONResponse(status_code=422, content={
                "feasible": False,
                "request_id": result.request_id,
                "conflict": res.conflict.conflict.value if res.conflict else "unknown",
                "detail": res.conflict.detail if res.conflict else "",
                "binding_constraint": res.conflict.binding_constraint if res.conflict else None,
                "suggestion": res.conflict.suggestion if res.conflict else None,
                "audit": audit,
            })

        return {
            "feasible": True,
            "request_id": result.request_id,
            **plans[0],
            "alternatives": plans[1:],
            "audit": audit,
            "data_provenance": {
                "poi_source": "OpenStreetMap (ODbL) + SLTDA + Wikidata",
                "accommodation_source": "SLTDA registered accommodation",
                "note": "Opening hours and entry fees are largely imputed from "
                        "category defaults; per-stop flags mark which are real.",
            },
        }
    except Exception as e:
        log.exception("Error in create_itinerary")
        return JSONResponse(status_code=500, content={"detail": f"Failed to plan itinerary: {str(e)}"})



def _stop_photos(session, ids: set[str]) -> dict[str, dict]:
    """Stored photo fields for the planned stops, in one query."""
    if not ids:
        return {}
    rows = session.run(
        "MATCH (p:POI) WHERE p.poi_id IN $ids "
        "RETURN p.poi_id AS poi_id, p.image_url AS image_url, "
        "p.image_source AS image_source, p.image_credit AS image_credit",
        ids=list(ids))
    out = {}
    for r in map(dict, rows):
        if r.get("poi_id"):
            out[r["poi_id"]] = {"image_url": r.get("image_url"),
                                "image_source": r.get("image_source"),
                                "image_credit": r.get("image_credit")}
    return out


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _transport(l) -> Optional[dict]:
    if l is None:
        return None
    return {"mode": l.mode, "label": l.mode_label, "distance_km": l.distance_km,
            "duration_min": l.duration_min, "emissions_g": l.emissions_g, "why": l.rationale,
            "from_name": l.from_name, "kind": l.kind, "basis": l.basis,
            "from_station": l.stations[0] if l.stations else None,
            "to_station": l.stations[1] if l.stations else None}


def _plan_payload(it, result, legs, emissions, stays, lodging, photos=None,
                  by_id=None, _legs=None, timing=None) -> dict:
    photos = photos or {}
    by_id = by_id or {}
    into = {l.to_id: l for l in legs}
    times, overflow, late = timing or ({}, {}, set())
    return {
        "itinerary": {
            "total_cost_lkr": it.total_cost_lkr,
            "utility": it.utility,
            "s_pref": it.s_pref,
            # Trip sustainability = 0.6 places + 0.4 journey (measured legs).
            # s_place and s_journey are its two parts, shown separately.
            "s_sust": emissions.get("trip_sustainability", it.s_sust),
            "s_place": it.s_sust,
            "s_journey": emissions.get("journey_score"),
            "kg_co2e_per_traveller_day": emissions.get("kg_co2e_per_traveller_day"),
            "optimal": it.optimal,
            "districts": sorted(it.districts),
            "days": [{
                "day": d.day,
                "cost_lkr": d.cost_lkr,
                # Minutes the day runs past 18:00 once legs are measured; 0 = fits.
                "overflow_min": overflow.get(d.day, 0),
                "starts_from": (into[d.stops[0].poi_id].from_name
                                if d.stops and d.stops[0].poi_id in into else None),
                "stops": [{
                    "poi_id": s.poi_id,
                    "name": s.name,
                    "day": d.day,
                    "slot": s.slot,
                    "lat": getattr(by_id.get(s.poi_id), "lat", None),
                    "lon": getattr(by_id.get(s.poi_id), "lon", None),
                    **photos.get(s.poi_id, {"image_url": None, "image_source": None,
                                            "image_credit": None}),
                    "category": s.category,
                    "district": s.district,
                    "arrive": _hhmm(times[s.poi_id][0]) if s.poi_id in times else s.arrive_hhmm,
                    "depart": _hhmm(times[s.poi_id][1]) if s.poi_id in times else s.depart_hhmm,
                    "closes_before_departure": s.poi_id in late,
                    "dwell_min": s.dwell_min,
                    "cost_lkr": s.cost_lkr,
                    "travel_from_prev_min": (into[s.poi_id].duration_min if s.poi_id in into
                                             else 0.0),
                    "preference_score": result.preference_scores.get(s.poi_id),
                    "sustainability_score": result.sustainability_scores.get(s.poi_id),
                    "transport": _transport(into.get(s.poi_id)),
                    # Provenance travels with every stop so a client can show
                    # which figures are real and which are estimates.
                    "estimates": {
                        "opening_hours": not s.hours_enforced,
                        "entry_fee": not s.fee_enforced,
                        "travel_time": (into[s.poi_id].estimated if s.poi_id in into
                                        else False),
                    },
                } for s in d.stops],
            } for d in it.days],
        },
        "accommodation": {
            "stays": [{
                "night": st.night, "acc_id": st.acc_id, "name": st.name,
                "type": st.type, "district": st.district, "rooms": st.rooms,
                "sustainability_score": st.sustainability_score,
                "distance_km": st.distance_km,
                "est_rate_lkr": st.est_rate_lkr,
                "estimates": {"nightly_rate": st.rate_estimated,
                              "location": st.location_estimated},
            } for st in stays],
            "summary": lodging,
        },
        "transport": {
            "legs": [{
                "from_id": l.from_id, "to_id": l.to_id, **_transport(l),
            } for l in legs],
            "summary": emissions,
        },
    }


@app.exception_handler(ValidationError)
def on_validation_error(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400,
                        content={"detail": "Invalid request", "errors": exc.errors()})
