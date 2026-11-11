# local_swarm.py
# =============================================================================
# Bear Synthetic Consumer Research Platform
# =============================================================================
# A bear-themed synthetic survey panel that runs locally on Ollama via CrewAI.
#
# Each bear is one synthetic consumer with a full persona. A run flows through
# four phases:
#
#   1. GENERATION  - build the panel from three sources (synthetic / custom /
#                    historical), optionally CALIBRATED to an uploaded dataset.
#   2. SCREENING   - the Screening Room decides which bears qualify.
#   3. SURVEY      - qualified bears answer the survey in parallel, in-character.
#   4. INSIGHTS    - the Survey Insights Bear aggregates everything, and the
#                    platform validates the synthetic panel against ground truth.
#
# Major subsystems
# ----------------
#   * Ground-truth ingestion (CSV / XLSX / JSON) + distribution modelling
#   * Calibrated persona generation + audience matching
#   * Response calibration from observed patterns
#   * Custom Character Studio + Persona Library (save / edit / tag / lock)
#   * Hybrid panel mode (mix synthetic / custom / historical by proportion)
#   * Validation dashboard + insight confidence score
#
# The roster is NEVER hardcoded. Every run builds a fresh set of bears sized by
# the user's panel_size and source mix.
# =============================================================================

import copy
import csv
import io
import json
import math
import os
import random
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Disable CrewAI's interactive trace prompt (the "[y/N] 20s timeout") and telemetry
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from crewai import Agent, Task, Crew, Process, LLM

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LOCAL MODEL CONFIG
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_PARALLEL_AGENTS = max(1, int(os.getenv("MAX_PARALLEL_AGENTS", "6")))
MAX_PANEL_SIZE = max(1, int(os.getenv("MAX_PANEL_SIZE", "500")))

# Rough cost estimate ("what would this cost on a hosted model?"). $/1M tokens.
COST_PER_1M_PROMPT = float(os.getenv("COST_PER_1M_PROMPT", "0.15"))
COST_PER_1M_COMPLETION = float(os.getenv("COST_PER_1M_COMPLETION", "0.60"))


def _new_llm(temperature=0.8):
    return LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=temperature)


# ---------------------------------------------------------------------------
# PERSONA GENERATION BUILDING BLOCKS
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Emma", "Noah", "Olivia", "Lucas", "Sophia", "Liam", "Ava", "Ethan",
    "Isabella", "Mason", "Mia", "Logan", "Charlotte", "James", "Amelia",
    "Benjamin", "Harper", "Elijah", "Evelyn", "Oliver", "Abigail", "Jacob",
    "Emily", "William", "Ella", "Michael", "Scarlett", "Daniel", "Grace",
    "Henry", "Chloe", "Jackson", "Victoria", "Sebastian", "Aria", "Jack",
    "Lily", "Aiden", "Zoe", "Owen", "Nora", "Samuel", "Hannah", "Matthew",
    "Layla", "Joseph", "Riley", "David", "Aurora", "Carter", "Priya",
    "Wei", "Amir", "Fatima", "Diego", "Yuki", "Omar", "Leila", "Kwame", "Sana",
]

CANADIAN_LOCATIONS = [
    ("Toronto", "Ontario"), ("Ottawa", "Ontario"), ("Mississauga", "Ontario"),
    ("Hamilton", "Ontario"), ("London", "Ontario"), ("Montreal", "Quebec"),
    ("Quebec City", "Quebec"), ("Laval", "Quebec"), ("Vancouver", "British Columbia"),
    ("Surrey", "British Columbia"), ("Victoria", "British Columbia"),
    ("Calgary", "Alberta"), ("Edmonton", "Alberta"), ("Winnipeg", "Manitoba"),
    ("Saskatoon", "Saskatchewan"), ("Regina", "Saskatchewan"),
    ("Halifax", "Nova Scotia"), ("St. John's", "Newfoundland and Labrador"),
    ("Fredericton", "New Brunswick"), ("Charlottetown", "Prince Edward Island"),
]

OCCUPATIONS = [
    "Nurse", "Software developer", "Elementary teacher", "Retail manager",
    "Electrician", "Accountant", "Graphic designer", "Warehouse associate",
    "Marketing coordinator", "Chef", "Truck driver", "Dental hygienist",
    "Small business owner", "Financial analyst", "Social worker", "Barista",
    "Construction supervisor", "Pharmacist", "Real estate agent", "Librarian",
    "Data analyst", "HR specialist", "Plumber", "University student",
    "Personal trainer", "Bank teller", "Paramedic", "Freelance writer",
    "Mechanic", "Stay-at-home parent", "Retired", "Customer service rep",
]

EDUCATION_LEVELS = [
    "High school diploma", "College diploma", "Some college",
    "Bachelor's degree", "Master's degree", "Trade certification",
    "Doctorate", "Professional designation",
]

FAMILY_STATUSES = [
    "Single, no children", "Married, no children", "Married with young children",
    "Married with teenagers", "Common-law partnership", "Divorced, shared custody",
    "Single parent", "Widowed", "Empty nester", "Living with roommates",
]

INTEREST_POOL = [
    "cooking", "hiking", "gaming", "gardening", "photography", "cycling",
    "reading", "travel", "yoga", "woodworking", "baking", "running",
    "podcasts", "thrifting", "home renovation", "fishing", "knitting",
    "board games", "camping", "live music", "volunteering", "painting",
]

BRAND_POOL = [
    "Loblaws", "No Name", "President's Choice", "Costco", "Metro", "Sobeys",
    "Whole Foods", "Farm Boy", "Walmart", "Trader-style local co-op",
    "local farmers market", "Amazon Fresh", "store brands", "premium organic labels",
]

SHOPPING_FREQ = [
    "Multiple times a week", "Weekly big shop", "Every couple of weeks",
    "Mostly online with occasional top-ups", "Daily small trips",
]

RISK_TOLERANCE = ["Cautious", "Balanced", "Adventurous", "Very cautious", "Impulsive"]
SUSTAINABILITY = [
    "Highly committed to sustainability",
    "Sustainability-conscious when affordable",
    "Neutral on sustainability",
    "Skeptical of green claims",
    "Prioritizes price over sustainability",
]
TECH_ADOPTION = [
    "Early adopter", "Fast follower", "Mainstream", "Late adopter", "Reluctant"
]
GENDERS = ["Female", "Male", "Non-binary"]


def _income_for(age, education):
    base = 38000
    if "Bachelor" in education:
        base += 22000
    if "Master" in education or "Doctorate" in education or "Professional" in education:
        base += 40000
    if "Trade" in education:
        base += 15000
    base += max(0, (min(int(age), 60) - 25)) * 1200
    base += random.randint(-8000, 12000)
    base = max(24000, base)
    return f"${round(base / 1000) * 1000:,}"


# ===========================================================================
# GROUND TRUTH  (uploaded real survey data used for calibration)
# ===========================================================================
ground_truth = {
    "loaded": False,
    "filename": "",
    "n": 0,
    "rows": [],
    "columns": [],
    "mapping": {},           # role -> column name
    "distributions": {},     # role -> sampleable distribution
    "response_patterns": {}, # survey column -> {value: pct}
}

_COL_HINTS = {
    "age": ["age", "âge", "years old", "yob", "birth"],
    "gender": ["gender", "sex", "genre"],
    "location": ["province", "state", "region", "city", "location", "geo"],
    "income": ["income", "salary", "revenue", "household income"],
    "education": ["education", "degree", "schooling"],
    "occupation": ["occupation", "job", "role", "profession", "employment"],
    "family_status": ["marital", "family", "household", "children", "kids"],
    "sustainability": ["sustain", "eco", "environment", "green"],
    "tech_adoption": ["tech", "innovation", "adopt", "digital"],
    "brand": ["brand", "prefer", "favourite", "favorite"],
}


def _detect_role(col_name):
    low = str(col_name).strip().lower()
    for role, hints in _COL_HINTS.items():
        if any(h in low for h in hints):
            return role
    return None


def _coerce_number(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, AttributeError, TypeError):
        return None


def _parse_ground_truth_bytes(filename, data):
    """Parse an uploaded CSV / JSON / XLSX byte payload into row dicts."""
    name = (filename or "").lower()
    rows = []

    if name.endswith(".json") or data[:1] in (b"[", b"{"):
        payload = json.loads(data.decode("utf-8", "replace"))
        if isinstance(payload, dict):
            for key in ("respondents", "data", "rows", "records"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = [payload]
        rows = [dict(r) for r in payload if isinstance(r, dict)]

    elif name.endswith((".xlsx", ".xlsm")):
        try:
            import openpyxl
        except ImportError as exc:
            raise HTTPException(
                status_code=422,
                detail="XLSX support needs openpyxl (pip install openpyxl). "
                       "You can also upload CSV or JSON.",
            ) from exc
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        it = ws.iter_rows(values_only=True)
        header = [str(h) if h is not None else f"col{i}"
                  for i, h in enumerate(next(it, []))]
        for r in it:
            if r is None:
                continue
            rows.append({header[i]: r[i] for i in range(min(len(header), len(r)))})

    else:  # CSV / TSV
        text = data.decode("utf-8", "replace")
        sample = text[:2000]
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delim)
        rows = [dict(r) for r in reader]

    if not rows:
        raise HTTPException(status_code=422, detail="No rows found in the uploaded file.")
    return rows


def _model_distributions(rows, mapping):
    dists = {}
    for role, col in mapping.items():
        values = [r.get(col) for r in rows if r.get(col) not in (None, "")]
        if not values:
            continue
        if role == "age":
            numbers = [n for n in (_coerce_number(v) for v in values) if n is not None]
            if numbers:
                dists[role] = {
                    "type": "numeric",
                    "min": min(numbers), "max": max(numbers),
                    "mean": round(sum(numbers) / len(numbers), 1),
                    "values": numbers,
                }
                continue
        counts = Counter(str(v).strip() for v in values)
        total = sum(counts.values())
        dists[role] = {
            "type": "categorical",
            "probs": {k: c / total for k, c in counts.most_common()},
        }
    return dists


def _model_response_patterns(rows, mapping):
    demo_cols = set(mapping.values())
    patterns = {}
    if not rows:
        return patterns
    columns = list(rows[0].keys())
    for col in columns:
        if col in demo_cols:
            continue
        values = [str(r.get(col)).strip() for r in rows if r.get(col) not in (None, "")]
        if not values:
            continue
        counts = Counter(values)
        if len(counts) > max(12, len(values) * 0.6):
            continue  # near-unique free text: not aggregatable
        total = sum(counts.values())
        patterns[col] = {k: round(c / total, 3) for k, c in counts.most_common(8)}
    return patterns


def _load_ground_truth(filename, rows):
    columns = list(rows[0].keys())
    mapping = {}
    for col in columns:
        role = _detect_role(col)
        if role and role not in mapping:
            mapping[role] = col
    ground_truth.update({
        "loaded": True,
        "filename": filename,
        "n": len(rows),
        "rows": rows,
        "columns": columns,
        "mapping": mapping,
        "distributions": _model_distributions(rows, mapping),
        "response_patterns": _model_response_patterns(rows, mapping),
    })
    return ground_truth


def _sample_from_dist(role, fallback_choices):
    """Sample a value for `role` from ground truth if available, else fallback."""
    dist = ground_truth["distributions"].get(role) if ground_truth["loaded"] else None
    if not dist:
        return random.choice(fallback_choices)
    if dist["type"] == "numeric":
        vals = dist.get("values") or []
        return random.choice(vals) if vals else random.uniform(dist["min"], dist["max"])
    probs = dist.get("probs", {})
    if not probs:
        return random.choice(fallback_choices)
    r, cum = random.random(), 0.0
    for value, p in probs.items():
        cum += p
        if r <= cum:
            return value
    return next(iter(probs))


# ---------------------------------------------------------------------------
# PERSONA SEEDING (calibration-aware)
# ---------------------------------------------------------------------------
def _seed_persona(index, calibrated=True):
    """Build one bear persona. When ground truth is loaded and `calibrated`,
    demographic attributes are sampled from the real distributions so the
    synthetic panel statistically resembles the uploaded respondents."""
    use_gt = calibrated and ground_truth["loaded"]

    name = random.choice(FIRST_NAMES)

    if use_gt and "age" in ground_truth["distributions"]:
        age = int(round(float(_sample_from_dist("age", [random.randint(18, 75)]))))
        age = max(16, min(95, age))
    else:
        age = random.randint(16, 78)

    if use_gt and "gender" in ground_truth["distributions"]:
        gender = str(_sample_from_dist("gender", GENDERS))
    else:
        gender = random.choice(GENDERS)

    # Location: prefer a real location string if present, else our CA pool.
    if use_gt and "location" in ground_truth["distributions"]:
        loc_val = str(_sample_from_dist("location", ["Toronto, Ontario"]))
        if "," in loc_val:
            city, province = [p.strip() for p in loc_val.split(",", 1)]
        else:
            city, province = loc_val, loc_val
        location = f"{city}, {province}" if city != province else city
    else:
        city, province = random.choice(CANADIAN_LOCATIONS)
        location = f"{city}, {province}"

    education = str(_sample_from_dist("education", EDUCATION_LEVELS)) if use_gt \
        else random.choice(EDUCATION_LEVELS)
    occupation = "Student" if age < 20 else (
        str(_sample_from_dist("occupation", OCCUPATIONS)) if use_gt
        else random.choice(OCCUPATIONS))
    sustainability = str(_sample_from_dist("sustainability", SUSTAINABILITY)) if use_gt \
        else random.choice(SUSTAINABILITY)
    tech = str(_sample_from_dist("tech_adoption", TECH_ADOPTION)) if use_gt \
        else random.choice(TECH_ADOPTION)

    return {
        "bear_name": f"{name} Bear",
        "person_name": name,
        "age": age,
        "gender": gender,
        "location": location,
        "province": province,
        "education": education,
        "occupation": occupation,
        "income": _income_for(age, education),
        "family_status": random.choice(FAMILY_STATUSES),
        "interests": random.sample(INTEREST_POOL, k=random.randint(2, 4)),
        "shopping_frequency": random.choice(SHOPPING_FREQ),
        "preferred_brands": random.sample(BRAND_POOL, k=random.randint(2, 3)),
        "risk_tolerance": random.choice(RISK_TOLERANCE),
        "sustainability": sustainability,
        "tech_adoption": tech,
        "source": "calibrated" if use_gt else "synthetic",
    }


def _persona_from_real_row(row, index):
    """Build a 'historical respondent profile' bear from a real dataset row."""
    m = ground_truth["mapping"]
    def g(role, default):
        col = m.get(role)
        v = row.get(col) if col else None
        return v if v not in (None, "") else default

    raw_age = _coerce_number(g("age", None))
    age = int(raw_age) if raw_age is not None else random.randint(25, 60)
    age = max(16, min(95, age))
    gender = str(g("gender", random.choice(GENDERS)))
    loc = str(g("location", "Toronto, Ontario"))
    if "," in loc:
        city, province = [p.strip() for p in loc.split(",", 1)]
    else:
        city = province = loc
    education = str(g("education", random.choice(EDUCATION_LEVELS)))
    occupation = str(g("occupation", random.choice(OCCUPATIONS)))
    income_val = g("income", None)
    income = f"${int(_coerce_number(income_val)):,}" if _coerce_number(income_val) \
        else _income_for(age, education)
    name = random.choice(FIRST_NAMES)

    # Attach the real answers so this profile can "voice" its actual responses.
    real_answers = {}
    for col in ground_truth["columns"]:
        if col in ground_truth["mapping"].values():
            continue
        val = row.get(col)
        if val not in (None, ""):
            real_answers[col] = str(val)

    return {
        "bear_name": f"{name} Bear",
        "person_name": name,
        "age": age,
        "gender": gender,
        "location": f"{city}, {province}" if city != province else city,
        "province": province,
        "education": education,
        "occupation": occupation,
        "income": income,
        "family_status": str(g("family_status", random.choice(FAMILY_STATUSES))),
        "interests": random.sample(INTEREST_POOL, k=2),
        "shopping_frequency": random.choice(SHOPPING_FREQ),
        "preferred_brands": [str(g("brand", random.choice(BRAND_POOL)))],
        "risk_tolerance": random.choice(RISK_TOLERANCE),
        "sustainability": str(g("sustainability", random.choice(SUSTAINABILITY))),
        "tech_adoption": str(g("tech_adoption", random.choice(TECH_ADOPTION))),
        "source": "historical",
        "real_answers": real_answers,
    }


def _persona_blurb(seed):
    """Tiny LLM enrichment: a 2-sentence personality description with fallback."""
    prompt = (
        "Write a 2-sentence personality description for a survey respondent. "
        "Be specific and human, no lists. Details:\n"
        f"- {seed['age']}yo {seed['gender']}, {seed['occupation']} in {seed['location']}\n"
        f"- {seed['family_status']}; enjoys {', '.join(seed['interests'])}\n"
        f"- Shops {seed['shopping_frequency']}; leans toward {', '.join(seed['preferred_brands'])}\n"
        f"- {seed['sustainability']}; {seed['tech_adoption'].lower()} with tech\n"
        "Respond with ONLY the 2 sentences, third person."
    )
    try:
        text = str(_new_llm(0.9).call(prompt)).strip()
        text = re.sub(r"^[\"'`*#\s]+|[\"'`*\s]+$", "", text)
        if 20 <= len(text) <= 600:
            return text
    except Exception:
        pass
    return (
        f"{seed['person_name']} is a {seed['age']}-year-old {seed['occupation'].lower()} "
        f"in {seed['location']} who is {seed['family_status'].lower()} and enjoys "
        f"{' and '.join(seed['interests'][:2])}. They are {seed['sustainability'].lower()} "
        f"and {seed['tech_adoption'].lower()} with new products."
    )


# ---------------------------------------------------------------------------
# CUSTOM CHARACTERS + PERSONA LIBRARY
# ---------------------------------------------------------------------------
# One in-memory store backs both features. A library entry may be a saved
# persona, a reusable template, and/or "locked" (always joins future panels).
persona_library = {}   # persona_id -> record
library_lock = Lock()

_PERSONA_FIELDS = [
    "person_name", "age", "gender", "location", "occupation", "education",
    "income", "family_status", "interests", "preferred_brands",
    "shopping_frequency", "risk_tolerance", "sustainability", "tech_adoption",
    "description",
]


def _normalize_custom(payload):
    """Turn a Character Studio payload into a full persona seed."""
    name = str(payload.get("person_name") or payload.get("name") or "Custom").strip()
    age = int(_coerce_number(payload.get("age")) or 35)
    province = ""
    location = str(payload.get("location") or "Toronto, Ontario").strip()
    province = location.split(",")[-1].strip() if "," in location else location

    def _listify(v, pool):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [p.strip() for p in re.split(r"[,;]", v) if p.strip()]
        return random.sample(pool, k=2)

    seed = {
        "bear_name": f"{name} Bear",
        "person_name": name,
        "age": max(16, min(95, age)),
        "gender": str(payload.get("gender") or random.choice(GENDERS)),
        "location": location,
        "province": province,
        "education": str(payload.get("education") or random.choice(EDUCATION_LEVELS)),
        "occupation": str(payload.get("occupation") or random.choice(OCCUPATIONS)),
        "income": str(payload.get("income") or _income_for(age, str(payload.get("education") or ""))),
        "family_status": str(payload.get("family_status") or random.choice(FAMILY_STATUSES)),
        "interests": _listify(payload.get("interests"), INTEREST_POOL),
        "shopping_frequency": str(payload.get("shopping_frequency") or random.choice(SHOPPING_FREQ)),
        "preferred_brands": _listify(payload.get("preferred_brands"), BRAND_POOL),
        "risk_tolerance": str(payload.get("risk_tolerance") or random.choice(RISK_TOLERANCE)),
        "sustainability": str(payload.get("sustainability") or random.choice(SUSTAINABILITY)),
        "tech_adoption": str(payload.get("tech_adoption") or random.choice(TECH_ADOPTION)),
        "description": str(payload.get("description") or "").strip(),
        "source": "custom",
    }
    return seed


# ---------------------------------------------------------------------------
# SCREENING
# ---------------------------------------------------------------------------
def _parse_audience(audience):
    text = (audience or "").lower()
    age_min, age_max = 0, 200
    m = re.search(r"(\d{1,3})\s*(?:-|to|–|—)\s*(\d{1,3})", text)
    if m:
        age_min, age_max = int(m.group(1)), int(m.group(2))
    else:
        m2 = re.search(r"(\d{1,3})\s*\+", text)
        if m2:
            age_min = int(m2.group(1))
    country = None
    if "canad" in text:
        country = "canada"
    elif "united states" in text or "u.s." in text or "usa" in text or "american" in text:
        country = "usa"
    return {"age_min": age_min, "age_max": age_max, "country": country, "raw": audience}


def _screen_bear(seed, criteria):
    reasons = []
    age = int(seed["age"])
    if age < criteria["age_min"] or age > criteria["age_max"]:
        return (False,
                f"Age {age} is outside the target range "
                f"{criteria['age_min']}–{criteria['age_max']}.")
    reasons.append(f"age {age} fits the {criteria['age_min']}–{criteria['age_max']} range")

    loc = seed.get("location", "").lower()
    looks_canadian = any(prov.lower() in loc for _, prov in CANADIAN_LOCATIONS) \
        or "canada" in loc or seed.get("source") in ("synthetic", "calibrated", "custom")
    if criteria["country"] == "usa" and looks_canadian and "united states" not in loc \
            and "usa" not in loc:
        return (False, "Located outside the United States for a US-only audience.")
    if criteria["country"] == "canada":
        reasons.append("based in Canada")

    return (True, "Qualified: " + ", ".join(reasons) + ".")


# ---------------------------------------------------------------------------
# RESPONSE CALIBRATION
# ---------------------------------------------------------------------------
def _response_calibration_hint():
    """Compact summary of observed real answer patterns, injected into each
    respondent prompt so synthetic answers drift toward real-world frequencies."""
    if not ground_truth["loaded"] or not ground_truth["response_patterns"]:
        return ""
    lines = []
    for col, dist in list(ground_truth["response_patterns"].items())[:6]:
        top = ", ".join(f"{v} ({int(p*100)}%)" for v, p in list(dist.items())[:4])
        lines.append(f"- On '{col}', real respondents answered: {top}")
    if not lines:
        return ""
    return (
        "\n\nCALIBRATION CONTEXT (real-world response patterns from comparable "
        "respondents — let these gently inform your answers, but stay true to "
        "your own persona):\n" + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# SURVEY INSIGHTS BEAR
# ---------------------------------------------------------------------------
INSIGHTS_BEAR = {
    "id": "insights",
    "name": "Survey Insights Bear",
    "role": "Survey Insights Analyst",
    "goal": "Aggregate synthetic respondent answers into a clear insights report",
    "backstory": (
        "You are the Survey Insights Bear. You never answer surveys yourself and "
        "you never invent respondents. You take the completed answers from the panel "
        "and turn them into distributions, segments, representative quotes, and "
        "recommendations a decision-maker can act on."
    ),
    "active_action": "review",
    "done_action": "present",
}


# ---------------------------------------------------------------------------
# LIVE STATE
# ---------------------------------------------------------------------------
def _activity_for_status(status):
    return {
        "SCREENING": "search", "QUALIFIED": "thumbs", "DISQUALIFIED": "window",
        "ANSWERING": "type", "COMPLETED": "thumbs", "ERROR": "panic",
        "STANDBY": "sleep", "WAITING": "wait", "ACTIVE": "review", "DONE": "present",
    }.get(status, "sleep")


def _empty_metrics():
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "successful_requests": 0, "elapsed_seconds": 0.0}


agent_states = {}
team_interactions = []
state_lock = Lock()
run_active = False
run_started_at = None
run_elapsed_seconds = 0.0
run_meta = {
    "survey": "", "audience": "", "panel_size": 0,
    "qualified": 0, "disqualified": 0, "phase": "idle",
    "source_mix": {}, "calibrated": False,
}
last_validation = {}   # populated after each run


def _bear_state(seed, status, log, **extra):
    persona = {k: seed.get(k) for k in [
        "bear_name", "age", "gender", "location", "province", "education",
        "occupation", "income", "family_status", "interests",
        "shopping_frequency", "preferred_brands", "risk_tolerance",
        "sustainability", "tech_adoption",
    ]}
    persona["description"] = seed.get("description", "")
    persona["source"] = seed.get("source", "synthetic")
    return {
        "id": seed["id"],
        "name": seed["bear_name"],
        "role": f"{seed['age']} · {seed['occupation']} · {seed['location']}",
        "status": status,
        "log": log,
        "source": seed.get("source", "synthetic"),
        "persona": persona,
        "screening_result": extra.get("screening_result"),
        "qualification_reason": extra.get("qualification_reason", ""),
        "result": extra.get("result", ""),
        "metrics": extra.get("metrics", _empty_metrics()),
        "started_at": extra.get("started_at"),
        "activity": extra.get("activity", _activity_for_status(status)),
    }


def _set_bear(seed, status, log, **extra):
    with state_lock:
        agent_states[seed["id"]] = _bear_state(seed, status, log, **extra)


def _insights_state(status, log, **extra):
    return {
        "id": INSIGHTS_BEAR["id"], "name": INSIGHTS_BEAR["name"],
        "role": INSIGHTS_BEAR["role"], "status": status, "log": log,
        "source": "insights", "persona": None, "screening_result": None,
        "qualification_reason": "", "result": extra.get("result", ""),
        "metrics": extra.get("metrics", _empty_metrics()),
        "started_at": extra.get("started_at"),
        "activity": extra.get("activity", _activity_for_status(status)),
        "is_insights": True,
    }


def _set_insights(status, log, **extra):
    with state_lock:
        agent_states[INSIGHTS_BEAR["id"]] = _insights_state(status, log, **extra)


def _metric_value(usage, *names):
    for name in names:
        if isinstance(usage, dict) and usage.get(name) is not None:
            return usage[name]
        value = getattr(usage, name, None)
        if value is not None:
            return value
    return 0


def _extract_metrics(output, crew, elapsed_seconds):
    candidates = [getattr(output, "token_usage", None),
                  getattr(output, "usage_metrics", None),
                  getattr(crew, "usage_metrics", None)]
    usage = next((c for c in candidates if c is not None), {})
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict") and not isinstance(usage, dict):
        usage = usage.dict()
    prompt_tokens = int(_metric_value(usage, "prompt_tokens", "input_tokens") or 0)
    completion_tokens = int(_metric_value(usage, "completion_tokens", "output_tokens") or 0)
    total_tokens = int(_metric_value(usage, "total_tokens") or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "successful_requests": int(_metric_value(usage, "successful_requests") or 0),
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def _set_interactions(items, ttl_seconds=5):
    global team_interactions
    expires_at = time.time() + ttl_seconds
    with state_lock:
        team_interactions = [{
            "source": it["source"], "target": it["target"],
            "kind": it.get("kind", "talk"), "message": it.get("message", "Collaborating"),
            "expires_at": expires_at,
        } for it in items]


# ---------------------------------------------------------------------------
# SURVEY RESPONSE
# ---------------------------------------------------------------------------
def _persona_card(seed):
    return (
        f"Name: {seed['person_name']} ({seed['age']}, {seed['gender']})\n"
        f"Lives in: {seed['location']}\n"
        f"Occupation: {seed['occupation']} | Education: {seed['education']}\n"
        f"Household income: {seed['income']} | Family: {seed['family_status']}\n"
        f"Interests: {', '.join(seed['interests'])}\n"
        f"Shopping: {seed['shopping_frequency']}; prefers {', '.join(seed['preferred_brands'])}\n"
        f"Values: {seed['sustainability']}; {seed['tech_adoption']} tech adopter; "
        f"{seed['risk_tolerance']} risk tolerance\n"
        f"About them: {seed.get('description', '')}"
    )


def _answer_survey(seed, survey, calib_hint=""):
    started = time.perf_counter()
    _set_bear(seed, "ANSWERING", "Answering the survey...",
              screening_result="QUALIFIED",
              qualification_reason=seed.get("qualification_reason", ""),
              started_at=time.time(),
              activity=random.choice(["type", "think", "talk"]))
    crew = None
    try:
        # Historical-profile bears with real answers voice their actual data
        # rather than hallucinating (keeps the panel grounded in real records).
        real = seed.get("real_answers") or {}
        real_block = ""
        if real:
            real_block = (
                "\n\nYou previously gave these real answers; stay consistent with them "
                "and phrase them naturally:\n"
                + "\n".join(f"- {k}: {v}" for k, v in list(real.items())[:12])
            )

        respondent = Agent(
            role="Survey Respondent",
            goal="Answer the survey honestly as this specific person would",
            backstory=("You are a real member of the public taking a survey. You answer "
                       "as yourself, based only on your own life and opinions."),
            llm=_new_llm(0.85), verbose=False,
        )
        task = Task(
            description=(
                "You ARE the following person. Answer the survey below completely in "
                "character.\n\nRULES:\n"
                "- Stay fully in character as this person.\n"
                "- Answer like a normal human being, in plain conversational language.\n"
                "- Never say you are an AI, a model, or a persona.\n"
                "- Never analyze the survey or comment on it as a researcher.\n"
                "- Be consistent with your own demographics, values, and habits.\n"
                "- If a question doesn't apply, say so briefly and honestly.\n\n"
                f"YOUR PERSONA:\n{_persona_card(seed)}"
                f"{real_block}{calib_hint}\n\n"
                f"THE SURVEY:\n{survey}\n\n"
                "Answer each question in order, labelled (Q1, Q2, ...). Keep answers "
                "realistic and personal."
            ),
            expected_output="In-character answers to each survey question.",
            agent=respondent,
        )
        crew = Crew(agents=[respondent], tasks=[task], process=Process.sequential)
        output = crew.kickoff()
        elapsed = time.perf_counter() - started
        metrics = _extract_metrics(output, crew, elapsed)
        result = str(output)
        _set_bear(seed, "COMPLETED", "Survey completed.",
                  screening_result="QUALIFIED",
                  qualification_reason=seed.get("qualification_reason", ""),
                  result=result, metrics=metrics, activity="thumbs")
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        metrics = _extract_metrics(None, crew, elapsed)
        _set_bear(seed, "ERROR", f"Crash: {str(exc)[:150]}",
                  screening_result="QUALIFIED",
                  qualification_reason=seed.get("qualification_reason", ""),
                  metrics=metrics, activity="panic")
        return None


# ---------------------------------------------------------------------------
# INSIGHTS AGGREGATION
# ---------------------------------------------------------------------------
def _composition_summary(seeds):
    if not seeds:
        return "No qualified respondents."
    n = len(seeds)
    ages = [int(s["age"]) for s in seeds]

    def _tally(key):
        counts = Counter(s[key] for s in seeds)
        return ", ".join(f"{k} ({v})" for k, v in counts.most_common(6))

    return (
        f"n = {n} qualified respondents\n"
        f"Age: {min(ages)}–{max(ages)} (avg {round(sum(ages)/n,1)})\n"
        f"Gender: {_tally('gender')}\n"
        f"Province: {_tally('province')}\n"
        f"Source mix: {_tally('source')}\n"
        f"Sustainability: {_tally('sustainability')}\n"
        f"Tech adoption: {_tally('tech_adoption')}"
    )


def _run_insights(survey, audience, qualified_seeds, answers, confidence):
    started = time.perf_counter()
    _set_insights("ACTIVE", "Aggregating responses and writing the report...",
                  started_at=time.time(), activity="review")
    crew = None
    per_budget = max(400, int(24000 / max(1, len(answers))))
    parts = []
    for seed, ans in zip(qualified_seeds, answers):
        if not ans:
            continue
        head = (f"[{seed['bear_name']} — {seed['age']}{seed['gender'][0]}, "
                f"{seed['occupation']}, {seed['location']}, {seed['source']}]")
        parts.append(f"{head}\n{ans[:per_budget]}")
    transcript = "\n\n".join(parts)
    composition = _composition_summary(qualified_seeds)
    conf_line = (f"Overall insight confidence for this study: "
                 f"{confidence['score']}/100 ({confidence['label']}).")

    try:
        analyst = Agent(role=INSIGHTS_BEAR["role"], goal=INSIGHTS_BEAR["goal"],
                        backstory=INSIGHTS_BEAR["backstory"], llm=_new_llm(0.4), verbose=True)
        task = Task(
            description=(
                "Aggregate the synthetic survey panel below into an insights report. "
                "Base every statement ONLY on the responses given. Do not invent "
                "respondents or numbers you cannot support.\n\n"
                f"SURVEY:\n{survey}\n\nTARGET AUDIENCE:\n{audience}\n\n"
                f"SAMPLE COMPOSITION:\n{composition}\n\n{conf_line}\n\n"
                f"RESPONSES:\n{transcript}\n\n"
                "Write the report using EXACTLY these Markdown headings, in order:\n"
                "# Executive Summary\n# Sample Composition\n# Qualified Respondents\n"
                "# Key Findings\n# Consumer Segments\n# Response Distribution\n"
                "# Representative Quotes\n# Consensus Areas\n# Areas of Disagreement\n"
                "# Recommendations\n\n"
                "Under Response Distribution give approximate counts/percentages per "
                "question. Under Representative Quotes, quote respondents by bear name. "
                "Mention the confidence level in the Executive Summary."
            ),
            expected_output="A structured Markdown insights report with the required headings.",
            agent=analyst,
        )
        crew = Crew(agents=[analyst], tasks=[task], process=Process.sequential)
        output = crew.kickoff()
        elapsed = time.perf_counter() - started
        metrics = _extract_metrics(output, crew, elapsed)
        _set_insights("DONE", "Insights report ready.",
                      result=str(output), metrics=metrics, activity="present")
        return str(output)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        metrics = _extract_metrics(None, crew, elapsed)
        _set_insights("ERROR", f"Crash: {str(exc)[:150]}", metrics=metrics, activity="panic")
        return None


# ---------------------------------------------------------------------------
# VALIDATION  (synthetic panel vs ground truth)
# ---------------------------------------------------------------------------
def _hist_from_ages(ages, edges=(0, 25, 35, 45, 55, 65, 200)):
    buckets = [f"{edges[i]}-{edges[i+1]-1 if edges[i+1] < 200 else '+'}"
               for i in range(len(edges) - 1)]
    counts = [0] * (len(edges) - 1)
    for a in ages:
        for i in range(len(edges) - 1):
            if edges[i] <= a < edges[i + 1]:
                counts[i] += 1
                break
    total = sum(counts) or 1
    return {buckets[i]: round(counts[i] / total, 3) for i in range(len(buckets))}


def _categorical_share(values):
    counts = Counter(str(v).strip() for v in values if str(v).strip())
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in counts.most_common(8)}


def _overlap_similarity(a, b):
    """Distribution overlap (1 - total variation distance) over shared keys.
    1.0 = identical shares, 0.0 = disjoint. Robust and easy to read."""
    keys = set(a) | set(b)
    if not keys:
        return None
    tvd = 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
    return round(max(0.0, 1.0 - tvd), 3)


def _real_ages():
    dist = ground_truth["distributions"].get("age")
    if dist and dist["type"] == "numeric":
        return [int(v) for v in dist.get("values", [])]
    return []


def _real_categorical(role):
    dist = ground_truth["distributions"].get(role)
    if dist and dist["type"] == "categorical":
        return dist["probs"]
    return {}


def _build_validation(qualified_seeds):
    """Compare the qualified synthetic panel to the uploaded ground truth."""
    if not ground_truth["loaded"] or not qualified_seeds:
        return {"available": False,
                "reason": "Upload ground-truth data to enable validation."}

    syn_ages = [int(s["age"]) for s in qualified_seeds]
    comparisons = []

    # Age
    real_ages = _real_ages()
    if real_ages:
        syn_h, real_h = _hist_from_ages(syn_ages), _hist_from_ages(real_ages)
        comparisons.append({
            "dimension": "Age", "synthetic": syn_h, "real": real_h,
            "similarity": _overlap_similarity(syn_h, real_h),
        })

    # Gender / location (province) / sustainability / tech — any that exist
    for role, key, label in [
        ("gender", "gender", "Gender"),
        ("location", "province", "Geography"),
        ("sustainability", "sustainability", "Sustainability"),
        ("tech_adoption", "tech_adoption", "Tech adoption"),
    ]:
        real = _real_categorical(role)
        if not real:
            continue
        syn = _categorical_share([s.get(key) for s in qualified_seeds])
        comparisons.append({
            "dimension": label, "synthetic": syn, "real": real,
            "similarity": _overlap_similarity(syn, real),
        })

    sims = [c["similarity"] for c in comparisons if c["similarity"] is not None]
    overall = round(sum(sims) / len(sims), 3) if sims else None
    return {
        "available": True,
        "ground_truth_n": ground_truth["n"],
        "filename": ground_truth["filename"],
        "comparisons": comparisons,
        "overall_similarity": overall,
    }


def _coverage_score(qualified_seeds):
    """Fraction of ground-truth demographic groups represented in the panel."""
    if not ground_truth["loaded"]:
        return None
    covered, total = 0, 0
    for role, key in [("gender", "gender"), ("location", "province")]:
        real = _real_categorical(role)
        if not real:
            continue
        syn_vals = {str(s.get(key)).strip() for s in qualified_seeds}
        for value in real:
            total += 1
            if value in syn_vals:
                covered += 1
    return round(covered / total, 3) if total else None


def _confidence_score(qualified_seeds, validation):
    """Insight confidence from sample size, audience match, response similarity,
    and demographic coverage. Returns a 0-100 score with a breakdown."""
    # 1. Ground-truth sample size (log-scaled; 0 if no GT).
    gt_n = ground_truth["n"] if ground_truth["loaded"] else 0
    size_component = 0.0 if gt_n == 0 else min(1.0, math.log10(gt_n + 1) / 3.0)  # 1000 -> ~1.0

    # 2. Audience/panel qualified size (do we have enough synthetic voices?).
    q = len(qualified_seeds)
    panel_component = min(1.0, math.log10(q + 1) / 2.3)  # ~200 -> ~1.0

    # 3. Response similarity to real distributions.
    sim_component = 0.0
    if validation.get("available") and validation.get("overall_similarity") is not None:
        sim_component = validation["overall_similarity"]

    # 4. Demographic coverage.
    cov = _coverage_score(qualified_seeds)
    cov_component = cov if cov is not None else 0.0

    if ground_truth["loaded"]:
        weights = {"size": 0.25, "panel": 0.15, "similarity": 0.40, "coverage": 0.20}
        raw = (weights["size"] * size_component +
               weights["panel"] * panel_component +
               weights["similarity"] * sim_component +
               weights["coverage"] * cov_component)
    else:
        # No ground truth: confidence is inherently limited and rests on panel size.
        raw = 0.35 * panel_component + 0.10

    score = int(round(raw * 100))
    label = ("High" if score >= 70 else "Moderate" if score >= 45 else "Low")
    return {
        "score": score, "label": label,
        "components": {
            "ground_truth_sample_size": round(size_component, 3),
            "panel_size": round(panel_component, 3),
            "response_similarity": round(sim_component, 3),
            "demographic_coverage": round(cov_component, 3),
        },
        "ground_truth_n": gt_n,
        "qualified_n": q,
    }


def _cost_estimate(perf):
    p = perf.get("prompt_tokens", 0) / 1_000_000 * COST_PER_1M_PROMPT
    c = perf.get("completion_tokens", 0) / 1_000_000 * COST_PER_1M_COMPLETION
    return round(p + c, 4)


def _build_dashboard():
    """Researcher dashboard: totals, composition, tokens, cost, runtime, quality."""
    bears = [s for sid, s in agent_states.items()
             if not sid.startswith("_") and not s.get("is_insights")]
    total = len(bears)
    qualified = [b for b in bears if b["status"] in ("QUALIFIED", "ANSWERING", "COMPLETED")]
    disqualified = [b for b in bears if b["status"] == "DISQUALIFIED"]
    ages = [int(b["persona"]["age"]) for b in bears if b.get("persona")]
    avg_age = round(sum(ages) / len(ages), 1) if ages else 0

    def _compose(field):
        return dict(Counter(
            b["persona"].get(field) for b in bears if b.get("persona")
        ).most_common(8))

    metrics = [b.get("metrics", {}) for b in agent_states.values()]
    perf = {
        "prompt_tokens": sum(int(m.get("prompt_tokens", 0)) for m in metrics),
        "completion_tokens": sum(int(m.get("completion_tokens", 0)) for m in metrics),
        "total_tokens": sum(int(m.get("total_tokens", 0)) for m in metrics),
    }
    elapsed = run_elapsed_seconds
    if run_active and run_started_at is not None:
        elapsed = max(0.0, time.time() - run_started_at)

    return {
        "total_bears": total,
        "qualified_bears": len(qualified),
        "disqualified_bears": len(disqualified),
        "average_age": avg_age,
        "gender_composition": _compose("gender"),
        "province_composition": _compose("province"),
        "source_composition": dict(Counter(b.get("source", "synthetic") for b in bears)),
        "token_usage": perf,
        "estimated_cost_usd": _cost_estimate(perf),
        "runtime_seconds": round(elapsed, 2),
        "ground_truth_similarity": last_validation.get("overall_similarity"),
        "ground_truth_loaded": ground_truth["loaded"],
    }


# ---------------------------------------------------------------------------
# HYBRID ROSTER BUILDING
# ---------------------------------------------------------------------------
def _build_roster(panel_size, source_mix, custom_characters, calibrated):
    """Assemble the panel from three sources by proportion, always including any
    locked library personas. Returns a list of persona seeds with ids."""
    seeds = []

    # 1. Locked personas always participate.
    with library_lock:
        locked = [copy.deepcopy(rec["persona"]) for rec in persona_library.values()
                  if rec.get("locked")]
    for seed in locked:
        seed.setdefault("source", "custom")
    seeds.extend(locked)

    # 2. Explicit custom characters passed in for this run.
    for payload in (custom_characters or []):
        seeds.append(_normalize_custom(payload))

    remaining = max(0, panel_size - len(seeds))

    # 3. Fill the rest by the requested source mix.
    mix = source_mix or {}
    p_syn = float(mix.get("synthetic", 1.0))
    p_cust = float(mix.get("custom", 0.0))
    p_hist = float(mix.get("historical", 0.0))
    total_w = p_syn + p_cust + p_hist
    if total_w <= 0:
        p_syn, total_w = 1.0, 1.0

    n_hist = int(round(remaining * (p_hist / total_w))) if ground_truth["loaded"] else 0
    n_cust_extra = int(round(remaining * (p_cust / total_w)))
    n_syn = remaining - n_hist - n_cust_extra
    if n_syn < 0:
        n_syn = 0

    # Historical: sample real rows into profile bears.
    if n_hist and ground_truth["rows"]:
        sampled = random.sample(ground_truth["rows"],
                                k=min(n_hist, len(ground_truth["rows"])))
        # if we need more than we have, allow resampling
        while len(sampled) < n_hist:
            sampled.append(random.choice(ground_truth["rows"]))
        for i, row in enumerate(sampled):
            seeds.append(_persona_from_real_row(row, i))

    # Extra custom-style bears (varied clones seeded from any custom templates,
    # else just synthetic — we don't fabricate fake "custom" identities).
    for _ in range(n_cust_extra):
        seeds.append(_seed_persona(0, calibrated=calibrated))

    # Synthetic / calibrated fill.
    for _ in range(n_syn):
        seeds.append(_seed_persona(0, calibrated=calibrated))

    # Trim/pad to exactly panel_size (locked personas may push us over).
    seeds = seeds[:max(panel_size, len(locked) + len(custom_characters or []))]

    for i, seed in enumerate(seeds):
        seed["id"] = f"bear_{i:03d}"
    return seeds


# ---------------------------------------------------------------------------
# THE FULL RUN
# ---------------------------------------------------------------------------
def run_panel(survey, audience, panel_size, source_mix, custom_characters, calibrated):
    global agent_states, run_active, run_started_at, run_elapsed_seconds
    global run_meta, last_validation

    panel_size = max(1, min(MAX_PANEL_SIZE, int(panel_size)))
    criteria = _parse_audience(audience)

    with state_lock:
        agent_states = {}
        run_started_at = time.time()
        run_elapsed_seconds = 0.0
        last_validation = {}
        run_meta = {
            "survey": survey, "audience": audience, "panel_size": panel_size,
            "qualified": 0, "disqualified": 0, "phase": "generating",
            "source_mix": source_mix or {}, "calibrated": bool(calibrated and ground_truth["loaded"]),
        }

    try:
        # ---- Phase 1: GENERATE -------------------------------------------
        seeds = _build_roster(panel_size, source_mix, custom_characters, calibrated)
        for i, seed in enumerate(seeds):
            _set_bear(seed, "SCREENING", "Entering the Screening Room...", activity="search")

        with state_lock:
            run_meta["phase"] = "screening"
            run_meta["panel_size"] = len(seeds)

        # Enrich personas lacking a description (skip ones that already have one).
        def _enrich(seed):
            if not seed.get("description"):
                seed["description"] = _persona_blurb(seed)
            return seed
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as ex:
            list(ex.map(_enrich, seeds))

        # ---- Phase 2: SCREENING ------------------------------------------
        qualified_seeds = []
        for seed in seeds:
            qualified, reason = _screen_bear(seed, criteria)
            seed["qualification_reason"] = reason
            if qualified:
                qualified_seeds.append(seed)
                _set_bear(seed, "QUALIFIED", "Qualified for the survey.",
                          screening_result="QUALIFIED", qualification_reason=reason,
                          activity="thumbs")
            else:
                _set_bear(seed, "DISQUALIFIED", "Screened out.",
                          screening_result="DISQUALIFIED", qualification_reason=reason,
                          activity="window")

        with state_lock:
            run_meta["qualified"] = len(qualified_seeds)
            run_meta["disqualified"] = len(seeds) - len(qualified_seeds)
            run_meta["phase"] = "surveying"

        _set_insights("WAITING", "Waiting for the panel to finish answering...",
                      activity="wait")

        # Build validation + confidence now (persona-level; doesn't need answers).
        validation = _build_validation(qualified_seeds)
        confidence = _confidence_score(qualified_seeds, validation)
        with state_lock:
            last_validation = validation

        if not qualified_seeds:
            _set_insights("ERROR",
                          "No bears qualified for this audience. Try widening the audience.",
                          activity="panic")
            with state_lock:
                run_meta["phase"] = "complete"
            return

        # ---- Phase 3: SURVEY (parallel) ----------------------------------
        sample_sources = [s["id"] for s in qualified_seeds[:6]]
        _set_interactions([{"source": sid, "target": INSIGHTS_BEAR["id"],
                            "kind": "handoff", "message": "Heading to the Survey Hall"}
                           for sid in sample_sources])
        calib_hint = _response_calibration_hint()

        answers = [None] * len(qualified_seeds)
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as ex:
            futures = {ex.submit(_answer_survey, seed, survey, calib_hint): idx
                       for idx, seed in enumerate(qualified_seeds)}
            for fut in futures:
                answers[futures[fut]] = fut.result()

        # ---- Phase 4: INSIGHTS -------------------------------------------
        with state_lock:
            run_meta["phase"] = "aggregating"
        _set_interactions([{"source": s["id"], "target": INSIGHTS_BEAR["id"],
                            "kind": "review", "message": "Submitting answers"}
                           for s in qualified_seeds[:6]])
        _run_insights(survey, audience, qualified_seeds, answers, confidence)

        with state_lock:
            run_meta["phase"] = "complete"
            run_meta["confidence"] = confidence
        _set_interactions([{"source": INSIGHTS_BEAR["id"],
                            "target": qualified_seeds[0]["id"],
                            "kind": "share", "message": "Insights ready"}],
                          ttl_seconds=12)

    except Exception as e:
        import traceback
        print("=" * 60); print("PANEL RUN CRASHED:"); traceback.print_exc(); print("=" * 60)
        _set_interactions([], ttl_seconds=0)
        with state_lock:
            run_meta["phase"] = "error"
            for state in agent_states.values():
                if state["status"] in ("SCREENING", "ANSWERING", "ACTIVE", "WAITING"):
                    state["status"] = "ERROR"
                    state["log"] = f"Crash: {str(e)[:150]}"
                    state["activity"] = "panic"
    finally:
        with state_lock:
            if run_started_at is not None:
                run_elapsed_seconds = max(0.0, time.time() - run_started_at)
            run_active = False


# ===========================================================================
# API
# ===========================================================================
class SurveyRequest(BaseModel):
    survey: str
    panel_size: int = 50
    audience: str = "Canadian grocery shoppers aged 25-65"
    # Hybrid panel: proportions need not sum to 1; they're normalized.
    source_mix: Optional[Dict[str, float]] = None
    custom_characters: Optional[List[Dict[str, Any]]] = None
    calibrated: bool = True


class PersonaRecord(BaseModel):
    persona: Dict[str, Any]
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    is_template: bool = False
    locked: bool = False


@app.post("/assign")
def assign_task(payload: SurveyRequest, background_tasks: BackgroundTasks):
    global run_active, run_started_at, run_elapsed_seconds
    survey = (payload.survey or "").strip()
    if not survey:
        raise HTTPException(status_code=422, detail="Survey cannot be empty.")
    panel_size = int(payload.panel_size or 0)
    if panel_size < 1:
        raise HTTPException(status_code=422, detail="panel_size must be at least 1.")
    if panel_size > MAX_PANEL_SIZE:
        raise HTTPException(status_code=422,
                            detail=f"panel_size is capped at {MAX_PANEL_SIZE} for local Ollama.")
    with state_lock:
        if run_active:
            raise HTTPException(status_code=409, detail="A panel run is already active.")
        run_active = True
        run_started_at = time.time()
        run_elapsed_seconds = 0.0

    background_tasks.add_task(
        run_panel, survey, payload.audience or "", panel_size,
        payload.source_mix, payload.custom_characters, payload.calibrated,
    )
    return {"message": "Synthetic panel started.",
            "panel_size": min(MAX_PANEL_SIZE, panel_size),
            "parallel_agents": MAX_PARALLEL_AGENTS,
            "calibrated": bool(payload.calibrated and ground_truth["loaded"])}


@app.get("/status")
def get_status():
    with state_lock:
        snapshot = copy.deepcopy(agent_states)
        now = time.time()
        for state in snapshot.values():
            if state.get("status") in ("ANSWERING", "ACTIVE", "SCREENING") \
                    and state.get("started_at") is not None:
                state["metrics"]["elapsed_seconds"] = round(max(0.0, now - state["started_at"]), 2)
        metrics = [s.get("metrics", {}) for s in agent_states.values()]
        elapsed = run_elapsed_seconds
        if run_active and run_started_at is not None:
            elapsed = max(0.0, time.time() - run_started_at)
        snapshot["_interactions"] = [it for it in team_interactions if it["expires_at"] > time.time()]
        perf = {
            "prompt_tokens": sum(int(m.get("prompt_tokens", 0)) for m in metrics),
            "completion_tokens": sum(int(m.get("completion_tokens", 0)) for m in metrics),
            "total_tokens": sum(int(m.get("total_tokens", 0)) for m in metrics),
            "successful_requests": sum(int(m.get("successful_requests", 0)) for m in metrics),
            "elapsed_seconds": round(elapsed, 2),
        }
        snapshot["_performance"] = perf
        snapshot["_run"] = {"active": run_active,
                            "max_parallel_agents": MAX_PARALLEL_AGENTS, **run_meta}
        snapshot["_validation"] = copy.deepcopy(last_validation)
        snapshot["_dashboard"] = _build_dashboard()
        snapshot["_ground_truth"] = {
            "loaded": ground_truth["loaded"], "filename": ground_truth["filename"],
            "n": ground_truth["n"], "mapping": ground_truth["mapping"],
            "columns": ground_truth["columns"],
        }
        return snapshot


@app.get("/roster")
def get_roster():
    with state_lock:
        bears = [{"id": s["id"], "name": s["name"], "role": s["role"],
                  "status": s["status"], "source": s.get("source", "synthetic")}
                 for sid, s in agent_states.items()
                 if not sid.startswith("_") and not s.get("is_insights")]
        bears.append({"id": INSIGHTS_BEAR["id"], "name": INSIGHTS_BEAR["name"],
                      "role": INSIGHTS_BEAR["role"],
                      "status": agent_states.get(INSIGHTS_BEAR["id"], {}).get("status", "STANDBY"),
                      "source": "insights"})
        return bears


# ---- Ground truth ---------------------------------------------------------
@app.post("/ground_truth")
async def upload_ground_truth(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    rows = _parse_ground_truth_bytes(file.filename, data)
    gt = _load_ground_truth(file.filename, rows)
    return {
        "message": "Ground truth loaded.",
        "filename": gt["filename"], "n": gt["n"],
        "columns": gt["columns"], "detected_mapping": gt["mapping"],
        "response_columns": list(gt["response_patterns"].keys()),
    }


@app.delete("/ground_truth")
def clear_ground_truth():
    ground_truth.update({"loaded": False, "filename": "", "n": 0, "rows": [],
                         "columns": [], "mapping": {}, "distributions": {},
                         "response_patterns": {}})
    return {"message": "Ground truth cleared."}


@app.get("/ground_truth")
def ground_truth_info():
    return {
        "loaded": ground_truth["loaded"], "filename": ground_truth["filename"],
        "n": ground_truth["n"], "columns": ground_truth["columns"],
        "mapping": ground_truth["mapping"],
        "distributions": {
            role: (d if d["type"] == "categorical"
                   else {"type": "numeric", "min": d["min"], "max": d["max"], "mean": d["mean"]})
            for role, d in ground_truth["distributions"].items()
        },
        "response_patterns": ground_truth["response_patterns"],
    }


# ---- Validation / dashboard (also embedded in /status) --------------------
@app.get("/validation")
def get_validation():
    with state_lock:
        return copy.deepcopy(last_validation) or {"available": False,
            "reason": "Run a panel with ground truth loaded to see validation."}


@app.get("/dashboard")
def get_dashboard():
    with state_lock:
        return _build_dashboard()


# ---- Persona Library + Character Studio ------------------------------------
@app.get("/personas")
def list_personas(q: Optional[str] = None, tag: Optional[str] = None):
    with library_lock:
        out = []
        for pid, rec in persona_library.items():
            if tag and tag not in (rec.get("tags") or []):
                continue
            if q:
                hay = json.dumps(rec).lower()
                if q.lower() not in hay:
                    continue
            out.append({"id": pid, **rec})
        return out


@app.post("/personas")
def save_persona(rec: PersonaRecord):
    seed = _normalize_custom(rec.persona)
    pid = uuid.uuid4().hex[:12]
    record = {
        "id": pid, "name": rec.name or seed["person_name"],
        "tags": rec.tags or [], "is_template": bool(rec.is_template),
        "locked": bool(rec.locked), "persona": seed,
    }
    with library_lock:
        persona_library[pid] = record
    return {"message": "Persona saved.", "id": pid, **record}


@app.put("/personas/{pid}")
def update_persona(pid: str, rec: PersonaRecord):
    with library_lock:
        if pid not in persona_library:
            raise HTTPException(status_code=404, detail="Persona not found.")
        seed = _normalize_custom(rec.persona)
        persona_library[pid] = {
            "id": pid, "name": rec.name or seed["person_name"],
            "tags": rec.tags or persona_library[pid].get("tags", []),
            "is_template": bool(rec.is_template), "locked": bool(rec.locked),
            "persona": seed,
        }
        return {"message": "Persona updated.", "id": pid, **persona_library[pid]}


@app.post("/personas/{pid}/lock")
def toggle_lock(pid: str):
    with library_lock:
        if pid not in persona_library:
            raise HTTPException(status_code=404, detail="Persona not found.")
        persona_library[pid]["locked"] = not persona_library[pid].get("locked")
        return {"id": pid, "locked": persona_library[pid]["locked"]}


@app.delete("/personas/{pid}")
def delete_persona(pid: str):
    with library_lock:
        if pid not in persona_library:
            raise HTTPException(status_code=404, detail="Persona not found.")
        del persona_library[pid]
        return {"message": "Persona deleted.", "id": pid}


@app.post("/preview_persona")
def preview_persona(rec: PersonaRecord):
    """Normalize a Character Studio draft into a full persona (no save)."""
    return _normalize_custom(rec.persona)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Bear Synthetic Consumer Research Platform",
        "max_panel_size": MAX_PANEL_SIZE,
        "max_parallel_agents": MAX_PARALLEL_AGENTS,
        "ground_truth_loaded": ground_truth["loaded"],
        "endpoints": {
            "assign": "POST /assign", "status": "GET /status", "roster": "GET /roster",
            "ground_truth": "POST/GET/DELETE /ground_truth",
            "validation": "GET /validation", "dashboard": "GET /dashboard",
            "personas": "GET/POST /personas, PUT/DELETE /personas/{id}, POST /personas/{id}/lock",
            "docs": "http://127.0.0.1:8000/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
