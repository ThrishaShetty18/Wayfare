"""
Shared model-loading and inference logic for the Wayfare recommendation API.

Scoring uses two signals, but they no longer play the same role:

  1. CONTENT  — a transparent, rule-based overlap score between what the
                traveller picked in the survey and each destination's own
                labelled attributes (its typical budget tier, top interest,
                trip style, season, climate, companions, accommodation).
                This is what's now shown to the user as "match %". It is
                literally (# of the traveller's applicable answers that this
                destination's own profile agrees with) / (# applicable) —
                so a destination that matches every dimension the traveller
                answered shows a genuine 100%, and the "why matched" list on
                the destination page is the exact arithmetic behind that
                number, not a black-box guess.
  2. NEURAL   — the trained MLP over [survey features] + [learned destination
                embedding], exactly as in ml/train.py. It no longer waters
                down the displayed percentage. Its only job now is breaking
                ties in ranking order between destinations that land on the
                same content-match percentage, using the co-occurrence
                patterns it learned from the real swipe data.

Why the split: blending the two into one displayed number meant even a
destination that matched 100% of what you picked could never show higher
than the high-80s/low-90s, because the neural term (a sigmoid, rarely near 1)
dragged the average down — the score looked "imprecise" for exactly the
inputs where it should have been most confident. Separating "what you'll see"
(content, exact and explainable) from "how ties are broken" (neural,
statistical) fixes that without throwing away the neural network — it's
still doing real work, just not in the number people read as a percentage.

Destinations with very few recorded swipes (see MIN_SWIPES below) never got
enough training signal for their embedding row to mean anything — e.g.
"Baalbek, Lebanon" had a single swipe and its embedding was close to random,
yet it was still scoring near the top for almost everyone. Two fixes for
that: (a) destinations below MIN_SWIPES are dropped from the candidate pool
entirely (their descriptive fields are blank anyway, so there is nothing
reliable to show on a destination page either), and (b) for the destinations
that remain, the neural tie-breaker is shrunk toward neutral in proportion to
how little training signal it had (POP_FULL_CONF), so a barely-trained
embedding can't swing ordering either.
"""
import csv
import json
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

# ---------------------------------------------------------------------------
# Popularity / confidence handling (fixes the "untrained embedding ranks
# #1 for everyone" bug)
# ---------------------------------------------------------------------------
MIN_SWIPES = 10          # destinations below this are excluded as candidates
POP_FULL_CONF = 50       # swipe count at which we trust the neural score 100%

# ---------------------------------------------------------------------------
# Displayed match % = CONTENT score alone (exact, explainable, can hit 100%).
# Ranking order = content score first, neural score only as a tie-breaker
# among destinations that land on the same (rounded) content percentage —
# TIE_BREAK_WEIGHT keeps its influence on the *order* small enough that it
# can never reorder two destinations whose content match differs by more
# than a couple of points.
# ---------------------------------------------------------------------------
TIE_BREAK_WEIGHT = 0.05

# Rough day-of-travel spend per person, in USD, by declared budget style —
# these are the BASELINE rates for a "typical mid-cost" destination (roughly
# the cost level of somewhere like Portugal or Croatia). Every destination
# then scales this baseline by its own COUNTRY_COST_INDEX multiplier below,
# so a shoestring week in Switzerland and a shoestring week in Vietnam no
# longer come out to the same number — that was the bug: estimate_budget()
# used to ignore which destination was picked entirely.
BUDGET_DAILY_USD = {
    "Shoestring / hostel budget": {"stay": 18, "food": 12, "activities": 8, "local_transport": 6},
    "Mid-range, value-conscious": {"stay": 45, "food": 28, "activities": 20, "local_transport": 12},
    "Comfortable, willing to splurge sometimes": {"stay": 90, "food": 50, "activities": 40, "local_transport": 20},
    "Luxury, money is not the constraint": {"stay": 220, "food": 100, "activities": 90, "local_transport": 40},
}

# Approximate relative cost-of-travel index by country, 1.0 = the baseline
# the BUDGET_DAILY_USD table above already represents (a mid-cost European/
# Latin American destination). This is a general, widely-recognized ballpark
# ranking (the kind you'd find comparing Numbeo/Big Mac Index-style cost
# tiers) — not live pricing data, and it deliberately is not part of the
# neural network's input; it only rescales the destination-page budget
# estimate so the number is destination-aware instead of identical for
# every place. Countries not listed fall back to DEFAULT_COST_MULTIPLIER.
DEFAULT_COST_MULTIPLIER = 1.0
COUNTRY_COST_INDEX = {
    # Very high cost (Western Europe premium, Gulf luxury, wealthy small states)
    "Switzerland": 2.3, "Norway": 2.1, "Iceland": 2.1, "Monaco": 2.6,
    "Liechtenstein": 2.2, "Luxembourg": 1.9, "Singapore": 1.8, "Denmark": 1.8,
    "Sweden": 1.5, "Qatar": 1.7, "United Arab Emirates": 1.6, "Bahamas": 1.9,
    "Bahrain": 1.4, "Kuwait": 1.4, "San Marino": 1.5, "Ireland": 1.6,
    "Australia": 1.6, "New Zealand": 1.5, "Japan": 1.4, "Israel": 1.5,
    "Finland": 1.5, "Netherlands": 1.6, "United Kingdom": 1.6, "United States": 1.7,
    "Canada": 1.4, "France": 1.5, "Germany": 1.4, "Belgium": 1.4, "Austria": 1.4,
    "South Korea": 1.3, "Maldives": 2.4, "Seychelles": 2.0, "Greenland (Denmark)": 2.0,
    # High-moderate (Southern/Central Europe, small islands, richer Asia/ME)
    "Italy": 1.3, "Spain": 1.1, "Portugal": 1.0, "Malta": 1.2, "Cyprus": 1.2,
    "Greece": 1.1, "Mauritius": 1.3, "Saudi Arabia": 1.1, "Oman": 1.0,
    "Taiwan": 0.9, "China": 0.9, "Brunei": 1.1, "Palau": 1.4, "Nauru": 1.5,
    "Marshall Islands": 1.4, "Micronesia": 1.3, "Fiji": 1.0, "Trinidad and Tobago": 1.0,
    "Barbados": 1.3, "St. Kitts and Nevis": 1.2, "Antigua and Barbuda": 1.3,
    "Andorra": 1.3, "Chile": 0.9,
    # Moderate (baseline-ish — matches the table above)
    "Croatia": 0.9, "Slovenia": 1.0, "Czech Republic": 0.85, "Poland": 0.75,
    "Estonia": 0.85, "Latvia": 0.8, "Lithuania": 0.8, "Hungary": 0.75,
    "Slovakia": 0.8, "Costa Rica": 0.9, "Panama": 0.85, "Uruguay": 1.0,
    "Argentina": 0.7, "Brazil": 0.75, "Mexico": 0.7, "South Africa": 0.65,
    "Botswana": 0.8, "Namibia": 0.75, "Jordan": 0.85, "Turkey": 0.6,
    "Jamaica": 0.9, "Dominican Republic": 0.85, "St. Lucia": 1.1,
    "St. Vincent and the Grenadines": 1.0, "Grenada": 1.0, "Suriname": 0.8,
    "Guyana": 0.75, "Belize": 0.9, "Cuba": 0.7, "Samoa": 0.9, "Vanuatu": 1.0,
    "Papua New Guinea": 0.9, "Solomon Islands": 0.9, "Cape Verde": 0.8,
    # Lower-moderate (Eastern Europe, Caucasus, much of Latin America)
    "Romania": 0.65, "Bulgaria": 0.6, "Serbia": 0.6, "Montenegro": 0.75,
    "Bosnia and Herzegovina": 0.6, "Albania": 0.55, "Moldova": 0.55,
    "Belarus": 0.55, "Russia": 0.6, "Ukraine": 0.5, "Georgia": 0.5,
    "Armenia": 0.5, "Azerbaijan": 0.6, "Kazakhstan": 0.55, "Peru": 0.6,
    "Ecuador": 0.6, "Colombia": 0.55, "Bolivia": 0.45, "Paraguay": 0.55,
    "Guatemala": 0.55, "Honduras": 0.5, "El Salvador": 0.55, "Nicaragua": 0.5,
    "Dominica": 0.9, "Palestine": 0.5,
    # Low cost (South/Southeast Asia, much of Africa, Central Asia)
    "India": 0.35, "Nepal": 0.3, "Sri Lanka": 0.4, "Vietnam": 0.4, "Laos": 0.4,
    "Myanmar": 0.35, "Indonesia": 0.4, "Philippines": 0.45, "Malaysia": 0.55,
    "Thailand": 0.5, "Egypt": 0.45, "Morocco": 0.5, "Tunisia": 0.45,
    "Algeria": 0.45, "Kenya": 0.5, "Tanzania": 0.5, "Uganda": 0.45,
    "Zambia": 0.5, "Zimbabwe": 0.5, "Ethiopia": 0.35, "Madagascar": 0.4,
    "Mozambique": 0.5, "Senegal": 0.5, "Sao Tome and Principe": 0.6,
    "Kyrgyzstan": 0.4, "Tajikistan": 0.4, "Uzbekistan": 0.4,
}
TRIP_LENGTH_DAYS = {
    "A long weekend (1–3 days)": 3,
    "About a week (4–7 days)": 6,
    "Two to three weeks": 15,
    "A month or more": 30,
}
COMPANION_COUNT = {
    "Solo": 1,
    "Partner": 2,
    "Friends or family group": 4,
}

FORM_CARD = {
    'form_a': 4, 'form_b': 4, 'form_c': 4,
    'form_f': 8, 'form_g': 8,
    'form_h': 3, 'form_i': 3, 'form_j': 3, 'form_k': 3,
}
FEAT_DIM = sum(FORM_CARD.values())  # 40 — unchanged, matches the trained weights

# 'region' is a candidate FILTER (India / abroad / both), not part of the
# neural network's input vector, so it's deliberately excluded from FORM_CARD.
QUESTIONS = [
    {"key": "region", "title": "Where do you want to explore?", "multi": False,
     "options": ["Within India", "Abroad — anywhere international", "Show me both"]},
    {"key": "form_a", "title": "How often do you travel?", "multi": False,
     "options": ["Rarely (once a year or less)", "A few times a year",
                 "Every couple of months", "Constantly — I live out of a suitcase"]},
    {"key": "form_b", "title": "How long is your typical trip?", "multi": False,
     "options": ["A long weekend (1–3 days)", "About a week (4–7 days)",
                 "Two to three weeks", "A month or more"]},
    {"key": "form_c", "title": "What's your travel budget style?", "multi": False,
     "options": ["Shoestring / hostel budget", "Mid-range, value-conscious",
                 "Comfortable, willing to splurge sometimes", "Luxury, money is not the constraint"]},
    {"key": "form_f", "title": "What draws you to a destination?", "multi": True,
     "options": ["Beaches & coastline", "Mountains & nature", "History & culture",
                 "Nightlife & entertainment", "Food & culinary scene", "Adventure sports",
                 "Relaxation & wellness", "Shopping & city life"]},
    {"key": "form_g", "title": "What kind of trip are you in the mood for?", "multi": True,
     "options": ["City break", "Beach holiday", "Road trip", "Backpacking",
                 "Luxury resort stay", "Cultural immersion", "Wildlife & nature", "Festivals & events"]},
    {"key": "form_h", "title": "Who do you usually travel with?", "multi": False,
     "options": ["Solo", "Partner", "Friends or family group"]},
    {"key": "form_i", "title": "Which season do you prefer to travel in?", "multi": False,
     "options": ["Warm season", "Shoulder season", "Cold season / winter travel"]},
    {"key": "form_j", "title": "What accommodation do you prefer?", "multi": False,
     "options": ["Hostel / budget stay", "Standard hotel", "Resort or high-end stay"]},
    {"key": "form_k", "title": "What climate do you prefer?", "multi": False,
     "options": ["Warm & tropical", "Temperate", "Cool / cold"]},
]

REGION_BY_INDEX = {0: "india", 1: "abroad", 2: "both"}

# ---------------------------------------------------------------------------
# Survey-option -> dataset-label maps, used for the content match score.
# A `None` means that survey option has no counterpart in the labelled
# dataset (it simply can't contribute a hit for that dimension).
# ---------------------------------------------------------------------------
BUDGET_LABEL = {
    "Shoestring / hostel budget": "Shoestring/budget",
    "Mid-range, value-conscious": "Mid-range",
    "Comfortable, willing to splurge sometimes": "Comfortable/splurge",
    "Luxury, money is not the constraint": "Luxury",
}
INTEREST_LABEL = {
    "Beaches & coastline": "Beaches & coastline",
    "Mountains & nature": "Mountains & nature",
    "History & culture": "History & culture",
    "Nightlife & entertainment": "Nightlife & entertainment",
    "Food & culinary scene": None,
    "Adventure sports": "Adventure sports",
    "Relaxation & wellness": None,
    "Shopping & city life": "Shopping & city life",
}
TRIP_STYLE_LABEL = {
    "City break": "City break",
    "Beach holiday": "Beach holiday",
    "Road trip": "Road trip",
    "Backpacking": "Backpacking",
    "Luxury resort stay": "Luxury resort stay",
    "Cultural immersion": None,
    "Wildlife & nature": None,
    "Festivals & events": None,
}
COMPANION_LABEL = {
    "Solo": "Solo",
    "Partner": "Partner",
    "Friends or family group": "Friends/family group",
}
SEASON_LABEL = {
    "Warm season": "Warm season",
    "Shoulder season": "Shoulder season",
    "Cold season / winter travel": "Cold season/winter",
}
ACCOM_LABEL = {
    "Hostel / budget stay": "Hostel/budget stay",
    "Standard hotel": "Standard hotel",
    "Resort or high-end stay": None,
}
CLIMATE_LABEL = {
    "Warm & tropical": "Warm & tropical",
    "Temperate": "Temperate",
    "Cool / cold": None,
}

# (survey key, is_multi, label_map, dest_meta_field, weight)
CONTENT_DIMENSIONS = [
    ("form_f", True, INTEREST_LABEL, "top_interest", 2.0),
    ("form_g", True, TRIP_STYLE_LABEL, "trip_style", 2.0),
    ("form_c", False, BUDGET_LABEL, "budget_tier", 1.5),
    ("form_k", False, CLIMATE_LABEL, "climate", 1.0),
    ("form_i", False, SEASON_LABEL, "season", 1.0),
    ("form_h", False, COMPANION_LABEL, "companions", 1.0),
    ("form_j", False, ACCOM_LABEL, "accommodation", 0.5),
]


class WayfareModel:
    """Loads trained_params.npz once and serves fast forward-pass inference."""

    def __init__(self):
        params = np.load(MODEL_DIR / "trained_params.npz")
        self.E = params["E"]          # (N_DEST, 16) destination embeddings
        self.W1, self.b1 = params["W1"], params["b1"]
        self.W2, self.b2 = params["W2"], params["b2"]
        self.W3, self.b3 = params["W3"], params["b3"]
        self.dest_names = json.loads((MODEL_DIR / "dest_names.json" if (MODEL_DIR / "dest_names.json").exists()
                                       else MODEL_DIR.parent / "data/processed/dest_names.json").read_text())
        self.n_dest = len(self.dest_names)
        self.dest_meta = self._load_enriched()
        # Candidate pool: destinations with enough real signal to trust.
        self.eligible_idx = np.array([i for i in range(self.n_dest)
                                       if (self.dest_meta.get(i, {}).get("popularity") or 0) >= MIN_SWIPES])

    def _load_enriched(self):
        """Row i of destinations_enriched.csv describes dest_names[i] (verified 1:1 by id)."""
        path = MODEL_DIR / "destinations_enriched.csv"
        if not path.exists():
            path = MODEL_DIR.parent / "data/processed/destinations_enriched.csv"
        meta = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                did = int(row["destination_id"])
                name = row["destination_name"]
                country = name.split(",")[-1].strip() if "," in name else name
                meta[did] = {
                    "country": country,
                    "region": "india" if country == "India" else "abroad",
                    "cost_multiplier": COUNTRY_COST_INDEX.get(country, DEFAULT_COST_MULTIPLIER),
                    "popularity": int(row["num_yes_swipes"]) if row.get("num_yes_swipes") else 0,
                    "budget_tier": row.get("typical_budget_tier") or None,
                    "top_interest": row.get("top_interest_among_fans") or None,
                    "trip_style": row.get("top_trip_style_among_fans") or None,
                    "companions": row.get("typical_companions") or None,
                    "season": row.get("typical_season_preference") or None,
                    "accommodation": row.get("typical_accommodation") or None,
                    "climate": row.get("typical_climate_preference") or None,
                }
        return meta

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def encode_survey(self, answers: dict) -> np.ndarray:
        """answers: {form_key: [selected option indices]} -> 40-dim multi-hot vector.
        Note: 'region' is intentionally not part of FORM_CARD, so it's skipped here —
        it's a candidate filter, not a neural network input."""
        vec = np.zeros(FEAT_DIM, dtype=np.float32)
        offset = 0
        for key, card in FORM_CARD.items():
            selected = answers.get(key, [])
            for idx in selected:
                if isinstance(idx, int) and 0 <= idx < card:
                    vec[offset + idx] = 1.0
            offset += card
        return vec

    def forward(self, user_vec: np.ndarray, dest_idx) -> np.ndarray:
        """Vectorized forward pass for one user against one or many destinations."""
        dest_idx = np.atleast_1d(dest_idx)
        de = self.E[dest_idx]                                   # (B, 16)
        uf = np.tile(user_vec, (len(dest_idx), 1))               # (B, 40)
        x0 = np.concatenate([uf, de], axis=1)                    # (B, 56)
        a1 = np.maximum(0, x0 @ self.W1 + self.b1)
        a2 = np.maximum(0, a1 @ self.W2 + self.b2)
        z3 = a2 @ self.W3 + self.b3
        return self._sigmoid(z3).flatten()

    def _selected_labels(self, answers: dict, key: str, label_map: dict, is_multi: bool):
        idxs = answers.get(key, [])
        opts = next(q["options"] for q in QUESTIONS if q["key"] == key)
        chosen = idxs if is_multi else idxs[:1]
        labels = set()
        for i in chosen:
            if isinstance(i, int) and 0 <= i < len(opts):
                mapped = label_map.get(opts[i])
                if mapped:
                    labels.add(mapped)
        return labels

    def content_score(self, answers: dict, dest_id: int) -> float:
        """Transparent rule-based match: fraction of applicable, weighted
        dimensions where the destination's own labelled attribute matches
        something the traveller picked in the survey."""
        meta = self.dest_meta.get(dest_id, {})
        num, den = 0.0, 0.0
        for key, is_multi, label_map, field, weight in CONTENT_DIMENSIONS:
            dest_val = meta.get(field)
            if not dest_val:
                continue
            den += weight
            selected = self._selected_labels(answers, key, label_map, is_multi)
            if dest_val in selected:
                num += weight
        return (num / den) if den > 0 else 0.5

    DIM_LABELS = {
        "form_f": "Interest", "form_g": "Trip style", "form_c": "Budget style",
        "form_k": "Climate", "form_i": "Season", "form_h": "Who you're with",
        "form_j": "Accommodation",
    }

    def content_breakdown(self, answers: dict, dest_id: int):
        """Per-dimension explainability: exactly which of the traveller's
        picks this specific destination does/doesn't share, so 'match %' is
        traceable to real answers instead of being a black box."""
        meta = self.dest_meta.get(dest_id, {})
        rows = []
        for key, is_multi, label_map, field, weight in CONTENT_DIMENSIONS:
            dest_val = meta.get(field)
            if not dest_val:
                continue
            selected = self._selected_labels(answers, key, label_map, is_multi)
            if not selected:
                continue  # question wasn't answered / not applicable -> skip, don't count against it
            rows.append({
                "dimension": self.DIM_LABELS.get(key, key),
                "destination_value": dest_val,
                "matched": dest_val in selected,
            })
        return rows

    def _region_filtered(self, region: str):
        if region == "both":
            return self.eligible_idx
        return np.array([i for i in self.eligible_idx if self.dest_meta[i]["region"] == region])

    def recommend(self, answers: dict, top_k: int = 10):
        region_idx = (answers.get("region") or [2])[0]
        region = REGION_BY_INDEX.get(region_idx, "both")

        candidates = self._region_filtered(region)
        if len(candidates) == 0:
            candidates = self.eligible_idx  # never return nothing

        user_vec = self.encode_survey(answers)
        neural_raw = self.forward(user_vec, candidates)

        content = np.zeros(len(candidates), dtype=np.float64)
        neural_shrunk = np.zeros(len(candidates), dtype=np.float64)
        for pos, dest_id in enumerate(candidates):
            popularity = self.dest_meta[int(dest_id)]["popularity"]
            conf = min(1.0, popularity / POP_FULL_CONF)
            neural_shrunk[pos] = 0.5 + (neural_raw[pos] - 0.5) * conf  # shrink toward neutral if low-signal
            content[pos] = self.content_score(answers, int(dest_id))

        # Ranking order: content score first, neural only nudges the order
        # within destinations that are (nearly) tied on content — it can
        # never outrank a clearly better content match.
        ranking_key = content + TIE_BREAK_WEIGHT * neural_shrunk
        order = np.argsort(-ranking_key)[:top_k]

        # Displayed "match %" = the content score alone, so a destination
        # that agrees with every applicable thing you picked shows exactly
        # 100 — not diluted by the neural tie-breaker.
        display = np.clip(100 * content[order], 1, 100)

        results = []
        for rank, pos in enumerate(order):
            dest_id = int(candidates[pos])
            payload = self._dest_payload(dest_id)
            payload["score"] = round(float(display[rank]) / 100.0, 4)          # 0..1, what the UI shows (= content match)
            payload["ai_confidence"] = round(float(neural_shrunk[pos]), 4)     # neural signal, shown separately if wanted
            payload["why_matched"] = self.content_breakdown(answers, dest_id)
            results.append(payload)
        return results

    def _dest_payload(self, dest_id: int, score=None):
        payload = {"destination_id": dest_id, "name": self.dest_names[dest_id]}
        payload.update(self.dest_meta.get(dest_id, {}))
        if score is not None:
            payload["score"] = score
        return payload

    def get_destination(self, dest_id: int):
        if dest_id < 0 or dest_id >= self.n_dest:
            return None
        return self._dest_payload(dest_id)

    @staticmethod
    def estimate_budget(answers: dict, dest_id_meta: dict, questions_by_key: dict):
        """Builds a per-category USD estimate from the traveller's own survey
        answers (budget style, trip length, who they travel with), then scales
        the whole thing by the CHOSEN DESTINATION's own cost-of-travel index
        (dest_id_meta['cost_multiplier']) — a shoestring week in Switzerland
        and a shoestring week in Vietnam are the same "budget style" but not
        the same number of dollars. Currency conversion to the traveller's
        local currency happens client-side, since that only needs a public FX
        API and the browser's own IP/locale, not anything the server knows."""
        budget_opts = questions_by_key["form_c"]["options"]
        length_opts = questions_by_key["form_b"]["options"]
        companion_opts = questions_by_key["form_h"]["options"]

        budget_label = budget_opts[answers.get("form_c", [1])[0]] if answers.get("form_c") else budget_opts[1]
        length_label = length_opts[answers.get("form_b", [1])[0]] if answers.get("form_b") else length_opts[1]
        companion_label = companion_opts[answers.get("form_h", [0])[0]] if answers.get("form_h") else companion_opts[0]

        daily = BUDGET_DAILY_USD.get(budget_label, BUDGET_DAILY_USD[budget_opts[1]])
        days = TRIP_LENGTH_DAYS.get(length_label, 6)
        travellers = COMPANION_COUNT.get(companion_label, 1)
        cost_mult = (dest_id_meta or {}).get("cost_multiplier", DEFAULT_COST_MULTIPLIER)
        country = (dest_id_meta or {}).get("country")

        breakdown = {k: round(v * cost_mult * days * travellers, 2) for k, v in daily.items()}
        total = round(sum(breakdown.values()), 2)
        return {
            "currency": "USD",
            "days": days,
            "travellers": travellers,
            "budget_style": budget_label,
            "country": country,
            "cost_multiplier": cost_mult,
            "breakdown": breakdown,
            "total_usd": total,
            "per_person_usd": round(total / travellers, 2),
        }


_model_singleton = None


def get_model() -> WayfareModel:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = WayfareModel()
    return _model_singleton
