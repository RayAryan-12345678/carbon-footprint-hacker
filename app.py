"""
app.py
======
Carbon Footprint Hacker — Flask backend.

Loads the trained ML models once at startup and serves:
  GET  /                 -> the frontend (templates/index.html)
  POST /api/predict      -> runs the ML pipeline on submitted lifestyle data
  GET  /api/model-info   -> training metrics for the header pills
  GET  /api/leaderboard  -> in-memory community leaderboard
  GET  /healthz          -> simple health check (useful for deploy platforms)
"""

import os
import json
import pickle
import random
from datetime import datetime

import numpy as np
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

GLOBAL_AVG_CO2 = 833  # kg/month — used for scoring + comparison

# ---------------------------------------------------------------------------
# Load all model files ONCE at startup
# ---------------------------------------------------------------------------
_model_files_present = all(
    os.path.exists(os.path.join(MODEL_DIR, f))
    for f in ["regressor.pkl", "classifier.pkl", "naive_bayes.pkl",
              "scaler.pkl", "encoders.pkl", "feature_cols.json", "metrics.json"]
)

if not _model_files_present:
    raise RuntimeError(
        "Model files not found in /model. Run `python train_model.py` first "
        "to generate the dataset and train the models before starting app.py."
    )

with open(os.path.join(MODEL_DIR, "regressor.pkl"), "rb") as f:
    regressor = pickle.load(f)
with open(os.path.join(MODEL_DIR, "classifier.pkl"), "rb") as f:
    classifier = pickle.load(f)
with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "rb") as f:
    naive_bayes = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

CATEGORICAL_COLS = list(encoders.keys())
CLASS_ORDER = METRICS["class_order"]  # order used by classifier.predict_proba

print(f"[startup] Loaded {len(FEATURE_COLS)} features, "
      f"R2={METRICS['r2_score']}, MAE={METRICS['mae_kg']}kg")

# In-memory leaderboard (resets on server restart — fine for a hackathon demo)
LEADERBOARD = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def encode(value, column):
    """Convert a text category to its trained integer code.
    Falls back to the most common / first known category if unseen."""
    mapping = encoders[column]
    if value in mapping:
        return mapping[value]
    return next(iter(mapping.values()))


def build_feature_vector(payload):
    row = []
    for col in FEATURE_COLS:
        val = payload.get(col)
        if col in CATEGORICAL_COLS:
            row.append(encode(val, col))
        else:
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)
    return np.array(row, dtype=float).reshape(1, -1)


def compute_breakdown(payload):
    """Recompute a human-readable category breakdown using the same
    real-world emission factors as the dataset, for the charts."""
    TRANSPORT_FACTORS = {
        "car_petrol": 0.21, "car_diesel": 0.17, "car_electric": 0.05,
        "motorbike": 0.11, "bus": 0.089, "train": 0.041, "cycle": 0.0,
    }
    GRID_FACTORS = {"coal": 1.6, "mixed": 1.0, "renewable": 0.25}
    HEATING_BASE = {"gas": 90, "electric": 60, "oil": 110, "none": 5}
    DIET_BASE = {"meat_heavy": 7.2, "omnivore": 5.0, "vegetarian": 2.8, "vegan": 1.5}
    FOOD_SOURCE_FACTOR = {"local": 0.8, "mixed": 1.0, "imported": 1.3}
    FOOD_WASTE_FACTOR = {"low": 0.9, "medium": 1.0, "high": 1.25}
    RECYCLING_FACTOR = {"most": 0.6, "some": 0.85, "none": 1.15}

    km = float(payload.get("km_per_month", 0) or 0)
    flights = float(payload.get("flights_per_year", 0) or 0)
    kwh = float(payload.get("electricity_kwh", 0) or 0)
    waste = float(payload.get("waste_kg", 0) or 0)
    spend = float(payload.get("shopping_spend", 0) or 0)

    transport_mode = payload.get("transport_mode", "car_petrol")
    energy_source = payload.get("energy_source", "mixed")
    heating_type = payload.get("heating_type", "gas")
    diet_type = payload.get("diet_type", "omnivore")
    food_source = payload.get("food_source", "mixed")
    food_waste_level = payload.get("food_waste_level", "medium")
    recycling = payload.get("recycling", "some")

    grid_factor = GRID_FACTORS.get(energy_source, 1.0)

    transport_co2 = km * TRANSPORT_FACTORS.get(transport_mode, 0.21) + (flights / 12) * 255
    electricity_co2 = kwh * grid_factor * 0.233
    heating_co2 = HEATING_BASE.get(heating_type, 90) * grid_factor
    food_co2 = (DIET_BASE.get(diet_type, 5.0) * FOOD_SOURCE_FACTOR.get(food_source, 1.0)
                * FOOD_WASTE_FACTOR.get(food_waste_level, 1.0) * 30)
    waste_co2 = waste * 1.2 * RECYCLING_FACTOR.get(recycling, 0.85)
    shopping_co2 = spend * 0.43

    return {
        "transport": round(transport_co2, 1),
        "electricity": round(electricity_co2, 1),
        "heating": round(heating_co2, 1),
        "food": round(food_co2, 1),
        "waste": round(waste_co2, 1),
        "shopping": round(shopping_co2, 1),
    }


def build_suggestions(breakdown, payload):
    """Rank the top emission categories and return actionable suggestions."""
    catalog = {
        "transport": [
            "Switch one car trip a week to cycling or public transport",
            "Consider carpooling or an electric vehicle for your next car",
        ],
        "electricity": [
            "Switch to a renewable energy tariff if available",
            "Unplug idle appliances and switch to LED bulbs",
        ],
        "heating": [
            "Lower your thermostat by 1-2°C and improve insulation",
            "Consider a heat pump instead of gas/oil heating",
        ],
        "food": [
            "Try 2-3 plant-based meals per week to cut food emissions",
            "Buy more local, seasonal produce over imported food",
        ],
        "waste": [
            "Increase recycling — it cuts waste emissions by up to 40%",
            "Compost food scraps instead of sending them to landfill",
        ],
        "shopping": [
            "Buy fewer, higher-quality items instead of fast fashion",
            "Repair or resell items before replacing them",
        ],
    }
    ranked = sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True)
    suggestions = []
    for category, amount in ranked:
        for tip in catalog.get(category, []):
            suggestions.append({
                "category": category,
                "tip": tip,
                "potential_saving_kg": round(amount * 0.25, 1),
            })
        if len(suggestions) >= 6:
            break
    return suggestions[:6]


def build_trend(co2_monthly):
    """Project a 12-month current vs optimised trend with light seasonal
    variation, for the dashboard trend chart."""
    seasonal = [1.08, 1.05, 1.0, 0.95, 0.92, 0.9, 0.93, 0.95, 0.98, 1.02, 1.06, 1.1]
    current = [round(co2_monthly * s, 1) for s in seasonal]
    optimised = [round(v * 0.78, 1) for v in current]  # ~22% achievable reduction
    return {"months": list(range(1, 13)), "current": current, "optimised": optimised}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/model-info")
def model_info():
    return jsonify({
        "r2_score": METRICS["r2_score"],
        "mae_kg": METRICS["mae_kg"],
        "ensemble_accuracy": METRICS["ensemble_accuracy"],
        "gb_classifier_accuracy": METRICS["gb_classifier_accuracy"],
        "naive_bayes_accuracy": METRICS["naive_bayes_accuracy"],
        "n_records": METRICS["n_records"],
        "n_features": METRICS["n_features"],
        "feature_importance": METRICS["feature_importance"],
        "global_avg_co2": GLOBAL_AVG_CO2,
    })


@app.route("/api/leaderboard")
def leaderboard():
    top = sorted(LEADERBOARD, key=lambda r: r["score"], reverse=True)[:20]
    return jsonify(top)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}

    missing = [c for c in FEATURE_COLS if payload.get(c) in (None, "")]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    X = build_feature_vector(payload)

    # 1. Regressor -> exact CO2 prediction
    co2_monthly = float(regressor.predict(X)[0])
    co2_monthly = max(0.0, co2_monthly)

    # 2. Ensemble classification (NB x 0.35 + GB x 0.65)
    gb_proba_row = classifier.predict_proba(X)[0]
    gb_classes = list(classifier.classes_)

    X_scaled = scaler.transform(X)
    nb_proba_row = naive_bayes.predict_proba(X_scaled)[0]
    nb_classes = list(naive_bayes.classes_)

    combined = {}
    for cls in CLASS_ORDER:
        gb_p = gb_proba_row[gb_classes.index(cls)] if cls in gb_classes else 0.0
        nb_p = nb_proba_row[nb_classes.index(cls)] if cls in nb_classes else 0.0
        combined[cls] = nb_p * 0.35 + gb_p * 0.65

    tier = max(combined, key=combined.get)
    confidence = round(combined[tier] * 100, 1)
    tier_proba = {k: round(v * 100, 1) for k, v in combined.items()}

    # 3. Carbon score: 0-100, higher = greener
    score = max(0, min(100, round(100 - (co2_monthly / GLOBAL_AVG_CO2) * 50)))

    # 4. Category breakdown + suggestions
    breakdown = compute_breakdown(payload)
    suggestions = build_suggestions(breakdown, payload)

    # 5. 12-month trend
    trend = build_trend(co2_monthly)

    vs_global_pct = round(((co2_monthly - GLOBAL_AVG_CO2) / GLOBAL_AVG_CO2) * 100, 1)

    result = {
        "co2_monthly": round(co2_monthly, 1),
        "co2_annual_t": round(co2_monthly * 12 / 1000, 2),
        "carbon_score": score,
        "tier": tier,
        "tier_confidence": confidence,
        "tier_proba": tier_proba,
        "vs_global_pct": vs_global_pct,
        "global_avg_co2": GLOBAL_AVG_CO2,
        "breakdown": breakdown,
        "suggestions": suggestions,
        "trend_current": trend["current"],
        "trend_optimised": trend["optimised"],
        "trend_months": trend["months"],
        "model_info": {
            "r2_score": METRICS["r2_score"],
            "mae_kg": METRICS["mae_kg"],
            "ensemble_accuracy": METRICS["ensemble_accuracy"],
            "top_feature": next(iter(METRICS["feature_importance"])),
        },
    }

    # Add to in-memory leaderboard
    LEADERBOARD.append({
        "name": payload.get("name") or f"Anon{random.randint(1000, 9999)}",
        "score": score,
        "tier": tier,
        "co2_monthly": round(co2_monthly, 1),
        "timestamp": datetime.utcnow().isoformat(),
    })
    if len(LEADERBOARD) > 200:
        del LEADERBOARD[0]

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
