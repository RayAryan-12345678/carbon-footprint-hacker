"""
Carbon Footprint Hacker — Flask Backend
Serves the frontend AND exposes the ML prediction API.
No flask-cors needed — frontend is served from the same Flask app.
"""

import os, json, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

# ─────────────────────────────────────────
# Load models once at startup
# ─────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

with open(f"{MODEL_DIR}/regressor.pkl",    "rb") as f: REGRESSOR   = pickle.load(f)
with open(f"{MODEL_DIR}/classifier.pkl",   "rb") as f: CLASSIFIER  = pickle.load(f)
with open(f"{MODEL_DIR}/naive_bayes.pkl",  "rb") as f: NAIVE_BAYES = pickle.load(f)
with open(f"{MODEL_DIR}/scaler.pkl",       "rb") as f: SCALER      = pickle.load(f)
with open(f"{MODEL_DIR}/encoders.pkl",     "rb") as f: ENCODERS    = pickle.load(f)
with open(f"{MODEL_DIR}/feature_cols.json")      as f: FEATURE_COLS = json.load(f)
with open(f"{MODEL_DIR}/metrics.json")           as f: METRICS      = json.load(f)

# ─────────────────────────────────────────
# Emission constants (mirrors training data)
# ─────────────────────────────────────────
TRANSPORT_F = {"car_petrol":0.21,"car_diesel":0.17,"car_electric":0.05,
               "motorbike":0.11,"bus":0.089,"train":0.041,"cycle":0.0}
GRID_F      = {"coal":1.0,"mixed":0.85,"partial_renew":0.5,"solar_wind":0.1}
HEAT_F      = {"gas":80,"electric":40,"heat_pump":20,"none":5}
DIET_F      = {"heavy_meat":3.3,"omnivore":2.5,"flexitarian":1.9,
               "pescatarian":1.5,"vegetarian":1.0,"vegan":0.7}
FSRC_F      = {"imported":1.2,"mixed":1.0,"local":0.8}
FWASTE_F    = {"high":1.3,"medium":1.1,"low":1.0}
REC_F       = {"none":1.2,"some":1.0,"most":0.7,"compost":0.5}
GLOBAL_AVG  = 833.0   # kg/month world average

SUGGESTIONS = {
    "transport":[
        ("Switch to public transport 3 days/week",0.30,"🚌"),
        ("Cycle or walk for trips under 5 km",0.15,"🚴"),
        ("Carpool to halve per-person emissions",0.20,"🤝"),
        ("Switch to an electric vehicle",0.75,"⚡"),
    ],
    "electricity":[
        ("Replace all bulbs with LED (75% less energy)",0.12,"💡"),
        ("Install rooftop solar panels",0.60,"☀️"),
        ("Switch to a green energy tariff",0.40,"🌬️"),
        ("Unplug devices on standby",0.05,"🔌"),
    ],
    "food":[
        ("Adopt a flexitarian diet — meat 2x/week",0.35,"🥦"),
        ("Buy local & seasonal produce",0.15,"🛒"),
        ("Reduce food waste through meal planning",0.20,"📋"),
        ("Try one fully plant-based day per week",0.10,"🌱"),
    ],
    "waste":[
        ("Compost food scraps (diverts ~30% of waste)",0.30,"♻️"),
        ("Increase recycling rate at home",0.25,"🗑️"),
    ],
    "shopping":[
        ("Buy secondhand or swap clothing",0.40,"👕"),
        ("Adopt a minimalist shopping mindset",0.30,"🧘"),
    ],
}

# ─────────────────────────────────────────
# Helper: encode raw form data → feature vector
# ─────────────────────────────────────────
def encode(data: dict) -> np.ndarray:
    def safe_enc(col, val):
        le = ENCODERS[col]
        val = val if val in le.classes_ else le.classes_[0]
        return int(le.transform([val])[0])

    return np.array([[
        float(data.get("km_per_month",    500)),
        float(data.get("flights_per_year", 2)),
        float(data.get("electricity_kwh", 200)),
        float(data.get("waste_kg",         30)),
        float(data.get("shopping_spend",   80)),
        safe_enc("transport_mode",   data.get("transport_mode",   "car_petrol")),
        safe_enc("energy_source",    data.get("energy_source",    "mixed")),
        safe_enc("heating_type",     data.get("heating_type",     "electric")),
        safe_enc("diet_type",        data.get("diet_type",        "omnivore")),
        safe_enc("food_source",      data.get("food_source",      "mixed")),
        safe_enc("food_waste_level", data.get("food_waste_level", "medium")),
        safe_enc("recycling",        data.get("recycling",        "some")),
    ]])


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    X     = encode(data)
    X_sc  = SCALER.transform(X)

    # — Regression: monthly CO2 kg —
    co2 = float(max(10.0, REGRESSOR.predict(X)[0]))

    # — Classification: tier (ensemble NB + GB) —
    nb_proba  = dict(zip(NAIVE_BAYES.classes_,  NAIVE_BAYES.predict_proba(X_sc)[0]))
    gb_proba  = dict(zip(CLASSIFIER.classes_,   CLASSIFIER.predict_proba(X)[0]))
    classes   = ["high","low","moderate"]
    ensemble  = {c: round(nb_proba.get(c,0)*0.35 + gb_proba.get(c,0)*0.65, 4) for c in classes}
    tier      = max(ensemble, key=ensemble.get)
    confidence= round(ensemble[tier]*100, 1)

    # — Per-category breakdown —
    km  = float(data.get("km_per_month",    0))
    fl  = float(data.get("flights_per_year",0))
    kwh = float(data.get("electricity_kwh", 0))
    wkg = float(data.get("waste_kg",        0))
    sh  = float(data.get("shopping_spend",  0))
    tf  = TRANSPORT_F.get(data.get("transport_mode","car_petrol"), 0.21)
    gf  = GRID_F.get(data.get("energy_source","mixed"), 0.85)
    hf  = HEAT_F.get(data.get("heating_type","electric"), 40)
    df_ = DIET_F.get(data.get("diet_type","omnivore"), 2.5)
    sf  = FSRC_F.get(data.get("food_source","mixed"), 1.0)
    wf  = FWASTE_F.get(data.get("food_waste_level","medium"), 1.1)
    rf  = REC_F.get(data.get("recycling","some"), 1.0)

    breakdown = {
        "transport":   round(km*tf + (fl/12)*255, 1),
        "electricity": round(kwh*gf*0.233, 1),
        "heating":     round(hf*gf, 1),
        "food":        round(df_*sf*wf*30, 1),
        "waste":       round(wkg*1.2*rf, 1),
        "shopping":    round(sh*0.43, 1),
    }

    # — Carbon score 0-100 (higher = greener) —
    score = max(0, min(100, round(100 - (co2/GLOBAL_AVG)*50)))

    # — Smart suggestions (top 6 ranked by impact) —
    suggestions = []
    for cat, co2v in sorted(breakdown.items(), key=lambda x: -x[1]):
        for text, pct, icon in SUGGESTIONS.get(cat, [])[:2]:
            suggestions.append({
                "category": cat,
                "icon": icon,
                "text": text,
                "save_kg": round(co2v*pct, 1),
                "save_pct": round(pct*100),
            })
        if len(suggestions) >= 6:
            break

    # — 12-month trend projection —
    months    = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal  = [round(co2*(0.92 + 0.16*abs(np.sin(i*0.52))), 1) for i in range(12)]
    optimised = [round(v*(1-i*0.025), 1) for i,v in enumerate(seasonal)]

    return jsonify({
        "co2_monthly":    round(co2, 1),
        "co2_annual":     round(co2*12, 1),
        "co2_annual_t":   round(co2*12/1000, 2),
        "carbon_score":   score,
        "tier":           tier,
        "confidence":     confidence,
        "tier_proba":     ensemble,
        "breakdown":      breakdown,
        "suggestions":    suggestions,
        "months":         months,
        "trend_current":  seasonal,
        "trend_optimised":optimised,
        "vs_global_pct":  round((co2/GLOBAL_AVG - 1)*100, 1),
        "model_info": {
            "regressor":        "GradientBoostingRegressor (200 trees)",
            "classifier":       "GB + NaiveBayes ensemble",
            "training_samples": METRICS["n_samples"],
            "mae":              METRICS["mae"],
            "r2":               METRICS["r2"],
            "gb_accuracy":      METRICS["gb_accuracy"],
            "top_features":     list(METRICS["feature_importances"].items())[:5],
        }
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(METRICS)


if __name__ == "__main__":
    print("\n" + "="*52)
    print("  Carbon Footprint Hacker — running at")
    print("  http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=False, port=5000)
