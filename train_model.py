"""
train_model.py
================
Carbon Footprint Hacker — Dataset generator + ML training pipeline.

Generates 5,000 synthetic-but-realistic lifestyle records using real
IPCC / IEA emission factors, then trains:
  1. GradientBoostingRegressor   -> predicts exact CO2 (kg/month)
  2. GradientBoostingClassifier  -> predicts emission tier (low/moderate/high)
  3. GaussianNB                  -> second classifier, combined in an ensemble

Run:
    python train_model.py

Outputs:
    carbon_dataset.csv
    model/regressor.pkl
    model/classifier.pkl
    model/naive_bayes.pkl
    model/scaler.pkl
    model/encoders.pkl
    model/feature_cols.json
    model/metrics.json
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

RANDOM_STATE = 42
N_RECORDS = 5000

np.random.seed(RANDOM_STATE)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Real-world emission factors (IPCC AR6 / IEA 2023 / ICAO)
# ---------------------------------------------------------------------------

TRANSPORT_FACTORS = {          # kg CO2 per km
    "car_petrol": 0.21,
    "car_diesel": 0.17,
    "car_electric": 0.05,
    "motorbike": 0.11,
    "bus": 0.089,
    "train": 0.041,
    "cycle": 0.000,
}

GRID_FACTORS = {                # relative multiplier on electricity emissions
    "coal": 1.6,
    "mixed": 1.0,
    "renewable": 0.25,
}

HEATING_BASE = {                # kg CO2 baseline per month
    "gas": 90,
    "electric": 60,
    "oil": 110,
    "none": 5,
}

DIET_BASE = {                   # kg CO2 baseline per day (x30 in formula)
    "meat_heavy": 7.2,
    "omnivore": 5.0,
    "vegetarian": 2.8,
    "vegan": 1.5,
}

FOOD_SOURCE_FACTOR = {
    "local": 0.8,
    "mixed": 1.0,
    "imported": 1.3,
}

FOOD_WASTE_FACTOR = {
    "low": 0.9,
    "medium": 1.0,
    "high": 1.25,
}

RECYCLING_FACTOR = {
    "most": 0.6,
    "some": 0.85,
    "none": 1.15,
}

TRANSPORT_MODES = list(TRANSPORT_FACTORS.keys())
GRID_TYPES = list(GRID_FACTORS.keys())
HEATING_TYPES = list(HEATING_BASE.keys())
DIET_TYPES = list(DIET_BASE.keys())
FOOD_SOURCES = list(FOOD_SOURCE_FACTOR.keys())
FOOD_WASTE_LEVELS = list(FOOD_WASTE_FACTOR.keys())
RECYCLING_LEVELS = list(RECYCLING_FACTOR.keys())

CATEGORICAL_COLS = [
    "transport_mode", "energy_source", "heating_type",
    "diet_type", "food_source", "food_waste_level", "recycling",
]

FEATURE_COLS = [
    "transport_mode", "km_per_month", "flights_per_year",
    "electricity_kwh", "energy_source", "heating_type",
    "diet_type", "food_source", "food_waste_level",
    "waste_kg", "recycling", "shopping_spend",
]


def generate_dataset(n=N_RECORDS):
    rows = []
    for _ in range(n):
        transport_mode = np.random.choice(TRANSPORT_MODES, p=[0.28, 0.12, 0.07, 0.08, 0.17, 0.13, 0.15])
        km_per_month = max(0, np.random.gamma(4, 120))
        flights_per_year = np.random.choice([0, 1, 2, 3, 4, 6, 10], p=[0.35, 0.2, 0.18, 0.12, 0.08, 0.05, 0.02])

        electricity_kwh = max(20, np.random.normal(250, 90))
        energy_source = np.random.choice(GRID_TYPES, p=[0.3, 0.5, 0.2])

        heating_type = np.random.choice(HEATING_TYPES, p=[0.45, 0.3, 0.15, 0.1])

        diet_type = np.random.choice(DIET_TYPES, p=[0.25, 0.45, 0.2, 0.1])
        food_source = np.random.choice(FOOD_SOURCES, p=[0.3, 0.5, 0.2])
        food_waste_level = np.random.choice(FOOD_WASTE_LEVELS, p=[0.3, 0.45, 0.25])

        waste_kg = max(2, np.random.normal(28, 10))
        recycling = np.random.choice(RECYCLING_LEVELS, p=[0.3, 0.45, 0.25])

        shopping_spend = max(0, np.random.gamma(3, 45))

        # --- Ground-truth CO2 using real emission factors ---
        transport_co2 = km_per_month * TRANSPORT_FACTORS[transport_mode] + (flights_per_year / 12) * 255
        grid_factor = GRID_FACTORS[energy_source]
        electricity_co2 = electricity_kwh * grid_factor * 0.233
        heating_co2 = HEATING_BASE[heating_type] * grid_factor
        food_co2 = (DIET_BASE[diet_type] * FOOD_SOURCE_FACTOR[food_source]
                    * FOOD_WASTE_FACTOR[food_waste_level] * 30)
        waste_co2 = waste_kg * 1.2 * RECYCLING_FACTOR[recycling]
        shopping_co2 = shopping_spend * 0.43

        total_co2 = (transport_co2 + electricity_co2 + heating_co2
                     + food_co2 + waste_co2 + shopping_co2)

        # 5% random noise for real-world messiness
        total_co2 = total_co2 * (1 + np.random.normal(0, 0.05))
        total_co2 = max(20, total_co2)

        rows.append({
            "transport_mode": transport_mode,
            "km_per_month": round(km_per_month, 1),
            "flights_per_year": int(flights_per_year),
            "electricity_kwh": round(electricity_kwh, 1),
            "energy_source": energy_source,
            "heating_type": heating_type,
            "diet_type": diet_type,
            "food_source": food_source,
            "food_waste_level": food_waste_level,
            "waste_kg": round(waste_kg, 1),
            "recycling": recycling,
            "shopping_spend": round(shopping_spend, 1),
            "co2_monthly_kg": round(total_co2, 2),
        })

    df = pd.DataFrame(rows)

    # Assign emission tiers by percentile -> balanced classes
    low_cut, high_cut = df["co2_monthly_kg"].quantile([1 / 3, 2 / 3])

    def tier_of(v):
        if v <= low_cut:
            return "low"
        elif v <= high_cut:
            return "moderate"
        return "high"

    df["emission_tier"] = df["co2_monthly_kg"].apply(tier_of)
    return df, float(low_cut), float(high_cut)


def main():
    print("Generating synthetic dataset...")
    df, low_cut, high_cut = generate_dataset()
    df.to_csv("carbon_dataset.csv", index=False)
    print(f"Saved carbon_dataset.csv ({len(df)} rows)")
    print(f"Tier cutoffs -> low <= {low_cut:.1f} kg | moderate <= {high_cut:.1f} kg | high above")

    # -----------------------------------------------------------------
    # Step 1: Encode categorical features
    # -----------------------------------------------------------------
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

    X = df_enc[FEATURE_COLS].values
    y_reg = df_enc["co2_monthly_kg"].values
    y_clf = df_enc["emission_tier"].values

    # -----------------------------------------------------------------
    # Step 2: Train / test split (80/20)
    # -----------------------------------------------------------------
    X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # -----------------------------------------------------------------
    # Step 3: Model 1 - GradientBoostingRegressor
    # -----------------------------------------------------------------
    print("Training GradientBoostingRegressor...")
    regressor = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08, random_state=RANDOM_STATE
    )
    regressor.fit(X_train, yreg_train)
    reg_preds = regressor.predict(X_test)
    r2 = r2_score(yreg_test, reg_preds)
    mae = mean_absolute_error(yreg_test, reg_preds)
    print(f"  R2 = {r2:.4f} | MAE = {mae:.2f} kg/month")

    # -----------------------------------------------------------------
    # Step 4: Model 2 - GradientBoostingClassifier
    # -----------------------------------------------------------------
    print("Training GradientBoostingClassifier...")
    classifier = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
    )
    classifier.fit(X_train, yclf_train)
    gb_acc = accuracy_score(yclf_test, classifier.predict(X_test))
    print(f"  GB Classifier accuracy = {gb_acc:.4f}")

    # -----------------------------------------------------------------
    # Step 5: StandardScaler + Model 3 - GaussianNB
    # -----------------------------------------------------------------
    print("Training GaussianNB (with StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train_scaled, yclf_train)
    nb_acc = accuracy_score(yclf_test, naive_bayes.predict(X_test_scaled))
    print(f"  NaiveBayes accuracy = {nb_acc:.4f}")

    # -----------------------------------------------------------------
    # Step 6: Ensemble accuracy (NB x 0.35 + GB x 0.65)
    # -----------------------------------------------------------------
    classes = classifier.classes_  # shared class order used by both models
    gb_proba = classifier.predict_proba(X_test)
    nb_proba_raw = naive_bayes.predict_proba(X_test_scaled)

    # Align NB's class order to the classifier's class order
    nb_class_index = {c: i for i, c in enumerate(naive_bayes.classes_)}
    nb_proba = np.zeros_like(gb_proba)
    for j, c in enumerate(classes):
        nb_proba[:, j] = nb_proba_raw[:, nb_class_index[c]]

    ensemble_proba = nb_proba * 0.35 + gb_proba * 0.65
    ensemble_preds = classes[np.argmax(ensemble_proba, axis=1)]
    ensemble_acc = accuracy_score(yclf_test, ensemble_preds)
    print(f"  Ensemble (NB*0.35 + GB*0.65) accuracy = {ensemble_acc:.4f}")

    # -----------------------------------------------------------------
    # Feature importance (from regressor)
    # -----------------------------------------------------------------
    importances = dict(zip(FEATURE_COLS, regressor.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # -----------------------------------------------------------------
    # Save everything
    # -----------------------------------------------------------------
    with open(os.path.join(MODEL_DIR, "regressor.pkl"), "wb") as f:
        pickle.dump(regressor, f)
    with open(os.path.join(MODEL_DIR, "classifier.pkl"), "wb") as f:
        pickle.dump(classifier, f)
    with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
        pickle.dump(naive_bayes, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODEL_DIR, "feature_cols.json"), "w") as f:
        json.dump(FEATURE_COLS, f, indent=2)

    metrics = {
        "r2_score": round(r2, 4),
        "mae_kg": round(mae, 2),
        "gb_classifier_accuracy": round(gb_acc, 4),
        "naive_bayes_accuracy": round(nb_acc, 4),
        "ensemble_accuracy": round(ensemble_acc, 4),
        "n_records": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "tier_cutoffs": {"low_max": round(low_cut, 1), "moderate_max": round(high_cut, 1)},
        "feature_importance": {k: round(v, 4) for k, v in importances.items()},
        "class_order": list(classes),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll model files saved to model/")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
