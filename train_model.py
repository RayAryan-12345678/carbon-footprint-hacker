"""
train_model.py — Carbon Footprint Hacker
Regenerates the dataset and retrains all ML models.

Usage:
    python train_model.py
    python train_model.py --samples 10000
"""

import os, json, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

SEED       = 42
OUT_DIR    = os.path.join(os.path.dirname(__file__), "model")
CSV_PATH   = os.path.join(os.path.dirname(__file__), "carbon_dataset.csv")

# ── Emission constants ──
TRANSPORT_F = {"car_petrol":0.21,"car_diesel":0.17,"car_electric":0.05,
               "motorbike":0.11,"bus":0.089,"train":0.041,"cycle":0.0}
DIET_F      = {"heavy_meat":3.3,"omnivore":2.5,"flexitarian":1.9,
               "pescatarian":1.5,"vegetarian":1.0,"vegan":0.7}
GRID_F      = {"coal":1.0,"mixed":0.85,"partial_renew":0.5,"solar_wind":0.1}
HEAT_F      = {"gas":80,"electric":40,"heat_pump":20,"none":5}
RECYCLING_F = {"none":1.2,"some":1.0,"most":0.7,"compost":0.5}
FOOD_SRC_F  = {"imported":1.2,"mixed":1.0,"local":0.8}
FOOD_WAS_F  = {"high":1.3,"medium":1.1,"low":1.0}


def generate_dataset(n: int) -> pd.DataFrame:
    np.random.seed(SEED)
    rows = []
    tm = list(TRANSPORT_F.keys())
    dt = list(DIET_F.keys())
    es = list(GRID_F.keys())
    ht = list(HEAT_F.keys())
    rc = list(RECYCLING_F.keys())
    fs = list(FOOD_SRC_F.keys())
    fw = list(FOOD_WAS_F.keys())

    for _ in range(n):
        mode  = np.random.choice(tm, p=[.30,.10,.08,.12,.20,.12,.08])
        km    = np.clip(np.random.exponential(400 if mode not in ("cycle","train","bus") else 200), 0, 3000)
        fl    = np.random.choice([0,1,2,4,6,10], p=[.30,.25,.20,.12,.08,.05])
        kwh   = np.clip(np.random.normal(250, 120), 20, 2000)
        grid  = np.random.choice(es, p=[.30,.40,.20,.10])
        heat  = np.random.choice(ht, p=[.35,.35,.20,.10])
        diet  = np.random.choice(dt, p=[.15,.35,.18,.10,.14,.08])
        fsrc  = np.random.choice(fs, p=[.25,.55,.20])
        fwas  = np.random.choice(fw, p=[.25,.50,.25])
        waste = np.clip(np.random.normal(35, 15), 2, 200)
        rec   = np.random.choice(rc, p=[.15,.45,.28,.12])
        shop  = np.clip(np.random.exponential(80), 5, 500)

        c_t = km*TRANSPORT_F[mode] + (fl/12)*255
        c_e = kwh*GRID_F[grid]*0.233
        c_h = HEAT_F[heat]*GRID_F[grid]
        c_f = DIET_F[diet]*FOOD_SRC_F[fsrc]*FOOD_WAS_F[fwas]*30
        c_w = waste*1.2*RECYCLING_F[rec]
        c_s = shop*0.43
        total = max(10.0, (c_t+c_e+c_h+c_f+c_w+c_s)*(1+np.random.normal(0,.05)))

        rows.append({
            "transport_mode":mode, "km_per_month":round(km,1),
            "flights_per_year":fl,  "electricity_kwh":round(kwh,1),
            "energy_source":grid,   "heating_type":heat,
            "diet_type":diet,       "food_source":fsrc,
            "food_waste_level":fwas,"waste_kg":round(waste,1),
            "recycling":rec,        "shopping_spend":round(shop,1),
            "co2_transport":round(c_t,2), "co2_electricity":round(c_e,2),
            "co2_heating":round(c_h,2),   "co2_food":round(c_f,2),
            "co2_waste":round(c_w,2),     "co2_shopping":round(c_s,2),
            "co2_monthly_kg":round(total,2),
        })

    df = pd.DataFrame(rows)
    p33 = df["co2_monthly_kg"].quantile(0.33)
    p66 = df["co2_monthly_kg"].quantile(0.66)
    df["emission_tier"] = df["co2_monthly_kg"].apply(
        lambda x: "low" if x < p33 else ("moderate" if x < p66 else "high"))
    return df


def train(n_samples=5000):
    print("\n" + "="*54)
    print("  Carbon Footprint Hacker — Training Pipeline")
    print("="*54)

    print(f"\n[1/5] Generating {n_samples:,} training records...")
    df = generate_dataset(n_samples)
    df.to_csv(CSV_PATH, index=False)
    print(f"      Saved → {CSV_PATH}")
    print(f"      CO2 range : {df['co2_monthly_kg'].min():.0f}–{df['co2_monthly_kg'].max():.0f} kg/mo")
    print(f"      Tiers     : {df['emission_tier'].value_counts().to_dict()}")

    print("\n[2/5] Encoding features...")
    CAT = ["transport_mode","energy_source","heating_type","diet_type",
           "food_source","food_waste_level","recycling"]
    encoders = {}
    df_e = df.copy()
    for col in CAT:
        le = LabelEncoder()
        df_e[col+"_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    FEAT = ["km_per_month","flights_per_year","electricity_kwh","waste_kg","shopping_spend",
            "transport_mode_enc","energy_source_enc","heating_type_enc","diet_type_enc",
            "food_source_enc","food_waste_level_enc","recycling_enc"]

    X   = df_e[FEAT].values
    y_r = df_e["co2_monthly_kg"].values
    y_c = df_e["emission_tier"].values
    X_tr,X_te,yr_tr,yr_te,yc_tr,yc_te = train_test_split(X,y_r,y_c,test_size=.2,random_state=SEED)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    print(f"      Train: {len(X_tr):,}  Test: {len(X_te):,}  Features: {len(FEAT)}")

    print("\n[3/5] Training GradientBoostingRegressor...")
    reg = GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.08,random_state=SEED)
    reg.fit(X_tr, yr_tr)
    mae = mean_absolute_error(yr_te, reg.predict(X_te))
    r2  = r2_score(yr_te, reg.predict(X_te))
    print(f"      R²  = {r2:.4f}")
    print(f"      MAE = {mae:.2f} kg/month")

    print("\n[4/5] Training classifiers (NB + GB ensemble)...")
    gnb = GaussianNB()
    gnb.fit(X_tr_s, yc_tr)
    nb_acc = accuracy_score(yc_te, gnb.predict(X_te_s))

    gbc = GradientBoostingClassifier(n_estimators=150,max_depth=4,random_state=SEED)
    gbc.fit(X_tr, yc_tr)
    gb_acc = accuracy_score(yc_te, gbc.predict(X_te))
    print(f"      Naive Bayes accuracy : {nb_acc:.4f}")
    print(f"      GB Classifier accuracy: {gb_acc:.4f}")
    print("\n" + classification_report(yc_te, gbc.predict(X_te)))

    feat_imp = sorted(zip(FEAT, reg.feature_importances_), key=lambda x: -x[1])
    print("      Feature importances:")
    for f,v in feat_imp[:6]:
        print(f"        {f:<32} {'█'*int(v*40)} {v:.4f}")

    print(f"\n[5/5] Saving to ./{os.path.basename(OUT_DIR)}/")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/regressor.pkl",   "wb") as f: pickle.dump(reg,     f)
    with open(f"{OUT_DIR}/classifier.pkl",  "wb") as f: pickle.dump(gbc,     f)
    with open(f"{OUT_DIR}/naive_bayes.pkl", "wb") as f: pickle.dump(gnb,     f)
    with open(f"{OUT_DIR}/scaler.pkl",      "wb") as f: pickle.dump(scaler,  f)
    with open(f"{OUT_DIR}/encoders.pkl",    "wb") as f: pickle.dump(encoders,f)
    with open(f"{OUT_DIR}/feature_cols.json","w") as f: json.dump(FEAT, f)
    with open(f"{OUT_DIR}/metrics.json",    "w") as f:
        json.dump({
            "mae":round(mae,2), "r2":round(r2,4),
            "nb_accuracy":round(nb_acc,4), "gb_accuracy":round(gb_acc,4),
            "n_samples":n_samples, "n_features":len(FEAT),
            "tier_thresholds":{"low_max":round(df["co2_monthly_kg"].quantile(.33),1),
                               "moderate_max":round(df["co2_monthly_kg"].quantile(.66),1)},
            "feature_importances":{k:round(v,4) for k,v in feat_imp},
        }, f, indent=2)

    for fn in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(f"{OUT_DIR}/{fn}")
        print(f"      {fn:<28} {sz/1024:.0f} KB")

    print("\n" + "="*54)
    print("  Done! Run:  python app.py")
    print("  Then open: http://127.0.0.1:5000")
    print("="*54 + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5000,
                   help="Number of training records to generate (default: 5000)")
    args = p.parse_args()
    train(args.samples)
