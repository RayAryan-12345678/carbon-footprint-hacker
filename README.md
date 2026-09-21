# 🌍 Carbon Footprint Hacker

AI-powered carbon emission calculator. A trained scikit-learn pipeline
(GradientBoostingRegressor + GradientBoostingClassifier + GaussianNB ensemble)
predicts a user's monthly CO₂ output, emission tier, and carbon score from
12 lifestyle inputs, served through a Flask API with a Chart.js dashboard.

## Project structure

```
carbon_footprint_hacker/
├── app.py                  # Flask server + API routes
├── train_model.py          # Dataset generator + model training
├── requirements.txt
├── Procfile                 # gunicorn start command (Render/Heroku/Railway)
├── carbon_dataset.csv       # generated training data (5,000 rows)
├── templates/
│   └── index.html           # frontend (calculator, dashboard, leaderboard, how it works)
└── model/                   # generated model artifacts
    ├── regressor.pkl
    ├── classifier.pkl
    ├── naive_bayes.pkl
    ├── scaler.pkl
    ├── encoders.pkl
    ├── feature_cols.json
    └── metrics.json
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train the models (only needed once, or whenever you want to retrain)
python train_model.py

# Start the server
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Deploy

The app reads the port from the `PORT` environment variable and binds to
`0.0.0.0`, so it works out of the box on Render, Railway, Fly.io, Heroku, or
any platform that runs a `Procfile` / `gunicorn app:app` process.

**Important:** the trained `model/*.pkl` files and `carbon_dataset.csv` must
exist before `app.py` starts, and they must have been trained with the same
`scikit-learn`/`numpy` versions pinned in `requirements.txt` — pickled
scikit-learn models are not guaranteed to load on a different library
version, and a mismatch will crash the server on startup with an error like
`ValueError: ... is not a known BitGenerator` or `InconsistentVersionWarning`.

The safest approach is to **always run `python train_model.py` as part of
your build step** (as shown below) rather than committing `.pkl` files
trained on a different machine — that way the models are always retrained
against whatever versions the build just installed, so they can never drift
out of sync with `requirements.txt`. If you'd rather commit the `.pkl` files
to skip retraining on every deploy, first confirm locally that your installed
`scikit-learn`/`numpy` versions exactly match `requirements.txt`.

### Render / Railway / Heroku (Procfile-based)
1. Push this folder to a GitHub repo (include the `model/` folder and
   `carbon_dataset.csv`, or add a build step that runs `python train_model.py`).
2. Create a new **Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt && python train_model.py`
   (or skip the `train_model.py` step if you committed the `model/` folder).
4. Start command: `gunicorn app:app` (already set in the `Procfile`).
5. No environment variables are required; `PORT` is provided automatically.

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python train_model.py
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

## API

| Route | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/predict` | POST | Runs the ML pipeline on lifestyle JSON, returns prediction |
| `/api/model-info` | GET | Training metrics (R², MAE, accuracy) |
| `/api/leaderboard` | GET | In-memory community leaderboard |
| `/healthz` | GET | Health check |

## Retraining

Edit the emission factors or dataset size at the top of `train_model.py`,
then re-run `python train_model.py` to regenerate `carbon_dataset.csv` and
every file in `model/`.
