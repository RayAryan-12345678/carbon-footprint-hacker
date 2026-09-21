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
exist before `app.py` starts — either commit them to your repo (simplest for
a hackathon demo) or run `python train_model.py` as a build/release step.

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
