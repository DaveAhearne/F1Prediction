# F1 Podium Predictor
A machine learning project to predict the probability of a Formula 1 driver finishing on the podium (top 3) for a given race.

## Project Overview
Using historical race data from 1990 to present, we train a classification model to predict podium finishes. The model outputs a probability for each driver in a given race, rather than a binary yes/no prediction.

## Objectives
- Predict the probability of a podium finish for each driver in a race
- Beat a pre-weekend rolling podium rate heuristic baseline (AUC > 0.79, Brier Score < 0.12)
- Validate against 2025 race results using live data from the Jolpica API

## Data Sources
- **Kaggle — Formula 1 Race Data** by jtrotman: [historical race results, qualifying, constructors (1950–2024)](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
- **Jolpica API**: live 2025 race results for model validation

## Methodology
1. Establish heuristic baselines from qualifying position data and rolling driver form
2. Train a baseline model on raw features with data validation
3. Iteratively engineer features and measure improvement
4. Validate final model against 2025 season results
5. Automate the full loop: ingest → validate → drift detection → conditional retraining → promotion

## Success Criteria
Defined upfront to avoid p-hacking. The qualifying heuristic (AUC 0.93, Brier 0.059) was rejected as a baseline because it relies on same-weekend data and leaks significant signal — qualifying position already encodes car setup, tyre performance, and driver form. The true pre-weekend baseline is the rolling podium rate heuristic:

| Metric | Qualifying Heuristic (rejected) | Rolling Rate Heuristic (baseline) | Target |
|--------|--------------------------------|-----------------------------------|--------|
| ROC-AUC | 0.93 | 0.79 | > 0.79 |
| Brier Score | 0.059 | 0.12 | < 0.12 |

## What We Know So Far

### EDA
- The dataset covers 1,171 races from 1950 to 2026 (calendar pre-populated) and 26,759 result rows
- **2025 is held out as the test set** — all EDA and feature intuitions are derived from pre-2025 data only
- Fastest lap, fastest lap speed, and rank columns have too many gaps to use reliably as features
- Missing `position` values almost always correspond to DNFs, inferable from `statusId`
- Constructor dominance is clearly visible across eras — constructor identity should carry real predictive signal
- Grid vs finish position difference has narrowed over time, suggesting the modern era is more "locked in" and overtaking is harder
- Driver podium rates vary significantly (Fangio 60.3%, Hamilton 56.7%, Verstappen 53.6% among drivers with 24+ starts) — driver identity or a skill proxy is worth including as a feature
- DNF rates have fallen dramatically from ~50% in the 1950s to ~15% today; constructor-specific reliability varies by era and regulation cycle

### Heuristic Baselines
Two baselines were evaluated on 2025 race results:

**Qualifying position heuristic** — maps each grid position to its historical podium rate (1990–2024). Produces strong metrics (AUC 0.93, Brier 0.059) but is not a valid baseline because it uses same-weekend qualifying data.

**Rolling 5-race podium rate** — for each driver, computes the fraction of their last 5 races (strictly prior to the current race) where they finished on the podium. Rookies and drivers with no prior history default to 0. This is the true pre-weekend baseline.

| Heuristic | ROC-AUC | Brier Score |
|-----------|---------|-------------|
| Qualifying position | 0.93 | 0.059 |
| Rolling 5-race podium rate | 0.79 | 0.12 |

### LightGBM Baseline
A binary LightGBM classifier trained on a minimal feature set to establish an ML baseline before feature engineering begins. Key decisions made here:

- **Training data**: 1990–2024, filtered before applying Pandera validation to avoid spurious failures from pre-1990 recording inconsistencies
- **Data validation**: Pandera `DataFrameModel` validates the training frame post-filter, using a warning-only approach — production pipelines would hard-stop on failures
- **Leakage discipline**: Rolling aggregations use `shift(1)` to ensure only prior-race data is visible at training time
- **Evaluation metrics**: Brier score (primary — measures calibration of predicted probabilities) and ROC-AUC (secondary — measures driver ranking quality), both defined before training to avoid unconscious cherry-picking
- **NaN handling**: NaN values in rolling rate features are primarily a cold-start issue for debut races, not DNF artefacts

### LightGBM with Full Feature Engineering
The current model adds a comprehensive feature set on top of the baseline:

- Driver rolling podium rate at 3, 5, and 10-race windows
- Constructor rolling podium rate at 3, 5, and 10-race windows
- Constructor mechanical DNF rate (5-race window)
- Driver age and career race count
- Circuit-specific driver podium rate
- Championship standings position
- Season podium rate
- Regulation era encoding
- Circuit type (street, permanent, hybrid)
- Grid size
- Home race flag

Training uses walk-forward cross-validation (10-year training window, 1-year validation window) with experiment tracking in MLflow and data versioning via LakeFS.

### Serving
The trained model is exported to ONNX and served via a FastAPI application. At startup the server loads the full historical dataset from LakeFS, engineers all rolling features, and holds the result in memory. Inference requests specify a year and round; if that race has already been run the features are read directly from the prepared frame, otherwise they are extrapolated from the most recent known race.

The server is packaged as a Docker image. Because MLflow and LakeFS run locally via Docker Compose, `host.docker.internal` is used to reach them from inside the container.

### Automated Pipeline
A Prefect flow runs on a weekly cron schedule and orchestrates the full post-race loop:

1. Ingest the latest race result from the Jolpica API into LakeFS via a staging branch
2. Validate and merge to main
3. Compute feature drift between the champion model's training distribution and the last 12 races using Evidently
4. Retrain if drift is detected or if a configurable race count threshold has been reached since the last training run
5. Promote the new model to the champion alias unconditionally — walk-forward cross-validation is the quality gate, not a post-hoc metric comparison

Hyperparameters and retraining config are stored as Prefect Variables and seeded automatically at `docker compose up`.

## Results

| Stage | Mean Val ROC-AUC | Agg Val ROC-AUC | Mean Val Brier Score | Notes |
|-------|-----------------|-----------------|----------------------|-------|
| Rolling podium rate heuristic | - | 0.79 | 0.12 | Pre-weekend baseline |
| LightGBM baseline | 0.81 | - | 0.12 | Minimal features, 1990–2024 |
| LightGBM full feature set | 0.876 | 0.883 | 0.088 | Full feature engineering, walk-forward CV |

## Project Structure
```
├── ingest/                         # Data ingestion (runs independently of training)
│   ├── bootstrap.py                # Initial load of raw CSVs into LakeFS
│   ├── update.py                   # Incremental updates via Jolpica API with staging branch strategy
│   └── settings.py                 # LakeFS connection settings
├── src/
│   └── f1_predictor/
│       ├── common/
│       │   └── config.py           # Shared settings (MLflow URI, LakeFS, hyperparameters)
│       ├── data/
│       │   ├── load.py             # Reads CSVs from LakeFS via io.BytesIO
│       │   ├── merge.py            # Joins raw tables into a single race frame
│       │   ├── clean.py            # Year filtering, target variable, column cleanup
│       │   └── validate.py         # Pandera schema validation
│       ├── features/
│       │   ├── driver.py           # Driver rolling rates, age, experience, circuit rate
│       │   ├── constructor.py      # Constructor rolling rates and DNF rates
│       │   ├── context.py          # Championship position, regulation era, circuit type, home race
│       │   └── features.py         # MODEL_FEATURES constant — single source of truth for feature list
│       ├── models/
│       │   ├── train.py            # Walk-forward training loop with MLflow logging
│       │   ├── eval.py             # Evaluation utilities and test set scoring
│       │   ├── export.py           # ONNX conversion and MLflow artifact logging
│       │   ├── fold.py             # Rolling window fold generation
│       │   └── types.py            # Shared types and constants
│       ├── pipelines/
│       │   ├── train_pipeline.py   # Orchestrates load → clean → validate → engineer → train
│       │   └── prepare.py          # Shared data preparation used by training and serving
│       ├── flows/
│       │   └── pipeline_flow.py    # Prefect flow: ingest → drift check → conditional retrain
│       └── serve/
│           ├── api.py              # FastAPI app with lifespan startup
│           ├── startup.py          # Data preparation and model loading at startup
│           ├── clients.py          # LakeFS and MLflow client wrappers
│           ├── prepare.py          # Inference request handling and feature extrapolation
│           ├── inference.py        # ONNX inference session wrapper
│           ├── log.py              # Logging configuration and request ID middleware
│           ├── templates_env.py    # Jinja2 template configuration
│           ├── templates/
│           │   └── home.html       # Prediction UI
│           └── routes/
│               ├── health.py       # GET /health
│               ├── home.py         # GET / (redirect)
│               └── predict.py      # GET|POST /predict
├── notebooks/                      # Exploratory and iterative work
├── Dockerfile.serve                # Docker image for the inference server
├── Dockerfile.worker               # Docker image for the Prefect worker
├── Dockerfile.evidently            # Docker image for the Evidently UI
├── docker-compose.yml              # Full local stack: MLflow, LakeFS, Prefect, Evidently
├── .env                            # Local config for running outside Docker (not committed)
├── .env.docker                     # Config for services running inside Docker Compose (not committed)
├── .env.example                    # Template for both env files
└── pyproject.toml
```

## Infrastructure

The full stack runs via Docker Compose. All services persist data to named volumes so state survives restarts. On first start, `lakefs-setup` and `prefect-setup` containers run automatically to initialise LakeFS and seed Prefect Variables.

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| MLflow tracking UI | http://localhost:5000 |
| LakeFS UI | http://localhost:8000 |
| Prefect UI | http://localhost:4200 |
| Evidently UI | http://localhost:8001 |

## Setup

Install dependencies (only needed for local development outside Docker):

```bash
pip install -e ".[train,data,dev]"
```

Copy `.env.example` to `.env` (for local development) and to `.env.docker` (for the Docker Compose stack). Fill in your LakeFS credentials, MLflow URI, and Prefect Variable values in both.

Download the raw CSVs from [Kaggle](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) and place them in `./data/`.

Bring up the full stack:

```bash
docker compose up
```

On first start:
- `lakefs-setup` initialises the LakeFS instance automatically
- `bootstrap` uploads the raw CSVs from `./data/` into LakeFS
- `prefect-setup` seeds the `lgbm_hyperparameters` and `retraining_config` Prefect Variables

Once the stack is running, trigger the first flow run manually from the Prefect UI at http://localhost:4200. The flow detects that no champion model exists and runs the initial training. From then on the cron schedule takes over.

## Serving with Docker

The inference server can also be run standalone outside of the Compose stack:

```bash
docker build --file Dockerfile.serve --tag f1-podium-predictor:latest .
```

```bash
docker run --name f1-podium-predictor -d -p 8080:1234 \
  -e LAKEFS_HOST=http://host.docker.internal:8000 \
  -e LAKEFS_INSTALLATION_ACCESS_KEY_ID=[your key] \
  -e LAKEFS_INSTALLATION_SECRET_ACCESS_KEY=[your secret] \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_EXPERIMENT_NAME=f1-podium-predictor \
  f1-podium-predictor:latest
```

The prediction UI is then available at http://localhost:8080/predict and the API at http://localhost:8080/docs.