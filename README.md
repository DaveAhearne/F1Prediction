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
│   ├── update.py                   # Incremental data updates
│   └── settings.py                 # LakeFS connection settings
├── src/
│   └── f1_predictor/
│       ├── common/
│       │   └── config.py           # Shared settings (MLflow URI, LakeFS, hyperparameters)
│       ├── data/
│       │   ├── load.py             # Reads CSVs from LakeFS, casts dtypes
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
│       │   ├── evaluate.py         # Evaluation utilities
│       │   └── register.py         # Model registry promotion
│       ├── pipelines/
│       │   └── train_pipeline.py   # Orchestrates load → clean → validate → engineer → train
│       └── serve/
│           ├── api.py              # FastAPI serving endpoint (planned)
│           └── routes/
│               └── health.py
├── notebooks/                      # Exploratory and iterative work
├── docker-compose.yml              # MLflow and LakeFS services
├── .env                            # Local config and hyperparameters (not committed)
└── pyproject.toml
```

## Infrastructure

The project uses Docker Compose to run MLflow and LakeFS locally. Both services persist data to named volumes so runs survive restarts.

```bash
docker compose up
```

| Service | URL |
|---------|-----|
| MLflow tracking UI | http://localhost:5000 |
| LakeFS UI | http://localhost:8000 |

## Setup

Install dependencies:

```bash
pip install -e ".[train,data,dev]"
```

Copy `.env.example` to `.env` and fill in your LakeFS credentials and MLflow URI.

Bootstrap data into LakeFS (first run only):

```bash
f1_bootstrap
```

Run the training pipeline:

```bash
f1_train
```