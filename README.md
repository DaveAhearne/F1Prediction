# F1 Podium Predictor
A machine learning project to predict the probability of a Formula 1 driver finishing on the podium (top 3) for a given race.

## Project Overview
Using historical race data from 1950 to present, we train a classification model to predict podium finishes. The model outputs a probability for each driver in a given race, rather than a binary yes/no prediction.

## Objectives
- Predict the probability of a podium finish for each driver in a race
- Beat a pre-weekend rolling podium rate heuristic baseline (AUC > 0.79, Brier Score < 0.12)
- Validate against 2025 race results using live data from the Jolpica API

## Data Sources
- **Kaggle — Formula 1 Race Data** by jtrotman: [historical race results, qualifying, constructors (1950–2024)](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data)
- **Jolpica API**: live 2025 race results for model validation

## Methodology
1. Establish heuristic baselines from qualifying position data and rolling driver form
2. Train a baseline model on raw features
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

## Project Structure
```
├── data/
│   └── raw/
│       ├── races.csv
│       ├── qualifying.csv
│       ├── results.csv
│       ├── drivers.csv
│       └── constructors.csv
├── notebooks/
│   ├── EDA.ipynb
│   └── Heuristic_Baseline.ipynb
└── README.md
```

## Setup
```bash
pip install pandas scikit-learn lightgbm matplotlib seaborn requests
```

## Results
| Stage | ROC-AUC | Brier Score | Notes |
|-------|---------|-------------|-------|
| Rolling podium rate heuristic | 0.79 | 0.12 | Pre-weekend baseline, evaluated on 2025 |