# F1 Podium Predictor

A machine learning project to predict the probability of a Formula 1 driver finishing on the podium (top 3) for a given race.

## Project Overview

Using historical race data from 1950 to present, we train a classification model to predict podium finishes. The model outputs a probability for each driver in a given race, rather than a binary yes/no prediction.

## Objectives

- Predict the probability of a podium finish for each driver in a race
- Beat a per-position qualifying heuristic baseline (AUC > 0.93, Brier Score < 0.065)
- Validate against 2025 race results using live data from the Jolpica API

## Data Sources
- **Kaggle — Formula 1 Race Data** by jtrotman: [historical race results, qualifying, constructors (1950–2024)](https://www.kaggle.com/datasets/jtrotman/formula-1-race-data) 
- **Jolpica API**: live 2025 race results for model validation

## Methodology

1. Establish heuristic baselines from qualifying position data
2. Train a baseline model on raw features
3. Iteratively engineer features and measure improvement
4. Validate final model against 2025 season results

## Success Criteria

Defined upfront to avoid p-hacking:

| Metric | Baseline | Target |
|--------|----------|--------|
| ROC-AUC | 0.93 | > 0.93 |
| Brier Score | 0.065 | < 0.065 |

## Project Structure

```
├── data/
│   └── raw/
│       ├── races.csv
│       ├── qualifying.csv
│       ├── results.csv
│       └── constructors.csv
├── F1Predictor.ipynb
└── README.md
```

## Setup

```bash
pip install pandas scikit-learn lightgbm matplotlib seaborn requests
```

## Results

*To be updated as the project progresses.*

