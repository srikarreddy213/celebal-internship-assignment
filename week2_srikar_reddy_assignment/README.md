# Tesla Deliveries Forecasting

Week 2 assignment — forecasting quarterly Tesla deliveries using lag features and linear regression models (Linear, Ridge, Lasso).

## About the Dataset

The raw file is `data/tesla_deliveries_dataset_2015_2025.csv`.

It has monthly data broken down by region and model (4 regions, 5 models → 20 rows per month). Main columns I used:

- `Year`, `Month` — time
- `Estimated_Deliveries` — target variable
- `Production_Units` — used as extra features

Raw data: 2,640 rows, no missing values, no duplicates.

For modeling, I aggregated everything to quarterly company totals (44 quarters from 2015 to 2025). The last 4 quarters are held out as the test set — no random split, since this is time series data.

## What I Did

1. Cleaned and aggregated monthly data to quarterly level
2. Built lag, rolling, and growth features from past deliveries
3. Added production-based features (`production_lag_1`, rolling stats, etc.) — all shifted so there's no data leakage
4. Used `StandardScaler` + model inside sklearn `Pipeline`
5. Trained Linear, Ridge, and Lasso regression
6. Tuned Ridge/Lasso alpha with `GridSearchCV` and `TimeSeriesSplit`
7. Evaluated on the last 4 quarters and forecasted the next 4

## Features

**From deliveries:** `lag_1`, `lag_2`, `lag_4`, `rolling_mean_4`, `rolling_std_4`, `pct_change`, `qoq_growth`

**From production:** `production_lag_1`, `production_lag_2`, `production_lag_4`, `production_rolling_mean_4`, `production_rolling_std_4`

**Calendar:** `year`, `quarter`

## Results

Test set = last 4 quarters (2025 Q1–Q4).

| Model | RMSE | MAE | R² | MAPE | Forecast Accuracy |
|-------|------|-----|-----|------|-------------------|
| Ridge Regression | 11,168 | 10,466 | -0.151 | 1.77% | 98.23% |
| Lasso Regression | 14,365 | 10,869 | -0.905 | 1.86% | 98.14% |
| Linear Regression | 33,682 | 31,814 | -9.472 | 5.40% | 94.60% |

Ridge worked best. Both Ridge and Lasso picked alpha = 1000 during tuning.

R² is negative on the test set because deliveries have been very flat recently — the model is only slightly off in absolute terms (MAPE ~1.8%), but the series doesn't vary much so even small errors hurt R².

Detailed outputs: `model_results.csv`, `test_predictions.csv`

## Forecast (next 4 quarters)

| Quarter | Forecast |
|---------|----------|
| 2026-Q1 | 596,067 |
| 2026-Q2 | 596,240 |
| 2026-Q3 | 596,487 |
| 2026-Q4 | 596,819 |

Charts saved as `actual_vs_predicted.png` and `future_forecast.png`.

## Takeaways

- Deliveries grew fast until around 2020, then flattened to roughly 590k–610k per quarter
- Production and deliveries track each other closely, so production lags helped
- Ridge beat plain Linear Regression — the lag features are correlated and Ridge handles that better
- Biggest prediction miss was 2025-Q2 — deliveries dropped more than recent patterns suggested
- Forecast points to stable deliveries going forward, no big jump expected

## Project Layout

```
week2_srikar_reddy_assignment/
├── data/
│   └── tesla_deliveries_dataset_2015_2025.csv
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
├── main.py
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

For EDA and plots step by step, open `notebooks/01_eda.ipynb`.
