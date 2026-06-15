# Tesla Deliveries Forecasting

**Week 2 Internship Assignment**

This project forecasts quarterly Tesla deliveries using historical delivery and production data. The objective is to apply core Week-2 machine learning concepts including feature engineering, regression modeling, hyperparameter tuning, and time-series forecasting.

---

## Dataset Overview

The dataset used is:

`data/tesla_deliveries_dataset_2015_2025.csv`

The raw dataset contains monthly Tesla delivery and production information segmented by region and vehicle model.

### Key Columns

* `Year`, `Month` – Time information
* `Estimated_Deliveries` – Target variable
* `Production_Units` – Production data used as additional predictors

### Data Summary

* Raw rows: **2,640**
* Missing values: **0**
* Duplicate rows: **0**

For forecasting purposes, the monthly data was aggregated into **quarterly company-level totals**, resulting in **44 quarterly observations (2015 Q1 – 2025 Q4)**.

The most recent **4 quarters** were reserved as the test set to preserve chronological order and avoid data leakage.

---

## Methodology

### Data Preprocessing

* Loaded and validated raw data
* Removed duplicates
* Aggregated monthly records into quarterly totals
* Created a chronological time index

### Feature Engineering

#### Delivery-Based Features

* `lag_1`
* `lag_2`
* `lag_4`
* `rolling_mean_4`
* `rolling_std_4`
* `pct_change`
* `qoq_growth`

#### Production-Based Features

* `production_lag_1`
* `production_lag_2`
* `production_lag_4`
* `production_rolling_mean_4`
* `production_rolling_std_4`

#### Calendar Features

* `year`
* `quarter`

All lag and rolling features were created using only past information to prevent data leakage.

---

## Models Used

The following regression models were trained and evaluated:

1. Linear Regression
2. Ridge Regression (L2 Regularization)
3. Lasso Regression (L1 Regularization)

### Hyperparameter Tuning

Ridge and Lasso models were tuned using:

* GridSearchCV
* TimeSeriesSplit cross-validation

---

## Model Performance

Test Set: **2025 Q1 – 2025 Q4**

| Model             |   RMSE |    MAE |     R² |  MAPE | Forecast Accuracy |
| ----------------- | -----: | -----: | -----: | ----: | ----------------: |
| Ridge Regression  | 11,168 | 10,466 | -0.151 | 1.77% |            98.23% |
| Lasso Regression  | 14,365 | 10,869 | -0.905 | 1.86% |            98.14% |
| Linear Regression | 33,682 | 31,814 | -9.472 | 5.40% |            94.60% |

### Best Model

**Ridge Regression**

Best alpha selected through tuning:

`alpha = 1000`

Although the test-set R² is negative, the model achieved a low MAPE (~1.8%), indicating relatively small forecasting errors compared to total quarterly deliveries.

---

## Forecast Results

### Predicted Deliveries

| Quarter | Forecast |
| ------- | -------: |
| 2026-Q1 |  596,067 |
| 2026-Q2 |  596,240 |
| 2026-Q3 |  596,487 |
| 2026-Q4 |  596,819 |

The model suggests relatively stable Tesla deliveries over the next four quarters.

---

## Key Findings

* Tesla deliveries grew strongly during earlier years and became relatively stable in recent periods.
* Production and deliveries are highly related, making production-based lag features useful predictors.
* Ridge Regression outperformed Linear Regression and Lasso Regression.
* The largest prediction error occurred in **2025 Q2**, where deliveries deviated from recent historical patterns.
* Forecasts indicate stable delivery volumes rather than rapid growth or decline.

---

## Project Structure

```text
week2_srikar_reddy_assignment/

├── data/
│   └── tesla_deliveries_dataset_2015_2025.csv
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── models.py
│
├── actual_vs_predicted.png
├── future_forecast.png
├── model_results.csv
├── test_predictions.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete pipeline:

```bash
python main.py
```

The script will:

* Load and preprocess data
* Create forecasting features
* Train and evaluate models
* Generate forecasts
* Save plots and result files

---

## Outputs Generated

* `actual_vs_predicted.png`
* `future_forecast.png`
* `model_results.csv`
* `test_predictions.csv`
