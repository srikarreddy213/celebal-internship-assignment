"""Train, tune, evaluate, and forecast with linear models."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS
from src.preprocessing import DATE_COL, PRODUCTION_COL, TARGET_COL

TEST_SIZE = 4
FORECAST_HORIZON = 4
CV_SPLITS = 5
RANDOM_STATE = 42


def chronological_split(
    df: pd.DataFrame, test_size: int = TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train on past quarters, test on the most recent held-out quarters."""
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test


def build_pipelines() -> Dict[str, Pipeline]:
    """
    StandardScaler + model in a Pipeline.

    Ridge and Lasso need scaling because regularization penalizes
    coefficients — unscaled features would be penalized unevenly.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(random_state=RANDOM_STATE, max_iter=100000, tol=1e-3)),
        ]),
    }


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Return RMSE, MAE, R², MAPE, and Forecast Accuracy."""
    mape_value = mape(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape_value,
        "Forecast_Accuracy": 100.0 - mape_value,
    }


def train_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Train Linear, Ridge, and Lasso with TimeSeriesSplit tuning."""
    train_df, test_df = chronological_split(df)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    alphas = [100, 1000]
    param_grid = {"model__alpha": alphas}

    fitted: Dict[str, Any] = {}
    best_alphas: Dict[str, float] = {}

    for name, pipeline in build_pipelines().items():
        if name == "Linear Regression":
            pipeline.fit(x_train, y_train)
            fitted[name] = pipeline
        else:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            search.fit(x_train, y_train)
            fitted[name] = search.best_estimator_
            best_alphas[name] = search.best_params_["model__alpha"]

    return {
        "models": fitted,
        "x_test": x_test,
        "y_test": y_test,
        "test_df": test_df,
        "train_df": train_df,
        "best_alphas": best_alphas,
    }


def compare_models(results: Dict[str, Any]) -> pd.DataFrame:
    """Build a model comparison table sorted by RMSE."""
    rows = []
    for name, model in results["models"].items():
        preds = model.predict(results["x_test"])
        metrics = evaluate_model(results["y_test"], preds)
        rows.append({"Model": name, **metrics})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def save_model_results(comparison: pd.DataFrame, output_path: Path) -> None:
    """Save model comparison metrics to CSV."""
    comparison.to_csv(output_path, index=False)


def build_test_predictions(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Build per-quarter test set predictions for error analysis."""
    quarters = test_df.apply(
        lambda row: f"{int(row['year'])}-Q{int(row['quarter'])}", axis=1
    )
    errors = y_true.values - y_pred
    return pd.DataFrame({
        "Quarter": quarters.values,
        "Actual": y_true.values,
        "Predicted": y_pred,
        "Error": errors,
        "Absolute_Error": np.abs(errors),
    })


def analyze_errors(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize prediction errors on the test set.

    Returns the predictions table with analysis metadata.
    """
    worst = predictions.loc[predictions["Absolute_Error"].idxmax()]
    avg_error = float(predictions["Absolute_Error"].mean())

    summary = (
        f"Largest prediction error occurred in {worst['Quarter']}.\n"
        f"  Actual: {worst['Actual']:,.0f} | Predicted: {worst['Predicted']:,.0f} | "
        f"Error: {worst['Error']:,.0f}\n"
        f"Average absolute error: {avg_error:,.0f}\n"
        "Possible reason: quarterly deliveries changed more abruptly than "
        "historical lag and rolling patterns suggested."
    )

    return {
        "predictions": predictions,
        "worst_quarter": worst["Quarter"],
        "avg_absolute_error": avg_error,
        "summary": summary,
    }


def describe_forecast_trend(forecast_df: pd.DataFrame) -> str:
    """Describe whether the forecast is stable, rising, or falling."""
    values = forecast_df["forecast"]
    change = float(values.iloc[-1] - values.iloc[0])
    if abs(change) < 5_000:
        return "relatively stable Tesla deliveries over the next four quarters"
    if change > 0:
        return "a slight increase in Tesla deliveries over the next four quarters"
    return "a slight decrease in Tesla deliveries over the next four quarters"


def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Bar chart comparing actual and predicted deliveries on the test set."""
    labels = test_df.apply(
        lambda row: f"{int(row['year'])}-Q{int(row['quarter'])}", axis=1
    )
    x_pos = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x_pos - width / 2, y_true, width, label="Actual Deliveries")
    ax.bar(x_pos + width / 2, y_pred, width, label="Predicted Deliveries")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title(f"Actual vs Predicted — {model_name}")
    ax.set_ylabel("Estimated Deliveries")
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_future_forecast(
    history: pd.DataFrame,
    forecast_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Line plot of historical deliveries and the next-quarter forecast."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        history[DATE_COL],
        history[TARGET_COL],
        marker="o",
        label="Historical Deliveries",
    )
    ax.plot(
        forecast_df[DATE_COL],
        forecast_df["forecast"],
        marker="s",
        linestyle="--",
        label="Forecasted Deliveries",
    )
    ax.set_title("Historical and Forecasted Deliveries")
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated Deliveries")
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _next_quarter(year: int, quarter: int) -> Tuple[int, int]:
    if quarter == 4:
        return year + 1, 1
    return year, quarter + 1


def _feature_row(history: pd.DataFrame) -> pd.Series:
    """Build one leakage-safe feature row from delivery and production history."""
    deliveries = history[TARGET_COL]
    production = history[PRODUCTION_COL].ffill()

    lag_1, lag_2, lag_4 = deliveries.iloc[-1], deliveries.iloc[-2], deliveries.iloc[-4]
    window = deliveries.iloc[-4:]
    rolling_std = window.std()
    if pd.isna(rolling_std):
        rolling_std = 0.0

    prod_lag_1 = production.iloc[-1]
    prod_lag_2 = production.iloc[-2]
    prod_lag_4 = production.iloc[-4]
    prod_window = production.iloc[-4:]
    prod_rolling_std = prod_window.std()
    if pd.isna(prod_rolling_std):
        prod_rolling_std = 0.0

    pct = (lag_1 - lag_2) / lag_2 if lag_2 else 0.0
    year, quarter = _next_quarter(
        int(history.iloc[-1]["year"]), int(history.iloc[-1]["quarter"])
    )

    return pd.Series({
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_4": lag_4,
        "rolling_mean_4": window.mean(),
        "rolling_std_4": rolling_std,
        "pct_change": pct,
        "qoq_growth": pct,
        "production_lag_1": prod_lag_1,
        "production_lag_2": prod_lag_2,
        "production_lag_4": prod_lag_4,
        "production_rolling_mean_4": prod_window.mean(),
        "production_rolling_std_4": prod_rolling_std,
        "year": year,
        "quarter": quarter,
    })


def forecast_quarters(
    history: pd.DataFrame,
    model: Any,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """Recursive multi-step forecast for the next N quarters."""
    working = history.copy().reset_index(drop=True)
    rows: List[Dict[str, Any]] = []

    for _ in range(horizon):
        features = _feature_row(working)
        x = pd.DataFrame([features[FEATURE_COLUMNS].values], columns=FEATURE_COLUMNS)
        prediction = float(model.predict(x)[0])

        year, quarter = int(features["year"]), int(features["quarter"])
        row = {
            DATE_COL: pd.Timestamp(year=year, month=(quarter - 1) * 3 + 1, day=1),
            "year": year,
            "quarter": quarter,
            TARGET_COL: prediction,
            PRODUCTION_COL: prediction,
            "forecast": prediction,
        }
        rows.append(row)
        working = pd.concat([working, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(rows)
