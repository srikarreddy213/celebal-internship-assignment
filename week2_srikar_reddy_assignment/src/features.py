"""Feature engineering for quarterly delivery forecasting."""

import pandas as pd

from src.preprocessing import PRODUCTION_COL, TARGET_COL

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_mean_4",
    "rolling_std_4",
    "pct_change",
    "qoq_growth",
    "production_lag_1",
    "production_lag_2",
    "production_lag_4",
    "production_rolling_mean_4",
    "production_rolling_std_4",
    "year",
    "quarter",
]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag features use shift() so the current target is never included."""
    out = df.copy()
    out["lag_1"] = out[TARGET_COL].shift(1)
    out["lag_2"] = out[TARGET_COL].shift(2)
    out["lag_4"] = out[TARGET_COL].shift(4)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling stats on past values only.

    shift(1) before rolling prevents the current quarter from leaking
    into the feature.
    """
    out = df.copy()
    past = out[TARGET_COL].shift(1)
    out["rolling_mean_4"] = past.rolling(window=4, min_periods=4).mean()
    out["rolling_std_4"] = past.rolling(window=4, min_periods=4).std()
    return out


def add_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Growth features based on past delivery counts."""
    out = df.copy()
    past = out[TARGET_COL].shift(1)
    out["pct_change"] = past.pct_change()
    out["qoq_growth"] = past.pct_change(periods=1)
    return out


def add_production_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features from past production values."""
    out = df.copy()
    out["production_lag_1"] = out[PRODUCTION_COL].shift(1)
    out["production_lag_2"] = out[PRODUCTION_COL].shift(2)
    out["production_lag_4"] = out[PRODUCTION_COL].shift(4)

    past_production = out[PRODUCTION_COL].shift(1)
    out["production_rolling_mean_4"] = past_production.rolling(
        window=4, min_periods=4
    ).mean()
    out["production_rolling_std_4"] = past_production.rolling(
        window=4, min_periods=4
    ).std()
    return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features for trend and seasonality."""
    out = df.copy()
    out["year"] = out["year"].astype(int)
    out["quarter"] = out["quarter"].astype(int)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature steps and drop rows without enough history."""
    featured = add_lag_features(df)
    featured = add_rolling_features(featured)
    featured = add_growth_features(featured)
    featured = add_production_features(featured)
    featured = add_date_features(featured)

    featured = featured.dropna(subset=FEATURE_COLUMNS + [TARGET_COL])
    return featured.reset_index(drop=True)
