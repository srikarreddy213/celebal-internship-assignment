"""Load and clean Tesla delivery data."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tesla_deliveries_dataset_2015_2025.csv"

TARGET_COL = "Estimated_Deliveries"
PRODUCTION_COL = "Production_Units"
DATE_COL = "date"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Place tesla_deliveries_dataset_2015_2025.csv in the data/ folder."
        )
    return pd.read_csv(path)


def aggregate_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly Region x Model rows to company-level quarterly totals.

    Raw data has 20 rows per month (4 regions x 5 models).
    Forecasting uses one row per quarter.
    """
    data = df.copy()
    data["quarter"] = ((data["Month"] - 1) // 3) + 1

    quarterly = (
        data.groupby(["Year", "quarter"], as_index=False)
        .agg({TARGET_COL: "sum", PRODUCTION_COL: "sum"})
        .rename(columns={"Year": "year"})
    )

    start_month = (quarterly["quarter"] - 1) * 3 + 1
    quarterly[DATE_COL] = pd.to_datetime(
        {"year": quarterly["year"], "month": start_month, "day": 1}
    )

    return quarterly.sort_values(DATE_COL).reset_index(drop=True)


def preprocess(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load, remove duplicates, and return sorted quarterly data.
    """
    raw = load_data(path)
    raw = raw.drop_duplicates()

    quarterly = aggregate_to_quarterly(raw)

    if quarterly[TARGET_COL].isna().any():
        raise ValueError("Missing values found in target column after cleaning.")

    if quarterly.duplicated(subset=[DATE_COL]).any():
        raise ValueError("Duplicate quarterly dates found.")

    return quarterly
