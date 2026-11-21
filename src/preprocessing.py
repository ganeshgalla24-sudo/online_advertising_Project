"""
Step 1: Preprocessing, feature selection and feature engineering.

Run on its own:
    python src/preprocessing.py
or from main.py:
    from src.preprocessing import run_preprocessing
"""

import os

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/online_advertising_performance_data.csv"
PROCESSED_PATH = "data/processed/cleaned_data.csv"

MONTH_MAP = {"April": 4, "May": 5, "June": 6}
CATEGORICAL_COLS = ["campaign_number", "user_engagement", "placement", "banner"]


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV."""
    df = pd.read_csv(path)
    print(f"[load] raw shape: {df.shape}")
    return df


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty columns, remove duplicates, fix missing values."""
    df = df.copy()

    # Drop columns that are completely empty (e.g. 'Unnamed: 12', 'Unnamed: 13')
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)
    print(f"[clean] dropped empty columns: {empty_cols}")

    # Tidy column names and text values
    df.columns = df.columns.str.strip()
    for col in ["month", "campaign_number", "user_engagement", "banner", "placement"]:
        df[col] = df[col].astype("string").str.strip()

    # Remove exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] removed {n_before - len(df)} duplicate rows")

    # Missing placement -> keep the row, label it 'Unknown'
    n_missing = int(df["placement"].isna().sum())
    df["placement"] = df["placement"].fillna("Unknown")
    print(f"[clean] filled {n_missing} missing placement values with 'Unknown'")

    # Sanity check: no negative numbers in the numeric columns
    num_cols = df.select_dtypes(include="number").columns
    negatives = int((df[num_cols] < 0).sum().sum())
    print(f"[clean] negative values found: {negatives}")

    return df


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning 0 where the denominator is 0 (instead of inf/NaN)."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business metrics and helper columns."""
    df = df.copy()

    # Efficiency metrics
    df["CTR"] = _safe_divide(df["clicks"], df["displays"])                       # click-through rate
    df["CPC"] = _safe_divide(df["cost"], df["clicks"])                           # cost per click
    df["CPM"] = _safe_divide(df["cost"], df["displays"]) * 1000                  # cost per 1000 displays
    df["conversion_rate"] = _safe_divide(df["post_click_conversions"], df["clicks"])
    df["avg_order_value"] = _safe_divide(df["post_click_sales_amount"], df["post_click_conversions"])

    # Profitability metrics
    df["profit"] = df["revenue"] - df["cost"]
    df["ROI"] = _safe_divide(df["profit"], df["cost"])                           # return on spend

    # Time feature (data has month name + day only, no year)
    df["month_num"] = df["month"].map(MONTH_MAP)

    # Banner size features, e.g. '160 x 600' -> width=160, height=600
    size = df["banner"].str.split(" x ", expand=True).astype(int)
    df["banner_width"] = size[0]
    df["banner_height"] = size[1]
    df["banner_area"] = df["banner_width"] * df["banner_height"]

    print(f"[features] new columns: CTR, CPC, CPM, conversion_rate, avg_order_value, "
          f"profit, ROI, month_num, banner_width, banner_height, banner_area")
    return df


# --------------------------------------------------------------------------
# 4. Encoding (used later by the ML model in Step 4)
# --------------------------------------------------------------------------
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns. Not saved; called by model.py."""
    cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return pd.get_dummies(df, columns=cols, drop_first=True)


# --------------------------------------------------------------------------
# 5. Summary + runner
# --------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> None:
    """Print a quick before-you-continue check."""
    print("\n[summary] shape:", df.shape)
    print("[summary] missing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_string() or "  none")
    print("[summary] rows with zero clicks:", int((df["clicks"] == 0).sum()))
    print("[summary] rows with zero cost  :", int((df["cost"] == 0).sum()))
    print("[summary] placements:", sorted(df["placement"].unique()))


def run_preprocessing(raw_path: str = RAW_PATH, save_path: str = PROCESSED_PATH) -> pd.DataFrame:
    """Full Step 1 pipeline. Returns the cleaned + engineered DataFrame."""
    df = load_data(raw_path)
    df = clean_data(df)
    df = engineer_features(df)
    summarize(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\n[save] cleaned data written to {save_path}")
    return df


if __name__ == "__main__":
    run_preprocessing()