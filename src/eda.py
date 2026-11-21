"""
Step 2: End-to-end Exploratory Data Analysis (EDA).

Reads the cleaned data from Step 1, prints key tables, and saves
charts to outputs/figures/ and tables to outputs/results/.

Run on its own:
    python src/eda.py
or from main.py:
    from src.eda import run_eda
"""

import os

import matplotlib

matplotlib.use("Agg")  # save figures to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROCESSED_PATH = "data/processed/cleaned_data.csv"
FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"

MONTH_ORDER = ["April", "May", "June"]
GROUP_COLS = ["month", "campaign_number", "user_engagement", "banner", "placement"]

sns.set_theme(style="whitegrid")


def _save_fig(name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()
    print(f"  saved figure: {FIG_DIR}/{name}")


# --------------------------------------------------------------------------
# 1. Overall KPIs
# --------------------------------------------------------------------------
def overall_kpis(df: pd.DataFrame) -> pd.Series:
    """Totals and ratios for the whole dataset."""
    t = df[["displays", "clicks", "cost", "revenue",
            "post_click_conversions", "post_click_sales_amount"]].sum()
    kpis = pd.Series({
        "total_displays": t["displays"],
        "total_clicks": t["clicks"],
        "total_cost": t["cost"],
        "total_revenue": t["revenue"],
        "total_profit": t["revenue"] - t["cost"],
        "total_conversions": t["post_click_conversions"],
        "total_sales_amount": t["post_click_sales_amount"],
        "overall_CTR": t["clicks"] / t["displays"],
        "overall_CPC": t["cost"] / t["clicks"],
        "overall_conversion_rate": t["post_click_conversions"] / t["clicks"],
        "overall_ROI": (t["revenue"] - t["cost"]) / t["cost"],
    })
    print("\n=== Overall KPIs ===")
    print(kpis.round(4).to_string())
    os.makedirs(RES_DIR, exist_ok=True)
    kpis.to_csv(os.path.join(RES_DIR, "overall_kpis.csv"), header=["value"])
    return kpis


# --------------------------------------------------------------------------
# 2. Performance by group (ratios computed from SUMS, not averages of ratios)
# --------------------------------------------------------------------------
def group_performance(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col).agg(
        displays=("displays", "sum"),
        clicks=("clicks", "sum"),
        cost=("cost", "sum"),
        revenue=("revenue", "sum"),
        conversions=("post_click_conversions", "sum"),
        sales_amount=("post_click_sales_amount", "sum"),
    )
    g["profit"] = g["revenue"] - g["cost"]
    g["ROI"] = g["profit"] / g["cost"]
    g["CTR"] = g["clicks"] / g["displays"]
    g["CPC"] = g["cost"] / g["clicks"]
    g["conversion_rate"] = g["conversions"] / g["clicks"]
    g["cost_share"] = g["cost"] / g["cost"].sum()
    g["revenue_share"] = g["revenue"] / g["revenue"].sum()
    return g.sort_values("ROI", ascending=False)


def plot_group(g: pd.DataFrame, col: str) -> None:
    """Two charts per dimension: ROI, and cost share vs revenue share."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    g["ROI"].plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"ROI by {col}")
    axes[0].set_ylabel("ROI = (revenue - cost) / cost")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=45)

    g[["cost_share", "revenue_share"]].plot(kind="bar", ax=axes[1],
                                            color=["#C44E52", "#55A868"])
    axes[1].set_title(f"Share of total cost vs revenue by {col}")
    axes[1].set_ylabel("share of total")
    axes[1].tick_params(axis="x", rotation=45)
    _save_fig(f"performance_by_{col}.png")


def performance_by_dimension(df: pd.DataFrame) -> dict:
    results = {}
    os.makedirs(RES_DIR, exist_ok=True)
    for col in GROUP_COLS:
        g = group_performance(df, col)
        if col == "month":
            g = g.reindex(MONTH_ORDER)
        results[col] = g
        print(f"\n=== Performance by {col} ===")
        print(g[["cost", "revenue", "profit", "ROI", "CTR", "CPC",
                 "conversion_rate", "cost_share", "revenue_share"]].round(4).to_string())
        g.to_csv(os.path.join(RES_DIR, f"performance_by_{col}.csv"))
        plot_group(g, col)
    return results


# --------------------------------------------------------------------------
# 3. Distributions (skewed data -> log scale)
# --------------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame) -> None:
    cols = ["displays", "clicks", "cost", "revenue"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(np.log1p(df[c]), bins=50, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution of {c} (log1p scale)")
        ax.set_xlabel(f"log(1 + {c})")
    _save_fig("distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, c in zip(axes, ["cost", "revenue", "clicks"]):
        sns.boxplot(data=df, x="user_engagement", y=np.log1p(df[c]),
                    order=["Low", "Medium", "High"], ax=ax)
        ax.set_title(f"{c} by engagement (log1p)")
        ax.set_ylabel(f"log(1 + {c})")
    _save_fig("boxplots_by_engagement.png")


# --------------------------------------------------------------------------
# 4. Correlations
# --------------------------------------------------------------------------
def plot_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["displays", "clicks", "cost", "revenue", "post_click_conversions",
            "post_click_sales_amount", "CTR", "CPC", "CPM", "conversion_rate"]
    corr = df[cols].corr(method="spearman")  # robust to skew
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Spearman correlation of key metrics")
    _save_fig("correlation_heatmap.png")
    corr.to_csv(os.path.join(RES_DIR, "correlations.csv"))
    print("\n=== Spearman correlation with revenue ===")
    print(corr["revenue"].drop("revenue").sort_values(ascending=False).round(3).to_string())
    return corr


# --------------------------------------------------------------------------
# 5. Trend over time
# --------------------------------------------------------------------------
def plot_daily_trend(df: pd.DataFrame) -> None:
    d = df.groupby(["month_num", "day"], as_index=False)[["cost", "revenue"]].sum()
    d["order"] = np.arange(len(d))
    d["ROI"] = (d["revenue"] - d["cost"]) / d["cost"]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(d["order"], d["cost"], label="cost", color="#C44E52")
    axes[0].plot(d["order"], d["revenue"], label="revenue", color="#55A868")
    axes[0].set_title("Daily cost vs revenue (April 1 to June 30)")
    axes[0].legend()
    axes[1].plot(d["order"], d["ROI"], color="#4C72B0")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Daily ROI")
    axes[1].set_xlabel("day index in the data")
    _save_fig("daily_trend.png")


# --------------------------------------------------------------------------
# 6. Data quality checks that matter for the analysis
# --------------------------------------------------------------------------
def quality_checks(df: pd.DataFrame) -> None:
    print("\n=== Data quality checks ===")
    print("rows with CTR > 1 (clicks > displays)       :", int((df["CTR"] > 1).sum()))
    print("rows with conversion_rate > 1               :", int((df["conversion_rate"] > 1).sum()))
    print("rows with cost > 0 but revenue = 0          :", int(((df["cost"] > 0) & (df["revenue"] == 0)).sum()))
    print("share of rows with zero clicks              :", round((df["clicks"] == 0).mean(), 3))
    loss = df[df["profit"] < 0]
    print("share of rows that lose money               :", round(len(loss) / len(df), 3))
    print("money lost in those rows / total cost       :",
          round(-loss["profit"].sum() / df["cost"].sum(), 3))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame = None, path: str = PROCESSED_PATH) -> None:
    if df is None:
        df = pd.read_csv(path)
    overall_kpis(df)
    performance_by_dimension(df)
    plot_distributions(df)
    plot_correlations(df)
    plot_daily_trend(df)
    quality_checks(df)
    print("\nEDA finished. Figures in outputs/figures, tables in outputs/results.")


if __name__ == "__main__":
    run_eda()