"""
Step 3: Descriptive and inferential statistics.

Descriptive: what do the numbers look like? (centre, spread, shape)
Inferential: are the differences we saw in EDA real, or just noise?

Run on its own:
    python src/stats_analysis.py
or from main.py:
    from src.stats_analysis import run_statistics
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

PROCESSED_PATH = "data/processed/cleaned_data.csv"
RES_DIR = "outputs/results"

ALPHA = 0.05
MIN_COST = 1.0          # rows with tiny cost give wild row-level ROI, so tests use cost >= 1
N_BOOT = 1000
RNG = np.random.default_rng(42)

NUM_COLS = ["displays", "clicks", "cost", "revenue", "post_click_conversions",
            "post_click_sales_amount", "CTR", "CPC", "conversion_rate", "ROI"]


def _save(df: pd.DataFrame, name: str) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name))


# --------------------------------------------------------------------------
# A. DESCRIPTIVE STATISTICS
# --------------------------------------------------------------------------
def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, median, spread and shape of each key column."""
    d = df[NUM_COLS]
    out = pd.DataFrame({
        "mean": d.mean(),
        "median": d.median(),
        "std": d.std(),
        "min": d.min(),
        "q1": d.quantile(0.25),
        "q3": d.quantile(0.75),
        "max": d.max(),
        "IQR": d.quantile(0.75) - d.quantile(0.25),
        "CV(std/mean)": d.std() / d.mean(),
        "skewness": d.skew(),
        "kurtosis": d.kurt(),
    })
    print("\n=== A1. Descriptive statistics ===")
    print(out.round(3).to_string())
    _save(out, "descriptive_stats.csv")
    return out


def descriptive_by_group(df: pd.DataFrame) -> None:
    """Median row-level metrics per engagement level and per placement."""
    for col in ["user_engagement", "placement"]:
        g = df.groupby(col)[["cost", "revenue", "clicks", "CTR", "CPC"]].median()
        g["rows"] = df.groupby(col).size()
        print(f"\n=== A2. Median metrics by {col} ===")
        print(g.round(4).to_string())
        _save(g, f"median_by_{col}.csv")


# --------------------------------------------------------------------------
# B. INFERENTIAL STATISTICS
# --------------------------------------------------------------------------
def normality_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Is the data normal? (Decides parametric vs non-parametric tests.)"""
    rows = []
    for c in ["cost", "revenue", "clicks", "CTR", "ROI"]:
        stat, p = stats.normaltest(df[c])
        rows.append({"column": c, "statistic": stat, "p_value": p,
                     "normal": p > ALPHA, "skewness": df[c].skew()})
    out = pd.DataFrame(rows).set_index("column")
    print("\n=== B1. Normality test (D'Agostino-Pearson) ===")
    print(out.round(4).to_string())
    print("-> p < 0.05 means NOT normal, so we use non-parametric tests below.")
    _save(out, "normality_tests.csv")
    return out


def kruskal_tests(df: pd.DataFrame, metric: str = "ROI") -> pd.DataFrame:
    """Kruskal-Wallis: does the metric differ across the groups of a column?"""
    d = df[df["cost"] >= MIN_COST]
    rows = []
    for col in ["month", "campaign_number", "user_engagement", "banner", "placement"]:
        groups = [g[metric].values for _, g in d.groupby(col) if len(g) >= 5]
        h, p = stats.kruskal(*groups)
        n, k = len(d), len(groups)
        eps2 = max((h - k + 1) / (n - k), 0)  # effect size (epsilon squared)
        rows.append({"factor": col, "groups": k, "H": h, "p_value": p,
                     "significant": p < ALPHA, "effect_size_eps2": eps2})
    out = pd.DataFrame(rows).set_index("factor")
    print(f"\n=== B2. Kruskal-Wallis on row-level {metric} (rows with cost >= {MIN_COST}) ===")
    print(out.round(4).to_string())
    print("-> effect size: <0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, >0.14 large")
    _save(out, f"kruskal_{metric}.csv")
    return out


def pairwise_tests(df: pd.DataFrame, col: str, metric: str = "ROI") -> pd.DataFrame:
    """Mann-Whitney U for every pair of groups, Bonferroni-corrected."""
    d = df[df["cost"] >= MIN_COST]
    levels = [lv for lv, g in d.groupby(col) if len(g) >= 20]
    pairs = list(itertools.combinations(levels, 2))
    rows = []
    for a, b in pairs:
        x = d.loc[d[col] == a, metric]
        y = d.loc[d[col] == b, metric]
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"group_a": a, "group_b": b,
                     "median_a": x.median(), "median_b": y.median(),
                     "p_raw": p, "p_bonferroni": min(p * len(pairs), 1.0)})
    out = pd.DataFrame(rows)
    out["significant"] = out["p_bonferroni"] < ALPHA
    print(f"\n=== B3. Pairwise Mann-Whitney on {metric} by {col} (Bonferroni) ===")
    print(out.round(4).to_string(index=False))
    out.to_csv(os.path.join(RES_DIR, f"pairwise_{col}_{metric}.csv"), index=False)
    return out


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Is 'row is profitable' related to the category? Chi-square + Cramer's V."""
    d = df[df["cost"] >= MIN_COST].copy()
    d["profitable"] = d["profit"] > 0
    rows = []
    for col in ["month", "campaign_number", "user_engagement", "banner", "placement"]:
        table = pd.crosstab(d[col], d["profitable"])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        rows.append({"factor": col, "chi2": chi2, "dof": dof, "p_value": p,
                     "significant": p < ALPHA, "cramers_V": v})
    out = pd.DataFrame(rows).set_index("factor")
    print("\n=== B4. Chi-square: is 'profitable row' related to the factor? ===")
    print(out.round(4).to_string())
    _save(out, "chi_square_profitable.csv")
    return out


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation with p-values for the main drivers of revenue."""
    rows = []
    for c in ["displays", "clicks", "cost", "post_click_conversions", "CTR", "CPC", "CPM"]:
        rho, p = stats.spearmanr(df[c], df["revenue"])
        rows.append({"variable": c, "spearman_rho_with_revenue": rho, "p_value": p})
    out = pd.DataFrame(rows).set_index("variable")
    print("\n=== B5. Spearman correlation with revenue ===")
    print(out.round(4).to_string())
    _save(out, "correlation_tests.csv")
    return out


# --------------------------------------------------------------------------
# C. BOOTSTRAP CONFIDENCE INTERVALS FOR AGGREGATE ROI
# --------------------------------------------------------------------------
def _boot_roi(cost: np.ndarray, rev: np.ndarray) -> np.ndarray:
    n = len(cost)
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        out[i] = (rev[idx].sum() - cost[idx].sum()) / cost[idx].sum()
    return out


def bootstrap_roi_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for level, g in df.groupby(col):
        b = _boot_roi(g["cost"].values, g["revenue"].values)
        rows.append({col: level, "ROI": (g["revenue"].sum() - g["cost"].sum()) / g["cost"].sum(),
                     "CI_low": np.percentile(b, 2.5), "CI_high": np.percentile(b, 97.5),
                     "rows": len(g)})
    return pd.DataFrame(rows).set_index(col).sort_values("ROI", ascending=False)


def bootstrap_roi_difference(df: pd.DataFrame, col: str, a: str, b: str) -> dict:
    ga, gb = df[df[col] == a], df[df[col] == b]
    ba = _boot_roi(ga["cost"].values, ga["revenue"].values)
    bb = _boot_roi(gb["cost"].values, gb["revenue"].values)
    diff = ba - bb
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"comparison": f"{a} minus {b} ({col})", "ROI_diff_mean": diff.mean(),
            "CI_low": lo, "CI_high": hi, "significant": (lo > 0) or (hi < 0)}


def bootstrap_section(df: pd.DataFrame) -> None:
    print("\n=== C1. Aggregate ROI with 95% bootstrap confidence intervals ===")
    for col in ["user_engagement", "placement", "campaign_number", "month"]:
        t = bootstrap_roi_table(df, col)
        print(f"\nby {col}")
        print(t.round(3).to_string())
        _save(t, f"roi_ci_{col}.csv")

    comps = [
        bootstrap_roi_difference(df, "placement", "ghi", "mno"),
        bootstrap_roi_difference(df, "user_engagement", "Medium", "High"),
        bootstrap_roi_difference(df, "campaign_number", "camp 2", "camp 1"),
        bootstrap_roi_difference(df, "month", "April", "June"),
        bootstrap_roi_difference(df, "banner", "240 x 400", "728 x 90"),
    ]
    out = pd.DataFrame(comps).set_index("comparison")
    print("\n=== C2. ROI differences (95% bootstrap CI; significant if CI excludes 0) ===")
    print(out.round(3).to_string())
    _save(out, "roi_differences.csv")


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_statistics(df: pd.DataFrame = None, path: str = PROCESSED_PATH) -> None:
    if df is None:
        df = pd.read_csv(path)
    descriptive_stats(df)
    descriptive_by_group(df)
    normality_tests(df)
    kruskal_tests(df, "ROI")
    for col in ["user_engagement", "month", "campaign_number", "placement"]:
        pairwise_tests(df, col, "ROI")
    chi_square_tests(df)
    correlation_tests(df)
    bootstrap_section(df)
    print("\nStatistics finished. Tables saved in outputs/results.")


if __name__ == "__main__":
    run_statistics()