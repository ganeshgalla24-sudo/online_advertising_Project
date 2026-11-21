"""
Step 4: Strategy and machine learning models.

Business problem: spend less on ads that lose money, put more behind ads that
earn well.

What this script builds
  1. Revenue model (regression): predicts revenue of an ad slot BEFORE spending.
  2. Profitability model (classification): predicts if an ad slot will make a profit.
  3. Backtest: pretends we skipped the slots the model flagged as unprofitable and
     measures how much profit we would have gained on data the model never saw.
  4. Budget reallocation strategy: segment-level scale-up / maintain / cut plan.

Only information known BEFORE the ad runs is used as input (campaign, engagement
level, banner, placement, date, planned cost). displays and clicks are results of
spending, so they are NOT used (that would be data leakage).

Run on its own:
    python src/model.py
or from main.py:
    from src.model import run_model
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_PATH = "data/processed/cleaned_data.csv"
FIG_DIR = "outputs/figures"
RES_DIR = "outputs/results"
SEED = 42

NUM_FEATURES = ["log_cost", "day", "month_num", "banner_width", "banner_height"]
CAT_FEATURES = ["campaign_number", "user_engagement", "placement"]
FEATURES = NUM_FEATURES + CAT_FEATURES


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    os.makedirs(RES_DIR, exist_ok=True)
    df.to_csv(os.path.join(RES_DIR, name), index=index)


def _save_fig(name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close()
    print(f"  saved figure: {FIG_DIR}/{name}")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add model inputs and targets."""
    d = df.copy()
    d["log_cost"] = np.log1p(d["cost"])          # cost is very skewed
    d["log_revenue"] = np.log1p(d["revenue"])    # regression target
    d["profitable"] = (d["profit"] > 0).astype(int)  # classification target
    return d


def _preprocessor(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


# --------------------------------------------------------------------------
# 1. Revenue model (regression)
# --------------------------------------------------------------------------
def revenue_model(d: pd.DataFrame):
    X, y = d[FEATURES], d["log_revenue"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": Pipeline([("prep", _preprocessor(True)), ("m", LinearRegression())]),
        "Random Forest": Pipeline([("prep", _preprocessor(False)),
                                   ("m", RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                               n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", _preprocessor(False)),
                                       ("m", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                           random_state=SEED))]),
    }
    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({"model": name,
                     "R2 (log revenue)": r2_score(y_te, pred),
                     "MAE (log revenue)": mean_absolute_error(y_te, pred),
                     "R2 (revenue, original scale)": r2_score(np.expm1(y_te), np.expm1(pred).clip(min=0))})
        fitted[name] = pipe
    res = pd.DataFrame(rows).set_index("model")
    print("\n=== 1. Revenue model (80/20 random split) ===")
    print(res.round(3).to_string())
    _save_table(res, "model_revenue_comparison.csv")

    best_name = res["R2 (log revenue)"].idxmax()
    best = fitted[best_name]
    print(f"best revenue model: {best_name}")

    # Which inputs matter? Permutation importance on the unseen test set.
    imp = permutation_importance(best, X_te, y_te, n_repeats=5, random_state=SEED, scoring="r2")
    imp_df = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("\npermutation importance (drop in R2 when the input is shuffled):")
    print(imp_df.round(4).to_string())
    _save_table(imp_df.rename("importance").to_frame(), "revenue_feature_importance.csv")

    imp_df.sort_values().plot(kind="barh", color="#4C72B0", figsize=(7, 4))
    plt.title(f"What drives revenue? ({best_name})")
    plt.xlabel("importance (drop in R2)")
    _save_fig("revenue_feature_importance.png")

    pred = best.predict(X_te)
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_te, pred, s=6, alpha=0.3)
    lim = [0, max(y_te.max(), pred.max())]
    plt.plot(lim, lim, color="red", linewidth=1)
    plt.xlabel("actual log(1+revenue)")
    plt.ylabel("predicted log(1+revenue)")
    plt.title("Revenue model: predicted vs actual (test set)")
    _save_fig("revenue_predicted_vs_actual.png")
    return best_name, best


# --------------------------------------------------------------------------
# 2. Profitability model (classification)
# --------------------------------------------------------------------------
def _backtest(test: pd.DataFrame, prob: np.ndarray, thresholds=(0.3, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """If we skipped slots with P(profitable) < threshold, what happens to profit?"""
    base_profit = test["profit"].sum()
    rows = []
    for t in thresholds:
        skip = prob < t
        rows.append({
            "threshold": t,
            "slots_skipped_%": 100 * skip.mean(),
            "cost_saved": test.loc[skip, "cost"].sum(),
            "revenue_lost": test.loc[skip, "revenue"].sum(),
            "profit_before": base_profit,
            "profit_after": base_profit - test.loc[skip, "profit"].sum(),
        })
    out = pd.DataFrame(rows).set_index("threshold")
    out["profit_gain"] = out["profit_after"] - out["profit_before"]
    out["ROI_before"] = base_profit / test["cost"].sum()
    out["ROI_after"] = out["profit_after"] / (test["cost"].sum() - out["cost_saved"])
    return out


def profitability_model(d: pd.DataFrame):
    X, y = d[FEATURES], d["profitable"]
    idx_tr, idx_te = train_test_split(d.index, test_size=0.2, random_state=SEED, stratify=y)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]

    models = {
        "Logistic Regression": Pipeline([("prep", _preprocessor(True)),
                                         ("m", LogisticRegression(max_iter=2000))]),
        "Random Forest": Pipeline([("prep", _preprocessor(False)),
                                   ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                                n_jobs=-1, random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", _preprocessor(False)),
                                       ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                            random_state=SEED))]),
    }
    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        rows.append({"model": name,
                     "ROC-AUC": roc_auc_score(y_te, prob),
                     "accuracy": accuracy_score(y_te, prob >= 0.5),
                     "F1": f1_score(y_te, prob >= 0.5)})
        fitted[name] = pipe
    res = pd.DataFrame(rows).set_index("model")
    print(f"\n=== 2. Profitability model (80/20 stratified split; {y.mean():.0%} of rows are profitable) ===")
    print(res.round(3).to_string())
    _save_table(res, "model_profitability_comparison.csv")

    best_name = res["ROC-AUC"].idxmax()
    best = fitted[best_name]
    print(f"best profitability model: {best_name}")

    # ---- 3a. Backtest on the random hold-out set
    test = d.loc[idx_te]
    bt = _backtest(test, best.predict_proba(X_te)[:, 1])
    print("\n=== 3a. Backtest on hold-out set: skip slots predicted unprofitable ===")
    print(bt.round(3).to_string())
    _save_table(bt, "backtest_random_split.csv")

    # ---- 3b. Stricter, time-based backtest: learn from April+May, test on June
    train_t, test_t = d[d["month_num"] <= 5], d[d["month_num"] == 6]
    m_t = clone(best).fit(train_t[FEATURES], train_t["profitable"])
    prob_t = m_t.predict_proba(test_t[FEATURES])[:, 1]
    auc_t = roc_auc_score(test_t["profitable"], prob_t)
    bt_t = _backtest(test_t, prob_t)
    print(f"\n=== 3b. Time-based backtest: train April+May, test June (ROC-AUC {auc_t:.3f}) ===")
    print(bt_t.round(3).to_string())
    _save_table(bt_t, "backtest_time_split.csv")

    # chart: profit before vs after for both backtests
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tbl, title in zip(axes, [bt, bt_t], ["Random hold-out", "June (time-based)"]):
        ax.bar([str(t) for t in tbl.index], tbl["profit_gain"], color="#55A868")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Profit gained by skipping flagged slots\n{title}")
        ax.set_xlabel("skip if P(profitable) below")
        ax.set_ylabel("profit gain")
    _save_fig("backtest_profit_gain.png")
    return best_name, best


# --------------------------------------------------------------------------
# 4. Budget reallocation strategy (segment level)
# --------------------------------------------------------------------------
def budget_strategy(df: pd.DataFrame, shift_fraction: float = 0.5) -> pd.DataFrame:
    keys = ["campaign_number", "user_engagement", "placement"]
    seg = df.groupby(keys).agg(rows=("cost", "size"), cost=("cost", "sum"), revenue=("revenue", "sum"))
    seg["profit"] = seg["revenue"] - seg["cost"]
    seg["ROI"] = seg["profit"] / seg["cost"]
    seg["cost_share"] = seg["cost"] / seg["cost"].sum()
    seg = seg[seg["cost_share"] >= 0.005].copy()  # ignore segments with <0.5% of spend

    seg["action"] = np.where(seg["ROI"] >= 1.0, "SCALE UP",
                     np.where(seg["ROI"] < 0.2, "CUT / FIX", "MAINTAIN"))
    seg = seg.sort_values("ROI", ascending=False)
    print("\n=== 4. Budget strategy: segments (campaign, engagement, placement) with >=0.5% of spend ===")
    print(seg.round(3).to_string())
    _save_table(seg, "budget_strategy_segments.csv")

    up, cut = seg[seg["action"] == "SCALE UP"], seg[seg["action"] == "CUT / FIX"]
    if up.empty or cut.empty:
        print("No segments to move budget between.")
        return seg

    moved = shift_fraction * cut["cost"].sum()
    roi_up = up["revenue"].sum() / up["cost"].sum() - 1
    base_profit = df["profit"].sum()
    loss_from_cut = shift_fraction * cut["profit"].sum()  # profit we give up in cut segments

    rows = []
    for eff in [1.0, 0.5, 0.25]:
        # eff = how much of the historical return the EXTRA money keeps (diminishing returns)
        gain = moved * ((1 + roi_up) * eff - 1)
        rows.append({"extra_money_efficiency": eff,
                     "budget_moved": moved,
                     "profit_change": gain - loss_from_cut,
                     "profit_change_%": 100 * (gain - loss_from_cut) / base_profit})
    scen = pd.DataFrame(rows).set_index("extra_money_efficiency")
    print(f"\nScenario: move {shift_fraction:.0%} of the spend in CUT / FIX segments into SCALE UP segments")
    print(f"(SCALE UP segments historical ROI = {roi_up:.2f}; baseline total profit = {base_profit:,.0f})")
    print(scen.round(2).to_string())
    print("Assumption: extra spend earns only 'efficiency' x the historical return (diminishing returns).")
    _save_table(scen, "budget_strategy_scenarios.csv")
    return seg


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_model(df: pd.DataFrame = None, path: str = PROCESSED_PATH) -> None:
    if df is None:
        df = pd.read_csv(path)
    d = prepare(df)
    revenue_model(d)
    profitability_model(d)
    budget_strategy(d)
    print("\nModelling finished. Figures in outputs/figures, tables in outputs/results.")


if __name__ == "__main__":
    run_model()