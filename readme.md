# Online Advertising Spend Optimization

Executive Post Graduate Certification in Data Science and AI, Assignment 1 (Intellipaat)

Author: Ganesh

## 1. Problem

An online advertising firm wants to lower its ad cost while increasing what it earns from ads.
This project analyses three months of ad performance data (April to June) and builds a
strategy and machine learning models to decide where the budget should go.

## 2. Data

File: `data/raw/online_advertising_performance_data.csv` (15,408 rows, 14 columns)

| Column | Meaning |
|---|---|
| month, day | When the ad ran (April to June, no year given) |
| campaign_number | camp 1, camp 2, camp 3 |
| user_engagement | High, Medium, Low |
| banner | Banner size (8 sizes, for example 240 x 400) |
| placement | Ad placement (abc, def, ghi, jkl, mno) |
| displays, clicks | How many times the ad was shown and clicked |
| cost | Money spent |
| revenue | Money earned from clicks |
| post_click_conversions, post_click_sales_amount | Conversions and sales after a click |

## 3. Project structure

```
online_advertising_Project/
├── main.py                      # runs the full pipeline (steps 1 to 4)
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                     # original CSV
│   └── processed/               # cleaned_data.csv (made by step 1)
├── src/
│   ├── preprocessing.py         # Step 1: cleaning and feature engineering
│   ├── eda.py                   # Step 2: exploratory data analysis
│   ├── stats_analysis.py        # Step 3: descriptive and inferential statistics
│   └── model.py                 # Step 4: ML models and budget strategy
└── outputs/
    ├── figures/                 # all charts (PNG)
    └── results/                 # all tables (CSV)
```

## 4. How to run

```
python -m venv venv
venv\Scripts\activate            # Windows (use: source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
python main.py
```

Each step can also be run alone, for example `python src/eda.py`.
Charts are saved to `outputs/figures/` and tables to `outputs/results/`.

## 5. Approach and findings

### Step 1: Preprocessing and feature engineering (`src/preprocessing.py`)

- Dropped 2 completely empty columns and 5 duplicate rows.
- Filled 413 missing `placement` values with "Unknown" (kept the rows, since they hold real spend).
- Created features: CTR, CPC, CPM, conversion rate, average order value, profit, ROI,
  month number, and banner width, height and area.
- Result: 15,403 rows and 23 columns with no missing values.

### Step 2: Exploratory data analysis (`src/eda.py`)

Ratios such as ROI are calculated from totals, not by averaging row-level values, because
row-level ROI has extreme values (up to 537).

- Overall: cost about 175K, revenue about 276K, profit about 101K, ROI 0.58.
- ROI falls every month: April 0.90, May 0.43, June -0.08.
- Campaign 1 takes 86% of the spend at ROI 0.53. Campaign 2 has the best ROI (1.05) with 10% of spend.
- Placement `ghi` earns 55% of revenue on 36% of spend (ROI 1.44). Placement `mno` takes 54% of
  spend and earns ROI 0.06.
- The 240 x 400 banner has the best ROI (1.30) among the large banners.
- Revenue follows clicks most closely (Spearman 0.98). CPM has almost no link to revenue (0.06).
- About 32% of rows have zero clicks, and 53% of rows lose money.
- Data quirks: 297 rows have more conversions than clicks (likely post-view tracking), and 1 row has
  more clicks than displays. These rows were kept.

### Step 3: Statistics (`src/stats_analysis.py`)

- Descriptive: all key columns are strongly right-skewed (ROI skewness 31), so medians and
  non-parametric tests are used.
- Normality tests reject normality for every key column.
- Kruskal-Wallis tests show ROI differs significantly by month, campaign, engagement, banner
  and placement. The effect is large for engagement (epsilon squared 0.40) and small for the others.
- Pairwise Mann-Whitney tests with Bonferroni correction confirm differences between all
  engagement levels, months and campaigns.
- Chi-square tests link "profitable row" to every factor, strongest for engagement (Cramer's V 0.63).
- Bootstrap 95% confidence intervals for total ROI show these differences are real:

| Comparison | ROI difference | 95% CI |
|---|---|---|
| ghi minus mno (placement) | 1.37 | 1.19 to 1.55 |
| Medium minus High (engagement) | 0.85 | 0.72 to 0.98 |
| camp 2 minus camp 1 | 0.51 | 0.38 to 0.63 |
| April minus June | 0.97 | 0.83 to 1.13 |
| 240 x 400 minus 728 x 90 (banner) | 1.08 | 0.85 to 1.30 |

Note: for placement, the typical row (median ROI) of ghi and mno is similar. ghi's advantage comes
from its large-spend rows, so total-money ROI and typical-row ROI tell different stories.

### Step 4: Models and strategy (`src/model.py`)

Only information known before an ad runs is used as input (campaign, engagement, banner,
placement, date, planned cost). Displays and clicks are results of spending, so they are excluded
to avoid data leakage.

**Revenue model (regression).** Predicts log revenue.

| Model | R2 (log revenue) |
|---|---|
| Linear Regression | 0.938 |
| Random Forest | 0.984 |
| Gradient Boosting (best) | 0.987 |

Planned cost is the strongest input, followed by engagement, placement and campaign.

**Profitability model (classification).** Predicts whether an ad slot makes a profit.

| Model | ROC-AUC | Accuracy |
|---|---|---|
| Logistic Regression | 0.840 | 0.762 |
| Random Forest (best) | 0.955 | 0.889 |
| Gradient Boosting | 0.953 | 0.885 |

**Backtest.** Skipping slots the model flags as unprofitable (threshold 0.5):

- On a random hold-out set, ROI goes from 0.50 to 1.12 and profit rises by about 4,350.
- On a stricter time-based test (train on April and May, test on June), ROI goes from -0.08 to 0.23
  and profit rises by about 6,300.

**Budget strategy.** Each campaign, engagement and placement segment with at least 0.5% of spend is
labelled SCALE UP (ROI of 1.0 or more), MAINTAIN, or CUT / FIX (ROI below 0.2).

- `camp 1 / High / mno` takes 46% of all spend at ROI -0.13. It is the biggest money sink.
- `camp 1 / High / ghi` takes 28% of spend at ROI +1.50. It is the best large segment.
- So High engagement is not weak on its own. Its result depends strongly on the placement.

## 6. Recommendations

1. Reduce or fix spend on `camp 1 / High / mno` and `camp 1 / High / def`.
2. Gradually shift budget toward `ghi` and `jkl` placements and toward Medium engagement segments,
   and track the return on each increase.
3. Use the profitability model to screen ad slots before buying, starting with a conservative threshold.
4. Review why ROI fell month after month, ending below zero in June. Check seasonality and campaign changes.
5. Start with a small budget shift and measure the result. The scenario in `model.py` shows the payoff
   depends strongly on how well the extra spend performs (+18% profit if it keeps half the historical
   return, -11% if it keeps only a quarter).

## 7. Limitations

- Only three months of data, so seasonality is unknown.
- Planned cost is used as a model input, but in practice spend is partly driven by how many ads get
  shown. This is a main reason for the high model accuracy.
- The backtest assumes skipping a slot removes only its own cost and revenue, with no effect on other slots.
- The bootstrap treats rows as independent, but rows from the same day or campaign are probably
  related, so the confidence intervals may be slightly too narrow.
- The data is observational, so differences between groups do not prove that moving budget will have the same effect.
- The budget scenario assumes extra spend keeps a fixed share of the historical return. This is an assumption, not a forecast.