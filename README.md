# Foreign Exchange Rate Prediction using Machine Learning

Predicting next-day EUR/USD closing price using technical indicators and tree-based ML models (Decision Tree, Random Forest, XGBoost), benchmarked against a naive persistence baseline.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Workflow](#2-workflow)
3. [Data Collection](#3-data-collection)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Multicollinearity Analysis](#6-multicollinearity-analysis)
7. [Target Variable & Avoiding Leakage](#7-target-variable--avoiding-leakage)
8. [Train-Test Split](#8-train-test-split)
9. [Baseline Model](#9-baseline-model)
10. [Modeling](#10-modeling)
11. [Final Results](#11-final-results)
12. [Feature Importance](#12-feature-importance)
13. [Key Findings](#13-key-findings)
14. [Limitations](#14-limitations)
15. [Future Scope](#15-future-scope)
16. [Repository Structure & How to Run](#16-repository-structure--how-to-run)

---

## 1. Project Overview

The FX market trades roughly $6.6 trillion daily and is close to informationally efficient, meaning short-horizon price movements are difficult to predict from public information alone.

**Objective:** Test whether classical technical indicators (RSI, MACD, Bollinger Bands, Parabolic SAR, Stochastic Oscillator) combined with tree-based ML models can predict next-day EUR/USD closing price better than a naive "no change" forecast.

This project treats an honest result — including where models fail to beat a trivial baseline, and why — as a legitimate, evidence-backed finding, not something to hide.

---

## 2. Workflow

```
Data Collection → EDA → Feature Engineering → Multicollinearity Check
→ Target Definition → Chronological Train/Test Split → Baseline
→ Model Training (Decision Tree → Random Forest → XGBoost)
→ Feature Importance → Conclusion
```

---

## 3. Data Collection

Two sources were evaluated (`Data_extraction_using_web_scraping.ipynb`):

| Source | Library | What it gives you | Verdict |
|---|---|---|---|
| FRED (`DEXUSEU`) | `pandas_datareader` | Single daily rate, no OHLCV | Explored only, not used further |
| Yahoo Finance (`EURUSD=X`) | `yfinance` | True Open/High/Low/Close/Adj Close/Volume | **Used for the final project** |

```python
df = yf.download("EURUSD=X", start="2000-01-01", end="2026-01-01", auto_adjust=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel('Ticker')
df = df.sort_index()
df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
```

**Data quality checks performed and verified:**
- Although the query requested data from 2000, Yahoo Finance's earliest available `EURUSD=X` data begins **2003-12-01** — 5,730 rows through 2025-12-31.
- No missing values in any raw OHLCV column (`isna().sum()` = 0 across the board).
- `Volume` is **100% zero** for every single row (`(Volume == 0).mean() = 1.0`) — expected, since FX is an OTC (over-the-counter) market with no centralized exchange to report volume the way equities do.
- `Close` equals `Adj Close` exactly for every row — expected, since "adjusted close" accounts for stock splits/dividends, which don't apply to currency pairs.
- Only **1** business day is missing across the entire ~22-year span — a remarkably complete series.
- **⚠️ Data anomaly found:** `df.describe()` shows `Low` has a minimum value of **0.072902**, while `Open`/`High`/`Close`/`Adj Close` all bottom out around **0.96–0.97**. EUR/USD has never traded near 0.07 in its history — this is very likely a bad tick from the data source. Since `Low` feeds directly into Bollinger Bands, the Stochastic Oscillator, and Parabolic SAR, this should be located and either corrected or removed before treating the dataset as fully clean.

---

## 4. Exploratory Data Analysis

Performed in `Feature_Engeneering.ipynb` before any feature engineering:

- `df.info()` / `df.describe()` — structural and statistical sanity check (see anomaly above).
- `df.index.is_monotonic_increasing` — confirmed chronological order.
- `df.index.duplicated().sum()` — confirmed 0 duplicate dates.
- Full business-day date range compared against the actual index (`pd.date_range(..., freq='B')`) to check for gaps — only 1 missing day found.
- `df['Close'].plot()` — visual inspection of the price series over time.

---

## 5. Feature Engineering

Built using the `ta` (technical analysis) library.

| Feature | Formula / Logic | What it captures |
|---|---|---|
| `momentum_rsi` | 100 − 100/(1 + avg gain/avg loss), 14-day window | Momentum strength; overbought (>70) / oversold (<30) |
| `trend_macd`, `trend_macd_signal`, `trend_macd_diff` | EMA(12) − EMA(26); 9-day EMA of that; their difference | Trend direction and momentum shift |
| `volatility_bbm/bbl/bbh` | 20-day SMA ± 2×rolling std | Volatility-adjusted price bands |
| `trend_psar` | Iterative stop-and-reverse trend indicator | Trend-following reversal signal |
| `momentum_stoch`, `momentum_stoch_signal` | Position of Close within 14-day High-Low range; 3-day SMA of that | Where price sits in its recent range |
| `Close_Lag1` | `Close.shift(1)` | Yesterday's price |

```python
df['momentum_rsi'] = RSIIndicator(close=df['Close']).rsi()

macd = MACD(close=df['Close'])
df['trend_macd'] = macd.macd()
df['trend_macd_signal'] = macd.macd_signal()
df['trend_macd_diff'] = macd.macd_diff()

bollinger = BollingerBands(close=df['Close'])
df['volatility_bbm'] = bollinger.bollinger_mavg()
df['volatility_bbl'] = bollinger.bollinger_lband()
df['volatility_bbh'] = bollinger.bollinger_hband()

psar = PSARIndicator(high=df['High'], low=df['Low'], close=df['Close'])
df['trend_psar'] = psar.psar()

stochastic = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
df['momentum_stoch'] = stochastic.stoch()
df['momentum_stoch_signal'] = stochastic.stoch_signal()

df['Close_Lag1'] = df['Close'].shift()
```

**Raw price-level features transformed into stationary versions**, since `volatility_bbm/bbl/bbh` and `trend_psar` are near-restatements of `Close` itself:

```python
df['bb_percent_b'] = (df['Close'] - df['volatility_bbl']) / (df['volatility_bbh'] - df['volatility_bbl'])
df['bb_width']      = (df['volatility_bbh'] - df['volatility_bbm']) / df['volatility_bbm']
df['above_psar']    = (df['Close'] > df['trend_psar']).astype(int)
```

Final engineered dataset: **5730 rows × 21 columns**, saved to `EURUSD_Official_Fetures_Included.csv`.

---

## 6. Multicollinearity Analysis

Checked via correlation heatmap and **Variance Inflation Factor (VIF)** (`statsmodels`).

```python
corr_matrix = X.corr()
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

At this stage of analysis, `X` still included `Close` — and the VIF results were, in hindsight, one of the first clear signals that `Close` was a problem:

| Feature | VIF |
|---|---|
| `Close` | **32,754.67** |
| `Close_Lag1` | **32,608.81** |
| `momentum_rsi` | 93.71 |
| `momentum_stoch` | 51.46 |
| `momentum_stoch_signal` | 47.25 |
| `bb_percent_b` | 35.79 |
| `above_psar` | 4.42 |
| `bb_width` | 4.03 |
| `trend_macd_diff` | 2.43 |

`Close` and `Close_Lag1` showing near-identical, extreme VIF values (~32,000+) is a direct signal that the two carry almost the same information — reinforcing the decision (see Section 7) to drop `Close` from the modeling features entirely. The remaining features sit in a much more reasonable range; the moderate collinearity among `momentum_rsi`/`momentum_stoch`/`momentum_stoch_signal`/`bb_percent_b` is expected, since they're different mathematical formulations of the same underlying "where is price relative to its recent range" concept.

Two **perfect** (deterministic) redundancies were identified and removed earlier in the process, before this table was produced:

| Identity | Why it's exactly redundant |
|---|---|
| `trend_macd_diff = trend_macd − trend_macd_signal` | True by construction, every row |
| `volatility_bbm = (volatility_bbh + volatility_bbl) / 2` | True by construction |

---

## 7. Target Variable & Avoiding Leakage

**Target:** next-day closing price.

```python
df['Target'] = df['Close'].shift(-1)
```

### The most important issue caught in this project: `Close` inside `X`

`Close` was initially included in the feature set. Since `Target` is literally tomorrow's `Close`, and tomorrow's price is almost always extremely close to today's, `Close` became a dominant, trivially strong predictor — confirmed directly by the VIF table above, and later confirmed again by feature importance (Section 12), where `Close`/`Close_Lag1` absorbed nearly all of the model's decision-making.

**Fix — final feature set excludes `Close` entirely:**

```python
X = df[['momentum_rsi', 'trend_macd_diff', 'momentum_stoch',
        'momentum_stoch_signal', 'Close_Lag1', 'bb_percent_b', 'bb_width']]
y = df['Target']
```

All results reported in Sections 9–12 use this corrected 7-feature set. `Close` is used only to compute the naive baseline (Section 9), never as a model input.

> **Note:** `above_psar` was engineered (Section 5) but is not present in the final `X` used for modeling — worth a deliberate decision either way (include it and re-run, or document why it was excluded) before calling the feature set final.

---

## 8. Train-Test Split

**Chronological split — never random.** Random splitting would let the model train on future dates and validate on past dates, a direct form of leakage for time-ordered data.

```python
split_date = '2022-01-01'
train_mask = df.index < split_date
test_mask = df.index >= split_date

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
```

Result: **4658 training rows** (2004–2021), **1038 test rows** (2022–2025).

---

## 9. Baseline Model

**Naive persistence forecast:** assume tomorrow's price equals today's price.

```python
baseline_pred = df.loc[X_test.index, 'Close']
```

| Metric | Value |
|---|---|
| MAE | 0.00400 |
| RMSE | 0.00534 |
| MAPE | **0.37%** |

Every model below is judged against this number.

---

## 10. Modeling

### 10.1 Decision Tree

```python
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
```

A single tree, depth-capped to prevent memorizing training data.

### 10.2 Random Forest

```python
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
```

### 10.3 Hyperparameter Tuning (exploratory — see caveat)

`RandomizedSearchCV` with `TimeSeriesSplit(n_splits=5)` was used to search Random Forest hyperparameters (never standard k-fold, for the same leakage reason as Section 8):

```python
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 8, 10, 15, None],
    'min_samples_leaf': [1, 3, 5, 10],
    'max_features': ['sqrt', 0.8, 1.0]
}
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, n_iter=20, scoring='neg_mean_absolute_error',
    cv=tscv, random_state=42, n_jobs=-1
)
```

Best params found: `n_estimators=300, max_depth=8, min_samples_leaf=5, max_features=1.0` (best CV MAE: 0.0239).

> **Caveat:** this tuning run was performed *before* `Close` was removed from `X` — so the params above were optimized against a feature set that included the leakage described in Section 7. The final results in Section 11 use the **base, untuned** Random Forest configuration (`n_estimators=200, max_depth=5`) on the corrected feature set. Re-running this search on the corrected 7-feature `X` is a natural next step (see Future Scope).

### 10.4 XGBoost

```python
from xgboost import XGBRegressor
xgb_model = XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
```

Unlike Random Forest (independent trees averaged in parallel), XGBoost builds trees **sequentially** — each new tree targets the residual errors of all previous trees combined.

---

## 11. Final Results

**All results below use the corrected feature set — `Close` excluded from `X`.**

| Model | Test MAE | Test RMSE | Test MAPE | Train MAPE |
|---|---|---|---|---|
| **Baseline (naive)** | 0.00400 | 0.00534 | **0.37%** | — |
| Decision Tree | 0.01196 | 0.02067 | 1.15% | 0.67% |
| Random Forest (base config) | 0.01086 | 0.01941 | 1.04% | 0.61% |
| XGBoost | **0.01051** | **0.01837** | **1.01%** | 0.43% |

**None of the three models beat the naive baseline.** XGBoost > Random Forest > Decision Tree, in the expected order — but all three land in a similar range, well short of 0.37%.

---

## 12. Feature Importance

From the final XGBoost model (7-feature set, `Close` excluded):

| Feature | Importance |
|---|---|
| `Close_Lag1` | **0.9467** |
| `momentum_stoch_signal` | 0.0143 |
| `momentum_rsi` | 0.0125 |
| `trend_macd_diff` | 0.0108 |
| `bb_width` | 0.0082 |
| `momentum_stoch` | 0.0050 |
| `bb_percent_b` | 0.0026 |

**94.7% of the model's decision-making rests on yesterday's price alone.** All five technical indicators combined contribute under 6%.

---

## 13. Key Findings

1. **Daily EUR/USD price movement is close to a random walk.** All three model families converged to a similar performance ceiling — evidence the limitation is the *information content of the features*, not model choice or tuning.
2. **Classical technical indicators, computed purely from price history, add only marginal predictive value beyond the most recent price.** Consistent with weak-form market efficiency.
3. **The VIF analysis and feature importance analysis independently pointed to the same conclusion** — `Close`/`Close_Lag1` dominate, everything else is a minor correction on top.

---

## 14. Limitations

- **Data anomaly:** `Low` column has a minimum value (0.0729) inconsistent with EUR/USD's actual historical range — likely a bad tick, not yet investigated or corrected.
- Feature set is limited to price-derived technical indicators — no macroeconomic data, news/sentiment, or order-flow data.
- Single currency pair (EUR/USD) — not tested for generalization across pairs.
- `above_psar` was engineered but not included in the final feature set — an open decision, not a deliberate exclusion with documented rationale.
- Hyperparameter tuning (Section 10.3) was performed on a feature set that still included `Close` — the final reported Random Forest uses base, untuned hyperparameters.
- 1-day-ahead price-level prediction is a hard target for a near-efficient market; return-based framing was not explored in this iteration.
- No walk-forward (rolling-origin) backtesting — a single chronological train/test split.

---

## 15. Future Scope

- Locate and correct/remove the `Low` column anomaly, then verify whether it changed any downstream feature values or results.
- Re-run `RandomizedSearchCV` hyperparameter tuning on the corrected (`Close`-excluded) feature set.
- Decide on and test `above_psar`'s inclusion in the final feature set.
- Reframe the target as **log return** (`log(Close_t+1 / Close_t)`) instead of raw price — removes the trivial "predict no change" shortcut entirely.
- Add macroeconomic features (interest rate differentials, CPI, central bank policy dates).
- Walk-forward validation across multiple rolling windows instead of one fixed split.

---

## 16. Repository Structure & How to Run

```
exchange-rate-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_exploratory.ipynb     # VIF, multicollinearity, hyperparameter search
│   └── 04_modeling_final.ipynb           # corrected feature set, final results
└── data/
    ├── EURUSD_Official_OHLCV.csv
    └── EURUSD_Official_Fetures_Included.csv
```

**Tech stack:** `Python` · `pandas` · `numpy` · `yfinance` · `ta` · `scikit-learn` · `xgboost` · `statsmodels` · `matplotlib` / `seaborn`

**To run:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_data_extraction.ipynb
```
Run notebooks in order — each writes a CSV the next one reads.
