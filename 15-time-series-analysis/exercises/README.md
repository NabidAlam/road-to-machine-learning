# Time Series Exercises (Practice Pack)

These exercises are designed to make you “industry-ready” at time series forecasting: correct splits, strong baselines, honest evaluation, and clear reporting.

## How to use

- Start with **Exercise 1–4** (fundamentals + baselines)
- Then do **5–8** (feature engineering + model comparison)
- Finally **9–10** (production thinking)

> Use a public dataset (download yourself). Do not commit datasets or model artifacts.

---

## Exercise 1: Plot and diagnose a series

- Plot raw series
- Plot rolling mean/std
- Decompose (trend/seasonality/residual)
- Identify stationarity (ADF test)

Deliverable: a short markdown section with plots + your conclusions.

## Exercise 2: Build a naive baseline

Implement and evaluate:

- last value (“persistence”)
- seasonal naive (last week / last year same period)

Report: MAE, RMSE, MAPE/SMAPE (if applicable).

## Exercise 3: Proper time split

Create:

- train
- validation (for tuning)
- test (final)

Compare random split vs time-based split and explain why random is wrong.

## Exercise 4: ARIMA/SARIMA baseline

- Grid small parameter ranges
- Pick best on validation
- Evaluate on test

Deliverable: metrics table + a plot of predictions vs actuals.

## Exercise 5: Feature engineering (lags + rolling stats)

Create features:

- lags: 1, 7, 14 (or dataset-dependent)
- rolling mean/std: 7, 14
- calendar features: day-of-week, month

Train a tree model (e.g., LightGBM/XGBoost if available, else RandomForest) and compare to ARIMA.

## Exercise 6: Forecast horizon experiment

Evaluate performance at:

- 1-step ahead
- 7-step ahead
- 30-step ahead

Discuss which horizon is hardest and why.

## Exercise 7: Walk-forward validation

Implement walk-forward validation and compare stability vs a single holdout split.

## Exercise 8: Deep learning baseline (optional)

- Build an LSTM/GRU baseline
- Compare to your best classical baseline

Focus on leakage prevention and honest evaluation.

## Exercise 9: Drift + monitoring plan (production thinking)

Write a short monitoring plan:

- what to log (latency, error rates, input stats)
- drift checks (mean/quantiles, missingness)
- when to retrain

## Exercise 10: Final report (portfolio-ready)

Write a 1–2 page report:

- problem statement
- data description
- baselines
- best model
- evaluation
- limitations + next steps

