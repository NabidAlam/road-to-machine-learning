# Time Series Quick Reference Guide

Quick reference for time series analysis techniques.

## Table of Contents

- [Key Concepts](#key-concepts)
- [Code Snippets](#code-snippets)
- [Model Selection](#model-selection)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Key Concepts

### Stationarity

```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
is_stationary = result[1] <= 0.05
```

### Time-based Split

```python
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
```

---

## Code Snippets

### ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(ts, order=(2, 1, 2))
fitted = model.fit()
forecast = fitted.forecast(steps=10)
```

### Auto ARIMA

```python
from pmdarima import auto_arima
model = auto_arima(ts, seasonal=True, m=12)
```

### LSTM

```python
model = Sequential([
    LSTM(50, input_shape=(seq_length, 1)),
    Dense(1)
])
```

---

## Model Selection

| Model | Use Case | Notes |
|-------|----------|-------|
| **ARIMA** | Univariate, stationary | Good baseline |
| **Prophet** | Strong seasonality | Handles holidays |
| **LSTM** | Complex patterns | Needs more data |
| **VAR** | Multivariate | Multiple series |

---

## Common Issues & Solutions

### Issue 1: Non-stationary Data

**Solution**: Differencing, log transform

### Issue 2: Data Leakage

**Solution**: Always use time-based split

---

## Best Practices Checklist

- [ ] Use time-based train/test split
- [ ] Check stationarity
- [ ] Decompose time series
- [ ] Try multiple models
- [ ] Use proper metrics (RMSE, MAE)
- [ ] Visualize predictions

---

**Remember**: Always respect temporal order!

