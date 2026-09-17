# Complete Time Series Project Tutorial

Step-by-step walkthrough of building a time series forecasting system.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Load and Explore Data](#step-1-load-and-explore-data)
- [Step 2: Check Stationarity](#step-2-check-stationarity)
- [Step 3: Build ARIMA Model](#step-3-build-arima-model)
- [Step 4: Build LSTM Model](#step-4-build-lstm-model)
- [Step 5: Compare Models](#step-5-compare-models)

---

## Project Overview

**Project**: Stock Price Forecasting

**Dataset**: Synthetic monthly close prices (swap in your CSV later)

**Goals**: Forecast future prices using ARIMA and LSTM

---

## Step 1: Load and Explore Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Demo series. Replace later with your own dated Close series from a local file.
rng = np.random.default_rng(42)
dates = pd.date_range("2020-01-01", periods=120, freq="ME")
close = 100 + np.cumsum(rng.normal(0, 1.5, size=len(dates)))
ts = pd.Series(close, index=dates, name="Close")

print(ts.describe())
ts.plot(figsize=(14, 6))
plt.title("Stock Prices Over Time")
plt.show()
```

---

## Step 2: Check Stationarity

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] <= 0.05

ts_model = ts.copy()
is_stationary = check_stationarity(ts_model)
if not is_stationary:
    ts_model = ts_model.diff().dropna()
print("Stationary after prep:", check_stationarity(ts_model))
```

---

## Step 3: Build ARIMA Model

```python
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error

split = int(len(ts_model) * 0.8)
ts_train, ts_test = ts_model.iloc[:split], ts_model.iloc[split:]

arima_model = auto_arima(
    ts_train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
)
arima_forecast = arima_model.predict(n_periods=len(ts_test))
arima_rmse = float(np.sqrt(mean_squared_error(ts_test, arima_forecast)))
print(f"ARIMA RMSE: {arima_rmse:.4f}")
```

---

## Step 4: Build LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

def create_sequences(values, seq_length=10):
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(values[i + seq_length])
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).ravel()
seq_length = 10
X, y = create_sequences(ts_scaled, seq_length=seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

split_i = int(len(X) * 0.8)
X_train, X_test = X[:split_i], X[split_i:]
y_train, y_test = y[:split_i], y[split_i:]

lstm_model = Sequential([
    Input(shape=(seq_length, 1)),
    LSTM(32),
    Dense(1),
])
lstm_model.compile(optimizer="adam", loss="mse")
lstm_model.fit(X_train, y_train, epochs=2, batch_size=16, verbose=0)

lstm_pred = lstm_model.predict(X_test, verbose=0).ravel()
lstm_rmse = float(np.sqrt(mean_squared_error(y_test, lstm_pred)))
print(f"LSTM RMSE (scaled): {lstm_rmse:.4f}")
```

---

## Step 5: Compare Models

```python
print(f"ARIMA RMSE: {arima_rmse:.4f}")
print(f"LSTM RMSE (scaled): {lstm_rmse:.4f}")
```

---

**Try next:** Open [Module 16 · Beginner projects](../16-projects-beginner/README.md) and pick one project README to implement end to end.
