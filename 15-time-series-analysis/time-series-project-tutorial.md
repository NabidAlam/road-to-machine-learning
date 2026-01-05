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

**Dataset**: Stock price data

**Goals**: Forecast future prices using ARIMA and LSTM

---

## Step 1: Load and Explore Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)
ts = df['Close']

# Explore
print(ts.describe())
ts.plot(figsize=(14, 6))
plt.title('Stock Prices Over Time')
plt.show()
```

---

## Step 2: Check Stationarity

```python
from statsmodels.tsa.stattools import adfuller

def check_stationarity(ts):
    result = adfuller(ts.dropna())
    return result[1] <= 0.05

is_stationary = check_stationarity(ts)
if not is_stationary:
    ts = ts.diff().dropna()
```

---

## Step 3: Build ARIMA Model

```python
from pmdarima import auto_arima

model = auto_arima(ts, seasonal=True, m=12)
forecast = model.predict(n_periods=30)
```

---

## Step 4: Build LSTM Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prepare sequences
X, y = create_sequences(ts_scaled, seq_length=10)

# Build and train
model = Sequential([LSTM(50), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50)
```

---

## Step 5: Compare Models

```python
# Evaluate both models
arima_rmse = calculate_rmse(y_test, arima_forecast)
lstm_rmse = calculate_rmse(y_test, lstm_forecast)

print(f"ARIMA RMSE: {arima_rmse:.4f}")
print(f"LSTM RMSE: {lstm_rmse:.4f}")
```

---

**Congratulations!** You've built a complete time series forecasting system!

