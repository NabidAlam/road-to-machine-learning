# Time Series Analysis - Complete Guide

Comprehensive guide to time series analysis and forecasting using statistical and deep learning methods.

## Table of Contents

- [Introduction](#introduction)
- [Time Series Fundamentals](#time-series-fundamentals)
- [Statistical Methods](#statistical-methods)
- [Deep Learning for Time Series](#deep-learning-for-time-series)
- [Feature Engineering](#feature-engineering)
- [Evaluation and Validation](#evaluation-and-validation)
- [Practice Exercises](#practice-exercises)

---

## Introduction

**Time Series** is a sequence of data points collected over time intervals. Examples include:
- Stock prices
- Sales data
- Temperature readings
- Website traffic
- Energy consumption

### Key Characteristics

- **Temporal Order**: Data points are ordered by time
- **Dependencies**: Current values depend on past values
- **Trends**: Long-term patterns
- **Seasonality**: Repeating patterns over fixed periods

---

## Time Series Fundamentals

### Components of Time Series

1. **Trend**: Long-term increase or decrease
2. **Seasonality**: Regular patterns that repeat over fixed periods
3. **Cyclical**: Patterns that don't have fixed periods
4. **Noise/Random**: Irregular, unpredictable variations

### Stationarity

A time series is **stationary** if:
- Mean is constant over time
- Variance is constant over time
- Autocorrelation doesn't depend on time

**Why Important**: Many models assume stationarity.

**Making Series Stationary**:
- Differencing: Take differences between consecutive values
- Log transformation: Reduce variance
- Detrending: Remove trend component

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load time series data
dates = pd.date_range('2020-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

# Check stationarity (Augmented Dickey-Fuller test)
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")
    
    return result[1] <= 0.05

# Check original series
check_stationarity(ts)

# Make stationary with differencing
ts_diff = ts.diff().dropna()
check_stationarity(ts_diff)
```

### Time Series Decomposition

Separate time series into components:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(ts, model='additive', period=12)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(ts, label='Original')
plt.legend()
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend()
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend()
plt.tight_layout()
plt.show()
```

### Autocorrelation

Measures correlation between series and lagged versions:

```python
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Calculate autocorrelation
autocorr = acf(ts, nlags=20)
partial_autocorr = pacf(ts, nlags=20)

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, lags=20, ax=axes[0])
plot_pacf(ts, lags=20, ax=axes[1])
plt.show()
```

---

## Statistical Methods

### ARIMA (AutoRegressive Integrated Moving Average)

ARIMA(p, d, q):
- **p**: Autoregressive order
- **d**: Differencing order
- **q**: Moving average order

```python
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(ts, order=(2, 1, 2))  # (p, d, q)
fitted_model = model.fit()

# Summary
print(fitted_model.summary())

# Forecast
forecast = fitted_model.forecast(steps=10)
print(f"Forecast: {forecast}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(ts, label='Original')
plt.plot(fitted_model.fittedvalues, label='Fitted')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()
```

### Auto ARIMA

Automatically find best ARIMA parameters:

```python
# Install: pip install pmdarima
from pmdarima import auto_arima

# Auto ARIMA
model = auto_arima(ts, seasonal=True, m=12, 
                   stepwise=True, suppress_warnings=True)
print(model.summary())

# Forecast
forecast = model.predict(n_periods=10)
```

### SARIMA (Seasonal ARIMA)

Handles seasonality:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA(p, d, q)(P, D, Q, s)
# s = seasonal period (e.g., 12 for monthly data)
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
fitted_model = model.fit()

forecast = fitted_model.forecast(steps=12)
```

### Exponential Smoothing

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Triple Exponential Smoothing (Holt-Winters)
model = ExponentialSmoothing(ts, seasonal='add', seasonal_periods=12)
fitted_model = model.fit()

forecast = fitted_model.forecast(steps=12)
```

### Prophet (Facebook)

Robust to missing data and outliers:

```python
# Install: pip install prophet
from prophet import Prophet

# Prepare data (Prophet expects 'ds' and 'y' columns)
df = pd.DataFrame({
    'ds': dates,
    'y': ts.values
})

# Fit model
model = Prophet()
model.fit(df)

# Create future dataframe
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

---

## Deep Learning for Time Series

### LSTM (Long Short-Term Memory)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Prepare data
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1))

# Create sequences
seq_length = 10
X, y = create_sequences(ts_scaled, seq_length)

# Split (time-based!)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, 
         input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X_train, y_train, epochs=50, 
                   validation_data=(X_test, y_test),
                   verbose=1, batch_size=32)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
```

### GRU (Gated Recurrent Unit)

Similar to LSTM but simpler:

```python
from tensorflow.keras.layers import GRU

model = Sequential([
    GRU(50, activation='relu', return_sequences=True,
        input_shape=(seq_length, 1)),
    Dropout(0.2),
    GRU(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

### CNN for Time Series

1D convolutions can capture patterns:

```python
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu',
           input_shape=(seq_length, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
```

---

## Feature Engineering

### Lag Features

```python
# Create lag features
def create_lag_features(df, lags=[1, 2, 3, 7, 14]):
    for lag in lags:
        df[f'lag_{lag}'] = df['value'].shift(lag)
    return df

df = create_lag_features(df)
```

### Rolling Statistics

```python
# Rolling mean, std, min, max
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
df['rolling_std_7'] = df['value'].rolling(window=7).std()
df['rolling_min_7'] = df['value'].rolling(window=7).min()
df['rolling_max_7'] = df['value'].rolling(window=7).max()
```

### Time-based Features

```python
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
```

### Fourier Features

Capture seasonality:

```python
def create_fourier_features(df, period, num_terms=3):
    for i in range(1, num_terms + 1):
        df[f'sin_{i}'] = np.sin(2 * np.pi * i * df.index.dayofyear / period)
        df[f'cos_{i}'] = np.cos(2 * np.pi * i * df.index.dayofyear / period)
    return df

df = create_fourier_features(df, period=365)
```

---

## Evaluation and Validation

### Time-based Splitting

**Never use random split for time series!**

```python
# Correct: Time-based split
split_date = '2023-01-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

# Or percentage-based
split_idx = int(len(df) * 0.8)
train = df[:split_idx]
test = df[split_idx:]
```

### Walk-Forward Validation

```python
def walk_forward_validation(data, n_train, n_test, model_func):
    """
    Walk-forward validation for time series
    """
    predictions = []
    actuals = []
    
    for i in range(n_test):
        # Training set
        train = data[i:i+n_train]
        
        # Test set (next value)
        test = data[i+n_train]
        
        # Train model
        model = model_func(train)
        
        # Predict
        pred = model.predict(test)
        predictions.append(pred)
        actuals.append(test)
    
    return predictions, actuals
```

### Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / 
                        (np.abs(y_true) + np.abs(y_pred)))

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred)
smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
```

---

## Practice Exercises

### Exercise 1: Basic Time Series Analysis

1. Load a time series dataset (e.g., airline passengers)
2. Plot the time series
3. Check for stationarity
4. Decompose into components
5. Plot ACF and PACF

### Exercise 2: ARIMA Modeling

1. Fit an ARIMA model
2. Use auto_arima to find best parameters
3. Forecast next 12 periods
4. Evaluate using RMSE and MAE

### Exercise 3: LSTM Forecasting

1. Prepare data for LSTM (create sequences)
2. Build and train LSTM model
3. Make predictions
4. Compare with ARIMA results

### Exercise 4: Feature Engineering

1. Create lag features
2. Add rolling statistics
3. Extract time-based features
4. Build model with engineered features

---

## Resources

### Libraries

- **statsmodels**: Statistical models (ARIMA, SARIMA)
- **pmdarima**: Auto ARIMA
- **prophet**: Facebook's forecasting tool
- **TensorFlow/Keras**: Deep learning models

### Datasets

- [Time Series Datasets](https://www.kaggle.com/datasets?search=time+series)
- [UCI Time Series](https://archive.ics.uci.edu/ml/datasets.php)
- [M4 Competition](https://www.m4.unic.ac.cy/)

### Books

- **"Forecasting: Principles and Practice"** by Rob J Hyndman
  - [Online Book](https://otexts.com/fpp3/)

---

## Key Takeaways

1. **Never Random Split**: Always use time-based splitting
2. **Check Stationarity**: Many models require stationary data
3. **Feature Engineering**: Lag and rolling features are crucial
4. **Multiple Methods**: Try both statistical and deep learning approaches
5. **Proper Evaluation**: Use time-series specific metrics and validation

---

**Remember**: Time series analysis requires understanding temporal dependencies. Always respect the time order of your data!

