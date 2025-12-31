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

**Time Series** is a sequence of data points collected over time intervals.

### What is Time Series Data?

Time series data differs from regular data because:
- **Order Matters**: Temporal sequence is crucial
- **Dependencies**: Current values depend on past values
- **Cannot Randomize**: Must preserve time order
- **Special Validation**: Cannot use random train/test split

### Examples of Time Series

- **Financial**: Stock prices, exchange rates, cryptocurrency
- **Business**: Sales, revenue, website traffic, user signups
- **Environmental**: Temperature, rainfall, air quality
- **Industrial**: Energy consumption, production output, sensor readings
- **Healthcare**: Patient vitals, disease cases, hospital admissions

### Key Characteristics

- **Temporal Order**: Data points are ordered by time (critical!)
- **Dependencies**: Current values depend on past values (autocorrelation)
- **Trends**: Long-term increase or decrease
- **Seasonality**: Repeating patterns over fixed periods (daily, weekly, yearly)
- **Cyclical**: Patterns without fixed periods (business cycles)
- **Noise**: Random variations

### Why Time Series is Different

```python
# WRONG: Random split (breaks temporal order)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data, test_size=0.2)  # DON'T DO THIS!

# CORRECT: Time-based split
split_date = '2023-01-01'
train = data[data.index < split_date]
test = data[data.index >= split_date]
```

---

## Time Series Fundamentals

### Components of Time Series

1. **Trend**: Long-term increase or decrease
2. **Seasonality**: Regular patterns that repeat over fixed periods
3. **Cyclical**: Patterns that don't have fixed periods
4. **Noise/Random**: Irregular, unpredictable variations

### Stationarity

A time series is **stationary** if its statistical properties don't change over time.

**Requirements for Stationarity:**
1. **Constant Mean**: Mean doesn't change over time
2. **Constant Variance**: Variance is constant (homoscedasticity)
3. **Constant Autocorrelation**: Autocorrelation depends only on lag, not time

**Why Important**: 
- Many models (ARIMA, etc.) assume stationarity
- Easier to model and forecast
- Statistical properties are consistent

**Making Series Stationary**:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load time series data
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.linspace(0, 10, 365)
seasonal = 5 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
noise = np.random.randn(365) * 0.5
ts = pd.Series(trend + seasonal + noise, index=dates)

# Check stationarity (Augmented Dickey-Fuller test)
def check_stationarity(timeseries, title="Time Series"):
    """Check if time series is stationary using ADF test"""
    result = adfuller(timeseries.dropna())
    
    print(f"\n{title} Stationarity Test:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")
    
    is_stationary = result[1] <= 0.05
    if is_stationary:
        print("✓ Series is STATIONARY (p-value <= 0.05)")
    else:
        print("✗ Series is NOT STATIONARY (p-value > 0.05)")
    
    return is_stationary

# Check original series
check_stationarity(ts, "Original")

# Method 1: Differencing (remove trend)
ts_diff = ts.diff().dropna()
check_stationarity(ts_diff, "First Difference")

# Method 2: Log transformation (stabilize variance)
ts_log = np.log(ts + 1)  # +1 to handle zeros
check_stationarity(ts_log, "Log Transformed")

# Method 3: Seasonal differencing (remove seasonality)
ts_seasonal_diff = ts.diff(periods=91).dropna()  # Quarterly difference
check_stationarity(ts_seasonal_diff, "Seasonal Difference")

# Method 4: Detrending (remove trend component)
decomposition = seasonal_decompose(ts, model='additive', period=91)
ts_detrended = ts - decomposition.trend
check_stationarity(ts_detrended.dropna(), "Detrended")

# Visualize transformations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].plot(ts)
axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(ts_diff)
axes[0, 1].set_title('First Difference', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(ts_log)
axes[0, 2].set_title('Log Transformed', fontsize=12, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(ts_seasonal_diff)
axes[1, 0].set_title('Seasonal Difference', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(ts_detrended)
axes[1, 1].set_title('Detrended', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()
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

ARIMA combines three components:
- **AR (p)**: Autoregressive - uses past values
- **I (d)**: Integrated - differencing to make stationary
- **MA (q)**: Moving Average - uses past forecast errors

**ARIMA(p, d, q) Parameters:**
- **p**: Number of lag observations (AR order)
- **d**: Degree of differencing (I order)
- **q**: Size of moving average window (MA order)

**How to Choose Parameters:**
- Use ACF/PACF plots
- Use auto_arima (recommended)
- Grid search with validation

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Check ACF/PACF to estimate p and q
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts, lags=20, ax=axes[0])
axes[0].set_title('ACF Plot', fontsize=12, fontweight='bold')
plot_pacf(ts, lags=20, ax=axes[1])
axes[1].set_title('PACF Plot', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# Step 2: Fit ARIMA model
model = ARIMA(ts, order=(2, 1, 2))  # (p, d, q)
fitted_model = model.fit()

# Summary with diagnostics
print(fitted_model.summary())

# Step 3: Check residuals (should be white noise)
residuals = fitted_model.resid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals plot
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Residuals distribution
axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Q-Q plot (should be linear if normal)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# ACF of residuals (should have no significant correlations)
plot_acf(residuals, lags=20, ax=axes[1, 1])
axes[1, 1].set_title('ACF of Residuals', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Step 4: Forecast
forecast_steps = 10
forecast = fitted_model.forecast(steps=forecast_steps)
forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()

# Plot with confidence intervals
plt.figure(figsize=(14, 6))
plt.plot(ts.index, ts.values, label='Original', linewidth=2)
plt.plot(fitted_model.fittedvalues.index, fitted_model.fittedvalues.values, 
         label='Fitted', linewidth=2, alpha=0.7)

# Forecast
forecast_index = pd.date_range(start=ts.index[-1], periods=forecast_steps+1, freq='D')[1:]
plt.plot(forecast_index, forecast, label='Forecast', linewidth=2, color='red')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                alpha=0.3, color='red', label='95% Confidence Interval')

plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('ARIMA Model: Original, Fitted, and Forecast', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nForecast for next {forecast_steps} periods:")
for i, (date, value) in enumerate(zip(forecast_index, forecast)):
    print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}")
```

### Auto ARIMA

Automatically find best ARIMA parameters using information criteria.

```python
# Install: pip install pmdarima
try:
    from pmdarima import auto_arima
    
    # Auto ARIMA with seasonal
    model = auto_arima(
        ts,
        seasonal=True,
        m=12,  # Seasonal period (12 for monthly data)
        stepwise=True,  # Use stepwise algorithm (faster)
        suppress_warnings=True,
        error_action='ignore',
        max_order=5,  # Maximum (p, d, q) to try
        information_criterion='aic',  # AIC, AICc, or BIC
        trace=True  # Print progress
    )
    
    print("\nBest Model Summary:")
    print(model.summary())
    
    # Get best parameters
    print(f"\nBest ARIMA order: {model.order}")
    if model.seasonal:
        print(f"Best seasonal order: {model.seasonal_order}")
    
    # Forecast
    forecast, conf_int = model.predict(n_periods=10, return_conf_int=True)
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(ts.index, ts.values, label='Original', linewidth=2)
    forecast_index = pd.date_range(start=ts.index[-1], periods=11, freq='D')[1:]
    plt.plot(forecast_index, forecast, label='Forecast', linewidth=2, color='red')
    plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1],
                     alpha=0.3, color='red', label='95% CI')
    plt.legend()
    plt.title('Auto ARIMA Forecast', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Install pmdarima: pip install pmdarima")
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

Robust forecasting tool that handles:
- Missing data
- Outliers
- Holidays
- Multiple seasonalities
- Trend changes

```python
# Install: pip install prophet
try:
    from prophet import Prophet
    import warnings
    warnings.filterwarnings('ignore')
    
    # Prepare data (Prophet expects 'ds' and 'y' columns)
    df = pd.DataFrame({
        'ds': dates,
        'y': ts.values
    })
    
    # Fit model with custom parameters
    model = Prophet(
        yearly_seasonality=True,  # Yearly seasonality
        weekly_seasonality=True,  # Weekly seasonality
        daily_seasonality=False,  # Daily (usually too noisy)
        seasonality_mode='additive',  # 'additive' or 'multiplicative'
        changepoint_prior_scale=0.05,  # Flexibility of trend changes
        holidays_prior_scale=10.0  # If using holidays
    )
    
    # Add custom seasonalities
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    # Add holidays (optional)
    # holidays = pd.DataFrame({
    #     'holiday': 'holiday_name',
    #     'ds': pd.to_datetime(['2020-12-25', '2021-12-25']),
    #     'lower_window': 0,
    #     'upper_window': 1,
    # })
    # model.holidays = holidays
    
    # Fit model
    model.fit(df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    
    # Plot forecast
    fig = model.plot(forecast)
    plt.title('Prophet Forecast', fontsize=14, fontweight='bold')
    plt.show()
    
    # Plot components (trend, seasonality)
    fig = model.plot_components(forecast)
    plt.show()
    
    # Access forecast components
    print("\nForecast Components:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend', 'yearly', 'weekly']].tail())
    
    # Evaluate model performance
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # Compare with actual (for in-sample)
    actual = df['y'].values
    predicted = forecast['yhat'].values[:len(actual)]
    
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    print(f"\nProphet Performance (in-sample):")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
except ImportError:
    print("Install prophet: pip install prophet")
```

---

## Deep Learning for Time Series

### Why Deep Learning for Time Series?

- **Non-linear Patterns**: Capture complex relationships
- **Multiple Features**: Handle multivariate time series
- **Long Dependencies**: LSTM/GRU remember long sequences
- **Automatic Feature Learning**: Learn relevant patterns

### LSTM (Long Short-Term Memory)

LSTMs are excellent for time series due to their ability to remember long-term dependencies.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prepare data function
def create_sequences(data, seq_length):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Scale data (important for neural networks!)
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

# Create sequences
seq_length = 10  # Look back 10 time steps
X, y = create_sequences(ts_scaled, seq_length)

# Split (time-based - NEVER random!)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Training sequences: {X_train.shape}")
print(f"Test sequences: {X_test.shape}")

# Build improved LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, 
         input_shape=(seq_length, 1)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(50, activation='relu', return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
]

# Train
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training History', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE', fontsize=12)
plt.title('MAE History', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)

print(f"\nLSTM Performance:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Plot predictions
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual', linewidth=2, marker='o', markersize=4)
plt.plot(predictions, label='Predicted', linewidth=2, marker='s', markersize=4)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('LSTM Predictions vs Actual', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
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

**Critical Rule: Never use random split for time series!**

Time series data has temporal dependencies. Random splitting would:
- Leak future information into training
- Break temporal relationships
- Give unrealistic performance estimates

```python
# CORRECT: Time-based split
split_date = '2023-01-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

# Or percentage-based (preserves order)
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# Visualize split
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['value'], label='Training', linewidth=2)
plt.plot(test.index, test['value'], label='Test', linewidth=2)
plt.axvline(x=split_date if 'split_date' in locals() else train.index[-1], 
           color='red', linestyle='--', linewidth=2, label='Split Point')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Train-Test Split (Time-based)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Walk-Forward Validation

More robust validation for time series.

```python
def walk_forward_validation(data, n_train, n_test, model_func):
    """
    Walk-forward validation for time series
    
    Parameters:
    - data: Time series data
    - n_train: Size of training window
    - n_test: Number of steps to forecast
    - model_func: Function that takes training data and returns fitted model
    """
    predictions = []
    actuals = []
    errors = []
    
    for i in range(n_test):
        # Training set (expanding window)
        train = data[i:i+n_train]
        
        # Test set (next value)
        test_value = data[i+n_train]
        
        # Train model
        model = model_func(train)
        
        # Predict
        pred = model.forecast(steps=1)[0]
        predictions.append(pred)
        actuals.append(test_value)
        
        # Calculate error
        error = abs(pred - test_value)
        errors.append(error)
        
        print(f"Step {i+1}/{n_test}: Predicted={pred:.2f}, Actual={test_value:.2f}, Error={error:.2f}")
    
    # Calculate overall metrics
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    mae = np.mean(errors)
    
    print(f"\nWalk-Forward Validation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return predictions, actuals, errors

# Example usage
# predictions, actuals, errors = walk_forward_validation(
#     ts.values, n_train=100, n_test=20,
#     model_func=lambda train: ARIMA(train, order=(1,1,1)).fit()
# )
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for train_idx, test_idx in tscv.split(X):
    X_train_cv, X_test_cv = X[train_idx], X[test_idx]
    y_train_cv, y_test_cv = y[train_idx], y[test_idx]
    
    # Train model
    model.fit(X_train_cv, y_train_cv)
    
    # Evaluate
    score = model.score(X_test_cv, y_test_cv)
    scores.append(score)
    print(f"Fold score: {score:.4f}")

print(f"\nCross-validation scores: {scores}")
print(f"Mean CV score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
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

1. **Never Random Split**: Always use time-based splitting - temporal order is critical
2. **Check Stationarity**: Many models require stationary data - use ADF test
3. **Feature Engineering**: Lag and rolling features are crucial for time series
4. **Multiple Methods**: Try both statistical (ARIMA) and deep learning (LSTM) approaches
5. **Proper Evaluation**: Use time-series specific metrics (RMSE, MAE, MAPE) and validation (walk-forward)
6. **Understand Components**: Decompose to understand trend, seasonality, and noise
7. **ACF/PACF**: Use these plots to identify ARIMA parameters
8. **Auto ARIMA**: Use when unsure about parameters

---

## Best Practices

### Data Preparation
- Always preserve temporal order
- Handle missing values appropriately (forward fill, interpolation)
- Check for outliers and handle them
- Normalize/scale data for neural networks

### Model Selection
- Start with simple methods (moving average, exponential smoothing)
- Try ARIMA for univariate time series
- Use Prophet for data with strong seasonality
- Use LSTM/GRU for complex patterns and multivariate data

### Evaluation
- Use time-based train/test split
- Use walk-forward validation for robust evaluation
- Report multiple metrics (RMSE, MAE, MAPE)
- Visualize predictions vs actuals
- Check residuals (should be white noise)

---

## Next Steps

- Practice with real time series datasets
- Experiment with different models
- Try multivariate time series
- Explore advanced techniques (state space models, VAR)
- Learn about anomaly detection in time series
- Move to next module or practice projects

**Remember**: Time series analysis requires understanding temporal dependencies. Always respect the time order of your data!

