# Working with Dates and Times - Complete Guide

Comprehensive guide to handling dates, times, and time-based data in Python using Pandas and datetime.

## Table of Contents

- [Introduction](#introduction)
- [Creating Date/Time Objects](#creating-datetime-objects)
- [Parsing Dates](#parsing-dates)
- [Date Arithmetic](#date-arithmetic)
- [Time Series Indexing](#time-series-indexing)
- [Date Components](#date-components)
- [Time Zones](#time-zones)
- [Resampling and Frequency](#resampling-and-frequency)
- [Practice Exercises](#practice-exercises)

---

## Introduction

### Why Dates and Times Matter

Date/time data is common in data science:
- **Time series**: Stock prices, weather data, sales
- **Event logs**: User actions, system events
- **Scheduling**: Appointments, deadlines
- **Analysis**: Trends over time, seasonality

### Key Libraries

- **datetime**: Python's built-in date/time module
- **pandas**: Powerful time series functionality
- **dateutil**: Additional parsing capabilities

---

## Creating Date/Time Objects

### Using datetime Module

```python
from datetime import datetime, date, time, timedelta

# Current date and time
now = datetime.now()
print(now)  # 2024-01-15 10:30:45.123456

# Specific date/time
dt = datetime(2024, 1, 15, 10, 30, 45)
print(dt)

# Date only
d = date(2024, 1, 15)
print(d)

# Time only
t = time(10, 30, 45)
print(t)
```

### Using Pandas

```python
import pandas as pd

# Create Timestamp
ts = pd.Timestamp('2024-01-15 10:30:45')
print(ts)

# Create date range
dates = pd.date_range('2024-01-01', periods=10, freq='D')
print(dates)

# Different frequencies
daily = pd.date_range('2024-01-01', periods=10, freq='D')
weekly = pd.date_range('2024-01-01', periods=10, freq='W')
monthly = pd.date_range('2024-01-01', periods=10, freq='M')
yearly = pd.date_range('2024-01-01', periods=10, freq='Y')

print(f"Daily: {daily}")
print(f"Weekly: {weekly}")
print(f"Monthly: {monthly}")
```

---

## Parsing Dates

### Converting Strings to Dates

```python
# Using pd.to_datetime
dates_str = ['2024-01-15', '2024-02-20', '2024-03-25']
dates = pd.to_datetime(dates_str)
print(dates)

# Various formats
dates_mixed = ['2024-01-15', '01/15/2024', 'Jan 15, 2024']
dates = pd.to_datetime(dates_mixed)
print(dates)

# Handle errors
dates_with_errors = ['2024-01-15', 'invalid', '2024-02-20']
dates = pd.to_datetime(dates_with_errors, errors='coerce')  # Invalid -> NaT
print(dates)

# Custom format
dates_custom = ['15-01-2024', '20-02-2024']
dates = pd.to_datetime(dates_custom, format='%d-%m-%Y')
print(dates)
```

### Common Date Formats

```python
# Format codes
formats = {
    '%Y-%m-%d': '2024-01-15',
    '%m/%d/%Y': '01/15/2024',
    '%d-%m-%Y': '15-01-2024',
    '%B %d, %Y': 'January 15, 2024',
    '%Y-%m-%d %H:%M:%S': '2024-01-15 10:30:45'
}

for fmt, example in formats.items():
    parsed = pd.to_datetime(example, format=fmt)
    print(f"{fmt}: {parsed}")
```

---

## Date Arithmetic

### Basic Operations

```python
from datetime import timedelta

# Add/subtract days
date1 = datetime(2024, 1, 15)
date2 = date1 + timedelta(days=30)
date3 = date1 - timedelta(days=10)

print(f"Original: {date1}")
print(f"Plus 30 days: {date2}")
print(f"Minus 10 days: {date3}")

# Add/subtract with Pandas
ts = pd.Timestamp('2024-01-15')
ts_plus_month = ts + pd.DateOffset(months=1)
ts_plus_year = ts + pd.DateOffset(years=1)

print(f"Plus 1 month: {ts_plus_month}")
print(f"Plus 1 year: {ts_plus_year}")
```

### Date Differences

```python
# Calculate difference
date1 = datetime(2024, 1, 15)
date2 = datetime(2024, 2, 20)
diff = date2 - date1
print(f"Difference: {diff.days} days")

# With Pandas
ts1 = pd.Timestamp('2024-01-15')
ts2 = pd.Timestamp('2024-02-20')
diff = ts2 - ts1
print(f"Difference: {diff.days} days")

# Series of dates
dates = pd.Series(pd.date_range('2024-01-01', periods=10, freq='D'))
diffs = dates.diff()  # Difference between consecutive dates
print(diffs)
```

---

## Time Series Indexing

### Setting Date Index

```python
# Create DataFrame with date index
df = pd.DataFrame({
    'value': range(10),
    'date': pd.date_range('2024-01-01', periods=10, freq='D')
})

# Set date as index
df = df.set_index('date')
print(df)

# Slicing by date
print(df['2024-01-05'])  # Single date
print(df['2024-01-01':'2024-01-05'])  # Date range
print(df['2024-01'])  # All January
print(df['2024'])  # All 2024
```

### Date Range Selection

```python
# Using loc with date ranges
df.loc['2024-01-01':'2024-01-05']

# Using query
df.query("index >= '2024-01-01' and index <= '2024-01-05'")

# Using between
df[df.index.to_series().between('2024-01-01', '2024-01-05')]
```

---

## Date Components

### Extracting Components

```python
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10, freq='D')
})

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['date'].dt.day_name()
df['month_name'] = df['date'].dt.month_name()
df['quarter'] = df['date'].dt.quarter
df['week'] = df['date'].dt.isocalendar().week
df['is_weekend'] = df['date'].dt.dayofweek >= 5

print(df)
```

### Time Components

```python
df = pd.DataFrame({
    'datetime': pd.date_range('2024-01-01 10:30:00', periods=10, freq='H')
})

# Extract time components
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['second'] = df['datetime'].dt.second
df['time'] = df['datetime'].dt.time

print(df)
```

---

## Time Zones

### Working with Time Zones

```python
# Create timezone-aware datetime
dt_utc = pd.Timestamp('2024-01-15 10:00:00', tz='UTC')
print(dt_utc)

# Convert timezone
dt_est = dt_utc.tz_convert('US/Eastern')
print(dt_est)

# Localize (add timezone to naive datetime)
dt_naive = pd.Timestamp('2024-01-15 10:00:00')
dt_localized = dt_naive.tz_localize('UTC')
print(dt_localized)

# Series with timezone
dates = pd.date_range('2024-01-01', periods=10, freq='D', tz='UTC')
df = pd.DataFrame({'value': range(10)}, index=dates)
df_est = df.tz_convert('US/Eastern')
print(df_est)
```

---

## Resampling and Frequency

### Resampling Time Series

```python
# Create sample time series
dates = pd.date_range('2024-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'value': np.random.randn(100).cumsum()
}, index=dates)

# Resample to weekly
df_weekly = df.resample('W').mean()
print(df_weekly.head())

# Resample to monthly
df_monthly = df.resample('M').sum()
print(df_monthly.head())

# Different aggregations
df_resampled = df.resample('W').agg({
    'value': ['mean', 'sum', 'min', 'max']
})
print(df_resampled.head())
```

### Common Frequencies

```python
frequencies = {
    'D': 'Daily',
    'W': 'Weekly',
    'M': 'Monthly',
    'Q': 'Quarterly',
    'Y': 'Yearly',
    'H': 'Hourly',
    'T': 'Minutely',
    'S': 'Secondly'
}

# Examples
for freq, name in frequencies.items():
    dates = pd.date_range('2024-01-01', periods=5, freq=freq)
    print(f"{name} ({freq}): {dates}")
```

### Upsampling and Interpolation

```python
# Upsample (increase frequency)
df_daily = pd.DataFrame({
    'value': [100, 120, 130, 140]
}, index=pd.date_range('2024-01-01', periods=4, freq='D'))

# Upsample to hourly
df_hourly = df_daily.resample('H').asfreq()  # Forward fill
df_hourly_interp = df_daily.resample('H').interpolate()  # Interpolate

print("Forward fill:")
print(df_hourly.head())
print("\nInterpolated:")
print(df_hourly_interp.head())
```

---

## Advanced Operations

### Rolling Windows

```python
# Create time series
dates = pd.date_range('2024-01-01', periods=30, freq='D')
df = pd.DataFrame({
    'value': np.random.randn(30).cumsum()
}, index=dates)

# Rolling mean (7-day window)
df['rolling_mean_7'] = df['value'].rolling(window=7).mean()

# Rolling sum
df['rolling_sum_7'] = df['value'].rolling(window=7).sum()

# Expanding window
df['expanding_mean'] = df['value'].expanding().mean()

print(df.head(10))
```

### Shifting and Lagging

```python
# Shift values (lag)
df['value_lag1'] = df['value'].shift(1)  # Previous day
df['value_lag7'] = df['value'].shift(7)  # 7 days ago

# Lead (future values)
df['value_lead1'] = df['value'].shift(-1)  # Next day

# Percentage change
df['pct_change'] = df['value'].pct_change()
df['pct_change_7d'] = df['value'].pct_change(periods=7)

print(df.head(10))
```

### Business Days

```python
# Business day range
bday_range = pd.bdate_range('2024-01-01', '2024-01-31')
print(f"Business days: {len(bday_range)}")

# Check if business day
dates = pd.date_range('2024-01-01', periods=10, freq='D')
is_bday = dates.isin(pd.bdate_range(dates[0], dates[-1]))
print(is_bday)
```

---

## Practice Exercises

### Exercise 1: Date Parsing

Parse dates from various formats and extract components.

```python
dates_str = [
    '2024-01-15',
    '01/15/2024',
    'Jan 15, 2024',
    '15-01-2024'
]

# Your solution here
```

### Exercise 2: Time Series Analysis

Create a time series, resample it, and calculate rolling statistics.

```python
# Your solution here
```

### Exercise 3: Date Filtering

Filter data based on date ranges and extract specific periods.

```python
# Your solution here
```

---

## Resources

### Documentation

- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Python datetime](https://docs.python.org/3/library/datetime.html)

### Cheat Sheets

- [Pandas Time Series Cheat Sheet](https://pandas.pydata.org/docs/user_guide/timeseries.html)

---

## Key Takeaways

1. **Use Pandas**: Powerful time series functionality
2. **Set Date Index**: Makes time-based operations easier
3. **Resample**: Change frequency of time series
4. **Extract Components**: Year, month, day, etc.
5. **Handle Time Zones**: Important for global data

---

**Remember**: Date/time handling is crucial for time series analysis. Master these operations for effective data analysis!

