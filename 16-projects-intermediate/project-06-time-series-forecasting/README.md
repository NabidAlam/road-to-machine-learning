# Project 6: Time Series Forecasting

Forecast future values in a time series (sales, stock prices, etc.).

## Difficulty
Intermediate

## Time Estimate
4-5 days

## Skills You'll Practice
- Time Series Analysis
- ARIMA Models
- LSTM Networks
- Feature Engineering for Time Series

## Learning Objectives

By completing this project, you will learn to:
- Preprocess time series data
- Identify trends and seasonality
- Apply ARIMA models
- Use LSTM for sequences
- Engineer time-based features
- Evaluate time series models

## Dataset

**Airline Passengers Dataset**
- [Kaggle Airline Passengers](https://www.kaggle.com/datasets/rakannimer/air-passengers)
- Monthly passenger numbers
- Clear trend and seasonality
- Good for learning

## Project Steps

### Step 1: Load and Explore Data
- Load time series data
- Visualize time series
- Check for missing values
- Identify trends and seasonality

### Step 2: Time Series Decomposition
- Decompose into trend, seasonality, residual
- Visualize components
- Understand patterns

### Step 3: Stationarity
- Check for stationarity (ADF test)
- Apply differencing if needed
- Make series stationary

### Step 4: ARIMA Model
- Identify ARIMA parameters (p, d, q)
- Use auto_arima or manual selection
- Train ARIMA model
- Forecast future values

### Step 5: LSTM Model
- Prepare data for LSTM
- Create sequences
- Build LSTM network
- Train and forecast

### Step 6: Feature Engineering
- Create lag features
- Add time-based features (month, day, etc.)
- Create rolling statistics
- Use external features if available

### Step 7: Model Evaluation
- Split into train/test (time-based!)
- Calculate RMSE, MAE
- Visualize predictions
- Compare ARIMA vs LSTM

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis
2. **Forecasted Values** for future periods
3. **Visualizations** of predictions vs actual
4. **Comparison Report** of different methods

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **Visual Inspection**: Plot predictions

## Models to Implement

1. **ARIMA**: Classical time series
2. **LSTM**: Deep learning approach
3. **Prophet**: Facebook's tool (optional)
4. **Simple Methods**: Moving average, exponential smoothing

## Tips

- Use time-based train/test split (not random!)
- Check for stationarity
- Handle seasonality properly
- Create lag features
- Visualize everything
- Compare multiple approaches

## Resources

- [Kaggle Time Series](https://www.kaggle.com/datasets/rakannimer/air-passengers)
- [ARIMA Guide](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
- [LSTM for Time Series](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)

## Extensions

- Multiple time series
- External features
- Real-time forecasting
- Confidence intervals
- Anomaly detection in time series

## Next Steps

After completing this project:
- Try more complex time series
- Experiment with advanced techniques
- Move to [Advanced Projects](../17-projects-advanced/README.md)

