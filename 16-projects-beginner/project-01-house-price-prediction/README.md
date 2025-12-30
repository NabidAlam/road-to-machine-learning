# Project 1: House Price Prediction

Predict house prices using features like size, location, number of rooms, etc.

## Difficulty
Beginner

## Time Estimate
2-3 days

## Skills You'll Practice
- Regression
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Data Cleaning

## Learning Objectives

By completing this project, you will learn to:
- Clean and preprocess real-world data
- Perform exploratory data analysis
- Engineer features from raw data
- Apply linear regression and regularization
- Evaluate regression models using multiple metrics
- Handle missing values and outliers

## Dataset

**Option 1: California Housing Dataset (Built-in)**
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
```

**Option 2: Kaggle House Prices Competition**
- [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- More features and real-world complexity

## Project Steps

### Step 1: Load and Explore Data
- Load the dataset
- Check data shape and basic statistics
- Identify missing values
- Explore feature distributions
- Check for outliers

### Step 2: Data Preprocessing
- Handle missing values
- Remove or transform outliers
- Encode categorical variables (if using Kaggle dataset)
- Feature scaling/normalization

### Step 3: Feature Engineering
- Create new features (e.g., total rooms per household, bedrooms per room)
- Handle skewed distributions (log transformation)
- Feature selection

### Step 4: Model Training
- Split data into train/test sets
- Train Linear Regression model
- Try Ridge and Lasso regression
- Compare model performance

### Step 5: Model Evaluation
- Calculate RMSE, MAE, R²
- Visualize predictions vs actual values
- Analyze residuals
- Identify areas for improvement

### Step 6: Model Improvement
- Tune hyperparameters
- Try polynomial features
- Feature selection
- Final model selection

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis:
   - EDA with visualizations
   - Data preprocessing steps
   - Model training and evaluation
   - Results and conclusions

2. **Python Script** (optional):
   - Clean, well-commented code
   - Can be run independently

3. **Results Summary**:
   - Best model performance metrics
   - Key insights from EDA
   - Feature importance

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Primary metric
- **MAE** (Mean Absolute Error): Average prediction error
- **R² Score**: Proportion of variance explained

## Tips

- Start with the California Housing dataset (simpler)
- Visualize everything: distributions, correlations, residuals
- Try different feature combinations
- Don't forget to scale features for regularization
- Compare multiple models before choosing the best one

## Resources

- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [California Housing Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Next Steps

After completing this project:
- Try the Kaggle competition for more challenge
- Experiment with ensemble methods
- Move to [Project 2: Iris Classification](../project-02-iris-classification/README.md) for classification practice

