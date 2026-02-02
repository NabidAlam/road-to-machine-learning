# Project 5: Wine Quality Prediction

Predict wine quality based on chemical properties. Can be approached as both regression and classification.

## Difficulty
Beginner

## Time Estimate
2-3 days

## Skills You'll Practice
- Regression/Classification
- Feature Selection
- Handling Imbalanced Data
- Model Tuning

## Learning Objectives

By completing this project, you will learn to:
- Work with chemical/analytical data
- Handle imbalanced datasets
- Apply both regression and classification
- Select important features
- Tune model hyperparameters

## Dataset

**UCI Wine Quality Dataset**
- [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Two datasets: red wine and white wine
- Chemical properties as features
- Quality score (0-10) as target

**Features:**
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol

**Target:**
- Quality (0-10 scale)

## Project Steps

### Step 1: Load and Explore Data
- Load red and white wine datasets
- Check data shape and basic statistics
- Analyze quality distribution
- Check for missing values
- Explore feature distributions

### Step 2: Data Preprocessing
- Handle missing values (if any)
- Check for outliers
- Feature scaling/normalization
- Create binary classification (good/bad wine)

### Step 3: Feature Analysis
- Correlation analysis
- Feature importance
- Visualize relationships
- Identify key features

### Step 4: Approach 1 - Regression
- Predict quality score (0-10)
- Train regression models:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regressor
- Evaluate using RMSE, MAE, R²

### Step 5: Approach 2 - Classification
- Convert quality to binary (good ≥ 7, bad < 7)
- Handle class imbalance
- Train classification models:
  - Logistic Regression
  - Random Forest
  - SVM
- Evaluate using accuracy, precision, recall, F1

### Step 6: Model Comparison
- Compare regression vs classification approaches
- Analyze feature importance
- Select best model
- Final evaluation

## Expected Deliverables

1. **Jupyter Notebook** with complete analysis:
   - EDA with visualizations
   - Both regression and classification approaches
   - Model comparison
   - Results and conclusions

2. **Analysis Report**:
   - Which chemical properties matter most?
   - Can we predict wine quality accurately?
   - Comparison of approaches

## Evaluation Metrics

**For Regression:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**For Classification:**
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix

## Key Insights to Explore

- Which chemical properties correlate with quality?
- Is there a difference between red and white wine?
- Can we predict quality from chemical properties alone?
- Which features are most important?

## Tips

- Try both regression and classification
- Quality scores are imbalanced (few high scores)
- Feature scaling is important
- Visualize correlations
- Compare red vs white wine models
- Try combining both datasets

## Resources

- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- [Wine Quality on Kaggle](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset)
- [Scikit-learn Regression](https://scikit-learn.org/stable/supervised_learning.html#regression)

## Next Steps

After completing this project:
- Try advanced feature engineering
- Experiment with ensemble methods
- Move to [Intermediate Projects](../../17-projects-intermediate/README.md)

