# Project 8: Ensemble Methods Comparison Project

Comprehensive comparison of ensemble methods: Bagging, Boosting, Stacking, and Voting.

## Difficulty
Intermediate

## Time Estimate
4-5 days

## Skills You'll Practice
- Ensemble Methods
- Bagging (Random Forest)
- Boosting (XGBoost, LightGBM, CatBoost)
- Stacking
- Voting Classifiers
- Hyperparameter Tuning
- Model Comparison

## Learning Objectives

By completing this project, you will learn to:
- Understand different ensemble methods and when to use them
- Implement and compare Bagging vs Boosting
- Use advanced boosting algorithms (XGBoost, LightGBM, CatBoost)
- Build Stacking and Voting ensembles
- Tune hyperparameters for ensemble methods
- Compare model performance comprehensively
- Understand trade-offs between different approaches
- Select the best ensemble for your problem

## Prerequisites

Before starting, you should have completed:
- Phase 2: Machine Learning Basics
- Phase 5: Model Evaluation & Optimization
- Phase 6: Ensemble Methods (all topics)
- Understanding of cross-validation and hyperparameter tuning

## Dataset

**Recommended Datasets:**

1. **Titanic Dataset** (Good for comparison)
   - [Kaggle Titanic](https://www.kaggle.com/c/titanic)
   - Medium size, mixed features
   - Classic benchmark dataset

2. **Credit Card Fraud Detection**
   - [Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Highly imbalanced
   - Tests ensemble robustness

3. **Customer Churn Prediction**
   - [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Business problem
   - Mixed data types

4. **House Prices** (for regression)
   - [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - Regression problem
   - Many features

## Project Steps

### Step 1: Data Preparation
- Load and explore dataset
- Handle missing values
- Encode categorical variables
- Split into train/validation/test sets
- Create baseline model (single decision tree or logistic regression)

### Step 2: Bagging Methods
- Implement Random Forest
- Tune hyperparameters (n_estimators, max_depth, min_samples_split)
- Evaluate performance
- Analyze feature importance
- Compare with baseline

### Step 3: Boosting Methods - AdaBoost
- Implement AdaBoost
- Understand how it works (re-weighting)
- Tune hyperparameters
- Compare with bagging

### Step 4: Advanced Boosting - XGBoost
- Implement XGBoost
- Tune key hyperparameters:
  - learning_rate, n_estimators
  - max_depth, min_child_weight
  - gamma, reg_alpha, reg_lambda
  - subsample, colsample_bytree
- Use early stopping
- Compare performance

### Step 5: Advanced Boosting - LightGBM
- Implement LightGBM
- Understand GOSS and EFB
- Tune hyperparameters
- Compare with XGBoost

### Step 6: Advanced Boosting - CatBoost
- Implement CatBoost
- Understand ordered boosting
- Handle categorical features automatically
- Compare with other boosters

### Step 7: Stacking
- Build base models (diverse set)
- Create meta-model
- Use cross-validation for meta-features
- Compare stacking with individual models

### Step 8: Voting Classifiers
- Implement Hard Voting
- Implement Soft Voting
- Compare voting with other ensembles

### Step 9: Comprehensive Comparison
- Create comparison table:
  - Accuracy/Score
  - Training time
  - Prediction time
  - Model complexity
  - Interpretability
- Visualize results
- Make recommendations

### Step 10: Final Model Selection
- Select best ensemble method
- Fine-tune selected model
- Evaluate on test set
- Document findings

## Code Structure

```
project-08-ensemble-comparison/
├── README.md
├── notebooks/
│   ├── 01-data-preparation.ipynb
│   ├── 02-bagging-methods.ipynb
│   ├── 03-boosting-adaboost.ipynb
│   ├── 04-boosting-xgboost.ipynb
│   ├── 05-boosting-lightgbm.ipynb
│   ├── 06-boosting-catboost.ipynb
│   ├── 07-stacking.ipynb
│   ├── 08-voting.ipynb
│   └── 09-comparison.ipynb
├── src/
│   ├── bagging_models.py
│   ├── boosting_models.py
│   ├── stacking.py
│   ├── voting.py
│   └── comparison.py
├── data/
├── models/
└── requirements.txt
```

## Implementation Examples

### 1. Random Forest (Bagging)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
```

### 2. XGBoost (Boosting)
```python
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

xgb_model = xgb.XGBClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=50, cv=5)
xgb_search.fit(X_train, y_train)
```

### 3. LightGBM
```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
```

### 4. CatBoost
```python
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train, cat_features=categorical_indices)
```

### 5. Stacking
```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

base_models = [
    ('rf', RandomForestClassifier(n_estimators=100)),
    ('xgb', xgb.XGBClassifier()),
    ('lgb', lgb.LGBMClassifier())
]

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_model.fit(X_train, y_train)
```

### 6. Voting
```python
from sklearn.ensemble import VotingClassifier

voting_hard = VotingClassifier(
    estimators=base_models,
    voting='hard'
)

voting_soft = VotingClassifier(
    estimators=base_models,
    voting='soft'
)
```

## Comparison Framework

Create a comprehensive comparison:

```python
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'Training Time': [],
    'Prediction Time': [],
    'Model Size': []
}

# Evaluate each model
for name, model in models.items():
    # Train and evaluate
    # Add results to dictionary

# Create comparison DataFrame
comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
```

## Evaluation Criteria

Your comparison should:
- ✅ Implement all major ensemble methods
- ✅ Tune hyperparameters appropriately
- ✅ Use proper cross-validation
- ✅ Compare on multiple metrics
- ✅ Include training/prediction time
- ✅ Visualize comparisons clearly
- ✅ Provide recommendations
- ✅ Document trade-offs

## Metrics to Compare

1. **Performance Metrics**
   - Accuracy/Score
   - Precision, Recall, F1-Score
   - ROC-AUC (for classification)
   - RMSE, MAE (for regression)

2. **Efficiency Metrics**
   - Training time
   - Prediction time
   - Memory usage

3. **Model Characteristics**
   - Interpretability
   - Robustness to overfitting
   - Handling of missing values
   - Categorical feature support

## Extensions

1. **Custom Ensemble**
   - Create your own ensemble method
   - Combine different approaches
   - Experiment with weights

2. **Ensemble of Ensembles**
   - Stack different ensemble types
   - Create multi-level stacking
   - Compare with single-level

3. **Feature Importance Comparison**
   - Compare feature importance across methods
   - Visualize differences
   - Understand model differences

4. **Hyperparameter Sensitivity**
   - Analyze sensitivity to hyperparameters
   - Create hyperparameter importance plots
   - Find robust configurations

## Resources

- [Ensemble Methods Guide](../../06-ensemble-methods/ensemble-methods.md)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)

## Tips for Success

1. **Start Simple**: Begin with basic ensembles, then add complexity
2. **Use Cross-Validation**: Always use CV for fair comparison
3. **Tune Systematically**: Use GridSearch or RandomizedSearch
4. **Document Everything**: Keep track of all results
5. **Visualize**: Create clear comparison charts
6. **Think Trade-offs**: Consider time, accuracy, interpretability
7. **Validate**: Always test on held-out test set

## Common Pitfalls to Avoid

- ❌ Overfitting on validation set
- ❌ Not using proper cross-validation
- ❌ Comparing untuned models
- ❌ Ignoring training time
- ❌ Not considering interpretability needs
- ❌ Using same base models for stacking

## Next Steps

After completing this project:
- Apply best ensemble to real problems
- Experiment with custom ensembles
- Learn about neural network ensembles
- Move to advanced projects requiring ensembles

---

**Ready to master ensembles?** Start with a baseline model and systematically compare each ensemble method!

