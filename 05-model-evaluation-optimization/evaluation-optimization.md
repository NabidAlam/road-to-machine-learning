# Model Evaluation & Optimization Complete Guide

Comprehensive guide to properly evaluating models and optimizing their performance.

## Table of Contents

- [Data Splitting](#data-splitting)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Learning Curves](#learning-curves)
- [Model Calibration](#model-calibration)
- [Practice Exercises](#practice-exercises)

---

## Data Splitting

### Train/Validation/Test Split

**Three Sets:**
1. **Training Set**: Train the model
2. **Validation Set**: Tune hyperparameters
3. **Test Set**: Final evaluation (only touched once!)

```python
from sklearn.model_selection import train_test_split

# First split: Train + (Validation + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Second split: Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Training: {X_train.shape[0]} samples")
print(f"Validation: {X_val.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# Typical split: 60% train, 20% validation, 20% test
```

### Stratified Split

Maintains class distribution in each split (important for imbalanced data).

```python
from sklearn.model_selection import train_test_split

# Stratified split (for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Verify distribution
print("Original distribution:")
print(pd.Series(y).value_counts(normalize=True))
print("\nTrain distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\nTest distribution:")
print(pd.Series(y_test).value_counts(normalize=True))
```

### Why Three Sets?

- **Training**: Model learns from this
- **Validation**: Tune hyperparameters (model selection)
- **Test**: Final unbiased evaluation

**Never use test set for tuning!**

### Common Split Ratios

```python
# Small dataset (< 10K samples)
# 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Medium dataset (10K - 100K samples)
# 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Large dataset (> 100K samples)
# 98% train, 1% validation, 1% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.02, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
```

---

## Cross-Validation

### K-Fold Cross-Validation

Divides data into k folds, trains on k-1, validates on 1, repeats k times.

**Advantages:**
- More reliable performance estimate
- Uses all data for both training and validation
- Reduces variance in performance estimate

**Disadvantages:**
- Computationally expensive (k times training)
- Not suitable for time series data

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# K-Fold CV
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Calculate confidence interval (95%)
mean_score = scores.mean()
std_score = scores.std()
n_splits = len(scores)
confidence_interval = 1.96 * std_score / np.sqrt(n_splits)
print(f"95% Confidence Interval: [{mean_score - confidence_interval:.3f}, "
      f"{mean_score + confidence_interval:.3f}]")
```

**Visualization:**
```python
# Visualize folds
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
    axes[fold].scatter(X[train_idx, 0], X[train_idx, 1], 
                      c='blue', alpha=0.3, label='Train')
    axes[fold].scatter(X[val_idx, 0], X[val_idx, 1], 
                      c='red', alpha=0.7, label='Validation')
    axes[fold].set_title(f'Fold {fold+1}')
    axes[fold].legend()

plt.tight_layout()
plt.show()
```

### Stratified K-Fold

Maintains class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"Stratified CV Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Leave-One-Out Cross-Validation

Extreme case: k = n (each sample is a fold). Very slow but uses all data.

```python
from sklearn.model_selection import LeaveOneOut

# Leave-One-Out (slow for large datasets!)
loo = LeaveOneOut()
scores = cross_val_score(model, X[:100], y[:100], cv=loo, scoring='accuracy')
print(f"LOO CV Mean: {scores.mean():.3f}")
```

### Time Series Cross-Validation

For time-dependent data, respect temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit

# Time series split
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    print(f"Train: {train_idx[0]} to {train_idx[-1]}, "
          f"Test: {test_idx[0]} to {test_idx[-1]}")
```

---

## Hyperparameter Tuning

### Grid Search

Exhaustive search over parameter grid. Tests all combinations of parameters.

**When to use:**
- Small parameter space (< 1000 combinations)
- Need to find exact best parameters
- Computational resources available

**Disadvantages:**
- Computationally expensive
- Doesn't scale well with many parameters

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Create model
model = RandomForestClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1,  # Show progress
    return_train_score=True  # Include train scores
)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# View all results
results_df = pd.DataFrame(grid_search.cv_results_)
print("\nTop 5 parameter combinations:")
print(results_df[['params', 'mean_test_score', 'std_test_score']]
      .sort_values('mean_test_score', ascending=False).head())

# Use best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Random Search

Random sampling of parameters (faster than grid search). Often finds good solutions with fewer iterations.

**When to use:**
- Large parameter space
- Limited computational resources
- Want quick results

**Advantages:**
- Faster than grid search
- Can explore wider parameter ranges
- Often finds good solutions quickly

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

# Random search
random_search = RandomizedSearchCV(
    model, 
    param_dist, 
    n_iter=50,  # Number of iterations
    cv=5, 
    scoring='accuracy', 
    random_state=42, 
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

# Compare with grid search
print(f"\nRandom Search found score: {random_search.best_score_:.3f}")
print(f"Grid Search found score: {grid_search.best_score_:.3f}")
print(f"Time saved: Random search is typically 5-10x faster")
```

### Bayesian Optimization (Optuna)

Smart search using previous results.

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Define objective function for Optuna"""
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
print("Best score:", study.best_value)
```

---

## Bias-Variance Tradeoff

### Understanding Bias and Variance

**Bias**: Error from oversimplifying assumptions (underfitting)
**Variance**: Error from sensitivity to small fluctuations (overfitting)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# High bias model (simple)
high_bias = LogisticRegression()
high_bias.fit(X_train, y_train)
train_score_bias = high_bias.score(X_train, y_train)
test_score_bias = high_bias.score(X_test, y_test)

# High variance model (complex)
high_variance = DecisionTreeClassifier(max_depth=20)
high_variance.fit(X_train, y_train)
train_score_var = high_variance.score(X_train, y_train)
test_score_var = high_variance.score(X_test, y_test)

print("High Bias Model:")
print(f"  Train: {train_score_bias:.3f}, Test: {test_score_bias:.3f}")
print(f"  Gap: {train_score_bias - test_score_bias:.3f}")

print("\nHigh Variance Model:")
print(f"  Train: {train_score_var:.3f}, Test: {test_score_var:.3f}")
print(f"  Gap: {train_score_var - test_score_var:.3f}")
```

### Finding the Balance

```python
# Test different model complexities
max_depths = range(1, 21)
train_scores = []
test_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, 'o-', label='Train')
plt.plot(max_depths, test_scores, 's-', label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Bias-Variance Tradeoff')
plt.legend()
plt.grid(True)
plt.show()

# Find optimal depth
optimal_depth = max_depths[np.argmax(test_scores)]
print(f"Optimal depth: {optimal_depth}")
```

---

## Learning Curves

### Plotting Learning Curves

Shows how model performance changes with training data size.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title):
    """Plot learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    plt.fill_between(train_sizes, train_mean - train_std, 
                     train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
    plt.fill_between(train_sizes, val_mean - val_std, 
                     val_mean + val_std, alpha=0.1, color='red')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot for different models
plot_learning_curve(LogisticRegression(), X, y, 'Logistic Regression')
plot_learning_curve(RandomForestClassifier(), X, y, 'Random Forest')
```

### Interpreting Learning Curves

**Underfitting:**
- Both curves converge to low performance
- Small gap between train and validation
- **Solution**: More complex model, better features

**Overfitting:**
- Large gap between train and validation
- Train performance much higher
- **Solution**: Regularization, more data, simpler model

**Good Fit:**
- Both curves converge to high performance
- Small gap between train and validation

---

## Practice Exercises

### Exercise 1: Cross-Validation Comparison

**Task:** Compare 5-fold, 10-fold, and stratified 5-fold CV.

**Solution:**
```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

model = RandomForestClassifier(n_estimators=100)

# 5-fold
kfold5 = KFold(n_splits=5, shuffle=True, random_state=42)
scores_5 = cross_val_score(model, X, y, cv=kfold5, scoring='accuracy')

# 10-fold
kfold10 = KFold(n_splits=10, shuffle=True, random_state=42)
scores_10 = cross_val_score(model, X, y, cv=kfold10, scoring='accuracy')

# Stratified 5-fold
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"5-Fold CV: {scores_5.mean():.3f} (+/- {scores_5.std():.3f})")
print(f"10-Fold CV: {scores_10.mean():.3f} (+/- {scores_10.std():.3f})")
print(f"Stratified 5-Fold: {scores_strat.mean():.3f} (+/- {scores_strat.std():.3f})")
```

### Exercise 2: Hyperparameter Tuning

**Task:** Tune Random Forest hyperparameters using Grid Search and compare with Random Search.

**Solution:**
```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
import time

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

# Grid Search
start_time = time.time()
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print("Grid Search Results:")
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
print(f"Time taken: {grid_time:.2f} seconds")

# Random Search
param_dist = {
    'n_estimators': randint(50, 201),
    'max_depth': [5, 10, None],
    'min_samples_split': randint(2, 11)
}

start_time = time.time()
random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=27,  # Same number of combinations as grid search
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

print("\nRandom Search Results:")
print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)
print(f"Time taken: {random_time:.2f} seconds")

# Evaluate on test set
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"\nTest score: {test_score:.3f}")
```

### Exercise 3: Learning Curve Analysis

**Task:** Plot learning curves for different models and diagnose issues.

**Solution:**
```python
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
    'Decision Tree (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, model) in enumerate(models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
    axes[idx].fill_between(train_sizes, train_mean - train_std, 
                          train_mean + train_std, alpha=0.1, color='blue')
    axes[idx].plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
    axes[idx].fill_between(train_sizes, val_mean - val_std, 
                          val_mean + val_std, alpha=0.1, color='red')
    axes[idx].set_xlabel('Training Set Size')
    axes[idx].set_ylabel('Accuracy')
    axes[idx].set_title(name)
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

# Diagnose issues
print("Diagnosis:")
print("- Logistic Regression: Check for underfitting (both curves low)")
print("- Decision Tree (depth=3): Check for good fit (curves converge)")
print("- Decision Tree (depth=10): Check for overfitting (large gap)")
```

---

## Evaluation Metrics Summary

### For Classification

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)

# Binary classification
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
```

### For Regression

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                            r2_score)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")
```

### Choosing the Right Metric

**Classification:**
- **Balanced data**: Accuracy
- **Imbalanced data**: F1-score, ROC-AUC, Precision-Recall
- **Cost-sensitive**: Consider precision or recall based on business needs

**Regression:**
- **Outliers matter**: RMSE
- **Outliers don't matter**: MAE
- **Interpretability**: R² (proportion of variance explained)

---

## Model Calibration

### What is Model Calibration?

**Model calibration** refers to how well a model's predicted probabilities match the actual observed frequencies. A well-calibrated model means that when it predicts a 70% probability, the outcome should be positive about 70% of the time.

**Example:**
- **Well-calibrated**: If model predicts 0.8 probability for 100 samples, ~80 should be positive
- **Poorly calibrated**: If model predicts 0.8 probability for 100 samples, but only 50 are positive (overconfident)

### Why Calibration Matters

1. **Decision Making**: When probabilities are used for business decisions (e.g., risk assessment)
2. **Cost-Benefit Analysis**: Need accurate probabilities to calculate expected values
3. **Threshold Selection**: Calibrated probabilities help choose optimal decision thresholds
4. **Trust**: Stakeholders need to trust probability estimates

### Calibration Curves

A calibration curve plots predicted probabilities against observed frequencies. A perfectly calibrated model would follow the diagonal line.

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Get predicted probabilities
rf_proba = rf.predict_proba(X_test)[:, 1]
lr_proba = lr.predict_proba(X_test)[:, 1]

# Calculate calibration curves
rf_fraction_of_positives, rf_mean_predicted_value = calibration_curve(
    y_test, rf_proba, n_bins=10
)
lr_fraction_of_positives, lr_mean_predicted_value = calibration_curve(
    y_test, lr_proba, n_bins=10
)

# Plot calibration curves
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.plot(rf_mean_predicted_value, rf_fraction_of_positives, 
         's-', label='Random Forest (uncalibrated)')
plt.plot(lr_mean_predicted_value, lr_fraction_of_positives, 
         'o-', label='Logistic Regression')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.title('Calibration Curves', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate Brier score (lower is better, 0 = perfect)
from sklearn.metrics import brier_score_loss

rf_brier = brier_score_loss(y_test, rf_proba)
lr_brier = brier_score_loss(y_test, lr_proba)

print(f"Brier Score (lower is better):")
print(f"  Random Forest: {rf_brier:.4f}")
print(f"  Logistic Regression: {lr_brier:.4f}")
```

### Calibration Methods

#### 1. Platt Scaling (Sigmoid Calibration)

Uses logistic regression to map uncalibrated probabilities to calibrated probabilities. Best for models that are already somewhat well-calibrated.

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, calibration_curve
import matplotlib.pyplot as plt

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train uncalibrated SVM (SVM probabilities are often poorly calibrated)
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# Calibrate using Platt scaling
svm_calibrated = CalibratedClassifierCV(svm, method='sigmoid', cv=3)
svm_calibrated.fit(X_train, y_train)

# Get probabilities
svm_proba = svm.predict_proba(X_test)[:, 1]
svm_calibrated_proba = svm_calibrated.predict_proba(X_test)[:, 1]

# Compare Brier scores
print("Brier Score (lower is better):")
print(f"  Uncalibrated SVM: {brier_score_loss(y_test, svm_proba):.4f}")
print(f"  Calibrated SVM (Platt): {brier_score_loss(y_test, svm_calibrated_proba):.4f}")

# Plot calibration curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Uncalibrated
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, svm_proba, n_bins=10
)
ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label='SVM (uncalibrated)')
ax1.set_xlabel('Mean Predicted Probability')
ax1.set_ylabel('Fraction of Positives')
ax1.set_title('Before Calibration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Calibrated
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, svm_calibrated_proba, n_bins=10
)
ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax2.plot(mean_predicted_value, fraction_of_positives, 's-', label='SVM (Platt calibrated)')
ax2.set_xlabel('Mean Predicted Probability')
ax2.set_ylabel('Fraction of Positives')
ax2.set_title('After Platt Calibration')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 2. Isotonic Regression

Uses a non-parametric approach (piecewise constant function) to map probabilities. More flexible than Platt scaling, can handle any monotonic relationship.

```python
# Calibrate using Isotonic Regression
svm_calibrated_isotonic = CalibratedClassifierCV(svm, method='isotonic', cv=3)
svm_calibrated_isotonic.fit(X_train, y_train)

svm_calibrated_isotonic_proba = svm_calibrated_isotonic.predict_proba(X_test)[:, 1]

print("Brier Score Comparison:")
print(f"  Uncalibrated SVM: {brier_score_loss(y_test, svm_proba):.4f}")
print(f"  Platt Scaling: {brier_score_loss(y_test, svm_calibrated_proba):.4f}")
print(f"  Isotonic Regression: {brier_score_loss(y_test, svm_calibrated_isotonic_proba):.4f}")

# Plot all three
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, svm_calibrated_isotonic_proba, n_bins=10
)
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Isotonic Regression')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Isotonic Regression Calibration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### When to Calibrate

**Models that often need calibration:**
- **Random Forest**: Often overconfident (probabilities too extreme)
- **SVM**: Probabilities can be poorly calibrated
- **Neural Networks**: Can be overconfident
- **Gradient Boosting**: Often needs calibration

**Models that are usually well-calibrated:**
- **Logistic Regression**: Usually well-calibrated by design
- **Naive Bayes**: Often well-calibrated
- **Linear models**: Generally well-calibrated

### Complete Calibration Workflow

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train base model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Calibrate model
rf_calibrated = CalibratedClassifierCV(
    rf, 
    method='isotonic',  # or 'sigmoid' for Platt scaling
    cv=3  # Use 3-fold cross-validation for calibration
)
rf_calibrated.fit(X_train, y_train)

# Evaluate both models
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_calibrated_proba = rf_calibrated.predict_proba(X_test)[:, 1]

print("Model Evaluation:")
print("=" * 50)
print(f"Brier Score (lower is better):")
print(f"  Uncalibrated: {brier_score_loss(y_test, rf_proba):.4f}")
print(f"  Calibrated: {brier_score_loss(y_test, rf_calibrated_proba):.4f}")
print(f"\nLog Loss (lower is better):")
print(f"  Uncalibrated: {log_loss(y_test, rf_proba):.4f}")
print(f"  Calibrated: {log_loss(y_test, rf_calibrated_proba):.4f}")
print(f"\nROC-AUC (higher is better):")
print(f"  Uncalibrated: {roc_auc_score(y_test, rf_proba):.4f}")
print(f"  Calibrated: {roc_auc_score(y_test, rf_calibrated_proba):.4f}")
```

### Key Points

1. **Calibration doesn't change ranking**: Well-calibrated probabilities don't necessarily improve ranking (ROC-AUC), but they improve probability estimates
2. **Use cross-validation**: Always use `CalibratedClassifierCV` with cross-validation to avoid overfitting
3. **Choose method**: Platt scaling for smooth calibration, Isotonic for more flexibility
4. **Evaluate with Brier Score**: Lower Brier score = better calibration
5. **Production importance**: Critical when probabilities are used for decision-making

### Common Issues

**Overconfident Models:**
- Predictions too extreme (e.g., always 0.9 or 0.1)
- Common with Random Forest, Neural Networks
- Solution: Calibration brings probabilities closer to 0.5

**Underconfident Models:**
- Predictions too conservative (e.g., always around 0.5)
- Less common but can occur
- Solution: Calibration can help, but may indicate model issues

### Best Practices

1. **Always check calibration** for models used in production
2. **Use cross-validation** for calibration to avoid overfitting
3. **Compare methods**: Try both Platt scaling and Isotonic regression
4. **Monitor in production**: Recalibrate if data distribution changes
5. **Document calibration**: Note which method was used and why

---

## Key Takeaways

1. **Three sets**: Train, Validation, Test - never touch test set until final evaluation
2. **Cross-validation**: More reliable performance estimate, reduces variance
3. **Hyperparameter tuning**: Grid search (exhaustive), Random search (faster), Bayesian optimization (smart)
4. **Bias-Variance**: Balance model complexity - high bias (underfitting) vs high variance (overfitting)
5. **Learning curves**: Diagnose overfitting/underfitting, determine if more data helps
6. **Evaluation metrics**: Choose appropriate metrics based on problem type and data characteristics
7. **Model calibration**: Ensure predicted probabilities match actual frequencies (critical for production)

## Common Mistakes to Avoid

1. **Using test set for tuning**: Test set should only be used for final evaluation
2. **Not using cross-validation**: Single train-test split can be misleading
3. **Ignoring class imbalance**: Accuracy is misleading for imbalanced data
4. **Overfitting to validation set**: Don't tune hyperparameters too much on validation set
5. **Not stratifying splits**: Important for imbalanced classification problems
6. **Choosing wrong metric**: Use metrics appropriate for your problem

---

## Next Steps

- Practice with different datasets
- Experiment with hyperparameter tuning methods
- Analyze learning curves for different models
- Move to [06-ensemble-methods](../06-ensemble-methods/README.md)

**Remember**: Never touch test set until final evaluation! Use validation set for all tuning and model selection.

