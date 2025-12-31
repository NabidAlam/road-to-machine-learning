# Model Evaluation & Optimization Complete Guide

Comprehensive guide to properly evaluating models and optimizing their performance.

## Table of Contents

- [Data Splitting](#data-splitting)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Bias-Variance Tradeoff](#bias-variance-tradeoff)
- [Learning Curves](#learning-curves)
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

## Key Takeaways

1. **Three sets**: Train, Validation, Test - never touch test set until final evaluation
2. **Cross-validation**: More reliable performance estimate, reduces variance
3. **Hyperparameter tuning**: Grid search (exhaustive), Random search (faster), Bayesian optimization (smart)
4. **Bias-Variance**: Balance model complexity - high bias (underfitting) vs high variance (overfitting)
5. **Learning curves**: Diagnose overfitting/underfitting, determine if more data helps
6. **Evaluation metrics**: Choose appropriate metrics based on problem type and data characteristics

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

