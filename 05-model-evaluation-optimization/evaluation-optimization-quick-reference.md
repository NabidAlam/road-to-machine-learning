# Model Evaluation & Optimization Quick Reference Guide

Quick reference for evaluation techniques, optimization methods, and best practices.

## Table of Contents

- [Data Splitting](#data-splitting)
- [Cross-Validation](#cross-validation)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Data Splitting

### Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

# First split: Train + (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# Second split: Validation + Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# Result: 60% train, 20% validation, 20% test
```

### Stratified Split (Classification)

```python
# Maintains class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Split Ratios by Dataset Size

| Dataset Size | Train | Validation | Test |
|--------------|-------|------------|------|
| Small (< 10K) | 70% | 15% | 15% |
| Medium (10K-100K) | 60% | 20% | 20% |
| Large (> 100K) | 98% | 1% | 1% |

---

## Cross-Validation

### K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Stratified K-Fold (Classification)

```python
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')
```

### Leave-One-Out

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
```

### Common CV Strategies

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **K-Fold** | General purpose | Reliable estimate | Not for time series |
| **Stratified K-Fold** | Imbalanced classification | Maintains distribution | Classification only |
| **Leave-One-Out** | Small datasets | Uses all data | Very slow |
| **Time Series Split** | Time-dependent data | Respects temporal order | Only for time series |

---

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
best_model = grid_search.best_estimator_
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, None],
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    model, param_dist, n_iter=50, cv=5,
    scoring='accuracy', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

### Bayesian Optimization (Optuna)

```python
import optuna

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print("Best params:", study.best_params)
```

### Tuning Method Comparison

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | Small parameter space | Exhaustive, finds best | Slow, doesn't scale |
| **Random Search** | Large parameter space | Fast, good results | May miss optimal |
| **Bayesian Optimization** | Expensive evaluations | Smart search | More complex |

---

## Model Calibration

### What is Calibration?

Model calibration ensures predicted probabilities match actual frequencies. A well-calibrated model: if it predicts 70%, outcome should be positive ~70% of the time.

### Calibration Techniques

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Platt Scaling (sigmoid)
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

# Isotonic Regression (non-parametric)
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
calibrated.fit(X_train, y_train)

# Evaluate with Brier Score
from sklearn.metrics import brier_score_loss
brier_score = brier_score_loss(y_test, y_pred_proba)
```

### When to Calibrate

- **Tree-based models** (Random Forest, XGBoost) often need calibration
- **Logistic Regression** usually well-calibrated
- **SVM** benefits from calibration
- Use when probabilities matter (cost-sensitive learning, decision thresholds)

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score,
                            confusion_matrix, classification_report)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# For binary classification
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Detailed report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

### Regression Metrics

```python
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                            r2_score)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

### Metric Selection Guide

**Classification:**
- Balanced data: Accuracy
- Imbalanced data: F1-score, ROC-AUC, Precision-Recall
- Cost-sensitive: Precision or Recall (based on cost)

**Regression:**
- Outliers matter: RMSE
- Outliers don't matter: MAE
- Interpretability: R²

---

## Learning Curves

### Plot Learning Curve

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='Training')
plt.plot(train_sizes, val_mean, 'o-', label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Diagnose Issues

**Underfitting:**
- Both curves converge to low performance
- Small gap between train and validation
- Solution: More complex model, better features

**Overfitting:**
- Large gap between train and validation
- Train performance much higher
- Solution: Regularization, more data, simpler model

**Good Fit:**
- Both curves converge to high performance
- Small gap between train and validation

---

## Bias-Variance Analysis

### Test Model Complexity

```python
max_depths = range(1, 21)
train_scores = []
val_scores = []

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    val_scores.append(model.score(X_val, y_val))

# Find optimal complexity
optimal_idx = np.argmax(val_scores)
optimal_depth = max_depths[optimal_idx]
```

### Bias-Variance Tradeoff

**High Bias (Underfitting):**
- Model too simple
- Poor performance on both train and validation
- Solution: Increase model complexity

**High Variance (Overfitting):**
- Model too complex
- Good train performance, poor validation
- Solution: Regularization, more data, simpler model

**Optimal:**
- Balance between bias and variance
- Good performance on both sets

---

## Common Issues & Solutions

### Issue 1: Overfitting

**Symptoms:**
- High train accuracy, low validation accuracy
- Large gap between train and validation

**Solutions:**
```python
# Solution 1: Regularization
model = LogisticRegression(C=0.1)  # Lower C = more regularization

# Solution 2: Reduce complexity
model = DecisionTreeClassifier(max_depth=5)

# Solution 3: More data
# Collect more training samples

# Solution 4: Dropout/Ensemble
model = RandomForestClassifier(n_estimators=100, max_depth=10)
```

### Issue 2: Underfitting

**Symptoms:**
- Low performance on both train and validation
- Small gap between train and validation

**Solutions:**
```python
# Solution 1: Increase complexity
model = DecisionTreeClassifier(max_depth=20)

# Solution 2: Better features
# Feature engineering

# Solution 3: Different algorithm
# Try non-linear models (SVM, Random Forest)
```

### Issue 3: Unreliable Performance Estimate

**Symptoms:**
- Large variance in cross-validation scores
- Test score very different from CV score

**Solutions:**
```python
# Solution 1: More folds
cv = KFold(n_splits=10)  # More folds = more stable

# Solution 2: Stratified CV
cv = StratifiedKFold(n_splits=5)  # For classification

# Solution 3: Nested CV
# Use nested cross-validation for unbiased estimate
```

### Issue 4: Slow Hyperparameter Tuning

**Symptoms:**
- Grid search takes too long
- Too many parameter combinations

**Solutions:**
```python
# Solution 1: Random search
random_search = RandomizedSearchCV(model, param_dist, n_iter=50)

# Solution 2: Smaller grid
param_grid = {
    'C': [0.1, 1, 10]  # Fewer values
}

# Solution 3: Parallel processing
grid_search = GridSearchCV(model, param_grid, n_jobs=-1)
```

---

## Best Practices Checklist

### Data Splitting
- [ ] Use proper train/validation/test split
- [ ] Stratify splits for classification
- [ ] Never use test set for tuning
- [ ] Verify class distribution maintained

### Cross-Validation
- [ ] Use appropriate CV method (K-Fold, Stratified, Time Series)
- [ ] Use enough folds (5-10 typically)
- [ ] Shuffle data before CV
- [ ] Use CV for model selection, not just evaluation

### Hyperparameter Tuning
- [ ] Tune on validation set, not test set
- [ ] Use cross-validation in grid search
- [ ] Start with random search for large spaces
- [ ] Don't overfit to validation set

### Evaluation
- [ ] Use appropriate metrics for problem type
- [ ] Don't rely only on accuracy
- [ ] Check confusion matrix for classification
- [ ] Compare multiple metrics

### Model Selection
- [ ] Compare multiple algorithms
- [ ] Use learning curves to diagnose issues
- [ ] Analyze bias-variance tradeoff
- [ ] Only use test set at the very end

---

## Quick Tips

1. **Always stratify** for classification problems
2. **Never touch test set** until final evaluation
3. **Use cross-validation** for reliable estimates
4. **Start simple** - baseline models first
5. **Visualize** - learning curves, confusion matrices
6. **Compare metrics** - don't rely on one metric
7. **Tune systematically** - grid search or random search
8. **Diagnose issues** - learning curves show problems
9. **Balance complexity** - find bias-variance sweet spot
10. **Document everything** - keep track of experiments

---

## Common Mistakes to Avoid

1. Using test set for hyperparameter tuning
2. Not stratifying splits for imbalanced data
3. Using accuracy for imbalanced classification
4. Overfitting to validation set
5. Not using cross-validation
6. Ignoring learning curves
7. Choosing wrong evaluation metric
8. Data leakage in preprocessing
9. Too many hyperparameter combinations
10. Not comparing multiple models

---

## Code Templates

### Complete Evaluation Pipeline

```python
# 1. Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 2. Baseline evaluation
baseline_scores = cross_val_score(
    model, X_train, y_train, cv=5, scoring='accuracy'
)

# 3. Hyperparameter tuning
grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 4. Final evaluation
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
```

### Learning Curve Template

```python
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label='Training')
plt.plot(train_sizes, val_mean, label='Validation')
plt.legend()
plt.show()
```

---

## Resources

- [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna Documentation](https://optuna.org/)

---

**Remember**: Proper evaluation is the foundation of reliable machine learning models!

