# Ensemble Methods Quick Reference Guide

Quick reference for ensemble methods, code snippets, and best practices.

## Table of Contents

- [Method Selection](#method-selection)
- [Code Snippets](#code-snippets)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Common Issues & Solutions](#common-issues--solutions)
- [Best Practices Checklist](#best-practices-checklist)

---

## Method Selection

### Quick Decision Tree

```
Need to improve model performance?
│
├─ High variance (overfitting)?
│  └─ YES → Bagging (Random Forest)
│
├─ High bias (underfitting)?
│  └─ YES → Boosting (XGBoost, LightGBM)
│
├─ Have diverse models?
│  ├─ YES → Stacking or Voting
│  └─ NO → Continue
│
├─ Need best performance?
│  ├─ YES → Stacking
│  └─ NO → Voting
│
└─ Want simplicity?
   └─ YES → Voting
```

### Method Comparison

| Method | Best For | Pros | Cons | Speed |
|--------|----------|------|------|-------|
| **Bagging** | High-variance models | Reduces variance, parallel | Doesn't reduce bias | Fast |
| **Boosting** | High-bias models | Reduces bias, powerful | Can overfit, slower | Medium |
| **Stacking** | Best performance | Very flexible, often best | Complex, slow | Slow |
| **Voting** | Quick ensemble | Simple, fast | Less sophisticated | Fast |

---

## Code Snippets

### Random Forest (Bagging)

```python
from sklearn.ensemble import RandomForestClassifier

# Basic
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Tuned
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")
```

### AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier

# Basic
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# Tuned
adaboost = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    algorithm='SAMME.R',
    random_state=42
)
```

### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Basic
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

# Tuned with early stopping
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)
```

### XGBoost

```python
import xgboost as xgb

# Basic
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Tuned
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,
    reg_lambda=1,
    random_state=42
)
```

### LightGBM

```python
import lightgbm as lgb

# Basic
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Tuned
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    random_state=42
)
```

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

# Hard voting
voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(random_state=42)),
        ('knn', KNeighborsClassifier())
    ],
    voting='hard'
)

# Soft voting
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft',
    weights=[2, 1, 1]  # Weighted voting
)
```

### Stacking Classifier

```python
from sklearn.ensemble import StackingClassifier

# Basic stacking
stacking = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Multi-level stacking
level1 = StackingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

level2 = StackingClassifier(
    estimators=[('level1', level1), ('svm', svm)],
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)
```

### Custom Bagging

```python
from sklearn.ensemble import BaggingClassifier

# Bagging with any estimator
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42
)
```

---

## Hyperparameter Tuning

### Random Forest Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

### XGBoost Tuning

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

### Stacking Meta-Learner Tuning

```python
# Try different meta-learners
meta_learners = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Ridge': RidgeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

for name, meta in meta_learners.items():
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=5
    )
    scores = cross_val_score(stacking, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f}")
```

---

## Common Issues & Solutions

### Issue 1: Ensemble Overfitting

**Symptoms:**
- High train accuracy, low validation accuracy
- Large gap between train and test

**Solutions:**
```python
# Solution 1: Regularization
rf = RandomForestClassifier(
    max_depth=10,           # Limit depth
    min_samples_split=10,   # Require more samples
    min_samples_leaf=5,     # Require more in leaf
    random_state=42
)

# Solution 2: Early stopping (boosting)
gb = GradientBoostingClassifier(
    n_estimators=1000,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)

# Solution 3: Reduce ensemble size
rf = RandomForestClassifier(n_estimators=50)  # Fewer trees
```

### Issue 2: Not Enough Diversity

**Symptoms:**
- Ensemble performs similar to single model
- Models make same errors

**Solutions:**
```python
# Solution 1: Different algorithms
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier()),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

# Solution 2: Different hyperparameters
rf_models = [
    RandomForestClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=10, random_state=42),
    RandomForestClassifier(max_depth=15, random_state=42)
]

# Solution 3: Different feature subsets
# Use feature selection to create diverse models
```

### Issue 3: Slow Training

**Symptoms:**
- Ensemble takes too long to train
- Not feasible for large datasets

**Solutions:**
```python
# Solution 1: Parallel processing
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Solution 2: Fewer estimators
rf = RandomForestClassifier(n_estimators=50)  # Reduce trees

# Solution 3: Use faster algorithms
# LightGBM is faster than XGBoost
lgb_model = lgb.LGBMClassifier(n_estimators=100)

# Solution 4: Sample data
from sklearn.utils import resample
X_sample, y_sample = resample(X_train, y_train, n_samples=10000)
```

### Issue 4: Poor Performance

**Symptoms:**
- Ensemble doesn't improve over single model
- Worse than baseline

**Solutions:**
```python
# Solution 1: Check base model quality
# Only use good base models
base_scores = []
for model in base_models:
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    base_scores.append(score)

# Filter weak models
good_models = [m for m, s in zip(base_models, base_scores) if s > 0.7]

# Solution 2: Tune hyperparameters
# Proper tuning is critical

# Solution 3: Increase diversity
# Use more diverse base models
```

---

## Best Practices Checklist

### Model Selection
- [ ] Start with baseline models
- [ ] Compare single models first
- [ ] Choose diverse base models
- [ ] Filter out weak models

### Ensemble Building
- [ ] Use appropriate method (bagging/boosting/stacking/voting)
- [ ] Ensure model diversity
- [ ] Tune hyperparameters
- [ ] Use cross-validation

### Evaluation
- [ ] Compare ensembles systematically
- [ ] Use proper train/validation/test split
- [ ] Check for overfitting
- [ ] Measure diversity

### Optimization
- [ ] Tune ensemble hyperparameters
- [ ] Try different meta-learners (stacking)
- [ ] Experiment with weights (voting)
- [ ] Use early stopping (boosting)

---

## Quick Tips

1. **Start simple**: Try voting before stacking
2. **Diversity matters**: Different algorithms work better together
3. **Tune hyperparameters**: Critical for performance
4. **Use cross-validation**: Reliable performance estimates
5. **Watch for overfitting**: Ensembles can overfit too
6. **Feature importance**: Understand what matters
7. **Compare systematically**: Use same evaluation method
8. **Consider computational cost**: Some ensembles are slow
9. **Early stopping**: Prevents overfitting in boosting
10. **Ensemble selection**: Not all models need to be included

---

## Common Mistakes to Avoid

1. Using identical models in ensemble
2. Not tuning hyperparameters
3. Overfitting to validation set
4. Ignoring base model quality
5. Too many models (diminishing returns)
6. Not using cross-validation
7. Ignoring computational cost
8. Not checking for overfitting
9. Using wrong ensemble method
10. Not measuring diversity

---

## Code Templates

### Complete Ensemble Pipeline

```python
# 1. Baseline models
baseline_models = {
    'RF': RandomForestClassifier(random_state=42),
    'GB': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# 2. Evaluate baselines
for name, model in baseline_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"{name}: {scores.mean():.3f}")

# 3. Build ensemble
ensemble = VotingClassifier(
    estimators=list(baseline_models.items()),
    voting='soft'
)

# 4. Evaluate ensemble
ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
print(f"Ensemble: {ensemble_scores.mean():.3f}")

# 5. Final evaluation
ensemble.fit(X_train, y_train)
test_score = ensemble.score(X_test, y_test)
print(f"Test: {test_score:.3f}")
```

### Feature Importance Template

```python
# For tree-based ensembles
if hasattr(model, 'feature_importances_'):
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.barh(importance['feature'], importance['importance'])
    plt.show()
```

---

## Resources

- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/en/docs/)

---

**Remember**: Ensembles often win competitions! Diversity and proper tuning are key to success!

