# Ensemble Methods Complete Guide

Comprehensive guide to combining multiple models for better performance.

## Table of Contents

- [Introduction to Ensembles](#introduction-to-ensembles)
- [Bagging](#bagging)
- [Boosting](#boosting)
- [Stacking](#stacking)
- [Voting](#voting)
- [Practice Exercises](#practice-exercises)

---

## Introduction to Ensembles

### Why Ensembles?

**Wisdom of the Crowd**: Multiple models often outperform single models.

**Benefits:**
- Reduces overfitting
- Improves generalization
- More robust predictions
- Better performance

**Trade-off:**
- More computational cost
- Less interpretable
- More complex

---

## Bagging

### Bootstrap Aggregating

Train multiple models on different subsets, average predictions.

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest (bagging with decision trees)
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

### Custom Bagging

```python
# Bagging with any base estimator
base_estimator = DecisionTreeClassifier(max_depth=5)
bagging = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Boosting

### AdaBoost

Adaptive boosting - focuses on misclassified samples.

```python
from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Gradient Boosting

Sequential error correction.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### XGBoost

Optimized gradient boosting.

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Stacking

Train meta-learner on base model predictions.

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50)),
    ('svm', SVC(probability=True)),
    ('knn', KNeighborsClassifier())
]

# Meta-learner
meta_learner = LogisticRegression()

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Voting

### Hard Voting

Majority class wins.

```python
from sklearn.ensemble import VotingClassifier

voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC()),
        ('knn', KNeighborsClassifier())
    ],
    voting='hard'
)
voting_hard.fit(X_train, y_train)
y_pred = voting_hard.predict(X_test)
print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Soft Voting

Average probabilities.

```python
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('knn', KNeighborsClassifier())
    ],
    voting='soft'
)
voting_soft.fit(X_train, y_train)
y_pred = voting_soft.predict(X_test)
print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Practice Exercises

### Exercise 1: Compare Ensemble Methods

**Task:** Compare Bagging, Boosting, and Voting on a dataset.

**Solution:**
```python
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'AdaBoost': AdaBoostClassifier(n_estimators=50),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True))
    ], voting='soft')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name:20s}: {accuracy_score(y_test, y_pred):.3f}")
```

---

## Key Takeaways

1. **Bagging**: Reduces variance, parallel training
2. **Boosting**: Reduces bias, sequential training
3. **Stacking**: Meta-learning approach
4. **Voting**: Simple ensemble

---

## Next Steps

- Practice with real datasets
- Experiment with hyperparameters
- Move to [07-feature-engineering](../07-feature-engineering/README.md)

**Remember**: Ensembles often win competitions!

