# Advanced Ensemble Methods Topics

Comprehensive guide to advanced ensemble techniques, optimization strategies, and best practices.

## Table of Contents

- [Ensemble Diversity](#ensemble-diversity)
- [Advanced Boosting Techniques](#advanced-boosting-techniques)
- [Ensemble Hyperparameter Tuning](#ensemble-hyperparameter-tuning)
- [Ensemble Selection](#ensemble-selection)
- [Feature Importance in Ensembles](#feature-importance-in-ensembles)
- [Ensemble Interpretability](#ensemble-interpretability)
- [Common Ensemble Pitfalls](#common-ensemble-pitfalls)

---

## Ensemble Diversity

### Why Diversity Matters

Ensembles work best when base models make different errors. If all models make the same mistakes, combining them won't help.

### Measuring Diversity

```python
from sklearn.metrics import cohen_kappa_score
import numpy as np

def ensemble_diversity(models, X, y):
    """Measure diversity of ensemble predictions"""
    predictions = []
    
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    n_models = len(models)
    
    # Pairwise agreement (kappa score)
    kappa_scores = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            kappa = cohen_kappa_score(predictions[i], predictions[j])
            kappa_scores.append(kappa)
    
    # Lower kappa = more diverse
    avg_kappa = np.mean(kappa_scores)
    diversity = 1 - avg_kappa  # Higher = more diverse
    
    return diversity, avg_kappa

# Example
models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    SVC(probability=True, random_state=42)
]

for model in models:
    if isinstance(model, SVC):
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

diversity, kappa = ensemble_diversity(models, X_test, y_test)
print(f"Ensemble Diversity: {diversity:.3f}")
print(f"Average Kappa: {kappa:.3f}")
```

### Creating Diverse Models

```python
# Strategy 1: Different algorithms
diverse_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('nb', GaussianNB())
]

# Strategy 2: Different hyperparameters
rf_models = [
    RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
]

# Strategy 3: Different feature subsets
from sklearn.feature_selection import SelectKBest, f_classif

feature_sets = []
for k in [5, 10, 15]:
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_train, y_train)
    feature_sets.append((selector, X_selected))
```

---

## Advanced Boosting Techniques

### Early Stopping in Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Split for validation
X_train_gb, X_val_gb, y_train_gb, y_val_gb = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Gradient boosting with early stopping
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 iterations
    tol=0.0001,
    random_state=42
)

gb.fit(X_train_gb, y_train_gb)

print(f"Number of estimators used: {gb.n_estimators_}")
print(f"Best validation score: {gb.train_score_[-1]:.3f}")
```

### XGBoost Advanced Features

```python
try:
    import xgboost as xgb
    
    # XGBoost with custom objective and evaluation
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        objective='multi:softprob',  # For multiclass
        eval_metric='mlogloss',       # Evaluation metric
        early_stopping_rounds=10,     # Early stopping
        random_state=42
    )
    
    # Fit with validation set
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Feature importance
    importance = xgb_model.feature_importances_
    
    # Plot importance
    xgb.plot_importance(xgb_model, max_num_features=10)
    plt.show()
    
except ImportError:
    print("Install XGBoost: pip install xgboost")
```

### LightGBM Advanced Features

```python
try:
    import lightgbm as lgb
    
    # LightGBM with categorical features
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        categorical_feature=[0, 1],  # Specify categorical columns
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42
    )
    
    lgb_model.fit(X_train, y_train)
    
    # Feature importance
    importance = lgb_model.feature_importances_
    
except ImportError:
    print("Install LightGBM: pip install lightgbm")
```

---

## Ensemble Hyperparameter Tuning

### Tuning Random Forest

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

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)
print(f"Best params: {rf_random.best_params_}")
print(f"Best score: {rf_random.best_score_:.3f}")
```

### Tuning XGBoost

```python
try:
    import xgboost as xgb
    
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42),
        xgb_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    xgb_grid.fit(X_train, y_train)
    print(f"Best XGBoost params: {xgb_grid.best_params_}")
    
except ImportError:
    print("XGBoost not available")
```

### Tuning Stacking Meta-Learner

```python
# Tune meta-learner in stacking
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Try different meta-learners
meta_learners = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Ridge': RidgeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

best_meta = None
best_score = 0

for name, meta in meta_learners.items():
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=5
    )
    
    scores = cross_val_score(
        stacking, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_meta = name
    
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")

print(f"\nBest meta-learner: {best_meta}")
```

---

## Ensemble Selection

### Greedy Ensemble Selection

```python
def greedy_ensemble_selection(base_models, X, y, max_models=5):
    """Greedily select best subset of models"""
    selected = []
    remaining = list(range(len(base_models)))
    best_score = 0
    
    for _ in range(min(max_models, len(base_models))):
        best_idx = None
        best_new_score = best_score
        
        for idx in remaining:
            # Try adding this model
            test_selected = selected + [idx]
            test_models = [base_models[i] for i in test_selected]
            
            # Create voting classifier
            voting = VotingClassifier(
                estimators=[(f'model_{i}', base_models[i]) for i in test_selected],
                voting='soft'
            )
            
            # Evaluate
            scores = cross_val_score(
                voting, X, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            score = scores.mean()
            
            if score > best_new_score:
                best_new_score = score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
            best_score = best_new_score
        else:
            break
    
    return selected, best_score

# Example usage
base_models_list = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    SVC(probability=True, random_state=42),
    KNeighborsClassifier(),
    LogisticRegression(random_state=42, max_iter=1000)
]

selected_indices, score = greedy_ensemble_selection(
    base_models_list, X_train, y_train, max_models=3
)
print(f"Selected models: {selected_indices}")
print(f"Ensemble score: {score:.3f}")
```

### Ensemble Pruning

```python
def ensemble_pruning(ensemble, X, y, threshold=0.01):
    """Remove models that don't improve ensemble"""
    base_models = ensemble.estimators
    selected = list(range(len(base_models)))
    
    # Start with all models
    current_score = cross_val_score(
        ensemble, X, y, cv=5, scoring='accuracy', n_jobs=-1
    ).mean()
    
    # Try removing each model
    improved = True
    while improved and len(selected) > 1:
        improved = False
        best_removal = None
        best_new_score = current_score
        
        for idx in selected:
            test_selected = [i for i in selected if i != idx]
            test_models = [base_models[i] for i in test_selected]
            
            test_ensemble = VotingClassifier(
                estimators=[(f'model_{i}', base_models[i]) for i in test_selected],
                voting='soft'
            )
            
            score = cross_val_score(
                test_ensemble, X, y, cv=5, scoring='accuracy', n_jobs=-1
            ).mean()
            
            if score > best_new_score + threshold:
                best_new_score = score
                best_removal = idx
                improved = True
        
        if best_removal is not None:
            selected.remove(best_removal)
            current_score = best_new_score
    
    return selected, current_score
```

---

## Feature Importance in Ensembles

### Random Forest Feature Importance

```python
# Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Mean decrease impurity
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance:")
print(importance_df)

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Permutation importance (more reliable)
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

perm_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nPermutation Importance:")
print(perm_df)
```

### XGBoost Feature Importance

```python
try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Different importance types
    importance_types = ['weight', 'gain', 'cover']
    
    for imp_type in importance_types:
        importance = xgb_model.get_booster().get_score(importance_type=imp_type)
        print(f"\nXGBoost Importance ({imp_type}):")
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feature}: {score:.3f}")
    
except ImportError:
    print("XGBoost not available")
```

---

## Ensemble Interpretability

### SHAP Values for Ensembles

```python
try:
    import shap
    
    # SHAP for Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test[:100])
    
    # Summary plot
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
    
    # Waterfall plot for single prediction
    shap.waterfall_plot(explainer.expected_value[0], shap_values[0][0], X_test[0])
    
except ImportError:
    print("Install SHAP: pip install shap")
```

### Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Partial dependence for Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Plot partial dependence
fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(
    rf, X_train, feature_names=feature_names,
    features=[0, 1, 2, 3],  # Features to plot
    ax=ax
)
plt.tight_layout()
plt.show()
```

---

## Common Ensemble Pitfalls

### Pitfall 1: Overfitting with Ensembles

**Problem:** Ensemble overfits to training data

**Solution:**
```python
# Use regularization
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,           # Limit depth
    min_samples_split=10,   # Require more samples to split
    min_samples_leaf=5,     # Require more samples in leaf
    max_features='sqrt',    # Limit features
    random_state=42
)

# Use early stopping for boosting
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    validation_fraction=0.2,
    n_iter_no_change=10,
    random_state=42
)
```

### Pitfall 2: Not Enough Diversity

**Problem:** All models make similar errors

**Solution:**
```python
# Use different algorithms
diverse_ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

# Use different hyperparameters
rf_models = [
    RandomForestClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(max_depth=10, random_state=42),
    RandomForestClassifier(max_depth=15, random_state=42)
]
```

### Pitfall 3: Too Many Models

**Problem:** Diminishing returns, slow prediction

**Solution:**
```python
# Use ensemble selection to find optimal subset
selected_models = greedy_ensemble_selection(
    base_models, X_train, y_train, max_models=5
)

# Use fewer, better models
best_ensemble = VotingClassifier(
    estimators=[base_models[i] for i in selected_models],
    voting='soft'
)
```

### Pitfall 4: Ignoring Base Model Quality

**Problem:** Combining weak models doesn't help

**Solution:**
```python
# Filter out weak models first
base_models = [rf, gb, svm, knn]
base_scores = []

for model in base_models:
    scores = cross_val_score(model, X_train, y_train, cv=5)
    base_scores.append(scores.mean())

# Only use models above threshold
threshold = 0.7
good_models = [
    (f'model_{i}', base_models[i])
    for i, score in enumerate(base_scores)
    if score >= threshold
]

if len(good_models) > 1:
    ensemble = VotingClassifier(estimators=good_models, voting='soft')
```

---

## Key Takeaways

1. **Diversity is crucial**: Ensembles work best with diverse base models
2. **Early stopping**: Prevents overfitting in boosting methods
3. **Hyperparameter tuning**: Critical for ensemble performance
4. **Ensemble selection**: Not all models need to be in ensemble
5. **Feature importance**: Use permutation importance for reliability
6. **Interpretability**: SHAP and partial dependence help understand ensembles
7. **Avoid pitfalls**: Watch for overfitting, lack of diversity, too many models

---

## Next Steps

- Practice with Kaggle competitions
- Experiment with different ensemble combinations
- Learn about advanced boosting libraries (XGBoost, LightGBM, CatBoost)
- Move to feature engineering module

**Remember**: Diversity and proper tuning are keys to successful ensembles!

