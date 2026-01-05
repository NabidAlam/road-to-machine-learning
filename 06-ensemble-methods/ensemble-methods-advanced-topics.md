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

#### XGBoost Regularization Parameters

XGBoost provides multiple ways to reduce overfitting:

```python
try:
    import xgboost as xgb
    
    # XGBoost with comprehensive regularization
    xgb_model = xgb.XGBClassifier(
        # Tree structure
        n_estimators=100,
        max_depth=6,              # Maximum tree depth (reduce to prevent overfitting)
        min_child_weight=1,       # Minimum sum of instance weight in child (increase to prevent overfitting)
        
        # Learning rate
        learning_rate=0.1,        # Shrinkage (lower = more conservative, prevents overfitting)
        
        # Regularization
        gamma=0,                  # Minimum loss reduction required for split (increase to prevent overfitting)
        reg_alpha=0,              # L1 regularization (increase to add L1 penalty)
        reg_lambda=1,             # L2 regularization (increase to add L2 penalty)
        
        # Sampling
        subsample=0.8,            # Row sampling (fraction of samples for each tree)
        colsample_bytree=0.8,     # Column sampling per tree
        colsample_bylevel=1.0,    # Column sampling per level
        colsample_bynode=1.0,     # Column sampling per node
        
        # Early stopping
        early_stopping_rounds=10,
        
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

#### XGBoost Regularization Strategies

**1. Gamma (Minimum Loss Reduction):**
```python
# Higher gamma = more conservative splits
gammas = [0, 0.1, 0.5, 1.0]
for gamma in gammas:
    model = xgb.XGBClassifier(gamma=gamma, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Gamma {gamma}: {score:.3f}")
```

**2. Max Depth:**
```python
# Lower max_depth = simpler trees, less overfitting
depths = [3, 5, 7, 10]
for depth in depths:
    model = xgb.XGBClassifier(max_depth=depth, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Max Depth {depth}: {score:.3f}")
```

**3. Min Child Weight:**
```python
# Higher min_child_weight = more conservative splits
weights = [1, 3, 5, 10]
for weight in weights:
    model = xgb.XGBClassifier(min_child_weight=weight, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Min Child Weight {weight}: {score:.3f}")
```

**4. L1 and L2 Regularization:**
```python
# L1 (alpha) and L2 (lambda) regularization
alphas = [0, 0.1, 0.5, 1.0]
lambdas = [1, 2, 5, 10]

for alpha, lam in zip(alphas, lambdas):
    model = xgb.XGBClassifier(
        reg_alpha=alpha,
        reg_lambda=lam,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Alpha {alpha}, Lambda {lam}: {score:.3f}")
```

**5. Subsampling:**
```python
# Row and column sampling
subsamples = [0.6, 0.8, 1.0]
col_samples = [0.6, 0.8, 1.0]

for sub, col in zip(subsamples, col_samples):
    model = xgb.XGBClassifier(
        subsample=sub,
        colsample_bytree=col,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Subsample {sub}, Colsample {col}: {score:.3f}")
```

#### XGBoost Optimizations

**1. Approximate Split Finding:**
```python
# XGBoost uses approximate algorithms for speed
model = xgb.XGBClassifier(
    tree_method='hist',  # Histogram-based (fast, approximate)
    # tree_method='exact',  # Exact greedy (slow, exact)
    # tree_method='approx',  # Approximate (balanced)
    n_estimators=100,
    random_state=42
)
```

**2. Quantiles Sketch:**
```python
# XGBoost uses quantile sketch for approximate split finding
# This reduces computation while maintaining accuracy

model = xgb.XGBClassifier(
    tree_method='hist',
    max_bin=256,  # Number of bins for histogram (more = more accurate, slower)
    n_estimators=100,
    random_state=42
)
```

**3. Weighted Quantiles Sketch:**
```python
# For weighted data, XGBoost uses weighted quantile sketch
# Automatically handles instance weights

# Create sample weights
sample_weights = np.ones(len(X_train))
sample_weights[y_train == 1] = 2.0  # Weight positive class more

model = xgb.XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

#### XGBoost Missing Value Handling

XGBoost has built-in handling for missing values:

```python
# Create data with missing values
X_with_missing = X_train.copy()
X_with_missing.iloc[0:100, 0] = np.nan  # Add missing values

# XGBoost automatically handles missing values
model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    tree_method='hist'
)

# No need to impute - XGBoost learns best direction for missing values
model.fit(X_with_missing, y_train)

# Predictions work with missing values too
X_test_missing = X_test.copy()
X_test_missing.iloc[0:50, 0] = np.nan
predictions = model.predict(X_test_missing)
```

**How XGBoost Handles Missing Values:**
1. During training, XGBoost learns the best direction (left or right child) for missing values
2. This is more sophisticated than simple imputation
3. Works automatically - no preprocessing needed

#### Complete Mathematics of XGBoost

**Objective Function:**
```
Obj(θ) = Σ L(y_i, ŷ_i) + Σ Ω(f_k)

Where:
- L: Loss function (e.g., squared error, log loss)
- Ω: Regularization term
- f_k: k-th tree
```

**Regularization Term:**
```
Ω(f) = γT + (1/2)λ||w||²

Where:
- T: Number of leaves
- w: Leaf weights
- γ: Minimum loss reduction (gamma)
- λ: L2 regularization (lambda)
```

**Tree Structure Score (Similarity Score):**
```
Similarity = (Σ g_i)² / (Σ h_i + λ)

Where:
- g_i: First-order gradient
- h_i: Second-order gradient (Hessian)
- λ: L2 regularization
```

**Gain (Split Quality):**
```
Gain = Similarity_left + Similarity_right - Similarity_parent - γ

Where:
- γ: Minimum gain threshold
```

**Example Calculation:**
```python
# Simplified XGBoost calculation example
def calculate_similarity(gradients, hessians, lambda_reg=1.0):
    """Calculate similarity score for a leaf"""
    sum_g = np.sum(gradients)
    sum_h = np.sum(hessians)
    similarity = (sum_g ** 2) / (sum_h + lambda_reg)
    return similarity

# Example gradients and hessians
gradients = np.array([0.5, -0.3, 0.8, -0.2])
hessians = np.array([0.25, 0.15, 0.4, 0.1])

similarity = calculate_similarity(gradients, hessians)
print(f"Similarity score: {similarity:.4f}")
```

#### XGBoost for Regression

```python
import xgboost as xgb

# Regression model
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',  # Regression objective
    eval_metric='rmse',              # Evaluation metric
    random_state=42
)

xgb_reg.fit(X_train, y_train_regression)
y_pred = xgb_reg.predict(X_test)
```

#### XGBoost for Classification

```python
# Binary classification
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',  # Binary classification
    eval_metric='logloss',
    random_state=42
)

# Multiclass classification
xgb_multi = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',   # Multiclass
    eval_metric='mlogloss',
    num_class=3,                  # Number of classes
    random_state=42
)
```

#### Key XGBoost Parameters Summary

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|--------|
| **n_estimators** | Number of trees | 50-500 | More = better but slower |
| **max_depth** | Tree depth | 3-10 | Lower = less overfitting |
| **learning_rate** | Shrinkage | 0.01-0.3 | Lower = more conservative |
| **gamma** | Min loss reduction | 0-5 | Higher = more regularization |
| **min_child_weight** | Min samples in leaf | 1-10 | Higher = more regularization |
| **subsample** | Row sampling | 0.6-1.0 | Lower = more regularization |
| **colsample_bytree** | Column sampling | 0.6-1.0 | Lower = more regularization |
| **reg_alpha** | L1 regularization | 0-10 | Higher = sparser model |
| **reg_lambda** | L2 regularization | 1-10 | Higher = smoother model |

#### Best Practices

1. **Start with defaults**: XGBoost has good defaults
2. **Tune learning_rate and n_estimators together**: Lower learning rate needs more trees
3. **Use early stopping**: Prevents overfitting automatically
4. **Scale features**: XGBoost is tree-based but scaling can help
5. **Handle missing values**: XGBoost does this automatically
6. **Use cross-validation**: For reliable performance estimates

### LightGBM Advanced Features

#### GOSS (Gradient-based One-Side Sampling)

GOSS keeps instances with large gradients and randomly samples instances with small gradients.

**Why GOSS?**
- Instances with large gradients contribute more to information gain
- Randomly sampling small-gradient instances maintains accuracy
- Reduces computational cost significantly

```python
try:
    import lightgbm as lgb
    
    # LightGBM with GOSS (automatic)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        
        # GOSS parameters
        top_rate=0.2,        # Keep top 20% instances with large gradients
        other_rate=0.1,      # Randomly sample 10% of remaining instances
        
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

**GOSS Algorithm:**
1. Sort instances by gradient magnitude
2. Keep top `a × 100%` instances (large gradients)
3. Randomly sample `b × 100%` from remaining (small gradients)
4. Use weighted sampling to compensate for bias

#### EFB (Exclusive Feature Bundling)

EFB bundles mutually exclusive features (features that rarely take non-zero values simultaneously) into a single feature.

**Why EFB?**
- Reduces number of features
- Speeds up training
- Maintains accuracy

```python
try:
    import lightgbm as lgb
    
    # LightGBM with EFB (automatic)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        
        # EFB parameters
        max_bin=255,         # Number of bins (affects bundling)
        min_data_in_bin=3,   # Minimum data in bin
        
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42
    )
    
    lgb_model.fit(X_train, y_train)
    
except ImportError:
    print("Install LightGBM: pip install lightgbm")
```

**EFB Algorithm:**
1. Identify mutually exclusive features
2. Bundle them into single features
3. Use different bin boundaries to distinguish
4. Reduces feature count without information loss

#### LightGBM with Categorical Features

```python
try:
    import lightgbm as lgb
    
    # LightGBM handles categorical features natively
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        
        # Categorical features (specify indices)
        categorical_feature=[0, 1, 2],  # Column indices
        
        # Or let LightGBM auto-detect
        # categorical_feature='auto',
        
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42
    )
    
    # Fit with categorical features
    lgb_model.fit(X_train, y_train, 
                  categorical_feature=[0, 1, 2])
    
except ImportError:
    print("Install LightGBM: pip install lightgbm")
```

#### LightGBM Leaf-wise Growth

Unlike level-wise growth (XGBoost), LightGBM uses leaf-wise growth:

**Level-wise (XGBoost):**
- Grows tree level by level
- All nodes at same level split
- More balanced but slower

**Leaf-wise (LightGBM):**
- Grows tree by finding best leaf to split
- More efficient
- Can create deeper trees

```python
# LightGBM uses leaf-wise growth by default
# Control with num_leaves (max number of leaves)
lgb_model = lgb.LGBMClassifier(
    num_leaves=31,  # Maximum leaves (2^max_depth - 1 for balanced tree)
    max_depth=5,    # Limit depth
    n_estimators=100,
    random_state=42
)
```

#### LightGBM vs XGBoost

| Feature | LightGBM | XGBoost |
|---------|----------|---------|
| **Growth Strategy** | Leaf-wise | Level-wise |
| **Speed** | Faster | Slower |
| **Memory** | Lower | Higher |
| **Categorical** | Native support | Needs encoding |
| **GOSS** | Yes | No |
| **EFB** | Yes | No |
| **Accuracy** | Similar | Similar |

**Choose LightGBM when:**
- Large datasets
- Many categorical features
- Speed is important
- Memory is limited

**Choose XGBoost when:**
- Need more control
- Smaller datasets
- Want more regularization options

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

