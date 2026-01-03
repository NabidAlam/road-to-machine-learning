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

**Wisdom of the Crowd**: Multiple models often outperform single models. The idea is that combining predictions from multiple models reduces errors and improves generalization.

**Benefits:**
- Reduces overfitting (especially bagging)
- Improves generalization
- More robust predictions
- Better performance (often top performers in competitions)
- Handles different types of errors from different models

**Trade-offs:**
- More computational cost (training multiple models)
- Less interpretable (harder to understand combined predictions)
- More complex (more hyperparameters to tune)
- Requires more memory

### When to Use Ensembles

**Use ensembles when:**
- You have diverse base models that make different errors
- Single models are overfitting
- You need the best possible performance
- Computational resources are available

**Don't use ensembles when:**
- Single model already performs well
- Interpretability is critical
- Limited computational resources
- Real-time predictions needed (some ensembles are slow)

---

## Bagging

### Bootstrap Aggregating

Train multiple models on different bootstrap samples (sampling with replacement), then average predictions. Reduces variance without increasing bias.

**How it works:**
1. Create multiple bootstrap samples from training data
2. Train a model on each sample
3. Average predictions (classification: majority vote, regression: mean)

**Key characteristics:**
- Models trained in parallel (fast)
- Reduces variance
- Works well with high-variance models (e.g., decision trees)
- Each model sees different data

### Random Forest

Most popular bagging method using decision trees.

```python
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Random Forest (bagging with decision trees)
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # Features to consider at each split
    bootstrap=True,        # Use bootstrap sampling
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
print("\nFeature Importance:")
print(importance)

# Out-of-bag score (evaluation without separate validation set)
print(f"\nOut-of-bag Score: {rf.oob_score_:.3f}")
```

### Hyperparameters

```python
# Tuned Random Forest
rf_tuned = RandomForestClassifier(
    n_estimators=200,           # More trees = better, but slower
    max_depth=15,               # Control tree depth
    min_samples_split=10,       # Prevent overfitting
    min_samples_leaf=4,         # Minimum samples in leaf
    max_features='sqrt',        # 'sqrt', 'log2', or number
    max_samples=0.8,            # Bootstrap sample size
    bootstrap=True,
    oob_score=True,             # Calculate out-of-bag score
    random_state=42
)
rf_tuned.fit(X_train, y_train)
print(f"Tuned RF Accuracy: {accuracy_score(y_test, rf_tuned.predict(X_test)):.3f}")
```

### Custom Bagging

Use any base estimator with bagging.

```python
# Bagging with any base estimator
base_estimator = DecisionTreeClassifier(max_depth=5)
bagging = BaggingClassifier(
    base_estimator=base_estimator,
    n_estimators=50,
    max_samples=0.8,        # Bootstrap sample size
    max_features=0.8,       # Feature sample size
    bootstrap=True,         # Use bootstrap
    bootstrap_features=False,  # Don't bootstrap features
    oob_score=True,         # Calculate out-of-bag score
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Out-of-bag Score: {bagging.oob_score_:.3f}")

# Bagging with different base estimators
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Bagging with SVM
bagging_svm = BaggingClassifier(
    base_estimator=SVC(probability=True),
    n_estimators=10,  # Fewer for SVM (slower)
    random_state=42
)
bagging_svm.fit(X_train_scaled, y_train)
print(f"Bagging SVM Accuracy: {accuracy_score(y_test, bagging_svm.predict(X_test_scaled)):.3f}")
```

### Bagging vs Single Model

```python
# Compare single tree vs bagged trees
single_tree = DecisionTreeClassifier(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)
single_score = accuracy_score(y_test, single_tree.predict(X_test))

bagged_trees = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    random_state=42
)
bagged_trees.fit(X_train, y_train)
bagged_score = accuracy_score(y_test, bagged_trees.predict(X_test))

print(f"Single Tree Accuracy: {single_score:.3f}")
print(f"Bagged Trees Accuracy: {bagged_score:.3f}")
print(f"Improvement: {bagged_score - single_score:.3f}")
```

---

## Boosting

### How Boosting Works

Boosting trains models sequentially, where each new model focuses on correcting errors of previous models. Reduces bias by combining weak learners.

**Key differences from bagging:**
- Sequential training (not parallel)
- Each model learns from previous model's errors
- Reduces bias (bagging reduces variance)
- Models are weighted based on performance

### AdaBoost

Adaptive boosting - focuses on misclassified samples by increasing their weights.

#### How AdaBoost Works

**Components:**
1. **Weak Learners**: Simple models (typically decision stumps - depth 1 trees)
2. **Weights**: Each instance has a weight that increases if misclassified
3. **Model Weights**: Each weak learner has a weight based on its accuracy
4. **Final Model**: Weighted combination of all weak learners

**Algorithm Steps:**
1. Initialize equal weights for all instances
2. For each iteration:
   - Train weak learner on weighted data
   - Calculate error rate
   - Compute learner weight (α)
   - Update instance weights (increase for misclassified)
   - Normalize weights
3. Final prediction: Weighted vote of all learners

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# AdaBoost with detailed configuration
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # Weak learner (stump)
    n_estimators=50,        # Number of weak learners
    learning_rate=1.0,      # Shrinkage factor (weight multiplier)
    algorithm='SAMME.R',    # 'SAMME' or 'SAMME.R' (more efficient)
    random_state=42
)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, adaboost.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Estimator weights (how much each model contributes)
print(f"\nEstimator Weights (first 5): {adaboost.estimator_weights_[:5]}")

# Error rate at each stage
print("\nError Rate by Stage:")
for i, estimator in enumerate(adaboost.estimators_[:10]):
    stage_pred = estimator.predict(X_test)
    error = 1 - accuracy_score(y_test, stage_pred)
    weight = adaboost.estimator_weights_[i]
    print(f"Stage {i+1}: Error={error:.3f}, Weight={weight:.3f}")
```

#### AdaBoost Mathematics

**Weight Calculation:**
```
α_t = (1/2) × ln((1 - ε_t) / ε_t)

Where:
- α_t: Weight of t-th weak learner
- ε_t: Error rate of t-th weak learner
```

**Instance Weight Update:**
```
w_i^(t+1) = w_i^(t) × exp(-α_t × y_i × h_t(x_i)) / Z_t

Where:
- w_i: Weight of instance i
- y_i: True label
- h_t(x_i): Prediction of t-th learner
- Z_t: Normalization factor
```

**Final Prediction:**
```
H(x) = sign(Σ α_t × h_t(x))

Where:
- H(x): Final prediction
- h_t(x): Prediction of t-th learner
- α_t: Weight of t-th learner
```

#### AdaBoost Components in Detail

**1. Weak Learners:**
```python
# Decision stumps (depth 1)
stump = DecisionTreeClassifier(max_depth=1)

# Or use other weak learners
from sklearn.naive_bayes import GaussianNB
adaboost_nb = AdaBoostClassifier(
    base_estimator=GaussianNB(),
    n_estimators=50,
    random_state=42
)
```

**2. Learning Rate:**
```python
# Lower learning rate = more conservative updates
learning_rates = [0.5, 1.0, 1.5]
for lr in learning_rates:
    model = AdaBoostClassifier(
        learning_rate=lr,
        n_estimators=50,
        random_state=42
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Learning Rate {lr}: {score:.3f}")
```

**3. Number of Estimators:**
```python
# More estimators = better but slower
n_estimators_list = [10, 25, 50, 100, 200]
for n in n_estimators_list:
    model = AdaBoostClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"n_estimators {n}: {score:.3f}")
```

#### AdaBoost for Regression

```python
from sklearn.ensemble import AdaBoostRegressor

adaboost_reg = AdaBoostRegressor(
    base_estimator=None,  # Default: DecisionTreeRegressor(max_depth=3)
    n_estimators=50,
    learning_rate=1.0,
    loss='linear',  # 'linear', 'square', 'exponential'
    random_state=42
)

adaboost_reg.fit(X_train, y_train_regression)
y_pred = adaboost_reg.predict(X_test)
```

#### Key Takeaways

1. **Weak Learners**: Simple models (decision stumps)
2. **Adaptive**: Focuses on hard-to-classify instances
3. **Weighted Combination**: Final model is weighted sum
4. **Sequential**: Each learner depends on previous
5. **Reduces Bias**: Unlike bagging which reduces variance

### Gradient Boosting

Sequential error correction using gradient descent. Each new model fits the residuals (errors) of the previous model.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    loss='deviance',          # 'deviance' or 'exponential'
    learning_rate=0.1,        # Shrinkage (lower = more conservative)
    n_estimators=100,
    subsample=1.0,            # Fraction of samples for each tree
    criterion='friedman_mse', # Splitting criterion
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=3,              # Shallow trees (weak learners)
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Staged predictions (predictions at each stage)
staged_predictions = list(gb.staged_predict(X_test))
print(f"\nAccuracy at different stages:")
for i in [10, 25, 50, 100]:
    if i <= len(staged_predictions):
        stage_acc = accuracy_score(y_test, staged_predictions[i-1])
        print(f"  Stage {i}: {stage_acc:.3f}")
```

### XGBoost

Optimized gradient boosting with regularization and parallel processing.

```python
try:
    import xgboost as xgb
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_child_weight=1,
        subsample=0.8,           # Row sampling
        colsample_bytree=0.8,    # Column sampling
        gamma=0,                 # Minimum loss reduction
        reg_alpha=0,             # L1 regularization
        reg_lambda=1,            # L2 regularization
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
    # Feature importance
    xgb.plot_importance(xgb_model, max_num_features=10)
    plt.show()
except ImportError:
    print("Install XGBoost: pip install xgboost")
```

### LightGBM

Fast gradient boosting framework.

```python
try:
    import lightgbm as lgb
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42
    )
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred):.3f}")
except ImportError:
    print("Install LightGBM: pip install lightgbm")
```

### CatBoost

Handles categorical features automatically with advanced techniques.

#### Key Features

1. **Automatic Categorical Handling**: No need to encode categorical features
2. **Ordered Boosting**: Reduces overfitting
3. **GPU Support**: Fast training on GPUs
4. **Built-in Cross-Validation**: Easy model validation

#### Basic Usage

```python
try:
    import catboost as cb
    
    # CatBoost with categorical features
    cat_model = cb.CatBoostClassifier(
        iterations=100,           # Number of trees
        learning_rate=0.1,        # Shrinkage
        depth=6,                  # Tree depth
        loss_function='Logloss',  # Loss function
        eval_metric='Accuracy',   # Evaluation metric
        random_seed=42,
        verbose=False,            # Suppress output
        early_stopping_rounds=10  # Early stopping
    )
    
    # Fit with categorical features (specify indices)
    cat_model.fit(
        X_train, y_train,
        cat_features=[0, 1, 2],  # Categorical feature indices
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    y_pred = cat_model.predict(X_test)
    print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    
except ImportError:
    print("Install CatBoost: pip install catboost")
```

#### CatBoost Technical Aspects

**1. Ordered Boosting:**
- Uses permutation-based scheme
- Reduces overfitting
- More robust than standard boosting

**2. Categorical Feature Handling:**
- Uses target statistics
- Handles high-cardinality categories
- No need for one-hot encoding

**3. Symmetric Trees:**
- Balanced tree structure
- Faster prediction
- Better regularization

#### Advanced CatBoost Parameters

```python
cat_model = cb.CatBoostClassifier(
    # Tree structure
    iterations=100,
    depth=6,
    learning_rate=0.1,
    
    # Regularization
    l2_leaf_reg=3,              # L2 regularization
    border_count=32,            # Number of splits for numerical features
    bagging_temperature=1,      # Bayesian bagging
    
    # Categorical features
    max_cat_to_onehot=4,        # One-hot encode if categories <= this
    
    # Sampling
    subsample=0.8,              # Row sampling
    colsample_bylevel=0.8,      # Column sampling
    
    # Early stopping
    early_stopping_rounds=10,
    
    # Other
    random_seed=42,
    verbose=False
)
```

#### CatBoost for Regression

```python
cat_reg = cb.CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',       # Regression loss
    eval_metric='RMSE',
    random_seed=42,
    verbose=False
)

cat_reg.fit(X_train, y_train_regression, cat_features=[0, 1, 2])
y_pred = cat_reg.predict(X_test)
```

#### CatBoost vs Other Boosting

| Feature | CatBoost | XGBoost | LightGBM |
|---------|----------|---------|----------|
| **Categorical** | Native | Needs encoding | Native |
| **Speed** | Medium | Slow | Fast |
| **Overfitting** | Low (ordered boosting) | Medium | Medium |
| **Ease of Use** | Very Easy | Medium | Easy |

#### Key Takeaways

1. **Best for Categorical**: Excellent with categorical features
2. **Easy to Use**: Minimal preprocessing needed
3. **Robust**: Ordered boosting reduces overfitting
4. **Fast**: Good performance-speed balance

---

## Stacking

Train a meta-learner on predictions from base models. More sophisticated than voting.

### How Stacking Works

**Steps:**
1. **Base Models**: Train diverse models (Level 0)
2. **Meta-Model**: Train on base model predictions (Level 1)
3. **Final Prediction**: Meta-model predicts from base predictions

**Key Principle**: Learn how to best combine base model predictions.

### Basic Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Define base models
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Define meta-model
meta_model = LogisticRegression(random_state=42)

# Create stacking classifier
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,  # Cross-validation for base predictions
    stack_method='predict_proba',  # Use probabilities
    n_jobs=-1
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Detailed Stacking Process

**Step 1: Train Base Models**
```python
from sklearn.model_selection import cross_val_predict

# Train base models and get out-of-fold predictions
base_predictions = {}

for name, model in base_models:
    # Get cross-validated predictions (out-of-fold)
    pred = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
    base_predictions[name] = pred
    print(f"{name} trained")

# Combine predictions
X_meta = np.hstack([base_predictions[name] for name, _ in base_models])
print(f"Meta features shape: {X_meta.shape}")
```

**Step 2: Train Meta-Model**
```python
# Train meta-model on base predictions
meta_model.fit(X_meta, y_train)

# Evaluate
meta_score = meta_model.score(X_meta, y_train)
print(f"Meta-model training score: {meta_score:.3f}")
```

**Step 3: Make Predictions**
```python
# Get base model predictions on test set
test_base_predictions = {}
for name, model in base_models:
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)
    test_base_predictions[name] = pred

# Combine test predictions
X_test_meta = np.hstack([test_base_predictions[name] for name, _ in base_models])

# Meta-model prediction
final_predictions = meta_model.predict(X_test_meta)
```

### Stacking Variations

**1. Multi-Level Stacking:**
```python
# Level 1: Base models
level1_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42))
]

# Level 2: Meta-model on Level 1 predictions
level2_model = LogisticRegression(random_state=42)

# Create multi-level stacking
stacking = StackingClassifier(
    estimators=level1_models,
    final_estimator=level2_model,
    cv=5
)
```

**2. Blending (Hold-out Validation):**
```python
# Split data for blending
X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models on training set
base_predictions_blend = {}
for name, model in base_models:
    model.fit(X_train_blend, y_train_blend)
    pred = model.predict_proba(X_val_blend)
    base_predictions_blend[name] = pred

# Train meta-model on validation predictions
X_meta_blend = np.hstack([base_predictions_blend[name] for name, _ in base_models])
meta_model.fit(X_meta_blend, y_val_blend)
```

**3. Different Meta-Models:**
```python
# Try different meta-models
meta_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=50, random_state=42)
}

for name, meta in meta_models.items():
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=5
    )
    stacking.fit(X_train, y_train)
    score = stacking.score(X_test, y_test)
    print(f"{name} as meta-model: {score:.3f}")
```

### Stacking Best Practices

**1. Diverse Base Models:**
```python
# Use different algorithms
diverse_models = [
    ('tree', DecisionTreeClassifier(random_state=42)),
    ('linear', LogisticRegression(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier()),
    ('nb', GaussianNB())
]
```

**2. Use Probabilities:**
```python
# Better than hard predictions
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method='predict_proba',  # Use probabilities
    cv=5
)
```

**3. Cross-Validation:**
```python
# Always use CV to prevent overfitting
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,  # Use cross-validation
    passthrough=False  # Don't include original features
)
```

**4. Feature Engineering for Meta-Model:**
```python
# Can include original features
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True  # Include original features in meta-model
)
```

### Stacking vs Blending

| Aspect | Stacking | Blending |
|--------|----------|----------|
| **Validation** | Cross-validation | Hold-out set |
| **Data Usage** | Uses all data | Splits data |
| **Complexity** | More complex | Simpler |
| **Overfitting Risk** | Lower (CV) | Higher (single split) |

### Key Takeaways

1. **Base Models**: Should be diverse and complementary
2. **Meta-Model**: Learns optimal combination
3. **Cross-Validation**: Essential to prevent overfitting
4. **Probabilities**: Better than hard predictions
5. **Diversity**: More important than individual performance

**How it works:**
1. Train multiple diverse base models
2. Use cross-validation to get out-of-fold predictions from base models
3. Train meta-learner on these predictions
4. Final prediction: meta-learner predicts from base model predictions

### Basic Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Base models (should be diverse)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# Meta-learner (usually simple model)
meta_learner = LogisticRegression(random_state=42, max_iter=1000)

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,                    # Cross-validation folds
    stack_method='predict_proba',  # 'predict', 'predict_proba', or 'decision_function'
    n_jobs=-1
)
stacking.fit(X_train_scaled, y_train)
y_pred = stacking.predict(X_test_scaled)
print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Compare with individual models
print("\nIndividual Model Performance:")
for name, model in base_models:
    if name == 'svm':
        model.fit(X_train_scaled, y_train)
        score = accuracy_score(y_test, model.predict(X_test_scaled))
    else:
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
    print(f"  {name}: {score:.3f}")
```

### Multi-Level Stacking

```python
# Level 1: Base models
level1_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# Level 2: Stack level 1 models
level2_model = StackingClassifier(
    estimators=level1_models,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

# Level 3: Final meta-learner
final_models = [
    ('level2', level2_model),
    ('svm', SVC(probability=True, random_state=42))
]

final_stacking = StackingClassifier(
    estimators=final_models,
    final_estimator=LogisticRegression(random_state=42),
    cv=5
)

final_stacking.fit(X_train_scaled, y_train)
y_pred = final_stacking.predict(X_test_scaled)
print(f"Multi-Level Stacking Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Blending

Similar to stacking but uses a hold-out validation set instead of CV.

```python
# Split training data
X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train base models on training set
base_predictions = []
for name, model in base_models:
    if name == 'svm':
        model.fit(X_train_blend, y_train_blend)
        pred = model.predict_proba(X_val_blend)
    else:
        model.fit(X_train_blend, y_train_blend)
        pred = model.predict_proba(X_val_blend)
    base_predictions.append(pred)

# Stack predictions
X_meta = np.hstack(base_predictions)

# Train meta-learner on validation predictions
meta_learner = LogisticRegression(random_state=42)
meta_learner.fit(X_meta, y_val_blend)

# Final predictions
final_base_preds = []
for name, model in base_models:
    if name == 'svm':
        pred = model.predict_proba(X_test)
    else:
        pred = model.predict_proba(X_test)
    final_base_preds.append(pred)

X_test_meta = np.hstack(final_base_preds)
y_pred_blend = meta_learner.predict(X_test_meta)
print(f"Blending Accuracy: {accuracy_score(y_test, y_pred_blend):.3f}")
```

---

## Voting

Simple ensemble method that combines predictions from multiple models.

### Hard Voting

Majority class wins. Each model votes for a class, most votes win.

```python
from sklearn.ensemble import VotingClassifier

voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='hard',
    weights=None  # Equal weights, or specify [2, 1, 1] for weighted voting
)
voting_hard.fit(X_train_scaled, y_train)
y_pred = voting_hard.predict(X_test_scaled)
print(f"Hard Voting Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Soft Voting

Average probabilities. Each model provides probability estimates, average them.

```python
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),  # Need probability=True
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft',
    weights=[2, 1, 1]  # Weighted average (RF gets 2x weight)
)
voting_soft.fit(X_train_scaled, y_train)
y_pred = voting_soft.predict(X_test_scaled)
print(f"Soft Voting Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Compare hard vs soft voting
print(f"\nHard Voting: {accuracy_score(y_test, voting_hard.predict(X_test_scaled)):.3f}")
print(f"Soft Voting: {accuracy_score(y_test, y_pred):.3f}")
```

### When to Use Voting

**Hard Voting:**
- Models don't provide probabilities
- Simple and fast
- Works well when models are diverse

**Soft Voting:**
- Models provide probabilities
- Usually performs better
- Can weight models differently

---

## Ensemble Comparison

### When to Use Each Method

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Bagging** | High-variance models, parallel training | Reduces variance, fast | Doesn't reduce bias |
| **Boosting** | High-bias models, sequential improvement | Reduces bias, powerful | Can overfit, slower |
| **Stacking** | Diverse base models, best performance | Very flexible, often best | Complex, slow |
| **Voting** | Quick ensemble, diverse models | Simple, fast | Less sophisticated |

### Algorithm Selection Guide

```
Need best performance?
│
├─ YES → Continue
│  │
│  ├─ Have diverse models?
│  │  ├─ YES → Stacking
│  │  └─ NO → Continue
│  │
│  ├─ Need fast training?
│  │  ├─ YES → Bagging (Random Forest)
│  │  └─ NO → Boosting (XGBoost, LightGBM)
│  │
│  └─ Want simplicity?
│     └─ YES → Voting
│
└─ NO → Use single best model
```

## Practice Exercises

### Exercise 1: Compare Ensemble Methods

**Task:** Compare Bagging, Boosting, Voting, and Stacking on a dataset.

**Solution:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Voting': VotingClassifier(estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ], voting='soft'),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier())
        ],
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
}

results = {}
for name, model in models.items():
    if name in ['Voting', 'Stacking']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_model, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Test score
    model.fit(X_train_model, y_train)
    test_score = accuracy_score(y_test, model.predict(X_test_model))
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score
    }
    
    print(f"{name:20s}: CV={cv_scores.mean():.3f} (+/- {cv_scores.std():.3f}), "
          f"Test={test_score:.3f}")

# Find best model
best_model = max(results, key=lambda x: results[x]['test_score'])
print(f"\nBest Model: {best_model}")
```

### Exercise 2: Tune Ensemble Hyperparameters

**Task:** Tune Random Forest and XGBoost hyperparameters.

**Solution:**
```python
from sklearn.model_selection import GridSearchCV

# Tune Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")
print(f"Best RF score: {rf_grid.best_score_:.3f}")

# Tune XGBoost (if available)
try:
    import xgboost as xgb
    
    xgb_param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=42),
        xgb_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    xgb_grid.fit(X_train, y_train)
    print(f"\nBest XGBoost params: {xgb_grid.best_params_}")
    print(f"Best XGBoost score: {xgb_grid.best_score_:.3f}")
except ImportError:
    print("\nXGBoost not available")
```

### Exercise 3: Build Winning Ensemble

**Task:** Create a multi-level ensemble combining different methods.

**Solution:**
```python
# Level 1: Diverse base models
level1_rf = RandomForestClassifier(n_estimators=100, random_state=42)
level1_gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
level1_svm = SVC(probability=True, random_state=42)

# Train level 1
level1_rf.fit(X_train, y_train)
level1_gb.fit(X_train, y_train)
level1_svm.fit(X_train_scaled, y_train)

# Get level 1 predictions
rf_pred = level1_rf.predict_proba(X_test)
gb_pred = level1_gb.predict_proba(X_test)
svm_pred = level1_svm.predict_proba(X_test_scaled)

# Level 2: Combine with voting
level2_input = np.hstack([rf_pred, gb_pred, svm_pred])
level2_model = LogisticRegression(random_state=42, max_iter=1000)
level2_model.fit(level2_input, y_test)  # In practice, use validation set

# Final prediction
final_pred = level2_model.predict(level2_input)
print(f"Multi-Level Ensemble Accuracy: {accuracy_score(y_test, final_pred):.3f}")
```

---

## Key Takeaways

1. **Bagging**: Reduces variance, parallel training, good for high-variance models
2. **Boosting**: Reduces bias, sequential training, powerful but can overfit
3. **Stacking**: Meta-learning approach, often best performance, most complex
4. **Voting**: Simple ensemble, fast, good starting point
5. **Diversity matters**: Ensembles work best with diverse base models
6. **Hyperparameter tuning**: Critical for ensemble performance
7. **Computational cost**: Ensembles are more expensive but often worth it

---

## Resources and Further Learning

### Books

1. **"Ensemble Methods: Foundations and Algorithms"** - Zhi-Hua Zhou
   - Comprehensive textbook on ensemble methods
   - Covers theory and practice

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani & Friedman
   - [Free Online](https://web.stanford.edu/~hastie/ElemStatLearn/)
   - Chapter 8: Model Inference and Averaging
   - Chapter 15: Random Forests

3. **"Hands-On Machine Learning"** - Aurélien Géron
   - [Book Website](https://github.com/ageron/handson-ml2)
   - Chapter 7: Ensemble Learning and Random Forests

### Important Papers

1. **"Random Forests"** - Breiman, 2001
2. **"Gradient Boosting Machines"** - Friedman, 2001
3. **"XGBoost: A Scalable Tree Boosting System"** - Chen & Guestrin, 2016
4. **"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"** - Ke et al., 2017
5. **"CatBoost: Unbiased Boosting with Categorical Features"** - Prokhorenkova et al., 2018
6. **"Stacked Generalization"** - Wolpert, 1992

### Online Courses

1. **Ensemble Methods** - Coursera (University of Washington)
   - Part of Machine Learning Specialization
   - Covers bagging, boosting, stacking

2. **Kaggle Learn: Intro to Machine Learning**
   - [Course Link](https://www.kaggle.com/learn/intro-to-machine-learning)
   - Includes ensemble methods

### Datasets

1. **Kaggle Competitions**: Great for practicing ensemble methods
   - [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - [Titanic](https://www.kaggle.com/c/titanic)
   - [Porto Seguro's Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)

2. **UCI Machine Learning Repository**
   - [Dataset Collection](https://archive.ics.uci.edu/ml/datasets.php)

### Tools and Libraries

1. **scikit-learn**: Ensemble methods
   - [Documentation](https://scikit-learn.org/)
   - Random Forest, Gradient Boosting, Voting, Stacking

2. **XGBoost**: Extreme Gradient Boosting
   - [Documentation](https://xgboost.readthedocs.io/)
   - Highly optimized gradient boosting

3. **LightGBM**: Light Gradient Boosting Machine
   - [Documentation](https://lightgbm.readthedocs.io/)
   - Fast and memory-efficient

4. **CatBoost**: Categorical Boosting
   - [Documentation](https://catboost.ai/)
   - Handles categorical features natively

---

## Next Steps

- Practice with real datasets (Kaggle competitions)
- Experiment with hyperparameters for each method
- Try combining different ensemble methods
- Move to [07-feature-engineering](../07-feature-engineering/README.md)

**Remember**: Ensembles often win competitions! Diversity and proper tuning are key to success.

