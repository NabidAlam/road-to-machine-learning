# Advanced Model Evaluation & Optimization Topics

Comprehensive guide to advanced evaluation techniques, optimization strategies, and best practices.

## Table of Contents

- [Nested Cross-Validation](#nested-cross-validation)
- [Custom Scoring Functions](#custom-scoring-functions)
- [Model Selection Strategies](#model-selection-strategies)
- [Early Stopping](#early-stopping)
- [Ensemble Model Selection](#ensemble-model-selection)
- [Performance Profiling](#performance-profiling)
- [Common Evaluation Pitfalls](#common-evaluation-pitfalls)

---

## Nested Cross-Validation

### Why Nested CV?

Standard cross-validation can overfit hyperparameters to the validation set. Nested CV provides unbiased performance estimates.

### Implementation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

# Outer CV: For final performance estimate
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: For hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

# Parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

outer_scores = []

for train_idx, test_idx in outer_cv.split(X):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    # Inner CV: Hyperparameter tuning
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=inner_cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_outer, y_train_outer)
    
    # Evaluate best model on outer test fold
    best_model = grid_search.best_estimator_
    score = best_model.score(X_test_outer, y_test_outer)
    outer_scores.append(score)
    
    print(f"Fold score: {score:.3f}, Best params: {grid_search.best_params_}")

print(f"\nNested CV Mean Score: {np.mean(outer_scores):.3f} (+/- {np.std(outer_scores):.3f})")
```

### When to Use

- Need unbiased performance estimate
- Hyperparameter tuning is part of model selection
- Comparing different algorithms fairly
- Research or publication

---

## Custom Scoring Functions

### Creating Custom Metrics

```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

# Example: Custom F1-score for multiclass
def custom_f1_score(y_true, y_pred):
    """Custom F1-score calculation"""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average='weighted')

# Create scorer
custom_scorer = make_scorer(custom_f1_score, greater_is_better=True)

# Use in cross-validation
scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring=custom_scorer
)
```

### Business-Specific Metrics

```python
# Example: Cost-sensitive metric
def cost_sensitive_score(y_true, y_pred):
    """Custom metric that penalizes false negatives more"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Cost matrix: [TN, FP, FN, TP]
    # Higher cost for false negatives
    cost_matrix = np.array([[0, 1, 5, 0]])  # FN costs 5x more
    costs = cm.ravel() * cost_matrix.ravel()
    
    return -np.sum(costs)  # Negative because we want to minimize

cost_scorer = make_scorer(cost_sensitive_score, greater_is_better=True)
```

### Regression Custom Metrics

```python
# Example: Percentage error
def percentage_error(y_true, y_pred):
    """Mean absolute percentage error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

percentage_scorer = make_scorer(percentage_error, greater_is_better=False)

# Use in grid search
grid_search = GridSearchCV(
    model, param_grid,
    cv=5,
    scoring=percentage_scorer
)
```

---

## Model Selection Strategies

### Multiple Models Comparison

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"{name:20s}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Select best model
best_model_name = max(results, key=lambda x: results[x]['mean'])
print(f"\nBest model: {best_model_name}")
```

### Statistical Significance Testing

```python
from scipy import stats

# Compare two models
model1_scores = results['Logistic Regression']['scores']
model2_scores = results['Random Forest']['scores']

# Paired t-test
t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")

if p_value < 0.05:
    print("Difference is statistically significant")
    if model1_scores.mean() > model2_scores.mean():
        print("Logistic Regression is significantly better")
    else:
        print("Random Forest is significantly better")
else:
    print("No significant difference")
```

---

## Early Stopping

### For Iterative Algorithms

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Gradient Boosting with early stopping
gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    validation_fraction=0.2,
    n_iter_no_change=10,  # Stop if no improvement for 10 iterations
    tol=0.0001,  # Minimum improvement threshold
    random_state=42
)

gb.fit(X_train, y_train)

print(f"Number of estimators used: {gb.n_estimators_}")
print(f"Best validation score: {gb.train_score_[-1]:.3f}")
```

### Manual Early Stopping

```python
from sklearn.metrics import accuracy_score

best_score = 0
best_model = None
patience = 10
no_improve = 0

for n_est in range(50, 1000, 50):
    model = GradientBoostingClassifier(
        n_estimators=n_est,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    val_score = model.score(X_val, y_val)
    
    if val_score > best_score:
        best_score = val_score
        best_model = model
        no_improve = 0
    else:
        no_improve += 1
        
    if no_improve >= patience:
        print(f"Early stopping at {n_est} estimators")
        break

print(f"Best validation score: {best_score:.3f}")
```

---

## Ensemble Model Selection

### Voting Classifier Tuning

```python
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Base models
models = [
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Voting classifier
voting_clf = VotingClassifier(estimators=models, voting='soft')

# Tune individual models
param_grid = {
    'lr__C': [0.1, 1, 10],
    'rf__max_depth': [5, 10, None],
    'svm__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    voting_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

### Stacking with Hyperparameter Tuning

```python
from sklearn.ensemble import StackingClassifier

# Base models
base_models = [
    ('lr', LogisticRegression(random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

# Meta-learner
meta_model = LogisticRegression(random_state=42)

# Stacking classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Tune meta-learner
param_grid = {
    'final_estimator__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(
    stacking_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

---

## Performance Profiling

### Model Training Time

```python
import time
from sklearn.model_selection import cross_val_score

def profile_model(model, X, y, cv=5):
    """Profile model training time"""
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    elapsed_time = time.time() - start_time
    
    return {
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'time': elapsed_time,
        'time_per_fold': elapsed_time / cv
    }

# Profile multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC()
}

results = {}
for name, model in models.items():
    results[name] = profile_model(model, X_train, y_train)
    print(f"{name:20s}: Score={results[name]['mean_score']:.3f}, "
          f"Time={results[name]['time']:.2f}s")
```

### Memory Usage

```python
import tracemalloc

def profile_memory(model, X, y):
    """Profile model memory usage"""
    tracemalloc.start()
    
    model.fit(X, y)
    current, peak = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    return {
        'current_mb': current / 1024 / 1024,
        'peak_mb': peak / 1024 / 1024
    }

# Profile memory
for name, model in models.items():
    memory = profile_memory(model, X_train, y_train)
    print(f"{name:20s}: Peak memory={memory['peak_mb']:.2f} MB")
```

---

## Common Evaluation Pitfalls

### Pitfall 1: Data Leakage in Cross-Validation

```python
# WRONG: Scaling before cross-validation
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses all data!
scores = cross_val_score(model, X_scaled, y, cv=5)

# CORRECT: Scale inside cross-validation
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
scores = cross_val_score(pipeline, X, y, cv=5)
```

### Pitfall 2: Using Test Set for Model Selection

```python
# WRONG: Tuning on test set
grid_search.fit(X_train, y_train)
test_score = grid_search.score(X_test, y_test)  # Used for tuning!

# CORRECT: Use validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2
)
grid_search.fit(X_train, y_train)
val_score = grid_search.score(X_val, y_val)  # Tune on validation

# Only use test set at the end
final_score = grid_search.best_estimator_.score(X_test, y_test)
```

### Pitfall 3: Not Stratifying

```python
# WRONG: For imbalanced data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# CORRECT: Stratify for classification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Verify distribution
print("Train distribution:", pd.Series(y_train).value_counts(normalize=True))
print("Test distribution:", pd.Series(y_test).value_counts(normalize=True))
```

### Pitfall 4: Overfitting to Validation Set

```python
# WRONG: Too many hyperparameter combinations
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
}  # 42 combinations - high risk of overfitting

# CORRECT: Reasonable grid
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1]
}  # 9 combinations - more reasonable
```

---

## Key Takeaways

1. **Nested CV**: Provides unbiased performance estimates when tuning hyperparameters
2. **Custom Metrics**: Create business-specific evaluation functions
3. **Statistical Testing**: Compare models with significance tests
4. **Early Stopping**: Prevent overfitting in iterative algorithms
5. **Performance Profiling**: Understand time and memory tradeoffs
6. **Avoid Pitfalls**: Watch out for data leakage, test set contamination, and overfitting

---

## Next Steps

- Practice nested cross-validation
- Create custom scoring functions for your domain
- Compare models statistically
- Profile model performance
- Move to ensemble methods module

**Remember**: Proper evaluation is crucial for reliable model performance estimates!

