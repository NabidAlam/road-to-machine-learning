# Complete Model Evaluation & Optimization Project Tutorial

Step-by-step walkthrough of properly evaluating and optimizing a machine learning model.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Initial Split](#step-1-data-loading-and-initial-split)
- [Step 2: Baseline Model Evaluation](#step-2-baseline-model-evaluation)
- [Step 3: Cross-Validation Analysis](#step-3-cross-validation-analysis)
- [Step 4: Hyperparameter Tuning](#step-4-hyperparameter-tuning)
- [Step 5: Learning Curve Analysis](#step-5-learning-curve-analysis)
- [Step 6: Bias-Variance Analysis](#step-6-bias-variance-analysis)
- [Step 7: Final Model Selection and Evaluation](#step-7-final-model-selection-and-evaluation)

---

## Project Overview

**Project**: Model Evaluation and Optimization for Classification

**Dataset**: Iris Dataset (or any classification dataset)

**Goal**: Properly evaluate models, tune hyperparameters, and diagnose issues

**Type**: Classification with comprehensive evaluation

**Difficulty**: Intermediate

**Time**: 1-2 hours

---

## Step 1: Data Loading and Initial Split

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                    KFold, StratifiedKFold, learning_curve,
                                    GridSearchCV, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Dataset loaded successfully!")
print(f"Shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
print(f"\nClass distribution:")
print(pd.Series(y).value_counts())
```

### Initial Train-Test Split

```python
# Initial split: 80% train, 20% test
# This test set will ONLY be used at the very end
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train_full.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Verify stratification
print("\nOriginal distribution:")
print(pd.Series(y).value_counts(normalize=True))
print("\nTrain distribution:")
print(pd.Series(y_train_full).value_counts(normalize=True))
print("\nTest distribution:")
print(pd.Series(y_test).value_counts(normalize=True))
```

### Split Training into Train and Validation

```python
# Split training data into train and validation
# Validation set for hyperparameter tuning
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print(f"\nFinal splits:")
print(f"Training: {X_train.shape[0]} samples (60% of total)")
print(f"Validation: {X_val.shape[0]} samples (20% of total)")
print(f"Test: {X_test.shape[0]} samples (20% of total)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
```

---

## Step 2: Baseline Model Evaluation

### Train Multiple Baseline Models

```python
# Initialize baseline models
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

baseline_results = {}

print("Baseline Model Performance:")
print("=" * 60)

for name, model in baseline_models.items():
    # Use scaled data for Logistic Regression and SVM
    if name in ['Logistic Regression', 'SVM']:
        X_train_model = X_train_scaled
        X_val_model = X_val_scaled
    else:
        X_train_model = X_train
        X_val_model = X_val
    
    # Train
    model.fit(X_train_model, y_train)
    
    # Evaluate on validation set
    train_score = model.score(X_train_model, y_train)
    val_score = model.score(X_val_model, y_val)
    
    baseline_results[name] = {
        'model': model,
        'train_score': train_score,
        'val_score': val_score,
        'gap': train_score - val_score
    }
    
    print(f"\n{name}:")
    print(f"  Train Accuracy: {train_score:.3f}")
    print(f"  Validation Accuracy: {val_score:.3f}")
    print(f"  Gap: {baseline_results[name]['gap']:.3f}")

# Find best baseline
best_baseline = max(baseline_results, key=lambda x: baseline_results[x]['val_score'])
print(f"\nBest Baseline Model: {best_baseline}")
print(f"Validation Accuracy: {baseline_results[best_baseline]['val_score']:.3f}")
```

### Visualize Baseline Comparison

```python
# Visualize baseline results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy comparison
model_names = list(baseline_results.keys())
train_scores = [baseline_results[m]['train_score'] for m in model_names]
val_scores = [baseline_results[m]['val_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, train_scores, width, label='Train', alpha=0.8)
axes[0].bar(x + width/2, val_scores, width, label='Validation', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Baseline Model Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Gap visualization
gaps = [baseline_results[m]['gap'] for m in model_names]
axes[1].barh(model_names, gaps, color='salmon')
axes[1].set_xlabel('Train - Validation Gap', fontsize=12)
axes[1].set_title('Overfitting Indicator (Larger = More Overfitting)', 
                  fontsize=14, fontweight='bold')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Step 3: Cross-Validation Analysis

### K-Fold Cross-Validation

```python
# Perform cross-validation on training data
cv_results = {}

print("Cross-Validation Results:")
print("=" * 60)

# Use StratifiedKFold for classification
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in baseline_models.items():
    if name in ['Logistic Regression', 'SVM']:
        X_model = X_train_scaled
    else:
        X_model = X_train
    
    scores = cross_val_score(
        model, X_model, y_train,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    cv_results[name] = {
        'scores': scores,
        'mean': scores.mean(),
        'std': scores.std()
    }
    
    print(f"\n{name}:")
    print(f"  CV Scores: {scores}")
    print(f"  Mean: {cv_results[name]['mean']:.3f} (+/- {cv_results[name]['std']:.3f})")

# Visualize CV results
fig, ax = plt.subplots(figsize=(12, 6))
positions = np.arange(len(cv_results))
means = [cv_results[m]['mean'] for m in cv_results.keys()]
stds = [cv_results[m]['std'] for m in cv_results.keys()]

ax.barh(list(cv_results.keys()), means, xerr=stds, capsize=5, alpha=0.8)
ax.set_xlabel('Cross-Validated Accuracy', fontsize=12)
ax.set_title('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()
```

### Compare CV with Single Split

```python
# Compare CV estimate with validation set performance
comparison = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'CV Mean': [cv_results[m]['mean'] for m in cv_results.keys()],
    'CV Std': [cv_results[m]['std'] for m in cv_results.keys()],
    'Validation Score': [baseline_results[m]['val_score'] for m in cv_results.keys()]
})

comparison['Difference'] = comparison['CV Mean'] - comparison['Validation Score']

print("\nCV vs Validation Set Comparison:")
print(comparison)

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comparison))
width = 0.35

ax.bar(x - width/2, comparison['CV Mean'], width, label='CV Mean', alpha=0.8)
ax.bar(x + width/2, comparison['Validation Score'], width, label='Validation', alpha=0.8)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Cross-Validation vs Validation Set', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Step 4: Hyperparameter Tuning

### Grid Search for Random Forest

```python
# Focus on Random Forest (best baseline)
print("Hyperparameter Tuning: Random Forest")
print("=" * 60)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Evaluate on validation set
best_rf = grid_search.best_estimator_
val_score_tuned = best_rf.score(X_val, y_val)
print(f"Validation score: {val_score_tuned:.3f}")

# Compare with baseline
baseline_val = baseline_results['Random Forest']['val_score']
print(f"\nImprovement: {val_score_tuned - baseline_val:.3f}")
```

### Random Search Comparison

```python
from scipy.stats import randint

# Random search
param_dist = {
    'n_estimators': randint(50, 201),
    'max_depth': [3, 5, 10, None],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=50,  # Sample 50 combinations
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"\nRandom Search Best parameters: {random_search.best_params_}")
print(f"Random Search Best CV score: {random_search.best_score_:.3f}")

# Compare methods
print(f"\nComparison:")
print(f"Grid Search CV score: {grid_search.best_score_:.3f}")
print(f"Random Search CV score: {random_search.best_score_:.3f}")
```

---

## Step 5: Learning Curve Analysis

### Plot Learning Curves

```python
def plot_learning_curves_comparison(models, X, y, title_suffix=""):
    """Plot learning curves for multiple models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models.items()):
        if name in ['Logistic Regression', 'SVM']:
            X_model = X_train_scaled
        else:
            X_model = X_train
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_model, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', 
                      label='Training', linewidth=2)
        axes[idx].fill_between(train_sizes, train_mean - train_std,
                              train_mean + train_std, alpha=0.1, color='blue')
        axes[idx].plot(train_sizes, val_mean, 'o-', color='red',
                      label='Validation', linewidth=2)
        axes[idx].fill_between(train_sizes, val_mean - val_std,
                              val_mean + val_std, alpha=0.1, color='red')
        axes[idx].set_xlabel('Training Set Size', fontsize=11)
        axes[idx].set_ylabel('Accuracy', fontsize=11)
        axes[idx].set_title(f'{name}{title_suffix}', fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot learning curves for baseline models
plot_learning_curves_comparison(baseline_models, X_train, y_train, " - Baseline")

# Plot learning curve for tuned model
tuned_models = {
    'Random Forest (Tuned)': grid_search.best_estimator_
}
plot_learning_curves_comparison(tuned_models, X_train, y_train, " - Tuned")
```

### Diagnose Issues from Learning Curves

```python
# Analyze learning curves
def diagnose_learning_curve(model, X, y, name):
    """Diagnose model issues from learning curve"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val
    
    print(f"\n{name} Diagnosis:")
    print(f"  Final Training Score: {final_train:.3f}")
    print(f"  Final Validation Score: {final_val:.3f}")
    print(f"  Gap: {gap:.3f}")
    
    if final_val < 0.7 and gap < 0.1:
        print("  -> UNDERFITTING: Both scores low, small gap")
        print("  -> Solution: More complex model, better features")
    elif gap > 0.15:
        print("  -> OVERFITTING: Large gap between train and validation")
        print("  -> Solution: Regularization, more data, simpler model")
    else:
        print("  -> GOOD FIT: Both scores high, small gap")

# Diagnose all models
for name, model in baseline_models.items():
    if name in ['Logistic Regression', 'SVM']:
        X_model = X_train_scaled
    else:
        X_model = X_train
    diagnose_learning_curve(model, X_model, y_train, name)
```

---

## Step 6: Bias-Variance Analysis

### Analyze Bias-Variance Tradeoff

```python
# Test different model complexities
max_depths = range(1, 21)
train_scores = []
val_scores = []
test_scores = []

for depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(tree.score(X_train, y_train))
    val_scores.append(tree.score(X_val, y_val))
    test_scores.append(tree.score(X_test, y_test))  # Only for analysis

# Plot bias-variance tradeoff
plt.figure(figsize=(12, 6))
plt.plot(max_depths, train_scores, 'o-', label='Train', linewidth=2)
plt.plot(max_depths, val_scores, 's-', label='Validation', linewidth=2)
plt.plot(max_depths, test_scores, '^-', label='Test', linewidth=2, alpha=0.5)
plt.xlabel('Max Depth (Model Complexity)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=5, color='green', linestyle='--', label='Optimal (estimated)')
plt.legend()
plt.tight_layout()
plt.show()

# Find optimal depth
optimal_idx = np.argmax(val_scores)
optimal_depth = max_depths[optimal_idx]
print(f"\nOptimal depth (based on validation): {optimal_depth}")
print(f"Validation score at optimal depth: {val_scores[optimal_idx]:.3f}")
```

---

## Step 7: Final Model Selection and Evaluation

### Select Best Model

```python
# Compare all tuned models
final_models = {
    'Baseline Random Forest': baseline_results['Random Forest']['model'],
    'Tuned Random Forest (Grid)': grid_search.best_estimator_,
    'Tuned Random Forest (Random)': random_search.best_estimator_
}

final_comparison = {}

print("Final Model Comparison:")
print("=" * 60)

for name, model in final_models.items():
    # Cross-validation on full training set
    cv_scores = cross_val_score(
        model, X_train_full, y_train_full,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1
    )
    
    final_comparison[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }
    
    print(f"\n{name}:")
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Select best model
best_model_name = max(final_comparison, key=lambda x: final_comparison[x]['cv_mean'])
best_model = final_comparison[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"CV Score: {final_comparison[best_model_name]['cv_mean']:.3f}")
```

### Final Evaluation on Test Set

```python
# NOW we can use the test set - only once!
print("\nFinal Evaluation on Test Set:")
print("=" * 60)

# Retrain best model on full training set
if best_model_name in ['Tuned Random Forest (Grid)', 'Tuned Random Forest (Random)']:
    # For tuned models, retrain with best parameters
    best_model.fit(X_train_full, y_train_full)
else:
    best_model.fit(X_train_full, y_train_full)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None

# Evaluation
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# Compare with CV estimate
cv_estimate = final_comparison[best_model_name]['cv_mean']
print(f"\nCV Estimate: {cv_estimate:.3f}")
print(f"Test Score: {test_accuracy:.3f}")
print(f"Difference: {abs(test_accuracy - cv_estimate):.3f}")

if abs(test_accuracy - cv_estimate) < 0.05:
    print("-> Good: Test score close to CV estimate (model generalizes well)")
else:
    print("-> Warning: Large difference (possible overfitting to CV)")
```

### Summary

```python
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"\n1. Baseline Models Evaluated: {len(baseline_models)}")
print(f"2. Best Baseline: {best_baseline} ({baseline_results[best_baseline]['val_score']:.3f})")
print(f"3. Hyperparameter Tuning: Grid Search and Random Search")
print(f"4. Best Model: {best_model_name}")
print(f"5. Cross-Validation Score: {final_comparison[best_model_name]['cv_mean']:.3f}")
print(f"6. Final Test Score: {test_accuracy:.3f}")
print(f"\n7. Key Learnings:")
print(f"   - Used proper train/validation/test split")
print(f"   - Applied cross-validation for reliable estimates")
print(f"   - Tuned hyperparameters on validation set")
print(f"   - Analyzed learning curves and bias-variance tradeoff")
print(f"   - Only touched test set at the very end")
```

---

## Key Takeaways

1. **Proper Data Splitting**: Train (60%), Validation (20%), Test (20%)
2. **Baseline Evaluation**: Always start with baseline models
3. **Cross-Validation**: More reliable than single split
4. **Hyperparameter Tuning**: Use validation set, not test set
5. **Learning Curves**: Diagnose overfitting/underfitting
6. **Bias-Variance**: Find optimal model complexity
7. **Final Evaluation**: Test set used only once at the end

---

**Congratulations!** You've completed a comprehensive model evaluation and optimization workflow!

