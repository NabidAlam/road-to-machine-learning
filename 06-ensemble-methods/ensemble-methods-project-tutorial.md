# Complete Ensemble Methods Project Tutorial

Step-by-step walkthrough of building and optimizing ensemble models for a classification problem.

## Table of Contents

- [Project Overview](#project-overview)
- [Step 1: Data Loading and Preparation](#step-1-data-loading-and-preparation)
- [Step 2: Baseline Models](#step-2-baseline-models)
- [Step 3: Bagging Ensemble](#step-3-bagging-ensemble)
- [Step 4: Boosting Ensemble](#step-4-boosting-ensemble)
- [Step 5: Voting Ensemble](#step-5-voting-ensemble)
- [Step 6: Stacking Ensemble](#step-6-stacking-ensemble)
- [Step 7: Ensemble Comparison and Selection](#step-7-ensemble-comparison-and-selection)
- [Step 8: Final Ensemble and Evaluation](#step-8-final-ensemble-and-evaluation)

---

## Project Overview

**Project**: Build Winning Ensemble for Classification

**Dataset**: Wine Quality Dataset (or any classification dataset)

**Goal**: Build and compare different ensemble methods to achieve best performance

**Type**: Classification with Multiple Ensemble Methods

**Difficulty**: Intermediate to Advanced

**Time**: 2-3 hours

---

## Step 1: Data Loading and Preparation

### Load Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, VotingClassifier,
                              StackingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                            confusion_matrix, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print("Dataset loaded successfully!")
print(f"Shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")
print(f"\nClass distribution:")
print(pd.Series(y).value_counts())
```

### Data Preparation

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

---

## Step 2: Baseline Models

### Train Individual Models

```python
# Initialize baseline models
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

baseline_results = {}

print("Baseline Model Performance:")
print("=" * 60)

for name, model in baseline_models.items():
    # Use scaled data for Logistic Regression and SVM
    if name in ['Logistic Regression', 'SVM']:
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
    
    # Train and evaluate
    model.fit(X_train_model, y_train)
    test_score = accuracy_score(y_test, model.predict(X_test_model))
    
    baseline_results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score
    }
    
    print(f"\n{name}:")
    print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test Score: {test_score:.3f}")

# Find best baseline
best_baseline = max(baseline_results, key=lambda x: baseline_results[x]['test_score'])
print(f"\nBest Baseline: {best_baseline} ({baseline_results[best_baseline]['test_score']:.3f})")
```

### Visualize Baseline Results

```python
# Visualize baseline comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CV scores
model_names = list(baseline_results.keys())
cv_means = [baseline_results[m]['cv_mean'] for m in model_names]
cv_stds = [baseline_results[m]['cv_std'] for m in model_names]
test_scores = [baseline_results[m]['test_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

axes[0].bar(x - width/2, cv_means, width, yerr=cv_stds, capsize=5,
           label='CV Mean', alpha=0.8)
axes[0].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Baseline Model Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Test scores comparison
axes[1].barh(model_names, test_scores, color='skyblue', alpha=0.8)
axes[1].set_xlabel('Test Accuracy', fontsize=12)
axes[1].set_title('Test Set Performance', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Step 3: Bagging Ensemble

### Random Forest

```python
print("Bagging Ensemble: Random Forest")
print("=" * 60)

# Random Forest with different configurations
rf_configs = {
    'RF (50 trees)': RandomForestClassifier(n_estimators=50, random_state=42),
    'RF (100 trees)': RandomForestClassifier(n_estimators=100, random_state=42),
    'RF (200 trees)': RandomForestClassifier(n_estimators=200, random_state=42),
    'RF (tuned)': RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
}

rf_results = {}

for name, rf in rf_configs.items():
    cv_scores = cross_val_score(
        rf, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    test_score = accuracy_score(y_test, rf.predict(X_test))
    
    rf_results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score,
        'model': rf
    }
    
    print(f"\n{name}:")
    print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test: {test_score:.3f}")

best_rf = max(rf_results, key=lambda x: rf_results[x]['test_score'])
print(f"\nBest Random Forest: {best_rf} ({rf_results[best_rf]['test_score']:.3f})")
```

### Custom Bagging

```python
# Bagging with different base estimators
bagging_configs = {
    'Bagging (DT)': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=50,
        random_state=42
    ),
    'Bagging (SVM)': BaggingClassifier(
        base_estimator=SVC(probability=True),
        n_estimators=10,  # Fewer for SVM (slower)
        random_state=42
    )
}

bagging_results = {}

for name, bagging in bagging_configs.items():
    if 'SVM' in name:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test
    
    cv_scores = cross_val_score(
        bagging, X_train_model, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    bagging.fit(X_train_model, y_train)
    test_score = accuracy_score(y_test, bagging.predict(X_test_model))
    
    bagging_results[name] = {
        'cv_mean': cv_scores.mean(),
        'test_score': test_score
    }
    
    print(f"\n{name}:")
    print(f"  CV: {cv_scores.mean():.3f}")
    print(f"  Test: {test_score:.3f}")
```

---

## Step 4: Boosting Ensemble

### AdaBoost

```python
print("\nBoosting Ensemble: AdaBoost")
print("=" * 60)

adaboost_configs = {
    'AdaBoost (50)': AdaBoostClassifier(n_estimators=50, random_state=42),
    'AdaBoost (100)': AdaBoostClassifier(n_estimators=100, random_state=42),
    'AdaBoost (tuned)': AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
}

adaboost_results = {}

for name, ab in adaboost_configs.items():
    cv_scores = cross_val_score(
        ab, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    ab.fit(X_train, y_train)
    test_score = accuracy_score(y_test, ab.predict(X_test))
    
    adaboost_results[name] = {
        'cv_mean': cv_scores.mean(),
        'test_score': test_score,
        'model': ab
    }
    
    print(f"\n{name}:")
    print(f"  CV: {cv_scores.mean():.3f}")
    print(f"  Test: {test_score:.3f}")
```

### Gradient Boosting

```python
print("\nBoosting Ensemble: Gradient Boosting")
print("=" * 60)

gb_configs = {
    'GB (100)': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'GB (200)': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'GB (tuned)': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )
}

gb_results = {}

for name, gb in gb_configs.items():
    cv_scores = cross_val_score(
        gb, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    gb.fit(X_train, y_train)
    test_score = accuracy_score(y_test, gb.predict(X_test))
    
    gb_results[name] = {
        'cv_mean': cv_scores.mean(),
        'test_score': test_score,
        'model': gb
    }
    
    print(f"\n{name}:")
    print(f"  CV: {cv_scores.mean():.3f}")
    print(f"  Test: {test_score:.3f}")
```

### XGBoost (if available)

```python
try:
    import xgboost as xgb
    
    print("\nBoosting Ensemble: XGBoost")
    print("=" * 60)
    
    xgb_configs = {
        'XGBoost (default)': xgb.XGBClassifier(random_state=42),
        'XGBoost (tuned)': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    }
    
    xgb_results = {}
    
    for name, xgb_model in xgb_configs.items():
        cv_scores = cross_val_score(
            xgb_model, X_train, y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        xgb_model.fit(X_train, y_train)
        test_score = accuracy_score(y_test, xgb_model.predict(X_test))
        
        xgb_results[name] = {
            'cv_mean': cv_scores.mean(),
            'test_score': test_score,
            'model': xgb_model
        }
        
        print(f"\n{name}:")
        print(f"  CV: {cv_scores.mean():.3f}")
        print(f"  Test: {test_score:.3f}")
        
except ImportError:
    print("\nXGBoost not available. Install: pip install xgboost")
    xgb_results = {}
```

---

## Step 5: Voting Ensemble

### Hard Voting

```python
print("\nVoting Ensemble: Hard Voting")
print("=" * 60)

# Hard voting with diverse models
voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='hard'
)

cv_scores = cross_val_score(
    voting_hard, X_train, y_train,
    cv=5, scoring='accuracy', n_jobs=-1
)

voting_hard.fit(X_train, y_train)
test_score = accuracy_score(y_test, voting_hard.predict(X_test))

print(f"Hard Voting:")
print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"  Test: {test_score:.3f}")
```

### Soft Voting

```python
print("\nVoting Ensemble: Soft Voting")
print("=" * 60)

# Soft voting (needs probabilities)
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='soft',
    weights=[2, 1, 1]  # Weight RF more
)

cv_scores = cross_val_score(
    voting_soft, X_train_scaled, y_train,
    cv=5, scoring='accuracy', n_jobs=-1
)

voting_soft.fit(X_train_scaled, y_train)
test_score = accuracy_score(y_test, voting_soft.predict(X_test_scaled))

print(f"Soft Voting:")
print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(f"  Test: {test_score:.3f}")

# Compare hard vs soft
print(f"\nComparison:")
print(f"  Hard Voting: {accuracy_score(y_test, voting_hard.predict(X_test)):.3f}")
print(f"  Soft Voting: {test_score:.3f}")
```

---

## Step 6: Stacking Ensemble

### Basic Stacking

```python
print("\nStacking Ensemble")
print("=" * 60)

# Base models (should be diverse)
base_models = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Try different meta-learners
meta_learners = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Ridge': RidgeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42)
}

stacking_results = {}

for meta_name, meta_learner in meta_learners.items():
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    cv_scores = cross_val_score(
        stacking, X_train_scaled, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    stacking.fit(X_train_scaled, y_train)
    test_score = accuracy_score(y_test, stacking.predict(X_test_scaled))
    
    stacking_results[meta_name] = {
        'cv_mean': cv_scores.mean(),
        'test_score': test_score,
        'model': stacking
    }
    
    print(f"\nStacking with {meta_name}:")
    print(f"  CV: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Test: {test_score:.3f}")

best_stacking = max(stacking_results, key=lambda x: stacking_results[x]['test_score'])
print(f"\nBest Stacking: {best_stacking} ({stacking_results[best_stacking]['test_score']:.3f})")
```

---

## Step 7: Ensemble Comparison and Selection

### Compare All Ensembles

```python
print("\n" + "=" * 60)
print("ENSEMBLE COMPARISON")
print("=" * 60)

# Collect all results
all_results = {
    'Baseline (Best)': baseline_results[best_baseline],
    'Random Forest (Best)': rf_results[best_rf],
    'AdaBoost (Best)': adaboost_results[max(adaboost_results, key=lambda x: adaboost_results[x]['test_score'])],
    'Gradient Boosting (Best)': gb_results[max(gb_results, key=lambda x: gb_results[x]['test_score'])],
    'Hard Voting': {'test_score': accuracy_score(y_test, voting_hard.predict(X_test))},
    'Soft Voting': {'test_score': test_score},
    'Stacking (Best)': stacking_results[best_stacking]
}

if xgb_results:
    all_results['XGBoost (Best)'] = xgb_results[max(xgb_results, key=lambda x: xgb_results[x]['test_score'])]

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Method': list(all_results.keys()),
    'Test Score': [r.get('test_score', 0) for r in all_results.values()],
    'CV Mean': [r.get('cv_mean', 0) for r in all_results.values()]
}).sort_values('Test Score', ascending=False)

print("\nEnsemble Comparison:")
print(comparison_df.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(comparison_df))
ax.barh(y_pos, comparison_df['Test Score'], color='skyblue', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_df['Method'])
ax.set_xlabel('Test Accuracy', fontsize=12)
ax.set_title('Ensemble Methods Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, score in enumerate(comparison_df['Test Score']):
    ax.text(score + 0.005, i, f'{score:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## Step 8: Final Ensemble and Evaluation

### Select Best Ensemble

```python
# Select best ensemble
best_ensemble_name = comparison_df.iloc[0]['Method']
print(f"\nBest Ensemble: {best_ensemble_name}")
print(f"Test Accuracy: {comparison_df.iloc[0]['Test Score']:.3f}")

# Get best model
if 'Random Forest' in best_ensemble_name:
    best_model = rf_results[best_rf]['model']
elif 'AdaBoost' in best_ensemble_name:
    best_model = adaboost_results[max(adaboost_results, key=lambda x: adaboost_results[x]['test_score'])]['model']
elif 'Gradient Boosting' in best_ensemble_name:
    best_model = gb_results[max(gb_results, key=lambda x: gb_results[x]['test_score'])]['model']
elif 'Stacking' in best_ensemble_name:
    best_model = stacking_results[best_stacking]['model']
elif 'Voting' in best_ensemble_name:
    best_model = voting_soft if 'Soft' in best_ensemble_name else voting_hard
else:
    best_model = baseline_results[best_baseline]['model']

# Final evaluation
y_pred = best_model.predict(X_test_scaled if 'Stacking' in best_ensemble_name or 'Soft Voting' in best_ensemble_name else X_test)

print("\nFinal Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_ensemble_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()
```

### Feature Importance (if applicable)

```python
# Feature importance for tree-based ensembles
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {best_ensemble_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
```

### Summary

```python
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"\n1. Baseline Models Tested: {len(baseline_models)}")
print(f"2. Best Baseline: {best_baseline} ({baseline_results[best_baseline]['test_score']:.3f})")
print(f"3. Ensemble Methods Tested:")
print(f"   - Bagging (Random Forest)")
print(f"   - Boosting (AdaBoost, Gradient Boosting)")
if xgb_results:
    print(f"   - XGBoost")
print(f"   - Voting (Hard and Soft)")
print(f"   - Stacking")
print(f"4. Best Ensemble: {best_ensemble_name}")
print(f"5. Final Test Accuracy: {comparison_df.iloc[0]['Test Score']:.3f}")
print(f"6. Improvement over baseline: "
      f"{comparison_df.iloc[0]['Test Score'] - baseline_results[best_baseline]['test_score']:.3f}")

print(f"\nKey Learnings:")
print(f"- Ensembles generally outperform single models")
print(f"- Diversity in base models is important")
print(f"- Stacking often provides best performance")
print(f"- Hyperparameter tuning improves ensemble performance")
```

---

## Key Takeaways

1. **Start with baselines**: Understand individual model performance
2. **Try different ensembles**: Bagging, Boosting, Voting, Stacking
3. **Compare systematically**: Use cross-validation and test set properly
4. **Tune hyperparameters**: Critical for ensemble performance
5. **Diversity matters**: Different algorithms work better together
6. **Stacking is powerful**: Often achieves best performance
7. **Feature importance**: Understand what features matter

---

**Congratulations!** You've built and compared multiple ensemble methods!

