# Introduction to ML Quick Reference

Quick lookup guide for ML concepts, algorithm selection, and workflow.

## Table of Contents

- [Algorithm Selection Decision Tree](#algorithm-selection-decision-tree)
- [ML Workflow Checklist](#ml-workflow-checklist)
- [Problem Type Identification](#problem-type-identification)
- [Quick Terminology Lookup](#quick-terminology-lookup)
- [Common Patterns and Anti-patterns](#common-patterns-and-anti-patterns)

---

## Algorithm Selection Decision Tree

```
Need to predict something?
│
├─ Continuous value (price, temperature)?
│  └─ REGRESSION
│     ├─ Linear relationship?
│     │  └─ Linear Regression
│     ├─ Non-linear relationship?
│     │  └─ Polynomial Regression, Random Forest, XGBoost
│     └─ Need regularization?
│        └─ Ridge, Lasso, Elastic Net
│
├─ Categorical value (spam/not spam, class)?
│  └─ CLASSIFICATION
│     ├─ Binary (2 classes)?
│     │  ├─ Need interpretability?
│     │  │  └─ Logistic Regression, Decision Tree
│     │  ├─ Need best performance?
│     │  │  └─ Random Forest, XGBoost, Neural Network
│     │  └─ Small dataset?
│     │     └─ SVM, KNN
│     │
│     └─ Multi-class (3+ classes)?
│        ├─ Logistic Regression (multinomial)
│        ├─ Random Forest
│        ├─ Neural Network
│        └─ One-vs-Rest / One-vs-One
│
└─ No labels available?
   └─ UNSUPERVISED LEARNING
      ├─ Find groups?
      │  └─ Clustering (K-Means, Hierarchical, DBSCAN)
      ├─ Reduce dimensions?
      │  └─ PCA, t-SNE, UMAP
      └─ Find anomalies?
         └─ Isolation Forest, One-Class SVM
```

---

## ML Workflow Checklist

### Problem Definition
- [ ] Understand business problem
- [ ] Define success metrics
- [ ] Identify constraints (time, resources)
- [ ] Determine data requirements

### Data Collection
- [ ] Identify data sources
- [ ] Collect initial dataset
- [ ] Check data quality
- [ ] Document data sources

### Data Preparation
- [ ] Handle missing values
- [ ] Remove duplicates
- [ ] Handle outliers
- [ ] Encode categorical variables
- [ ] Scale/normalize features
- [ ] Split data (train/val/test)

### Feature Engineering
- [ ] Create new features
- [ ] Select important features
- [ ] Remove irrelevant features
- [ ] Handle feature interactions

### Model Training
- [ ] Choose baseline model
- [ ] Train model
- [ ] Evaluate on validation set
- [ ] Tune hyperparameters
- [ ] Try different algorithms
- [ ] Compare models

### Evaluation
- [ ] Evaluate on test set
- [ ] Check for overfitting
- [ ] Analyze errors
- [ ] Calculate metrics
- [ ] Compare with baseline

### Deployment
- [ ] Save model
- [ ] Create API/interface
- [ ] Deploy to production
- [ ] Set up monitoring
- [ ] Document model

### Monitoring
- [ ] Track predictions
- [ ] Monitor performance
- [ ] Check for drift
- [ ] Plan retraining

---

## Problem Type Identification

### Regression Problems

**Characteristics:**
- Target is continuous (numbers)
- Examples: Price, temperature, sales, age

**Key Indicators:**
- "Predict the value of..."
- "Estimate how much..."
- "Forecast..."

**Metrics:**
- MSE, RMSE, MAE, R²

### Classification Problems

**Characteristics:**
- Target is categorical (classes)
- Examples: Spam/Not spam, Cat/Dog/Bird, High/Medium/Low

**Key Indicators:**
- "Classify as..."
- "Identify which..."
- "Predict the category..."

**Metrics:**
- Accuracy, Precision, Recall, F1, ROC-AUC

### Clustering Problems

**Characteristics:**
- No labels available
- Find groups in data
- Examples: Customer segments, document topics

**Key Indicators:**
- "Group similar..."
- "Find patterns in..."
- "Segment..."

**Metrics:**
- Silhouette score, Inertia, Davies-Bouldin index

---

## Quick Terminology Lookup

### Data Terms

| Term | Definition |
|------|------------|
| **Feature** | Input variable (X) |
| **Label/Target** | Output variable (y) |
| **Training Data** | Data used to train model |
| **Validation Data** | Data used to tune hyperparameters |
| **Test Data** | Data used for final evaluation |
| **Overfitting** | Model memorizes training data |
| **Underfitting** | Model too simple, can't learn patterns |

### Model Terms

| Term | Definition |
|------|------------|
| **Algorithm** | Method to learn from data |
| **Model** | Learned function that makes predictions |
| **Hyperparameter** | Configuration set before training |
| **Parameter** | Values learned during training |
| **Epoch** | One pass through training data |
| **Batch** | Subset of data processed together |

### Evaluation Terms

| Term | Definition |
|------|------------|
| **Accuracy** | Percentage of correct predictions |
| **Precision** | Of positive predictions, how many correct |
| **Recall** | Of actual positives, how many found |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Area under ROC curve |
| **Confusion Matrix** | Breakdown of predictions vs actual |

---

## Common Patterns and Anti-patterns

### Good Patterns

```python
# Pattern 1: Proper train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pattern 2: Scale features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Don't fit on test!

# Pattern 3: Cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Mean CV score: {scores.mean():.3f}")

# Pattern 4: Evaluate on test set only once
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
```

### Anti-patterns

```python
# Anti-pattern 1: Data leakage
# DON'T: Use future information
df['target'] = df['feature'].shift(-1)  # Using future data!

# Anti-pattern 2: Fitting on test data
# DON'T: Fit scaler on test data
scaler.fit(X_test)  # Wrong!

# Anti-pattern 3: Overfitting
# DON'T: Use too complex model without validation
model = ComplexModel(max_depth=1000)  # Will overfit!

# Anti-pattern 4: Testing multiple times
# DON'T: Evaluate on test set multiple times
for i in range(10):
    model = train_model()
    score = model.score(X_test, y_test)  # Wrong! Test set used multiple times
```

---

## Quick Algorithm Reference

### Regression Algorithms

| Algorithm | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **Linear Regression** | Linear relationships | Simple, interpretable | Assumes linearity |
| **Polynomial Regression** | Non-linear relationships | Handles curves | Can overfit |
| **Random Forest** | General purpose | Good performance | Less interpretable |
| **XGBoost** | Best performance needed | Excellent performance | Complex, slow |

### Classification Algorithms

| Algorithm | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **Logistic Regression** | Baseline, interpretable | Simple, fast | Assumes linearity |
| **Decision Tree** | Need interpretability | Very interpretable | Prone to overfitting |
| **Random Forest** | General purpose | Good performance | Less interpretable |
| **SVM** | Small-medium datasets | Effective | Slow on large data |
| **KNN** | Local patterns | Simple | Slow prediction |

### Clustering Algorithms

| Algorithm | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **K-Means** | Spherical clusters | Fast, simple | Need to specify k |
| **Hierarchical** | Unknown number of clusters | Visual dendrogram | Slow on large data |
| **DBSCAN** | Non-spherical clusters | Finds outliers | Sensitive to parameters |

---

## Common Mistakes to Avoid

1. **Data Leakage**: Using future information to predict past
2. **Overfitting**: Model too complex, memorizes training data
3. **Underfitting**: Model too simple, can't learn patterns
4. **Wrong Metrics**: Using accuracy for imbalanced data
5. **No Validation**: Not using validation set
6. **Test Set Contamination**: Using test set multiple times
7. **Ignoring Baseline**: Not comparing with simple baseline
8. **No Documentation**: Not documenting experiments

---

## Quick Tips

1. **Start Simple**: Begin with simple models (linear/logistic regression)
2. **Use Baseline**: Always compare with a simple baseline
3. **Validate Properly**: Use cross-validation or hold-out validation
4. **Check for Overfitting**: Compare train vs validation performance
5. **Use Appropriate Metrics**: Choose metrics based on problem type
6. **Document Everything**: Keep track of experiments and results
7. **Iterate**: ML is iterative, improve based on results

---

**Remember**: This is a quick reference. For detailed explanations, see the main guides!

