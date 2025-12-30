# Common Pitfalls and Best Practices in Machine Learning

Essential guide to avoiding common mistakes and following best practices when building ML models.

## Table of Contents

- [Data-Related Pitfalls](#data-related-pitfalls)
- [Model Training Pitfalls](#model-training-pitfalls)
- [Evaluation Pitfalls](#evaluation-pitfalls)
- [Deployment Pitfalls](#deployment-pitfalls)
- [Best Practices](#best-practices)
- [Checklist](#checklist)

---

## Data-Related Pitfalls

### Pitfall 1: Data Leakage

**What it is**: Using information from test set or future data during training.

**Examples:**
```python
# ❌ WRONG: Scaling before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses all data including test!
X_train, X_test = train_test_split(X_scaled, y)

# ✅ CORRECT: Split first, then scale
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)  # Transform test
```

**Common Leakage Sources:**
- Scaling/normalizing before splitting
- Feature selection using test set
- Using future information (time series)
- Target encoding with test data

**How to Avoid:**
- Always split data first
- Fit transformers only on training data
- Use cross-validation for feature selection
- Be careful with time-based data

### Pitfall 2: Insufficient Data

**Problem**: Not enough data for model to learn patterns.

**Signs:**
- High variance in performance
- Model can't generalize
- Overfitting even with simple models

**Solutions:**
- Collect more data
- Use data augmentation
- Use simpler models
- Transfer learning (for deep learning)

### Pitfall 3: Imbalanced Data

**Problem**: Classes are not equally represented.

**Example:**
```python
# 99% class A, 1% class B
# Model predicts A all the time → 99% accuracy but useless!
```

**Solutions:**
- Use appropriate metrics (Precision, Recall, F1)
- Resample data (SMOTE, undersampling)
- Use class weights
- Collect more minority class data

### Pitfall 4: Poor Data Quality

**Problems:**
- Missing values not handled properly
- Outliers not addressed
- Inconsistent formats
- Wrong data types

**Solutions:**
- Always check data quality first
- Handle missing values appropriately
- Detect and handle outliers
- Validate data types

---

## Model Training Pitfalls

### Pitfall 5: Overfitting

**What it is**: Model memorizes training data, fails on new data.

**Signs:**
```python
train_accuracy = 0.99  # Perfect on training
test_accuracy = 0.65  # Poor on test
# Large gap = overfitting!
```

**Solutions:**
- More training data
- Simpler model (reduce complexity)
- Regularization (L1, L2, dropout)
- Early stopping
- Cross-validation

### Pitfall 6: Underfitting

**What it is**: Model too simple to capture patterns.

**Signs:**
```python
train_accuracy = 0.55  # Poor on training
test_accuracy = 0.53   # Poor on test
# Both low = underfitting
```

**Solutions:**
- More complex model
- Better features
- Remove regularization
- Train longer
- Increase model capacity

### Pitfall 7: Wrong Algorithm Choice

**Problem**: Using algorithm that doesn't fit the problem.

**Examples:**
- Using linear model for non-linear problem
- Using classification for regression
- Using complex model when simple works

**Solution**: Follow algorithm selection guide, start simple.

### Pitfall 8: Ignoring Hyperparameters

**Problem**: Using default hyperparameters without tuning.

**Impact**: Suboptimal performance

**Solution**: Use validation set or cross-validation to tune hyperparameters.

```python
# ❌ WRONG: Use defaults without checking
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ✅ BETTER: Tune hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## Evaluation Pitfalls

### Pitfall 9: Wrong Evaluation Metric

**Problem**: Using metric that doesn't match the problem.

**Examples:**
- Using accuracy for imbalanced data
- Using MSE when outliers matter
- Not considering business impact

**Solutions:**
- Choose metric based on problem
- Use multiple metrics
- Consider business context

### Pitfall 10: No Validation Set

**Problem**: Only using train/test split, no validation for tuning.

**Impact**: Overfitting to test set, unrealistic performance estimates

**Solution**: Use train/validation/test split or cross-validation.

```python
# ✅ CORRECT: Three-way split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Use validation for tuning
# Use test only for final evaluation
```

### Pitfall 11: Testing on Training Data

**Problem**: Evaluating model on data it was trained on.

**Why it's wrong**: Gives overly optimistic results, doesn't reflect real performance.

**Solution**: Always use separate test set that model has never seen.

### Pitfall 12: Not Checking for Data Drift

**Problem**: Model performance degrades over time as data changes.

**Solution**: Monitor model performance, retrain periodically.

---

## Deployment Pitfalls

### Pitfall 13: Not Versioning Models

**Problem**: Can't reproduce results or rollback if needed.

**Solution**: Version control for models, data, and code.

```python
# Use MLflow or similar
import mlflow
mlflow.log_model(model, "iris_classifier")
mlflow.log_params(model.get_params())
```

### Pitfall 14: Ignoring Model Interpretability

**Problem**: Can't explain predictions when needed.

**Solution**: Use interpretable models or explainability tools (SHAP, LIME).

### Pitfall 15: No Monitoring

**Problem**: Don't know when model performance degrades.

**Solution**: Set up monitoring for:
- Prediction accuracy
- Prediction latency
- Data quality
- Model drift

---

## Best Practices

### 1. Start Simple

**Principle**: Begin with the simplest solution that works.

**Why:**
- Easier to understand and debug
- Faster to implement
- Often performs well
- Can always upgrade later

**Example:**
```
1. Try Linear Regression first
2. If not good enough, try Random Forest
3. If still not enough, try XGBoost
4. Last resort: Neural Networks
```

### 2. Follow the Workflow

**Structured Approach:**
1. Problem definition
2. Data collection and exploration
3. Data preparation
4. Model selection and training
5. Evaluation
6. Deployment
7. Monitoring

**Don't skip steps!**

### 3. Validate Everything

**Check:**
- Data quality
- Model assumptions
- Performance metrics
- Predictions make sense

### 4. Use Cross-Validation

**Why:**
- Better performance estimate
- Use all data effectively
- Reduce overfitting risk

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### 5. Document Everything

**Document:**
- Data sources and collection methods
- Preprocessing steps
- Model choices and rationale
- Hyperparameters and tuning process
- Results and interpretations

### 6. Version Control

**Track:**
- Code versions
- Data versions
- Model versions
- Experiment results

### 7. Test on Unseen Data

**Always:**
- Keep test set completely separate
- Only use test set for final evaluation
- Never tune on test set

### 8. Consider Business Context

**Ask:**
- What does success look like?
- What are the costs of errors?
- What metrics matter to stakeholders?
- What are the constraints?

### 9. Iterate and Improve

**Process:**
1. Build baseline model
2. Evaluate and identify issues
3. Improve (more data, better features, different model)
4. Repeat

### 10. Understand Your Model

**Know:**
- How it works
- What it's good/bad at
- When it will fail
- How to interpret predictions

---

## Checklist

### Before Training

- [ ] Data is clean and validated
- [ ] Data is split properly (train/val/test)
- [ ] No data leakage
- [ ] Appropriate algorithm selected
- [ ] Baseline model established

### During Training

- [ ] Monitoring training progress
- [ ] Checking for overfitting
- [ ] Validating on validation set
- [ ] Tuning hyperparameters
- [ ] Documenting experiments

### After Training

- [ ] Evaluated on test set (once!)
- [ ] Multiple metrics calculated
- [ ] Results make sense
- [ ] Model is interpretable (if needed)
- [ ] Performance is acceptable

### Before Deployment

- [ ] Model is versioned
- [ ] Code is documented
- [ ] Tests are written
- [ ] Monitoring is set up
- [ ] Rollback plan exists

---

## Resources

- [Google's ML Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Scikit-learn Best Practices](https://scikit-learn.org/stable/developers/contributing.html)

---

## Key Takeaways

1. **Avoid Data Leakage**: Always split before preprocessing
2. **Start Simple**: Don't overcomplicate
3. **Validate Properly**: Use appropriate metrics and validation
4. **Document**: Keep track of everything
5. **Iterate**: ML is iterative, not one-shot
6. **Understand**: Know your model and data

---

**Remember**: Learning from mistakes is part of the journey. Follow best practices to avoid common pitfalls!

