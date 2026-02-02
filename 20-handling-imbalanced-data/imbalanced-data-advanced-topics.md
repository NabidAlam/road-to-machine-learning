# Advanced Imbalanced Data Topics

Advanced techniques for handling imbalanced data.

## Table of Contents

- [Advanced Resampling](#advanced-resampling)
- [Ensemble Methods for Imbalanced Data](#ensemble-methods-for-imbalanced-data)
- [Cost-Sensitive Learning](#cost-sensitive-learning)
- [Common Pitfalls](#common-pitfalls)

---

## Advanced Resampling

### SMOTE Variants

```python
from imblearn.over_sampling import BorderlineSMOTE, ADASYN, SVMSMOTE, SMOTENC

# Borderline SMOTE
borderline = BorderlineSMOTE(random_state=42)
X_res, y_res = borderline.fit_resample(X_train, y_train)

# ADASYN (adaptive)
adasyn = ADASYN(random_state=42)
X_res, y_res = adasyn.fit_resample(X_train, y_train)
```

---

## Ensemble Methods for Imbalanced Data

### Balanced Random Forest

```python
from imblearn.ensemble import BalancedRandomForestClassifier

# Balanced Random Forest
brf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
brf.fit(X_train, y_train)
```

---

## Cost-Sensitive Learning

When classes are imbalanced, accuracy often lies. Cost-sensitive learning makes the model care more about the errors that are expensive for the business.

### 1) Class weights (simple and common)

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight="balanced", max_iter=1000)
clf.fit(X_train, y_train)
```

### 2) Custom weights (when you know costs)

Example: false negatives are 5x more expensive than false positives:

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(class_weight={0: 1.0, 1: 5.0}, random_state=42)
clf.fit(X_train, y_train)
```

### 3) Threshold tuning (almost always needed)

Even with class weights, choosing a threshold matters:

```python
import numpy as np

proba = clf.predict_proba(X_val)[:, 1]
threshold = 0.2  # pick using PR curve / business target
y_pred = (proba >= threshold).astype(int)
```

Practical goal: pick a threshold that hits a target like “Recall >= 0.85 at Precision >= 0.30”.

---

## Common Pitfalls

- **Evaluating with accuracy** instead of PR-AUC / recall/precision
- **Data leakage**: resampling before train/test split (always resample inside training only)
- **Wrong split**: using random split for time-ordered problems
- **No threshold tuning**: default 0.5 is rarely optimal for imbalanced tasks
- **Ignoring calibration**: predicted probabilities may be miscalibrated
- **Not reporting costs**: a “better metric” may still be worse for business outcomes

---

## Key Takeaways

1. **Advanced Techniques**: Use ensemble methods for severe imbalance
2. **Cost-Sensitive**: Incorporate business costs
3. **Validation**: Use proper evaluation metrics

---

**Remember**: Advanced techniques help with extreme imbalance!

