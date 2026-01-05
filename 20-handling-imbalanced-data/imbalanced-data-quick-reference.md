# Imbalanced Data Quick Reference

Quick reference for handling imbalanced data.

## Resampling

```python
# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

# Class weights
model = RandomForestClassifier(class_weight='balanced')
```

## Metrics

```python
# Use F1-score, PR-AUC for imbalanced data
from sklearn.metrics import f1_score, average_precision_score
```

---

**Remember**: Don't use accuracy for imbalanced data!

