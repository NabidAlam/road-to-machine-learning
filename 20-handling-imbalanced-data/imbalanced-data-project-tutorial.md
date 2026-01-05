# Imbalanced Data Project Tutorial

Step-by-step tutorial for handling imbalanced data.

## Project: Fraud Detection

### Step 1: Analyze Imbalance

```python
from collections import Counter

counter = Counter(y)
print(counter)
```

### Step 2: Apply SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Step 3: Train and Evaluate

```python
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_resampled, y_resampled)
```

---

**Congratulations!** You've handled imbalanced data!

