# ML Debugging Lab (Silent Failure Modes)

Bugs that return a number can still be wrong. This lab lists silent failures and a first check for each.

## 1. Train/serve skew

**Symptom.** Offline metrics look fine. Online metrics fall over.

**First check.** Log a hash or summary of features at train time and at serve time. Compare distributions for the same traffic window.

## 2. Target leakage

**Symptom.** Unrealistically strong validation score. Collapse after launch.

**First check.** List every feature that is only knowable after the label time. Drop anything that uses post-outcome information. Re-run CV.

## 3. Bad split for the problem

**Symptom.** Random split scores high. Time-based or group-based split tanks.

**First check.** If rows share a user, device, or day, split on that key. For time series, never shuffle across time.

## 4. Scaling outside the fold

**Symptom.** CV looks a bit too good. Pipeline in production is worse.

**First check.** Put scalers and encoders inside a `Pipeline` or fold loop. Fit transforms on train folds only.

## 5. Imbalance vanity accuracy

**Symptom.** 99% accuracy. Minority recall near zero.

**First check.** Print a confusion matrix and PR-AUC. Optimize the metric that matches the cost of a miss.

## 6. Shuffled labels still "work"

**Symptom.** You distrust a pipeline but metrics still look okay.

**First check.** Permute `y` and retrain. If score barely moves, the model is not using the label signal you think it is. Or your metric is constant.

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def sanity_shuffle_score(model, X, y, cv=5, seed=0):
    real = cross_val_score(model, X, y, cv=cv).mean()
    y_bad = np.random.default_rng(seed).permutation(y)
    fake = cross_val_score(model, X, y_bad, cv=cv).mean()
    return {"real": float(real), "shuffled_y": float(fake)}
```

## 7. Duplicate rows across splits

**Symptom.** Near-perfect test score on scraped or joined tables.

**First check.** Deduplicate on a business key before splitting. Confirm no identical feature rows sit in both train and test.

## 8. Silent NaN / inf paths

**Symptom.** Model trains. A subset of production rows always predicts the same class.

**First check.** Count NaNs per column before and after transforms. Assert finite values on a production sample batch.

## 9. Threshold copied from the wrong fold

**Symptom.** Precision/recall in the report do not match the deployed cut.

**First check.** Choose thresholds on validation only. Freeze them. Score test once.

## 10. Metric computed on probabilities as if they were labels

**Symptom.** Absurd F1 or accuracy values.

**First check.** Confirm `predict` vs `predict_proba`. Metrics that need labels must see 0/1 (or class ids), not raw scores, unless the metric explicitly takes scores.

## Lab loop

1. State the failure mode you fear.
2. Write one assert or print that would catch it.
3. Only then change the model.

## Related

- [ml_model_testing.md](ml_model_testing.md)
- [common_errors.md](common_errors.md)
- Module 05 pack. [../05-model-evaluation-optimization/exercises/README.md](../05-model-evaluation-optimization/exercises/README.md)
