"""Module 04 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def make_preds():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    return y_true, y_pred


def exercise_1_accuracy(y_true, y_pred):
    raise NotImplementedError


def exercise_2_precision(y_true, y_pred):
    """Precision for the positive class (1)."""
    raise NotImplementedError


def exercise_3_recall(y_true, y_pred):
    """Recall for the positive class (1)."""
    raise NotImplementedError


def exercise_4_f1(y_true, y_pred):
    """F1 for the positive class (1)."""
    raise NotImplementedError


def exercise_5_confusion(y_true, y_pred):
    """Return 2x2 confusion matrix (sklearn default labels)."""
    raise NotImplementedError


def exercise_6_false_positives(y_true, y_pred):
    """Return count of false positives for class 1."""
    raise NotImplementedError


def run_checks(ns):
    y_true, y_pred = make_preds()
    acc = ns["exercise_1_accuracy"](y_true, y_pred)
    prec = ns["exercise_2_precision"](y_true, y_pred)
    rec = ns["exercise_3_recall"](y_true, y_pred)
    f1 = ns["exercise_4_f1"](y_true, y_pred)
    cm = ns["exercise_5_confusion"](y_true, y_pred)
    fp = ns["exercise_6_false_positives"](y_true, y_pred)

    assert abs(acc - accuracy_score(y_true, y_pred)) < 1e-9
    assert abs(prec - precision_score(y_true, y_pred)) < 1e-9
    assert abs(rec - recall_score(y_true, y_pred)) < 1e-9
    assert abs(f1 - f1_score(y_true, y_pred)) < 1e-9
    assert np.array_equal(cm, confusion_matrix(y_true, y_pred))
    assert int(fp) == int(cm[0, 1])
    print("module04 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
