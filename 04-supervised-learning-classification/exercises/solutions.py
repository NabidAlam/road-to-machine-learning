"""Module 04 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from tasks import run_checks


def exercise_1_accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def exercise_2_precision(y_true, y_pred):
    return float(precision_score(y_true, y_pred))


def exercise_3_recall(y_true, y_pred):
    return float(recall_score(y_true, y_pred))


def exercise_4_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred))


def exercise_5_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def exercise_6_false_positives(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return int(cm[0, 1])


if __name__ == "__main__":
    run_checks(globals())
