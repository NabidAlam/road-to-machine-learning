"""Module 02 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from tasks import run_checks


def exercise_1_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def exercise_2_fit_score(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def exercise_3_feature_count(X):
    return int(X.shape[1])


def exercise_4_class_balance(y):
    counts = np.bincount(y.astype(int))
    return float(counts.min() / counts.sum())


def exercise_5_no_leak_shapes(X_train, X_test):
    return bool(X_train.shape[0] > 0 and X_test.shape[0] > 0)


def exercise_6_predict_labels(model, X_test):
    return model.predict(X_test).astype(int)


if __name__ == "__main__":
    run_checks(globals())
