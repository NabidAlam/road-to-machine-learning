"""Module 02 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def make_toy(n=100, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def exercise_1_split(X, y, test_size=0.2, random_state=42):
    """Return X_train, X_test, y_train, y_test with a fixed random_state."""
    raise NotImplementedError


def exercise_2_fit_score(X_train, y_train, X_test, y_test):
    """Fit LogisticRegression and return test accuracy as float."""
    raise NotImplementedError


def exercise_3_feature_count(X):
    """Return number of columns (features)."""
    raise NotImplementedError


def exercise_4_class_balance(y):
    """Return minority class fraction in [0, 0.5]."""
    raise NotImplementedError


def exercise_5_no_leak_shapes(X_train, X_test):
    """Return True if train and test row counts sum to original n and both nonempty."""
    raise NotImplementedError


def exercise_6_predict_labels(model, X_test):
    """Return integer predictions for X_test."""
    raise NotImplementedError


def run_checks(ns):
    X, y = make_toy()
    X_train, X_test, y_train, y_test = ns["exercise_1_split"](X, y, test_size=0.2, random_state=42)
    assert X_train.shape[0] == 80 and X_test.shape[0] == 20
    assert y_train.shape[0] == 80 and y_test.shape[0] == 20

    acc = ns["exercise_2_fit_score"](X_train, y_train, X_test, y_test)
    assert isinstance(acc, float)
    assert 0.7 <= acc <= 1.0

    assert ns["exercise_3_feature_count"](X) == 3

    bal = ns["exercise_4_class_balance"](y)
    assert 0.0 <= bal <= 0.5

    assert ns["exercise_5_no_leak_shapes"](X_train, X_test) is True

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, y_train)
    pred = ns["exercise_6_predict_labels"](clf, X_test)
    assert pred.shape == (20,)
    assert set(np.unique(pred)).issubset({0, 1})
    print("module02 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
