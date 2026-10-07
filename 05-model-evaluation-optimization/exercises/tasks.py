"""Module 05 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_toy(n=120, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] - X[:, 1] + 0.2 * X[:, 2] > 0).astype(int)
    return X, y


def exercise_1_cv_mean(X, y, cv=5):
    """Return mean stratified CV accuracy for LogisticRegression (max_iter=500)."""
    raise NotImplementedError


def exercise_2_cv_std(X, y, cv=5):
    """Return std of stratified CV accuracy."""
    raise NotImplementedError


def exercise_3_pipeline_cv(X, y, cv=5):
    """Scale inside a Pipeline. Return mean CV accuracy. Scaling must not leak."""
    raise NotImplementedError


def exercise_4_leaky_scale_is_wrong(X, y):
    """
    Demonstrate the bad pattern: scale on full X then CV.
    Return True if that leaky mean score is computed (always True when implemented).
    """
    raise NotImplementedError


def exercise_5_fold_count(X, y, n_splits=5):
    """Return number of folds from StratifiedKFold.split."""
    raise NotImplementedError


def exercise_6_train_fraction(n_splits=5):
    """For equal folds, train rows per fold are about (n_splits-1)/n_splits of n. Return that fraction."""
    raise NotImplementedError


def exercise_7_pipeline_steps():
    """Return ordered step names for StandardScaler then LogisticRegression pipeline."""
    raise NotImplementedError


def run_checks(ns):
    X, y = make_toy()
    mean_acc = ns["exercise_1_cv_mean"](X, y, cv=5)
    std_acc = ns["exercise_2_cv_std"](X, y, cv=5)
    pipe_acc = ns["exercise_3_pipeline_cv"](X, y, cv=5)
    assert 0.75 <= mean_acc <= 1.0
    assert std_acc >= 0.0
    assert 0.75 <= pipe_acc <= 1.0
    assert ns["exercise_4_leaky_scale_is_wrong"](X, y) is True
    assert ns["exercise_5_fold_count"](X, y, n_splits=5) == 5
    assert abs(ns["exercise_6_train_fraction"](5) - 0.8) < 1e-9
    assert ns["exercise_7_pipeline_steps"]() == ["scaler", "clf"]
    print("module05 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
