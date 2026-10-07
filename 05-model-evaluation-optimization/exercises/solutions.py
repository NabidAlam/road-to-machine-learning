"""Module 05 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tasks import run_checks


def exercise_1_cv_mean(X, y, cv=5):
    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=42))
    return float(scores.mean())


def exercise_2_cv_std(X, y, cv=5):
    clf = LogisticRegression(max_iter=500, random_state=42)
    scores = cross_val_score(clf, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=42))
    return float(scores.std())


def exercise_3_pipeline_cv(X, y, cv=5):
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=42))
    return float(scores.mean())


def exercise_4_leaky_scale_is_wrong(X, y):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=500, random_state=42)
    _ = cross_val_score(clf, Xs, y, cv=5).mean()
    return True


def exercise_5_fold_count(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return sum(1 for _ in skf.split(X, y))


def exercise_6_train_fraction(n_splits=5):
    return float((n_splits - 1) / n_splits)


def exercise_7_pipeline_steps():
    return ["scaler", "clf"]


if __name__ == "__main__":
    run_checks(globals())
