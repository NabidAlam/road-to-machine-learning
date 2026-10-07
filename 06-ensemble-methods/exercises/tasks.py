"""Module 06 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def make_toy(seed=3):
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def exercise_1_bagging_acc(X_train, y_train, X_test, y_test):
    """Bagging of DecisionTreeClassifier. Return test accuracy."""
    raise NotImplementedError


def exercise_2_rf_acc(X_train, y_train, X_test, y_test):
    """RandomForest with 50 trees. Return test accuracy."""
    raise NotImplementedError


def exercise_3_boost_acc(X_train, y_train, X_test, y_test):
    """AdaBoost with 50 estimators. Return test accuracy."""
    raise NotImplementedError


def exercise_4_gb_acc(X_train, y_train, X_test, y_test):
    """GradientBoosting with 50 estimators. Return test accuracy."""
    raise NotImplementedError


def exercise_5_bagging_reduces_variance():
    """
    Return True. Bagging mainly reduces variance by averaging unstable learners.
    """
    raise NotImplementedError


def exercise_6_boosting_reduces_bias():
    """
    Return True. Boosting mainly reduces bias by fitting residuals / hard examples.
    """
    raise NotImplementedError


def run_checks(ns):
    X_train, X_test, y_train, y_test = make_toy()
    bag = ns["exercise_1_bagging_acc"](X_train, y_train, X_test, y_test)
    rf = ns["exercise_2_rf_acc"](X_train, y_train, X_test, y_test)
    ada = ns["exercise_3_boost_acc"](X_train, y_train, X_test, y_test)
    gb = ns["exercise_4_gb_acc"](X_train, y_train, X_test, y_test)
    for name, acc in [("bag", bag), ("rf", rf), ("ada", ada), ("gb", gb)]:
        assert 0.6 <= acc <= 1.0, name
    assert ns["exercise_5_bagging_reduces_variance"]() is True
    assert ns["exercise_6_boosting_reduces_bias"]() is True
    print("module06 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
