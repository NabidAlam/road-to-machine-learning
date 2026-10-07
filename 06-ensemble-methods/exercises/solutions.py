"""Module 06 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

from tasks import run_checks


def exercise_1_bagging_acc(X_train, y_train, X_test, y_test):
    model = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=30,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def exercise_2_rf_acc(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def exercise_3_boost_acc(X_train, y_train, X_test, y_test):
    model = AdaBoostClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def exercise_4_gb_acc(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def exercise_5_bagging_reduces_variance():
    return True


def exercise_6_boosting_reduces_bias():
    return True


if __name__ == "__main__":
    run_checks(globals())
