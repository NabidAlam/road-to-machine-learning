"""Module 10 exercise checks. Prefer numpy/sklearn. Torch is optional and unused here."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def make_toy(seed=5):
    X, y = make_classification(
        n_samples=240,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=seed,
    )
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def exercise_1_mlp_score(X_train, y_train, X_test, y_test):
    """Fit a small MLPClassifier. Return test accuracy."""
    raise NotImplementedError


def exercise_2_n_layers_param():
    """Return the hidden_layer_sizes tuple used in exercise_1 (document your choice)."""
    raise NotImplementedError


def exercise_3_torch_optional_note():
    """
    Return a short string stating that torch/tensorflow are optional for this pack.
    Must contain the substring 'optional'.
    """
    raise NotImplementedError


def exercise_4_predict_proba_sum(model, X_row):
    """Return sum of predict_proba for one row. Should be about 1.0."""
    raise NotImplementedError


def exercise_5_class_count(y):
    """Return number of unique classes."""
    raise NotImplementedError


def run_checks(ns):
    X_train, X_test, y_train, y_test = make_toy()
    acc = ns["exercise_1_mlp_score"](X_train, y_train, X_test, y_test)
    assert 0.7 <= acc <= 1.0
    h = ns["exercise_2_n_layers_param"]()
    assert isinstance(h, tuple) and len(h) >= 1
    note = ns["exercise_3_torch_optional_note"]()
    assert "optional" in note.lower()
    mlp = MLPClassifier(
        hidden_layer_sizes=h,
        max_iter=300,
        solver="lbfgs",
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    s = ns["exercise_4_predict_proba_sum"](mlp, X_test[:1])
    assert abs(s - 1.0) < 1e-6
    assert ns["exercise_5_class_count"](y_train) == 2
    print("module10 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
