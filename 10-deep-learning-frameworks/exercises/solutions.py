"""Module 10 solutions. Numpy/sklearn only. Torch not required."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.neural_network import MLPClassifier

from tasks import run_checks

HIDDEN = (16,)


def exercise_1_mlp_score(X_train, y_train, X_test, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        max_iter=300,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return float(model.score(X_test, y_test))


def exercise_2_n_layers_param():
    return HIDDEN


def exercise_3_torch_optional_note():
    return "PyTorch and TensorFlow are optional extras. This pack stays on sklearn MLP."


def exercise_4_predict_proba_sum(model, X_row):
    return float(np.sum(model.predict_proba(X_row)))


def exercise_5_class_count(y):
    return int(len(np.unique(y)))


if __name__ == "__main__":
    run_checks(globals())
