"""Module 09 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from tasks import run_checks


def exercise_1_relu(z):
    z = np.asarray(z, dtype=float)
    return np.maximum(0.0, z)


def exercise_2_sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def exercise_3_linear(X, W, b):
    return X @ W + b


def exercise_4_forward(X, W1, b1, W2, b2):
    h = exercise_1_relu(exercise_3_linear(X, W1, b1))
    return exercise_2_sigmoid(exercise_3_linear(h, W2, b2))


def exercise_5_binary_preds(yhat, threshold=0.5):
    return (np.asarray(yhat) >= threshold).astype(int)


def exercise_6_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


if __name__ == "__main__":
    run_checks(globals())
