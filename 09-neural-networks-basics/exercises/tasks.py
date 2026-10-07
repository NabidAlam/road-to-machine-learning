"""Module 09 exercise checks. Pure numpy forward pass. No downloads."""
from __future__ import annotations

import numpy as np


def relu(z):
    raise NotImplementedError


def sigmoid(z):
    raise NotImplementedError


def exercise_1_relu(z):
    """Elementwise ReLU."""
    raise NotImplementedError


def exercise_2_sigmoid(z):
    """Elementwise sigmoid."""
    raise NotImplementedError


def exercise_3_linear(X, W, b):
    """Return X @ W + b."""
    raise NotImplementedError


def exercise_4_forward(X, W1, b1, W2, b2):
    """
    Two-layer net: h = relu(X @ W1 + b1), yhat = sigmoid(h @ W2 + b2).
    Return yhat with shape (n, 1).
    """
    raise NotImplementedError


def exercise_5_binary_preds(yhat, threshold=0.5):
    """Threshold probabilities to 0/1 ints."""
    raise NotImplementedError


def exercise_6_accuracy(y_true, y_pred):
    """Mean equality as float."""
    raise NotImplementedError


def run_checks(ns):
    z = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(ns["exercise_1_relu"](z), np.array([0.0, 0.0, 2.0]))
    s = ns["exercise_2_sigmoid"](np.array([0.0]))
    assert abs(float(s[0]) - 0.5) < 1e-9

    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    W1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.zeros(2)
    W2 = np.array([[1.0], [-1.0]])
    b2 = np.array([0.0])
    yhat = ns["exercise_4_forward"](X, W1, b1, W2, b2)
    assert yhat.shape == (3, 1)
    lin = ns["exercise_3_linear"](X, W1, b1)
    assert lin.shape == (3, 2)
    pred = ns["exercise_5_binary_preds"](yhat)
    assert set(np.unique(pred)).issubset({0, 1})
    y_true = np.array([[1], [0], [1]])
    acc = ns["exercise_6_accuracy"](y_true, pred)
    assert 0.0 <= acc <= 1.0
    print("module09 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
