"""Module 03 exercise checks. Tiny synthetic data only. No downloads."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_toy(n=80, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=n)
    return X, y


def exercise_1_fit_predict(X, y):
    """Fit LinearRegression on all rows. Return predictions shaped like y."""
    raise NotImplementedError


def exercise_2_mse(y_true, y_pred):
    """Return mean squared error."""
    raise NotImplementedError


def exercise_3_rmse(y_true, y_pred):
    """Return root mean squared error."""
    raise NotImplementedError


def exercise_4_mae(y_true, y_pred):
    """Return mean absolute error."""
    raise NotImplementedError


def exercise_5_r2(y_true, y_pred):
    """Return R^2 score."""
    raise NotImplementedError


def exercise_6_residual_mean(y_true, y_pred):
    """Return mean residual (y_true - y_pred). Should be near zero for a good fit."""
    raise NotImplementedError


def run_checks(ns):
    X, y = make_toy()
    pred = ns["exercise_1_fit_predict"](X, y)
    assert pred.shape == y.shape

    mse = ns["exercise_2_mse"](y, pred)
    rmse = ns["exercise_3_rmse"](y, pred)
    mae = ns["exercise_4_mae"](y, pred)
    r2 = ns["exercise_5_r2"](y, pred)
    resid = ns["exercise_6_residual_mean"](y, pred)

    assert abs(mse - mean_squared_error(y, pred)) < 1e-9
    assert abs(rmse - np.sqrt(mean_squared_error(y, pred))) < 1e-9
    assert abs(mae - mean_absolute_error(y, pred)) < 1e-9
    assert abs(r2 - r2_score(y, pred)) < 1e-9
    assert r2 > 0.95
    assert abs(resid) < 0.05
    print("module03 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
