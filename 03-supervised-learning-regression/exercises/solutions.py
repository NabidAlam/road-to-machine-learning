"""Module 03 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tasks import run_checks


def exercise_1_fit_predict(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.predict(X)


def exercise_2_mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))


def exercise_3_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def exercise_4_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def exercise_5_r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def exercise_6_residual_mean(y_true, y_pred):
    return float(np.mean(y_true - y_pred))


if __name__ == "__main__":
    run_checks(globals())
