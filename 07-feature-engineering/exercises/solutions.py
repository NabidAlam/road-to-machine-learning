"""Module 07 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tasks import run_checks


def exercise_1_one_hot_shape(df):
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    out = enc.fit_transform(df[["color"]])
    return out.shape


def exercise_2_one_hot_cols(df):
    return int(exercise_1_one_hot_shape(df)[1])


def exercise_3_ordinal_size(df):
    enc = OrdinalEncoder(categories=[["S", "M", "L"]])
    return enc.fit_transform(df[["size"]]).ravel().astype(int)


def exercise_4_log1p_price(df):
    return np.log1p(df["price"].to_numpy(dtype=float))


def exercise_5_price_zscore(df):
    p = df["price"].to_numpy(dtype=float)
    return (p - p.mean()) / p.std(ddof=0)


def exercise_6_combine_features(df):
    oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    color = oh.fit_transform(df[["color"]])
    size = exercise_3_ordinal_size(df).reshape(-1, 1).astype(float)
    price = exercise_5_price_zscore(df).reshape(-1, 1)
    return np.hstack([color, size, price])


if __name__ == "__main__":
    run_checks(globals())
