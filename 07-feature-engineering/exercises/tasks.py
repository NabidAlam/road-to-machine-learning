"""Module 07 exercise checks. Tiny synthetic pandas frames only. No downloads."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def make_frame():
    return pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue"],
            "size": ["S", "M", "L", "M", "S"],
            "price": [10.0, 12.0, 9.5, 11.0, 13.5],
        }
    )


def exercise_1_one_hot_shape(df):
    """One-hot encode color. Return encoded array shape (rows, cols)."""
    raise NotImplementedError


def exercise_2_one_hot_cols(df):
    """Return number of one-hot columns for color (dense)."""
    raise NotImplementedError


def exercise_3_ordinal_size(df):
    """Ordinal-encode size with categories S < M < L. Return 1d int array."""
    raise NotImplementedError


def exercise_4_log1p_price(df):
    """Return np.log1p of price as a 1d float array."""
    raise NotImplementedError


def exercise_5_price_zscore(df):
    """Return z-scored price (sample std, ddof=0)."""
    raise NotImplementedError


def exercise_6_combine_features(df):
    """
    Return a dense float matrix with one-hot color + ordinal size + z-scored price.
    Shape should be (5, 3 + 1 + 1) = (5, 5) when color has 3 levels.
    """
    raise NotImplementedError


def run_checks(ns):
    df = make_frame()
    shape = ns["exercise_1_one_hot_shape"](df)
    assert shape[0] == 5
    assert shape[1] == 3
    assert ns["exercise_2_one_hot_cols"](df) == 3
    ord_size = ns["exercise_3_ordinal_size"](df)
    assert list(ord_size) == [0, 1, 2, 1, 0]
    logp = ns["exercise_4_log1p_price"](df)
    assert np.allclose(logp, np.log1p(df["price"].to_numpy()))
    z = ns["exercise_5_price_zscore"](df)
    p = df["price"].to_numpy()
    assert np.allclose(z, (p - p.mean()) / p.std(ddof=0))
    combo = ns["exercise_6_combine_features"](df)
    assert combo.shape == (5, 5)
    print("module07 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
