"""Module 19 exercise checks. In-memory SQLite + pandas. No downloads."""
from __future__ import annotations

import sqlite3

import pandas as pd


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL, qty INTEGER)"
    )
    rows = [
        (1, "EU", 10.0, 2),
        (2, "EU", 15.0, 1),
        (3, "US", 8.0, 4),
        (4, "US", 20.0, 2),
        (5, "APAC", 12.0, 3),
    ]
    conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return conn


def exercise_1_count_rows(conn):
    """Return total row count in sales."""
    raise NotImplementedError


def exercise_2_eu_total(conn):
    """Return SUM(amount) for region = 'EU'."""
    raise NotImplementedError


def exercise_3_avg_qty(conn):
    """Return AVG(qty) across all rows as float."""
    raise NotImplementedError


def exercise_4_region_totals(conn):
    """
    Return a DataFrame with columns region, total_amount sorted by region ascending.
    """
    raise NotImplementedError


def exercise_5_filter_amount(conn, min_amount=12.0):
    """Return DataFrame of rows with amount >= min_amount, sorted by id."""
    raise NotImplementedError


def exercise_6_join_dim(conn):
    """
    Create dim_region(region, manager) with EU->Ada, US->Bob, APAC->Chi.
    Return a DataFrame of sales joined to manager, columns id, region, manager, sorted by id.
    """
    raise NotImplementedError


def run_checks(ns):
    conn = make_conn()
    assert ns["exercise_1_count_rows"](conn) == 5
    assert abs(ns["exercise_2_eu_total"](conn) - 25.0) < 1e-9
    assert abs(ns["exercise_3_avg_qty"](conn) - 2.4) < 1e-9
    totals = ns["exercise_4_region_totals"](conn)
    assert list(totals["region"]) == ["APAC", "EU", "US"]
    assert abs(float(totals.loc[totals["region"] == "US", "total_amount"].iloc[0]) - 28.0) < 1e-9
    filt = ns["exercise_5_filter_amount"](conn, 12.0)
    assert list(filt["id"]) == [2, 4, 5]
    joined = ns["exercise_6_join_dim"](conn)
    assert list(joined.columns) == ["id", "region", "manager"]
    assert joined.loc[joined["id"] == 1, "manager"].iloc[0] == "Ada"
    conn.close()
    print("module19 exercises OK")


if __name__ == "__main__":
    print("Implement solutions in solutions.py then run that file.")
