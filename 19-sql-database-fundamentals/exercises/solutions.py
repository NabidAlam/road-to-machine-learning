"""Module 19 solutions. Run as main to verify against tasks.run_checks."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from tasks import run_checks


def exercise_1_count_rows(conn):
    return int(conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0])


def exercise_2_eu_total(conn):
    return float(
        conn.execute("SELECT SUM(amount) FROM sales WHERE region = 'EU'").fetchone()[0]
    )


def exercise_3_avg_qty(conn):
    return float(conn.execute("SELECT AVG(qty) FROM sales").fetchone()[0])


def exercise_4_region_totals(conn):
    return pd.read_sql_query(
        """
        SELECT region, SUM(amount) AS total_amount
        FROM sales
        GROUP BY region
        ORDER BY region ASC
        """,
        conn,
    )


def exercise_5_filter_amount(conn, min_amount=12.0):
    return pd.read_sql_query(
        "SELECT * FROM sales WHERE amount >= ? ORDER BY id ASC",
        conn,
        params=(min_amount,),
    )


def exercise_6_join_dim(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS dim_region (region TEXT PRIMARY KEY, manager TEXT)")
    conn.executemany(
        "INSERT OR REPLACE INTO dim_region VALUES (?, ?)",
        [("EU", "Ada"), ("US", "Bob"), ("APAC", "Chi")],
    )
    conn.commit()
    return pd.read_sql_query(
        """
        SELECT s.id, s.region, d.manager
        FROM sales s
        JOIN dim_region d ON s.region = d.region
        ORDER BY s.id ASC
        """,
        conn,
    )


if __name__ == "__main__":
    run_checks(globals())
