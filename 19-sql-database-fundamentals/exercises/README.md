# Module 19 Exercise Pack

In-memory SQLite queries with pandas. No server required.

## How to run

From this `exercises/` folder:

```bash
python solutions.py
```

That loads your implementations (in `solutions.py`) and runs the asserts in `tasks.py`.

Work pattern:

1. Read each `exercise_N_...` docstring in `tasks.py`.
2. Implement the same function names in `solutions.py` (starter solutions are already filled).
3. Re-run `python solutions.py` until you see the module OK line.

Deps: `numpy`, `pandas`, and `scikit-learn` where noted. No dataset downloads.

## Tasks

1. Count rows in the sales table.
2. Sum EU amounts.
3. Average qty across all rows.
4. Group total amount by region.
5. Filter rows with amount >= 12.
6. Join a small region dimension to attach manager names.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod19-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/19-sql-database-fundamentals/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod19-exercises-smoke OK")
```
