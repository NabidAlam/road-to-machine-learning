# Module 06 Exercise Pack

Bagging vs boosting concept checks on tiny data.

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

1. Score bagged decision trees on the toy split.
2. Score a small random forest.
3. Score AdaBoost.
4. Score gradient boosting.
5. Confirm the bagging-vs-variance idea (return True with the stated meaning).
6. Confirm the boosting-vs-bias idea (return True with the stated meaning).

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod06-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/06-ensemble-methods/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod06-exercises-smoke OK")
```
