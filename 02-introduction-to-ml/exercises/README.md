# Module 02 Exercise Pack

ML workflow. Train and test split. Simple fit and score.

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

1. Split the toy matrix into train and test (80/20, fixed seed).
2. Fit logistic regression and report test accuracy.
3. Count features (columns) on X.
4. Compute the minority class fraction.
5. Confirm train and test are both nonempty after the split.
6. Predict integer labels on the test set.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod02-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/02-introduction-to-ml/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod02-exercises-smoke OK")
```
