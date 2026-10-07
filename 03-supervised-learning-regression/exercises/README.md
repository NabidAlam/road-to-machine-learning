# Module 03 Exercise Pack

Regression metrics on tiny synthetic targets.

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

1. Fit linear regression and predict on the same toy X.
2. Compute MSE.
3. Compute RMSE.
4. Compute MAE.
5. Compute R^2.
6. Compute mean residual and expect it near zero.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod03-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/03-supervised-learning-regression/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod03-exercises-smoke OK")
```
