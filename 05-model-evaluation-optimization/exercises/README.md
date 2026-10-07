# Module 05 Exercise Pack

Cross-validation and leakage-aware scaling.

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

1. Report mean stratified CV accuracy for logistic regression.
2. Report the std of those CV scores.
3. Put StandardScaler inside a Pipeline and CV again (no leakage).
4. Show the leaky pattern (scale all rows, then CV) so you can contrast it.
5. Count StratifiedKFold folds.
6. Compute the train-row fraction for equal k-fold.
7. List pipeline step names in order.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod05-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/05-model-evaluation-optimization/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod05-exercises-smoke OK")
```
