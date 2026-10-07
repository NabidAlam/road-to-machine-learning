# Module 04 Exercise Pack

Classification metrics on a tiny labeled matrix.

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

1. Compute accuracy on the fixed y_true / y_pred pair.
2. Compute precision for class 1.
3. Compute recall for class 1.
4. Compute F1 for class 1.
5. Build the 2x2 confusion matrix.
6. Count false positives from that matrix.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod04-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/04-supervised-learning-classification/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod04-exercises-smoke OK")
```
