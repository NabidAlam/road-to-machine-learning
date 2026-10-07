# Module 10 Exercise Pack

Framework-free smoke. Numpy and sklearn only so CI works without torch.

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

1. Fit a small sklearn MLPClassifier and score the test set.
2. Return the hidden_layer_sizes tuple you chose.
3. Write a one-line note that torch/tensorflow are optional here.
4. Sum predict_proba for one row (should be ~1).
5. Count unique classes in y_train.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod10-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/10-deep-learning-frameworks/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod10-exercises-smoke OK")
```
