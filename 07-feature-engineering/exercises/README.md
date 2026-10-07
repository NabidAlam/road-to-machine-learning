# Module 07 Exercise Pack

Encoding categoricals and simple numeric transforms.

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

1. One-hot encode color and report the array shape.
2. Count one-hot columns for color.
3. Ordinal-encode size with S < M < L.
4. Apply log1p to price.
5. Z-score price with population std.
6. Stack one-hot + ordinal + z-score into one matrix.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod07-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/07-feature-engineering/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod07-exercises-smoke OK")
```
