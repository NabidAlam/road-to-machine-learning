# Module 09 Exercise Pack

Tiny numpy forward pass. No framework required.

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

1. Implement ReLU.
2. Implement sigmoid.
3. Implement a linear layer X @ W + b.
4. Compose a two-layer forward pass (ReLU then sigmoid).
5. Threshold probabilities to hard labels.
6. Compute accuracy from y_true and y_pred.

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod09-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/09-neural-networks-basics/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod09-exercises-smoke OK")
```
