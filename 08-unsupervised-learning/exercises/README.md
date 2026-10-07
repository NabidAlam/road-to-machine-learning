# Module 08 Exercise Pack

KMeans clustering and silhouette on a tiny blob set.

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

1. Fit KMeans(k=3) and return labels.
2. Count unique clusters.
3. Compute silhouette score.
4. Pick best k in {2,3,4} by silhouette.
5. Report inertia for k=3.
6. Confirm label vector length equals n_samples (120).

## Hub smoke (do not edit)

CI runs the fence below from the hub repo root.

```python snippet-id=mod08-exercises-smoke
import runpy
import sys
from pathlib import Path

p = Path("road-to-machine-learning/08-unsupervised-learning/exercises/solutions.py").resolve()
sys.path.insert(0, str(p.parent))
runpy.run_path(str(p), run_name="__main__")
print("mod08-exercises-smoke OK")
```
