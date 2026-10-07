# Experiment Design and A/B Testing (Short Primer)

A compact lab note for shipping model or product changes without fooling yourself.

## What you are deciding

You change one thing (model, threshold, UI copy). You ask whether the change improved a metric for real users or real traffic. Correlation in a backtest is not the same as a controlled lift.

## Core pieces

1. **Unit.** User, session, or request. Pick one and keep it stable for the whole test.
2. **Assignment.** Randomize units into control vs treatment. Stratify if a segment is tiny.
3. **Primary metric.** One number you will ship on. Example. Conversion rate. Or minority-class recall at a fixed precision.
4. **Guardrails.** Latency, crash rate, complaint rate. A win on the primary metric that burns a guardrail is still a no-ship.
5. **Duration and power.** Run long enough to cover weekly cycles. Do not peek every hour and stop on the first green blip.

## Minimum design checklist

- Written hypothesis before launch.
- Fixed primary metric and minimum detectable effect.
- Clear stop rule (end date or sample size), not "when it looks good".
- Same logging in both arms.
- A holdout or reverse test plan if the change is sticky (cached models, trained-on-clicks loops).

## Simple analysis sketch

```python
import numpy as np

# Toy counts: successes / trials
control = (120, 2000)
treatment = (150, 2000)

p_c = control[0] / control[1]
p_t = treatment[0] / treatment[1]
lift = p_t - p_c

# Two-proportion z-test (quick lab version)
p_pool = (control[0] + treatment[0]) / (control[1] + treatment[1])
se = np.sqrt(p_pool * (1 - p_pool) * (1 / control[1] + 1 / treatment[1]))
z = lift / se
print({"p_control": p_c, "p_treatment": p_t, "lift": lift, "z": z})
```

For real decisions, use a stats library, pre-register the test, and correct for multiple metrics if you peeked at many.

## Leakage and contamination

- Users in both arms (shared accounts, shared devices).
- Spillover (network effects, shared inventory).
- Train/serve skew where the new model only sees treatment traffic and then "proves" itself.

## When not to A/B

- Sample is too small for the effect you care about.
- The change is a safety fix that must ship now.
- Offline eval already fails hard gates (calibration, fairness slice, latency).

## Related

- [causal_inference_guide.md](causal_inference_guide.md)
- Module 05 evaluation habits in [../05-model-evaluation-optimization/README.md](../05-model-evaluation-optimization/README.md)
