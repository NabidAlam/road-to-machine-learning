# Transfer strengths: physics to ML

You already think in models, units, and measurement error. This chapter maps that habit onto Modules `00`–`25`. It does not replace a physics degree. It does not guarantee a research post or hire.

## What you already do well

| Habit you bring | Where it helps on the spine |
|-----------------|-----------------------------|
| Equations, units, approximate models | Cleaner features and sanity checks in [Module 01](../../../01-python-for-data-science/README.md)–[03](../../../03-supervised-learning-regression/README.md) |
| Measurement noise and systematics | Honest evaluation in [Module 05](../../../05-model-evaluation-optimization/README.md) and [Data Validation](../../data_validation.md) |
| Time-ordered detector or lab traces | [Module 15](../../../15-time-series-analysis/README.md) |
| Explaining fits to collaborators | [Module 21](../../../21-model-explainability/README.md) |
| Respect for limits of a model | Saying what the predictor must not claim |

## Module map for common physical-science tasks

| Question shape | Start here |
|----------------|------------|
| Continuous observable from features | [Module 03](../../../03-supervised-learning-regression/README.md) then [Module 05](../../../05-model-evaluation-optimization/README.md) |
| Signal or sensor stream over time | [Module 15](../../../15-time-series-analysis/README.md) |
| Discrete state labels | [Module 04](../../../04-supervised-learning-classification/README.md) |
| Why did the model miss this point? | [Module 21](../../../21-model-explainability/README.md) |

Keep [Math Formulas](../../math_formulas.md) and [Prerequisites Cheatsheet](../../prerequisites_cheatsheet.md) nearby while you shore up Module 00 gaps.

## Worked sketch

You have a lab or simulation table. Rows are runs or time samples. The target is a continuous measurable.

1. Wrangle units and missing values in [Module 01](../../../01-python-for-data-science/README.md).
2. Fit a simple regressor ([Module 03](../../../03-supervised-learning-regression/README.md)).
3. Split so calibration drift or run identity cannot fake the score ([Module 05](../../../05-model-evaluation-optimization/README.md)).
4. If the series is ordered in time, move the holdout to the newest window ([Module 15](../../../15-time-series-analysis/README.md)).
5. Report residual structure, not only a single RMSE number.

## What to delay

- Deep nets before a strong linear or tree baseline has failed for a clear reason.
- Deployment modules until you actually need a service path.
- Surrogate models for expensive simulations until you can evaluate them against held-out physics checks.

## Honesty

- A low training loss is not a measurement.
- Portfolio practice is not a hire promise.

**Try next:** Open [Module 03](../../../03-supervised-learning-regression/README.md) after [Module 01](../../../01-python-for-data-science/README.md), then refuse to skip [Module 05](../../../05-model-evaluation-optimization/README.md).
