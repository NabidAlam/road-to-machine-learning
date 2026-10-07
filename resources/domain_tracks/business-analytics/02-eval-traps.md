# Eval traps: business analytics and ops

Business tables make leakage look like a KPI win. This chapter names the field-specific traps. Methods still live in Modules `00`–`25`.

## Trap 1. Join leakage

You join fact tables after the decision time. The model “predicts” churn using columns that only exist once the customer already left.

- Freeze features as of the prediction timestamp ([Module 07](../../../07-feature-engineering/README.md)).
- Validate join keys and grain before modeling ([Data Validation](../../data_validation.md), [Module 19](../../../19-sql-database-fundamentals/README.md)).
- Ask: would this column exist if I scored a brand-new row today?

## Trap 2. KPI leakage and circular labels

The label is built from the same report metric you put in the features. Accuracy looks perfect. The model learned the KPI definition, not a useful signal.

- Separate the label recipe from the feature recipe in writing.
- Ban transformed versions of the target (including rolling versions that include the label window).
- Prefer a baseline that uses only prior-period inputs.

## Trap 3. Shuffled time and campaign data

Random train/test splits on weekly sales or campaign logs mix tomorrow into yesterday.

- Split by time for forecasts and most ops scores ([Module 05](../../../05-model-evaluation-optimization/README.md), [Module 15](../../../15-time-series-analysis/README.md)).
- Hold out the newest weeks for the final check.
- Watch seasonality. A model that “wins” on one holiday week may fail the next.

## Trap 4. Stakeholder metrics vs model metrics

Accuracy looks fine in the notebook. Ops cares about false alarms that flood a queue, or missed high-value accounts.

- Translate errors into business costs before you tune ([Stakeholder Communication](../../stakeholder_communication.md)).
- Report precision/recall or RMSE in the units people already use.
- Agree on the decision threshold with the owner of the process, not only with the notebook default.

## Trap 5. Cohort and segment mirages

Overall AUC looks strong. One region or customer tier carries all the lift. Another segment is noise.

- Slice metrics by the segments stakeholders already track.
- Do not average away a failing cohort.
- Document where the model must not be used.

## Trap 6. Dashboard vanity without a holdout

A live dashboard of “model scores” is not proof. Without a holdout or backtest rule, you are watching a story.

- Keep an offline evaluation report beside any dashboard ([Module 05](../../../05-model-evaluation-optimization/README.md)).
- Version the query and the training cutoff date.
- Read [Ethics in ML](../../ethics_in_ml.md) when scores affect customers or employees.

## Quick checklist before you claim a win

- [ ] Prediction time and feature cutoff written down
- [ ] Label definition matches the KPI doc without circular features
- [ ] Time-aware or entity-aware split
- [ ] Metric mapped to stakeholder cost
- [ ] Segment slices reviewed
- [ ] Explicit non-claims (no guaranteed revenue, no investment advice)

**Try next:** Open [Module 05](../../../05-model-evaluation-optimization/README.md), then practice one join-safe feature set with [Module 19](../../../19-sql-database-fundamentals/README.md).
