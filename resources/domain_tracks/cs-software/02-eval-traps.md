# Eval traps: CS and software backgrounds

Software habits can hide ML failure modes. This chapter names the traps that show up when you treat models like ordinary services. Methods still live in Modules `00`–`25`.

## Trap 1. Offline green, online wrong

You unit-test the handler. The model returns JSON. Latency looks fine. Users still get worse outcomes.

- Offline metrics must match the product question ([Module 05](../../../05-model-evaluation-optimization/README.md)).
- Train/serve skew appears when feature code in training differs from the live path ([Module 13](../../../13-model-deployment/README.md)–[14](../../../14-mlops-basics/README.md)).
- Log the features you actually scored on. Do not trust a notebook-only pipeline.

## Trap 2. Random split on time-ordered or user-linked rows

Shuffling rows is normal for many coding datasets. It is often wrong for product logs.

- Split by time when the world drifts.
- Split by user or entity id when the same person appears in train and test.
- Write the split rule in the README before you tune models.

## Trap 3. Leakage through future joins

You join “conversion” or “churn” tables that already include outcomes known only after the prediction time.

- Build features as of a cutoff timestamp.
- Ban post-decision columns from the feature set ([Module 07](../../../07-feature-engineering/README.md)).
- Prefer a simple baseline that uses only known-at-score fields.

## Trap 4. Accuracy theater on rare events

Fraud, abuse, and rare defects look like ordinary classification until positives are 0.1%.

- Prefer precision, recall, PR curves, and cost-aware thresholds ([Module 20](../../../20-handling-imbalanced-data/README.md)).
- Do not celebrate 99% accuracy on a 1% positive class.

## Trap 5. Wrapping a foundation model without an eval harness

Calling an API is easy. Knowing whether the product improved is not.

- Define graded examples and failure classes before launch.
- Track regressions when prompts or models change ([Module 14](../../../14-mlops-basics/README.md)).
- Read [Ethics in ML](../../ethics_in_ml.md) when user text or decisions can harm people.

## Trap 6. System design without a measurement plan

A clean architecture diagram does not prove the model helps.

- Pair [ML System Design Guide](../../ml_system_design_guide.md) with Module 05 metrics.
- Name the online signal you would watch if you had production traffic. Even a portfolio project can invent a honest proxy metric.

## Quick checklist before you claim a win

- [ ] Split rule written (time, entity, or both)
- [ ] Features available at prediction time only
- [ ] Metric matches the product cost of errors
- [ ] Baseline beaten for a reason you can explain
- [ ] Failure cases listed for the next iteration

**Try next:** Open [Module 05](../../../05-model-evaluation-optimization/README.md), then skim [ML System Design Guide](../../ml_system_design_guide.md) with one concrete service in mind.
