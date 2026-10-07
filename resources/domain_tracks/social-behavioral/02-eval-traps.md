# Eval traps: social and behavioral research data

Survey waves, sites, and sampling bias can fake clean accuracy. This chapter names research-style traps. It is research framing only. Not therapy. Not counseling. Not diagnosis.

## Trap 1. Wave, site, and participant leakage

The same person, classroom, platform, or survey wave appears in train and test. The model memorizes the cluster.

- Split by participant id, site, or wave ([Module 05](../../../05-model-evaluation-optimization/README.md)).
- Hold out a later wave when the study is longitudinal.
- Validate that IDs do not leak through feature joins ([Data Validation](../../data_validation.md)).

## Trap 2. Sampling and non-response mirages

Convenience samples and dropouts reshape the label distribution. Metrics on the observed table may not match the population you care about.

- Write who is missing and why.
- Slice metrics by groups already in the study design.
- Do not treat a volunteer web sample as a census.

## Trap 3. Imbalanced rare outcomes

Rare events make accuracy look strong while the model predicts the majority every time.

- Prefer precision, recall, and PR-style views ([Module 20](../../../20-handling-imbalanced-data/README.md)).
- Keep a majority baseline in the write-up.
- Do not oversample before the split.

## Trap 4. Causal question, predictive method

A collaborator asks what changed attitudes or outcomes. You only built a predictor of who already matches the label.

- Prediction is not causation.
- Read the [Causal Inference Guide](../../causal_inference_guide.md) when the question is interventional or attributional.
- Match claims to the design you have (observational survey vs experiment).

## Trap 5. Construct and label leakage

Proxy labels, post-survey coder notes, or outcome-derived composites sneak into features.

- Freeze features at the prediction or survey time you claim.
- Document how the label was coded.
- Prefer simple, auditable feature sets before complex embeddings.

## Trap 6. Harmful overclaim

Research metrics are not clinical or counseling advice. Educational projects must not diagnose people.

- State research-only use in the README.
- Avoid publishing identifying survey responses.
- Read [Ethics in ML](../../ethics_in_ml.md) and practice plain limits with [Stakeholder Communication](../../stakeholder_communication.md).

## Quick checklist before you claim a win

- [ ] Split respects person, site, or wave
- [ ] Missingness and sampling limits written down
- [ ] Metrics fit rare outcomes when needed
- [ ] Predictive vs causal claim type is explicit
- [ ] Non-therapy / non-diagnosis disclaimer present
- [ ] One explained error for a collaborator ([Module 21](../../../21-model-explainability/README.md))

**Try next:** Open [Module 05](../../../05-model-evaluation-optimization/README.md), then the [Causal Inference Guide](../../causal_inference_guide.md) if your question is about what changed outcomes.
