# Eval traps: life sciences research data

Batch effects and rare labels can fake a breakthrough. This chapter names the traps for research-style ML. Educational content only. Not clinical advice. Not diagnosis or treatment guidance.

## Trap 1. Batch and site leakage

Samples from the same batch, plate, site, or sequencing run share technical artifacts. A random split lets the model memorize the batch, not the biology.

- Split by batch, site, or patient/sample id ([Module 05](../../../05-model-evaluation-optimization/README.md)).
- Report metrics on held-out batches, not only shuffled rows.
- Plot feature distributions by batch before you trust a score ([Data Validation](../../data_validation.md)).

## Trap 2. Imbalance theater

Rare positives make accuracy look excellent while the model predicts the majority class every time.

- Use precision, recall, PR curves, and cost-aware thresholds ([Module 20](../../../20-handling-imbalanced-data/README.md)).
- Keep a majority-class baseline in the write-up.
- Do not oversample before the split. That leaks test information.

## Trap 3. Causal question, predictive method

A collaborator asks what intervention changed an outcome. You shipped a predictor of who already has the label.

- Prediction answers “who looks like the label.” It does not answer “what caused the change.”
- Read the [Causal Inference Guide](../../causal_inference_guide.md) when the research question is causal.
- Keep claims matched to the design you actually have.

## Trap 4. Label leakage from processing pipelines

Normalized values, QC flags, or post-selection filters can encode the label indirectly.

- Rebuild features with only information available before the label decision.
- Document every preprocessing step that drops samples.
- Prefer a simple pipeline you can re-run from raw exports.

## Trap 5. Cross-study mirages

A model trained on one cohort collapses on another lab’s protocol.

- Treat external cohorts as the real test when you can.
- Name domain shift explicitly in the limits section.
- Do not market a single-site notebook as general biology truth.

## Trap 6. Clinical overclaim

Research metrics are not clinical validation. Educational projects must not advise diagnosis or treatment.

- State non-clinical use in the README.
- Do not publish patient-identifying data.
- Read [Ethics in ML](../../ethics_in_ml.md) early.

## Quick checklist before you claim a win

- [ ] Split respects batch, site, or sample identity
- [ ] Metrics fit rare positives when needed
- [ ] Claim type matches the design (predictive vs causal)
- [ ] Preprocessing cannot see the test labels
- [ ] Explicit non-clinical disclaimer
- [ ] One explained false positive for a domain collaborator ([Module 21](../../../21-model-explainability/README.md))

**Try next:** Open [Module 05](../../../05-model-evaluation-optimization/README.md), then [Module 20](../../../20-handling-imbalanced-data/README.md) if your positives are rare.
