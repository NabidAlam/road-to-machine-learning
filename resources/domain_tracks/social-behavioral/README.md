# Social and behavioral sciences (research): ML domain track

You already think in studies, surveys, and careful inference. This track maps that habit onto the shared Road to ML spine for research support. It is research framing only. It is not therapy. It is not counseling. It does not diagnose. It does not guarantee a hire.

**Core idea.** Keep social-science judgment. Add Python, supervised learning, imbalance awareness, and explainability. Methods live in Modules `00`–`25`.

## Who this is for

- Psychology, sociology, political science, education research, and related students
- Researchers who need classifiers or predictors on survey-style or study tables
- Switchers who want Data Scientist or Data Analyst emphasis with honest limits

## What you already bring

- Experimental or survey design instincts
- Respect for sampling bias, non-response, and noisy labels
- Domain labels that are expensive or imperfect

## ML problem types you will meet

- Classification of outcomes, attitudes, or group labels from features
- Imbalanced labels (rare events in research data)
- Careful evaluation across sites, waves, or cohorts so leakage does not fake performance
- Explainability when a collaborator asks which features drove a call
- Optional causal reading when the question is “what changed outcomes,” not only “what predicts”

## Gaps this track closes

- [Module 01](../../../01-python-for-data-science/README.md)
- [Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md)
- [Module 04](../../../04-supervised-learning-classification/README.md) as a common first modeling step
- [Module 20](../../../20-handling-imbalanced-data/README.md) when classes are rare
- [Module 21](../../../21-model-explainability/README.md)
- [Causal Inference Guide](../../causal_inference_guide.md) when prediction is not enough

## Related resources

- [Causal Inference Guide](../../causal_inference_guide.md). When the question is what changed outcomes, not only what predicts.
- [Ethics in ML](../../ethics_in_ml.md). Research data can harm people. Read this early.
- [Stakeholder Communication](../../stakeholder_communication.md). Explain metrics to collaborators without overclaim.
- [Imbalanced Data Cheatsheet](../../imbalanced_data_cheatsheet.md). Rare events and skewed labels are common in studies.
- [Model Explainability Cheatsheet](../../model_explainability_cheatsheet.md). Quick sheet beside Module 21.
- [Data Validation](../../data_validation.md). Survey and study tables need honest checks before modeling.

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | Data Scientist | Models plus evaluation match research questions |
| Alternate | Data Analyst | Strong when clean reporting and careful slices come first |

See [Career Paths](../../../README.md#career-paths) and [Career Roadmap Guide](../../career_roadmap_guide.md). Treat times as emphasis maps only.

## Intensity maps

### Research support (tier B)

1. [Module 00](../../../00-prerequisites/README.md) if needed
2. [Module 01](../../../01-python-for-data-science/README.md)
3. [Module 02](../../../02-introduction-to-ml/README.md)
4. [Module 04](../../../04-supervised-learning-classification/README.md)
5. [Module 05](../../../05-model-evaluation-optimization/README.md)
6. [Module 20](../../../20-handling-imbalanced-data/README.md)
7. [Module 21](../../../21-model-explainability/README.md)
8. One [Module 16](../../../16-projects-beginner/README.md) classification-style project you can describe without hype

Optional: read the [Causal Inference Guide](../../causal_inference_guide.md) after Module 05 is solid.

### Job-oriented study (tier B)

1. Complete Research support
2. Add [Module 07](../../../07-feature-engineering/README.md) and [Module 19](../../../19-sql-database-fundamentals/README.md)
3. Add [Module 03](../../../03-supervised-learning-regression/README.md) if you predict continuous endpoints
4. Keep deployment modules for later unless the role truly ships models

## Ordered study checklist

- [ ] Pick intensity map
- [ ] Finish Module 01 with a survey-style or study table (public or own)
- [ ] Finish Modules 02, 04, and 05 with a written split by wave, site, or participant id
- [ ] Finish Module 20 if positives are rare
- [ ] Finish Module 21 and explain one false positive as you would to a research collaborator
- [ ] Document ethics: what this model must never claim about therapy, counseling, or diagnosis

## Start here

Open [Module 00 README](../../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../../01-python-for-data-science/README.md).

## Honesty and traps

- Educational content only. Not clinical, counseling, or therapy advice.
- Prediction is not causation. Say so when stakeholders mix them up.
- Do not publish identifying participant data in public repos.
- Deep learning is optional. Many tables need strong classical baselines first.

**Try next:** Open [Module 04: Supervised Learning Classification](../../../04-supervised-learning-classification/README.md) after you can wrangle a table in Module 01.
