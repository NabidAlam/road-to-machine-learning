# Life sciences: ML domain track

You already think in experiments, controls, and biological variation. This track maps that habit onto the shared Road to ML spine. It is for research support and careful analysis. It is not clinical advice. It does not diagnose. It does not guarantee a hire.

**Core idea.** Keep biology or biotech depth. Add Python, supervised learning, imbalance awareness, and explainability. Study methods in Modules `00`–`25`. Use this syllabus for order.

## Who this is for

- Biology, biotech, bioinformatics-curious students
- Lab scientists who need classifiers or predictors on assay-style tables
- Switchers who want a Data Scientist or Data Analyst emphasis with honest limits

## What you already bring

- Experimental design instincts (controls, replicates, batches)
- Respect for biological noise and batch effects
- Domain labels that are expensive or imperfect

## ML problem types you will meet

- Classification of samples, states, or outcomes from features
- Imbalanced labels (rare events, rare conditions in research data)
- Feature sets from assays, sequence summaries, or imaging-derived tables
- Explainability when a collaborator asks which features drove a call
- Careful evaluation across batches or sites so leakage does not fake performance

## Gaps this track closes

- [Module 01](../../../01-python-for-data-science/README.md) for tables and plots
- [Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md) for supervised learning and eval
- [Module 20](../../../20-handling-imbalanced-data/README.md) when classes are rare
- [Module 21](../../../21-model-explainability/README.md) for trust with domain experts
- [Module 19](../../../19-sql-database-fundamentals/README.md) when data lives in warehouses

## Related resources

- [Ethics in ML](../../ethics_in_ml.md). Research data can harm people. Not clinical advice.
- [Imbalanced Data Cheatsheet](../../imbalanced_data_cheatsheet.md). Rare labels are common in assays and studies.
- [Causal Inference Guide](../../causal_inference_guide.md). When prediction is not enough for the research question.
- [Data Validation](../../data_validation.md). Batch effects and messy tables need early checks.
- [Model Explainability Cheatsheet](../../model_explainability_cheatsheet.md). Quick sheet beside Module 21.

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | Data Scientist | Models plus evaluation match research questions |
| Alternate | Data Analyst | Strong when the job is clean reporting and careful slices first |

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

Optional: [Module 12](../../../12-natural-language-processing/README.md) only for text-heavy literature tasks. [Module 11](../../../11-computer-vision/README.md) only for image assays after Module 05 is solid.

### Job-oriented study (tier B)

1. Complete Research support
2. Add [Module 07](../../../07-feature-engineering/README.md) and [Module 19](../../../19-sql-database-fundamentals/README.md)
3. Add [Module 03](../../../03-supervised-learning-regression/README.md) if you predict continuous endpoints
4. Keep deployment modules for later unless the role truly ships models

## Ordered study checklist

- [ ] Pick Research support or Job-oriented study
- [ ] Finish Module 01 with a real assay-style table (public or own)
- [ ] Finish Modules 02, 04, and 05 with a written split rule by batch or patient/sample id
- [ ] Finish Module 20 if positives are rare
- [ ] Finish Module 21 and explain one false positive as you would to a biology collaborator
- [ ] Document ethics: what this model must never claim in a clinical setting

## Start here

[Module 01 README](../../../01-python-for-data-science/README.md) unless you need [Module 00](../../../00-prerequisites/README.md) first.

## Honesty and traps

- Educational content only. Not medical advice.
- Batch effects can look like “great accuracy.” Split with that in mind.
- Do not publish patient-identifying data in public repos.
- Deep learning is optional. Many tables need strong classical baselines first.

**Try next:** Open [Module 04: Supervised Learning Classification](../../../04-supervised-learning-classification/README.md) after you can wrangle a table in Module 01.
