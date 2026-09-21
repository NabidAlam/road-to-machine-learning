# Chemistry and materials: ML domain track

You already work with spectra, compositions, process parameters, and lab notebooks. This track orders the shared ML modules for chemistry and materials-style tables and signals. It does not replace lab safety training. It does not guarantee a job.

**Core idea.** Keep chemistry or materials expertise. Add regression, evaluation, and optional imaging or unsupervised structure finding. Methods live in Modules `00`–`25`.

## Who this is for

- Chemistry, chemical engineering, and materials students
- Lab scientists predicting properties from composition or process settings
- People who want a Data Scientist path with optional ML Engineer depth later

## What you already bring

- Comfort with quantitative lab records
- Intuition for noise, calibration, and instrument drift
- Domain constraints (physical bounds, stoichiometry, process limits)

## ML problem types you will meet

- Property prediction (regression) from features
- Classification of material classes or pass/fail quality labels
- Unsupervised grouping of spectra or formulations
- Optional vision on micrographs after you can evaluate simpler models
- Careful splits so related samples do not leak across train and test

## Gaps this track closes

- [Module 01](../../01-python-for-data-science/README.md)
- [Modules 02](../../02-introduction-to-ml/README.md)–[05](../../05-model-evaluation-optimization/README.md)
- [Module 03](../../03-supervised-learning-regression/README.md) as the usual first modeling step
- [Module 08](../../08-unsupervised-learning/README.md) for exploration
- [Module 21](../../21-model-explainability/README.md)
- Optional [Module 11](../../11-computer-vision/README.md) for imaging

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | Data Scientist | Property models and evaluation |
| Alternate | ML Engineer | When models must run in a plant or lab pipeline |

See [Career Paths](../../README.md#career-paths) and [Career Roadmap Guide](../career_roadmap_guide.md). Treat times as emphasis maps only.

## Intensity maps

### Research support (tier B)

1. [Module 00](../../00-prerequisites/README.md) if needed
2. [Module 01](../../01-python-for-data-science/README.md)
3. [Module 02](../../02-introduction-to-ml/README.md)
4. [Module 03](../../03-supervised-learning-regression/README.md)
5. [Module 05](../../05-model-evaluation-optimization/README.md)
6. [Module 08](../../08-unsupervised-learning/README.md) when exploring unlabeled spectra
7. [Module 21](../../21-model-explainability/README.md)
8. One [Module 16](../../16-projects-beginner/README.md) regression-style project

### Job-oriented study (tier B toward C)

1. Complete Research support
2. Add [Module 04](../../04-supervised-learning-classification/README.md) and [Module 07](../../07-feature-engineering/README.md)
3. Add [Module 19](../../19-sql-database-fundamentals/README.md) if LIMS-style data is tabular at scale
4. Add [Module 13](../../13-model-deployment/README.md)–[14](../../14-mlops-basics/README.md) only for production roles
5. Add [Module 11](../../11-computer-vision/README.md) only for image-heavy work after Module 05

## Ordered study checklist

- [ ] Pick intensity map
- [ ] Finish Module 01 with a composition or spectra feature table
- [ ] Finish Modules 02, 03, and 05 with a split that respects sample families
- [ ] Enforce domain bounds on predictions (no impossible physical values without a note)
- [ ] Finish Module 21 for one stakeholder explanation
- [ ] Optional Module 08 cluster exploration with chemistry sense-checks

## Start here

Open [Module 00 README](../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../01-python-for-data-science/README.md).

## Honesty and traps

- A low RMSE can still violate chemistry. Check residuals against domain limits.
- Do not treat vendor “AI for materials” marketing as a methods curriculum.
- Lab safety and chemical handling stay outside this repo.

**Try next:** Open [Module 03: Supervised Learning Regression](../../03-supervised-learning-regression/README.md) after Module 01.
