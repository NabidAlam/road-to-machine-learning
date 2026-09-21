# Physics and physical sciences: ML domain track

You already think in models, measurement error, and experiments. This track shows which parts of the shared Road to ML spine matter first for physical-science work. It does not replace a physics degree. It does not guarantee a research post or industry hire.

**Core idea.** Keep your domain depth. Add a thin, honest ML methods path. Use Modules `00`–`25` for methods. Use this syllabus for order and emphasis.

## Who this is for

- Physics / astronomy / applied physics students who need ML for thesis or lab data
- Researchers who want careful supervised learning and uncertainty habits
- Switchers from physical sciences who want a Data Scientist emphasis without skipping evaluation

## What you already bring

- Comfort with equations, units, and approximate models
- Lab or simulation data with noise and systematic error
- Respect for measurement limits (that maps cleanly onto leakage and generalization)

## ML problem types you will meet

- Regression on continuous observables
- Time series and signals (detectors, sensors, orbits, lab traces)
- Anomaly and quality flags on instrument streams
- Surrogate models that approximate expensive simulations (only after you can evaluate them)
- Classification when labels are discrete states of a system

## Gaps this track closes on the shared spine

- Python and tabular workflows ([Module 01](../../../01-python-for-data-science/README.md))
- Supervised learning and evaluation discipline ([Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md))
- Time series habits ([Module 15](../../../15-time-series-analysis/README.md))
- Explaining model behavior when a collaborator asks why ([Module 21](../../../21-model-explainability/README.md))

## Role emphasis (not destiny)

| Emphasis | Hub / README role | Why |
|----------|-------------------|-----|
| Primary | Data Scientist | Predictive models, evaluation, and careful experiments match lab culture |
| Alternate (later) | ML Engineer | When you must ship a model into a pipeline, not only analyze offline |

See the main [Career Paths](../../../README.md#career-paths) table and [Career Roadmap Guide](../../career_roadmap_guide.md). Treat times as emphasis maps only.

## Intensity maps

Pick one. You can change later.

### Research support (tier B)

Goal: trustworthy analysis for papers, theses, and lab decisions.

1. [Module 00](../../../00-prerequisites/README.md) if math or Python is shaky
2. [Module 01](../../../01-python-for-data-science/README.md)
3. [Module 02](../../../02-introduction-to-ml/README.md)
4. [Module 03](../../../03-supervised-learning-regression/README.md)
5. [Module 05](../../../05-model-evaluation-optimization/README.md) (do not skip)
6. [Module 15](../../../15-time-series-analysis/README.md) if your data is ordered in time
7. [Module 21](../../../21-model-explainability/README.md)
8. One beginner project from [Module 16](../../../16-projects-beginner/README.md) that practices regression or forecasting habits

Optional later: [Module 08](../../../08-unsupervised-learning/README.md) for structure discovery. [Module 09](../../../09-neural-networks-basics/README.md)–[10](../../../10-deep-learning-frameworks/README.md) only when simpler models fail for a clear reason.

### Job-oriented study (tier B toward C)

Goal: portfolio evidence that you can clean data, train, evaluate, and explain. Still no hire guarantee.

1. Complete the Research support list
2. Add [Module 04](../../../04-supervised-learning-classification/README.md) and [Module 07](../../../07-feature-engineering/README.md)
3. Add [Module 19](../../../19-sql-database-fundamentals/README.md) if your workplace data lives in tables
4. Add [Module 13](../../../13-model-deployment/README.md)–[14](../../../14-mlops-basics/README.md) only when you need production habits
5. Prefer one intermediate project from [Module 17](../../../17-projects-intermediate/README.md) that you can defend end to end

## Ordered study checklist

Use this as your weekly spine.

- [ ] Skim this syllabus and pick Research support or Job-oriented study
- [ ] Finish Module 01 until you can load, clean, and plot a noisy CSV on your own
- [ ] Finish Module 02 until you can state train vs serve vs eval in your own words
- [ ] Finish Module 03 with a physics-flavored dataset of your own (even a small public one)
- [ ] Finish Module 05 and write down your split rule (time, run id, or experiment batch)
- [ ] If signals matter, finish Module 15 before deep models
- [ ] Read Module 21 and practice explaining one wrong prediction
- [ ] Ship one small project README with data source, split, metric, and failure case

## Start here

Open [Module 00 README](../../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../../01-python-for-data-science/README.md).

## Honesty and traps

- Fancy deep models without Module 05 habits will impress nobody who reads papers carefully.
- Do not call a notebook “production ML.”
- Medical imaging that sits near physics stays educational. It is not clinical advice.
- Surrogate modeling of simulations needs error bars and domain checks. A low loss is not physical truth.

## Where this connects

Domain chapters for physics (literacy, uncertainty, methods map) will land in this folder next. Until then, the Module NN links above are the teaching content.

**Try next:** Open [Module 01: Python for Data Science](../../../01-python-for-data-science/README.md) and complete its core path with a noisy measurement CSV of your choosing.
