# Earth and environment: ML domain track

You already work with spatial, seasonal, and messy observational data. This track orders the shared ML spine for earth science, climate-adjacent analysis, and environmental monitoring. It does not guarantee policy impact or a hire. Claims about the planet need careful evaluation language.

**Core idea.** Keep domain judgment. Add Python, regression or classification, time series, and honest metrics. Methods live in Modules `00`–`25`.

## Who this is for

- Earth science, geography, ecology, and environmental studies learners
- People analyzing sensors, stations, or remote-sensing derived tables
- Analysts who need forecasting habits without jumping straight to deep models

## What you already bring

- Comfort with maps, seasons, and spatial dependence (even if informal)
- Respect for missing data and instrument outages
- Stakeholder questions that need clear uncertainty talk

## ML problem types you will meet

- Time series forecasting and anomaly detection on station or sensor streams
- Regression on continuous environmental variables
- Classification of land cover or event labels from features
- Careful train/test splits that respect time and location
- Communication of error so decisions are not oversold

## Gaps this track closes

- [Module 01](../../../01-python-for-data-science/README.md)
- [Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md)
- [Module 15](../../../15-time-series-analysis/README.md)
- [Module 03](../../../03-supervised-learning-regression/README.md) or [Module 04](../../../04-supervised-learning-classification/README.md) by task
- [Module 21](../../../21-model-explainability/README.md)
- [Module 19](../../../19-sql-database-fundamentals/README.md) when archives are tabular

## Related resources

- [Data Validation](../../data_validation.md). Station and sensor series need checks before modeling.
- [Ethics in ML](../../ethics_in_ml.md). Environmental claims in public need caution.
- [Common Errors](../../common_errors.md). Time and space leakage are common failure modes.
- [Model Explainability Cheatsheet](../../model_explainability_cheatsheet.md). Quick sheet beside Module 21.
- [Imbalanced Data Cheatsheet](../../imbalanced_data_cheatsheet.md). Rare events and extremes need careful metrics.

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | Data Scientist | Models plus evaluation on observational data |
| Alternate | Data Analyst | Strong when reporting and SQL matter more than new models |

See [Career Paths](../../../README.md#career-paths) and [Career Roadmap Guide](../../career_roadmap_guide.md). Treat times as emphasis maps only.

## Intensity maps

### Research support (tier B)

1. [Module 00](../../../00-prerequisites/README.md) if needed
2. [Module 01](../../../01-python-for-data-science/README.md)
3. [Module 02](../../../02-introduction-to-ml/README.md)
4. [Module 03](../../../03-supervised-learning-regression/README.md) or [Module 04](../../../04-supervised-learning-classification/README.md)
5. [Module 05](../../../05-model-evaluation-optimization/README.md)
6. [Module 15](../../../15-time-series-analysis/README.md)
7. [Module 21](../../../21-model-explainability/README.md)
8. Time-aware project practice via [Module 15](../../../15-time-series-analysis/README.md) exercises or a [Module 16](../../../16-projects-beginner/README.md) project you adapt carefully

### Job-oriented study (tier B)

1. Complete Research support
2. Add [Module 19](../../../19-sql-database-fundamentals/README.md) and [Module 07](../../../07-feature-engineering/README.md)
3. Add [Module 20](../../../20-handling-imbalanced-data/README.md) for rare events
4. Keep deep learning optional until baselines and Module 15 habits are solid

## Ordered study checklist

- [ ] Pick intensity map
- [ ] Finish Module 01 with a public environmental CSV
- [ ] Write a time-based split before any fancy model
- [ ] Finish Module 05 and Module 15 for ordered data
- [ ] Practice one forecast plot with error bands described in plain words
- [ ] Document what the model must not be used for (policy overclaim)

## Start here

Open [Module 00 README](../../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../../01-python-for-data-science/README.md).

## Honesty and traps

- Random shuffles destroy time series truth. Prefer time order.
- Spatial leakage is real when nearby points share train and test.
- Climate and environment claims in public need caution. This track teaches methods, not advocacy scripts.

**Try next:** Open [Module 15: Time Series Analysis](../../../15-time-series-analysis/README.md) after Modules 01, 02, 05, and one supervised module (03 or 04).
