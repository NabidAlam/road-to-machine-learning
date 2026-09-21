# Agriculture and agronomy (applied): ML domain track

You already work with seasons, fields, sensors, and messy observational records. This track orders the shared Road to ML spine for applied agriculture and agronomy-style analysis. It teaches methods. It does not prescribe farm practices. It does not guarantee yield, compliance, or a hire.

**Core idea.** Keep domain judgment about crops, livestock, soil, and weather context. Add Python, regression or classification, time series, and honest metrics. Methods live in Modules `00`–`25`.

## Who this is for

- Agriculture, agronomy, and agri-engineering learners
- Practitioners analyzing stations, IoT sensors, or remote-sensing derived tables
- Analysts who need forecasting habits without jumping straight to deep models

## What you already bring

- Comfort with seasons, growth stages, and field variation (even if informal)
- Respect for missing data, sensor drift, and site differences
- Stakeholder questions that need clear uncertainty talk

## ML problem types you will meet

- Time series forecasting and anomaly detection on sensor or station streams
- Regression on continuous yield or quality-style variables
- Classification of stress, quality, or event labels from features
- Careful train/test splits that respect time and location
- Communication of error so decisions are not oversold as prescriptions

## Gaps this track closes

- [Module 01](../../../01-python-for-data-science/README.md)
- [Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md)
- [Module 15](../../../15-time-series-analysis/README.md)
- [Module 03](../../../03-supervised-learning-regression/README.md) or [Module 04](../../../04-supervised-learning-classification/README.md) by task
- [Module 21](../../../21-model-explainability/README.md)
- [Module 19](../../../19-sql-database-fundamentals/README.md) when archives are tabular

## Related resources

- [Data Validation](../../data_validation.md). Sensors drift. Validate before you trust a forecast.
- [Ethics in ML](../../ethics_in_ml.md). Field and farm data still need careful use framing.
- [Common Errors](../../common_errors.md). Leakage and shuffle mistakes hit seasonal data hard.
- [Model Explainability Cheatsheet](../../model_explainability_cheatsheet.md). Quick sheet beside Module 21.
- [Stakeholder Communication](../../stakeholder_communication.md). Uncertainty talk without agronomic prescriptions.

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | Data Scientist | Models plus evaluation on observational field data |
| Alternate | ML Engineer | When models must run in a monitoring or ops pipeline |

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
8. Time-aware practice via [Module 15](../../../15-time-series-analysis/README.md) or a [Module 16](../../../16-projects-beginner/README.md) project you adapt carefully

### Job-oriented study (tier B toward C)

1. Complete Research support
2. Add [Module 19](../../../19-sql-database-fundamentals/README.md) and [Module 07](../../../07-feature-engineering/README.md)
3. Add [Module 13](../../../13-model-deployment/README.md)–[14](../../../14-mlops-basics/README.md) only for production monitoring roles
4. Keep deep learning optional until baselines and Module 15 habits are solid

## Ordered study checklist

- [ ] Pick intensity map
- [ ] Finish Module 01 with a public agricultural or sensor CSV
- [ ] Write a time-based (and site-aware) split before any fancy model
- [ ] Finish Module 05 and Module 15 for ordered data
- [ ] Practice one forecast or anomaly plot with error described in plain words
- [ ] Document what the model must not be used for (agronomic prescription, guaranteed yield)

## Start here

Open [Module 00 README](../../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../../01-python-for-data-science/README.md).

## Honesty and traps

- Educational content only. Not farm advice, chemical recommendations, or regulatory compliance guidance.
- Random shuffles destroy time series truth. Prefer time order.
- Spatial leakage is real when nearby fields share train and test.
- Vendor “AI for yield” marketing is not a methods curriculum.

**Try next:** Open [Module 15: Time Series Analysis](../../../15-time-series-analysis/README.md) after Modules 01, 02, 05, and one supervised module (03 or 04).
