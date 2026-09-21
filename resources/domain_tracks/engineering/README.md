# Non-software engineering: ML domain track

You already think in systems, sensors, tolerances, and failure modes. This track maps that mindset onto the shared ML spine for mechanical, electrical, civil, and similar engineering backgrounds. It does not replace professional engineering certification. It does not guarantee a hire.

**Core idea.** Keep engineering judgment. Add data workflows, supervised learning, evaluation, then deployment awareness when you need models in a loop. Methods live in Modules `00`–`25`.

## Who this is for

- Engineering students outside pure software tracks
- Working engineers adding predictive maintenance, quality, or sensor analytics
- People aiming at ML Engineer emphasis with a solid Data Scientist base first

## What you already bring

- Systems thinking and unit awareness
- Sensor and logging intuition
- Respect for safety and failure consequences

## ML problem types you will meet

- Regression on continuous sensor outputs
- Classification for fault or quality states
- Time series on equipment streams
- Optional vision for inspection after strong baselines
- Deployment and monitoring when a model affects a real process

## Gaps this track closes

- [Module 01](../../01-python-for-data-science/README.md)
- [Modules 02](../../02-introduction-to-ml/README.md)–[05](../../05-model-evaluation-optimization/README.md)
- [Module 15](../../15-time-series-analysis/README.md) for streams
- [Module 13](../../13-model-deployment/README.md)–[14](../../14-mlops-basics/README.md) for production habits
- [Module 21](../../21-model-explainability/README.md) for reviews and audits

## Role emphasis (not destiny)

| Emphasis | Role | Why |
|----------|------|-----|
| Primary | ML Engineer | Production and reliability fit engineering culture |
| Alternate | Data Scientist | Strong when the work is still offline analysis |

See [Career Paths](../../README.md#career-paths) and [Career Roadmap Guide](../career_roadmap_guide.md). Treat times as emphasis maps only.

## Intensity maps

### Research support or R&D analysis (tier B)

1. [Module 00](../../00-prerequisites/README.md) if needed
2. [Module 01](../../01-python-for-data-science/README.md)
3. [Module 02](../../02-introduction-to-ml/README.md)
4. [Module 03](../../03-supervised-learning-regression/README.md) and/or [Module 04](../../04-supervised-learning-classification/README.md)
5. [Module 05](../../05-model-evaluation-optimization/README.md)
6. [Module 15](../../15-time-series-analysis/README.md) for sensor streams
7. [Module 21](../../21-model-explainability/README.md)

### Job-oriented study (tier C)

1. Complete the R&D list
2. Add [Module 07](../../07-feature-engineering/README.md) and [Module 19](../../19-sql-database-fundamentals/README.md)
3. Add [Module 09](../../09-neural-networks-basics/README.md)–[10](../../10-deep-learning-frameworks/README.md) only when needed
4. Add [Module 13](../../13-model-deployment/README.md)–[14](../../14-mlops-basics/README.md)
5. Optional [Module 11](../../11-computer-vision/README.md) for inspection imaging
6. One end-to-end project from [Module 17](../../17-projects-intermediate/README.md) or [Module 18](../../18-projects-advanced/README.md) you can operate and explain

## Ordered study checklist

- [ ] Pick R&D or Job-oriented map
- [ ] Finish Module 01 with a sensor CSV
- [ ] Finish Module 05 with a split by machine, run, or time
- [ ] Add Module 15 before deep sequence models
- [ ] If shipping, finish Modules 13 and 14 with a tiny service mindset
- [ ] Write a failure mode note (what happens when the model is wrong)

## Start here

Open [Module 00 README](../../00-prerequisites/README.md) if you need math or environment help. Otherwise open [Module 01 README](../../01-python-for-data-science/README.md).

## Honesty and traps

- Safety-critical systems need review beyond a tutorial notebook.
- “Predictive maintenance” marketing often skips evaluation and monitoring. Do not.
- Professional licensure and plant procedures stay outside this curriculum.

**Try next:** Open [Module 05: Model Evaluation and Optimization](../../05-model-evaluation-optimization/README.md) early. Engineering stakeholders will ask for it.
