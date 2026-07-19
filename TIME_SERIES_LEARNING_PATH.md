# Time Series: Pick One Path

Time series appears in **three places** in this curriculum. You do **not** need all three unless you are specializing in forecasting.

| Path | Location | Best for | Depth |
|------|----------|----------|-------|
| **A. Module** | [15-time-series-analysis](15-time-series-analysis/README.md) | Theory + methods (ARIMA, Prophet, LSTM) | Full module + [10 exercises](15-time-series-analysis/exercises/README.md) |
| **B. Project** | [Intermediate project 6](17-projects-intermediate/project-06-time-series-forecasting/README.md) | Portfolio piece after Stages 3–4 | Applied brief + `starter.py` |
| **C. Advanced** | [Advanced project 3](18-projects-advanced/project-03-time-series-forecasting/README.md) | End-to-end pipeline + deployment narrative | Capstone-style brief |

## Recommended choice

| Your goal | Pick |
|-----------|------|
| Data scientist / analyst needing forecasting | **Path A** (module), then optionally Path B for portfolio |
| Already know ARIMA basics; want a GitHub project fast | **Path B** only |
| ML engineer building production forecasting | **Path A** → **Path C** |
| Not doing forecasting roles | **Skip all three**, use Stage 6 time on CV/NLP or tabular projects |

## Prerequisites (all paths)

- Module 01: Pandas, dates, basic plotting
- Module 05: train/test discipline (no random split on time!)
- Path A/C deep learning sections: Modules 09–10

## Anti-patterns

- Doing module 15, intermediate project 6, **and** advanced project 3 back-to-back without new datasets or methods
- Random `train_test_split` on time-ordered data (see module 15 evaluation section)
- Starting module 15 before you can load and resample a datetime index in Pandas

## Related content elsewhere

- Date/time wrangling: [01 → Working with Dates & Times](01-python-for-data-science/08-working-with-dates-times.md)
- TS features in tabular ML: [07-feature-engineering](07-feature-engineering/feature-engineering-advanced-topics.md#time-series-feature-engineering)
- TensorFlow TS intro: [10-deep-learning-frameworks](10-deep-learning-frameworks/deep-learning-frameworks.md)

---

**Back to:** [START-HERE.md](START-HERE.md) · [LEARNING_ROADMAP.md](LEARNING_ROADMAP.md)
