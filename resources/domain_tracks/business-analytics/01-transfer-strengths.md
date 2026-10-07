# Transfer strengths: business analytics to ML

You already live in tables, KPIs, and stakeholder questions. This chapter maps that habit onto Modules `00`–`25`. It is not an MBA. It does not give investment advice. It does not guarantee revenue or a hire.

## What you already do well

| Habit you bring | Where it helps on the spine |
|-----------------|-----------------------------|
| Spreadsheets and pivots | [Module 01](../../../01-python-for-data-science/README.md) plus [Excel Data Analysis Guide](../../excel_data_analysis_guide.md) |
| Dashboard storytelling | [Stakeholder Communication](../../stakeholder_communication.md) and [Module 21](../../../21-model-explainability/README.md) |
| KPI definitions and cohorts | [Module 07](../../../07-feature-engineering/README.md) and careful splits in [Module 05](../../../05-model-evaluation-optimization/README.md) |
| Warehouse filters and joins | [Module 19](../../../19-sql-database-fundamentals/README.md) and [Enterprise Data Tools](../../enterprise_data_tools.md) |
| Seasonality gut feel | [Module 15](../../../15-time-series-analysis/README.md) when the target is ordered in time |

## Excel to Python glue (practical bridge)

Keep Excel for exploration with stakeholders if that is your shared language. Move modeling work into Python so splits and metrics stay reproducible.

1. Export a flat CSV or query result with one row per decision unit (order, customer-day, ticket).
2. Rebuild the same filters in pandas ([Module 01](../../../01-python-for-data-science/README.md)).
3. Write the label definition in plain words. Match the KPI doc, not a vague “success” column.
4. Train a simple baseline in [Module 02](../../../02-introduction-to-ml/README.md)–[04](../../../04-supervised-learning-classification/README.md).
5. Report errors the way a manager already thinks (missed churners, over-forecast units) via [Stakeholder Communication](../../stakeholder_communication.md).

Power BI or similar reporting can stay for distribution. It is not a substitute for Module 05 evaluation. See [Power BI Guide](../../powerbi_guide.md) only as a reporting bridge.

## Module map for common business tasks

| Business question shape | Start here |
|-------------------------|------------|
| Will this row convert / churn / escalate? | [Module 04](../../../04-supervised-learning-classification/README.md) then [Module 05](../../../05-model-evaluation-optimization/README.md) |
| How many units next week? | [Module 15](../../../15-time-series-analysis/README.md) after Module 05 habits |
| Which drivers explain a wrong call? | [Module 21](../../../21-model-explainability/README.md) |
| Can I join and filter without guessing? | [Module 19](../../../19-sql-database-fundamentals/README.md) |

## Worked sketch

Marketing wants a “likely to buy” score. You already have a CRM export and a dashboard of past campaigns.

1. Define the label as purchase within N days after a score date.
2. Build features known on that score date only.
3. Split by time so future campaigns do not train the past ([Module 05](../../../05-model-evaluation-optimization/README.md)).
4. Compare to a simple rules baseline (recent buyers, high spenders).
5. Explain two false positives to a stakeholder without promising lift.

## Honesty

- A polished deck is not an evaluation.
- “AI for growth” slides are not a methods path.
- Finishing modules is practice. It is not a hire or revenue promise.

**Try next:** Open [Module 01](../../../01-python-for-data-science/README.md) with one real business-style CSV, then [Module 19](../../../19-sql-database-fundamentals/README.md) when joins get messy.
