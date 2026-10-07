# Transfer strengths: CS and software to ML

You already ship code. This chapter maps that habit onto Modules `00`–`25` so you spend time where software people usually under-invest. It is not a shortcut past evaluation. It does not guarantee a hire.

## What you already do well

| Habit you bring | Where it helps on the spine |
|-----------------|-----------------------------|
| Writing and reviewing code | [Module 01](../../../01-python-for-data-science/README.md) feels familiar. Still practice pandas-style tables, not only scripts. |
| Debugging and tests | [Module 05](../../../05-model-evaluation-optimization/README.md) and [Module 14](../../../14-mlops-basics/README.md). Treat metrics like failing tests. |
| APIs, stores, failure modes | [Module 13](../../../13-model-deployment/README.md), [System Design](../../../system-design/README.md), [ML System Design Guide](../../ml_system_design_guide.md) |
| Git and review | [Git Guide](../../git_guide.md) plus portfolio READMEs with clear metrics |
| Complexity and data structures | [DSA for ML Guide](../../dsa_for_ml_guide.md) when interview or pipeline shape matters |

## Fast path toward ML Engineer habits

If your target emphasis is ML Engineer or AI Engineer, prefer this order after you can wrangle a table:

1. [Modules 02](../../../02-introduction-to-ml/README.md)–[05](../../../05-model-evaluation-optimization/README.md). Do not skip Module 05.
2. Role modules from the main [Career Paths](../../../README.md#career-paths) table for ML Engineer.
3. [Module 13](../../../13-model-deployment/README.md)–[14](../../../14-mlops-basics/README.md).
4. [System Design](../../../system-design/README.md) and the [ML System Design Guide](../../ml_system_design_guide.md).
5. One project from [Modules 16](../../../16-projects-beginner/README.md)–[18](../../../18-projects-advanced/README.md) with offline metrics and a failure note.

Optional later: GenAI, CV, or NLP modules when the role needs them. Optional product surfaces: [Full-Stack AI Track](../../full_stack_track/README.md).

## What you can skip early (and why)

Business-analytics overlays lean on Excel and dashboard bridges first. You do not need that as your default entry.

- Skip treating [Excel Data Analysis Guide](../../excel_data_analysis_guide.md) or [Power BI Guide](../../powerbi_guide.md) as your primary path unless your job is reporting-first.
- Do not stall on Tableau-style BI catalogs before Module 05. Your bottleneck is usually statistical humility and eval design, not chart tools.
- Do not jump to wrapping a model API before you can explain a held-out score and a split rule.

## Worked sketch

You have a recommendation or ranking service idea. You can already sketch the API and the store. Pause before coding the wrapper.

1. Define the prediction task and the label source in Module 02 language.
2. Write a split rule that does not leak user history across train and test ([Module 05](../../../05-model-evaluation-optimization/README.md)).
3. Ship a boring baseline. Log offline metrics.
4. Only then design the service boundary with system design notes.

That sequence is the transfer. Code fluency gets you to the glue faster. It does not replace the eval loop.

## Honesty

- Software skill does not replace careful splits.
- A green CI build is not an ML quality gate.
- Finishing modules is practice evidence. It is not a hire promise.

**Try next:** Open [Module 05: Model Evaluation and Optimization](../../../05-model-evaluation-optimization/README.md) if Modules 02–04 are already comfortable. Otherwise start at [Module 02](../../../02-introduction-to-ml/README.md).
