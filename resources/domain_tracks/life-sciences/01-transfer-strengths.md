# Transfer strengths: life sciences to ML

You already think in experiments, batches, and noisy biological variation. This chapter maps that habit onto Modules `00`–`25` for research-style analysis. It is educational content only. It is not clinical advice. It does not diagnose or treat. It does not guarantee a hire.

## What you already do well

| Habit you bring | Where it helps on the spine |
|-----------------|-----------------------------|
| Controls, replicates, protocols | Cleaner labels and honest baselines in [Module 02](../../../02-introduction-to-ml/README.md)–[04](../../../04-supervised-learning-classification/README.md) |
| Batch and site awareness | Split design in [Module 05](../../../05-model-evaluation-optimization/README.md) |
| Rare events and skewed assays | [Module 20](../../../20-handling-imbalanced-data/README.md) and [Imbalanced Data Cheatsheet](../../imbalanced_data_cheatsheet.md) |
| Expensive or imperfect labels | Careful metrics and error review, not accuracy theater |
| Explaining results to lab collaborators | [Module 21](../../../21-model-explainability/README.md) |

## Module map for common research tables

| Research question shape | Start here |
|-------------------------|------------|
| Classify samples or states from features | [Module 04](../../../04-supervised-learning-classification/README.md) then [Module 05](../../../05-model-evaluation-optimization/README.md) |
| Rare positive class | [Module 20](../../../20-handling-imbalanced-data/README.md) |
| Continuous endpoint | [Module 03](../../../03-supervised-learning-regression/README.md) after Module 05 habits |
| Which features drove a call? | [Module 21](../../../21-model-explainability/README.md) |
| Prediction is not enough. You need “what changed?” | [Causal Inference Guide](../../causal_inference_guide.md) after Module 05 |

Optional later: [Module 11](../../../11-computer-vision/README.md) for image assays only after Module 05 is solid. [Module 12](../../../12-natural-language-processing/README.md) only for text-heavy literature tasks.

## Worked sketch

You have an assay-style table. Rows are samples. Columns are measured features. A binary research label marks a state of interest.

1. Wrangle and plot in [Module 01](../../../01-python-for-data-science/README.md). Check missingness by batch ([Data Validation](../../data_validation.md)).
2. Write a split by batch or sample id so the same batch does not flood train and test ([Module 05](../../../05-model-evaluation-optimization/README.md)).
3. Train a simple classifier ([Module 04](../../../04-supervised-learning-classification/README.md)).
4. If positives are rare, switch metrics and tactics ([Module 20](../../../20-handling-imbalanced-data/README.md)).
5. Explain one false positive to a biology collaborator ([Module 21](../../../21-model-explainability/README.md)).
6. Write what the model must never claim in a clinical setting ([Ethics in ML](../../ethics_in_ml.md)).

## What transfers. What does not.

- Experimental humility transfers. Random optimistic splits do not.
- Protocol discipline transfers. Publishing patient-identifying data does not belong in public repos.
- Domain labels transfer as careful targets. They are not a license to give medical advice.

## Honesty

- Educational content only. Not medical advice.
- Strong classical baselines beat unearned deep learning on many tables.
- Finishing modules is practice evidence. It is not a hire promise.

**Try next:** Open [Module 01](../../../01-python-for-data-science/README.md) with one assay-style table, then [Module 04](../../../04-supervised-learning-classification/README.md).
