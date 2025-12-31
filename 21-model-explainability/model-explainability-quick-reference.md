# Model Explainability Quick Reference

Quick reference for model explainability.

## SHAP

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## LIME

```python
from lime import lime_tabular
explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names)
explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
```

---

**Remember**: Explainability builds trust!

