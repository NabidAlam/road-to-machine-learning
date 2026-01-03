# Causal Inference Guide

Comprehensive guide to causal inference in machine learning and data science.

## Table of Contents

- [Introduction to Causal Inference](#introduction-to-causal-inference)
- [Key Concepts](#key-concepts)
- [Causal Inference Methods](#causal-inference-methods)
- [Applications](#applications)
- [Tools and Libraries](#tools-and-libraries)
- [Resources](#resources)

---

## Introduction to Causal Inference

### What is Causal Inference?

**Causal Inference** is the process of determining whether a cause-effect relationship exists between variables, beyond mere correlation.

**Key Difference:**
- **Correlation**: Variables change together (association)
- **Causation**: One variable causes change in another (causal effect)

### Why Causal Inference Matters

**Traditional ML** focuses on prediction: "What will happen?"
**Causal Inference** focuses on understanding: "What would happen if we intervene?"

**Applications:**
- **Healthcare**: Does a treatment cause better outcomes?
- **Economics**: Does a policy cause economic change?
- **Marketing**: Does an ad campaign cause increased sales?
- **A/B Testing**: Does a change cause improved metrics?

---

## Key Concepts

### Potential Outcomes Framework

**Potential Outcomes** framework defines causal effects:

- **Y(1)**: Outcome if treated
- **Y(0)**: Outcome if not treated
- **Causal Effect**: Y(1) - Y(0)

**Fundamental Problem**: We can only observe one potential outcome per unit.

### Confounding

**Confounding** occurs when a third variable affects both treatment and outcome.

**Example:**
- Treatment: Exercise
- Outcome: Health
- Confounder: Income (affects both exercise and health)

### Randomized Controlled Trials (RCTs)

**RCTs** randomly assign treatment to eliminate confounding.

**Gold Standard** for causal inference, but often impractical.

---

## Causal Inference Methods

### 1. Randomized Experiments

**Randomized Controlled Trials (RCTs)**:
- Randomly assign treatment
- Eliminates confounding
- Gold standard for causal inference

### 2. Observational Studies

When RCTs are not possible, use observational methods:

#### A. Propensity Score Matching

**Propensity Score**: Probability of receiving treatment given covariates.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

def propensity_score_matching(treated, control, covariates):
    # Estimate propensity scores
    X = np.vstack([treated[covariates], control[covariates]])
    y = np.hstack([np.ones(len(treated)), np.zeros(len(control))])
    
    ps_model = LogisticRegression()
    ps_model.fit(X, y)
    
    treated_ps = ps_model.predict_proba(treated[covariates])[:, 1]
    control_ps = ps_model.predict_proba(control[covariates])[:, 1]
    
    # Match on propensity scores
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control_ps.reshape(-1, 1))
    distances, indices = nn.kneighbors(treated_ps.reshape(-1, 1))
    
    # Calculate treatment effect
    matched_control = control.iloc[indices.flatten()]
    treatment_effect = treated['outcome'].mean() - matched_control['outcome'].mean()
    
    return treatment_effect
```

#### B. Difference-in-Differences (DiD)

**DiD** compares changes over time between treatment and control groups.

**Formula:**
```
DiD = (Y_treated_after - Y_treated_before) - (Y_control_after - Y_control_before)
```

#### C. Instrumental Variables (IV)

**Instrumental Variables** use a variable that affects treatment but not outcome directly.

**Requirements:**
1. Relevance: IV affects treatment
2. Exclusion: IV affects outcome only through treatment
3. Independence: IV is independent of confounders

#### D. Regression Discontinuity Design (RDD)

**RDD** exploits a threshold that determines treatment assignment.

**Example**: Students above a test score threshold receive a scholarship.

### 3. Causal Machine Learning

#### A. Causal Trees

**Causal Trees** partition data to estimate heterogeneous treatment effects.

#### B. Meta-Learners

**Meta-Learners** use ML models to estimate treatment effects:

- **T-Learner**: Two separate models for treated/control
- **S-Learner**: Single model with treatment as feature
- **X-Learner**: Combines T-learner and S-learner
- **R-Learner**: Uses cross-fitting for robustness

```python
from sklearn.ensemble import RandomForestRegressor

class TLearner:
    def __init__(self):
        self.model_treated = RandomForestRegressor()
        self.model_control = RandomForestRegressor()
    
    def fit(self, X, treatment, outcome):
        treated_mask = treatment == 1
        self.model_treated.fit(X[treated_mask], outcome[treated_mask])
        self.model_control.fit(X[~treated_mask], outcome[~treated_mask])
    
    def predict_effect(self, X):
        return (self.model_treated.predict(X) - 
                self.model_control.predict(X))
```

#### C. Double Machine Learning

**Double Machine Learning** uses cross-fitting to reduce bias.

---

## Applications

### 1. Healthcare

**Question**: Does a new drug improve patient outcomes?

**Method**: Randomized controlled trial or propensity score matching

### 2. Economics

**Question**: Does minimum wage increase cause unemployment?

**Method**: Difference-in-differences, instrumental variables

### 3. Marketing

**Question**: Does an ad campaign cause increased sales?

**Method**: A/B testing, causal ML

### 4. Policy Evaluation

**Question**: Does a policy intervention cause desired outcomes?

**Method**: Regression discontinuity, difference-in-differences

---

## Tools and Libraries

### Python Libraries

#### 1. DoWhy

**DoWhy** provides a unified interface for causal inference.

```python
from dowhy import CausalModel

# Define causal model
model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    common_causes=['confounder1', 'confounder2']
)

# Identify causal effect
identified_estimand = model.identify_effect()

# Estimate effect
causal_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)
```

#### 2. EconML

**EconML** provides ML methods for causal inference.

```python
from econml.dml import DML

# Double Machine Learning
dml = DML(model_y=RandomForestRegressor(),
          model_t=RandomForestClassifier(),
          model_final=LinearRegression())

dml.fit(Y, T, X=X)
treatment_effect = dml.effect(X)
```

#### 3. CausalML

**CausalML** provides ML-based causal inference methods.

```python
from causalml.inference.meta import LRSRegressor

# Meta-learner
learner = LRSRegressor()
learner.fit(X, treatment, y)
treatment_effect = learner.predict(X)
```

### R Packages

- **causalTree**: Causal trees
- **grf**: Generalized random forests for causal inference
- **ivreg**: Instrumental variables regression

---

## Resources

### Books

1. **"Causal Inference: The Mixtape"** - Scott Cunningham
2. **"Mostly Harmless Econometrics"** - Angrist & Pischke
3. **"Causal Inference in Statistics"** - Pearl, Glymour, Jewell

### Courses

1. **"Causal Inference"** - Harvard (edX)
2. **"Causal Inference"** - MIT OpenCourseWare

### Papers

1. **"The Book of Why"** - Judea Pearl
2. **"Double Machine Learning"** - Chernozhukov et al.

---

## Key Takeaways

1. **Causation ≠ Correlation**: Need methods to identify causal effects
2. **Confounding**: Major challenge in observational studies
3. **RCTs**: Gold standard but often impractical
4. **Observational Methods**: Propensity scores, DiD, IV, RDD
5. **Causal ML**: Modern methods using machine learning
6. **Tools**: DoWhy, EconML, CausalML for implementation

---

**Next Steps**: Apply causal inference methods to your domain of interest, starting with understanding your causal question and available data.

