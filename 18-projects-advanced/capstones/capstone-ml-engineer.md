# Capstone: ML Engineer — Real-Time Risk Scoring System (GDPR-Safe)

Build a **production-style ML system** that scores events in real time (e.g., fraud/risk/churn propensity) and includes: data pipeline, training, evaluation, API serving, monitoring, and retraining triggers.

This blueprint is written to be **safe for a public GitHub repo** (no PII, no secrets, no datasets committed).

---

## Why this is “industry-ready”

Companies don’t hire for “a model”, they hire for:

- translating a business problem into measurable metrics
- building reliable pipelines (data → features → model → service)
- monitoring and iterative improvement
- basic security/cost/reliability reasoning

---

## Target role

- ML Engineer / Applied ML Engineer / Data Scientist (production-leaning)

---

## Deliverables (what your public repo should contain)

- `README.md` with requirements, decisions, metrics, and screenshots
- Training + evaluation code with reproducible runs (seeded)
- Model card (what it does / limitations / fairness considerations)
- A small API service (FastAPI) for scoring
- Basic monitoring plan (what to log, what alerts you’d set)
- CI checks (lint + unit tests)

---

## GDPR-safe data options (choose one)

### Option A (recommended): Synthetic event dataset

Generate a dataset that mimics “transactions/events” without any real user data.

Example generator (small):

```python
import random

def make_synthetic_events(n=20000, seed=42):
    random.seed(seed)
    events = []
    for _ in range(n):
        amount = round(random.expovariate(1/50), 2)
        hour = random.randint(0, 23)
        device_risk = random.choice([0, 1, 2])
        velocity_5m = random.randint(0, 10)
        # Simple rule for label (you’ll improve it later)
        risk = (amount > 200) + (device_risk == 2) + (velocity_5m >= 6) + (hour in [0,1,2,3])
        label = 1 if risk >= 2 else 0
        events.append(
            {
                "amount": amount,
                "hour": hour,
                "device_risk": device_risk,
                "velocity_5m": velocity_5m,
                "label": label,
            }
        )
    return events
```

### Option B: Public anonymized dataset (download yourself)

Use a dataset where features are already anonymized. **Do not commit it**; only link to it in your README.

---

## System architecture (minimal)

```
            (batch)                         (online)
Raw events --------> Feature pipeline -----> Scoring API (FastAPI)
   |                      |                     |
   |                      v                     v
   |                Train/Eval + Registry    Logs/Metrics
   |                      |                     |
   |                      v                     v
   +---------------> Model artifact (local)  Monitoring/Alerts (plan)
```

---

## Milestones (step-by-step)

### Milestone 1: Problem framing

- Define target: “flag risky events”
- Define metrics:
  - **Primary**: PR-AUC (good for imbalance), Recall@Precision threshold
  - **Secondary**: latency p95 (e.g., < 100ms), false-positive cost estimate

### Milestone 2: Baseline model

- Simple baseline: logistic regression or lightgbm/xgboost (if available)
- Train/val/test split (time-based if events are time-like)
- Save evaluation report (markdown)

### Milestone 3: Feature pipeline (reproducible)

- Create `src/features.py` to:
  - validate schema
  - impute/encode
  - compute simple features (velocity buckets, time-of-day)

### Milestone 4: API serving

- Build `FastAPI` endpoint:
  - `POST /score` → returns risk score + decision + model version
- Add input validation (pydantic)

### Milestone 5: Monitoring plan (what you would do in production)

- Log (structured):
  - request id (random), model version, score, decision, latency
- Track drift signals:
  - feature means/quantiles, missing rates
- Track model quality:
  - delayed label evaluation (if labels arrive later)

### Milestone 6: CI + tests

- Unit tests:
  - feature engineering invariants
  - API input validation
- Basic CI workflow (lint + tests)

---

## Public repo GDPR checklist (apply before publishing)

- No PII fields in synthetic/public data
- No datasets committed (`.gitignore` already ignores `*.csv`, `data/`, etc.)
- No secrets committed (`.env` is ignored)
- If you log sample requests, ensure they contain **only synthetic** inputs

---

## What to say in interviews (talking points)

- Why you chose PR-AUC / recall tradeoffs
- How you’d prevent training-serving skew
- How you’d do safe rollout (shadow, canary, A/B)
- What you’d alert on (drift + latency + error rates)
- What you’d do if false positives are too high (thresholding, calibration, cost-sensitive)

---

## Recommended references inside this repo

- `13-model-deployment/` (APIs, Docker, deployment)
- `14-mlops-basics/` (tracking, CI/CD, monitoring concepts)
- `resources/ml_model_testing.md` (testing mindset)
- `resources/ml_system_design_guide.md` (architecture tradeoffs)

