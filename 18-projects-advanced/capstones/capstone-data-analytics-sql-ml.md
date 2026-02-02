# Capstone: Data/Analytics — SQL → Metrics → Dashboard → ML (GDPR-Safe)

Build a **realistic analytics workflow** (tables, SQL, metrics, dashboard) and then add a production-style ML model (e.g., churn or purchase propensity).

This blueprint is designed to be safe for a **public repo** by using **synthetic event data**.

---

## Target role

- Data Analyst → Analytics Engineer → Data Scientist (product analytics)
- Great for roles that require strong SQL + practical modeling

---

## Deliverables

- A clear `README.md` with:
  - business questions + metric definitions
  - SQL queries + results interpretation
  - dashboard screenshots (optional)
  - ML model summary and limitations
- A synthetic dataset generator (so no PII)
- SQL schema + example queries (joins, window functions, funnels, retention)
- A small ML model predicting a business outcome

---

## Data model (example)

Use a standard product analytics schema:

- `users(user_id, signup_date, country, channel)` (synthetic)
- `events(event_id, user_id, ts, event_type, value)` (synthetic)
- `orders(order_id, user_id, ts, amount)` (synthetic)

> `user_id` should be a random identifier (not derived from real data).

---

## Synthetic data generator (example)

```python
import random
from datetime import datetime, timedelta

def gen_users(n=5000, seed=42):
    random.seed(seed)
    countries = ["DE", "FR", "NL", "ES", "US"]
    channels = ["seo", "ads", "referral", "social", "direct"]
    start = datetime(2024, 1, 1)
    users = []
    for i in range(n):
        signup = start + timedelta(days=random.randint(0, 120))
        users.append(
            {
                "user_id": f"u_{i:06d}",
                "signup_date": signup.date().isoformat(),
                "country": random.choice(countries),
                "channel": random.choice(channels),
            }
        )
    return users
```

---

## Milestones

### Milestone 1: SQL fundamentals in practice

Write queries for:

- DAU/WAU/MAU
- Retention (cohorts)
- Funnel conversion (view → add_to_cart → purchase)
- Revenue metrics (AOV, LTV proxy)
- Window functions (rolling 7-day revenue)

### Milestone 2: “Analytics engineering” style transformations

Create “clean” derived tables:

- `fact_orders`
- `fact_events`
- `dim_users`
- `user_daily_metrics`

Document metric definitions in your README so interviewers can follow your thinking.

### Milestone 3: Dashboard (optional but portfolio-strong)

- Streamlit or a simple notebook chart pack
- Focus on clarity: KPIs + trend charts + segmentation filters

### Milestone 4: ML model

Predict one clear target:

- churn risk (no activity in next 14 days)
- purchase propensity (purchase in next 7 days)

Features (examples):

- activity counts last 7/30 days
- recency/frequency/monetary (RFM)
- funnel steps completed
- country/channel segments

Metrics:

- classification: PR-AUC, ROC-AUC, calibration, confusion matrix at chosen threshold
- explainability: feature importance / SHAP summary (optional)

---

## GDPR-safe checklist

- Synthetic identifiers only
- No PII fields
- Do not commit generated datasets (repo `.gitignore` already ignores `*.csv`, `data/`)
- If you add dashboard screenshots, ensure they contain only synthetic values

---

## Interview talking points

- How you defined business metrics
- Why you chose the prediction target + time horizon
- How you avoided leakage (predicting future with future data)
- How you’d productionize (batch scoring, scheduling, monitoring)

---

## Useful repo references

- `19-sql-database-fundamentals/` (joins, window functions, schema thinking)
- `01-python-for-data-science/` (EDA + dashboards)
- `05-model-evaluation-optimization/` (metrics + validation)
- `13-model-deployment/` (serving patterns)

