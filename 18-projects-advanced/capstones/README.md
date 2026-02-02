# Capstone Blueprints (Industry-Ready, GDPR-Safe)

These capstones are **portfolio-grade** project blueprints designed to make you **company job-ready** (ML Engineer / LLM Engineer / Data & Analytics). They are written to be **safe for a public GitHub repo**:

- Use **public datasets** (download yourself) or **synthetic data**
- Do **not** commit datasets, model artifacts, logs, or credentials (this repo’s `.gitignore` already enforces that)
- Avoid any personally identifiable information (PII) or private company documents

> Note: This is educational content, not legal advice. See `DISCLAIMER.md` for data/privacy context.

---

## Capstones

### 1) ML Engineer Capstone: Real-Time Risk Scoring System
- Blueprint: [capstone-ml-engineer.md](capstone-ml-engineer.md)
- Focus: end-to-end pipeline, API serving, monitoring, retraining, cost/reliability thinking

### 2) LLM Engineer Capstone: RAG Knowledge Assistant (with evaluation + guardrails)
- Blueprint: [capstone-llm-rag-engineer.md](capstone-llm-rag-engineer.md)
- Focus: ingestion → embeddings → retrieval → generation, eval harness, prompt-injection defenses

### 3) Data/Analytics Capstone: SQL → Metrics → Dashboard → ML
- Blueprint: [capstone-data-analytics-sql-ml.md](capstone-data-analytics-sql-ml.md)
- Focus: analytics engineering, SQL case studies, business metrics, and a productionized ML model

---

## Public Repo GDPR-Safe Checklist (quick)

- **No PII**: no names, emails, addresses, phone numbers, IPs, exact GPS, device identifiers
- **No secrets**: `.env` stays local; never commit API keys/tokens
- **No datasets committed**: only provide download links or synthetic generators
- **No raw logs**: logs can contain identifiers; keep them local
- **Document tradeoffs**: explain privacy decisions and data minimization in your README

