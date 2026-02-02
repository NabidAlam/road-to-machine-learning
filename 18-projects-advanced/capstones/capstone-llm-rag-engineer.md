# Capstone: LLM Engineer — RAG Knowledge Assistant (GDPR-Safe)

Build a **production-style RAG (Retrieval-Augmented Generation)** system with evaluation, guardrails, and cost awareness.

This blueprint is designed to be **safe for a public GitHub repo**:

- Use **public documents** (or the repo’s own markdown) as your corpus
- Never commit private company docs, PII, or API keys

---

## Target role

- LLM Engineer / GenAI Engineer / AI Engineer

---

## Deliverables (what your public repo should contain)

- A clear `README.md` explaining:
  - what the assistant does, its limits, and example queries
  - evaluation approach and results
  - security/abuse considerations (prompt injection)
- Ingestion pipeline (chunking + metadata)
- Retrieval + generation pipeline
- Evaluation harness (ground truth + automatic metrics)
- Basic API or Streamlit UI demo

---

## GDPR-safe corpus options

### Option A (recommended): Public docs

- Python docs pages, FastAPI docs, scikit-learn docs, OWASP cheat sheets, etc.

### Option B: Use this repo as the corpus

Point ingestion at `road-to-ml/` markdown files. This is **safe** because it’s already public and contains no personal data.

---

## Architecture (book-style)

```
Documents -> Parse -> Chunk -> Embed -> Vector Index
                                 |
User Query -> (optional rewrite) -> Retrieve Top-K -> Prompt -> LLM -> Answer + Citations
                                                   |
                                               Safety checks
```

---

## Milestones

### Milestone 1: Baseline RAG

- Chunking strategy (start simple):
  - chunk size: 500–1000 chars
  - overlap: 100–200 chars
- Store metadata:
  - source path/url, title, section heading

### Milestone 2: Evaluation (mandatory for “industry-ready”)

Create a small eval set:

- 30–80 Q/A pairs that can be answered from your docs
- Include “hard” questions that require multi-chunk retrieval

Track:

- **Retrieval**: hit rate (does top-k contain the supporting chunk?)
- **Answer quality**: faithfulness / citation coverage
- **Latency + cost**: tokens per query, time per query

### Milestone 3: Guardrails (practical security)

Implement:

- prompt injection resilience:
  - system message with strict rules
  - only answer using retrieved context
  - refuse if context is insufficient
- input filtering:
  - disallow requests for private data
  - rate limit (basic)

### Milestone 4: Production concerns (what interviewers care about)

- Caching:
  - cache embeddings for unchanged docs
  - cache answers for repeated queries (optional)
- Index updates:
  - incremental indexing for changed docs
- Observability plan:
  - retrieval top-k sources, latency, tokens, refusal rates

---

## Public repo GDPR checklist

- Never ingest:
  - HR docs, customer tickets, user chats, emails, Slack exports
  - anything with names/emails/phone numbers
- Keep keys out of git:
  - use `.env` locally (already ignored)
- Don’t upload your vector DB files if they contain private content
- If you publish sample logs, ensure they contain **only non-sensitive** prompts

---

## Interview talking points

- How you chose chunking + top-k
- How you measured retrieval quality
- How you reduced hallucinations (grounding + refusals)
- How you handled prompt injection
- Cost controls (token budgets, caching, smaller models)

---

## Useful repo references

- `25-generative-ai-llms/` (RAG, agents, evaluation)
- `resources/rag_comprehensive_guide.md`
- `resources/ai_agents_guide.md`
- `13-model-deployment/` (serving patterns)

