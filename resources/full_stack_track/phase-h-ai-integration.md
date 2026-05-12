# Phase H — AI Integration for Full-Stack Products (In-Repo Lessons)

**Maps to:** [Blueprint Phase H](../full_stack_ai_engineer_roadmap.md#phase-h-ai-integration-for-full-stack-products)

**Goal:** Ship LLM features that are observable, cost-aware, and grounded when the product requires factual reliability.

---

## Table of Contents

- [Lesson 1 — Where models live in your architecture](#lesson-1-where-models-live-in-your-architecture)
- [Lesson 2 — Prompting and structured outputs](#lesson-2-prompting-and-structured-outputs)
- [Lesson 3 — Streaming and latency UX](#lesson-3-streaming-and-latency-ux)
- [Lesson 4 — RAG and evaluation](#lesson-4-rag-and-evaluation)
- [Lesson 5 — Safety, abuse, and cost controls](#lesson-5-safety-abuse-and-cost-controls)
- [Pair with ML modules in this repo](#pair-with-ml-modules-in-this-repo)
- [Exercises](#exercises)

---

## Lesson 1 — Where models live in your architecture

Avoid calling provider APIs directly from every UI component. Prefer:

- a small **AI service** module in your backend
- centralized **logging**, **timeouts**, and **retries**
- rate limits per user/org

---

## Lesson 2 — Prompting and structured outputs

Define stable system prompts for behavior and user prompts for task inputs.

When you need machine-readable results, use **JSON schema** or tool calling patterns supported by your provider instead of fragile “please return JSON” strings.

---

## Lesson 3 — Streaming and latency UX

Stream tokens when long answers are expected; show partial UI states and cancellation.

Always handle provider errors: show a fallback message and log correlation IDs.

---

## Lesson 4 — RAG and evaluation

RAG is not “embed everything and hope.” You need:

- chunking strategy
- retrieval metrics (even simple offline sets help)
- user-visible citations when factual grounding matters

Deep dive in this repo: [RAG Comprehensive Guide](../rag_comprehensive_guide.md).

---

## Lesson 5 — Safety, abuse, and cost controls

Threat model prompt injection for any tool-enabled agent. Cap token usage, throttle requests, and monitor anomalous spikes.

---

## Pair with ML modules in this repo

- **25-generative-ai-llms** for LLM fundamentals and app-facing patterns
- **19-sql-database-fundamentals** when retrieval uses Postgres
- **13-14** for deployment and operational discipline

---

## Exercises

1. Add server-side logging for each LLM call: user id, model, latency, token counts (if available).
2. Build a minimal RAG endpoint and return citations with each answer.
3. Write a rollback plan: what happens if the provider is down?
