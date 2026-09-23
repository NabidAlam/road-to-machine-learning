# Phase H: AI Integration for Full-Stack Products (In-Repo Lessons)

**Maps to:** [Blueprint Phase H](../full_stack_ai_engineer_roadmap.md#phase-h-ai-integration-for-full-stack-products)

**Goal:** Ship LLM features that are observable, cost-aware, and grounded when the product requires factual reliability. This chapter teaches **product wiring**. Deep theory and recipes live in Module **25** and the RAG / agents / LangChain guides.

**Next chapter:** [Phase H2: FastAPI AI service](phase-h2-fastapi-ai-service.md) for a Python service boundary, queues, and Node/Next integration.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Lesson 1: Where models live in your architecture](#lesson-1-where-models-live-in-your-architecture)
- [Lesson 2: Prompting and structured outputs](#lesson-2-prompting-and-structured-outputs)
- [Lesson 3: Streaming and latency UX](#lesson-3-streaming-and-latency-ux)
- [Lesson 4: RAG with citations and evaluation](#lesson-4-rag-with-citations-and-evaluation)
- [Lesson 5: Agents, tools, and human-in-the-loop](#lesson-5-agents-tools-and-human-in-the-loop)
- [Lesson 6: Safety, abuse, and cost controls](#lesson-6-safety-abuse-and-cost-controls)
- [Pair with ML modules and guides in this repo](#pair-with-ml-modules-and-guides-in-this-repo)
- [Self-check exercises](#self-check-exercises)
- [Official docs (keep current)](#official-docs-keep-current)
- [Next step](#next-step)

---

## Prerequisites

- Phases **A–G** (or equivalent TypeScript, API, Postgres, Next.js, and delivery practice).
- Comfort calling HTTP APIs and reading JSON.
- Optional but useful: skim [25-generative-ai-llms](../../25-generative-ai-llms/README.md) so tokens, embeddings, and RAG vocabulary are not brand new.

You do **not** need to train models for this phase. You integrate providers and retrieval into a product you control.

---

## Lesson 1: Where models live in your architecture

Avoid calling provider APIs directly from every UI component. Prefer a clear boundary:

1. **Product API** (Node/Express or Nest-style modules you already built in Phases B–E). Owns auth, tenancy, rate limits, and audit logs.
2. **AI capability layer** (same process at first, or a dedicated service in Phase H2). Owns prompts, tools, retrieval, and provider SDKs.
3. **Provider** (OpenAI, Anthropic, or a local model gateway). Owns model weights and inference.

Why this split matters:

- One place for **timeouts**, **retries**, and **fallback** when a provider fails.
- One place to attach **correlation IDs** so support can debug a bad answer.
- Easier to swap models without rewriting every React page.

Minimal request shape to log on every call:

- `userId` / `orgId` (or anonymous session id)
- `feature` name (for example `doc_qa`, `support_draft`)
- `model` id
- latency ms
- token counts when the API returns them
- success / error class (timeout, rate limit, content filter, unknown)

```text
Browser (Next.js)
    |
    v
Product API (auth, RBAC, quotas)
    |
    v
AI module or AI service
    |-- embeddings / vector store
    |-- LLM provider
    '-- tools (search, DB, tickets)
```

---

## Lesson 2: Prompting and structured outputs

Separate **system** instructions (stable behavior) from **user** content (task input). Do not paste secrets into prompts.

When the product needs machine-readable results, prefer provider-supported **structured outputs** or **tool/function calling** over “please reply in JSON.” Validate the payload with a schema library you already use (for example Zod on Node) before writing to the database.

Practical rules:

- Keep system prompts versioned in code or config. Review changes like any other production config.
- Bound input size. Truncate or summarize oversized documents before the model call.
- Treat model text as **untrusted** until validated. Especially when you will execute tools or write SQL.

For prompt patterns and failure modes, see [25-generative-ai-llms](../../25-generative-ai-llms/generative-ai-llms.md) and the [AI Engineering Glossary](../ai_engineering_glossary.md).

---

## Lesson 3: Streaming and latency UX

Long answers feel broken if the UI waits for the full completion. Stream tokens (or chunked events) when responses can be long.

Product expectations:

- Show a **partial** answer state and a clear **cancel** control.
- On provider error mid-stream, stop cleanly, keep what is safe to show, and log the correlation id.
- Prefer server-sent events or chunked HTTP from your product API. Do not expose provider API keys to the browser.

Next.js App Router can consume streams from a Route Handler or from your BFF. Keep auth cookies on the product origin. Details belong in Phase F plus official Next.js streaming docs (linked below).

---

## Lesson 4: RAG with citations and evaluation

Retrieval-augmented generation is not “embed everything and hope.” A production-shaped loop looks like this:

1. **Ingest:** parse docs, chunk with overlap, store text + metadata + embedding.
2. **Retrieve:** vector search (often `pgvector` on Postgres you already use), optionally hybrid with keyword search, then optional rerank.
3. **Generate:** prompt with retrieved passages and ask for **citations** tied to chunk ids.
4. **Evaluate:** keep a small golden question set. Measure retrieval hit rate and answer groundedness offline before you celebrate demos.

```text
Docs -> chunk + embed -> vector index
User question -> retrieve top-k -> (optional rerank) -> LLM -> answer + citations
```

Honesty rules for learners and products:

- If retrieval returns nothing useful, say so. Do not invent sources.
- Citations should point to real chunks the user can open.
- Measure before scaling the corpus.

Deep dive (keep this chapter short on theory): [RAG Comprehensive Guide](../rag_comprehensive_guide.md). Pair SQL skills with [19-sql-database-fundamentals](../../19-sql-database-fundamentals/README.md) when the index lives in Postgres.

---

## Lesson 5: Agents, tools, and human-in-the-loop

An **agent** is a model loop that can call tools (search, create ticket, run a query) until it stops. Use agents when the task needs multi-step decisions. Prefer a plain RAG or single-shot prompt when one retrieve-then-answer pass is enough.

Industry practice (LangGraph and similar runtimes):

- Explicit **state** (messages, tool results, flags).
- Bounded loops (max steps / recursion limits).
- **Checkpoints** so a crash can resume a thread.
- **Human-in-the-loop** before irreversible actions (refunds, deletes, sends).

Threat model **prompt injection** whenever tools have side effects. Tool results and retrieved docs can contain instructions that try to override your system prompt. Design tools with least privilege. Confirm destructive actions out of band.

Deeper reading: [AI Agents Guide](../ai_agents_guide.md), [LangChain Guide](../langchain_guide.md), and official [LangGraph overview (Python)](https://docs.langchain.com/oss/python/langgraph/overview) / [LangGraph overview (JS)](https://docs.langchain.com/oss/javascript/langgraph/overview).

---

## Lesson 6: Safety, abuse, and cost controls

Ship controls before you scale traffic:

| Control | Why |
|---------|-----|
| Per-user / per-org rate limits | Stops quota burn and abuse |
| Max tokens in and out | Caps cost per request |
| Model routing | Cheap model for drafts. Stronger model for hard tasks |
| Content / policy filters | Reduce harmful outputs where your product requires them |
| Kill switch | Disable a feature without redeploying the whole site |
| Provider fallback | Degraded mode or honest error when the primary API is down |

Cost is a product feature. Log token usage by feature. Alert on spikes. Do not promise unlimited AI in a free tier without a budget.

---

## Pair with ML modules and guides in this repo

| Need | Go here |
|------|---------|
| LLM and app-facing patterns | [25-generative-ai-llms](../../25-generative-ai-llms/README.md) |
| RAG depth | [RAG Comprehensive Guide](../rag_comprehensive_guide.md) |
| Agents | [AI Agents Guide](../ai_agents_guide.md) |
| LangChain concepts | [LangChain Guide](../langchain_guide.md) |
| Deploy and ops discipline | Modules [13](../../13-model-deployment/README.md)–[14](../../14-mlops-basics/README.md) |
| FastAPI AI service boundary | [Phase H2](phase-h2-fastapi-ai-service.md) |

---

## Self-check exercises

1. Add server-side logging for each LLM call: user or session id, feature name, model, latency, token counts when available, and error class.
2. Build a minimal RAG endpoint that returns **citations** (chunk id + snippet) with each answer. Include one golden question that fails when retrieval is empty.
3. Write a one-page rollback plan: what the UI shows if the provider is down, and how you flip the kill switch.
4. For one tool-enabled flow, list two prompt-injection risks and the least-privilege tool design that reduces them.
5. Sketch (text or diagram) where auth lives versus where the model SDK lives. Confirm the browser never holds the provider key.

---

## Official docs (keep current)

Prefer primary docs over random blog clones:

- [OpenAI API docs](https://platform.openai.com/docs)
- [Anthropic API docs](https://docs.anthropic.com/)
- [LangGraph (Python)](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph (JavaScript)](https://docs.langchain.com/oss/javascript/langgraph/overview)
- [pgvector](https://github.com/pgvector/pgvector)
- [Next.js docs](https://nextjs.org/docs) (streaming / App Router as needed)

Optional automation tools (n8n and similar) can glue workflows. They are **not** a substitute for an owned AI service with auth, evals, and logs.

---

## Next step

Complete the exercises with a small feature on your Phase D–F app. Then continue to [Phase H2: FastAPI AI service](phase-h2-fastapi-ai-service.md) to split Python AI work from the Node product API and add queues plus streaming consumption.
