# Phase H2: FastAPI AI Service (In-Repo Lessons)

**Maps to:** [Blueprint Phase H](../full_stack_ai_engineer_roadmap.md#phase-h-ai-integration-for-full-stack-products) (service boundary extension)

**Goal:** Put LLM work behind a **Python AI service** so your Node/Next product API stays focused on auth, tenancy, and product rules. This chapter is about boundaries, queues, and streaming. Model theory stays in Module **25** and Phase **H**.

**Previous chapter:** [Phase H: AI integration](phase-h-ai-integration.md)

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Lesson 1: Why a separate AI service](#lesson-1-why-a-separate-ai-service)
- [Lesson 2: FastAPI service shape](#lesson-2-fastapi-service-shape)
- [Lesson 3: Auth between Node and Python](#lesson-3-auth-between-node-and-python)
- [Lesson 4: Sync vs async jobs](#lesson-4-sync-vs-async-jobs)
- [Lesson 5: Redis, workers, and idempotency](#lesson-5-redis-workers-and-idempotency)
- [Lesson 6: Streaming back to Next.js](#lesson-6-streaming-back-to-nextjs)
- [Lesson 7: Observability and cost at the service edge](#lesson-7-observability-and-cost-at-the-service-edge)
- [Pair with this repo](#pair-with-this-repo)
- [Self-check exercises](#self-check-exercises)
- [Official docs](#official-docs)
- [Next step](#next-step)

---

## Prerequisites

- Phase **H** lessons (product API boundary, RAG/agents at a product level).
- Comfort with HTTP APIs and environment variables.
- Optional: skim [13-model-deployment](../../13-model-deployment/README.md) for service thinking, and FastAPI docs linked below.

You do **not** need Kubernetes for this chapter. One FastAPI process plus a worker process is enough to learn the patterns.

---

## Lesson 1: Why a separate AI service

Keep the **product API** (Node/Express or Nest) as the source of truth for users, orgs, and permissions. Put model SDKs, embeddings, and long-running retrieval in a **Python service** when:

- You want Python libraries (LangGraph, evaluation tooling, scientific stacks) without dragging them into Node.
- Inference or RAG jobs can take seconds to minutes.
- You may scale AI workers independently of the web API.

Anti-pattern: browser talks to the AI service with the provider key. Always go **Browser → Product API → AI service → Provider**.

```text
Browser (Next.js)
    |
    v
Product API (Node)  -- auth, RBAC, quotas, audit
    |
    v
AI service (FastAPI) -- prompts, tools, RAG, provider SDKs
    |-- Redis / queue (optional)
    '-- Worker processes (optional)
```

Start with an in-process AI module (Phase H). Split the service when latency, language, or scaling pressure shows up. Do not split for fashion.

---

## Lesson 2: FastAPI service shape

A minimal production-shaped layout:

| Piece | Role |
|-------|------|
| `POST /v1/complete` | Short sync completions with a hard timeout |
| `POST /v1/rag/query` | Retrieve then generate. Return citations |
| `POST /v1/jobs` | Enqueue long work. Return `jobId` |
| `GET /v1/jobs/{id}` | Job status and result (or error class) |
| `GET /healthz` | Liveness for Compose / load balancers |

Request body should carry a **correlation id** from the product API, a **feature** name, and enough tenant context for logging. Do not invent a second user database inside the AI service. Trust the product API after mutual auth.

Validate inputs with Pydantic models. Reject oversized prompts early. Prefer structured outputs for anything you will store.

---

## Lesson 3: Auth between Node and Python

The AI service is an **internal** dependency. Treat it like a database with an HTTP face.

Practical options (pick one and document it):

1. **Shared secret** header (`Authorization: Bearer <service-token>`) rotated via secrets manager.
2. **mTLS** inside a private network (stronger ops cost).
3. **Short-lived JWT** minted by the product API for each call (aud claim = AI service).

Never expose the AI service port publicly in Compose without auth. Rate-limit by org at the **product** API. The AI service should still enforce a global concurrency cap so one tenant cannot starve others.

---

## Lesson 4: Sync vs async jobs

| Path | Use when | Failure mode |
|------|----------|--------------|
| Sync HTTP | Chat turns under a few seconds | Timeouts. User sees error or retry |
| Async job | Indexing, bulk summarization, multi-tool agents | User polls or gets a webhook/SSE when done |

Rule of thumb: if p95 can exceed your HTTP gateway timeout, use a job. Return `202` with `jobId`. Store status in Redis or Postgres. The product API owns the UX of “still working.”

Idempotency keys matter. If the client retries `POST /v1/jobs`, the same key should return the same `jobId`, not a duplicate billable run.

---

## Lesson 5: Redis, workers, and idempotency

A simple worker loop:

1. Product API calls AI service `POST /v1/jobs`.
2. AI service writes job record + pushes queue message (Redis list or stream).
3. Worker pops job, runs RAG/agent, writes result, marks complete.
4. Client polls `GET /v1/jobs/{id}` or listens on an SSE channel scoped to that job.

Keep workers **stateless**. Put secrets in the environment. Cap max job runtime. Dead-letter failed jobs with an error class the UI can show (“timeout”, “provider_rate_limit”, “unsafe_tool”).

Redis is fine for queues and short-lived status. Durable audit of who asked what still belongs in the product database.

---

## Lesson 6: Streaming back to Next.js

Users expect tokens as they arrive. Two common patterns:

1. **Product API proxies the stream.** Browser opens SSE or fetch stream to Node. Node opens a stream to FastAPI. Browser never sees the AI service host.
2. **Job + partial events.** Worker publishes chunk events keyed by `jobId`. Product API forwards them after auth checks.

Cancel must work. If the user hits Stop, the product API should abort the upstream request or mark the job cancelled so the worker stops spending tokens.

Next.js Route Handlers can pipe streams. Keep cookies and session checks on the product origin. Details live in Next.js streaming docs (linked below) plus Phase F.

---

## Lesson 7: Observability and cost at the service edge

Log every call with:

- correlation id
- org / user ids (from the trusted product API)
- feature name
- model id
- latency ms
- token in/out when available
- error class

Alert on error rate and cost spikes per feature. Add a **kill switch** (feature flag or config) so you can disable expensive routes without redeploying Next.js.

Honesty for learners: a fancy queue does not fix a bad prompt or empty retrieval. Measure groundedness on a small golden set before you scale workers.

---

## Pair with this repo

| Need | Go here |
|------|---------|
| Product-side AI wiring | [Phase H](phase-h-ai-integration.md) |
| LLM concepts | [25-generative-ai-llms](../../25-generative-ai-llms/README.md) |
| RAG depth | [RAG Comprehensive Guide](../rag_comprehensive_guide.md) |
| Agents | [AI Agents Guide](../ai_agents_guide.md) |
| Deploy habits | [13-model-deployment](../../13-model-deployment/README.md), [14-mlops-basics](../../14-mlops-basics/README.md) |
| SQL / pgvector home | [19-sql-database-fundamentals](../../19-sql-database-fundamentals/README.md) |

---

## Self-check exercises

1. Sketch the call path for one chat feature with auth at Node and model calls only in FastAPI. Name where the provider key lives.
2. Implement (or stub) `POST /v1/jobs` + `GET /v1/jobs/{id}` with an idempotency key. Show what a duplicate POST returns.
3. Add a worker that fails on purpose with `provider_rate_limit` and confirm the product UI can show a calm retry message.
4. Proxy a short SSE stream from FastAPI through Node to the browser. Confirm Stop cancels upstream work.
5. Write a one-page capacity note: max concurrent sync calls, max queue depth, and what happens when Redis is down.

---

## Official docs

- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [Redis documentation](https://redis.io/docs/)
- [OpenAI API docs](https://platform.openai.com/docs)
- [Anthropic API docs](https://docs.anthropic.com/)
- [LangGraph (Python)](https://docs.langchain.com/oss/python/langgraph/overview)
- [Next.js docs](https://nextjs.org/docs) (streaming / Route Handlers)

---

## Next step

Return to the [Full-Stack AI Engineer Blueprint](../full_stack_ai_engineer_roadmap.md#phase-h-ai-integration-for-full-stack-products) and ship the Phase H portfolio deliverable with a clear service boundary. Keep ML depth moving through modules **00** to **25** in parallel.
