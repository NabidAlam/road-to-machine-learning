# Phase E — Advanced Backend Engineering (In-Repo Lessons)

**Maps to:** [Blueprint Phase E](../full_stack_ai_engineer_roadmap.md#phase-e-advanced-backend-engineering)

**Goal:** Handle money, files, background work, and failure modes without turning your codebase into copy-pasted routes.

---

## Table of Contents

- [Lesson 1: Layering routes, controllers, services](#lesson-1-layering-routes-controllers-services)
- [Lesson 2: Idempotency and webhooks](#lesson-2-idempotency-and-webhooks)
- [Lesson 3: File uploads and async processing](#lesson-3-file-uploads-and-async-processing)
- [Lesson 4: Query patterns at scale](#lesson-4-query-patterns-at-scale)
- [Lesson 5: Observability for backends](#lesson-5-observability-for-backends)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1 — Layering routes, controllers, services

**Route:** HTTP parsing, status mapping  
**Controller:** request/response shaping, authz checks  
**Service:** business rules and DB transactions

This separation makes testing easier: services should not depend on Express-specific objects.

---

## Lesson 2 — Idempotency and webhooks

Payment providers retry webhooks. Your handler must be **safe to run twice**.

Patterns:

- Store provider event IDs you have processed
- Use DB constraints to prevent double-capture
- Return `2xx` only after durable state updates (or queue work transactionally)

---

## Lesson 3 — File uploads and async processing

Validate **MIME type**, **size limits**, and **filename hygiene**. Prefer scanning and virus scanning policies for real products.

Offload heavy transforms to a worker queue when response latency matters.

---

## Lesson 4 — Query patterns at scale

Avoid unbounded `OFFSET` for huge tables when you can use **keyset** pagination.

Add indexes that match real `WHERE` + `ORDER BY` clauses; verify with `EXPLAIN`.

---

## Lesson 5 — Observability for backends

Minimum viable production awareness:

- Structured logs with **request IDs**
- Error reporting (Sentry-class tools)
- Basic metrics: latency, error rate, queue depth

---

## Exercises

1. Implement a webhook endpoint with an idempotency table.
2. Add a background job that runs every N minutes and cleans stale rows.
3. Pick one slow query and document an index plan.

---

## Next step

[Phase F: Advanced frontend](phase-f-frontend-advanced.md)
