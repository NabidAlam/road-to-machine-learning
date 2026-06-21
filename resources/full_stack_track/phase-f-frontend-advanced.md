# Phase F — Advanced Frontend Engineering (In-Repo Lessons)

**Maps to:** [Blueprint Phase F](../full_stack_ai_engineer_roadmap.md#phase-f-advanced-frontend-engineering)

**Goal:** Build dashboards and data-heavy UIs with predictable auth, caching, and URL-driven state.

---

## Table of Contents

- [Lesson 1: Typed API contracts](#lesson-1-typed-api-contracts)
- [Lesson 2: Auth in the browser vs server](#lesson-2-auth-in-the-browser-vs-server)
- [Lesson 3: Data fetching patterns](#lesson-3-data-fetching-patterns)
- [Lesson 4: Tables, filters, and URL state](#lesson-4-tables-filters-and-url-state)
- [Lesson 5: Performance and UX guardrails](#lesson-5-performance-and-ux-guardrails)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1 — Typed API contracts

Generate or hand-maintain types for API responses. The goal is to catch drift when the backend changes.

Even without codegen, **Zod** schemas can validate unknown JSON at runtime.

---

## Lesson 2 — Auth in the browser vs server

**Cookies (HttpOnly)** are often preferred for session tokens to reduce XSS blast radius compared to localStorage access tokens.

Whatever you choose, document:

- refresh strategy
- what happens on `401`
- how role-based UI gates map to server authorization (never trust UI alone)

---

## Lesson 3 — Data fetching patterns

You will mix:

- server-rendered data for first paint
- client queries for interactive views

Understand cache keys and stale times; avoid refetch storms on navigation.

---

## Lesson 4 — Tables, filters, and URL state

Make list pages shareable: encode filters/sort/page in the query string.

Debounce search inputs; cancel in-flight requests when parameters change.

---

## Lesson 5 — Performance and UX guardrails

- Virtualize huge lists when needed
- Prefer skeletons over hard blocking spinners for slow queries
- Measure bundle size impact of chart/table libraries

---

## Exercises

1. Build a paginated table where sorting and filters round-trip through the URL.
2. Implement a global `401` handler that refreshes tokens once and retries safely.
3. Add optimistic updates for one mutation and document rollback behavior.

---

## Next step

[Phase G: Containers & delivery](phase-g-containers-cloud.md)
