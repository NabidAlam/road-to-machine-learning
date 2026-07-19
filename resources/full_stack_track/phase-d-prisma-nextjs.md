# Phase D: Prisma and Next.js (In-Repo Lessons)

**Maps to:** [Blueprint Phase D](../full_stack_ai_engineer_roadmap.md#phase-d-prisma-nextjs-full-stack-build)

**Goal:** Move from ad-hoc SQL strings to a schema-first data layer (Prisma) and ship a user-facing UI with Next.js while keeping secrets and auth boundaries correct.

---

## Table of Contents

- [Lesson 1: Schema-first data with Prisma](#lesson-1-schema-first-data-with-prisma)
- [Lesson 2: Migrations and environments](#lesson-2-migrations-and-environments)
- [Lesson 3: Next.js App Router mental model](#lesson-3-nextjs-app-router-mental-model)
- [Lesson 4: Server vs client boundaries](#lesson-4-server-vs-client-boundaries)
- [Lesson 5: Forms, validation, and errors](#lesson-5-forms-validation-and-errors)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1: Schema-first data with Prisma

Prisma models live in `schema.prisma`. You generate a type-safe client and use it from route handlers or services.

Design checklist:

- Prefer **stable IDs** (`uuid` or `cuid`) for public identifiers
- Add **`createdAt` / `updatedAt`** early
- Model **relations explicitly** (`@relation`) and decide `onDelete` behavior up front

**Exercise:** Model `User`, `Post`, and `Comment` with correct FKs and indexes on foreign keys you query often.

---

## Lesson 2: Migrations and environments

Treat migrations like code review items: they change production data shapes.

- **Dev:** iterate quickly, reset when needed
- **Prod:** backward-compatible steps when possible (add nullable column → backfill → enforce)

Keep **`.env.example`** updated with required variables (`DATABASE_URL`, etc.).

---

## Lesson 3: Next.js App Router mental model

Routes map to folders under `app/`. Layouts nest. Loading and error UI improve perceived quality.

You will repeatedly decide:

- What runs on the **server** by default
- What must be **`"use client"`** because it uses browser-only APIs or event handlers

---

## Lesson 4: Server vs client boundaries

**Rule of thumb:** fetch sensitive data on the server; pass only what the client needs to render.

Avoid leaking service role keys to the browser. If you need user context in a server component, use your auth library’s server session helpers (patterns vary by stack).

**Exercise:** List three things that must never be imported into a client bundle.

---

## Lesson 5: Forms, validation, and errors

Use a schema validator (for example **Zod**) to share validation rules between client and server where practical.

Return actionable errors (`fieldErrors`) instead of generic “bad request” when validation fails.

---

## Exercises

1. Implement CRUD for `Post` with Prisma and JSON APIs.
2. Add a Next page that lists posts from the server without exposing DB credentials to the client.
3. Add optimistic UI *or* explicit loading states; pick one and document the trade-off.

---

## Next step

[Phase E: Advanced backend](phase-e-backend-advanced.md)
