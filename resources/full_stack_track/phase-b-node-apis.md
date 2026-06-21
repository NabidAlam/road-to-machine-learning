# Phase B — Node.js, HTTP, and API Core (In-Repo Lessons)

**Maps to:** [Blueprint Phase B](../full_stack_ai_engineer_roadmap.md#phase-b-nodejs-express-and-api-core)

**Goal:** Understand how a request becomes a response, then ship a small REST API with validation and clear errors.

---

## Table of Contents

- [Lesson 1: How the web talks](#lesson-1-how-the-web-talks)
- [Lesson 2: Node mental model](#lesson-2-node-mental-model)
- [Lesson 3: Raw HTTP in Node (why it matters)](#lesson-3-raw-http-in-node-why-it-matters)
- [Lesson 4: Express routing and middleware](#lesson-4-express-routing-and-middleware)
- [Lesson 5: Auth basics you will reuse everywhere](#lesson-5-auth-basics-you-will-reuse-everywhere)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1 — How the web talks

A browser or client sends an **HTTP request**: method (`GET`, `POST`, …), path (`/users`), headers, optional body. The server returns a **response**: status (`200`, `404`, `500`), headers, body.

**REST-ish JSON APIs** usually use:

- `GET /resources` list, `GET /resources/:id` read
- `POST /resources` create
- `PUT` / `PATCH` update
- `DELETE` remove

**Exercise:** Sketch JSON request/response shapes for `POST /users` (create) and `GET /users/:id` (read).

---

## Lesson 2 — Node mental model

Node is **event-driven** and **non-blocking** for I/O. Heavy CPU work still blocks the event loop, so you offload CPU jobs or split work.

**Modules:** prefer **ESM** (`import` / `export`) in new projects; many tutorials still show **CommonJS** (`require`). Pick one style per repo and stay consistent.

**Environment config:** never commit secrets. Use `.env` locally and document required variables in `README`.

---

## Lesson 3 — Raw HTTP in Node (why it matters)

Frameworks hide details; building once with `http` teaches routing, status codes, and body parsing.

Conceptual flow:

1. Accept socket → parse HTTP
2. Match route + method
3. Read body stream (for `POST`)
4. Write JSON response with correct `Content-Type`

**Exercise:** Implement `GET /health` returning `{ "ok": true }` using only Node’s `http` module (TypeScript optional).

---

## Lesson 4 — Express routing and middleware

**Middleware** is a pipeline: each function can end the response or call `next()`.

Patterns you want early:

- Central **error handler** (four-arg `(err, req, res, next)` in Express)
- **Validation** before route handlers (reject bad input with `400`)
- Consistent **JSON error shape**, for example `{ "error": { "code": "...", "message": "..." } }`

**Exercise:** Add `POST /echo` that validates `{ "message": string }` and echoes it back; return `400` on bad input.

---

## Lesson 5 — Auth basics you will reuse everywhere

- **Authentication:** who are you? (login)
- **Authorization:** what may you do? (roles, scopes)

Passwords must be **hashed** (for example bcrypt). **JWTs** are signed tokens; they are not a substitute for thinking through **refresh**, **revocation**, and **storage** (cookies vs localStorage trade-offs show up again in Phase F).

**Exercise:** Write a short design note: where will you store access and refresh tokens for a web app vs a mobile app?

---

## Exercises

1. Build `/users` CRUD against an in-memory array first; swap to Postgres in Phase C/D.
2. Add request logging: method, path, status, duration.
3. Add global error middleware and map known errors to stable HTTP codes.

---

## Next step

[Phase C: SQL & databases](phase-c-sql-databases.md)
