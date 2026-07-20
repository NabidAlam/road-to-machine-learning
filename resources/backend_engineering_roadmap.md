# Backend Engineering from First Principles

A language-agnostic path for building production APIs and services. This guide maps **what to learn**, **where it already lives in this repo**, **what is still thin**, and **free resources** you can use today.

**Who this is for:** learners who want backend engineering as a first-class skill, not only as a side effect of ML or full-stack study.

**How this repo splits backend content today**

| Layer | Where | What you get |
|-------|--------|--------------|
| **Architecture vocabulary** | [System Design for Beginners](../system-design/README.md) | HTTP, caching, load balancing, CAP, queues, sharding |
| **Hands-on Node/Postgres** | [Full-Stack Track](../resources/full_stack_track/README.md) Phases B, C, E | Express, SQL, layering, webhooks, observability |
| **ML serving context** | [Module 13](../13-model-deployment/README.md), [ML System Design Guide](ml_system_design_guide.md) | FastAPI, model APIs, ML service design |
| **This guide** | You are here | Reading order, gaps, portfolio milestones, free links |

Pair this roadmap with the in-repo tracks above. Do not read system design and Node lessons as competing paths. They stack.

---

## Table of Contents

- [Phase 0: Orientation](#phase-0-orientation)
- [Phase 1: The request path](#phase-1-the-request-path)
- [Phase 2: API contracts and identity](#phase-2-api-contracts-and-identity)
- [Phase 3: Application structure](#phase-3-application-structure)
- [Phase 4: Data and search](#phase-4-data-and-search)
- [Phase 4: Speed, async, and decoupling](#phase-4-speed-async-and-decoupling)
- [Phase 5: Reliability and operations](#phase-5-reliability-and-operations)
- [Phase 6: Security](#phase-6-security)
- [Phase 7: Scale and concurrency](#phase-7-scale-and-concurrency)
- [Expert gaps we added](#expert-gaps-we-added)
- [Portfolio milestones](#portfolio-milestones)
- [Free resource library](#free-resource-library)
- [Try next](#try-next)

---

## Phase 0: Orientation

Read this phase once before you touch frameworks.

### 1. Roadmap for backend from first principles

**Goal:** See the full journey before you optimize for tools.

**Order in this guide:** Phase 0 → 7. Each phase ends with a small deliverable. Skip nothing in Phase 1–3 even if you already know Express or Django.

**In-repo:** [System Design README](../system-design/README.md) (side track overview), [Full-Stack AI Blueprint](full_stack_ai_engineer_roadmap.md) Phases B–E (implementation track).

**Free resources**

- [The Twelve-Factor App](https://12factor.net/) (how production services are organized)
- [Backend Developer Roadmap (roadmap.sh)](https://roadmap.sh/backend) (visual checklist)

### 2. Walk the path of a true backend engineer

**Goal:** Know what “good” looks like at junior, mid, and senior levels.

| Level | You can… |
|-------|----------|
| **Junior** | Build CRUD APIs, write SQL, handle auth basics, deploy one service |
| **Mid** | Design REST contracts, tune queries, add caching and queues, ship observability |
| **Senior** | Reason about consistency, failure modes, scaling bottlenecks, and security tradeoffs |

**In-repo:** [Career Roadmap Guide](career_roadmap_guide.md#backend-engineer) (role slice in this repo).

**Free resources**

- [Google SRE Book (free online)](https://sre.google/sre-book/table-of-contents/)
- [Staff Engineer archetypes (Will Larson)](https://staffeng.com/guides/staff-archetypes/)

### 3. What is a backend, how do they work, and why do we need them?

**Goal:** Separate **client**, **API/service**, **data store**, and **async workers** in your mental model.

A backend accepts requests, enforces rules, reads and writes durable state, and returns responses. It hides data layout, business rules, and integration with third parties from the client.

**In-repo:** [Application Architecture](../system-design/01-application-architecture.md), [Phase B Lesson 1](../resources/full_stack_track/phase-b-node-apis.md#lesson-1-how-the-web-talks).

**Free resources**

- [MDN: What is a web server?](https://developer.mozilla.org/en-US/docs/Learn/Server-side/First_steps/Introduction)
- [Microsoft Learn: What is an API?](https://learn.microsoft.com/en-us/training/modules/introduction-to-web-apis/)

### 4. Benefits of learning backend engineering from first principles

**Goal:** Resist tutorial-driven learning that skips the request lifecycle.

First-principles backend work means you can debug any stack: you know where routing, validation, auth, and persistence sit even when the framework magic breaks.

**Why it matters for ML learners:** model serving is still a backend problem (latency, batching, auth, observability). See [Module 13](../13-model-deployment/README.md).

**Free resources**

- [HTTP: The Definitive Guide (O'Reilly sample chapters)](https://www.oreilly.com/library/view/http-the-definitive/1565925092/) (conceptual reference)

---

## Phase 1: The request path

Everything in backend engineering hangs off one question: **what happens between socket open and response sent?**

### 5. Understanding HTTP for backend engineers

**Goal:** Methods, status codes, headers, cookies, content negotiation, keep-alive, and HTTP/2 basics.

**In-repo (strong):** [HTTP](../system-design/06-http.md), [Networking basics](../system-design/03-networking-basics.md), [Phase B](../resources/full_stack_track/phase-b-node-apis.md).

**Free resources**

- [MDN HTTP overview](https://developer.mozilla.org/en-US/docs/Web/HTTP/Overview)
- [HTTP Status Codes (httpstatuses.com)](https://httpstatuses.com/)
- [curl manual](https://curl.se/docs/manual.html) (practice requests without Postman)

### 6. Routing: how requests find their handler

**Goal:** Path params, query strings, route tables, middleware order, and 404 vs 405 semantics.

**In-repo (strong):** [Phase B Lesson 4](../resources/full_stack_track/phase-b-node-apis.md#lesson-4-express-routing-and-middleware), [Flask web development](../01-python-for-data-science/10-flask-web-development.md) (Python alternative).

**Free resources**

- [Express routing guide](https://expressjs.com/en/guide/routing.html)
- [FastAPI path operations](https://fastapi.tiangolo.com/tutorial/path-params/)

### 7. Serialization and deserialization

**Goal:** JSON as the default wire format, schema validation, date/decimal pitfalls, and versioning of payloads.

**Gap in repo:** ML guides cover **model** serialization (pickle, ONNX). This phase is about **request/response bodies**.

**Practice:** Accept JSON, validate with a schema library, return typed errors on bad input.

**In-repo (partial):** [Phase B](../resources/full_stack_track/phase-b-node-apis.md), [Phase D validation](../resources/full_stack_track/phase-d-prisma-nextjs.md), [API Design](../system-design/09-api-design.md).

**Free resources**

- [JSON Schema tutorial](https://json-schema.org/learn/getting-started-step-by-step)
- [Zod documentation](https://zod.dev/) (TypeScript) or [Pydantic](https://docs.pydantic.dev/) (Python)

---

## Phase 2: API contracts and identity

### 8. Authentication and authorization

**Goal:** Sessions vs JWT, refresh tokens, RBAC, OAuth2/OIDC at a practical level, and “authn vs authz.”

**In-repo (strong):** [Phase B Lesson 5](../resources/full_stack_track/phase-b-node-apis.md#lesson-5-auth-basics-you-will-reuse-everywhere), [Full-Stack Blueprint Modules 08–09](full_stack_ai_engineer_roadmap.md), [API Design auth section](../system-design/09-api-design.md).

**Free resources**

- [OAuth 2.0 Simplified](https://www.oauth.com/)
- [JWT.io debugger](https://jwt.io/)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

### 9. Validations and transformations

**Goal:** Validate at the boundary, transform for the domain layer, never trust the client.

**Patterns:** DTO in → domain model → DTO out. Reject early with stable error shapes.

**In-repo (moderate):** [Phase B](../resources/full_stack_track/phase-b-node-apis.md), [Phase D](../resources/full_stack_track/phase-d-prisma-nextjs.md).

**Free resources**

- [OWASP Input Validation Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html)

### 10. Controllers, services, repositories, middleware, and request context

**Goal:** One direction of dependency: route → controller → service → repository → database. Middleware for cross-cutting concerns. Request context for user id, trace id, and locale.

**In-repo (moderate):** [Phase E Lesson 1](../resources/full_stack_track/phase-e-backend-advanced.md#lesson-1-layering-routes-controllers-services). Repository pattern is named here as a gap to practice explicitly.

**Free resources**

- [Martin Fowler: Patterns of Enterprise Application Architecture (online catalog)](https://martinfowler.com/eaaCatalog/)
- [NestJS docs (layering example)](https://docs.nestjs.com/) even if you use Express

### 11. Complete REST API design

**Goal:** Resources, nouns not verbs, idempotent methods, pagination, filtering, error format, and OpenAPI contracts.

**In-repo (strong):** [API Paradigms](../system-design/08-api-paradigms.md), [API Design](../system-design/09-api-design.md), Phase B portfolio builds.

**Free resources**

- [Microsoft REST API design guidelines](https://github.com/microsoft/api-guidelines/blob/vNext/Guidelines.md)
- [OpenAPI Specification](https://swagger.io/specification/)
- [JSON:API specification](https://jsonapi.org/) (one opinionated standard)

---

## Phase 3: Data and search

### 12. Mastering databases with Postgres

**Goal:** Schema design, indexes, transactions, isolation levels, migrations, connection pooling, and EXPLAIN for slow queries.

**In-repo (strong):** [Phase C](../resources/full_stack_track/phase-c-sql-databases.md), [Module 19](../19-sql-database-fundamentals/README.md), [SQL in system design](../system-design/14-sql.md).

**Free resources**

- [PostgreSQL Tutorial](https://www.postgresqltutorial.com/)
- [Use The Index, Luke](https://use-the-index-luke.com/) (indexing and SQL performance)
- [PgBouncer docs](https://www.pgbouncer.org/) (connection pooling)

### 15. Full text search (Elasticsearch and alternatives)

**Goal:** Inverted indexes, analyzers, relevance scoring, and when to use Postgres `tsvector` vs Elasticsearch vs OpenSearch.

**Gap in repo:** Design problems mention search. No dedicated Elasticsearch lesson yet. Phase H covers **pgvector** for semantic search, which complements but does not replace keyword search.

**Free resources**

- [Elasticsearch Guide (official)](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
- [OpenSearch documentation](https://opensearch.org/docs/latest/)
- [PostgreSQL Full Text Search docs](https://www.postgresql.org/docs/current/textsearch.html)

---

## Phase 4: Speed, async, and decoupling

### 13. Caching

**Goal:** Cache-aside, write-through, TTL strategy, cache invalidation, and CDN vs application cache vs database cache.

**In-repo (strong):** [Caching](../system-design/10-caching.md), [CDNs](../system-design/11-cdns.md).

**Free resources**

- [Redis University (free courses)](https://university.redis.com/)
- [AWS Caching best practices (concepts apply beyond AWS)](https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html)

### 14. Task queues and background jobs

**Goal:** Why async work exists, at-least-once delivery, retries, dead-letter queues, and worker scaling.

**In-repo (moderate):** [Message Queues](../system-design/19-message-queues.md), [Phase E Lesson 3](../resources/full_stack_track/phase-e-backend-advanced.md#lesson-3-file-uploads-and-async-processing).

**Free resources**

- [Celery documentation](https://docs.celeryq.dev/) (Python)
- [BullMQ documentation](https://docs.bullmq.io/) (Node)
- [RabbitMQ tutorials](https://www.rabbitmq.com/tutorials)

---

## Phase 5: Reliability and operations

### 16. Error handling and fault-tolerant systems

**Goal:** Stable error envelopes, retries with backoff, circuit breakers, timeouts, bulkheads, and idempotency keys.

**In-repo (moderate):** [Phase B error handling](../resources/full_stack_track/phase-b-node-apis.md), [Phase E idempotency](../resources/full_stack_track/phase-e-backend-advanced.md#lesson-2-idempotency-and-webhooks), [API Design errors](../system-design/09-api-design.md).

**Free resources**

- [Microsoft Azure Architecture: Retry pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/retry)
- [Release It! (summary articles on circuit breakers)](https://pragprog.com/titles/mnee2/release-it-second-edition/)

### 17. Production-grade configuration management

**Goal:** Environment separation, secrets vs config, feature flags, and config validation at startup.

**Gap in repo:** `.env` appears in Phase B/G. This phase adds hierarchy: defaults → env → secrets manager.

**In-repo (partial):** [Phase G env](../resources/full_stack_track/phase-g-containers-cloud.md), [Phase B Lesson 2](../resources/full_stack_track/phase-b-node-apis.md).

**Free resources**

- [The Twelve-Factor App: Config](https://12factor.net/config)
- [HashiCorp Vault tutorials (free tier)](https://developer.hashicorp.com/vault/tutorials)

### 18. Logging, monitoring, and observability

**Goal:** Structured logs, metrics (RED/USE), traces, correlation ids, and alerting that pages humans for symptoms not causes.

**In-repo (moderate):** [Phase E Lesson 5](../resources/full_stack_track/phase-e-backend-advanced.md#lesson-5-observability-for-backends), [Application Architecture](../system-design/01-application-architecture.md).

**Free resources**

- [OpenTelemetry documentation](https://opentelemetry.io/docs/)
- [Google SRE: Monitoring distributed systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Prometheus getting started](https://prometheus.io/docs/prometheus/latest/getting_started/)

### 19. Graceful shutdown

**Goal:** Stop accepting new work, drain in-flight requests, flush buffers, close DB pools, and respect Kubernetes termination grace.

**Gap in repo:** Not covered yet. Treat as a required production checklist item.

**Practice:** Handle `SIGTERM`, expose `/health` and `/ready`, set server `keepAliveTimeout` below load balancer idle timeout.

**Free resources**

- [Kubernetes: Pod lifecycle and termination](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination)
- [Node.js graceful shutdown (official guide)](https://nodejs.org/en/learn/asynchronous-work/graceful-shutdown)

---

## Phase 6: Security

### 20. Backend security: everything you need to know

**Goal:** OWASP API Top 10, injection, SSRF, rate limiting, secrets hygiene, dependency scanning, and least-privilege DB roles.

**Gap in repo:** OWASP links exist in the Full-Stack blueprint. No standalone backend security guide yet. This roadmap collects the minimum bar.

**In-repo (partial):** [Phase B auth](../resources/full_stack_track/phase-b-node-apis.md), [deployment advanced topics](../13-model-deployment/deployment-advanced-topics.md) (cloud IAM).

**Free resources**

- [OWASP API Security Top 10](https://owasp.org/API-Security/)
- [OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/)
- [Mozilla Web Security Guidelines](https://infosec.mozilla.org/guidelines/web_security)

---

## Phase 7: Scale and concurrency

### 21–22. Backend scaling and performance engineering

**Goal:** Vertical vs horizontal scale, load balancers, read replicas, sharding intro, backpressure, and profiling before guessing.

**In-repo (strong):** [Proxies and load balancing](../system-design/12-proxies-and-load-balancing.md), [Replication and sharding](../system-design/16-replication-and-sharding.md), [Consistent hashing](../system-design/13-consistent-hashing.md), [CAP theorem](../system-design/17-cap-theorem.md), [Phase G Nginx](../resources/full_stack_track/phase-g-containers-cloud.md).

**Free resources**

- [High Scalability blog archive](http://highscalability.com/)
- [System Design Primer (GitHub)](https://github.com/donnemartin/system-design-primer)

### 23. Concurrency and parallelism: IO bound vs CPU bound

**Goal:** Event loop for IO, thread/process pools for CPU, when to offload to workers, and why blocking the event loop hurts Node.

**Gap in repo:** Phase B mentions the event loop. This phase needs explicit IO vs CPU framing.

**In-repo (partial):** [Phase B Lesson 2](../resources/full_stack_track/phase-b-node-apis.md#lesson-2-node-mental-model), [Computer Architecture](../system-design/00-computer-architecture.md).

**Free resources**

- [Node.js event loop explained](https://nodejs.org/en/learn/asynchronous-work/event-loop-timers-and-nexttick)
- [Python concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html)

---

## Expert gaps we added

These topics rarely appear in beginner playlists but separate production engineers from tutorial graduates.

| Topic | Why it matters | In-repo touchpoint |
|-------|----------------|-------------------|
| **Health checks (liveness/readiness)** | Orchestrators need to know when to route traffic | Phase G, Module 13 deployment |
| **Idempotency keys** | Safe retries on payments and writes | [Phase E Lesson 2](../resources/full_stack_track/phase-e-backend-advanced.md) |
| **Rate limiting and backpressure** | Protect dependencies under load | [Design a Rate Limiter](../system-design/22-design-rate-limiter.md) |
| **Database migrations** | Schema evolution without downtime fear | Phase C, Phase D Prisma |
| **OpenAPI / contract testing** | Clients and servers stay in sync | [API Design](../system-design/09-api-design.md) |
| **Distributed tracing** | Debug latency across services | Phase E observability |
| **Outbox / transactional messaging** | Reliable events without dual-write bugs | [Message Queues](../system-design/19-message-queues.md) |
| **Webhooks** | Inbound async integration | Phase E Lesson 2 |
| **CI/CD for APIs** | Ship small, ship often, roll back safely | Phase G, [Docker Tutorial](docker_tutorial.md) |
| **Backup and restore drills** | Postgres is only durable if you test recovery | Phase C, Module 19 |

---

## Portfolio milestones

Build these in order. Each proves a slice of the roadmap.

| Milestone | Proves | Suggested stack |
|-----------|--------|-----------------|
| **M1: Raw HTTP API** | Routing, JSON, status codes | Node `http` or Python FastAPI |
| **M2: Auth + Postgres CRUD** | Validation, SQL, migrations | Express + Postgres or FastAPI + SQLAlchemy |
| **M3: Cache + queue** | Redis cache-aside, background worker | BullMQ/Celery + Redis |
| **M4: Search** | Full text or semantic search | Elasticsearch or Postgres FTS + pgvector |
| **M5: Production slice** | Logs, metrics, graceful shutdown, Docker | OpenTelemetry + Compose + health routes |

Show **M3–M5** in README with architecture diagram, env sample, and one `curl` example per endpoint.

---

## Free resource library

Curated links grouped by phase. All free at time of writing.

### Core references

- [MDN Web Docs](https://developer.mozilla.org/en-US/docs/Web)
- [The Twelve-Factor App](https://12factor.net/)
- [System Design for Beginners (in-repo)](../system-design/README.md)

### HTTP, APIs, and design

- [MDN HTTP](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [OpenAPI Specification](https://swagger.io/specification/)
- [Postman Learning Center](https://learning.postman.com/)

### Postgres and data

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Use The Index, Luke](https://use-the-index-luke.com/)
- [SQLBolt (interactive SQL)](https://sqlbolt.com/)

### Caching, queues, search

- [Redis University](https://university.redis.com/)
- [RabbitMQ Get Started](https://www.rabbitmq.com/tutorials)
- [Elasticsearch Guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

### Security and reliability

- [OWASP API Security Top 10](https://owasp.org/API-Security/)
- [Google SRE Books](https://sre.google/books/)
- [OpenTelemetry Docs](https://opentelemetry.io/docs/)

### Node and Python (pick one primary stack)

- [Node.js Learn](https://nodejs.org/en/learn)
- [Express.js Guide](https://expressjs.com/en/guide/routing.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Flask Tutorial (in-repo)](../01-python-for-data-science/10-flask-web-development.md)

---

## Try next

1. Read [System Design: HTTP](../system-design/06-http.md) and [Phase B](../resources/full_stack_track/phase-b-node-apis.md) in parallel this week.
2. Build **M1** without Express first, then rebuild with middleware and validation.
3. Skim [Career Roadmap: Backend Engineer](career_roadmap_guide.md#backend-engineer) and pick modules 19 and 13 when you reach data and ML serving.
