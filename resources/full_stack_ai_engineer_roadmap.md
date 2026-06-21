# Full-Stack AI Engineer Blueprint (Beginner to Advanced)

This guide extends the ML journey in this repository with software engineering depth, so you can move from training models to shipping complete AI products.

Think of it as a practical "build and deploy" companion to the core ML roadmap.

**Naming note:** Topic labels such as **Module 01** in this blueprint are **steps inside this guide only**. They are **not** the same as this repository’s numbered folders (**`00-prerequisites` … `25-generative-ai-llms`**). When this file says “pair with core ML modules,” it always means those **repo** folders.

**In-repo lesson chapters:** Step-by-step chapters live under [`resources/full_stack_track/`](full_stack_track/README.md). Read the matching chapter for each phase, then return here for deliverables and external free links.

---

## Why This Blueprint

- **ML + Engineering Together**: Learn how model work connects to backend, frontend, and cloud systems
- **Beginner to Advanced**: Start from TypeScript fundamentals and grow into production architecture
- **Project First**: Each phase ends with a concrete deliverable you can add to your portfolio
- **Industry Ready**: Focus on authentication, reliability, testing, observability, and deployment

---

## How to Study This Path

1. Follow phases in order.
2. Build the deliverable at the end of each phase.
3. Pair this with core ML modules from the main roadmap.
4. Use AI assistants for speed, but keep core reasoning and debugging manual.

---

## Phase A: TypeScript Foundations

**In-repo lesson:** [Phase A. TypeScript foundations](full_stack_track/phase-a-typescript-foundations.md)

### Outcome
Write clean, type-safe TypeScript and reason confidently about types.

### Module Coverage
- **Module 01**: Setup, Node via NVM, data types, arrays/objects, functions, optional/literal types, spread/rest, destructuring, type alias, unions/intersections, ternary/nullish/optional chaining, nullable/unknown/never
- **Module 02**: Type assertions, interfaces, generics, generic constraints, `keyof`, enum vs `as const`, conditional/mapped/utility types
- **Module 03**: OOP in TypeScript (classes, inheritance, guards, access modifiers, getters/setters, static members, polymorphism, abstraction, encapsulation)
- **Module 04**: Assignment - solve real programming problems with strong typing

### Portfolio Deliverable
Build and publish a small TypeScript utility library with tests and examples.

---

## Phase B: Node.js, Express, and API Core

**In-repo lesson:** [Phase B. Node and APIs](full_stack_track/phase-b-node-apis.md)

### Outcome
Understand server fundamentals and build secure REST APIs.

### Module Coverage
- **Module 05**: Web basics, frontend vs backend, Node architecture, event-driven runtime, streams/buffers, CommonJS and ESM
- **Module 06**: Raw Node.js HTTP server in TypeScript, request/response lifecycle, CRUD flow, file organization, env-based config
- **Module 07**: Express + TypeScript + PostgreSQL CRUD APIs
- **Module 08**: Modular backend structure, authentication vs authorization, bcrypt, JWT basics
- **Module 09**: Middleware patterns, RBAC, refresh tokens, business-rule checks, global error handling
- **Module 10**: Assignment - authenticated API with SQL integration

### Portfolio Deliverable
Ship a user-management backend with role-based access and refresh-token authentication.

---

## Phase C: Database and SQL Mastery

**In-repo lesson:** [Phase C. SQL and databases](full_stack_track/phase-c-sql-databases.md)

### Outcome
Design scalable relational schemas and write reliable SQL for real applications.

### Module Coverage
- **Module 11**: Data modeling, key systems, cardinality, ER diagrams
- **Module 12**: Normalization (1NF/2NF/3NF), anomalies, junction tables, PostgreSQL setup
- **Module 13**: PostgreSQL essentials, data types, DDL/DML, constraints
- **Module 14**: Advanced querying, filtering, sorting, aggregates, scalar functions
- **Module 15**: `COALESCE`, pagination, update/delete, joins, grouping and HAVING
- **Module 16**: Assignment - relational design + implementation for a realistic case

### Portfolio Deliverable
Create an ERD-backed SQL project with migration scripts and documented query examples.

---

## Phase D: Prisma + Next.js Full-Stack Build

**In-repo lesson:** [Phase D. Prisma and Next.js](full_stack_track/phase-d-prisma-nextjs.md)

### Outcome
Build production-style applications with clean backend architecture and modern frontend patterns.

### Module Coverage
- **Modules 17-24 (Backend Product Layer)**:
  - Prisma setup, schema design, migrations, CRUD and upsert
  - Requirement analysis and ERD translation
  - Auth flows (email/password + social), verification, RBAC
  - Filtering/search/sort/pagination patterns
  - Comment system, nested relations, admin operations, analytics
  - Centralized error handling and Prisma-specific exception management
- **Modules 25-29 (Next.js Core Layer)**:
  - Next.js routing and layout architecture
  - Loading and error boundaries
  - Session/token/cookie auth patterns
  - SSR/SSG/ISR and environment safety
  - Server actions, revalidation, URL-state and pagination
- **Module 30**: Assignment - full-stack project without over-relying on AI generation

### Portfolio Deliverable
Build a full-stack content platform with authentication, moderation flow, and analytics.

---

## Phase E: Advanced Backend Engineering

**In-repo lesson:** [Phase E. Advanced backend](full_stack_track/phase-e-backend-advanced.md)

### Outcome
Move from "feature-complete" to "production-reliable."

### Module Coverage
- **Modules 31-37**:
  - Scalable route-controller-service architecture
  - JWT lifecycle and secure account endpoints (`/me`, logout, reset, refresh)
  - OTP/email verification workflows
  - File uploads with Cloudinary
  - Reusable query builder patterns
  - Scheduling, payments (Stripe), and webhooks
  - Seed automation and robust error handling

### Portfolio Deliverable
Develop a domain backend (for example healthcare booking) with payments and automation jobs.

---

## Phase F: Advanced Frontend Engineering

**In-repo lesson:** [Phase F. Advanced frontend](full_stack_track/phase-f-frontend-advanced.md)

### Outcome
Build complex, maintainable frontends with role-aware dashboards and data-heavy UIs.

### Module Coverage
- **Modules 38-45**:
  - Large-scale Next.js project setup
  - Typed API client patterns
  - Zod schema validation and secure auth forms
  - Route guards and token refresh logic
  - Role-based dashboard navigation/layout
  - Charts, tables, and reusable UI data components
  - Search/filter/sort/pagination patterns with URL state

### Portfolio Deliverable
Create a multi-role admin dashboard with analytics cards/charts and advanced table workflows.

---

## Phase G: Cloud, Containers, and Delivery

**In-repo lesson:** [Phase G. Containers and delivery](full_stack_track/phase-g-containers-cloud.md)

### Outcome
Containerize services and run multi-service systems with confidence.

### Module Coverage
- **Modules 46-52**:
  - Docker fundamentals and container lifecycle
  - Images, containers, volumes, bind mounts, `.env`, `.dockerignore`
  - Container networking and communication
  - Multi-container architecture design
  - Docker Compose orchestration and Makefile automation
  - Nginx reverse proxy and load balancing fundamentals

### Portfolio Deliverable
Run a composed stack (frontend + API + database + reverse proxy) with reproducible local deployment.

---

## Phase H: AI Integration for Full-Stack Products

**In-repo lesson:** [Phase H. AI integration](full_stack_track/phase-h-ai-integration.md)

### Outcome
Design AI features that are useful, grounded, and production-aware.

### Module Coverage
- **Modules 53-61**:
  - AI/ML/LLM essentials for developers
  - Prompt design and token-aware optimization
  - Node.js AI API integration
  - LangChain orchestration and tool usage
  - Streaming chat experiences in Next.js
  - Document summarization workflows
  - Embeddings and semantic search with pgvector
  - RAG pipeline design and retrieval evaluation
  - Fine-tuning vs RAG decision-making
  - AI workflow automation with n8n

### Portfolio Deliverable
Ship an AI assistant feature with retrieval grounding, monitoring hooks, and fallback/error handling.

---

## Bonus Tracks

- **R1 / Software Engineering Essentials**: architecture, testing, deployment, maintainability, ethics
- **R2 / Testing Mastery**: unit/integration/API testing, advanced mocking, debugging workflows
- **R3 / AWS and DevOps Foundations**: IAM, S3, Linux, EC2, Nginx, CI/CD with GitHub Actions

---

## Free Resources by Phase

Use these free resources alongside each phase. They are practical, well-maintained, and beginner-friendly.

### TypeScript (Phase A)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [Total TypeScript - Free Tutorials](https://www.totaltypescript.com/tutorials)

### Node.js and APIs (Phase B)
- [Node.js Learn](https://nodejs.org/en/learn)
- [Express Guide](https://expressjs.com/en/guide/routing.html)
- [JWT Introduction](https://jwt.io/introduction)

### SQL and Databases (Phase C)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [SQLBolt (Interactive SQL)](https://sqlbolt.com/)
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/)

### Prisma and Next.js (Phase D and F)
- [Prisma Documentation](https://www.prisma.io/docs)
- [Next.js Learn](https://nextjs.org/learn)
- [shadcn/ui Documentation](https://ui.shadcn.com/docs)
- [Zod Documentation](https://zod.dev/)

### Backend Reliability and Security (Phase E)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [Stripe Docs](https://docs.stripe.com/)

### Containers and Cloud (Phase G)
- [Docker Get Started](https://docs.docker.com/get-started/)
- [Play with Docker Classroom](https://training.play-with-docker.com/)
- [Nginx Beginner's Guide](https://nginx.org/en/docs/beginners_guide.html)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

### AI Product Integration (Phase H)
- [OpenAI API Docs](https://platform.openai.com/docs)
- LangChain: [Python docs](https://python.langchain.com/docs/introduction/) (general concepts) · [JavaScript / TypeScript docs](https://docs.langchain.com/oss/javascript/langchain/overview) (use this track when your stack is Node or Next.js)
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Hugging Face Course](https://huggingface.co/learn)
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/)
- [Made With ML](https://madewithml.com/)

---

## Pairing with This Repository's ML Roadmap

For balanced growth, run both tracks:

- **ML depth**: Continue core modules in this repo (`00-15`, `19-21`, `25`)
- **Engineering depth**: Complete Phases A-H in this blueprint
- **Portfolio target**:
  - 2 ML-focused projects
  - 2 full-stack AI products
  - 1 deployment/operations case study

If you complete both tracks, you will not only train models, but also design and ship AI systems that real users can trust.
