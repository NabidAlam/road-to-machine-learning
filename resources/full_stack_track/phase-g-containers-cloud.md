# Phase G — Containers, Networking, and Delivery (In-Repo Lessons)

**Maps to:** [Blueprint Phase G](../full_stack_ai_engineer_roadmap.md#phase-g-cloud-containers-and-delivery)

**Goal:** Run the same stack locally and in deployment with reproducible images, compose files, and a reverse proxy at the edge.

---

## Table of Contents

- [Lesson 1: Images vs containers](#lesson-1-images-vs-containers)
- [Lesson 2: Volumes and configuration](#lesson-2-volumes-and-configuration)
- [Lesson 3: Multi-service Compose](#lesson-3-multi-service-compose)
- [Lesson 4: Nginx as reverse proxy](#lesson-4-nginx-as-reverse-proxy)
- [Lesson 5: Makefiles and developer ergonomics](#lesson-5-makefiles-and-developer-ergonomics)
- [Exercises](#exercises)
- [Next step](#next-step)

---

## Lesson 1 — Images vs containers

An **image** is a filesystem snapshot + metadata. A **container** is a running instance.

Rebuild images when dependencies change; bump tags for anything deployed.

---

## Lesson 2 — Volumes and configuration

- **Volumes** persist database data across container restarts
- **Bind mounts** help local dev hot reload
- `.env` files should not be baked into public images

---

## Lesson 3 — Multi-service Compose

Compose links services on a shared network. Define explicit dependencies and healthchecks so APIs do not start before the database is ready.

---

## Lesson 4 — Nginx as reverse proxy

Nginx can terminate TLS (in production), route paths to services, and serve static assets.

At minimum, learn:

- `proxy_pass`
- headers like `Host` and `X-Forwarded-For` (trust carefully)

---

## Lesson 5 — Makefiles and developer ergonomics

Short commands (`make dev`, `make test`) reduce onboarding friction and match CI commands.

---

## Exercises

1. Compose `api` + `db` + `web` with a private network and published ports only where needed.
2. Add a healthcheck endpoint and wire it into Compose health dependencies.
3. Put Nginx in front of two API replicas and document how you would test load balancing locally.

---

## Next step

[Phase H: AI integration](phase-h-ai-integration.md)
