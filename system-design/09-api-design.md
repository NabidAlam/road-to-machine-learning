# 09. API Design

Designing an API is mostly about predictability. A good API feels like the next thing the developer was going to try works. A bad one feels like a maze.

This chapter is opinionated. There's no single right way to design an API, but there is a sane defaults list.

## Resource naming

Use plural nouns for collections, singular for items:

```
GET  /users          # list
GET  /users/42       # one user
POST /users          # create
```

Nested resources are fine, but stay shallow. Two levels max:

```
GET /users/42/posts        # ok
GET /users/42/posts/9      # ok
GET /users/42/posts/9/comments/3/likes   # too deep
```

For the last case, give comments their own top-level endpoint and link by ID:

```
GET /comments/3
GET /comments?post_id=9
```

Use verbs only when the action doesn't fit a resource. `POST /payments/42/refund` is fine.

## Status codes that pull their weight

Pick the right one. Don't 200 OK every response with `{"error": "..."}`. You're making future you grep server logs by hand.

| Code | Use when                                      |
| ---- | --------------------------------------------- |
| 200  | Got the thing                                 |
| 201  | Created the thing                             |
| 204  | Did the thing, nothing to return              |
| 400  | Request was malformed                         |
| 401  | Not logged in                                 |
| 403  | Logged in, but not allowed                    |
| 404  | Doesn't exist                                 |
| 409  | Conflict (duplicate email, version mismatch)  |
| 422  | Validation failed                             |
| 429  | Rate limited                                  |
| 500  | Server crashed                                |
| 503  | Server overloaded                             |

A common point of confusion: 401 vs 403. 401 is "you haven't authenticated". 403 is "I know who you are, but no".

## Pagination

Never return unbounded lists. Two main styles:

**Offset / limit**:

```
GET /posts?limit=20&offset=40
```

Easy. Breaks under heavy writes (page 3 might skip or duplicate items if rows are added/removed).

**Cursor-based**:

```
GET /posts?limit=20&after=cursor_abc
```

Better for infinite scroll. Stable under writes. Slightly more work on the client.

GitHub, Twitter, Stripe all use cursor pagination for anything important. Offset works fine for an admin panel.

## Filtering and sorting

A consistent convention saves a lot of pain:

```
GET /posts?status=published&author=42&sort=-created_at&limit=20
```

- Filters as query params.
- Sort as a single `sort` param. Prefix with `-` for descending.
- Don't invent ten ways to do the same thing.

## Errors should be a contract

When something goes wrong, return a structured error, not a wall of text:

```json
{
  "error": {
    "code": "validation_failed",
    "message": "Email is required.",
    "field": "email",
    "request_id": "req_abc123"
  }
}
```

Include a `request_id`. When a user opens a support ticket, you'll find this request in the logs in seconds instead of hours.

Stripe's error format is the gold standard. Copy it.

## Idempotency

We touched on this in the HTTP chapter. Briefly: any non-idempotent operation (POST, especially payments) should accept an idempotency key:

```
POST /payments
Idempotency-Key: 6f8b3e6e-...

{
  "amount": 1000,
  "currency": "usd"
}
```

Server stores the key with the result for, say, 24 hours. If the client retries, the server returns the cached result instead of charging twice.

## Rate limiting

Two reasons to do it: protect yourself from abuse, and protect noisy clients from breaking everyone else.

Common headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1716808800
```

When the limit hits, return 429 with `Retry-After: 60` so the client knows when to come back. Don't just drop the connection.

Algorithms in order of simplicity:
- **Fixed window**: count requests per minute, reset at the top of every minute. Easy, but bursty around the boundary.
- **Sliding window**: smooth version of fixed window.
- **Token bucket**: tokens drip in at a steady rate, requests consume tokens. Allows bursts. The most flexible.
- **Leaky bucket**: requests queued at a fixed drain rate.

Redis is the workhorse for distributed rate limiting because it has fast atomic counters.

A naive token bucket in pseudocode:

```python
def allowed(user_id, max_tokens=10, refill_per_sec=1):
    tokens, last = redis.hget(user_id, "tokens", "last") or (max_tokens, now())
    elapsed = now() - last
    tokens = min(max_tokens, tokens + elapsed * refill_per_sec)
    if tokens >= 1:
        tokens -= 1
        redis.hset(user_id, tokens=tokens, last=now())
        return True
    return False
```

In production you'd write this as a Lua script for atomicity.

## Authentication

Three flavors you'll see:

- **API keys**: a single secret string sent in a header. Good for server-to-server. Hard to scope or revoke per-user.
- **OAuth 2.0 / OIDC**: full delegated auth (login with Google, Facebook). Complex, well understood.
- **JWTs (JSON Web Tokens)**: signed tokens that hold a user identity. Stateless, the server doesn't have to look anything up. Easy to leak, awkward to revoke. Useful for short-lived tokens.

A typical mobile app: user logs in with email + password, server returns a short-lived JWT + a longer-lived refresh token. The app uses the JWT until it expires, then exchanges the refresh token for a new one.

## Versioning recap

From the previous chapter:

```
URL:    /v1/users/42       (most common)
Header: Accept: application/vnd.app.v2+json
Date:   Stripe style, header like X-API-Version: 2026-05-01
```

Pick one, document it, stick to it. The worst is invisible versioning where the same endpoint changes behavior over time.

## Documentation

If you publish an API, you publish docs. Two big options:

- **OpenAPI / Swagger**: a YAML or JSON spec of every endpoint. Generates interactive docs, client SDKs, tests. The industry default for REST.
- **GraphQL introspection**: the schema is the docs. Tools like GraphiQL render it.

For internal services, even a simple Markdown file with example curls beats nothing.

## A worked example

You're designing the API for a habit tracker (look familiar?). What does it look like?

```
GET    /habits                       # list user's habits
POST   /habits                       # create new habit
GET    /habits/42                    # one habit
PATCH  /habits/42                    # rename
DELETE /habits/42                    # delete

POST   /habits/42/check-ins          # mark done today
GET    /habits/42/check-ins?from=2026-05-01&to=2026-05-26

GET    /me/stats                     # summary: total habits, done today, best streak
```

A check-in needs an idempotency key (don't double-count if the network flakes):

```
POST /habits/42/check-ins
Idempotency-Key: 5a3e... 
```

Pagination on check-ins because they grow forever:

```
GET /habits/42/check-ins?limit=50&after=cursor_abc
```

All errors structured. All responses JSON. Auth via JWT in the `Authorization: Bearer ...` header.

That's a fine API. Predictable, paginated, idempotent. Future-you can extend it without breaking past-you.

## Things to remember

- Plural nouns. HTTP verbs. Status codes that mean something.
- Always paginate lists. Cursor pagination beats offset for anything growing.
- Errors are part of the API contract. Structured, with codes and request IDs.
- Idempotency keys for anything that costs money or changes state.
- Rate limit early. Token bucket is a sane default.
- Version explicitly. URL versioning is the easiest.
- Document with OpenAPI or your team will hate you.

## Going deeper

- Stripe API docs: https://stripe.com/docs/api. The benchmark.
- GitHub REST API docs: https://docs.github.com/en/rest. Comprehensive, well versioned.
- "Best practices for REST API design" by Microsoft: https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design.
- *API Design Patterns* by JJ Geewax. Modern, opinionated, very good.
- OpenAPI Specification: https://www.openapis.org/.
