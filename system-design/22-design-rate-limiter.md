# 22. Design a Rate Limiter

A rate limiter says "no" to too many requests. It's the seatbelt of an API. Every public service has one. If you don't, somebody will scrape you, brute-force you, or accidentally DDoS you with a buggy script.

We covered the algorithms in Chapter 09. This chapter is about turning those algorithms into a real distributed service.

## Clarify

| Question                      | Example answer                                                |
| ----------------------------- | ------------------------------------------------------------- |
| Per user, IP, or API key?     | API key (paid plan) and IP (anonymous), with different limits |
| Same limit for all endpoints? | No, `POST /login` is stricter than `GET /products`           |
| What happens when limited?    | Return HTTP 429 with `Retry-After` header                     |
| Soft or hard limit?           | Hard. No grace overage                                        |
| Where does it run?            | Edge (API gateway) and per-service for sensitive endpoints    |

## Estimate

- **Active users:** 10M
- **Requests per second peak:** 100K
- **Rules:** ~50 different limit configs

The rate-limiter call itself must be fast. If the limit check costs 50 ms, you've doubled API latency. Target: **under 1 ms** for the hit/miss decision.

## High-level design

```
[ client ]
    |
    v
[ Load balancer ]
    |
    v
+---------------------+
|  API gateway        |
|  - reads rule set   |
|  - calls limiter    |
+---------+-----------+
          |
          v
   +-------------+
   |   Redis     |   counts per key, per window
   +-------------+
          ^
          | rules synced from
   +-------------+
   |  Postgres   |   source of truth for limit configs
   +-------------+
```

The gateway calls Redis on every request. Redis is the only thing on the hot path. Rules live in Postgres and are reloaded into the gateway every few seconds.

## Deep dive 1: Picking an algorithm

From Chapter 09, the four common algorithms:

| Algorithm              | Pros                           | Cons                        |
| ---------------------- | ------------------------------ | --------------------------- |
| Fixed window           | Simple, one counter per window | Spikes at window boundaries |
| Sliding window log     | Most accurate                  | Memory grows with traffic   |
| Sliding window counter | Good balance, ~exact           | A bit more math             |
| Token bucket           | Allows bursts, smooth refill   | Two values to track         |

For most APIs, **token bucket** wins. It's what AWS, Stripe, and GitHub use. Bursts are friendly to clients, the refill rate caps long-term abuse, and the math is two numbers.

```python
def allow(key, capacity, refill_per_sec):
    now = time.time()
    tokens, last = redis.hmget(key, "tokens", "last")
    tokens = float(tokens or capacity)
    last = float(last or now)
    tokens = min(capacity, tokens + (now - last) * refill_per_sec)
    if tokens >= 1:
        redis.hmset(key, {"tokens": tokens - 1, "last": now})
        return True
    redis.hmset(key, {"tokens": tokens, "last": now})
    return False
```

To stay atomic under load, run this as a Lua script inside Redis. One round-trip, no race.

## Deep dive 2: Where to run it

**Three layers, three trade-offs.**

1. **Edge / CDN / API gateway.** Cheapest to run. Stops bad traffic before it hits your servers. Good for blanket per-IP limits.
2. **Service mesh / sidecar.** Per-service rules. Useful when one team's hot endpoint shouldn't sink another's.
3. **In the app.** Most flexible (you know the user and the business rule). Most expensive (your app paid for the request already).

Real systems do all three.

## Deep dive 3: Failure modes

What if Redis is down?

- **Fail open:** allow the request through. Risk: you got DDoSed during the outage.
- **Fail closed:** reject everything. Risk: your own outage just became user-visible.

Best practice: **fail open at the edge, fail closed for sensitive endpoints** (login, password reset, payment). Log every fail-open event loudly so you know it happened.

What if Redis returns stale numbers across regions? Token bucket tolerates this well. You're slightly more permissive across regions but never wildly off.

## Response shape

Always tell the client what happened. Don't just drop the connection.

```
HTTP/1.1 429 Too Many Requests
Retry-After: 13
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1710000000

{ "error": "rate_limited", "retry_after_seconds": 13 }
```

Clients with exponential backoff (Chapter 09) will recover gracefully.

## Things to remember

- Rate limiting protects against accidents, abuse, and bad clients alike. It is not optional.
- Token bucket is the default. Implement it as a Lua script inside Redis.
- Run limiters at multiple layers; tune at the edge for cheap blanket rules and in the app for business logic.
- Always include `Retry-After` and remaining-quota headers.
- Decide fail-open vs fail-closed per endpoint, not globally.

## Going deeper

- Cloudflare blog posts on their rate limiter design.
- Stripe's "Scaling your API with rate limiters": https://stripe.com/blog/rate-limiters.
- Redis `RateLimit` and `redis-cell` modules.
- Envoy rate-limit service: https://www.envoyproxy.io/docs/envoy/latest/configuration/other_features/global_rate_limiting.
