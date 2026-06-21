# 21. Designing Systems

You've read 20 chapters on individual pieces. This one is about putting them together.

In an interview (or a real design review), nobody asks "explain consistent hashing." They say: "Design a URL shortener." You have 45 minutes. You need a process.

## The four steps

Use this order every time. Skipping step 1 is the most common mistake.

```
1. Clarify requirements  (5–10 min)
2. Estimate scale          (5 min)
3. High-level design       (10–15 min)
4. Deep dive               (15–20 min)
```

### Step 1: Clarify requirements

Ask questions. Write answers on the board (or say them out loud).

**Functional**. What does it do?
- Create a short link from a long URL?
- Redirect short → long?
- Custom aliases (`go.company.com/promo`)?
- Analytics (click counts)?
- Expiration?

**Non-functional**. How well?
- Read-heavy or write-heavy?
- Latency target for redirects?
- Availability (can we lose a redirect for a minute)?
- Consistency (must every region see the same link immediately)?

Don't assume. "Do we need user accounts?" changes the design a lot.

### Step 2: Estimate scale

Back-of-the-envelope math from Chapter 2. You need rough numbers to pick SQL vs cache vs queue.

Example prompts you can ask the interviewer:
- Daily active users?
- Writes per second? Reads per second?
- How long do we store data?
- Average URL length? Short code length?

Even if they say "make it up," pick numbers and state them. Interviewers want to see you think in orders of magnitude.

### Step 3: High-level design

Draw boxes and arrows. Name the main components. Don't dive into Redis internals yet.

Typical boxes:
- Client (browser / mobile)
- Load balancer
- API servers (stateless)
- Cache
- Database
- Object storage / queue (if needed)

Say which path is hot (usually reads) and which is cold (writes).

### Step 4: Deep dive

Pick 1–2 areas the interviewer cares about, or that your math says will break:
- How do you generate short codes without collisions?
- How do you scale reads?
- What happens when the cache is empty?
- Multi-region?

Go deep on those. Leave other boxes as "standard replication + monitoring."

---

## Worked example: URL shortener

**Prompt:** Design a service like `bit.ly`. Users submit a long URL and get a short link. Visiting the short link redirects to the long URL.

### Clarify

| Question            | Answer (example)                                |
| ------------------- | ----------------------------------------------- |
| Custom short codes? | Optional, user-chosen if available              |
| Analytics?          | Yes, click count per link                       |
| Auth?               | Yes, registered users; anonymous create allowed |
| Link expiration?    | Optional TTL per link                           |
| Redirect type?      | HTTP 302 (temporary) is fine                    |

**Non-functional (we'll assume):**
- 100M links created per month
- 10:1 read-to-write ratio on redirects vs creates
- Redirect p99 < 100 ms
- 99.9% availability

### Estimate

```
Creates:
  100M / month ≈ 100M / (30 × 24 × 3600) ≈ 40 writes/sec
  Peak × 5 ≈ 200 writes/sec

Redirects:
  200 × 10 = 2,000 redirects/sec peak
  (round to ~2K reads/sec. Modest for a cache-heavy service)

Storage (5 years, rough):
  100M × 12 × 5 = 6B links
  ~500 bytes per row (short code, long URL, metadata)
  6B × 500 B ≈ 3 TB
```

3 TB fits one beefy Postgres cluster for years, but you'll still add a cache because **reads dominate** and latency matters.

### High-level design

```
                    +------------------+
 [ Browser ] -----> |  Load balancer   |
                    +--------+---------+
                             |
              +--------------+--------------+
              v              v              v
        [ API server ] [ API server ] [ API server ]   (stateless)
              |              |              |
              +------+-------+-------+------+
                     |               |
                     v               v
              +-----------+    +-----------+
              |   Redis   |    | Postgres  |
              |  (cache)  |    |  (source  |
              +-----------+    | of truth) |
                               +-----------+
```

**Two APIs:**

1. `POST /v1/links`. Body: `{ "url": "https://...", "custom_code": "promo" }` → `{ "short_url": "https://short.io/promo" }`
2. `GET /{code}`, 302 redirect to long URL (this is the hot path)

For analytics, don't block the redirect. Emit an event (Chapter 19):

```
GET /abc  --> 302 redirect (fast)
         --> async: publish { code, timestamp, user_agent } to Kafka
                    --> analytics consumer updates click counts
```

### Deep dive 1: Generating short codes

You need a unique, short string. Options:

| Approach                   | Pros                                | Cons                                              |
| -------------------------- | ----------------------------------- | ------------------------------------------------- |
| Hash long URL (MD5/base62) | Deterministic, same URL → same code | Collisions; can't support custom codes easily     |
| Auto-increment ID → base62 | Simple, no collisions               | Predictable; need a central ID generator at scale |
| Random string (6–8 chars)  | Unpredictable                       | Must check DB for collision                       |

**Practical choice:** random 7-character base62 (`a-zA-Z0-9`) → 62^7 ≈ 3.5 trillion possibilities. At 6B links you're fine. On collision, retry.

```python
import secrets
import string

ALPHABET = string.ascii_letters + string.digits

def new_code(length=7):
    return "".join(secrets.choice(ALPHABET) for _ in range(length))
```

Custom codes: check uniqueness in Postgres before insert. Return 409 if taken.

**Schema (simplified):**

```sql
CREATE TABLE links (
    id           BIGSERIAL PRIMARY KEY,
    short_code   VARCHAR(16) UNIQUE NOT NULL,
    long_url     TEXT NOT NULL,
    user_id      BIGINT,
    expires_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_links_code ON links(short_code);
```

### Deep dive 2: The redirect path (read-heavy)

Every redirect should hit Redis first (Chapter 10):

```
GET short_code from Redis
  hit  → 302 to long_url
  miss → SELECT from Postgres → SET Redis → 302
```

Cache-aside. TTL: 24 hours (or forever until explicit delete). At 2K reads/sec, a small Redis cluster is trivial.

**Why not only Postgres?** 2K indexed lookups/sec is doable on one machine, but p99 latency and DB connection limits hurt under spikes. Cache keeps the DB for writes and cold keys.

### Deep dive 3: Scaling beyond one region

If users are global:
- **Reads:** CDN won't help much (dynamic redirect). Use read replicas or multi-region Redis with the same codes everywhere.
- **Writes:** still low (200/sec). Single primary Postgres is OK for a long time.
- **Custom domains:** DNS (Chapter 5) points `short.io` to your load balancer.

CAP reminder (Chapter 17): redirects are **AP**. Stale cache might 302 to an old URL for seconds after an update. Acceptable for most shorteners. Use cache invalidation on update/delete.

### What to mention if time allows

- **Rate limiting** (Chapter 9) on `POST /links` to stop abuse.
- **HTTPS** everywhere (Chapter 6).
- **Monitoring** (Chapter 01): redirect latency p99, cache hit rate, 5xx rate.
- **Idempotency** if clients retry creates with the same key.

---

## A second sketch: news feed (30-second version)

Interviewers often ask feed systems after URL shorteners. Same four steps:

1. **Clarify:** Fan-out on write vs read? Celebrity users? Real-time?
2. **Estimate:** 300M users, 500 friends avg, 10 posts/day → posts/sec and fan-out write load.
3. **High-level:**
   - Post service writes to DB + publishes `post.created`.
   - **Fan-out on write:** precompute each follower's feed in Redis/DB (fast read, heavy write for celebrities).
   - **Fan-out on read:** merge posts at request time (cheap writes, slow reads at scale).
   - Hybrid: fan-out on write for normal users; fan-out on read for celebrities.
4. **Deep dive:** feed cache key design, ranking, pagination (cursor from Chapter 9).

You don't need to finish every box. Show you know the trade-off.

---

## Checklist before you say "we're done"

Run through this mentally:

- [ ] Functional and non-functional requirements stated
- [ ] Rough QPS and storage estimated
- [ ] Hot path identified (usually reads)
- [ ] Stateless app tier behind a load balancer
- [ ] Cache for hot reads
- [ ] Database as source of truth
- [ ] Async path for analytics or heavy work (queue)
- [ ] Failure modes: what if cache dies? DB primary dies?
- [ ] One bottleneck called out with a fix

---

## Things to remember

- Design is a **process**, not a memorized diagram. Clarify → estimate → boxes → deep dive.
- Most products are **read-heavy**. Cache and CDN show up constantly.
- **Writes** are where you need IDs, queues, and careful consistency.
- Tie each box back to a chapter you already read: HTTP, DNS, LB, cache, SQL, queues, CAP.
- In interviews, **communication** beats perfection. Think out loud.

## Going deeper

- Alex Xu, *System Design Interview* Vol 1. URL shortener, rate limiter, Twitter feed chapters.
- [System Design Primer](https://github.com/donnemartin/system-design-primer): many worked examples.
- [ML System Design Guide](../resources/ml_system_design_guide.md): same process applied to recommendation and model serving.
- [Interview Preparation Guide](../resources/interview_prep.md#system-design-questions): ML-flavored design prompts.
- Excalidraw or a whiteboard: practice drawing in 10 minutes without a template.
