# 02. Design Requirements

Before you draw a single box on a system design diagram, you need to know what the system is supposed to do. This is where most interviews and most real projects go off the rails.

## Two flavors of requirements

**Functional**: what the system does.
- "Users can post tweets up to 280 characters."
- "Drivers see new ride requests within 2 seconds."

**Non-functional**: how the system behaves.
- "99.9% uptime."
- "Handle 10,000 requests per second."
- "Read latency under 200 ms at p99."

Functional requirements are usually obvious. Non-functional ones are where the architecture is born or dies.

## The four non-negotiables

Every serious system asks these four questions:

```
+------------------+--------------------+
|   Scalability    |   Performance      |
+------------------+--------------------+
|   Availability   |   Consistency      |
+------------------+--------------------+
```

### Scalability

Can it grow? If 10x more users show up tomorrow, what breaks first?

There are two ways to scale:

- **Vertical**: bigger machine. Cheap up to a point.
- **Horizontal**: more machines. Harder to set up, almost unlimited ceiling.

Modern systems are horizontal by default because hardware stopped getting faster (see Chapter 0). The cost is complexity: load balancers, replication, distributed databases.

### Performance

Two metrics, both important:

- **Latency**: how long one request takes. "The page loads in 300 ms."
- **Throughput**: how many requests you handle per second. "We do 50,000 reqs/sec."

These are not the same thing. A system can be fast for one user and crawl under load. They trade off:

```
            high latency, low throughput  --> bad
            low latency, low throughput   --> okay for tiny systems
            high throughput, high latency --> okay for batch jobs
            low latency, high throughput  --> the goal
```

Latency is also a distribution, not a single number. People usually quote:

- **p50** (median): half of requests are faster than this.
- **p95**: 95% are faster. The other 5% suffer.
- **p99**: 99% are faster. This is what your worst users see.
- **p99.9**: tail latency. Important for big systems where 0.1% is still thousands of users.

Why this matters: a system with 100 ms p50 and 5,000 ms p99 is unusable for the unlucky 1%. The average looks fine. The tail is on fire.

### Availability

Is it up?

```
availability = uptime / total time
```

A year is 525,600 minutes. To hit 99.99%, you can be down at most ~52 minutes per year. That's not "we'll restart the server". That's "everything is redundant, automated, and tested".

Two tactics:
- **Redundancy**: have spare copies of everything. If one machine dies, another takes over.
- **Failover**: automatic detection plus a switch to the spare. Manual failover doesn't count at four nines.

### Consistency

When you write data, when does everyone else see it?

- **Strong consistency**: the next read is guaranteed to see the latest write. Like a bank account.
- **Eventual consistency**: it'll show up... eventually. Could be milliseconds. Could be seconds. Like a Twitter like count.

Pick wrong and you either lose money (bank with eventual consistency) or make a slow product (Twitter with strong consistency on a billion users).

We'll come back to this in the CAP chapter.

## How to actually do back-of-the-envelope math

A real interview question:

> "Design Twitter. How do you size it?"

Here's the recipe:

**Step 1: estimate users.**
- 500 million daily active users (current real number is around there).

**Step 2: estimate writes.**
- Average user posts ~0.1 tweets/day.
- 500M × 0.1 = 50M tweets/day.
- 50M / 86,400 seconds = ~600 tweets/sec on average.
- Peak is 5-10x average, so plan for ~6,000 tweets/sec.

**Step 3: estimate reads.**
- Average user reads ~100 tweets/day.
- 500M × 100 = 50B reads/day.
- 50B / 86,400 = ~580,000 reads/sec on average. Peak ~5 million/sec.

**Step 4: estimate storage.**
- One tweet ~ 500 bytes (text + metadata).
- 50M × 500 bytes = 25 GB/day.
- 25 GB × 365 = ~9 TB/year.

**Step 5: estimate bandwidth.**
- 500,000 reads/sec × 500 bytes = 250 MB/sec.

Now you know:
- Reads >> writes (100:1 ratio). Cache aggressively, optimize reads.
- 9 TB/year is small. One database can hold it.
- 250 MB/sec bandwidth needs a CDN.

You didn't need a calculator. You just needed to multiply.

## Useful numbers to keep in your head

| Operation                                             | Time            |
| ----------------------------------------------------- | --------------- |
| L1 cache                                              | 1 ns            |
| RAM                                                   | 100 ns          |
| SSD read                                              | 100 microseconds |
| Network round trip in same datacenter                 | 0.5 ms          |
| Network round trip cross-country (US east to US west)   | ~70 ms          |
| Network round trip across continents (US to EU)       | ~100 ms         |
| HDD seek                                              | 10 ms           |

| Data                       | Size       |
| -------------------------- | ---------- |
| One tweet                  | ~500 bytes |
| Average web page           | ~2 MB      |
| Average photo (compressed) | ~200 KB    |
| Average song (MP3)         | ~5 MB      |
| Movie (1080p, 2 hrs)       | ~5 GB      |

| Scale             | Order           |
| ----------------- | --------------- |
| 1 thousand (10^3) | small product   |
| 1 million (10^6)  | growing product |
| 1 billion (10^9)  | big tech        |

## A worked example

> "Design a URL shortener like bit.ly."

Requirements meeting:

**Functional**: paste a long URL, get a short one. Visit the short one, redirect to the long one. Track click counts.

**Non-functional**:
- 100 million URLs created per month
- 10:1 read-to-write ratio (people click links more than they create them)
- 99.9% availability for redirects (broken links destroy trust)
- Sub-200 ms redirect latency

Quick math:
- 100M / month = ~40 URLs/sec average write.
- 400 reads/sec.
- 100M × 12 months × 100 bytes per record = ~120 GB/year. Fits in one Postgres.

What you'd build:
- A simple service that hashes long URLs to short ones.
- A database. SQL is fine at this scale.
- A cache (Redis) for hot URLs. 80% of clicks go to 20% of URLs.
- A CDN if you're serving lots of geographies.

Notice how the requirements drove the architecture. No CDN if all your users are in one city. No cache if every URL is equally popular. The numbers tell you what to build.

## Things to remember

- Functional vs non-functional requirements. Both matter.
- Four key non-functional dimensions: scalability, performance, availability, consistency.
- Latency is a distribution. Always ask "p99 of what?"
- 99.99% availability means about 52 minutes of downtime per year.
- Back-of-the-envelope math: estimate users, multiply by behavior, divide by seconds.

## Going deeper

- "Numbers Every Programmer Should Know" (Jeff Dean): https://gist.github.com/jboner/2841832.
- *System Design Interview Volume 1* by Alex Xu, Chapter 2. Best concrete walkthrough of estimation.
- Brendan Gregg's USE Method for performance: https://www.brendangregg.com/usemethod.html.
- Google SRE Book on SLOs and SLAs: https://sre.google/sre-book/service-level-objectives/.
