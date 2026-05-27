# 10. Caching

The fastest query is the one you don't run. Caching is the art of remembering an answer so you don't have to compute it again.

It's also the source of half the production bugs in the world, because cache invalidation is famously hard. Phil Karlton: "There are only two hard things in computer science: cache invalidation and naming things."

## Why caching works

Most data is read way more often than it's written. Think about Twitter: most tweets are written once and read thousands of times. The home page of nytimes.com is rendered once and served a million times an hour.

If you can store the answer somewhere fast (RAM is ~100,000x faster than disk, remember Chapter 0), you save the database an enormous amount of work.

```
              without cache                      with cache
              =============                      ==========

  user ----> app ----> db                 user ----> app -> cache (hit)
                                                       \--> db (miss)
```

A cache hit costs microseconds. A database query costs milliseconds. Multiply by a million requests per minute and you see the impact.

## Where caches live

Anywhere there's slow data being read often. In practice:

```
+------------------+
| Browser cache    |  user's machine, JS/CSS/images
+------------------+
        |
+------------------+
| CDN              |  Cloudflare, Akamai, near the user
+------------------+
        |
+------------------+
| Reverse proxy    |  Nginx, Varnish, in front of your app
+------------------+
        |
+------------------+
| Application      |  in-process memory (LRU map)
+------------------+
        |
+------------------+
| Distributed cache|  Redis, Memcached
+------------------+
        |
+------------------+
| Database         |  Postgres also has its own page cache in RAM
+------------------+
```

Every layer caches what the next layer down would have to compute. The trick is putting the right data in the right layer.

## Cache strategies

How does data get into the cache and stay fresh?

### Cache-aside (lazy loading)

The most common pattern. Your app checks the cache. If miss, fetch from DB and store.

```python
def get_user(user_id):
    key = f"user:{user_id}"
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    user = db.query("SELECT * FROM users WHERE id = %s", user_id)
    redis.setex(key, 300, json.dumps(user))   # 5 min TTL
    return user
```

Pros: simple, only caches what's actually requested.

Cons: first miss is slow. Stale data is possible if the source changes and you don't invalidate.

### Read-through

The cache itself knows how to fetch from the source. Your app just talks to the cache.

Less code in your app, more magic in the cache layer. Less common in practice unless you're using a fancy cache library.

### Write-through

Every write goes to both the cache and the database, at the same time.

```python
def update_user(user_id, data):
    db.update("users", user_id, data)
    redis.setex(f"user:{user_id}", 300, json.dumps(data))
```

Pros: cache is never stale.

Cons: every write costs double. You cache things people might never read.

### Write-back (write-behind)

Writes go to the cache only. The cache flushes to the database later, in batches.

Fast writes. Risk of losing data if the cache crashes before flushing. Use only when you really know what you're doing and can tolerate some data loss.

### Write-around

Writes skip the cache. Only reads populate it.

Pros: avoids caching one-time writes.

Cons: first read after a write is slow.

## Choosing a strategy

Most real apps use **cache-aside** with a TTL. It's simple, gets you 80% of the benefit, and stale data is bounded by the TTL.

Use write-through when the data must always be fresh and the cache is critical.

Avoid write-back unless you have a reason. The data loss risk bites everyone eventually.

## Eviction: what to throw away when the cache is full

Caches are finite. When they fill up, they have to drop something. The policy:

- **LRU (Least Recently Used)**: drop the thing not used in the longest time. The default everywhere.
- **LFU (Least Frequently Used)**: drop the thing accessed least often. Better for skewed access patterns.
- **FIFO**: drop the oldest. Rarely best.
- **TTL-based**: drop anything past its expiration time.

Redis defaults to noeviction (return error) but you can pick `allkeys-lru`, `allkeys-lfu`, `volatile-lru`, etc.

## TTL: the slowest invalidation strategy that still works

The easiest way to keep a cache fresh is to expire entries on a timer.

```python
redis.setex("user:42", 300, json.dumps(user))   # 5 minutes
```

Five minutes of staleness on a user profile is almost always fine. Five minutes of staleness on a stock price is not.

Pick a TTL based on:
- How often the underlying data changes.
- How wrong the answer can be.
- How much load the underlying DB can handle on miss.

If your cache TTL is 60 seconds and your data changes once an hour, you're being wasteful. Bump it to 30 minutes.

## Active invalidation

When the data **really** can't be stale, you invalidate on write:

```python
def update_user(user_id, data):
    db.update("users", user_id, data)
    redis.delete(f"user:{user_id}")
```

Now the next read misses and repopulates. The trade-off: you have to remember to invalidate every place. Miss one and you have a stale-forever bug that's hard to find.

For complex objects (e.g. a user with cached "friend count"), this gets ugly fast. Some teams switch to event-driven invalidation: the database emits a "user updated" event and any cache that listens drops the key.

## Two failure modes everyone hits eventually

### Thundering herd / cache stampede

The cache expires. A million users hit the app at the same moment. The app does a million database queries. The database falls over.

Fixes:
- **Lock**: only one process refills the cache while others wait.
- **Probabilistic early refresh**: as the TTL nears expiration, occasionally refresh ahead.
- **Stale-while-revalidate**: serve the stale value, kick off a background refresh.

Redis cookbook example:

```python
def get_or_set(key, fetch, ttl=300):
    val = redis.get(key)
    if val:
        return val
    # try to acquire a lock so only one worker fetches
    if redis.setnx(f"lock:{key}", "1", ex=10):
        val = fetch()
        redis.setex(key, ttl, val)
        redis.delete(f"lock:{key}")
        return val
    else:
        # someone else is fetching, wait briefly and retry
        time.sleep(0.1)
        return redis.get(key) or fetch()
```

### Cache penetration

Users query keys that don't exist. Every miss hits the database.

Fix: cache the "not found" too, with a short TTL.

```python
user = redis.get(f"user:{user_id}")
if user == "MISSING":
    return None
if user:
    return json.loads(user)
# ... fetch and cache, and cache a "MISSING" if not found
```

For really nasty workloads (lookups against random non-existent IDs), a **Bloom filter** in front of the cache lets you say "definitely doesn't exist" without hitting Redis or the DB.

## Redis vs Memcached

The two big distributed caches.

| Feature      | Redis                                            | Memcached                           |
| ------------ | ------------------------------------------------ | ----------------------------------- |
| Data types   | Strings, lists, sets, hashes, streams, geo, etc. | Strings only                        |
| Persistence  | Optional (RDB/AOF snapshots)                     | None                                |
| Replication  | Yes (primary-replica)                            | No, in core. Some forks support it. |
| Cluster mode | Yes                                              | Yes (consistent hashing in client)  |
| Pub/sub      | Yes                                              | No                                  |
| Use case     | Cache + lightweight database, queues, locks      | Pure cache                          |

In 2026, Redis is the default. Memcached is still great when you genuinely only need a giant LRU and want to save memory.

## A practical mental model

When you add caching, ask:

1. **What's the hit rate?** If under 80%, the cache isn't doing much. Tune key design or scope.
2. **What's the failure mode?** What if the cache returns nothing? What if it returns a stale value?
3. **How does it get invalidated?** TTL? On write? Both?
4. **What's the budget?** Cache is RAM. RAM costs money. Don't cache 1 TB of data "just in case".

## Things to remember

- Caches turn "expensive lookup" into "cheap lookup".
- Cache-aside with TTL is the everyday default.
- LRU is the everyday eviction policy.
- Two pain points: stampedes and stale-forever bugs.
- The right cache lives in the right layer. Browser, CDN, app memory, Redis, DB page cache.

## Going deeper

- *Designing Data-Intensive Applications*, Chapter 1 and 3 (storage and caching mentions).
- Redis docs are excellent: https://redis.io/docs/.
- "Caches everywhere" by AWS: https://aws.amazon.com/caching/.
- "Cache Stampedes Are Hard" by Hacker News conversations and various blog posts.
- Mathias Verraes "Cache invalidation: a sneaky monster": https://verraes.net/2019/05/patterns-for-decoupling-distributed-systems-explicit-public-events/.
