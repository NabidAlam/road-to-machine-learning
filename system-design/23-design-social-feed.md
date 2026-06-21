# 23. Design a Social Feed

Think Twitter, Threads, Mastodon, Bluesky, LinkedIn. A user opens the app and sees recent posts from people they follow, newest at the top, infinite scroll.

It sounds simple. Then you notice one user has 200 million followers, and another follows 5,000 people. Those two corner cases break almost every naive design.

## Clarify

| Question                 | Example answer                              |
| ------------------------ | ------------------------------------------- |
| Chronological or ranked? | Chronological for v1; ranking later         |
| Replies/threads?         | Yes                                         |
| Media?                   | Yes, images and short video                 |
| How fresh?               | A few seconds is fine; not real-time        |
| Public only?             | Followers-only too                          |
| Edit/delete posts?       | Yes; deletes must remove from feeds quickly |

## Estimate

- **Users:** 300M, 100M active daily
- **Avg follows per user:** 200
- **Posts per day:** 500M total (5 per active user)
- **Reads per second:** 1M (everyone refreshing)
- **Writes per second:** ~6K (500M posts / 86,400 s)

Read-heavy: ~150 reads for every write. Caching and precomputation will dominate.

## High-level design

```
[ client ] --post--> [ API ] --> [ Post DB ]
                       |
                       v
                  [ Kafka: post.created ]
                       |
            +----------+------------+
            v                       v
   [ Fanout workers ]      [ Search/analytics ]
            |
            v
   [ Feed cache (Redis) ]
            ^
            |
[ client ] --GET /feed--> [ API ] reads from cache
```

Two paths: write path produces an event; fanout workers spread it to follower feeds; read path is a fast cache hit.

## Deep dive 1: Fanout on write vs read

This is the classic feed trade-off.

| Approach        | What happens on post                   | What happens on read                                        |
| --------------- | -------------------------------------- | ----------------------------------------------------------- |
| Fanout on write | Insert into each follower's feed cache | Read your own feed cache directly                           |
| Fanout on read  | Just save the post                     | At read time, fetch follows, then their recent posts, merge |

**Fanout on write** is great for normal users. Reads are instant. But for a user with 200M followers, posting once does 200M Redis writes. Bad.

**Fanout on read** is cheap on write, but a user following 5,000 people pays at every refresh. Also bad.

**Hybrid:**

- Normal users (< 10K followers): fanout on write.
- Celebrities (≥ 10K followers): fanout on read.
- At feed time, merge "fanned-out" posts with recent celebrity posts. Sort by timestamp.

This is what most real platforms do. The threshold (10K) is a knob you tune.

## Deep dive 2: Feed storage

For each user, store recent feed entries:

```
feed:{user_id}  ->  sorted set in Redis
                    score = timestamp, member = post_id
                    capped at ~1,000 entries
```

A `ZREVRANGE feed:42 0 49` gets the latest 50 posts for user 42 in microseconds. Then the API hydrates the post IDs from the post DB or post cache.

For pagination, use a **cursor** (Chapter 09):

```
GET /feed?cursor=<last_timestamp>&limit=50
```

Don't use offset on a feed. Posts come and go; offsets are wrong by the time you scroll.

## Deep dive 3: Post storage

You need posts indexed by ID and by author.

```sql
CREATE TABLE posts (
    id          BIGINT PRIMARY KEY,
    author_id   BIGINT NOT NULL,
    body        TEXT,
    media_url   TEXT,
    created_at  TIMESTAMPTZ DEFAULT now(),
    deleted_at  TIMESTAMPTZ
);

CREATE INDEX idx_posts_author_time ON posts(author_id, created_at DESC);
```

At your scale, **shard by post_id** (Chapter 16). Author timeline queries hit the index. For celebrity timelines specifically, also keep a Redis sorted set per author so reads never touch SQL.

## Deep dive 4: Deletes

A user deletes a post. You can't easily yank it from millions of feed caches.

Two options:
- **Tombstone:** mark the post as deleted in the post DB. Feed renders skip tombstoned posts. Cheap on write; one extra check on read.
- **Lazy purge:** background job sweeps and removes the post ID from caches when convenient.

Most platforms do tombstone + lazy purge.

## Deep dive 5: Ranking (v2)

Chronological is easy. Once product wants ranking:

```
score = recency * w1 + author_affinity * w2 + post_engagement * w3 + ...
```

Now you're in ML territory: feature store, model serving, freshness. See Chapter 30 and the [ML System Design Guide](../resources/ml_system_design_guide.md).

## What about hashtags and search?

Out of scope for the feed itself. They are a separate index (Elasticsearch or similar). Same `post.created` Kafka topic feeds them.

## Things to remember

- Feed = read-heavy, write-light, dominated by caching and fanout.
- Fanout on write for normal users; fanout on read for celebrities. Hybrid is mandatory at scale.
- Each user has a capped Redis sorted set as their precomputed feed.
- Paginate with cursors, not offsets.
- Deletes use tombstones; don't try to scrub every cache.
- Ranking turns the feed into an ML system. Start chronological.

## Going deeper

- "The Architecture of Twitter's Timeline" (older but still useful).
- LinkedIn's "Follow Feed Architecture" engineering posts.
- Mastodon source. Same problem, smaller scale, readable code.
- Chapter 30 here for the ML ranking side.
