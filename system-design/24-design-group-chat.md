# 24. Design a Group Chat System

Slack, Discord, WhatsApp, Microsoft Teams. Users join rooms, send messages, see typing indicators and read receipts. The expectation: messages arrive within a second or two, anywhere in the world.

The challenge isn't the algorithm. It's that millions of long-lived connections must each receive only the messages they care about.

## Clarify

| Question                      | Example answer                   |
| ----------------------------- | -------------------------------- |
| 1:1 DMs or rooms or both?     | Both                             |
| Max room size?                | 100K members (large communities) |
| History?                      | Yes, infinite scroll back        |
| Media?                        | Images, files, short video       |
| Reactions / replies?          | Yes                              |
| Typing indicators / presence? | Yes                              |
| End-to-end encryption?        | Not for v1                       |

## Estimate

- **Users:** 50M, 10M concurrent online
- **Messages/sec peak:** 100K
- **Average rooms per user:** 30
- **Storage:** 1 KB/message × 100K/sec × 86,400 ≈ 8 TB/day

Messages add up fast. Plan for cold storage and per-room sharding.

## High-level design

```
[ client ]  <-- persistent WebSocket -->  [ Gateway ]
                                              |
                                              v
                                    +-----------------+
                                    |   Chat API      |
                                    +--------+--------+
                                             |
                +----------------+-----------+-------------+
                v                v                          v
        [ Kafka:                [ Postgres or            [ Redis:
          messages ]              Cassandra:               presence,
                                  messages ]               typing ]
                |
                v
   [ Fanout workers push to gateways ]
                |
                v
        [ Other connected clients ]
```

WebSockets (Chapter 07) for the live connection. Kafka for the durable backbone. Cassandra-style storage for messages (Chapter 15) because the access pattern is "give me last N messages of room X."

## Deep dive 1: Connection layer

Each online user holds a WebSocket to a **gateway** node. Gateways are stateless w.r.t. the user (any gateway can hold any user) but the **routing table** (which user is on which gateway) must be globally known.

```
+-----------+      +-----------+
|  Gateway  |      |  Gateway  |  ... thousands of these
+-----------+      +-----------+
       ^                 ^
       |                 |
       v                 v
+-----------------------------+
|     Pub/sub backbone        |   Redis cluster or Kafka topic per room
+-----------------------------+
```

When a message arrives:
1. API persists it.
2. API publishes `room:42` to the pub/sub backbone.
3. Every gateway subscribed to `room:42` pushes to its connected users in that room.

Gateways subscribe **only** to rooms with at least one online member they serve.

## Deep dive 2: Message storage

Messages have a simple access pattern: "last 50 messages in room X, before timestamp T." Wide-column databases (Cassandra, ScyllaDB) shine here.

```
Partition key:  room_id
Clustering key: created_at DESC
Columns:        message_id, author_id, body, attachments
```

For Postgres-style, shard by `room_id` (Chapter 16). For very large rooms, sub-shard by time bucket.

```sql
CREATE TABLE messages (
    room_id     BIGINT,
    created_at  TIMESTAMPTZ,
    message_id  BIGINT,
    author_id   BIGINT,
    body        TEXT,
    PRIMARY KEY (room_id, created_at, message_id)
);
```

Reads pull the latest page. Older history can move to cheaper storage after 30 days.

## Deep dive 3: Presence and typing

Presence is "who is online". Not durable. Living in memory is fine.

```
presence:user:42  ->  "online", TTL 30s
```

Clients heartbeat every 15 seconds. Miss two beats and you're "away."

Typing indicators are noisier. Don't store them. Publish `user 42 typing in room 7` to the pub/sub backbone, gateways forward, clients show the dot. Drop on the floor if the connection blinks.

## Deep dive 4: Read receipts and unread counts

For each `(user, room)` pair, store the last read message ID.

```
SET HSET reads user:42  room:7 -> msg_id 9912
```

Unread = count of messages where `created_at > last_read_at`. Per-room counter cached in Redis, incremented when new messages arrive for a user who hasn't read them.

## Deep dive 5: Delivery guarantees

WebSockets can drop. A message must not be lost.

- Server **always** writes to the durable store **before** broadcasting.
- Client acks each message ID. On reconnect, client sends "last seen message ID per room"; gateway streams the gap.
- Use Kafka-style append-only log per room for replay (Chapter 19).

You don't need full Paxos here. Order within a room is what matters, and a single partition per room gives you that for free.

## Deep dive 6: Scaling out

- **By room.** Shard rooms across DB and pub/sub backbone.
- **By gateway.** Add gateways behind a load balancer with sticky sessions (Chapter 12). Use a consistent hash (Chapter 13) to assign users to gateways.
- **By region.** Pin large rooms to a "home" region; replicate read-only copies elsewhere. Cross-region writes pay a latency tax. Accept it.

## Things to remember

- WebSockets for the live channel; Kafka or Redis pub/sub for the broadcast backbone.
- Persist the message before you broadcast, every time.
- Wide-column storage with `(room_id, created_at)` keys fits chat reads perfectly.
- Presence and typing live in memory with TTLs. Don't store them.
- Shard by room; very large rooms are special cases worth their own routing.
- Order matters per room, not globally. Design accordingly.

## Going deeper

- Discord's blog series on Cassandra → ScyllaDB migration.
- Slack engineering posts on real-time messaging.
- "Building a chat application in 2024" by various sources for current libraries.
- *Designing Data-Intensive Applications* on stream processing (Chapter 11).
