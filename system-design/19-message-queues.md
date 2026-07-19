# 19. Message Queues

So far, every system we've designed has been synchronous: a request comes in, you do work, you respond. That's fine for fast operations. It breaks down when the work is slow, when it might fail, or when you want to spread it across many workers.

The answer is a queue. Producers push messages onto it. Consumers pull messages off and do the work. Producer and consumer don't have to be running at the same time.

## A picture worth a thousand words

```
                                                       worker 1
                                                          ^
                                                          |
   producer -->  [ queue: M, M, M, M, M ]  --> dispatch -->
                                                          |
                                                          v
                                                       worker 2
                                                       worker 3
```

The producer (e.g. a web server handling a signup) drops a message: "send welcome email to ada@example.com". The queue holds it. A worker picks it up, sends the email, acknowledges, the message is removed. If the worker dies, another picks it up.

The web server didn't wait for SMTP. It returned a response in 50 ms. The email goes out in the background.

## Why it changes the shape of a system

Without queues, every part of your system has to be online and fast at all times. With queues, you decouple:

- **Time decoupling**: producer doesn't wait for the consumer. Burst of 10,000 signups? Queue absorbs it, workers chew through it.
- **Failure decoupling**: SMTP server down? Messages sit in the queue. Producer keeps going. Workers retry later.
- **Scale decoupling**: too many emails? Add 10 more workers. No code change in the producer.
- **Heterogeneous workers**: same queue feeds Python workers, Go workers, whatever.

Patterns this enables:
- Slow tasks (image resizing, video transcoding, analytics).
- Cross-service communication (microservices talking via events).
- Buffering against spikes (Black Friday traffic, viral posts).
- Retries and dead-letter handling.

## Two main flavors: queue vs pub/sub

**Queue (point-to-point)**: each message goes to exactly one consumer. Work is divided.

```
                worker 1 (gets M1, M3)
                       ^
                       |
   producer --> queue --
                       |
                       v
                worker 2 (gets M2, M4)
```

Used for task queues: "send this email", "resize this image". Each task happens once.

**Pub/Sub (topic-based)**: each message goes to every subscriber. Work is fanned out.

```
                   subscriber A
                       ^
                       |
   producer -> topic --+
                       |
                       v
                   subscriber B
                   subscriber C
```

Used for events: "user signed up". Email service sends a welcome. Analytics service logs the event. Marketing service adds them to a campaign. Same event, multiple consumers.

A modern broker like Kafka does both.

## The big players

| Name                      | Type                      | Strengths                                |
| ------------------------- | ------------------------- | ---------------------------------------- |
| RabbitMQ                  | Queue + pub/sub           | Mature, easy, great for task queues      |
| Kafka                     | Distributed log + pub/sub | Huge throughput, long retention, replay  |
| AWS SQS                   | Queue                     | Managed, simple, scales to anything      |
| AWS SNS                   | Pub/sub                   | Pairs with SQS, push to many subscribers |
| Google Pub/Sub            | Pub/sub                   | Like SNS+SQS but Google                  |
| Redis Streams             | Queue + pub/sub           | Lightweight, if you already use Redis    |
| NATS                      | Pub/sub (lightweight)     | Tiny, very fast                          |
| BullMQ / Celery / Sidekiq | Task queue libraries      | Built on Redis or Postgres               |

For a small app, Redis with a task queue library (Celery in Python, Sidekiq in Ruby, BullMQ in Node) is plenty.

For a serious distributed system or analytics pipeline, Kafka is the standard.

## Code: a basic Redis-backed queue with Celery

Producer side (your web request handler):

```python
from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/0")

@app.task
def send_welcome(user_email):
    smtp.send(to=user_email, subject="Welcome", body="...")

# in your signup handler:
send_welcome.delay("ada@example.com")
```

Worker side (a separate process):

```bash
celery -A tasks worker --loglevel=info
```

Now `send_welcome.delay(...)` returns immediately. The actual SMTP send happens in the worker. Run more workers, you handle more emails.

## At-least-once vs exactly-once vs at-most-once

This is the "are you sure my message will be processed?" question.

**At-most-once**: maybe delivered, never delivered twice. Use when duplicates are worse than misses (e.g. notification spam).

**At-least-once**: always delivered, possibly more than once. Use when misses are worse than duplicates (e.g. order processing). The consumer must be idempotent.

**Exactly-once**: delivered exactly once. The holy grail. Practically very hard, expensive in some systems, sometimes available (Kafka with transactions, some careful designs).

Most production systems pick **at-least-once + idempotent consumers**. The queue might deliver the same message twice on retry. Your code handles that gracefully (using idempotency keys, dedup on a unique field, etc.).

## Acknowledgments and retries

How does the queue know a worker finished?

1. Worker pulls message.
2. Queue marks it "in-progress" with a timeout (visibility timeout).
3. Worker does the job.
4. Worker sends "ack" (acknowledgment).
5. Queue deletes the message.

If the worker dies (no ack within the timeout), the queue puts the message back. Another worker picks it up.

The risk: the worker might have actually completed but crashed before the ack. That's why you need idempotent consumers.

## Dead-letter queues

What if a message keeps failing? Maybe it's malformed. Maybe a bug in your code. You don't want it bouncing forever.

The pattern, after N retries, the queue moves the message to a "dead-letter queue". A human (or alert) looks at the DLQ to figure out what's wrong.

```
            attempt 1 -> fails
            attempt 2 -> fails
            attempt 3 -> fails
                        |
                        v
                [ Dead-letter queue ]
                        |
                       alert/email engineer
```

Set this up early. It saves hours of pain in production.

## Kafka in one paragraph (because it's special)

Kafka is technically a queue, but really it's a **distributed log**. Messages aren't deleted after a consumer reads them. They sit in a partitioned log, retained for days or weeks. Multiple consumer groups can read the same data independently.

```
   Topic "orders" with 3 partitions
   
   P0: [M1, M5, M9, ...]
   P1: [M2, M6, M10, ...]
   P2: [M3, M7, M11, ...]

   Consumer group A reads all partitions
   Consumer group B reads all partitions (independently)
```

This is why Kafka is the standard for event-driven architectures and big data pipelines. You produce once, consume many times, replay from any point, scale by partitioning.

A tiny Kafka example with `kafka-python`:

```python
from kafka import KafkaProducer, KafkaConsumer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode(),
)
producer.send("user-events", {"user_id": 42, "event": "signup"})

consumer = KafkaConsumer(
    "user-events",
    bootstrap_servers="localhost:9092",
    group_id="analytics-service",
    value_deserializer=lambda v: json.loads(v.decode()),
)
for msg in consumer:
    print(msg.value)
```

## When you shouldn't use a queue

- For requests that are fast and synchronous. Just respond.
- For data that needs strong consistency with the caller's transaction.
- When you really only have one consumer and one producer in the same service. Sometimes a queue is overkill.

A common antipattern: putting everything through a queue "just in case we need to scale". You don't. Start synchronous. Add queues when there's a real reason.

## A real-world example: photo upload

User uploads a photo. What needs to happen?

1. Store the photo in S3.
2. Generate thumbnails (small, medium, large).
3. Run a face-detection model.
4. Update the user's gallery in Postgres.
5. Send a push notification to their friends.

If you do all of this in the HTTP request, the user waits 5 seconds. Bad.

Queue it:

```
   user uploads
        |
        v
   [ API ]  --> stores photo metadata in DB, returns success
        |
        v
   [ "photo.uploaded" event on Kafka ]
        |
   +----+-----+-----+
   v    v     v     v
   thumb gen  ML  notification  analytics
```

Each consumer does its job independently. Failures in one don't affect others. You can add a new "send to ML moderation" consumer without changing the API.

## Things to remember

- Queues decouple producers and consumers in time, scale, and failure mode.
- Two flavors: queue (one consumer per message) and pub/sub (many).
- Kafka is special: it's a durable, replayable log, not just a queue.
- Most systems run at-least-once delivery. Make consumers idempotent.
- Use dead-letter queues for poison messages. Set this up early.
- Don't queue something unless the latency or scale or failure isolation matters.

## Going deeper

- RabbitMQ tutorials: https://www.rabbitmq.com/tutorials. Hands-on, clear.
- "Kafka: The Definitive Guide" by Confluent. The book.
- Martin Kleppmann's talk "Turning the database inside out": https://www.confluent.io/blog/turning-the-database-inside-out-with-apache-samza/.
- AWS SQS vs SNS vs EventBridge tutorial.
- *Designing Data-Intensive Applications*, Chapter 11.
