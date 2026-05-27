# System Design for Beginners

Notes I'm writing as I go through "System Design for Beginners" (21 lessons). Each file is a chapter you can read on its own. No video required.

The goal is not to memorize buzzwords. By the end you should be able to look at a product like Twitter or Uber and have a rough idea of what's running behind the scenes, and why.

Reading style is borrowed from Alex Xu's books: short sentences, real numbers, concrete examples, trade-offs spelled out.

## Where this fits in the repo

This folder is a **foundational side track** that pairs with the main [Road to Machine Learning](../README.md) curriculum. It teaches the general backend vocabulary the ML guides assume you know.

- **Start here** if you're heading toward ML Engineer, MLOps, Data Engineer, AI Engineer, or Full-Stack AI Engineer.
- **Then** read the [ML System Design Guide](../resources/ml_system_design_guide.md) for how these ideas apply to serving models, drift, and MLOps.
- **For interviews,** see the [Interview Preparation Guide](../resources/interview_prep.md#system-design-questions).

You can read this in parallel with the numbered modules. It pairs especially well with module 13 (deployment) and 14 (MLOps).

## How to read this

Go in order if you're new. The first few chapters set up the vocabulary the later ones lean on.

If you already work in software, jump around. The CAP, sharding, and message queue chapters are the ones engineers usually come back to during interviews.

## Lessons

### Background
- [00. Computer Architecture](./00-computer-architecture.md)
- [01. Application Architecture](./01-application-architecture.md)
- [02. Design Requirements](./02-design-requirements.md)

### Networking
- [03. Networking Basics](./03-networking-basics.md)
- [04. TCP and UDP](./04-tcp-and-udp.md)
- [05. DNS](./05-dns.md)

### APIs
- [06. HTTP](./06-http.md)
- [07. WebSockets](./07-websockets.md)
- [08. API Paradigms](./08-api-paradigms.md)
- [09. API Design](./09-api-design.md)

### Caching
- [10. Caching](./10-caching.md)
- [11. CDNs](./11-cdns.md)

### Proxies
- [12. Proxies and Load Balancing](./12-proxies-and-load-balancing.md)
- [13. Consistent Hashing](./13-consistent-hashing.md)

### Storage
- [14. SQL](./14-sql.md)
- [15. NoSQL](./15-nosql.md)
- [16. Replication and Sharding](./16-replication-and-sharding.md)
- [17. CAP Theorem](./17-cap-theorem.md)
- [18. Object Storage](./18-object-storage.md)

### Big Data
- [19. Message Queues](./19-message-queues.md)
- [20. MapReduce](./20-mapreduce.md)

## Books and resources I keep coming back to

- *Designing Data-Intensive Applications* by Martin Kleppmann. The single best book on this topic.
- *System Design Interview Volume 1 & 2* by Alex Xu. Short chapters, real examples, the style I'm copying here.
- *Site Reliability Engineering* by Google (free online: https://sre.google/books/).
- ByteByteGo newsletter and YouTube channel. Good for visual recaps.
- High Scalability blog: http://highscalability.com/. Real architectures from real companies.
- AWS Architecture Center: https://aws.amazon.com/architecture/.

## A note on numbers

When I quote a number like "Redis handles 100k ops/sec" or "S3 stores 11 nines of durability", those numbers are real-world ballparks. They move year to year. Use them to get a sense of scale, not as gospel.
