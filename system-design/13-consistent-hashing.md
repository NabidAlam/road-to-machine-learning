# 13. Consistent Hashing

You have 4 cache servers. You want to spread keys evenly across them. Easiest approach:

```python
server_index = hash(key) % 4
```

Works great. Then you add a 5th server. Now your formula is `% 5`. Almost every key suddenly hashes to a different server. Your entire cache is invalidated overnight. Disaster.

This is the problem consistent hashing solves.

## The naive approach and why it breaks

With `hash(key) % N`:

```
4 servers:  key "abc" -> hash 7 -> server 7 % 4 = 3
5 servers:  key "abc" -> hash 7 -> server 7 % 5 = 2
```

Add or remove one server, and the mapping for most keys changes. That means your cache empties out and your database gets hammered. Or in a distributed database, data has to be reshuffled across all nodes at once.

We need a hashing scheme where adding or removing a node moves only a small fraction of keys.

## The ring

Consistent hashing places servers and keys on a circle (the "hash ring"). Both servers and keys are hashed to a number in some range, say 0 to 2^32.

```
                  0 / 2^32
                     |
                     |
                   [ A ]
                  /
                 /
          [ K1 ] 
                                  [ K2 ]
                                       \
                                        \
       [ D ]                          [ B ]
            \                        /
             \                      /
                       [ C ]
                         |
                      key K3
```

Picture a clock. Servers (A, B, C, D) are positions on the clock. Each key (K1, K2, K3) is also a position. To find which server a key belongs to, you start at the key's position and walk clockwise until you hit a server. That's its home.

So:
- K1 sits between A and B going clockwise → owned by B.
- K2 between B and C → owned by C.
- K3 between C and D → owned by D.

## What happens when you add a server

You add server E somewhere on the ring:

```
                  0 / 2^32
                     |
                   [ A ]
                  /
          [ K1 ]
                                  [ K2 ]
                                       \
       [ D ]                          [ B ]
            \                        /
             \                  [ E ]
                       [ C ]
                         |
                      K3
```

Now K2, which was owned by C, might be closer to E. The only keys that move are the ones between E and the next server counterclockwise. Roughly `keys / N` of them, not all of them.

Same when removing: only keys that pointed at the dead server need a new home.

This is the magic. Adding or removing a node moves ~`1/N` of the keys, not all of them.

## Virtual nodes (the trick that makes it work in practice)

Plain consistent hashing has a problem: if you only have 4 servers on a giant ring, they probably won't be evenly spaced. One server might own a quarter of the ring. Another might own a tiny slice.

The fix is to place each server in **many positions** on the ring, called **vnodes** or **virtual nodes**. Each physical server gets 100 to 1000 spots.

```
Server A -> hashed 200 times -> 200 spots on the ring
Server B -> hashed 200 times -> 200 spots on the ring
Server C -> hashed 200 times -> 200 spots on the ring
```

Now the load is spread evenly because of the law of large numbers. Add a fourth server with its own 200 vnodes, and it picks up keys from all the others fairly.

## A tiny Python implementation

```python
import hashlib
from bisect import bisect_right, insort

class ConsistentHash:
    def __init__(self, vnodes=200):
        self.vnodes = vnodes
        self.ring = []           # sorted list of (hash, server)
        self.hash_to_server = {} # hash -> server name

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_server(self, name):
        for i in range(self.vnodes):
            h = self._hash(f"{name}-{i}")
            insort(self.ring, h)
            self.hash_to_server[h] = name

    def remove_server(self, name):
        for i in range(self.vnodes):
            h = self._hash(f"{name}-{i}")
            self.ring.remove(h)
            del self.hash_to_server[h]

    def get_server(self, key):
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.ring, h) % len(self.ring)
        return self.hash_to_server[self.ring[idx]]

ring = ConsistentHash()
ring.add_server("A")
ring.add_server("B")
ring.add_server("C")

print(ring.get_server("user:42"))
print(ring.get_server("user:7"))
print(ring.get_server("session:abc"))
```

Try adding a server and checking what changes:

```python
before = {k: ring.get_server(k) for k in ["user:1", "user:2", "user:3", "user:4", "user:5"]}
ring.add_server("D")
after = {k: ring.get_server(k) for k in ["user:1", "user:2", "user:3", "user:4", "user:5"]}
print("Moved:", {k for k in before if before[k] != after[k]})
```

Most keys stay put. That's the win.

## Where you'll see this used

- **Distributed caches**: Memcached client libraries (libmemcached, twemproxy) use consistent hashing to spread keys across servers.
- **Distributed databases**: Cassandra, DynamoDB, Riak. They use it to decide which node owns which row.
- **CDNs**: deciding which edge cache handles which URL.
- **Load balancers**: when you want sticky-ish routing without a session store. Hash the user ID, always send them to the same backend.
- **Service meshes**: Envoy supports consistent hashing for upstream load balancing.

Anywhere data is partitioned across nodes and you want adding/removing nodes to be cheap.

## Replication on top of consistent hashing

In a real distributed DB, you don't just put a key on the next server clockwise. You put it on the next **N** servers for replication. If N=3 and your key K2 lands at C, it's also replicated to D and E.

```
K2 -> primary on C, replicas on D and E
```

Now if C dies, you still have the data on D and E. We'll come back to this in the replication chapter.

## Quick honorable mentions

- **Rendezvous hashing (HRW)**: another scheme with the same "minimal disruption" property, simpler to implement, often performs as well or better. Worth knowing.
- **Jump consistent hash**: by Google. Tiny, fast, but requires servers numbered 0..N. Used in some Google internal systems.

If you ever read about Dynamo (the Amazon paper, not DynamoDB), consistent hashing is the foundation of how they did sharding.

## Things to remember

- `hash(key) % N` breaks badly when N changes.
- Consistent hashing maps both keys and servers to a ring; each key goes to the next server clockwise.
- Adding/removing a server only moves ~`1/N` of keys.
- Virtual nodes (many positions per physical server) keep the load balanced.
- It's the backbone of distributed caches, NoSQL databases, and modern load balancers.

## Going deeper

- The original paper: "Consistent Hashing and Random Trees" by Karger et al., 1997.
- Amazon's Dynamo paper, 2007. Foundational reading for distributed systems: https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf.
- "A Guide to Consistent Hashing" by Toptal: https://www.toptal.com/big-data/consistent-hashing.
- Mailgun's blog post on rendezvous hashing, which is a great comparison: https://blog.mailgun.com/.
- ByteByteGo on consistent hashing has good diagrams: https://blog.bytebytego.com/.
