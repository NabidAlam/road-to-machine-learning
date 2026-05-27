# 08. API Paradigms

We've covered the plumbing (HTTP, WebSockets). Now the shape of the API itself. When you're designing a service, you pick a paradigm. Three big ones dominate today: REST, GraphQL, and gRPC.

There's no single right answer. Each one optimizes for different things.

## REST

The default for most public web APIs. Built on top of HTTP. The idea: every "thing" in your system is a resource at a URL, and you act on it with HTTP verbs.

```
GET    /users         -> list users
GET    /users/42      -> get user 42
POST   /users         -> create user
PATCH  /users/42      -> update fields
DELETE /users/42      -> delete
```

That's 90% of what REST is in practice. The other 10% is the formal stuff (HATEOAS, hypermedia, statelessness) that almost nobody implements strictly. Most "REST" APIs are really "HTTP APIs with sensible URLs".

**Good at**: simple, cacheable, debuggable (curl, browser), works through every proxy and firewall.

**Not so good at**: under-fetching (need data from 3 endpoints to build one page), over-fetching (response has 50 fields, you needed 3), no schema enforcement out of the box.

A REST response:

```json
GET /users/42

{
  "id": 42,
  "name": "Ada Lovelace",
  "email": "ada@example.com",
  "created_at": "2026-01-15T10:23:00Z",
  "posts_count": 17
}
```

## GraphQL

Built by Facebook around 2015. One endpoint. The client describes exactly what it wants.

```graphql
query {
  user(id: 42) {
    name
    posts(last: 5) {
      title
      comments {
        author { name }
      }
    }
  }
}
```

The server returns exactly that shape. No more, no less.

**Good at**: clients with varying needs (web vs mobile vs partner integrations), avoiding round trips, strict type system.

**Not so good at**: caching (everything is POST to one URL, traditional HTTP caching doesn't work), authorization complexity (every field needs a check), exposing too much (an N+1 query problem can sneak in).

The standard server library is Apollo. The standard client lib is also Apollo. There's a whole ecosystem.

A tiny GraphQL example with Python + Strawberry:

```python
import strawberry

@strawberry.type
class User:
    id: int
    name: str

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: int) -> User:
        return User(id=id, name="Ada")

schema = strawberry.Schema(query=Query)
```

## gRPC

Google's RPC framework. Uses HTTP/2 under the hood, with messages serialized via Protocol Buffers (Protobuf). You define the contract in a `.proto` file and gRPC generates client and server stubs in many languages.

A `.proto` file:

```proto
syntax = "proto3";

service UserService {
  rpc GetUser (UserRequest) returns (User);
  rpc StreamUsers (Empty) returns (stream User);
}

message UserRequest {
  int32 id = 1;
}

message User {
  int32 id = 1;
  string name = 2;
  string email = 3;
}
```

Run the codegen, and you get typed clients and servers in Go, Python, Java, etc. Calling looks like a normal function:

```python
user = stub.GetUser(UserRequest(id=42))
print(user.name)
```

**Good at**: backend-to-backend microservice calls, streaming (HTTP/2), strong typing, small payloads (binary).

**Not so good at**: browsers (you need grpc-web, an extra proxy), debugging (binary payloads, can't curl), public APIs.

## REST vs GraphQL vs gRPC: when to pick what

| If you're building... | Pick |
|---|---|
| A public API (Stripe, GitHub) | REST |
| A complex frontend that needs flexible data | GraphQL |
| Internal microservices, performance-critical | gRPC |
| A mobile app on slow networks | GraphQL or REST |
| A simple CRUD admin | REST |

Many companies use more than one. Stripe is REST for the public API and gRPC internally. GitHub offers both REST and GraphQL.

## A few honorable mentions

### Webhooks

The reverse of normal APIs. Instead of asking, the server calls **you** when something happens. Stripe uses webhooks heavily: "a payment succeeded", "a refund was issued".

```
POST https://yourapp.com/webhooks/stripe
{
  "type": "payment.succeeded",
  "data": { "amount": 1000, "currency": "usd" }
}
```

You expose an endpoint, give Stripe the URL, they POST to it. Authentication is usually a signed header.

### Long polling and SSE

Mentioned in the WebSocket chapter. Still useful for one-way streams of events without the complexity of full WebSockets.

### Webhooks vs polling vs WebSockets

A practical decision tree:

- Need bidirectional, low latency, lots of messages? WebSockets.
- Server pushes occasional events to one endpoint per customer? Webhooks.
- Server streams to one user's browser, no client messages? SSE.
- Client just needs to check sometimes? Plain HTTP polling.

## Versioning

Whatever paradigm you pick, you'll need to evolve it without breaking clients. Three common patterns:

```
1. URL versioning:    GET /v1/users/42
2. Header versioning: GET /users/42  with  Accept: application/vnd.app.v1+json
3. Query versioning:  GET /users/42?version=1
```

URL versioning is the most common because it's the easiest to debug. Big providers (Stripe, GitHub) use a different approach: date-based versions, with a per-account "default version" that doesn't change automatically. That way old integrations keep working forever.

## Things to remember

- REST: resources at URLs, HTTP verbs, easy to debug, easy to cache.
- GraphQL: one endpoint, clients ask for exactly the data they need, harder to cache.
- gRPC: binary, typed, fast, great for microservices, bad for browsers.
- Webhooks for "tell me when something happens".
- You can mix paradigms. REST for the public, gRPC for internal traffic, GraphQL for the frontend.

## Going deeper

- Roy Fielding's original REST dissertation (Chapter 5): https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm. Heavy but the source.
- GraphQL docs: https://graphql.org/learn/.
- gRPC concepts: https://grpc.io/docs/what-is-grpc/.
- "How we built rich text editing into Linear with collaborative GraphQL" and similar engineering blog posts. Linear, Shopify, GitHub all publish good ones.
- "Why GraphQL?" by Lee Byron: https://medium.com/@leeb/why-graphql-9d70b9d92fdd.
