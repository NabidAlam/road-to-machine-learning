# 06. HTTP

HTTP is the language web servers speak. Browsers, mobile apps, microservices, scrapers, IoT devices. All of them speak some flavor of HTTP. If you know one protocol cold. This is the one.

## A request is just text

Strip away the abstraction. An HTTP request is plain text over a TCP connection. Here's one:

```
GET /index.html HTTP/1.1
Host: example.com
User-Agent: curl/8.0.1
Accept: */*

```

That's it. A method, a path, a protocol version, some headers, a blank line, and optionally a body.

The response is also text:

```
HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 1256

<html><body>...</body></html>
```

You can do it by hand:

```bash
# Open a TCP connection and type the request yourself
nc example.com 80
GET / HTTP/1.1
Host: example.com

# (press enter twice)
```

You'll see the response come back. That's all HTTP is, under the polish.

## Methods (verbs)

You probably know the main ones:

| Method  | Meaning                    | Has body?   | Cached? |
| ------- | -------------------------- | ----------- | ------- |
| GET     | Fetch a resource           | Usually not | Yes     |
| POST    | Create / submit            | Yes         | No      |
| PUT     | Replace a resource         | Yes         | No      |
| PATCH   | Partial update             | Yes         | No      |
| DELETE  | Remove a resource          | Sometimes   | No      |
| HEAD    | Like GET but no body       | No          | Yes     |
| OPTIONS | Ask what's allowed (CORS)  | No          | No      |

A clean REST API uses these verbs predictably:

```
GET    /users          -> list users
GET    /users/42       -> get user 42
POST   /users          -> create a user
PUT    /users/42       -> replace user 42
PATCH  /users/42       -> partial update
DELETE /users/42       -> delete user 42
```

## Status codes

Three-digit numbers grouped by hundred:

| Range | Meaning | Common |
|---|---|---|
| 1xx | Informational | 101 (switching protocols, for WebSockets) |
| 2xx | Success | 200 OK, 201 Created, 204 No Content |
| 3xx | Redirect | 301 (moved permanently), 302 (temp), 304 (not modified) |
| 4xx | Client error | 400 (bad request), 401 (not authenticated), 403 (forbidden), 404 (not found), 429 (rate limited) |
| 5xx | Server error | 500 (internal), 502 (bad gateway), 503 (overloaded), 504 (gateway timeout) |

A useful mental model:
- **2xx**: I did what you asked.
- **3xx**: Try over there.
- **4xx**: You messed up.
- **5xx**: I messed up.

## Headers

Headers are metadata. Most of what makes HTTP interesting is in the headers.

Common request headers:

```
Host: example.com
Authorization: Bearer eyJhbGciOi...
Cookie: session=abc123
User-Agent: Mozilla/5.0 ...
Accept: application/json
Content-Type: application/json
```

Common response headers:

```
Content-Type: application/json
Cache-Control: max-age=3600, public
Set-Cookie: session=abc123; HttpOnly; Secure
Access-Control-Allow-Origin: *
ETag: "deadbeef"
```

The `Cache-Control` and `ETag` ones are how browsers and CDNs decide what to cache. The `Authorization` header is how you authenticate. The CORS headers (`Access-Control-...`) are how browsers decide whether to even let JavaScript see the response.

## HTTPS in one paragraph

HTTPS is HTTP over TLS. TLS does two things: it encrypts the connection (no one on the path can read it), and it authenticates the server (you know you're talking to the real example.com, not a man-in-the-middle).

The cost is a handshake on top of the TCP handshake. For HTTP/1.1 and HTTP/2 that's a few hundred ms on a cold connection. HTTP/3 over QUIC eliminates most of it.

Everyone uses HTTPS now. Browsers mark HTTP sites as insecure. Get a free cert from Let's Encrypt and move on.

## HTTP versions briefly

| Version | Year | Highlight |
|---|---|---|
| HTTP/1.0 | 1996 | One request per connection |
| HTTP/1.1 | 1997 | Persistent connections, pipelining |
| HTTP/2 | 2015 | Multiplexed streams, header compression, binary framing |
| HTTP/3 | 2022 | QUIC (over UDP), no head-of-line blocking, faster handshake |

For most apps it doesn't matter. The framework and CDN handle this for you. If you're optimizing real performance, HTTP/2 vs HTTP/3 matters at the edges.

## Code: making and serving HTTP

Making a request in Python:

```python
import requests

r = requests.get("https://api.github.com/users/octocat")
print(r.status_code)        # 200
print(r.headers["x-ratelimit-remaining"])
print(r.json()["name"])
```

A minimal server with Flask:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

users = {}

@app.get("/users/<int:uid>")
def get_user(uid):
    if uid not in users:
        return {"error": "not found"}, 404
    return jsonify(users[uid])

@app.post("/users")
def create_user():
    data = request.get_json()
    uid = len(users) + 1
    users[uid] = data
    return jsonify({"id": uid, **data}), 201

app.run(port=8000)
```

Try it:

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Ada"}'

curl http://localhost:8000/users/1
```

Notice how cleanly REST maps to HTTP. `POST /users` creates. `GET /users/1` reads. Status codes carry meaning. The server returns 404 for missing, 201 for created. That's the whole point of REST: use HTTP the way it was designed.

## Idempotency (the underrated concept)

A method is **idempotent** if doing it twice has the same effect as doing it once.

- GET, PUT, DELETE: idempotent.
- POST, not idempotent.

This matters because networks are unreliable. If your client sends a POST and times out, you can't safely retry. You might charge the card twice. But a PUT? Safe to retry.

In production you handle this with **idempotency keys**: the client sends a unique ID with the request, the server stores it, and if it sees the same key twice, it ignores the second one. Stripe's API is built on this.

```
POST /payments
Idempotency-Key: abc-123-unique-id
```

## Caching basics

HTTP has built-in cache headers. The two big ones:

```
Cache-Control: max-age=3600          # cache for 1 hour
Cache-Control: no-store              # don't cache at all
Cache-Control: private, max-age=300  # only cache for one user
```

And conditional caching with ETag or Last-Modified:

```
# First request
GET /image.jpg
-> 200 OK, ETag: "abc"

# Later, browser asks "is it still abc?"
GET /image.jpg
If-None-Match: "abc"
-> 304 Not Modified   (no body, save bandwidth)
```

We'll go deeper in the Caching and CDN chapters.

## Things to remember

- HTTP is text over TCP. Read a few raw requests and it stops feeling abstract.
- Methods have meaning. GET reads, POST creates, PUT replaces, DELETE deletes.
- Status codes group meaning by the hundreds: 2xx good, 4xx your fault, 5xx my fault.
- Headers carry the metadata: auth, caching, content type, CORS.
- HTTPS is HTTP + TLS. Use it everywhere.
- Idempotent methods are safe to retry. Non-idempotent need idempotency keys.

## Going deeper

- *HTTP: The Definitive Guide* by Gourley & Totty. Old but still good.
- MDN's HTTP docs: https://developer.mozilla.org/en-US/docs/Web/HTTP. Excellent reference.
- HPBN Chapter 11 (HTTP/2): https://hpbn.co/http2/.
- "Idempotency Keys" by Stripe: https://stripe.com/blog/idempotency.
- The classic RFC 7230 to 7235 if you want to read the spec.
