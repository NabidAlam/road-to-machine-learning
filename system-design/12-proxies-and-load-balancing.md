# 12. Proxies and Load Balancing

A proxy is a server that stands between two other servers and passes traffic between them. That sounds boring. It is also one of the most useful patterns in all of system design.

Two flavors:

- **Forward proxy**: sits in front of clients. Hides them from servers. Examples: corporate web filter, a VPN.
- **Reverse proxy**: sits in front of servers. Hides them from clients. Examples: Nginx, Cloudflare, HAProxy.

In system design we almost always mean **reverse proxy**.

## Reverse proxy

```
              +---------------+
   user --->  | reverse proxy | ---> [ app server 1 ]
              |               | ---> [ app server 2 ]
              |               | ---> [ app server 3 ]
              +---------------+
```

The user thinks they're talking to one server. Behind the proxy, there might be dozens. The proxy handles:

- Routing requests to whichever backend is healthy.
- TLS termination (decrypt HTTPS once at the edge).
- Caching common responses.
- Compression.
- Rate limiting.
- Adding headers, rewriting URLs, redirecting.

Every serious web app puts a reverse proxy in front of its app servers. Nginx, HAProxy, Envoy, Caddy, and Traefik are the popular open-source ones. AWS ALB and GCP Cloud Load Balancing are the managed ones.

## Load balancing is just one job of a reverse proxy

A load balancer distributes incoming traffic across multiple backend servers. The point: no single server gets crushed, and if one goes down, the others pick up the slack.

The simplest possible LB:

```nginx
# Nginx config
http {
  upstream app {
    server app1:3000;
    server app2:3000;
    server app3:3000;
  }

  server {
    listen 80;
    location / {
      proxy_pass http://app;
    }
  }
}
```

That's a working load balancer. Six lines. Nginx will round-robin requests across the three backends.

## How does it pick a backend?

This is the interesting part. A few common algorithms:

### Round robin

Send request 1 to server A, request 2 to server B, request 3 to server C, then loop.

Pros: dead simple, fair.

Cons: doesn't consider server load. If server B is slow, it still gets its share.

### Least connections

Pick the backend with the fewest open connections.

Pros: handles servers with different speeds or long requests.

Cons: needs the LB to track state.

### Least response time

Pick the backend that's responded fastest recently.

Pros: very smart about real-world performance.

Cons: more state, more compute.

### Hash-based (IP hash or URL hash)

Hash some property of the request (client IP, URL) and use that to pick a backend. The same client always lands on the same server.

Pros: useful for session stickiness or caching by URL.

Cons: hot keys can imbalance the cluster.

### Weighted variants

Same algorithms but with weights. A 2x bigger server gets 2x the traffic.

```nginx
upstream app {
  server app1:3000 weight=3;
  server app2:3000 weight=1;
}
```

In practice, **round-robin** or **least connections** is what you'll see 90% of the time.

## L4 vs L7 load balancers

Quick refresher from Chapter 3:

- **L4** (transport layer): the LB shuffles TCP/UDP connections without looking inside. Fast and simple. Doesn't know it's serving HTTP.
- **L7** (application layer): the LB understands HTTP. Can route by URL, header, cookie. Can do redirects, rewrite, terminate TLS.

An L7 LB:

```
   /api/*        ---> [ api servers ]
   /images/*     ---> [ image servers ]
   /             ---> [ web servers ]
```

An L4 LB doesn't know any of this. It just hashes the connection and picks a backend.

For most web traffic, you want L7. AWS Application Load Balancer (ALB), Nginx, and Envoy are L7. AWS Network Load Balancer (NLB) is L4. Cloud providers happily charge you for both.

## Health checks

The LB needs to know which backends are alive. Standard pattern:

```nginx
upstream app {
  server app1:3000 max_fails=3 fail_timeout=30s;
  server app2:3000 max_fails=3 fail_timeout=30s;
}
```

Or more explicitly, every backend exposes a `/health` endpoint that returns 200 if it's healthy. The LB pings it every few seconds. If a backend fails 3 in a row, it's removed from rotation until it's healthy again.

```python
# Flask example
@app.get("/health")
def health():
    if db.is_alive() and redis.is_alive():
        return {"ok": True}, 200
    return {"ok": False}, 503
```

This is also what Kubernetes uses for "readiness probes" and "liveness probes".

## Sticky sessions (and why to avoid them)

Sticky sessions mean "this user always lands on the same backend". Useful when the server holds state (in-memory session, WebSocket connection).

The cost: load isn't really balanced anymore. If a popular user is stuck on one server, that server gets hot. And if that server dies, the user loses state.

Better fix: make your backends stateless. Move session state to Redis or a database. Then any backend can serve any user, and round-robin works again.

WebSockets are the legitimate use of sticky sessions, because the connection is the state.

## Failover and high availability

What if the load balancer itself dies? Then you've moved the single point of failure, not eliminated it. Real systems run multiple LBs:

```
                DNS (returns multiple IPs)
                       |
            +----------+----------+
            |                     |
       [ LB 1 ]              [ LB 2 ]
        |    \                /    |
        |     \              /     |
   [ app 1 ] [ app 2 ] [ app 3 ] [ app 4 ]
```

If LB 1 dies, DNS keeps directing some users to LB 2. Combine with short DNS TTLs and clients re-resolve quickly. Or use anycast IPs (one IP advertised from multiple locations, like Cloudflare and Google use).

## Reverse proxy as cache, compression, and gateway

The same Nginx or Envoy you use as a load balancer also commonly does:

- **Static file serving**: don't bother your app with image requests.
- **Gzip / Brotli**: compress responses before sending.
- **Rate limiting**: drop excessive requests at the edge.
- **TLS termination**: HTTPS in, HTTP to backends.
- **WAF (Web Application Firewall)**: block bad patterns.
- **Header rewrites**: add CORS headers, strip internal headers.

This is why "reverse proxy" and "API gateway" overlap so much. Modern API gateways like Kong, Tyk, and AWS API Gateway are reverse proxies with extra features (auth, schema validation, plan management).

## A real reverse proxy config

A more complete Nginx setup:

```nginx
upstream api {
  least_conn;
  server api1:3000 max_fails=3 fail_timeout=30s;
  server api2:3000 max_fails=3 fail_timeout=30s;
  server api3:3000 max_fails=3 fail_timeout=30s;
}

server {
  listen 443 ssl http2;
  server_name api.example.com;

  ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

  # gzip everything text-y
  gzip on;
  gzip_types text/plain text/css application/json application/javascript;

  # rate limit
  limit_req_zone $binary_remote_addr zone=basic:10m rate=10r/s;

  location / {
    limit_req zone=basic burst=20 nodelay;

    proxy_pass http://api;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

That's TLS termination, HTTP/2, gzip, rate limiting, load balancing, and forwarding all in one file. This is the bread and butter of running things on the open internet.

## Things to remember

- A reverse proxy is the front door of your app. It hides your backends.
- Load balancing is one of many jobs of a reverse proxy.
- Round-robin and least connections cover most cases.
- L7 LBs understand HTTP and can route smartly. L4 is dumb but fast.
- Health checks decide which backends are in rotation.
- Make backends stateless. It makes everything easier.

## Going deeper

- Nginx's official docs: https://nginx.org/en/docs/.
- Envoy's docs: https://www.envoyproxy.io/docs/. Much more powerful, used by service meshes.
- HAProxy: still the gold standard for raw load balancing performance.
- "Load Balancing" chapter of Google SRE Book: https://sre.google/sre-book/load-balancing-frontend/.
- AWS docs on ALB vs NLB: https://docs.aws.amazon.com/elasticloadbalancing/.
