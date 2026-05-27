# 11. CDNs

A user in Sydney loads a website hosted on a server in Virginia. The packet has to fly across the Pacific and back. That's at least 250 ms of round-trip time before the server has even read the request. For a page that needs 50 assets, you can do the math: the page is going to feel slow.

The fix is obvious: put a copy of the content close to the user. That's what a CDN (Content Delivery Network) does.

## What a CDN actually is

A CDN is a network of servers spread around the world, all serving the same content. Cloudflare, Akamai, Fastly, AWS CloudFront, and others operate them. They have hundreds of "edge locations" or "points of presence" (PoPs).

Your DNS routes users to the nearest one (GeoDNS, remember Chapter 5):

```
   User in Sydney                       User in Mumbai
        |                                    |
        v                                    v
   [ Sydney PoP ]                       [ Mumbai PoP ]
        |                                    |
        +----------- origin server -----------+
                   (your actual app)
```

The first request from a region misses and reaches your origin. The PoP caches the response and serves the next requests directly.

## What gets cached

CDNs were originally for static content: images, CSS, JS, video. Those are easy: same URL, same bytes, cache for hours or days.

Modern CDNs also cache:
- HTML pages (with shorter TTLs or revalidation).
- API responses (when safe to cache).
- Personalized content (using clever rules and cache keys).

What doesn't make sense to cache:
- User-specific data (your inbox, your account page).
- Real-time data (live chat, stock prices).
- Anything with `Cache-Control: no-store`.

## How a CDN decides what to do

The CDN sits between client and origin and follows HTTP cache headers. The two big ones:

```
Cache-Control: max-age=3600, public
```

Cache for 3600 seconds (1 hour). `public` means CDN can cache it for everyone. `private` means only the user's browser can.

```
Cache-Control: no-store
```

Never cache. Always go to origin.

ETag-based revalidation is the other lever. The CDN can ask the origin "is this still the latest?" instead of refetching the whole asset.

```
GET /image.jpg HTTP/1.1
If-None-Match: "v1abc"

HTTP/1.1 304 Not Modified
```

The 304 response has no body, just "yes, your cached copy is fine". The CDN serves the cached body.

## Push vs pull CDNs

Two models:

- **Pull** (most common): the CDN doesn't have your content until it's requested. First request from each region triggers a fetch from origin, which the CDN caches.
- **Push**: you upload assets to the CDN ahead of time. Useful for big static sites, video libraries.

Cloudflare and Fastly are pull. CloudFront supports both. For 99% of use cases, pull is what you want. It's hands-off.

## Origin shielding

When a CDN has hundreds of edges, a cold cache means hundreds of edges all hit your origin on the first request. Origin shielding adds one "shield" PoP in between that absorbs those misses. Other PoPs pull from the shield, not directly from origin.

```
            +------- edge --+
            |               |
   origin --+---- shield --+--- edge --- user
            |               |
            +------- edge --+
```

You set this up in the CDN config. Big traffic spikes will thank you for it.

## Cache invalidation

Files change. You want users to see the new version, not the cached old one.

Three ways:

**1. Versioned URLs (the best way)**: change the URL when the content changes. Bundlers do this automatically: `app.a3f8b1.js` becomes `app.e7c92d.js` after a build. Users get the new file, the old one just sits in the cache until it ages out.

**2. Cache busting query strings**: `app.js?v=2`. Same idea, uglier.

**3. Purge / invalidate**: tell the CDN to throw out a specific URL.

```bash
# Cloudflare example
curl -X POST "https://api.cloudflare.com/client/v4/zones/<zone>/purge_cache" \
  -H "Authorization: Bearer <token>" \
  -d '{"files":["https://example.com/style.css"]}'
```

Purges are slow (sometimes minutes) and have rate limits. Don't rely on them for frequent updates. Use versioned URLs as the default.

## CDNs as more than caches

Modern CDNs are full edge platforms:

- **TLS termination**: certs and HTTPS handshake at the edge, not at origin.
- **HTTP/2 and HTTP/3 termination**: even if your origin only speaks HTTP/1.1, users get the modern protocol.
- **Compression**: gzip and brotli on the fly.
- **Image optimization**: resize and convert formats on demand.
- **WAF (Web Application Firewall)**: block SQL injection, bot traffic.
- **DDoS protection**: absorb floods of malicious traffic before they reach you.
- **Edge functions**: run JS or WASM at the edge for personalization, A/B tests, simple APIs. Cloudflare Workers and Fastly Compute@Edge are the popular options.

For many projects, the CDN is now half the architecture.

## Code: a tiny Cloudflare Worker

This is the kind of code that runs at every edge PoP, microseconds from the user:

```js
export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return new Response("ok", { headers: { "Cache-Control": "no-store" } });
    }

    // pass everything else to origin, but add a header
    const res = await fetch(request);
    const out = new Response(res.body, res);
    out.headers.set("X-Edge", "hello-from-the-edge");
    return out;
  },
};
```

You deploy it to one URL, and it runs in 300+ cities at the same time. That's the value proposition.

## Things to remember

- A CDN is a network of caches close to users.
- It speeds up static assets by an order of magnitude on global traffic.
- You control it with HTTP cache headers, mostly `Cache-Control` and `ETag`.
- Versioned URLs are the best cache invalidation strategy.
- Modern CDNs do a lot more: TLS, compression, WAF, edge compute.

## Going deeper

- Cloudflare Learning, CDN section: https://www.cloudflare.com/learning/cdn/what-is-a-cdn/.
- HPBN Chapter 8 (optimizing application delivery): https://hpbn.co/.
- "How CDNs Work" by KeyCDN: https://www.keycdn.com/blog/how-cdns-work.
- Mozilla's `Cache-Control` reference: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control.
- *High Performance Browser Networking* (Chapter 8) and the AWS CloudFront documentation are the practical references.
