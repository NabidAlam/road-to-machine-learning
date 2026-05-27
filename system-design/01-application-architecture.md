# 01. Application Architecture

In the last chapter we looked at one computer. Now let's zoom out. What does a real application running on the internet actually look like?

## The simplest possible web app

A user types a URL into a browser. Something somewhere on the internet sends back HTML. That's it.

```
[ Browser ]  --- request --->  [ Server ]
            <--- HTML ---
```

This works for a personal blog. It breaks the moment you have:
- More than a handful of users
- A login system
- User-generated content
- Anything that needs to remember state

So we add pieces.

## The standard three-tier web app

This is the architecture behind 90% of products you use:

```
[ Browser / Mobile ]
        |
        v
+------------------+
|   Web Server     |   serves HTML, handles auth, routes requests
+------------------+
        |
        v
+------------------+
| Application code |   business logic, your Python/Go/Node code
+------------------+
        |
        v
+------------------+
|    Database      |   Postgres, MySQL, MongoDB
+------------------+
```

Three tiers: presentation (frontend), application (backend logic), data (database). On a small site they might all run on one box. On big sites they run on hundreds.

## Developer view vs production view

When you `git push`, your code travels through several systems before users see it.

```
You write code
      |
      v
   Git repo  (GitHub, GitLab)
      |
      v
   CI/CD     (GitHub Actions, CircleCI)  -- runs tests, builds artifact
      |
      v
   Registry  (Docker Hub, ECR)           -- stores the build
      |
      v
   Servers   (EC2, Kubernetes)           -- runs the build
      |
      v
   Users
```

The big idea: code on your laptop is not code in production. There's a whole pipeline that gets it there safely.

## Monitoring and logging (the boring but critical bit)

When something breaks at 3am, you need to know:
1. **Did it break?** (alerting)
2. **What broke?** (logging)
3. **How badly?** (metrics)

A real production app always has three companion systems:

| Tool family | Examples | What it tells you |
|---|---|---|
| Logs | ELK stack, Datadog Logs, Loki | What happened, in detail |
| Metrics | Prometheus, Grafana | How often, how fast, how many |
| Traces | Jaeger, Honeycomb | Where the time went in one request |

You don't need all three on day one. But if your app makes any money, you'll need all three eventually.

A quick example with Python logging:

```python
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

def charge_user(user_id, amount):
    log.info("charge_started", extra={"user_id": user_id, "amount": amount})
    try:
        result = stripe.charge(user_id, amount)
        log.info("charge_succeeded", extra={"user_id": user_id})
        return result
    except Exception as e:
        log.error("charge_failed", extra={"user_id": user_id, "error": str(e)})
        raise
```

In production, these logs flow into something like Datadog or Loki where you can search across all servers at once.

## The CAP triangle of an app's life

Every real product is trying to optimize three things at the same time. You can pick two:

- **Availability**: is the app up?
- **Performance**: is it fast?
- **Cost**: is it cheap?

A static site is fast and cheap but limited. Netflix is fast and available but eats billions in infrastructure. A grad student's project is cheap and (sort of) available but slow.

We'll see this trade-off again in the CAP theorem chapter, but it shows up everywhere.

## Reliability and the language of nines

When someone says "this service has four nines of availability", they mean it's up 99.99% of the time.

| Nines | Uptime | Downtime per year |
|---|---|---|
| 99% (two) | "two nines" | ~3.65 days |
| 99.9% (three) | | ~8.76 hours |
| 99.99% (four) | | ~52 minutes |
| 99.999% (five) | | ~5 minutes |
| 99.9999% (six) | | ~30 seconds |

Each extra nine costs roughly 10x more engineering effort. Most products live at three or four nines. Stock exchanges and core cloud infrastructure aim for five or six.

This is what SLAs (Service Level Agreements) are about. AWS S3 promises 99.99% availability and 99.999999999% durability (eleven nines for "we won't lose your data").

## Things to remember

- A modern web app is split into frontend, backend, and database.
- Code moves from your laptop to production through a pipeline (CI/CD).
- Production also needs monitoring: logs, metrics, traces.
- You can't be infinitely available, fast, and cheap. Pick.
- "Five nines" is jargon for 99.999% uptime.

## Going deeper

- *The Twelve-Factor App*: https://12factor.net/. Short, classic. The default playbook for cloud apps.
- Google SRE Book, Chapter 4 "Service Level Objectives": https://sre.google/sre-book/service-level-objectives/.
- AWS Well-Architected Framework: https://aws.amazon.com/architecture/well-architected/.
- Martin Fowler on logging vs metrics vs tracing: https://martinfowler.com/articles/201701-event-driven.html.
