# 03. Networking Basics

Two computers want to talk. How does that actually work?

In this chapter we'll go from IP addresses to ports to packets. No deep CS theory, just the stuff you need to read a network diagram.

## IP addresses

Every machine on a network has an address. On the internet, that's an IP address.

Two formats:

- **IPv4**: four numbers, each 0-255. Like `142.250.80.46` (Google).
- **IPv6**: longer, hex-y. Like `2607:f8b0:4004:c1b::6a`.

We ran out of IPv4 addresses around 2012 because there are only about 4 billion of them. IPv6 has, for practical purposes, infinite. Adoption is slow but rising.

You can find yours:

```bash
# Your public IP
curl ifconfig.me

# Your private IP (on your laptop)
ipconfig getifaddr en0   # macOS
ip addr show              # Linux
```

You'll see your laptop has something like `192.168.1.20`. That's a **private IP**, only meaningful inside your home network. The address that the rest of the internet sees you as is your public IP, which your router has.

## Public vs private addresses

Certain ranges of IPv4 are reserved as "private". They never go on the public internet:

- `10.0.0.0` to `10.255.255.255`
- `172.16.0.0` to `172.31.255.255`
- `192.168.0.0` to `192.168.255.255`

Your home router gives every device on the WiFi a private address. When you visit google.com, your router translates that to its single public IP using NAT (Network Address Translation). Without NAT, the internet would have run out of addresses in the 90s.

## Packets

You don't send a whole video or webpage in one shot. The network chops it into small chunks called **packets**, each typically ~1,500 bytes.

Each packet carries a header that says:

```
+--------------------+-------------------+
|   from: 1.2.3.4    |   to: 5.6.7.8     |
+--------------------+-------------------+
|   protocol: TCP    |   port: 443       |
+--------------------+-------------------+
|   sequence: 17     |                   |
+--------------------+-------------------+
|              payload (data)            |
+--------------------+-------------------+
```

Packets can take different routes, arrive out of order, or get lost. Whether you care depends on the protocol (TCP cares, UDP doesn't, more on that next chapter).

## Ports

An IP gets you to the machine. A port gets you to the program.

Imagine an apartment building. The IP is the street address. The port is the apartment number.

A machine has 65,535 ports. Some are reserved by convention:

| Port | Used by |
|---|---|
| 22 | SSH |
| 25 | SMTP (email) |
| 53 | DNS |
| 80 | HTTP |
| 443 | HTTPS |
| 3306 | MySQL |
| 5432 | Postgres |
| 6379 | Redis |
| 27017 | MongoDB |

You can check who is listening on what:

```bash
# macOS / Linux
lsof -iTCP -sTCP:LISTEN -n -P

# Linux alternative
ss -tlnp
```

When your browser connects to `https://google.com`, it really connects to `142.250.80.46:443`. Port 443 means HTTPS, which means a TLS-encrypted HTTP conversation.

## The OSI model (and why you can mostly ignore it)

If you ask a textbook, you get this:

```
7. Application      (HTTP, gRPC)
6. Presentation     (encryption, encoding)
5. Session          (managing the connection)
4. Transport        (TCP, UDP)
3. Network          (IP)
2. Data link        (Ethernet, WiFi)
1. Physical         (cables, radio waves)
```

In practice you'll deal with three layers as a software engineer:

- **L7 (application)**: HTTP, WebSockets, gRPC. The stuff your code touches.
- **L4 (transport)**: TCP, UDP. Decides reliability vs speed.
- **L3 (network)**: IP. Just gets the packet to the right machine.

When someone says "L7 load balancer", they mean it understands HTTP and can route by URL. An L4 load balancer just shuffles TCP connections without knowing what's inside.

## Public vs private networks (in the cloud)

Same idea as your home WiFi, but at company scale.

```
                       Internet
                          |
                          v
                  +----------------+
                  |    Gateway     |
                  +----------------+
                          |
        +-----------------+-----------------+
        |                                   |
  +-----------+                       +-----------+
  | Web tier  |                       | Web tier  |   <- public subnet
  +-----------+                       +-----------+
        |                                   |
        +-----------------+-----------------+
                          |
                  +----------------+
                  |   App servers  |  <- private subnet, no public IP
                  +----------------+
                          |
                  +----------------+
                  |    Database    |  <- private subnet
                  +----------------+
```

The database has no internet IP. The only way to reach it is from inside the network. That's the default in any cloud (AWS VPC, GCP VPC, Azure VNet). Forgetting this is how you accidentally expose a database to the public internet and end up on the news.

## A quick test

Pretend you typed `https://github.com` into your browser. What happens, in order?

1. Browser asks the OS "what's the IP of github.com?". OS asks DNS. (Chapter 5.)
2. DNS replies with `140.82.114.4`.
3. Browser opens a TCP connection to `140.82.114.4:443`. (Chapter 4.)
4. They do a TLS handshake (encryption).
5. Browser sends an HTTP request. (Chapter 6.)
6. Server sends HTML back.
7. Browser parses, makes more requests for images, JS, CSS.
8. You see the page.

Each numbered step is a chapter or two in this course. You can already see the whole picture.

## Things to remember

- IP gets you to the machine. Port gets you to the program.
- Packets are ~1,500 bytes, sent independently, possibly out of order.
- IPv4 is running out, IPv6 is the future, NAT is the duct tape between them.
- L4 = transport (TCP/UDP). L7 = application (HTTP).
- In the cloud, databases sit in a private subnet. They shouldn't have a public IP.

## Going deeper

- *Computer Networking: A Top-Down Approach* by Kurose & Ross. The textbook everyone learns from.
- High Performance Browser Networking (free online): https://hpbn.co/. Ilya Grigorik. Excellent.
- Cloudflare Learning Center: https://www.cloudflare.com/learning/. Short, clear.
- `tcpdump` and Wireshark. If you've never watched real packets fly past, do it once.
