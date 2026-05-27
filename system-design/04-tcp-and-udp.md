# 04. TCP and UDP

Two computers want to send packets to each other. They need to agree on the rules. The two protocols that 99% of the internet uses are TCP and UDP.

TCP is the careful one. UDP is the fast one. Picking the right one is one of those decisions that bakes itself into your architecture for years, so it's worth understanding the trade.

## TCP: reliable, ordered, slow

TCP guarantees three things:

1. **Delivery**: every packet arrives. If it doesn't, it's resent.
2. **Order**: packets are reassembled in the order they were sent.
3. **No duplicates**: each packet is delivered once.

To do this, TCP starts every conversation with a handshake.

```
Client                                Server
  |                                     |
  | ----- SYN (let's talk) -----------> |
  |                                     |
  | <---- SYN-ACK (sure, ok) ---------- |
  |                                     |
  | ----- ACK (great) ----------------> |
  |                                     |
  |   connection established            |
  |                                     |
  | ----- data --------------------->   |
  | <---- data ---------------------    |
```

That's three round trips before the first byte of real data. If the server is 100 ms away, you pay 300 ms before you've sent anything useful. This is fine for normal web traffic. It's a problem for things like games.

Every packet TCP sends is also acknowledged. If it doesn't hear back fast enough, it resends. This is what makes TCP "reliable", and also why it can stall: one missing packet holds up everything behind it.

**TCP is used for**: HTTP, HTTPS, SSH, email (SMTP/IMAP), most databases. Anything where wrong data is worse than slow data.

## UDP: fast, fire-and-forget, lossy

UDP does almost nothing. It just sends a packet and walks away.

```
Client                                Server
  |                                     |
  | ----- packet 1 ------------------> |
  | ----- packet 2 ------------------> |  (lost in transit, oh well)
  | ----- packet 3 ------------------> |
```

No handshake. No ordering. No retries. No acknowledgments. The packet either arrives or it doesn't, and the application has to handle the difference.

Why use it then? Three reasons:

1. **Speed**: no handshake, no waiting for ACKs.
2. **No head-of-line blocking**: one lost packet doesn't stop the others.
3. **Small overhead**: simpler headers, fewer bytes per packet.

**UDP is used for**: DNS, video calls, online games, live streaming, DNS. Anything where stale data is worse than missing data.

If you're watching a livestream and one frame drops, you'd rather skip it than rewind. That's UDP's whole pitch.

## Side by side

| Feature | TCP | UDP |
|---|---|---|
| Connection | Yes (handshake) | No |
| Reliability | Guaranteed | None |
| Order | Preserved | None |
| Speed | Slower | Faster |
| Header size | 20 bytes min | 8 bytes |
| Used by | HTTP, SSH, DB | DNS, video, games |

## Code: a tiny example in Python

A TCP echo server:

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TCP
s.bind(("0.0.0.0", 9000))
s.listen()

while True:
    conn, addr = s.accept()
    data = conn.recv(1024)
    print("got:", data, "from", addr)
    conn.sendall(data)  # echo back
    conn.close()
```

A UDP version:

```python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
s.bind(("0.0.0.0", 9000))

while True:
    data, addr = s.recvfrom(1024)
    print("got:", data, "from", addr)
    s.sendto(data, addr)
```

The TCP version has the concept of a "connection" (`accept`, `conn.send`, `conn.close`). The UDP version doesn't. You just receive datagrams from anyone.

Test it:

```bash
# Talk to the TCP server
nc 127.0.0.1 9000

# Talk to the UDP server
nc -u 127.0.0.1 9000
```

## TCP's congestion control (briefly)

When the internet gets crowded, TCP backs off. It uses an algorithm called congestion control to slow down sending until things clear up. UDP doesn't do this, which is why a poorly written UDP app can swamp a network.

You probably don't need to know the specific algorithms (Tahoe, Reno, Cubic, BBR), but it's good to know they exist. When someone says "TCP is fair to other traffic", this is what they mean.

## HTTP and the move to HTTP/3

For most of the web's life, HTTP ran on TCP. HTTP/2 still does. But HTTP/3, the newest version, runs on **QUIC**, which is built on UDP.

Why the change? TCP's head-of-line blocking and slow handshake hurt the modern web, where pages have hundreds of small requests. QUIC keeps the reliability ideas TCP had, but builds them on top of UDP so it can do parallel streams without one slow packet stopping everything else.

So even when you're using UDP, you can have reliability. It's just done at a higher layer instead of in the kernel.

## When to use which

The decision usually answers itself:

- **Order and accuracy matter**: TCP. (Bank transactions, file downloads, web pages.)
- **Speed and freshness matter**: UDP. (Live video, online games, voice calls, DNS lookups.)

Real systems mix both. A video call uses UDP for the video stream and TCP for the chat sidebar.

## Things to remember

- TCP guarantees delivery and order. Pays for it with handshake and retries.
- UDP sends and forgets. Faster, but the app must handle loss.
- TCP for "must arrive correctly". UDP for "must arrive fast".
- HTTP/3 is changing this by putting reliability into UDP via QUIC.

## Going deeper

- *TCP/IP Illustrated, Vol 1* by Stevens. Old but still the gold standard.
- HPBN Chapter 2 (TCP) and Chapter 3 (UDP): https://hpbn.co/.
- Google's QUIC design doc: https://www.chromium.org/quic/.
- Cloudflare on HTTP/3 and QUIC: https://blog.cloudflare.com/http3-the-past-present-and-future/.
