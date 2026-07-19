# 00. Computer Architecture

Before we talk about servers and load balancers, let's zoom in on a single computer. A server is just a fancy computer. If you understand how one machine handles work, the rest of system design feels less magical.

## The four layers of storage

A computer has a strict hierarchy of where it keeps data. Closer to the CPU means faster but smaller. Farther away means slower but cheaper and bigger.

```
            +------------------+
   fastest  |   CPU registers  |  ~1 ns,    KBs
            +------------------+
            |    L1 / L2 / L3  |  ~1-10 ns,  MBs
            +------------------+
            |       RAM        |  ~100 ns,   GBs
            +------------------+
            |    SSD / NVMe    |  ~100 us,   TBs
            +------------------+
     slow   |    Hard Disk     |  ~10 ms,    TBs
            +------------------+
```

A few sanity-check numbers worth memorizing:

| Layer              | Read time        | Capacity   | Notes                          |
| ------------------ | ---------------- | ---------- | ------------------------------ |
| L1 cache           | 1 ns             | ~64 KB     | per core                       |
| RAM                | 100 ns           | 16-256 GB  | volatile                       |
| SSD                | 100 microseconds | 1-4 TB     | survives reboot                |
| HDD                | 10 ms            | 4-20 TB    | spinning rust, slow random reads |
| Network round trip | 1-100 ms         | n/a        | depends on distance            |

Reading from RAM is roughly 100,000 times faster than reading from a hard disk. That's the gap a cache buys you.

## Why this matters for system design

When you put data in Redis, what you're really saying is "keep this in RAM so I don't pay the disk tax". When someone says "we serve from cache", they mean "the answer is sitting in RAM closer to the CPU than the database".

When you hear "this query is slow", nine times out of ten it's because the database has to leave RAM and go to disk.

## RAM vs disk: what's the difference really

RAM is volatile. Pull the plug, it's gone. It's fast because there are no moving parts and it's wired close to the CPU.

Disk is persistent. Even if the machine reboots, the data is still there. But traditional disks have a physical arm that needs to swing to the right spot. SSDs got rid of the arm. Which is why they're ~100x faster than HDDs.

Most databases (Postgres, MySQL, MongoDB) keep frequently-accessed data in RAM and write to disk for safety. It's the best of both worlds.

## The CPU and what cores really mean

A modern server CPU has somewhere between 4 and 128 cores. A core is essentially a mini-CPU that can run one task at a time.

```
+----------------------------------+
|    CPU (8 cores)                 |
|  +-----+ +-----+ +-----+ +-----+ |
|  |Core1| |Core2| |Core3| |Core4| |
|  +-----+ +-----+ +-----+ +-----+ |
|  +-----+ +-----+ +-----+ +-----+ |
|  |Core5| |Core6| |Core7| |Core8| |
|  +-----+ +-----+ +-----+ +-----+ |
+----------------------------------+
```

More cores means you can do more things in parallel. But not everything benefits from parallelism. If task B needs the result of task A, you can't speed it up by adding cores.

This is also why Node.js. Which is single-threaded by default, sometimes feels limiting. One core does the work, the other 7 sit idle. You fix this with multiple processes (PM2 cluster mode, Kubernetes replicas, etc.).

## Moore's Law (and why we stopped caring)

For 40 years, CPUs got faster every 18 months. Now they mostly don't, because we've hit physical limits. Instead, chips ship more cores.

Implication for system design: you can't just buy a faster server. You need to design software that uses multiple cores, and multiple servers.

This is the whole reason horizontal scaling is the default in modern systems. We added machines because we couldn't add clock speed anymore.

## Vertical vs horizontal scaling

You'll hear this constantly. Two flavors of scaling:

**Vertical (scale up)**: bigger machine. More RAM, more cores, faster SSD. Easy to do. Hits a ceiling. Single point of failure.

**Vertical scaling**:
```
[ server 4 core ] -> [ server 16 core ]
```

**Horizontal (scale out)**: more machines. No ceiling. Way more complexity (now you need load balancers, replication, etc.).

**Horizontal scaling**:
```
[ server ] -> [ server ] [ server ] [ server ]
```

The realistic answer is "both". Start by vertical scaling because it's free engineering effort. When that runs out, you go horizontal.

## A quick code thing: cache locality

Even within one program, the memory hierarchy matters. Compare these two ways of summing a 2D array in Python (works the same in C, Java, etc.):

```python
N = 10_000
matrix = [[1] * N for _ in range(N)]

# Row-major (fast): walk memory in order
total = 0
for row in matrix:
    for x in row:
        total += x

# Column-major (slow): jump around in memory
total = 0
for col in range(N):
    for row in range(N):
        total += matrix[row][col]
```

The first is much faster, often 5-10x, even though it's the same number of additions. The CPU's cache loads memory in chunks. If your access pattern matches that, you stay in L1. If it jumps around, you keep going back to RAM.

System design version of this rule: keep related data together. It's why we denormalize tables, batch network calls, and group writes.

## Things to remember

- Memory hierarchy: registers > cache > RAM > SSD > HDD > network.
- Going to RAM is ~100,000x faster than going to disk.
- A server has many cores. Plan to use them, or you waste hardware.
- Vertical scaling is easy and limited. Horizontal scaling is hard and unlimited.

## Going deeper

- *Computer Architecture: A Quantitative Approach* by Hennessy & Patterson. The textbook.
- Brendan Gregg's "Systems Performance" book and his blog: https://www.brendangregg.com/. Famous for the latency numbers chart.
- "Latency Numbers Every Programmer Should Know" by Jeff Dean: https://gist.github.com/jboner/2841832.
- Crash Course on YouTube has a great series on CPU architecture if you like videos.
