# 20. MapReduce

You have 10 TB of log files. You want to count how many users visited each page. One machine can't hold the data, let alone process it. So you spread the work across hundreds of machines.

MapReduce is the model that made that easy. Google published the paper in 2004. The world rebuilt itself around it.

The pattern is older than the name, and you don't need to use Hadoop (the original implementation) to use the ideas. They show up in Spark, Flink, BigQuery, Snowflake, and the data pipelines at every big company.

## The two steps

You write two functions. The framework handles everything else.

**Map**: take an input record, emit zero or more key-value pairs.

```python
def map(line):
    # line: "GET /home from user 42 at 10:00"
    parts = line.split()
    path = parts[1]
    yield (path, 1)
```

**Reduce**: take a key and all the values that share that key, emit the final result.

```python
def reduce(path, counts):
    yield (path, sum(counts))
```

That's it. Run this against 10 TB of logs across 1,000 machines and you get the page-view counts. The framework figures out how to split the input, run the maps, shuffle the data so the same key ends up on the same reducer, and run the reduces.

## What's actually happening behind the scenes

```
  Input (split into chunks)
  +------+ +------+ +------+ +------+
  |  C1  | |  C2  | |  C3  | |  C4  |
  +------+ +------+ +------+ +------+
      |       |        |       |
      v       v        v       v
   Map     Map      Map     Map        (run in parallel, one per chunk)
      |       |        |       |
      v       v        v       v
      (each emits key-value pairs)
      
  ===== Shuffle / sort =====
  (move records so the same key is on the same reducer)
  
      |       |        |       |
      v       v        v       v
   Reduce  Reduce   Reduce  Reduce      (one per group of keys)
      |       |        |       |
      v       v        v       v
        Output (one part per reducer)
```

Three big phases:
1. **Map** runs in parallel on the input.
2. **Shuffle** moves records around so the same key lands on the same reducer. This is where the network gets pounded.
3. **Reduce** combines values per key.

## The classic example: word count

The "hello world" of MapReduce.

```python
def map(text):
    for word in text.split():
        yield (word.lower(), 1)

def reduce(word, counts):
    yield (word, sum(counts))
```

Run on the complete works of Shakespeare:

```
("the", 28944)
("and", 27437)
("i", 22107)
...
```

Doesn't matter if the input is 1 KB or 1 PB. Same code. The framework scales it out.

## Where it shines

- **Batch analytics**: process all the logs from yesterday.
- **ETL pipelines**: extract from raw data, transform, load into a warehouse.
- **Search index building**: process the whole web, build an inverted index.
- **Machine learning preprocessing**: featurize a billion records.
- **Reporting**: nightly job that builds dashboards.

Anything where the input is huge, the computation is data-parallel, and you can wait minutes or hours for the answer.

## Where it doesn't shine

- **Real-time queries**: MapReduce jobs take minutes. For interactive analytics, use a columnar DB (BigQuery, Snowflake, ClickHouse) or a stream processor.
- **Iterative algorithms**: many ML algorithms loop over the data dozens of times. MapReduce writes intermediate results to disk between iterations. Spark fixed this by keeping data in memory.
- **Low-latency event processing**: streams, not batches. Use Kafka Streams, Flink, or similar.

This is why Spark and Flink overshadowed Hadoop MapReduce in the 2010s. Same model, but much faster.

## Spark in two minutes

Apache Spark is what most people use today when they want MapReduce-style processing without Hadoop's overhead.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("page-views").getOrCreate()

logs = spark.read.text("s3://my-bucket/logs/2026-05-26/*.log")

# tokenize, count
counts = (
    logs.rdd
    .flatMap(lambda row: row.value.split())
    .filter(lambda w: w.startswith("/"))
    .map(lambda path: (path, 1))
    .reduceByKey(lambda a, b: a + b)
)

counts.saveAsTextFile("s3://my-bucket/out/page-counts/")
```

Same map and reduce pattern. But Spark keeps data in memory across stages, so iterative jobs are 10-100x faster than vanilla Hadoop. It also has a SQL-like API:

```python
df = spark.read.parquet("s3://my-bucket/events/")
df.groupBy("page").count().show()
```

That's a MapReduce-equivalent job in one line.

## Combiners: a small optimization that matters

Some reducers are commutative and associative (like sum, max, count). For those, you can run a "mini reduce" on each mapper's output before the shuffle. That cuts down the data moved across the network.

In word count:

```python
def map(text):
    counts = {}
    for word in text.split():
        counts[word.lower()] = counts.get(word.lower(), 0) + 1
    for word, c in counts.items():
        yield (word, c)
```

Instead of emitting `("the", 1)` 10,000 times from one mapper, you emit `("the", 10000)` once. Massive bandwidth savings during shuffle.

Spark does this for you with `reduceByKey`. Hadoop calls it a combiner.

## Skew: the thing that ruins your day

If one key is way more popular than others, one reducer gets crushed. Imagine counting tweets per hashtag, and 30% of tweets are `#trump`. That reducer is doing 30% of the work.

Fixes:
- **Salt the key**: add a random suffix during map, sum up at the end.
- **Combiners**: less data to shuffle in the first place.
- **Custom partitioner**: split popular keys across multiple reducers.
- **Move to a different model**: a streaming system might handle skew better.

Almost every real MapReduce job has at least one skewed key. Look for it.

## Modern alternatives

You may not write MapReduce code directly anymore. Most teams use:

- **BigQuery, Snowflake, Redshift**: SQL on petabytes. The engine does MapReduce-like things underneath.
- **Spark with DataFrames**: same pattern, nicer API.
- **Flink, Kafka Streams**: streaming versions for continuous processing.
- **Dataflow / Beam**: Google's unified batch + streaming model.

But the ideas all come from the original MapReduce paper. If you understand map, shuffle, reduce, and skew, you can read the docs for any of these and follow along.

## A worked example: counting unique users per day

You want to know how many unique users hit each page each day. Across a billion log lines.

**Map**: emit `((page, day), user_id)`.

```python
def map(log_line):
    parsed = parse(log_line)
    yield ((parsed.path, parsed.date), parsed.user_id)
```

**Reduce**: count distinct user_ids per key.

```python
def reduce(key, user_ids):
    yield (key, len(set(user_ids)))
```

That `set()` is expensive at scale. So in practice you'd use HyperLogLog (an approximate distinct-counting algorithm) instead of an exact set. The output is 99% accurate and uses constant memory per key. Real big-data systems are full of these "give up exact answers for speed" tricks.

## Things to remember

- MapReduce: two simple functions, map and reduce, with a shuffle in between.
- The framework handles parallelism, scheduling, fault tolerance, and shuffle.
- Use it for batch processing huge data. Don't use it for interactive queries.
- Spark replaced classic Hadoop MapReduce by keeping data in memory.
- Watch out for skew, use combiners, pre-aggregate when possible.

## Going deeper

- The original MapReduce paper: https://research.google/pubs/pub62/. Surprisingly readable.
- "MapReduce: Simplified Data Processing on Large Clusters" by Dean & Ghemawat (same paper).
- Spark docs: https://spark.apache.org/docs/latest/.
- *Designing Data-Intensive Applications*, Chapter 10 (Batch Processing).
- "The Datacenter as a Computer" (Google book): https://research.google/pubs/the-datacenter-as-a-computer-an-introduction-to-the-design-of-warehouse-scale-machines/.
