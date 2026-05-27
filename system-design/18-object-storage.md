# 18. Object Storage

A relational DB is great for structured rows. It's terrible at storing a 4 GB video. Object storage is what you reach for instead.

If you've ever used Google Drive or Dropbox, you've used object storage. If you've ever uploaded a profile photo to a website, it probably ended up in S3.

## What it actually is

Object storage is a flat key-value store where:

- The key is a path-like string: `users/42/avatar.jpg`.
- The value is a blob: bytes, plus some metadata (content type, size, custom tags).

```
+----------------------+----------------------+--------+
|         Key          |        Value         | Size   |
+----------------------+----------------------+--------+
| users/42/avatar.jpg  | (jpeg bytes)         | 87 KB  |
| videos/abc/raw.mp4   | (mp4 bytes)          | 1.2 GB |
| backups/2026-05-26.tar | (tar bytes)        | 4.7 GB |
+----------------------+----------------------+--------+
```

That's it. No tables. No queries. No indexes (except by key prefix). Just "put this blob at this key", "get this blob by key", "list keys starting with this prefix", "delete this key".

The big providers:

| Provider | Service |
|---|---|
| AWS | S3 |
| Google Cloud | Cloud Storage (GCS) |
| Azure | Blob Storage |
| Cloudflare | R2 |
| Self-hosted | MinIO, Ceph |

They all speak roughly the same model and similar HTTP APIs.

## What it's good at

- **Huge files**: gigabytes to terabytes per object are normal.
- **Cheap**: about $0.02 per GB per month for standard S3 in 2026. Glacier (cold storage) is ~10x cheaper.
- **Durable**: AWS S3 claims eleven nines (99.999999999%) for data. That's "you will not lose this file in your lifetime" durability.
- **Massively scalable**: petabytes, trillions of objects per bucket.
- **Available**: served behind a global CDN, accessible via HTTPS from anywhere.

## What it's bad at

- **Updating part of a file**: you upload, you download. You don't seek into the middle and change a byte.
- **Per-byte latency**: a small read still costs a network round trip. Don't use it for things you read 1000 times a second.
- **Listing huge prefixes**: listing all keys under `users/` when you have 10 million users is slow.
- **Transactions**: no atomic updates across objects.

## The API in three calls

S3 SDK, Python:

```python
import boto3

s3 = boto3.client("s3", region_name="us-east-1")

# Upload
s3.put_object(
    Bucket="my-app-uploads",
    Key="users/42/avatar.jpg",
    Body=open("avatar.jpg", "rb"),
    ContentType="image/jpeg",
)

# Download
obj = s3.get_object(Bucket="my-app-uploads", Key="users/42/avatar.jpg")
data = obj["Body"].read()

# List
result = s3.list_objects_v2(Bucket="my-app-uploads", Prefix="users/42/")
for item in result["Contents"]:
    print(item["Key"], item["Size"])
```

That's almost the whole API. There's also `delete_object`, `copy_object`, and some metadata calls.

## Pre-signed URLs

A common pattern: you want users to upload directly to S3 without proxying through your server.

The server creates a "pre-signed URL" that authorizes the user to PUT to a specific key, for a short time:

```python
url = s3.generate_presigned_url(
    "put_object",
    Params={"Bucket": "my-app-uploads", "Key": "users/42/avatar.jpg"},
    ExpiresIn=300,   # 5 minutes
)
return {"upload_url": url}
```

Frontend then does:

```js
await fetch(uploadUrl, { method: "PUT", body: file });
```

Your server never touches the bytes. Great for huge files, terrible for your bandwidth bill if you didn't think of this.

## Storage classes

Real-world data isn't all equal. Some files are hit constantly. Some haven't been read in a year. S3 lets you put them in different tiers:

| Tier | Cost per GB/mo | Retrieval cost | Use case |
|---|---|---|---|
| S3 Standard | ~$0.023 | None | Hot data |
| S3 Standard-IA | ~$0.0125 | Higher | Backups read occasionally |
| S3 One Zone-IA | ~$0.01 | Higher | Non-critical backups |
| S3 Glacier Instant | ~$0.004 | Higher | Archive, sometimes needed |
| S3 Glacier Flexible | ~$0.0036 | Minutes to hours | Long-term backups |
| S3 Glacier Deep Archive | ~$0.00099 | 12+ hours | "Set and forget" |

You can move objects through tiers automatically with lifecycle rules. A typical setup: "move objects to IA after 30 days, Glacier after 90, delete after 7 years".

## Object storage as the backbone of modern systems

Look behind the curtain of many products and you'll find object storage doing the heavy lifting:

- **Netflix**: every movie sits in S3 (Open Connect CDN pulls from there).
- **Instagram**: every photo and video is in S3.
- **Dropbox**: ran on S3 for years, eventually built their own.
- **GitHub**: large files (LFS), Actions artifacts, release downloads.
- **ML training**: datasets live in S3 or GCS, fed into compute clusters.
- **Data lakes**: Parquet files in S3 queried by Athena, Spark, Snowflake.

When you see "petabyte data lake" or "object store", it's the same idea: flat namespace, blobs, cheap, durable.

## A few practical patterns

### Don't put PII in the key

The key `users/ada@example.com/avatar.jpg` leaks email addresses. Use opaque IDs: `users/8f3a-2b1c/avatar.jpg`.

### Use a CDN in front

S3 is global but not optimized for latency. Putting CloudFront (or Cloudflare's R2-with-cache) in front of your bucket is the easy speed-up.

### Versioning and lifecycle

Turn on versioning if you're storing important data. It keeps old copies when you overwrite or delete. Combine with lifecycle to delete the old versions after N days, otherwise the bucket grows forever.

### Encryption

Server-side encryption (SSE-S3 or SSE-KMS) costs nothing and avoids a class of compliance headaches. Turn it on by default.

### Don't make buckets public by accident

Famously, lots of data breaches are "public S3 bucket". Bucket policies are restrictive by default in modern AWS, but it's still a thing to double check.

## Where it sits in your architecture

```
                +-----------+
       user --> | CloudFront| (CDN)
                +-----------+
                      |
                      v
                +-----------+
                |    S3     |   (object storage)
                +-----------+
                      ^
                      | uploads via pre-signed URLs
                      |
                +-----------+
                |  App API  |   (issues URLs, stores metadata in Postgres)
                +-----------+
                      |
                      v
                +-----------+
                | Postgres  |   (users, paths, ownership)
                +-----------+
```

The DB stores "user 42 has avatar at key X". The object store stores the bytes. The CDN serves them quickly.

## Things to remember

- Object storage = flat key-value store of blobs.
- Cheap, durable, scalable. Good for files. Bad for small high-frequency reads.
- Pre-signed URLs let clients upload directly without touching your server.
- Tier old data to colder storage to save real money.
- Pair with a CDN for latency. Pair with a DB for metadata.

## Going deeper

- AWS S3 docs: https://docs.aws.amazon.com/s3/.
- "Building and operating a pretty big storage system called S3" by Andy Warfield (AWS): https://www.youtube.com/watch?v=v3HfUNQ0JOE. Excellent talk.
- The original "Dynamo: Amazon's Highly Available Key-Value Store" paper, even though it's KV not blob, the ideas are foundational.
- MinIO docs if you want S3-compatible on your own hardware: https://min.io/docs/minio/.
- Cloudflare R2 vs S3 pricing comparison (R2 has no egress fee): https://developers.cloudflare.com/r2/.
