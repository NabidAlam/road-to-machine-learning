# 25. Design a Video Streaming Platform

YouTube, Vimeo, TikTok, Netflix. Users upload video, the system processes it, the world streams it. Sounds like one product. It's actually three: upload, processing, and playback. Each has its own bottleneck.

## Clarify

| Question                  | Example answer               |
| ------------------------- | ---------------------------- |
| User-uploaded or curated? | User-uploaded                |
| Live or on-demand?        | On-demand for v1; live later |
| Resolutions?              | 240p to 4K                   |
| Comments / likes?         | Yes                          |
| Subscriptions?            | Yes                          |
| Mobile and TV clients?    | Yes                          |
| Region availability?      | Global                       |

## Estimate

- **Daily uploads:** 500K videos averaging 100 MB → 50 TB/day in source
- **Each transcoded into 6 renditions** → ~300 TB/day stored
- **Daily watch hours:** 1B → ~5 Tbps peak egress
- **Storage growth:** ~100 PB/year before any retention policy

Numbers like these mean three things: object storage from day one, CDN from day one, and asynchronous processing for everything that isn't the playback path.

## High-level design

```
                            +-------------+
   uploader  -->  API  -->  | Object store|   raw video
                            +------+------+
                                   |
                                   v
                     [ "video.uploaded" event on Kafka ]
                                   |
                +------------------+--------------------+
                v                  v                    v
       [ Transcoder workers ] [ Thumbnail gen ] [ Captions / ML moderation ]
                |                  |                    |
                v                  v                    v
            +-------------+   +-----------+      +--------------+
            | Object      |   | Thumbnails|      |  Captions DB |
            | store       |   +-----------+      +--------------+
            | (renditions)|
            +------+------+
                   |
                   v
              [   CDN   ]
                   ^
                   |
            viewer player
```

The upload and processing pipeline lives off the hot path. The playback path is **player → CDN → object store**. Most viewers never hit your origin.

## Deep dive 1: Upload

Don't proxy a 4 GB upload through your API server. Use a **pre-signed URL** from object storage (Chapter 18):

```
1. Client: POST /uploads/init   -> server returns upload ID + pre-signed URL
2. Client: PUT raw bytes directly to S3 (chunked + resumable)
3. Client: POST /uploads/{id}/complete
4. Server publishes "video.uploaded" event
```

Resumable upload (S3 multipart, GCS resumable) so a mobile user with flaky wifi doesn't lose 30 minutes of work.

## Deep dive 2: Transcoding

A 1080p video must be cut into 240p, 360p, 480p, 720p, 1080p, 4K. And into chunks of 4–10 seconds. And packaged for HLS or DASH (adaptive streaming).

```
input.mp4
  -> ffmpeg -> 240p chunks (.ts)
  -> ffmpeg -> 360p chunks
  -> ffmpeg -> ... 1080p, 4K
  -> generate .m3u8 (HLS) or .mpd (DASH) manifest
```

This is parallelizable per rendition and per chunk. Run it on a worker fleet you can scale to thousands of cores when a viral upload hits. AWS MediaConvert, GCP Transcoder API, or your own Kubernetes-scheduled ffmpeg pool.

A video is "watchable" when at least one rendition is done. Don't make users wait for 4K.

## Deep dive 3: Playback

The player downloads a manifest:

```
#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=480x270
240p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=854x480
480p/index.m3u8
...
```

It picks a rendition based on current bandwidth, fetches chunks, switches up or down as the network changes. This is **adaptive bitrate streaming (ABR)**. The magic that makes the video keep playing even when wifi gets bad.

Every chunk is a static file on the CDN. Versioned URLs (`/v/abc/720p/00012.ts`) so caching is trivial.

## Deep dive 4: Storage tiering

Old, rarely-watched videos cost real money to keep on hot storage.

```
Upload day:    S3 Standard (hot)
After 30d:     S3 Standard-IA (infrequent access)
After 180d:    S3 Glacier Instant Retrieval
After 1y:      S3 Glacier Deep Archive (cheap, restore in hours)
```

Promote back to hot if traffic spikes (someone tweets an old video).

## Deep dive 5: Metadata

Posts, titles, views, comments, likes. None of this is in object storage. It's a relational store.

```sql
CREATE TABLE videos (
    id            BIGINT PRIMARY KEY,
    uploader_id   BIGINT,
    title         TEXT,
    description   TEXT,
    duration_s    INT,
    status        TEXT,         -- 'processing', 'live', 'failed'
    view_count    BIGINT DEFAULT 0,
    created_at    TIMESTAMPTZ
);
```

Shard by `video_id`. View counts: do not update Postgres on every play. Buffer counts in Redis or Kafka and flush every minute (Chapter 19).

## Deep dive 6: Recommendations

Out of scope for the storage and serving system, but worth a sentence: every play, like, and watch-time event flows through Kafka into a feature store and a model service. See Chapter 30.

## Things to remember

- Three sub-systems: upload, processing, playback. Don't mix them.
- Use pre-signed URLs for upload; raw bytes never touch your API.
- Transcode into many renditions; package as HLS/DASH for adaptive streaming.
- The CDN serves chunks; origin almost never gets hit on the hot path.
- Tier old videos to colder storage tiers automatically.
- Aggregate view counts; never `UPDATE videos SET views = views + 1` per play.

## Going deeper

- Netflix's "Open Connect" CDN blog posts.
- "How Video Streaming Works" by Mux and Bitmovin.
- HLS spec and DASH spec for the manifest formats.
- AWS Elemental MediaConvert and GCP Transcoder API docs.
- Chapter 11 (CDNs) and Chapter 18 (Object Storage) here.
