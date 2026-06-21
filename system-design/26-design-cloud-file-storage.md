# 26. Design a Cloud File Storage Service

Google Drive, Dropbox, iCloud, OneDrive. A user has folders, files, version history, sharing, search, and offline sync. The same file is edited on three devices and somehow shows up correctly on all of them.

The hard part isn't the storage. It's **sync**.

## Clarify

| Question                | Example answer                    |
| ----------------------- | --------------------------------- |
| Max file size?          | 50 GB                             |
| Concurrent device sync? | Yes, multiple devices per user    |
| Folder structure?       | Yes, nested                       |
| Sharing?                | User-to-user and link-based       |
| Version history?        | Keep last 30 days                 |
| Conflict resolution?    | Keep both versions, mark conflict |
| Search?                 | File names; full-text later       |

## Estimate

- **Users:** 100M, 30M active daily
- **Files per user:** 5,000 average
- **Total files:** 500B
- **Average file size:** 1 MB → 500 PB total
- **Writes per second peak:** 100K small uploads + 1K large ones
- **Reads per second peak:** 500K downloads

Five hundred petabytes is the scale at which a file system pretends to be a database.

## High-level design

```
[ client (desktop sync, web, mobile) ]
                |
                v
          [ API gateway ]
                |
   +------------+------------+
   v            v            v
[ Metadata ] [ Upload   ] [ Notification
   service ]   service ]    service ]
   |            |             |
   v            v             v
[ Postgres ]  [ Object   ]  [ WebSocket
              storage ]      gateway ]
              chunks
```

Two storage layers, deliberately split:
- **Metadata** in Postgres (or sharded relational): file names, folder tree, permissions, version pointers, sharing.
- **Bytes** in object storage (Chapter 18): the actual file contents, chunked.

## Deep dive 1: Chunking and deduplication

Big files are split into fixed-size chunks (usually 4 MB).

```
file "report.pdf" (50 MB)
   -> chunk_0  (sha256 = a1b2...)
   -> chunk_1  (sha256 = c3d4...)
   -> ...
   -> chunk_12 (sha256 = e5f6...)
```

Each chunk is named by its hash. **Same chunk uploaded twice = stored once.** This is content-addressable storage.

```
files table:
  file_id, name, folder_id, owner_id, current_version_id

versions table:
  version_id, file_id, created_at, size

chunks table:
  version_id, chunk_index, chunk_hash

blobs (object store):
  key = chunk_hash, value = bytes
```

Benefits:
- **Resumable uploads.** Re-upload only missing chunks.
- **Cheap version history.** A new version that changed 1 chunk costs 4 MB, not 50 MB.
- **Cross-user dedup.** A million people uploading the same PDF? One copy on disk.

## Deep dive 2: Sync protocol

Each user's "drive" is a tree. The client cares about: **what changed since I last synced?**

Use a **monotonically increasing change ID** per user.

```
client: GET /changes?since=98123
server: [
  { id: 98124, type: "create", path: "/notes.txt", version_id: ... },
  { id: 98125, type: "delete", path: "/old.docx" }
]
```

On the server, every metadata mutation appends to an event log per user. Clients pull, apply locally, and store the new high-water mark.

For live updates (other devices), a WebSocket tells the client "you have changes." The client then pulls.

## Deep dive 3: Conflict resolution

Two devices edit the same file offline. They both push.

Strategies:

| Strategy              | Behavior                                      | Where used              |
| --------------------- | --------------------------------------------- | ----------------------- |
| Last-writer-wins      | Newer timestamp overwrites                    | Simple, lossy           |
| Keep-both             | Both versions saved; one renamed `(conflict)` | Common in file sync     |
| Operational transform | Edits merged op by op                         | Live collaborative docs |
| CRDTs                 | Conflict-free merge by design                 | Modern collab tools     |

For file sync, **keep-both** is the safest default. The user can compare and pick. Live collaborative editing (Google Docs–style) needs OT/CRDTs and is a different beast. Out of scope here.

## Deep dive 4: Sharing and permissions

Permissions live in metadata.

```
permissions table:
  resource_id, principal_id, role  ('owner', 'editor', 'viewer')
```

Shared links are signed URLs (Chapter 18 pre-signed URLs, but for the metadata layer):

```
GET /share/9aJ2x...   -> server verifies token + scope -> grants read
```

A revoked link must stop working immediately. Cache invalidation on permission change is mandatory (Chapter 10).

## Deep dive 5: Search

For names, an index in Postgres or Elasticsearch keyed by `(owner_id, name)` is enough.

For full-text search across file contents, run an async pipeline:
1. On new chunk: detect content type.
2. If `pdf`/`docx`/`txt`: extract text.
3. Index into Elasticsearch with `(owner_id, file_id, text)`.

Don't try to do this synchronously during upload.

## Deep dive 6: Multi-region

Object storage is global. Metadata is the hard part.

- Pin a user to a home region for writes.
- Read replicas in other regions for fast reads.
- Cross-region writes for shared files are eventual (Chapter 17. AP).

Users barely notice if a file shared by a friend in another region takes 2 seconds to appear.

## Things to remember

- Split metadata (relational) and bytes (object store).
- Chunk big files, name chunks by hash, store each unique chunk once.
- Sync uses a per-user change log + a `since` cursor. Pull, apply, advance.
- Default conflict policy is keep-both; live collab is a different system.
- Permissions and sharing live in metadata; signed links carry scope and expiry.
- Full-text search runs async, never inline with upload.

## Going deeper

- Dropbox's "Magic Pocket" blog series on building their own storage.
- Google Drive API docs on change tokens and revisions.
- "Content-addressable storage" and the Rabin–Karp chunking variants.
- *Designing Data-Intensive Applications* Chapter 5 (replication) and 9 (consistency).
