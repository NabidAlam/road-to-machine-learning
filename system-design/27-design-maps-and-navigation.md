# 27. Design a Maps and Navigation Service

Google Maps, Apple Maps, OpenStreetMap, Waze. A user types "coffee near me," sees a map with pins, picks one, gets walking directions, and a live ETA that updates with traffic.

This problem is unusual in this course: it's mostly about **how data is laid out on disk**, not how services talk. Tiles, indices, and graphs.

## Clarify

| Question                             | Example answer    |
| ------------------------------------ | ----------------- |
| Static map tiles?                    | Yes               |
| Search (places)?                     | Yes               |
| Driving / walking / transit routing? | Driving + walking |
| Real-time traffic?                   | Yes               |
| Turn-by-turn navigation?             | Yes               |
| Offline maps?                        | Out of scope      |

## Estimate

- **Daily map views:** 10B tile requests
- **Daily route requests:** 200M
- **Search QPS peak:** 100K
- **Map data:** 100M roads, 200M POIs (points of interest), updated daily

Tile views dominate by two orders of magnitude. The CDN does almost all the heavy lifting.

## High-level design

```
[ client app ]
    |
    +-- tiles -----------> [ CDN ] -> [ Tile origin ]
    |
    +-- search ----------> [ Search API ] -> [ Elasticsearch / Postgres + PostGIS ]
    |
    +-- routing ---------> [ Routing API ] -> [ Road graph + traffic store ]
    |
    +-- navigation ------> [ Nav service ] -> WebSocket updates
```

Each capability is its own service, with very different storage needs.

## Deep dive 1: Map tiles

A "tile" is a 256×256-pixel PNG (or vector blob) of a small geographic area at one zoom level.

```
URL pattern:  /tiles/{z}/{x}/{y}.png
             z = zoom (0 = world, 20 = building)
             x, y = tile coordinates
```

There are about 4^20 ≈ a trillion tiles at max zoom. You don't pre-render all of them. Most are ocean or empty desert. Pre-render the popular ones; generate the rest on demand and cache.

Tiles are static-ish (regenerated when map data changes). Perfect for the CDN. Versioned URLs (`/tiles/v17/{z}/{x}/{y}.png`) make invalidation trivial (Chapter 11).

## Deep dive 2: Geospatial search

"Find coffee shops within 500 m of me" is a geospatial range query.

**Two indexing approaches** that real systems use:

1. **Geohash.** Encode `(lat, lng)` into a short string where nearby points share prefixes. `gbsuv7zt` means "in central London." Range queries become string prefix scans, which any DB does fast.
2. **S2 cells / Quadtree.** Recursive subdivision of the globe. Used by Google internally. Each cell has an ID; queries map a radius to a cover of cells, then filter.

Postgres with PostGIS or Elasticsearch with `geo_point` will give you good range queries out of the box at most scales.

```sql
-- PostGIS
SELECT id, name
FROM places
WHERE ST_DWithin(location, ST_MakePoint(-0.13, 51.51), 500)
  AND category = 'coffee_shop'
ORDER BY ST_Distance(location, ST_MakePoint(-0.13, 51.51));
```

For a billion POIs, shard by geohash prefix (Chapter 16).

## Deep dive 3: Routing

The road network is a **graph**: intersections are nodes, road segments are edges weighted by travel time.

Classical algorithms (Dijkstra, A*) work on small graphs. Continent-scale graphs require **preprocessing**:

- **Contraction Hierarchies (CH).** Pre-compute shortcut edges. A query that would visit a million nodes now visits a few thousand. Open-source implementations: OSRM, GraphHopper.
- **Customizable Route Planning (CRP).** Used at Google. Allows turn restrictions, multiple vehicle types, and traffic changes without redoing all preprocessing.

A routing service typically holds the whole road graph in memory per region. Query latency goal: under 100 ms for cross-country routes.

## Deep dive 4: Real-time traffic

Live ETAs adjust based on current speed on each road segment.

```
client phone --GPS samples--> [ Ingestion ] --> Kafka
                                                  |
                                                  v
                                      [ Aggregator: speed per segment ]
                                                  |
                                                  v
                                          [ Traffic store
                                            segment_id -> avg_speed ]
                                                  ^
                                                  |
                                         [ Routing service reads ]
```

The aggregator buckets GPS samples by road segment and time window (e.g., last 5 minutes). The routing service multiplies edge weights by the live ratio (`current_speed / free_flow_speed`) before searching.

Predictive traffic (what the speed *will* be in 20 minutes) is an ML problem with the same data pipeline. See Chapter 30.

## Deep dive 5: Turn-by-turn navigation

Once the user starts driving:

- Open a WebSocket to a nav service.
- Client streams GPS samples.
- Server checks: are you on the planned route? If not, **reroute** (cheap with CH).
- Push ETA updates and next-turn instructions back.

The same pipeline produces anonymized traffic data. Your trip improves everyone else's ETA.

## Deep dive 6: Map updates

Map data changes daily: new roads, new businesses, road closures. The pipeline:

1. Ingest changes (OSM diffs, business listings, satellite imagery).
2. Update the master graph.
3. Re-run CH/CRP preprocessing (incremental in modern systems).
4. Re-render affected tiles.
5. Bump tile version → CDN starts serving new ones.

## Things to remember

- Tiles are static files behind a CDN. Versioned URLs.
- Geospatial search uses geohash, S2, or PostGIS. Same idea: turn 2D space into something an index can scan.
- Routing on continents needs preprocessing (CH/CRP), not raw Dijkstra.
- Real-time traffic = stream of GPS → segment-level aggregation → into the edge weights.
- Turn-by-turn is a WebSocket plus rerouting.

## Going deeper

- OSRM and GraphHopper open-source projects. Read their docs.
- Google's "Customizable Route Planning" paper.
- PostGIS tutorials and the S2 geometry library (https://s2geometry.io).
- "How Maps Work" by Mapbox engineering blog.
