# Memory-Bounded, Leak-Free Retrieval — Design

**Date:** 2026-06-14
**Branch context:** `refactor/public-ready` (post `a73b8ef`)
**Status:** Approved, ready for implementation planning

## Problem

On the Pi (16 GB RAM, ~13 GB usable), survivorlibrary content polluted *almost
every* answer — modern queries (React `useEffect`, Docker, devops, modern
medicine) returned old/wrong-domain survivorlibrary sources, sometimes
exclusively.

Root cause (confirmed by code reading + Pi logs):

1. **`CollectionCache.get()` evicts the current query's working set.** It
   computes `now = time.time()` once per call, so every collection added in one
   query ties on `last_used`. With `max_size` smaller than the per-query
   collection set, it evicts *within the single call*, keeping only the last
   `max_size` added.
2. **`route()` always appends `_other` last** (the unassigned-collections
   bucket). So `_other` is at the end of the load order — exactly the entries
   that survive eviction — while the correctly-routed group collections (e.g.
   `web` → `devdocs_en_react`) are evicted before retrieval queries them.
3. **`_other` is polluted**: `resegment_survivorlibrary.py --drop-old` never
   completed, leaving 7 redundant per-part collections
   (`survivorlibrary_com_en_all_2025_03_e0000{1000,1500,1500__building,2000,2500,3000,3500}_chunks`)
   that match no group pattern → they live in `_other` and were searched on
   every query.

The `--max-cache-size 2` set in the OOM fix (`a73b8ef`) unmasked (1); at the
old `max_size=8` the working set fit and the bug was hidden. Routing itself is
correct (verified in logs: React→web, Docker→devops).

The hard constraint: `Retriever.retrieve()` holds *all* queried collections in
RAM simultaneously, and collections range from a few MB (most `devdocs`) to
6.7 GB (`survivorlibrary_reference`, `askubuntu`). No single collection *count*
correctly bounds memory.

## Goals

- Modern-domain queries retrieve correct-domain content; no survivorlibrary on
  confident routes.
- No OOM (hard memory ceiling), and the current query's chosen collections are
  never silently dropped.
- Eval scores recover (once the Ollama backend is healthy — see Out of Scope).

## Design

### A. Byte-budgeted collection cache (core fix)

Replace `CollectionCache`'s count-based `max_size` with a **byte budget**
(default ~11 GB, leaving headroom under `MemoryMax=13G` plus the base process /
embedding model). `get(names)` semantics:

- Track total resident bytes across cached collections.
- The current request's `names` form a **protected working set** — never
  evicted during this call.
- Load candidates in the given priority order. Before loading one, evict LRU
  collections **not in the current working set** until it fits. If it still
  does not fit, **skip it and log** (`skipped <name> (<N> GB) — over budget`).
- Return only the collections that actually loaded.
- Edge: a single collection larger than the whole budget (none today; max
  6.7 GB) is loaded alone on a best-effort basis and logged.
- Cross-query LRU retention + the existing `evict_stale` TTL daemon stay.

This structurally prevents both failure modes: it cannot OOM (hard byte
ceiling) and cannot drop the query's real collections to keep `_other`.

### B. Collection-size source

Build a `name → index_bytes` map by reading `chroma.sqlite3`
(`collections` → `segments`) once at startup to map each collection to its
segment directory id(s), then size each collection's segment dir(s)
(`vector_db/<segment-id>/`) **lazily** on first reference. No full
151-collection startup scan; size is needed only just before a collection
becomes a load candidate, and is cached after first measurement.

### C. Routing: `_other` fallback-only

`route()` returns `_other` **only** on the below-threshold fallback branch.
Confident routes (`best_score >= route_threshold`) return their top groups with
no `_other`. Removes the leak from all correctly-routed queries while keeping a
search target for queries that match no group confidently.

### D. Retune (byte budget now guards memory)

- Raise `max_per_group` 3 → 5 (more relevant candidates considered; the byte
  budget — not a count — bounds memory).
- Keep `top_groups=2`, `route_threshold=0.20`, `MemoryMax=13G`.
- Service flag: drop `--max-cache-size`, add `--max-cache-bytes`
  (default ~11 GB / `11_000_000_000`).
- Two-stage candidate selection per query: `select_collections` ranks by
  name-relevance and caps at `max_per_group` (relevance), then the byte budget
  decides which of those actually load (memory).

### E. Data cleanup (with verification)

For each of the 7 `_e0000XXXX_chunks` leftovers (plus the
`__building` temp collection):

1. **Verify redundancy** — confirm its content is already represented in a
   topic collection (sample sources / compare counts). Do *not* assume; the
   `--drop-old` step never ran.
2. **Drop** the redundant ones via ChromaDB `delete_collection`.
3. **Re-run resegment** for any part that was never merged into a topic
   collection.

Apply drops to **both** the Mac (source-of-truth) DB and the Pi DB — deletes
are cheap and this avoids a 297 GB re-sync.

### F. Tests (TDD)

- `CollectionCache` with an injected size-fn + fake client (no real ChromaDB):
  - never evicts the current working set,
  - respects the byte budget,
  - loads prioritized candidates that fit; skips those that don't,
  - cross-query LRU evicts only non-current collections.
- `route()`: confident route excludes `_other`; below-threshold includes it
  (use injected group embeddings).
- Follow the existing 48-test suite patterns.

## Out of Scope (track separately)

- **Ollama 500 / generation timeout.** Universal `500`s during the eval are the
  LLM host (cori-desktop / Jetson Ollama) being down/overloaded — every
  `requests.post` to `/api/generate` fails (caught at `server.py:220`). Must be
  healthy to re-run the eval, but it is a separate infra fix, not retrieval.
- Re-embedding / splitting the large collections.

### Future work (not this effort)

- **Audit `_other` and fan its content into groups/routes.** After the 7
  leftovers are dropped, `_other` holds legit-but-unassigned collections
  (`scifi_stackexchange`, `openstreetmap_wiki`, …) reachable only via the
  below-threshold fallback. Adding new groups (e.g. an entertainment/sci-fi
  group, a geography/maps group) or extending existing group patterns would
  give them a *confident-route* path to the surface, so good answers aren't
  gated behind fallback. Periodic `_other` audits as the corpus grows.

## Success Criteria

- React → React/devdocs; devops → serverfault/devdocs; no survivorlibrary on
  confident routes.
- No OOM; `NRestarts` stays 0 under a full eval; no working-set drops.
- Eval scores recover once Ollama is back.

## Verification

Implement + unit tests green → redeploy package to Pi → drop verified-redundant
leftovers → (Ollama healthy) re-run the eval groups → confirm correct-domain
retrieval and memory stability.
