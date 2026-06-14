from unittest.mock import MagicMock
from kiwix_rag.server import CollectionCache


def make_cache(sizes, max_bytes):
    client = MagicMock()
    client.get_collection.side_effect = lambda n: f"COL:{n}"
    return CollectionCache(client, max_bytes=max_bytes, size_fn=lambda n: sizes[n])


def test_loads_all_when_within_budget():
    cache = make_cache({"a": 2, "b": 3}, max_bytes=10)
    got = cache.get(["a", "b"])
    assert got == {"a": "COL:a", "b": "COL:b"}


def test_skips_collections_over_budget_keeps_working_set():
    # Budget fits 2 of these 3 same-call collections; the LAST is skipped,
    # the earlier (working-set) ones are NOT evicted.
    cache = make_cache({"a": 5, "b": 5, "c": 5}, max_bytes=10)
    got = cache.get(["a", "b", "c"])
    assert set(got) == {"a", "b"}
    assert "c" not in got


def test_cross_query_evicts_non_current():
    cache = make_cache({"a": 6, "b": 6}, max_bytes=8)
    cache.get(["a"])            # a resident
    got = cache.get(["b"])      # b needs room -> a (non-current) evicted
    assert got == {"b": "COL:b"}
    assert "a" not in cache._cache


def test_single_collection_over_budget_loads_alone():
    cache = make_cache({"big": 99}, max_bytes=10)
    got = cache.get(["big"])
    assert got == {"big": "COL:big"}


def test_resident_never_exceeds_budget_for_multi_query():
    cache = make_cache({"a": 6, "b": 6, "c": 6}, max_bytes=8)
    cache.get(["a"])
    cache.get(["b"])
    cache.get(["c"])
    assert cache._resident_bytes() <= 8


def test_cache_hit_refreshes_last_used():
    import time
    cache = make_cache({"a": 5, "b": 5}, max_bytes=8)
    cache.get(["a"])
    old_ts = cache._cache["a"]["last_used"]
    time.sleep(0.01)
    cache.get(["a"])   # cache hit must refresh the timestamp
    assert cache._cache["a"]["last_used"] > old_ts


def test_eviction_order_prefers_oldest():
    import time
    cache = make_cache({"a": 6, "b": 6, "c": 6}, max_bytes=12)
    cache.get(["a"])
    time.sleep(0.01)
    cache.get(["b"])         # b newer than a
    got = cache.get(["c"])   # needs room; should evict the older (a), keep b
    assert got == {"c": "COL:c"}
    assert "a" not in cache._cache
    assert "b" in cache._cache
