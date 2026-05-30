#!/usr/bin/env python3
"""
Phase 1 Step 2: Test - Obsidian Context Caching

Tests the caching mechanism:
1. Cache hit/miss behavior
2. TTL expiration
3. Cache invalidation
4. Statistics tracking
"""

import sys
import time
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_caching():
    """Test basic caching operations"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic caching operations")
    print("=" * 60)

    from mobile_app.obsidian_context_cache import ObsidianContextCache

    cache = ObsidianContextCache(ttl_seconds=2)

    # Test 1: Cache miss
    result = cache.get("test query")
    assert result is None, "Should return None on cache miss"
    print("✅ Cache miss returns None")

    # Test 2: Cache set and get
    data = {"hits": [{"path": "note1", "snippet": "text"}], "digest": "summary"}
    cache.set("test query", data)
    result = cache.get("test query")
    assert result == data, "Should return cached data"
    print("✅ Cache set/get works")

    # Test 3: Different queries have different cache entries
    cache.set("another query", {"hits": [], "digest": ""})
    result = cache.get("test query")
    assert result == data, "First query should still be cached"
    result = cache.get("another query")
    assert result["hits"] == [], "Second query should be separate"
    print("✅ Multiple cache entries work")

    # Test 4: TTL expiration
    time.sleep(2.1)  # Wait for TTL to expire
    result = cache.get("test query")
    assert result is None, "Should expire after TTL"
    print("✅ TTL expiration works")

    return True


def test_statistics():
    """Test cache statistics"""
    print("\n" + "=" * 60)
    print("TEST 2: Cache statistics")
    print("=" * 60)

    from mobile_app.obsidian_context_cache import ObsidianContextCache

    cache = ObsidianContextCache(ttl_seconds=3600)

    # Reset stats
    cache.clear()

    # Generate hits and misses
    cache.set("query1", {"hits": [1, 2, 3]})
    cache.get("query1")  # Hit
    cache.get("query1")  # Hit
    cache.get("query2")  # Miss
    cache.get("query2")  # Miss
    cache.get("query3")  # Miss

    stats = cache.get_stats()

    assert stats["hits"] == 2, f"Expected 2 hits, got {stats['hits']}"
    assert stats["misses"] == 3, f"Expected 3 misses, got {stats['misses']}"
    assert stats["size"] == 1, f"Expected 1 entry, got {stats['size']}"
    hit_rate = stats["hit_rate_percent"]
    assert 39 < hit_rate < 41, f"Expected ~40% hit rate, got {hit_rate}%"

    print(f"✅ Statistics tracking works:")
    print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"   Hit rate: {hit_rate:.1f}%")

    return True


def test_invalidation():
    """Test cache invalidation"""
    print("\n" + "=" * 60)
    print("TEST 3: Cache invalidation")
    print("=" * 60)

    from mobile_app.obsidian_context_cache import ObsidianContextCache

    cache = ObsidianContextCache(ttl_seconds=3600)

    # Set some data
    cache.set("query1", {"hits": [1, 2]})
    cache.set("query2", {"hits": [3, 4]})

    assert cache.get("query1") is not None, "Should be cached"
    print("✅ Data cached")

    # Invalidate all
    cache.invalidate()
    assert cache.get("query1") is None, "Should be cleared"
    assert cache.get("query2") is None, "Should be cleared"
    print("✅ Global invalidation works")

    # Set data again
    cache.clear()  # Reset stats
    cache.set("query1", {"hits": [1]})
    cache.set("query2", {"hits": [2]})
    cache.set("query3", {"hits": [3]})

    # Invalidate one query
    cache.invalidate_query("query2")
    assert cache.get("query1") is not None, "Other queries should remain"
    assert cache.get("query2") is None, "Specific query should be cleared"
    assert cache.get("query3") is not None, "Other queries should remain"
    print("✅ Selective invalidation works")

    return True


def test_thread_safety():
    """Test thread-safe operations"""
    print("\n" + "=" * 60)
    print("TEST 4: Thread safety")
    print("=" * 60)

    from mobile_app.obsidian_context_cache import ObsidianContextCache
    import threading

    cache = ObsidianContextCache(ttl_seconds=3600)
    errors = []

    def worker(thread_id):
        try:
            for i in range(100):
                query = f"thread_{thread_id}_query_{i}"
                cache.set(query, {"hits": [i]})
                result = cache.get(query)
                assert result is not None, f"Cache miss in thread {thread_id}"
        except Exception as e:
            errors.append(e)

    # Create multiple threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors in threads: {errors}"
    print(f"✅ Thread-safe operations work (500 operations completed)")

    return True


def test_hash_consistency():
    """Test that hash function is consistent"""
    print("\n" + "=" * 60)
    print("TEST 5: Hash consistency")
    print("=" * 60)

    from mobile_app.obsidian_context_cache import ObsidianContextCache

    cache = ObsidianContextCache(ttl_seconds=3600)

    # Same query with different whitespace should hash to the same value
    query1 = "what is lease risk"
    query2 = "  what is lease risk  "

    cache.set(query1, {"hits": [1, 2, 3]})

    # Should find cache entry with normalized query
    result = cache.get(query2)
    assert result is not None, "Should find cache with different whitespace"
    assert result["hits"] == [1, 2, 3], "Should return correct data"
    print("✅ Hash consistency works (whitespace normalized)")

    return True


def main():
    print("\n" + "=" * 60)
    print("PHASE 1 STEP 2: CACHING MECHANISM - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Basic caching", test_basic_caching),
        ("Statistics tracking", test_statistics),
        ("Cache invalidation", test_invalidation),
        ("Thread safety", test_thread_safety),
        ("Hash consistency", test_hash_consistency),
    ]

    results = []
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✅ Phase 1 Step 2: Caching mechanism verified!")
        print("\nNext steps:")
        print("1. Integrate caching into chat_assistant.py")
        print("2. Monitor cache hit rate in production")
        print("3. Proceed to Step 3: Auto index update")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
