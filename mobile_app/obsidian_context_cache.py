"""
Phase 1 Step 2: Obsidian Context Cache

Implements a simple in-memory cache for Obsidian context to reduce latency.
- TTL: 5 minutes (configurable)
- Invalidation on Vault changes
- Thread-safe operations
"""

from __future__ import annotations

import hashlib
import threading
import time
from datetime import datetime, timedelta
from typing import Any


class ObsidianContextCache:
    """In-memory cache for Obsidian search results with TTL and size limits."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 5 minutes)
            max_size: Maximum number of cache entries (default: 1000)
        """
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "size_limit_evictions": 0,
        }

    def _hash_key(self, query: str) -> str:
        """Generate a consistent hash key for the query."""
        return hashlib.md5(query.strip().encode()).hexdigest()

    def get(self, query: str) -> dict[str, Any] | None:
        """
        Get cached context for a query.

        Args:
            query: The search query

        Returns:
            Cached data if found and not expired, None otherwise
        """
        with self.lock:
            key = self._hash_key(query)
            entry = self.cache.get(key)

            if not entry:
                self.stats["misses"] += 1
                return None

            # Check TTL
            if datetime.now() > entry["expires"]:
                del self.cache[key]
                self.stats["evictions"] += 1
                self.stats["misses"] += 1
                return None

            # Record hit and update access time
            self.stats["hits"] += 1
            entry["accessed"] = datetime.now()
            entry["access_count"] = entry.get("access_count", 0) + 1
            return entry["data"]

    def set(self, query: str, data: dict[str, Any]) -> None:
        """
        Cache context for a query.

        Args:
            query: The search query
            data: The data to cache (dict with 'hits' and 'digest' keys)
        """
        with self.lock:
            key = self._hash_key(query)

            # Check if we need to evict old entries due to size limit
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Find and remove the least recently accessed entry
                lru_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]["accessed"],
                )
                del self.cache[lru_key]
                self.stats["size_limit_evictions"] += 1

            # Add new entry
            self.cache[key] = {
                "data": data,
                "expires": datetime.now() + timedelta(seconds=self.ttl),
                "created": datetime.now(),
                "accessed": datetime.now(),
                "access_count": 1,
            }

    def invalidate(self) -> None:
        """Invalidate all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats["invalidations"] += 1

    def invalidate_query(self, query: str) -> None:
        """Invalidate cache for a specific query."""
        with self.lock:
            key = self._hash_key(query)
            if key in self.cache:
                del self.cache[key]
                self.stats["invalidations"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                (self.stats["hits"] / total_requests * 100)
                if total_requests > 0
                else 0.0
            )
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "hit_rate_percent": hit_rate,
                "evictions": self.stats["evictions"],
                "size_limit_evictions": self.stats["size_limit_evictions"],
                "invalidations": self.stats["invalidations"],
                "total_requests": total_requests,
            }

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self.lock:
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "invalidations": 0,
            }


# Global cache instance
# Max 1000 entries, 5-minute TTL
_obsidian_context_cache = ObsidianContextCache(ttl_seconds=300, max_size=1000)


def get_cache() -> ObsidianContextCache:
    """Get the global cache instance."""
    return _obsidian_context_cache


def invalidate_global_cache() -> None:
    """Invalidate the global cache (called on Vault changes)."""
    _obsidian_context_cache.invalidate()


# ===== Integration helpers

def cached_collect_obsidian_context(
    query: str,
    *,
    collect_fn: callable,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """
    Collect Obsidian context with caching.

    Args:
        query: The search query
        collect_fn: The function to call if cache misses (collect_obsidian_context)
        limit: Number of results to return

    Returns:
        List of search hits
    """
    cache = get_cache()

    # Check cache first
    cached = cache.get(query)
    if cached:
        return cached.get("hits", [])

    # Cache miss: collect from Vault
    hits = collect_fn(query, limit=limit)

    # Store in cache for next request
    cache.set(query, {"hits": hits, "digest": {}})

    return hits


# ===== Monitoring helpers

def log_cache_stats(logger: Any) -> None:
    """Log cache statistics."""
    cache = get_cache()
    stats = cache.get_stats()
    logger.info(
        f"CACHE_STATS | "
        f"size={stats['size']} | "
        f"hits={stats['hits']} | "
        f"misses={stats['misses']} | "
        f"hit_rate={stats['hit_rate_percent']:.1f}% | "
        f"total_requests={stats['total_requests']}"
    )
