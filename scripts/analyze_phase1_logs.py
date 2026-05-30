#!/usr/bin/env python3
"""
Phase 1: Performance monitoring script

Analyzes Phase 1 latency and cache statistics logs
and generates daily reports.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any


def parse_phase1_logs(log_file: Path) -> list[dict[str, Any]]:
    """Parse PHASE1_LATENCY log lines."""
    entries = []

    if not log_file.exists():
        print(f"⚠️  Log file not found: {log_file}")
        return entries

    with open(log_file, "r") as f:
        for line in f:
            if "PHASE1_LATENCY" not in line:
                continue

            # Example: PHASE1_LATENCY | obsidian_search=0.150s | obsidian_digest=0.050s | ...
            try:
                # Extract metric values
                metrics = {}

                # Parse obsidian_search
                m = re.search(r"obsidian_search=([\d.]+)s", line)
                if m:
                    metrics["obsidian_search"] = float(m.group(1))

                # Parse obsidian_digest
                m = re.search(r"obsidian_digest=([\d.]+)s", line)
                if m:
                    metrics["obsidian_digest"] = float(m.group(1))

                # Parse web_search
                m = re.search(r"web_search=([\d.]+)s", line)
                if m:
                    metrics["web_search"] = float(m.group(1))

                # Parse gemini
                m = re.search(r"gemini=([\d.]+)s", line)
                if m:
                    metrics["gemini"] = float(m.group(1))

                # Parse total
                m = re.search(r"total=([\d.]+)s", line)
                if m:
                    metrics["total"] = float(m.group(1))

                # Parse cache_status
                m = re.search(r"cache_status=(\w+)", line)
                if m:
                    metrics["cache_status"] = m.group(1)

                # Parse timestamp
                m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if m:
                    metrics["timestamp"] = m.group(1)

                if metrics:
                    entries.append(metrics)
            except Exception as e:
                print(f"⚠️  Failed to parse line: {e}")
                continue

    return entries


def analyze_logs(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze log entries and generate statistics."""
    if not entries:
        return {"error": "No entries found"}

    # Calculate statistics
    obsidian_search_times = [e.get("obsidian_search", 0) for e in entries]
    obsidian_digest_times = [e.get("obsidian_digest", 0) for e in entries]
    web_search_times = [e.get("web_search", 0) for e in entries]
    gemini_times = [e.get("gemini", 0) for e in entries]
    total_times = [e.get("total", 0) for e in entries]

    cache_statuses = [e.get("cache_status", "unknown") for e in entries]
    cache_status_counts = defaultdict(int)
    for status in cache_statuses:
        cache_status_counts[status] += 1

    def stats(times):
        """Calculate min, max, avg, p95."""
        if not times:
            return {}
        sorted_times = sorted(times)
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "p95": sorted_times[int(len(times) * 0.95)] if len(times) > 1 else sorted_times[0],
        }

    return {
        "total_requests": len(entries),
        "obsidian_search": stats(obsidian_search_times),
        "obsidian_digest": stats(obsidian_digest_times),
        "web_search": stats(web_search_times),
        "gemini": stats(gemini_times),
        "total": stats(total_times),
        "cache_status_distribution": dict(cache_status_counts),
        "cache_hit_rate_percent": (
            (cache_status_counts.get("cache_hit", 0) / len(entries) * 100)
            if entries
            else 0
        ),
    }


def parse_cache_stats(log_file: Path) -> list[dict[str, Any]]:
    """Parse CACHE_STATS log lines."""
    entries = []

    if not log_file.exists():
        return entries

    with open(log_file, "r") as f:
        for line in f:
            if "CACHE_STATS" not in line:
                continue

            try:
                metrics = {}

                # Parse size
                m = re.search(r"size=(\d+)", line)
                if m:
                    metrics["size"] = int(m.group(1))

                # Parse hits
                m = re.search(r"hits=(\d+)", line)
                if m:
                    metrics["hits"] = int(m.group(1))

                # Parse misses
                m = re.search(r"misses=(\d+)", line)
                if m:
                    metrics["misses"] = int(m.group(1))

                # Parse hit_rate
                m = re.search(r"hit_rate=([\d.]+)%", line)
                if m:
                    metrics["hit_rate_percent"] = float(m.group(1))

                # Parse total_requests
                m = re.search(r"total_requests=(\d+)", line)
                if m:
                    metrics["total_requests"] = int(m.group(1))

                if metrics:
                    entries.append(metrics)
            except Exception as e:
                print(f"⚠️  Failed to parse CACHE_STATS: {e}")
                continue

    return entries


def main():
    """Generate Phase 1 performance report."""
    log_dir = Path.home() / "Library" / "Logs" / "tunelease"

    print("\n" + "=" * 70)
    print("PHASE 1: PERFORMANCE MONITORING REPORT")
    print("=" * 70)

    # Analyze latency logs
    latency_log = log_dir / "phase1_latency.log"
    print(f"\n📊 Latency Analysis")
    print(f"   Log file: {latency_log}")

    latency_entries = parse_phase1_logs(latency_log)
    latency_stats = analyze_logs(latency_entries)

    if "error" in latency_stats:
        print(f"   {latency_stats['error']}")
    else:
        print(f"\n   Total requests: {latency_stats['total_requests']}")

        print(f"\n   Latency Breakdown:")
        print(f"   ├─ Obsidian Search:")
        for key, val in latency_stats["obsidian_search"].items():
            print(f"   │  ├─ {key:5s}: {val*1000:6.1f}ms")

        print(f"   ├─ Obsidian Digest:")
        for key, val in latency_stats["obsidian_digest"].items():
            print(f"   │  ├─ {key:5s}: {val*1000:6.1f}ms")

        print(f"   ├─ Web Search:")
        for key, val in latency_stats["web_search"].items():
            print(f"   │  ├─ {key:5s}: {val*1000:6.1f}ms")

        print(f"   ├─ Gemini API:")
        for key, val in latency_stats["gemini"].items():
            print(f"   │  ├─ {key:5s}: {val*1000:6.1f}ms")

        print(f"   └─ Total:")
        for key, val in latency_stats["total"].items():
            print(f"      ├─ {key:5s}: {val:6.3f}s")

        print(f"\n   Cache Status Distribution:")
        for status, count in latency_stats["cache_status_distribution"].items():
            pct = (count / latency_stats["total_requests"] * 100) if latency_stats["total_requests"] > 0 else 0
            print(f"   ├─ {status:20s}: {count:4d} ({pct:5.1f}%)")

        print(f"\n   Cache Hit Rate: {latency_stats['cache_hit_rate_percent']:.1f}%")

    # Analyze cache stats
    print(f"\n📊 Cache Statistics")
    cache_entries = parse_cache_stats(latency_log)

    if cache_entries:
        latest_cache = cache_entries[-1]
        print(f"\n   Latest Cache State:")
        print(f"   ├─ Size: {latest_cache.get('size', 'N/A')} entries")
        print(f"   ├─ Total Requests: {latest_cache.get('total_requests', 'N/A')}")
        print(f"   ├─ Hits: {latest_cache.get('hits', 'N/A')}")
        print(f"   ├─ Misses: {latest_cache.get('misses', 'N/A')}")
        print(f"   └─ Hit Rate: {latest_cache.get('hit_rate_percent', 'N/A'):.1f}%")

    # Assessment
    print(f"\n🎯 Assessment")
    if latency_stats.get("total_requests", 0) < 50:
        print(f"   ⚠️  Not enough data (< 50 requests). Collect more logs.")
    else:
        obsidian_time = latency_stats["obsidian_search"]["avg"] + latency_stats["obsidian_digest"]["avg"]
        cache_hit_rate = latency_stats.get("cache_hit_rate_percent", 0)

        print(f"   ✅ Obsidian latency: {obsidian_time*1000:.1f}ms avg")
        if obsidian_time < 0.5:
            print(f"      Status: ✅ GOOD (< 500ms target)")
        else:
            print(f"      Status: ⚠️  SLOW (> 500ms target)")

        print(f"   ✅ Cache hit rate: {cache_hit_rate:.1f}%")
        if cache_hit_rate >= 30:
            print(f"      Status: ✅ GOOD (>= 30% target)")
        else:
            print(f"      Status: ⚠️  LOW (< 30% target)")

    print(f"\n" + "=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
