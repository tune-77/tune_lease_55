#!/usr/bin/env python3
"""Compatibility entrypoint for the lease-news collector."""

from __future__ import annotations

from collect_lease_news_to_obsidian import main


if __name__ == "__main__":
    raise SystemExit(main())
