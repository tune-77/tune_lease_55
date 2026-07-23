#!/usr/bin/env python3
"""紫苑の未完了調査タスク(shion_pending_tasks)を日次で整理するメンテナンス。

放置された pending を expired へ降格し、done/expired 履歴を上限まで刈り込む
（実体は lease_intelligence_pending.reconcile_pending）。対話経由でも自己修復されるが、
会話が途切れた環境では走らないため、日次パイプラインからも明示的に回して件数膨張を防ぐ。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lease_intelligence_pending import PENDING_PATH, reconcile_pending


def main() -> int:
    tasks = reconcile_pending()
    counts = Counter(str(t.get("status")) for t in tasks if isinstance(t, dict))
    print(
        "reconcile_pending: "
        f"path={PENDING_PATH} "
        f"total={len(tasks)} "
        f"pending={counts.get('pending', 0)} "
        f"expired={counts.get('expired', 0)} "
        f"done={counts.get('done', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
