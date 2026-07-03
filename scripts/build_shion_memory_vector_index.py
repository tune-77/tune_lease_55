"""紫苑記憶索引をChromaDBベクトルコレクションへ同期する（ハイブリッド想起用）。

前提: `scripts/build_shion_memory_index.py` で data/shion_memory_index.json を
生成済みであること。chromadb / sentence-transformers が必要（ローカル環境向け）。
有効化は環境変数 `SHION_MEMORY_HYBRID=1`（/api/chat 側の想起が自動でベクトル併用になる）。

使い方:
    python3 scripts/build_shion_memory_index.py
    python3 scripts/build_shion_memory_vector_index.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_INDEX = REPO_ROOT / "data" / "shion_memory_index.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶のベクトル索引を構築")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    from api.shion_memory_vector import is_available, sync_from_index

    if not args.index.exists():
        print(f"記憶索引がありません: {args.index}")
        print("先に scripts/build_shion_memory_index.py を実行してください")
        return 1

    summary = sync_from_index(args.index)
    for key, value in summary.items():
        print(f"{key}={value}")
    if summary.get("synced", 0) == 0 and summary.get("available", 0) > 0:
        print("同期できませんでした（chromadb / sentence-transformers の導入を確認）")
        return 1
    print(f"vector_search_ready={is_available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
