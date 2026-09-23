#!/usr/bin/env python3
"""ChromaDBのGCSスナップショットをCloud Run起動時に復元する。

api/knowledge/chroma_snapshot.py の薄いCLIラッパー。
scripts/restore_lease_db_snapshot.py と同じ位置づけ
（start_api_cloud_run.sh がバンドル復元・uvicorn起動の前に呼ぶ）。
復元できなくても起動は止めない（api/main.py側の既存フル索引にフォールバックする）。
"""

from __future__ import annotations


def main() -> None:
    from api.knowledge.chroma_snapshot import restore

    result = restore()
    print(f"[RestoreChromaSnapshot] {result}")


if __name__ == "__main__":
    main()
