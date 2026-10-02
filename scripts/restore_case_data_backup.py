#!/usr/bin/env python3
"""暗号化バックアップ（*.tar.gz.enc）を復号して、マニフェストのハッシュと SQLite を検証する。

  python scripts/restore_case_data_backup.py <archive> --out <dir>
  python scripts/restore_case_data_backup.py <archive> --out <dir> --key-prompt   # Mac を失った時（控えた鍵を入力）

元の data/ には書き戻さない。--out に展開するだけなので、中身を確かめてから手で戻す。
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backup_case_data import BackupKeyError, decode_key, load_key, restore_archive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Decrypt and verify an encrypted case data backup.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="展開先（空のディレクトリ推奨）")
    parser.add_argument("--key-prompt", action="store_true", help="キーチェーンではなく、控えた鍵を入力する")
    args = parser.parse_args()
    try:
        key = decode_key(getpass.getpass("backup key (base64): ")) if args.key_prompt else load_key()
        result = restore_archive(args.archive.expanduser(), args.out.expanduser(), key)
    except (BackupKeyError, ValueError, OSError) as exc:
        print(f"RESTORE FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
