#!/usr/bin/env python3
"""暗号化バックアップ（*.tar.gz.enc）を Cloudflare R2 にも置く（iCloud とは別系統のオフサイト）。

backup_case_data.py が iCloud への保存に成功した後に呼ぶ。送るのは MAGIC で始まる暗号化アーカイブだけ。
認証はキーチェーンの Cloudflare API トークン（service=cloudflare-api-token / account=tune-lease-55、
権限 Account / Workers R2 Storage / Edit）。新しい秘密はどこにも保存しない。
経路は api.cloudflare.com の R2 REST API。S3 互換エンドポイント（*.r2.cloudflarestorage.com）は
この Mac の回線から TLS ハンドシェイクで拒否されるため使わない（2026-10-04 確認）。
REST API は失敗でも HTTP 200 + success:false を返すことがあるので、JSON の success で判定する。

  python scripts/r2_offsite.py --list
  python scripts/r2_offsite.py --get case_data/case_data_20261004_013000.tar.gz.enc --out /tmp/x.tar.gz.enc
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backup_case_data import ARCHIVE_SUFFIX, MAGIC  # noqa: E402

BUCKET = os.environ.get("R2_BACKUP_BUCKET", "tune-lease-55-backups")
TOKEN_SERVICE = "cloudflare-api-token"
TOKEN_ACCOUNT = "tune-lease-55"
CF_API = "https://api.cloudflare.com/client/v4"
NO_SUCH_BUCKET = 10006
TIMEOUT = 300


class R2Error(RuntimeError):
    pass


@dataclass
class R2Credentials:
    account_id: str
    token: str = field(repr=False)


def _call(token: str, method: str, url: str, body: bytes | None = None, content_type: str = "") -> bytes:
    headers = {"Authorization": f"Bearer {token}", **({"Content-Type": content_type} if content_type else {})}
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        return exc.read()  # エラー本文（JSON の errors）で判定する
    except (urllib.error.URLError, OSError) as exc:
        raise R2Error(f"{method} 接続失敗: {getattr(exc, 'reason', exc)}") from None


def _json(raw: bytes, what: str) -> dict:
    """success:false なら Cloudflare のエラーコード付きで R2Error（トークンは含めない）。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise R2Error(f"{what}: JSON ではない応答") from None
    if not data.get("success"):
        codes = [e.get("code") for e in data.get("errors") or []]
        err = R2Error(f"{what}: Cloudflare errors {codes}")
        err.codes = codes  # type: ignore[attr-defined]
        raise err
    return data


def load_credentials() -> R2Credentials:
    proc = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", TOKEN_SERVICE, "-a", TOKEN_ACCOUNT, "-w"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise R2Error(f"キーチェーンに Cloudflare API トークンが無い（security exit {proc.returncode}）")
    token = proc.stdout.strip()
    account_id = os.environ.get("R2_ACCOUNT_ID", "")
    if not account_id:
        accounts = _json(_call(token, "GET", f"{CF_API}/accounts"), "accounts")["result"]
        if len(accounts) != 1:
            raise R2Error(f"アカウントが{len(accounts)}件あるので R2_ACCOUNT_ID で指定が必要")
        account_id = accounts[0]["id"]
    return R2Credentials(account_id, token)


def r2_request(creds: R2Credentials, method: str, key: str = "", query: dict | None = None, body: bytes | None = None) -> bytes:
    """key="" はバケット作成（POST）か一覧（GET）。GET でキー指定はオブジェクト本体（生バイト）を返す。"""
    base = f"{CF_API}/accounts/{creds.account_id}/r2/buckets"
    if method == "POST":
        return _call(creds.token, "POST", base, json.dumps({"name": BUCKET}).encode(), "application/json")
    url = f"{base}/{BUCKET}/objects" + (f"/{urllib.parse.quote(key, safe='/')}" if key else "")
    url += f"?{urllib.parse.urlencode(query)}" if query else ""
    return _call(creds.token, method, url, body, "application/octet-stream" if body is not None else "")


def list_objects(creds: R2Credentials, prefix: str = "", request=r2_request) -> list[dict]:
    objects: list[dict] = []
    cursor = ""
    while True:
        query = {"prefix": prefix, "per_page": 1000, **({"cursor": cursor} if cursor else {})}
        data = _json(request(creds, "GET", "", query), "list")
        objects.extend({"key": o["key"], "size": int(o.get("size") or 0)} for o in data.get("result") or [])
        info = data.get("result_info") or {}
        cursor = info.get("cursor") or ""
        if not (cursor and info.get("is_truncated")):
            return objects


def sync_archive(archive: Path, profile: str, keep: int, creds: R2Credentials | None = None, request=r2_request) -> dict:
    """1つアップロードし、同じ profile の古い世代を keep まで削り、バケット全体の合計サイズを返す。"""
    blob = archive.read_bytes()
    if not archive.name.endswith(ARCHIVE_SUFFIX) or not blob.startswith(MAGIC):
        raise R2Error(f"暗号化アーカイブではないので送らない: {archive.name}")
    creds = creds or load_credentials()
    key = f"{profile}/{archive.name}"
    try:
        _json(request(creds, "PUT", key, body=blob), "put")
    except R2Error as exc:
        if NO_SUCH_BUCKET not in getattr(exc, "codes", []):
            raise
        _json(request(creds, "POST"), "create bucket")  # 初回だけバケットを作る
        _json(request(creds, "PUT", key, body=blob), "put")
    # キー名にタイムスタンプが入っているので辞書順＝古い順
    ours = sorted(o["key"] for o in list_objects(creds, f"{profile}/", request) if o["key"].endswith(ARCHIVE_SUFFIX))
    removed = ours[:-keep] if keep > 0 else []
    for old in removed:
        _json(request(creds, "DELETE", old), "delete")
    total = sum(o["size"] for o in list_objects(creds, "", request))
    return {"bucket": BUCKET, "key": key, "bytes": len(blob), "removed": removed, "bucket_total_bytes": total}


def main() -> int:
    parser = argparse.ArgumentParser(description="R2 上の暗号化バックアップを一覧・取得する。")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--get", metavar="KEY")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        creds = load_credentials()
        if args.get:
            if not args.out:
                parser.error("--get には --out が必要")
            blob = r2_request(creds, "GET", args.get)
            if not blob.startswith(MAGIC):
                _json(blob, "get")  # エラー JSON ならここで R2Error
                raise R2Error("取得したものが暗号化アーカイブではない")
            args.out.write_bytes(blob)
            print(f"saved: {args.out}")
        else:
            for obj in list_objects(creds):
                print(f"{obj['size']:>12,}  {obj['key']}")
    except R2Error as exc:
        print(f"R2 FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
