#!/usr/bin/env python3
"""暗号化バックアップ（*.tar.gz.enc）を Cloudflare R2 にも置く（iCloud とは別系統のオフサイト）。

backup_case_data.py が iCloud への保存に成功した後に呼ぶ。送るのは MAGIC で始まる暗号化アーカイブだけ。
認証はキーチェーンの Cloudflare API トークン（service=cloudflare-api-token / account=tune-lease-55）。
S3 互換キーはトークンから導出する（Access Key=トークンID、Secret=トークン値の SHA-256。Cloudflare 公式の規則）
ので、新しい秘密はどこにも保存しない。依存は標準ライブラリだけ（SigV4 を自前で署名）。

  python scripts/r2_offsite.py --list
  python scripts/r2_offsite.py --get case_data/case_data_20261004_013000.tar.gz.enc --out /tmp/x.tar.gz.enc
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backup_case_data import ARCHIVE_SUFFIX, MAGIC  # noqa: E402

BUCKET = os.environ.get("R2_BACKUP_BUCKET", "tune-lease-55-backups")
TOKEN_SERVICE = "cloudflare-api-token"
TOKEN_ACCOUNT = "tune-lease-55"
CF_API = "https://api.cloudflare.com/client/v4"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
TIMEOUT = 300


class R2Error(RuntimeError):
    pass


@dataclass
class R2Credentials:
    account_id: str
    access_key: str
    secret_key: str


def _cf_get(token: str, path: str) -> dict:
    req = urllib.request.Request(CF_API + path, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise R2Error(f"Cloudflare API {path} HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError) as exc:
        raise R2Error(f"Cloudflare API {path} 接続失敗: {getattr(exc, 'reason', exc)}") from None


def load_credentials() -> R2Credentials:
    """トークン値はこの関数の外に出さない（ログ・例外メッセージにも入れない）。"""
    proc = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", TOKEN_SERVICE, "-a", TOKEN_ACCOUNT, "-w"],
        capture_output=True, text=True, timeout=30,
    )
    if proc.returncode != 0:
        raise R2Error(f"キーチェーンに Cloudflare API トークンが無い（security exit {proc.returncode}）")
    token = proc.stdout.strip()
    token_id = _cf_get(token, "/user/tokens/verify")["result"]["id"]
    account_id = os.environ.get("R2_ACCOUNT_ID", "")
    if not account_id:
        accounts = _cf_get(token, "/accounts")["result"]
        if len(accounts) != 1:
            raise R2Error(f"アカウントが{len(accounts)}件あるので R2_ACCOUNT_ID で指定が必要")
        account_id = accounts[0]["id"]
    return R2Credentials(account_id, token_id, hashlib.sha256(token.encode()).hexdigest())


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode(), hashlib.sha256).digest()


def s3_request(creds: R2Credentials, method: str, path: str, query: dict | None = None, body: bytes = b"") -> bytes:
    """R2 の S3 互換 API を AWS SigV4（region=auto）で呼ぶ。失敗は S3 のエラーコード付きで R2Error。"""
    host = f"{creds.account_id}.r2.cloudflarestorage.com"
    amz_date = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    scope = f"{amz_date[:8]}/auto/s3/aws4_request"
    payload_hash = hashlib.sha256(body).hexdigest()
    quote = lambda s: urllib.parse.quote(str(s), safe="-_.~")  # noqa: E731
    uri = urllib.parse.quote(path, safe="/-_.~")
    qs = "&".join(f"{quote(k)}={quote(v)}" for k, v in sorted((query or {}).items()))
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    signed = ";".join(sorted(headers))
    canonical = "\n".join(
        [method, uri, qs, "".join(f"{k}:{headers[k]}\n" for k in sorted(headers)), signed, payload_hash]
    )
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
    k = ("AWS4" + creds.secret_key).encode()
    for part in (amz_date[:8], "auto", "s3", "aws4_request"):
        k = _hmac(k, part)
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={creds.access_key}/{scope}, SignedHeaders={signed}, "
        f"Signature={hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()}"
    )
    url = f"https://{host}{uri}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, data=body if method == "PUT" else None, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        try:
            code = ET.fromstring(exc.read()).findtext("Code") or ""
        except ET.ParseError:
            code = ""
        raise R2Error(f"R2 {method} HTTP {exc.code} {code}".strip()) from None
    except (urllib.error.URLError, OSError) as exc:
        raise R2Error(f"R2 {method} 接続失敗: {getattr(exc, 'reason', exc)}") from None


def list_objects(creds: R2Credentials, prefix: str = "", request=s3_request) -> list[dict]:
    objects: list[dict] = []
    token = ""
    while True:
        query = {"list-type": "2", "prefix": prefix, **({"continuation-token": token} if token else {})}
        root = ET.fromstring(request(creds, "GET", f"/{BUCKET}", query))
        for item in root.iter(f"{S3_NS}Contents"):
            objects.append({"key": item.findtext(f"{S3_NS}Key"), "size": int(item.findtext(f"{S3_NS}Size") or 0)})
        token = root.findtext(f"{S3_NS}NextContinuationToken") or ""
        if not token:
            return objects


def sync_archive(archive: Path, profile: str, keep: int, creds: R2Credentials | None = None, request=s3_request) -> dict:
    """1つアップロードし、同じ profile の古い世代を keep まで削り、バケット全体の合計サイズを返す。"""
    blob = archive.read_bytes()
    if not archive.name.endswith(ARCHIVE_SUFFIX) or not blob.startswith(MAGIC):
        raise R2Error(f"暗号化アーカイブではないので送らない: {archive.name}")
    creds = creds or load_credentials()
    key = f"{profile}/{archive.name}"
    try:
        request(creds, "PUT", f"/{BUCKET}/{key}", body=blob)
    except R2Error as exc:
        if "NoSuchBucket" not in str(exc):
            raise
        request(creds, "PUT", f"/{BUCKET}")  # 初回だけバケットを作る
        request(creds, "PUT", f"/{BUCKET}/{key}", body=blob)
    # キー名にタイムスタンプが入っているので辞書順＝古い順
    ours = sorted(o["key"] for o in list_objects(creds, f"{profile}/", request) if o["key"].endswith(ARCHIVE_SUFFIX))
    removed = ours[:-keep] if keep > 0 else []
    for old in removed:
        request(creds, "DELETE", f"/{BUCKET}/{old}")
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
            args.out.write_bytes(s3_request(creds, "GET", f"/{BUCKET}/{args.get}"))
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
