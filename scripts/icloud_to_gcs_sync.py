"""iCloud → GCS 差分アップロードスクリプト（タイプ2ファイル用）

Obsidian Vault の *.md ファイルを GCS に差分アップロードする。
ローカルファイルの mtime をカスタムメタデータ `local_mtime` として GCS に保存し、
次回実行時に比較して変化があるファイルのみアップロードする。

Usage:
    python scripts/icloud_to_gcs_sync.py
"""
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Optional

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from runtime_paths import resolve_obsidian_vault  # noqa: E402
from obsidian_query import list_vault_md_files  # noqa: E402

GCS_BUCKET = os.environ.get("GCS_BUCKET", "tune-lease-55-data")
GCS_VAULT_PREFIX = os.environ.get("GCS_VAULT_PREFIX", "vault/")
GCS_UPLOAD_BACKEND = os.environ.get("GCS_UPLOAD_BACKEND", "gcloud").lower()
# LOCAL_VAULT_DIR は明示指定用に残す。未指定なら runtime_paths の解決結果
# （env → iCloud）を使い、GCS に上がる Vault と RAG の索引先を一致させる。
LOCAL_VAULT_DIR = os.environ.get("LOCAL_VAULT_DIR") or str(resolve_obsidian_vault())
INCLUDED_REL_PREFIXES = tuple(
    item.strip().strip("/")
    for item in os.environ.get(
        "GCS_VAULT_INCLUDED_PREFIXES",
        ",".join(
            [
                "リース知識",
                "Projects/tune_lease_55/Research",
                "Projects/tune_lease_55/News",
                "Projects/tune_lease_55/Asset Knowledge",
                "Projects/tune_lease_55/Asset Finance",
                "Projects/tune_lease_55/Lease Intelligence/Public",
                "Projects/tune_lease_55/Industry",
                "Projects/tune_lease_55/Judgment Assets",
                "05-クリップ_記事/業界リスクニュース",
                "05-クリップ_記事/リースニュース",
            ]
        ),
    ).split(",")
    if item.strip()
)
EXCLUDED_REL_PREFIXES = tuple(
    item.strip().strip("/")
    for item in os.environ.get(
        "GCS_VAULT_EXCLUDED_PREFIXES",
        ",".join(
            [
                "Daily",
                "Private Reflection",
                "チャット記録",
                "Codex",
                "Projects/tune_lease_55/Cloud SQL Summaries",
                "Projects/tune_lease_55/Cloud Run Inputs",
            ]
        ),
    ).split(",")
    if item.strip()
)


# gcloud バックエンドは GCS メタデータの事前照会ができず全件再アップロードになるため、
# 前回アップロード時の mtime をローカル状態ファイルに覚えて未変更ファイルをスキップする
STATE_FILE = Path(__file__).parent / ".sync_state_icloud_gcs.json"


def _load_upload_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def _save_upload_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    except OSError as exc:
        print(f"警告: アップロード状態ファイルを書けませんでした: {exc}", file=sys.stderr)


def _local_mtime_str(path: Path) -> str:
    """ローカルファイルの mtime を ISO 形式文字列で返す。"""
    return str(path.stat().st_mtime)


def _gcs_local_mtime(blob) -> Optional[str]:
    """GCS オブジェクトのカスタムメタデータ `local_mtime` を返す。なければ None。"""
    meta = blob.metadata or {}
    return meta.get("local_mtime")


def upload_file(
    bucket,
    local_path: Path,
    vault_dir: Path,
    gcs_prefix: str,
) -> str:
    """単一ファイルを差分チェックしてアップロードする。戻り値は操作の説明文字列。"""
    rel = local_path.relative_to(vault_dir)
    gcs_path = gcs_prefix + rel.as_posix()
    local_mtime = _local_mtime_str(local_path)

    blob = bucket.blob(gcs_path)
    try:
        blob.reload()
        gcs_mtime = _gcs_local_mtime(blob)
    except Exception:
        gcs_mtime = None

    if gcs_mtime is not None and gcs_mtime == local_mtime:
        return f"[SKIP]  {rel}"

    blob.metadata = {"local_mtime": local_mtime}
    try:
        blob.upload_from_filename(str(local_path))
    except Exception as exc:
        if not _upload_with_gcloud(local_path, bucket.name, gcs_path, local_mtime):
            raise exc
    return f"[UP]    {rel} → gs://{bucket.name}/{gcs_path}"


_access_token_cache: dict[str, str] = {}


def _get_access_token(force_refresh: bool = False) -> str:
    """ログイン済み gcloud CLI からアクセストークンを取得する（プロセス内キャッシュ）。"""
    if force_refresh or "token" not in _access_token_cache:
        token = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        _access_token_cache["token"] = token
    return _access_token_cache["token"]


def _upload_with_gcloud(local_path: Path, bucket_name: str, gcs_path: str, local_mtime: str) -> bool:
    """ADCが壊れている環境では、ログイン済み gcloud CLI のトークンで GCS JSON API に直接アップロードする。

    `gcloud storage cp` / `gsutil cp` はどちらも、ソース・宛先のいずれかに
    `[5440]` のような角括弧を含むパスがあるとワイルドカードパターンとして
    誤解釈し、バックスラッシュエスケープを試みても「該当URLなし」エラーで
    失敗する（例: 業界リスクニュースの銘柄コード表記のファイル名で実際に発生）。
    JSON API の multipart アップロードはオブジェクト名を単なる文字列として
    扱いパターン解釈しないため、この問題を回避できる。
    """
    boundary = uuid.uuid4().hex
    metadata_json = json.dumps(
        {"name": gcs_path, "metadata": {"local_mtime": local_mtime}}, ensure_ascii=False
    )
    body = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{metadata_json}\r\n"
        f"--{boundary}\r\n"
        "Content-Type: text/markdown; charset=UTF-8\r\n\r\n"
    ).encode("utf-8") + local_path.read_bytes() + f"\r\n--{boundary}--".encode("utf-8")

    for attempt, force_refresh in enumerate([False, True]):
        try:
            token = _get_access_token(force_refresh=force_refresh)
            resp = requests.post(
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket_name}/o",
                params={"uploadType": "multipart"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": f"multipart/related; boundary={boundary}",
                },
                data=body,
                timeout=60,
            )
        except Exception as fallback_exc:
            print(
                f"警告: GCS REST アップロードも失敗しました: {type(fallback_exc).__name__} {fallback_exc}",
                file=sys.stderr,
            )
            return False

        if resp.status_code == 200:
            return True
        if resp.status_code == 401 and attempt == 0:
            continue
        print(
            f"警告: GCS REST アップロードも失敗しました: HTTP {resp.status_code} {resp.text[:300]}",
            file=sys.stderr,
        )
        return False
    return False


def _matches_prefix(rel: str, prefixes: tuple[str, ...]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in prefixes)


def collect_md_files(vault_dir: Path) -> list[Path]:
    """Cloud Run AIチャットに渡してよい Obsidian *.md だけを列挙する。"""
    result = []
    for p in list_vault_md_files(vault_dir):
        if ".obsidian" in p.parts:
            continue
        rel = p.relative_to(vault_dir).as_posix()
        if EXCLUDED_REL_PREFIXES and _matches_prefix(rel, EXCLUDED_REL_PREFIXES):
            continue
        if INCLUDED_REL_PREFIXES and not _matches_prefix(rel, INCLUDED_REL_PREFIXES):
            continue
        result.append(p)
    return result


def main() -> None:
    vault_dir = Path(LOCAL_VAULT_DIR)
    print(f"iCloud → GCS 同期開始: {vault_dir} → gs://{GCS_BUCKET}/{GCS_VAULT_PREFIX}")

    md_files = collect_md_files(vault_dir)
    print(f"対象ファイル数: {len(md_files)}")
    print(f"許可prefix: {', '.join(INCLUDED_REL_PREFIXES) or '(all)'}")
    print(f"除外prefix: {', '.join(EXCLUDED_REL_PREFIXES) or '(none)'}")

    if not md_files:
        print("アップロード対象ファイルなし。正常終了。")
        return

    bucket = None
    if GCS_UPLOAD_BACKEND != "gcloud":
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)

    upload_state = _load_upload_state() if GCS_UPLOAD_BACKEND == "gcloud" else {}
    uploaded = 0
    skipped = 0
    try:
        for path in md_files:
            if GCS_UPLOAD_BACKEND == "gcloud":
                rel = path.relative_to(vault_dir)
                local_mtime = _local_mtime_str(path)
                if upload_state.get(rel.as_posix()) == local_mtime:
                    result = f"[SKIP]  {rel}"
                else:
                    gcs_path = GCS_VAULT_PREFIX + rel.as_posix()
                    if not _upload_with_gcloud(path, GCS_BUCKET, gcs_path, local_mtime):
                        raise RuntimeError(f"gcloud upload failed: {rel}")
                    upload_state[rel.as_posix()] = local_mtime
                    result = f"[UP]    {rel} → gs://{GCS_BUCKET}/{gcs_path}"
            else:
                assert bucket is not None
                result = upload_file(bucket, path, vault_dir, GCS_VAULT_PREFIX)
            print(result)
            if result.startswith("[UP]"):
                uploaded += 1
            else:
                skipped += 1
    finally:
        # 途中失敗でも成功済み分は記録し、次回は残りだけ再アップロードする
        if GCS_UPLOAD_BACKEND == "gcloud":
            _save_upload_state(upload_state)

    print(f"完了: アップロード {uploaded} 件 / スキップ {skipped} 件")


if __name__ == "__main__":
    main()
