#!/usr/bin/env python3
"""Backup case and learning data to timestamped snapshot folders.

profile:
  case_data       週次（日曜 01:30）。案件DB・係数・紫苑記憶・判断資産・Jev判定ログ・改善ログ・相談・対話ログ
  judgment_daily  日次（23:30）。判断資産の正本と Jev ラベルだけ（その日の昇格・統合を翌週まで裸にしない）

1バックアップ＝1つの暗号化アーカイブ（<prefix>_<ts>.tar.gz.enc、AES-256-GCM）。
平文はローカルの一時ディレクトリでだけ組み立て、iCloud には暗号文しか置かない。
鍵は macOS キーチェーン（service=tune-lease-backup-key）。取れなければ平文に落とさず失敗にする。
初回だけ `--init-key` で鍵を作る。復号は scripts/restore_case_data_backup.py。
成否は data/backup_status.json に profile ごとに残し、AURION CORE 朝報が読む。
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import io
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = Path(
    os.environ.get(
        "CASE_DATA_BACKUP_ROOT",
        Path.home()
        / "Library"
        / "Mobile Documents"
        / "com~apple~CloudDocs"
        / "tune_lease_55_backups"
        / "case_data",
    )
)

# 判断資産の正本と、人手で付けた Jev ラベル（再取得できない）。日次と週次の両方に入れる
JUDGMENT_DAILY_TARGETS = [
    "data/canonical_judgment_rules.json",
    "data/autoresearch_judgment_asset_candidate_state.json",
    "data/judgment_asset_merge_candidates.json",
    "data/jev_judgment_log.jsonl",
    "data/jev_*label_eval*.jsonl",
    "data/jev_*calibration*.jsonl",
    "data/jev_memory_review_human_*",
    "data/same_event_dedup_labels_*.json",
    "data/same_event_dedup_pending_*.json",
]

DEFAULT_TARGETS = [
    "data/lease_data.db",
    "data/screening_db.sqlite",
    "data/users.db",
    "data/novelist_agent.db",
    "data/math_discoveries.db",
    "data/lease_news_metrics.json",
    "data/model_review_state.json",
    "data/training_meta.json",
    "data/coeff_auto.json",
    "data/coeff_overrides.json",
    "data/ensemble_config.json",
    # data/weekly_plot.json は REV-182（2026-07-02）で廃止済み・生成元も止まっているため外した
    # 紫苑記憶システムの派生物。特に revisions（改訂宣言の真実の源）と
    # usage_log（鮮度更新の材料）は失うと再構築できない
    "data/shion_memory_index.json",
    "data/shion_memory_usage_log.jsonl",
    "data/shion_memory_revisions.jsonl",
    "data/shion_memory_health_state.json",
    "data/shion_memory_promotions.jsonl",
    # 判断資産（正本・昇格state・統合履歴・利用実績）と Jev 判定ログ・ラベル
    *JUDGMENT_DAILY_TARGETS,
    "data/autoresearch_judgment_asset_candidates.jsonl",
    "data/judgment_asset_dedup_*",
    "data/judgment_asset_merge_report_*",
    "data/judgment_asset_usage_feedback.jsonl",
    "data/judgment_asset_growth_history.jsonl",
    "data/judgment_asset_feedback_drops.jsonl",
    "data/judgment_asset_next_case_targets.json",
    "data/same_event_dedup_jev_eval_*",
    "data/jev_safe_gateway_audit.jsonl",
    # 改善ログ・相談キュー・チャットで教わったノウハウ・対話ログ
    "data/cloudrun_improvement_log.jsonl",
    "data/shion_agent_consultation_queue.jsonl",
    "data/shion_reasoner_consultation_queue.jsonl",
    "data/shion_reasoning_consultations.jsonl",
    "data/consultation_memory.jsonl",
    "data/shion_teaching_funnel.jsonl",
    "data/ai_teach_rules.json",
    "data/cloudrun_chat_log.jsonl",
    "data/chat_logs.jsonl",
]

PROFILES = {
    "case_data": {"targets": DEFAULT_TARGETS, "prefix": "case_data", "keep": 12},
    "judgment_daily": {"targets": JUDGMENT_DAILY_TARGETS, "prefix": "judgment_daily", "keep": 30},
}
STATUS_PATH = REPO_ROOT / "data" / "backup_status.json"

KEYCHAIN_SERVICE = "tune-lease-backup-key"
KEYCHAIN_ACCOUNT = "tune-lease-backup"
ARCHIVE_SUFFIX = ".tar.gz.enc"
# 形式: MAGIC + nonce(12) + AES-256-GCM(tar.gz)。MAGIC は AAD にも入れて改ざん・形式違いを弾く
MAGIC = b"TLBK1\n"
NONCE_LEN = 12
MANIFEST_NAME = "backup_manifest.json"


class BackupKeyError(RuntimeError):
    pass


def load_key() -> bytes:
    """キーチェーンから鍵（base64 の32バイト）を読む。値はログに出さない。"""
    try:
        proc = subprocess.run(
            ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BackupKeyError(f"キーチェーンを呼べない（平文バックアップは作っていない）: {type(exc).__name__}") from exc
    if proc.returncode != 0:
        raise BackupKeyError(
            f"キーチェーンから鍵を取得できない（平文バックアップは作っていない）: security exit {proc.returncode}"
        )
    return decode_key(proc.stdout.strip())


def decode_key(text: str) -> bytes:
    try:
        key = base64.b64decode(text.strip(), validate=True)
    except ValueError as exc:
        raise BackupKeyError("鍵の形式が不正（base64 ではない）") from exc
    if len(key) != 32:
        raise BackupKeyError(f"鍵の長さが不正（{len(key)}バイト、32バイトが必要）")
    return key


def init_key() -> None:
    """鍵をランダム生成してキーチェーンに入れる。値は stdin で渡し、argv・画面・ファイルに残さない。"""
    exists = subprocess.run(
        ["/usr/bin/security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT],
        capture_output=True,
    )
    if exists.returncode == 0:
        raise BackupKeyError("鍵は既にある（上書きすると過去のバックアップを復号できなくなるので中止）")
    value = base64.b64encode(secrets.token_bytes(32)).decode()
    proc = subprocess.run(
        ["/usr/bin/security", "add-generic-password", "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT,
         "-l", "tune_lease_55 backup encryption key", "-w"],
        input=f"{value}\n{value}\n", capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise BackupKeyError(f"キーチェーンへの保存に失敗: security exit {proc.returncode}")
    load_key()  # 読み戻せることを確かめる


def encrypt_bytes(key: bytes, plaintext: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(NONCE_LEN)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, plaintext, MAGIC)


def decrypt_bytes(key: bytes, blob: bytes) -> bytes:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not blob.startswith(MAGIC):
        raise ValueError("暗号化バックアップの形式ではない")
    nonce = blob[len(MAGIC): len(MAGIC) + NONCE_LEN]
    try:
        return AESGCM(key).decrypt(nonce, blob[len(MAGIC) + NONCE_LEN:], MAGIC)
    except InvalidTag as exc:
        raise ValueError("復号失敗（鍵が違うか、ファイルが壊れている）") from exc


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BackupEntry:
    source: str
    destination: str
    size_bytes: int
    method: str
    sha256: str = ""


@dataclass
class BackupSummary:
    created_at: str
    destination: str
    backed_up: list[BackupEntry]
    missing: list[str]
    removed_old: list[str]


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _archive_path(backup_root: Path, ts: str, prefix: str = "case_data") -> Path:
    base = backup_root / f"{prefix}_{ts}{ARCHIVE_SUFFIX}"
    if not base.exists():
        return base
    counter = 1
    while True:
        candidate = backup_root / f"{prefix}_{ts}_{counter}{ARCHIVE_SUFFIX}"
        if not candidate.exists():
            return candidate
        counter += 1


def _sqlite_integrity_ok(db_path: Path) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
    except sqlite3.Error:
        return False
    return bool(row and row[0] == "ok")


def _backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(src) as source_conn:
        with sqlite3.connect(dst) as dest_conn:
            source_conn.backup(dest_conn)
    if not _sqlite_integrity_ok(dst):
        raise RuntimeError(f"backup integrity check failed: {dst}")


def _backup_file(src: Path, dst: Path) -> str:
    suffix = src.suffix.lower()
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        _backup_sqlite(src, dst)
        return "sqlite_backup_api"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copy2"


def _cleanup_old_snapshots(backup_root: Path, keep: int, prefix: str = "case_data") -> list[str]:
    """世代管理は暗号化アーカイブだけ。暗号化前の平文フォルダには触らない（消すかは人が決める）。"""
    snapshots = sorted(
        [path for path in backup_root.glob(f"{prefix}_*{ARCHIVE_SUFFIX}") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for old in snapshots[keep:]:
        old.unlink()
        removed.append(str(old))
    return removed


def _expand_targets(targets: list[str]) -> tuple[list[str], list[str]]:
    """glob を展開する。1件も当たらないパターンと存在しないファイルは missing に回す。"""
    found: dict[str, None] = {}
    missing: list[str] = []
    for target in targets:
        if any(ch in target for ch in "*?["):
            matches = sorted(str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.glob(target) if p.is_file())
            if not matches:
                missing.append(target)
            found.update(dict.fromkeys(matches))
        elif (REPO_ROOT / target).exists():
            found[target] = None
        else:
            missing.append(target)
    return list(found), missing


def backup_case_data(
    backup_root: Path, targets: list[str], keep: int, prefix: str = "case_data", key: bytes | None = None
) -> BackupSummary:
    key = key if key is not None else load_key()  # 鍵が無ければ何も書かずにここで失敗
    ts = _timestamp()
    backup_root.mkdir(parents=True, exist_ok=True)
    archive = _archive_path(backup_root, ts, prefix)

    backed_up: list[BackupEntry] = []
    existing, missing = _expand_targets(targets)

    # 平文の組み立てはローカル一時領域だけ（iCloud 配下に平文を作らない）
    with tempfile.TemporaryDirectory(prefix="tlbk_") as tmp:
        stage = Path(tmp) / archive.name.removesuffix(ARCHIVE_SUFFIX)
        for rel_target in existing:
            src = (REPO_ROOT / rel_target).resolve()
            dst = stage / rel_target
            method = _backup_file(src, dst)
            backed_up.append(
                BackupEntry(
                    source=str(src),
                    destination=rel_target,
                    size_bytes=dst.stat().st_size,
                    method=method,
                    sha256=_sha256(dst),
                )
            )
        manifest = BackupSummary(
            created_at=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            destination=str(archive),
            backed_up=backed_up,
            missing=missing,
            removed_old=[],
        )
        (stage / MANIFEST_NAME).write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(stage, arcname=stage.name)

    blob = encrypt_bytes(key, buf.getvalue())
    part = archive.with_name(archive.name + ".part")
    part.write_bytes(blob)
    part.replace(archive)
    manifest.removed_old.extend(_cleanup_old_snapshots(backup_root, keep, prefix))
    return manifest


def restore_archive(archive: Path, out_dir: Path, key: bytes) -> dict:
    """復号→展開→マニフェストのサイズ・sha256 照合→SQLite integrity_check。結果を dict で返す。"""
    data = decrypt_bytes(key, archive.read_bytes())
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not (member.isfile() or member.isdir()) or not target.is_relative_to(out_dir.resolve()):
                raise ValueError(f"不正なアーカイブ要素: {member.name}")
        tar.extractall(out_dir)
    roots = [p for p in out_dir.iterdir() if (p / MANIFEST_NAME).exists()]
    if len(roots) != 1:
        raise ValueError("マニフェストが見つからない")
    root = roots[0]
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    mismatched: list[str] = []
    sqlite_failed: list[str] = []
    for entry in manifest["backed_up"]:
        path = root / entry["destination"]
        if not path.exists() or path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
            mismatched.append(entry["destination"])
        elif entry["method"] == "sqlite_backup_api" and not _sqlite_integrity_ok(path):
            sqlite_failed.append(entry["destination"])
    return {
        "restored_to": str(root),
        "files": len(manifest["backed_up"]),
        "sqlite_checked": sum(1 for e in manifest["backed_up"] if e["method"] == "sqlite_backup_api"),
        "mismatched": mismatched,
        "sqlite_failed": sqlite_failed,
        "ok": not mismatched and not sqlite_failed,
    }


def record_status(profile: str, *, ok: bool, summary: BackupSummary | None = None, error: str = "", path: Path | None = None) -> None:
    """profile ごとの最終試行・最終成功を残す（朝報が読む）。書けなくてもバックアップ自体は止めない。"""
    path = path or STATUS_PATH
    try:
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        status = {}
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    entry = dict(status.get(profile) or {})
    entry.update({"last_attempt": now, "ok": ok, "error": error[:300]})
    if ok and summary is not None:
        entry.update(
            {
                "last_success": now,
                "destination": summary.destination,
                "files": len(summary.backed_up),
                "bytes": sum(e.size_bytes for e in summary.backed_up),
                "missing": summary.missing,
            }
        )
    status[profile] = entry
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"[backup] status write failed: {exc}", file=sys.stderr)


STALE_DAYS = 7
OBSIDIAN_BACKUP_ROOT = DEFAULT_BACKUP_ROOT.parent / "obsidian"


def _obsidian_last_success(root: Path) -> dt.datetime | None:
    """Obsidian バックアップは別スクリプトなので、フォルダ名（Vault名_YYYYmmdd_HHMMSS）から最新を読む。"""
    stamps = []
    for path in root.glob("*_????????_??????") if root.exists() else []:
        try:
            stamps.append(dt.datetime.strptime("_".join(path.name.rsplit("_", 2)[-2:]), "%Y%m%d_%H%M%S"))
        except ValueError:
            continue
    return max(stamps).astimezone() if stamps else None


def morning_report_line(
    status_path: Path | None = None,
    obsidian_root: Path | None = None,
    now: dt.datetime | None = None,
) -> str:
    """バックアップの成否と最終成功日時を朝報に1行で出す。7日以上成功していない対象があれば警告。"""
    now = now or dt.datetime.now().astimezone()
    path = status_path or STATUS_PATH
    try:
        status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        status = {}
    last: dict[str, dt.datetime | None] = {}
    failed: list[str] = []
    for name in PROFILES:
        entry = status.get(name) or {}
        last[name] = dt.datetime.fromisoformat(entry["last_success"]) if entry.get("last_success") else None
        if entry and not entry.get("ok"):
            failed.append(name)
    last["obsidian"] = _obsidian_last_success(obsidian_root or OBSIDIAN_BACKUP_ROOT)
    stale = [name for name, ts in last.items() if ts is None or (now - ts).days >= STALE_DAYS]
    parts = [f"{name} {ts.strftime('%m/%d %H:%M') if ts else 'なし'}" for name, ts in last.items()]
    head = "⚠️ " if stale or failed else ""
    warn = []
    if failed:
        warn.append("直近失敗: " + ", ".join(failed))
    if stale:
        warn.append(f"{STALE_DAYS}日以上成功なし: " + ", ".join(stale))
    return f"- {head}バックアップ最終成功: " + " / ".join(parts) + (f"（{'；'.join(warn)}）" if warn else "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup case and learning data.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="case_data")
    parser.add_argument("--init-key", action="store_true", help="初回のみ: 鍵を生成してキーチェーンに保存する。")
    parser.add_argument("--backup-root", default=None, help="Snapshot root directory.")
    parser.add_argument("--keep", type=int, default=None, help="Number of snapshots to keep.")
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help="Relative path under repo to back up. Can be repeated. Defaults to core case data.",
    )
    args = parser.parse_args()
    if args.init_key:
        try:
            init_key()
        except BackupKeyError as exc:
            print(f"INIT KEY FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"鍵を生成してキーチェーンに保存した（service={KEYCHAIN_SERVICE}）。値は表示しない。")
        return 0
    profile = PROFILES[args.profile]
    # case_data の保存先（CASE_DATA_BACKUP_ROOT）と同じ階層に profile 名のフォルダを並べる
    default_root = DEFAULT_BACKUP_ROOT if args.profile == "case_data" else DEFAULT_BACKUP_ROOT.parent / args.profile

    try:
        summary = backup_case_data(
            backup_root=Path(args.backup_root or default_root).expanduser(),
            targets=args.targets or profile["targets"],
            keep=args.keep or profile["keep"],
            prefix=profile["prefix"],
        )
    except Exception as exc:  # noqa: BLE001 - 失敗を朝報に出すため記録してから非0で終わる
        record_status(args.profile, ok=False, error=f"{type(exc).__name__}: {exc}")
        print(f"BACKUP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    record_status(args.profile, ok=True, summary=summary)
    total_size = sum(entry.size_bytes for entry in summary.backed_up)
    print(
        f"CASE DATA BACKED UP (encrypted): {len(summary.backed_up)} files, "
        f"{total_size / 1024 / 1024:.1f} MB"
    )
    if summary.missing:
        print(f"MISSING: {', '.join(summary.missing)}")
    print(f"destination: {summary.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
