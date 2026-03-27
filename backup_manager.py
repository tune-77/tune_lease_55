"""
データバックアップモジュール。
lease_data.db / coeff_overrides.json / coeff_auto.json を data/backups/ に日次保存。
MAX_GENS 世代（デフォルト7日分）を保持し、それ以前は自動削除する。
"""
from __future__ import annotations
import os
import shutil
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_SCRIPT_DIR, "data")
_BACKUP_DIR = os.path.join(_DATA_DIR, "backups")

MAX_GENS = 7  # 保持する世代数

_TARGETS = [
    (os.path.join(_DATA_DIR, "lease_data.db"),         "lease_data.db"),
    (os.path.join(_DATA_DIR, "novelist_agent.db"),      "novelist_agent.db"),
    (os.path.join(_DATA_DIR, "coeff_overrides.json"),   "coeff_overrides.json"),
    (os.path.join(_DATA_DIR, "coeff_auto.json"),        "coeff_auto.json"),
    (os.path.join(_DATA_DIR, "training_meta.json"),     "training_meta.json"),
]


# ─────────────────────────────────────────────────────────────────────────────
# バックアップ実行
# ─────────────────────────────────────────────────────────────────────────────

def run_backup(force: bool = False) -> dict:
    """
    バックアップを実行する。

    Args:
        force: True の場合、当日バックアップ済みでも強制実行。

    Returns:
        {"backed_up": [...], "skipped": [...], "ts": "YYYYMMDD"}
    """
    os.makedirs(_BACKUP_DIR, exist_ok=True)
    today   = datetime.now().strftime("%Y%m%d_%H%M")
    day_str = today[:8]  # YYYYMMDD

    backed_up, skipped = [], []

    for src, name in _TARGETS:
        if not os.path.exists(src):
            continue
        # 当日バックアップ済みかチェック（force=False 時のみ）
        if not force:
            existing = [
                f for f in os.listdir(_BACKUP_DIR)
                if f.startswith(name + "." + day_str)
            ]
            if existing:
                skipped.append(name)
                continue
        dst = os.path.join(_BACKUP_DIR, f"{name}.{today}")
        shutil.copy2(src, dst)
        backed_up.append({"file": name, "dst": dst})

    _cleanup(_BACKUP_DIR, MAX_GENS)
    return {"backed_up": backed_up, "skipped": skipped, "ts": today}


def _cleanup(backup_dir: str, max_gens: int) -> None:
    """各ファイル名ベースごとに古い世代を削除する。"""
    by_base: dict[str, list[str]] = {}
    for fname in os.listdir(backup_dir):
        # 形式: <name>.<YYYYMMDD_HHMM>
        parts = fname.rsplit(".", 2)
        if len(parts) >= 2:
            base = ".".join(parts[:-1])  # 拡張子含む元ファイル名部分
            by_base.setdefault(base, []).append(fname)
    for base, files in by_base.items():
        files.sort(reverse=True)  # 新しい順
        for old_file in files[max_gens:]:
            try:
                os.remove(os.path.join(backup_dir, old_file))
            except OSError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# バックアップ一覧取得
# ─────────────────────────────────────────────────────────────────────────────

def list_backups() -> list[dict]:
    """バックアップファイルの一覧を返す（新しい順）。"""
    if not os.path.exists(_BACKUP_DIR):
        return []
    entries = []
    for fname in sorted(os.listdir(_BACKUP_DIR), reverse=True):
        fpath = os.path.join(_BACKUP_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        entries.append({
            "filename": fname,
            "size_kb":  round(stat.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return entries


def get_last_backup_time() -> str | None:
    """最新バックアップの日時文字列（なければ None）。"""
    entries = list_backups()
    return entries[0]["modified"] if entries else None


# ─────────────────────────────────────────────────────────────────────────────
# 自動バックアップ（起動時トリガー）
# ─────────────────────────────────────────────────────────────────────────────

def auto_backup_on_startup() -> bool:
    """
    アプリ起動時に1日1回だけバックアップを実行する。
    当日バックアップ済みなら何もしない。True=実行、False=スキップ。
    """
    today = datetime.now().strftime("%Y%m%d")
    if os.path.exists(_BACKUP_DIR):
        for fname in os.listdir(_BACKUP_DIR):
            if fname.endswith(".db." + today) or ("." + today + "_") in fname:
                return False  # 当日済み
    result = run_backup(force=False)
    return len(result["backed_up"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# サイドバーウィジェット（Streamlit）
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_backup() -> None:
    """サイドバーにバックアップステータスとボタンを表示する。"""
    import streamlit as st

    with st.sidebar.expander("💾 データバックアップ", expanded=False):
        last_ts = get_last_backup_time()
        if last_ts:
            st.caption(f"最終バックアップ: {last_ts}")
        else:
            st.caption("バックアップなし")

        backups = list_backups()
        st.caption(f"保存済み: {len(backups)} ファイル（最大 {MAX_GENS} 世代）")

        if st.button("🔄 今すぐバックアップ", use_container_width=True):
            with st.spinner("バックアップ中..."):
                result = run_backup(force=True)
            n = len(result["backed_up"])
            if n > 0:
                st.success(f"✅ {n} ファイルをバックアップしました")
            else:
                st.info("スキップ（すべて最新）")
