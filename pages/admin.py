"""pages/admin.py — モデル管理画面。手動再学習トリガーと再学習履歴を表示する。"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from retraining_pipeline import check_retraining_needed, run_retraining

DB_PATH = "data/lease_data.db"
MODEL_DIR = "models/"

st.set_page_config(page_title="管理画面 — モデル管理", layout="wide")
st.title("モデル管理")


@st.cache_data(ttl=30)
def _get_counts(db_path: str) -> tuple[int, int]:
    try:
        conn = sqlite3.connect(db_path)
        (total,) = conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()
        (confirmed,) = conn.execute(
            "SELECT COUNT(*) FROM screening_records WHERE outcome IS NOT NULL"
        ).fetchone()
        conn.close()
        return int(total), int(confirmed)
    except Exception:  # noqa: BLE001
        return 0, 0


@st.cache_data(ttl=30)
def _get_last_retraining(db_path: str) -> dict | None:
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT started_at, new_auc FROM retraining_log "
            "WHERE status='success' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return {"started_at": row[0], "new_auc": row[1]} if row else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=30)
def _get_retraining_log(db_path: str) -> list[dict]:
    try:
        import pandas as pd
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT started_at, records_used, new_auc, status, rollback_reason "
            "FROM retraining_log ORDER BY id DESC LIMIT 10"
        ).fetchall()
        conn.close()
        return [
            {
                "日時": r[0],
                "件数": r[1],
                "AUC": f"{r[2]:.3f}" if r[2] is not None else "—",
                "状態": r[3],
                "ロールバック理由": r[4] or "",
            }
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        return []


# ── データ件数 ──
total, confirmed = _get_counts(DB_PATH)
last = _get_last_retraining(DB_PATH)

col1, col2 = st.columns(2)
col1.metric("現在の審査データ", f"{total}件", f"うち結果確定: {confirmed}件")
if last:
    auc_str = f"{last['new_auc']:.3f}" if last["new_auc"] is not None else "—"
    col2.metric("最終再学習", last["started_at"], f"AUC: {auc_str}")
else:
    col2.metric("最終再学習", "なし")

# ── 自動チェック通知 ──
_retrain_check = check_retraining_needed(db_path=DB_PATH)
_retrain_needed = _retrain_check["needed"] if isinstance(_retrain_check, dict) else bool(_retrain_check)
_retrain_reason = _retrain_check.get("reason", "") if isinstance(_retrain_check, dict) else ""
_delinquent_cnt = _retrain_check.get("delinquent_count", 0) if isinstance(_retrain_check, dict) else 0
if _retrain_needed:
    st.toast("再学習データが蓄積されました", icon="ℹ️")
elif _delinquent_cnt < 5:
    st.info(f"💡 {_retrain_reason}")

st.divider()

# ── 手動再学習トリガー ──
if st.button("モデル再学習を実行", type="primary"):
    with st.spinner("再学習中..."):
        result = run_retraining(
            triggered_by="manual_streamlit",
            db_path=DB_PATH,
            model_dir=MODEL_DIR,
        )
    _get_counts.clear()
    _get_last_retraining.clear()
    _get_retraining_log.clear()

    if result["status"] == "success":
        st.success(f"再学習完了: AUC {result['new_auc']:.3f}")
    elif result["status"] == "rolled_back":
        st.warning(f"ロールバック: {result['rollback_reason']}")
    elif result["status"] == "skipped":
        st.info(f"スキップ: データ不足（{result['records_used']}件）")
    else:
        st.error(f"エラー: {result['error']}")

# ── 再学習履歴 ──
st.subheader("再学習履歴")
rows = _get_retraining_log(DB_PATH)
if not rows:
    st.info("再学習履歴はまだありません。")
else:
    st.dataframe(rows, use_container_width=True)
