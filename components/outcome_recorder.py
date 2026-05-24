"""outcome_recorder.py — 支払状況登録コンポーネント（流体化 Phase 0 UI）

審査後の実際の支払い状況（正常/延滞/デフォルト）を screening_outcomes に記録する。
このデータが蓄積されると retraining_pipeline.py が自動的に学習を開始する。

使い方（tune_lease_55.py のルーティングから呼ぶ）:
    from components.outcome_recorder import render_outcome_recorder
    render_outcome_recorder()
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from typing import Optional

import streamlit as st

_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_DIR, "data", "lease_data.db")

_STATUS_LABELS = {
    "normal":    "✅ 正常（支払い継続中）",
    "late_30":   "⚠️ 30日延滞",
    "late_90":   "🔴 90日延滞（要注意）",
    "default":   "❌ デフォルト（回収不能）",
    "completed": "🏁 正常完了（リース満了）",
}

_STATUS_TO_DELINQUENT = {
    "normal":    0,
    "late_30":   1,
    "late_90":   1,
    "default":   1,
    "completed": 0,
}


# ──────────────────────────────────────────────────────────────────────────────
# DB ヘルパー
# ──────────────────────────────────────────────────────────────────────────────

def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _get_case_info(case_id: str) -> Optional[dict]:
    """past_cases から案件情報を取得する。"""
    import json
    conn = _open_db()
    row = conn.execute(
        "SELECT id, final_status, timestamp, data FROM past_cases WHERE id = ?",
        (case_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = json.loads(row["data"]) if row["data"] else {}
    result = d.get("result", {}) or {}
    inputs = d.get("inputs", {}) or {}
    return {
        "case_id":       row["id"],
        "final_status":  row["final_status"],
        "timestamp":     row["timestamp"],
        "score":         result.get("score"),
        "asset_score":   result.get("asset_score"),
        "hantei":        result.get("hantei"),
        "company_name":  inputs.get("company_name", "（社名非表示）"),
        "industry":      d.get("industry_major", ""),
    }


def _get_existing_outcome(case_id: str) -> Optional[dict]:
    """既登録の支払状況を取得する。"""
    conn = _open_db()
    row = conn.execute(
        "SELECT * FROM screening_outcomes WHERE case_id = ? ORDER BY id DESC LIMIT 1",
        (case_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _save_outcome(
    case_id: str,
    actual_status: str,
    contract_date: Optional[str] = None,
    scheduled_end_date: Optional[str] = None,
    loss_given_default: Optional[float] = None,
    notes: str = "",
) -> bool:
    """screening_outcomes に支払状況を保存（INSERT OR REPLACE）。"""
    delinquent = _STATUS_TO_DELINQUENT.get(actual_status, 0)
    conn = _open_db()
    try:
        # 既存レコードがあれば UPDATE、なければ INSERT
        existing = conn.execute(
            "SELECT id FROM screening_outcomes WHERE case_id = ? ORDER BY id DESC LIMIT 1",
            (case_id,)
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE screening_outcomes
                   SET actual_status=?, delinquent=?, contract_date=?,
                       scheduled_end_date=?, loss_given_default=?,
                       notes=?, checked_at=datetime('now'), updated_at=datetime('now')
                   WHERE id=?""",
                (actual_status, delinquent, contract_date,
                 scheduled_end_date, loss_given_default, notes, existing["id"])
            )
        else:
            conn.execute(
                """INSERT INTO screening_outcomes
                   (case_id, actual_status, delinquent, contract_date,
                    scheduled_end_date, loss_given_default, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (case_id, actual_status, delinquent, contract_date,
                 scheduled_end_date, loss_given_default, notes)
            )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"保存エラー: {e}")
        return False
    finally:
        conn.close()


def _get_outcome_summary() -> dict:
    """登録状況サマリーを返す。"""
    conn = _open_db()
    total_contracted = conn.execute(
        "SELECT COUNT(*) FROM screening_records WHERE outcome IN ('contracted','completed')"
    ).fetchone()[0]
    registered = conn.execute(
        "SELECT COUNT(DISTINCT case_id) FROM screening_outcomes"
    ).fetchone()[0]
    delinquent = conn.execute(
        "SELECT COUNT(*) FROM screening_outcomes WHERE delinquent=1"
    ).fetchone()[0]
    status_dist = conn.execute(
        "SELECT actual_status, COUNT(*) FROM screening_outcomes GROUP BY actual_status ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    return {
        "total_contracted": total_contracted,
        "registered": registered,
        "delinquent": delinquent,
        "status_dist": [(r[0], r[1]) for r in status_dist],
        "coverage_pct": round(registered / total_contracted * 100, 1) if total_contracted > 0 else 0,
    }


def _search_cases(query: str, status_filter: str = "全て", limit: int = 20) -> list[dict]:
    """past_cases を検索する。"""
    import json
    conn = _open_db()

    # outcome フィルタ
    status_cond = ""
    params: list = []
    if status_filter == "登録済み":
        status_cond = "AND EXISTS (SELECT 1 FROM screening_outcomes so WHERE so.case_id = pc.id)"
    elif status_filter == "未登録":
        status_cond = "AND NOT EXISTS (SELECT 1 FROM screening_outcomes so WHERE so.case_id = pc.id)"

    # ID 検索
    if query:
        rows = conn.execute(
            f"SELECT id, final_status, timestamp, data FROM past_cases "
            f"WHERE (id LIKE ? OR data LIKE ?) "
            f"AND final_status IN ('成約','検収完了') {status_cond} "
            f"ORDER BY timestamp DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT id, final_status, timestamp, data FROM past_cases "
            f"WHERE final_status IN ('成約','検収完了') {status_cond} "
            f"ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()

    result = []
    for row in rows:
        d = json.loads(row["data"]) if row["data"] else {}
        r = d.get("result", {}) or {}
        inp = d.get("inputs", {}) or {}
        # screening_outcomes の登録状況
        existing = _get_existing_outcome(row["id"])
        result.append({
            "case_id":      row["id"],
            "final_status": row["final_status"],
            "timestamp":    row["timestamp"][:10] if row["timestamp"] else "",
            "score":        r.get("score", 0),
            "company":      inp.get("company_name", "（非表示）"),
            "industry":     d.get("industry_major", ""),
            "has_outcome":  existing is not None,
            "outcome_status": existing["actual_status"] if existing else None,
        })
    return result


# ──────────────────────────────────────────────────────────────────────────────
# UI レンダリング
# ──────────────────────────────────────────────────────────────────────────────

def render_outcome_recorder() -> None:
    """支払状況登録ページをレンダリングする。"""

    st.title("📋 支払状況登録")
    st.caption(
        "成約案件の実際の支払い状況を記録します。"
        "このデータが **5件以上** 蓄積されると、審査モデルが自動的に再学習します。"
    )

    # サマリーカード
    summary = _get_outcome_summary()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("成約・検収完了", f"{summary['total_contracted']}件")
    col2.metric(
        "支払状況 登録済み",
        f"{summary['registered']}件",
        f"{summary['coverage_pct']}% カバー率",
    )
    col3.metric(
        "延滞/デフォルト確認",
        f"{summary['delinquent']}件",
        "5件で自動学習開始" if summary['delinquent'] < 5 else "✅ 学習可能",
        delta_color="off" if summary['delinquent'] < 5 else "normal",
    )
    # 学習までの進捗バー
    progress = min(summary['delinquent'] / 5, 1.0)
    col4.write("**自動学習まで**")
    col4.progress(progress, text=f"{summary['delinquent']}/5件")

    if summary['delinquent'] >= 5:
        st.success(
            "✅ 延滞/デフォルト実績が5件以上蓄積されました。"
            "次の案件登録時にモデルが自動再学習されます。"
        )

    st.divider()

    # タブ: 一覧検索 / 個別登録
    tab_list, tab_single = st.tabs(["📋 一覧から登録", "🔍 案件IDで登録"])

    with tab_list:
        _render_list_tab()

    with tab_single:
        _render_single_tab()


def _render_list_tab() -> None:
    """一覧検索から支払状況を登録するタブ。"""
    col_s, col_f = st.columns([3, 1])
    with col_s:
        query = st.text_input(
            "案件ID・会社名で検索",
            placeholder="例: 202603 または 株式会社",
            key="outcome_search_query",
        )
    with col_f:
        status_filter = st.selectbox(
            "表示フィルタ",
            ["全て", "未登録", "登録済み"],
            key="outcome_status_filter",
        )

    cases = _search_cases(query or "", status_filter)

    if not cases:
        st.info("該当する案件が見つかりません。")
        return

    for case in cases:
        badge = "✅ 登録済み" if case["has_outcome"] else "⬜ 未登録"
        status_text = f" ({_STATUS_LABELS.get(case['outcome_status'], case['outcome_status'])})" \
            if case["outcome_status"] else ""

        with st.expander(
            f"{badge}{status_text}  |  {case['timestamp']}  |  "
            f"スコア {case['score']:.1f}点  |  {case['final_status']}  |  ID: ...{case['case_id'][-6:]}",
            expanded=False,
        ):
            _render_outcome_form(case["case_id"])


def _render_single_tab() -> None:
    """案件ID直接入力フォーム。"""
    case_id = st.text_input(
        "案件ID",
        placeholder="例: 20260323083756178083",
        key="outcome_single_case_id",
    )
    if case_id:
        info = _get_case_info(case_id)
        if info is None:
            st.error(f"案件ID `{case_id}` が見つかりません。")
        else:
            st.write(f"**審査スコア**: {info['score']:.1f}点  |  **判定**: {info['hantei']}  |  **ステータス**: {info['final_status']}")
            _render_outcome_form(case_id)


def _render_outcome_form(case_id: str) -> None:
    """支払状況入力フォームを描画する（case_id ごとに一意なキー）。"""
    existing = _get_existing_outcome(case_id)
    default_status = existing["actual_status"] if existing else "normal"

    key_pfx = f"outcome_{case_id[-8:]}"

    status = st.selectbox(
        "支払状況",
        options=list(_STATUS_LABELS.keys()),
        format_func=lambda x: _STATUS_LABELS[x],
        index=list(_STATUS_LABELS.keys()).index(default_status),
        key=f"{key_pfx}_status",
    )

    col1, col2 = st.columns(2)
    with col1:
        contract_date = st.date_input(
            "成約日",
            value=date.fromisoformat(existing["contract_date"]) if existing and existing.get("contract_date") else None,
            key=f"{key_pfx}_contract_date",
        )
    with col2:
        end_date = st.date_input(
            "リース満了予定日",
            value=date.fromisoformat(existing["scheduled_end_date"]) if existing and existing.get("scheduled_end_date") else None,
            key=f"{key_pfx}_end_date",
        )

    loss = None
    if status in ("late_90", "default"):
        loss = st.number_input(
            "実損額（千円）",
            min_value=0,
            value=int(existing["loss_given_default"] / 1000) if existing and existing.get("loss_given_default") else 0,
            key=f"{key_pfx}_loss",
        )

    notes = st.text_area(
        "メモ",
        value=existing.get("notes", "") if existing else "",
        placeholder="例: 2026年3月より30日延滞。担当に確認中。",
        height=80,
        key=f"{key_pfx}_notes",
    )

    if st.button("💾 保存", key=f"{key_pfx}_save", type="primary"):
        ok = _save_outcome(
            case_id=case_id,
            actual_status=status,
            contract_date=str(contract_date) if contract_date else None,
            scheduled_end_date=str(end_date) if end_date else None,
            loss_given_default=float(loss * 1000) if loss else None,
            notes=notes,
        )
        if ok:
            st.success(f"✅ 保存しました: {_STATUS_LABELS[status]}")
            # FluidPipeline への通知（存在する場合）
            try:
                from fluid_pipeline import FluidPipeline
                FluidPipeline().on_outcome_registered(case_id=case_id, status=status)
            except Exception:
                pass
            st.rerun()
