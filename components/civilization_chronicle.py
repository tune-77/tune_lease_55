"""
🌌 文明年代記 — ガバナンス変更履歴・エントロピー監視・ロールバック UI
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from data_cases import (
    load_all_cases,
    load_coeff_history,
    load_coeff_overrides,
    load_governance_snapshots,
    save_coeff_overrides,
)
from scoring_core import APPROVAL_LINE

_ENTROPY_WARN_THRESHOLD = 0.15  # 承認率が基準から15%以上乖離したら警告


# ── ヘルパー ────────────────────────────────────────────────────────────────────

def _compute_approval_rates(cases: list[dict]) -> tuple[float | None, float | None]:
    """
    全案件と直近30件の承認率を返す。
    returns: (baseline_rate, recent_rate)  — データ不足時は None
    """
    scored = [
        c for c in cases
        if c.get("result") and c["result"].get("score") is not None
    ]
    if len(scored) < 10:
        return None, None
    all_scores = [c["result"]["score"] for c in scored]
    baseline_rate = sum(1 for s in all_scores if s >= APPROVAL_LINE) / len(all_scores)
    recent = all_scores[-30:]
    recent_rate = sum(1 for s in recent if s >= APPROVAL_LINE) / len(recent)
    return baseline_rate, recent_rate


# ── セクション別レンダラー ───────────────────────────────────────────────────────

def _render_entropy_section() -> None:
    st.subheader("📡 スコア分布モニタリング")

    cases = load_all_cases()
    baseline_rate, recent_rate = _compute_approval_rates(cases)

    if baseline_rate is None:
        st.info("案件データが10件未満のため、モニタリングは案件蓄積後に有効になります。")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("全体ベースライン承認率", f"{baseline_rate:.1%}", help="全スコア済み案件の承認ライン超え割合")
    col2.metric("直近30件 承認率", f"{recent_rate:.1%}",
                delta=f"{recent_rate - baseline_rate:+.1%}",
                delta_color="normal")
    col3.metric("乖離幅", f"{abs(recent_rate - baseline_rate):.1%}")

    drift = abs(recent_rate - baseline_rate)
    if drift >= _ENTROPY_WARN_THRESHOLD:
        st.warning(
            f"⚠️ スコア分布が基準から **{drift:.1%}** 乖離しています。"
            " 係数や閾値の見直し、または審査傾向の変化を確認してください。"
        )
    else:
        st.success("✅ スコア分布は安定しています。")


def _render_history_section() -> None:
    st.subheader("📜 係数変更履歴")

    history = load_coeff_history()
    if not history:
        st.info("係数変更履歴がまだありません。")
        return

    rows = []
    for rec in history[:100]:
        changed_keys = rec.get("changed_keys", {})
        rows.append({
            "日時": rec.get("timestamp", ""),
            "種別": rec.get("change_type", ""),
            "コメント": rec.get("comment", ""),
            "変更キー数": len(changed_keys),
            "変更キー": ", ".join(list(changed_keys.keys())[:5]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_rollback_section() -> None:
    st.subheader("⏪ ロールバック")

    snapshots = load_governance_snapshots()  # 新しい順

    if not snapshots:
        st.info("スナップショットがまだありません。係数を変更すると自動的に保存されます。")
        return

    st.caption(f"保存済みスナップショット: {len(snapshots)} 件（最大50件）")

    for snap in snapshots:
        snap_id = snap.get("id", "")
        ts = snap.get("ts", "")
        comment = snap.get("comment", "（コメントなし）")
        overrides = snap.get("overrides", {})
        key_count = len(overrides)

        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{ts}**　`{comment}`")
                st.caption(f"オーバーライドキー数: {key_count}　/ ID: `{snap_id}`")
            with col2:
                if st.button("⏪ 戻す", key=f"rollback_{snap_id}"):
                    st.session_state[f"_rollback_confirm_{snap_id}"] = True

            if st.session_state.get(f"_rollback_confirm_{snap_id}"):
                st.warning("このスナップショットに係数を戻します。よろしいですか？")
                c1, c2 = st.columns(2)
                if c1.button("✅ 確認してロールバック", key=f"confirm_ok_{snap_id}", type="primary"):
                    success = save_coeff_overrides(overrides, comment=f"rollback from {snap_id}")
                    if success:
                        st.success("ロールバック完了。係数を復元しました。")
                        st.session_state.pop(f"_rollback_confirm_{snap_id}", None)
                        st.rerun()
                    else:
                        st.error("ロールバックに失敗しました。")
                if c2.button("❌ キャンセル", key=f"confirm_cancel_{snap_id}"):
                    st.session_state.pop(f"_rollback_confirm_{snap_id}", None)
                    st.rerun()


# ── メインエントリポイント ────────────────────────────────────────────────────────

def render_civilization_chronicle() -> None:
    """🌌 文明年代記 メイン画面"""
    st.title("🌌 文明年代記")
    st.caption("ガバナンス変更の記録・スコア分布の監視・過去状態へのロールバック")
    st.divider()

    tab_entropy, tab_history, tab_rollback = st.tabs([
        "📡 スコア分布モニタリング",
        "📜 係数変更履歴",
        "⏪ ロールバック",
    ])

    with tab_entropy:
        _render_entropy_section()

    with tab_history:
        _render_history_section()

    with tab_rollback:
        _render_rollback_section()
