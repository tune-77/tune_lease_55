"""
案件結果登録ページ（成約/失注）
screening_db.sqlite の screening_records テーブルを直接読み書きする。
"""
import json
import os
import sqlite3
import time
from contextlib import closing

import streamlit as st

# DB パス（customer_db.py と同じ解決方法）
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "screening_db.sqlite")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _load_pending() -> list[dict]:
    """final_status が NULL/空/"未登録" の全件を新しい順で返す"""
    if not os.path.exists(_DB_PATH):
        return []
    with closing(_get_conn()) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, industry_major, industry_sub,
                   customer_type, score, judgment, contract_prob, memo
            FROM screening_records
            WHERE (
                memo IS NULL
                OR memo = ''
                OR memo NOT LIKE '%final_status%'
                OR memo LIKE '%"final_status": "未登録"%'
                OR memo LIKE '%"final_status":"未登録"%'
            )
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _parse_memo(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _save_final_status(record_id: int, final_status: str, extra: dict) -> bool:
    """memo JSON に final_status と追加情報をマージして保存する"""
    if not os.path.exists(_DB_PATH):
        return False
    with closing(_get_conn()) as conn:
        row = conn.execute("SELECT memo FROM screening_records WHERE id = ?", (record_id,)).fetchone()
        if row is None:
            return False
        memo = _parse_memo(row["memo"])
        memo["final_status"] = final_status
        memo.update(extra)
        conn.execute(
            "UPDATE screening_records SET memo = ? WHERE id = ?",
            (json.dumps(memo, ensure_ascii=False), record_id),
        )
        conn.commit()
    return True


def render_status_registration():
    """案件結果登録 (成約/失注) タブを描画する"""
    st.title("📝 案件結果登録")
    st.info("審査案件に対して、最終的な結果（成約・失注）を登録します。")

    if not os.path.exists(_DB_PATH):
        st.error(f"DBファイルが見つかりません: `{_DB_PATH}`")
        return

    # --- 全件ロード ---
    all_pending = _load_pending()
    total_count = all_pending.__len__()

    st.caption(f"未登録件数: **{total_count}件** （`screening_records` より直接読込）")

    if total_count == 0:
        st.success("全ての案件が登録済みです！")
        return

    # --- 絞り込みフィルター ---
    with st.expander("🔍 絞り込み検索", expanded=False):
        search_kw = st.text_input("会社名・業種・日付で絞り込み", placeholder="例: 建設、株式会社テスト、2026-04")

    if search_kw:
        kw = search_kw.lower()
        filtered = []
        for r in all_pending:
            memo = _parse_memo(r["memo"])
            company = memo.get("company_name", "") or ""
            if (
                kw in company.lower()
                or kw in (r["industry_sub"] or "").lower()
                or kw in (r["industry_major"] or "").lower()
                or kw in (r["created_at"] or "").lower()
            ):
                filtered.append(r)
    else:
        filtered = all_pending

    st.caption(f"絞り込み後: {len(filtered)}件")

    # --- 表示件数スライダー ---
    max_show = max(5, len(filtered))
    default_show = min(20, max_show)
    if len(filtered) > 5:
        show_n = st.slider("表示件数", 5, min(100, max_show), default_show)
    else:
        show_n = len(filtered)

    display_list = filtered[:show_n]

    # --- 案件カード ---
    for idx, record in enumerate(display_list):
        rec_id = record["id"]
        memo = _parse_memo(record["memo"])

        company_name = memo.get("company_name", "").strip()
        industry = record.get("industry_sub", "") or record.get("industry_major", "") or "業種不明"
        display_name = company_name if company_name else industry

        score = record.get("score") or 0.0
        judgment = record.get("judgment", "—")
        created = (record.get("created_at") or "")[:16]

        with st.expander(f"🏢 {display_name}　｜　{created}　｜　スコア: {score:.1f}　｜　{judgment}", expanded=False):
            col_info, col_form = st.columns([1, 2])

            with col_info:
                st.write(f"**業種**: {industry}")
                st.write(f"**顧客区分**: {record.get('customer_type', '—')}")
                if company_name:
                    st.write(f"**会社名**: {company_name}")
                st.write(f"**判定**: {judgment}")
                st.write(f"**スコア**: {score:.1f}")
                if record.get("contract_prob") is not None:
                    st.write(f"**成約確率**: {record['contract_prob']:.1f}%")
                # memo から追加情報
                if memo.get("lease_amount"):
                    st.write(f"**リース物件**: {memo.get('lease_amount')}")

            with col_form:
                with st.form(f"status_form_{rec_id}_{idx}"):
                    res_status = st.radio("結果", ["成約", "失注"], horizontal=True, key=f"radio_{rec_id}")
                    final_rate = st.number_input(
                        "獲得レート (%)", value=0.0, step=0.01, format="%.2f",
                        help="成約した場合の決定金利",
                        key=f"rate_{rec_id}"
                    )
                    lost_reason = st.text_input(
                        "失注理由（失注の場合）",
                        placeholder="例: 金利で他社に負けた",
                        key=f"lost_{rec_id}"
                    )
                    competitor_name = st.text_input(
                        "競合他社",
                        placeholder="例: ○○銀行",
                        key=f"comp_{rec_id}"
                    )
                    loan_condition_options = [
                        "本件限度", "次回決算まで本件限度", "金融機関と協調",
                        "独立・新設向け条件", "親会社等保証", "担保・保全あり", "その他"
                    ]
                    loan_conditions = st.multiselect(
                        "承認条件",
                        loan_condition_options,
                        key=f"cond_{rec_id}"
                    )

                    submitted = st.form_submit_button("✅ 登録する", type="primary")

                if submitted:
                    if res_status == "成約" and final_rate == 0.0:
                        st.warning("💡 獲得レートを入力すると分析精度が向上します")
                    if res_status == "失注" and not lost_reason.strip():
                        st.warning("💡 失注理由を入力すると定性分析の精度が向上します")

                    extra = {
                        "final_rate": final_rate,
                        "loan_conditions": loan_conditions,
                        "competitor_name": competitor_name.strip(),
                    }
                    if res_status == "失注":
                        extra["lost_reason"] = lost_reason.strip()

                    if _save_final_status(rec_id, res_status, extra):
                        st.success(f"✅ 登録完了: {display_name} → **{res_status}**")
                        # BN 証拠重みを実績から再学習
                        try:
                            from components.shinsa_gunshi import refresh_evidence_weights
                            refresh_evidence_weights()
                        except Exception:
                            pass
                        # 自動係数最適化チェック
                        try:
                            from auto_optimizer import run_auto_optimization, get_training_status
                            _opt = run_auto_optimization()
                            if _opt:
                                _auc = _opt.get("auc_borrower_asset")
                                _auc_str = f" AUC: {_auc:.3f}" if _auc else ""
                                st.success(f"🧠 係数を自動更新（{_opt['n_cases']}件{_auc_str}）")
                        except Exception:
                            pass
                        time.sleep(0.8)
                        st.rerun()
                    else:
                        st.error("保存に失敗しました。")
