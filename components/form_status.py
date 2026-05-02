"""
案件結果登録ページ（成約/失注）
業務データは lease_data.db の past_cases のみを読み書きする。
screening_records は統計バッチ用（読み取り専用）。
"""
import json
import os
import sqlite3
import time
from contextlib import closing

import streamlit as st

# DB パス (絶対パス固定)
_DB_ROOT = os.path.join("/Users/kobayashiisaoryou/clawd/tune_lease_55", "data")
_LEASE_DB_PATH = os.path.join(_DB_ROOT, "lease_data.db")


def _load_workflow_cases() -> dict[str, list[dict]]:
    """past_casesの全件をロードし、ステータスごとに分類して返す"""
    categories = {
        "審査中": [],
        "見積もり提示": [],
        "稟議中": [],
        "成約": [],
        "失注": [],
        "検収完了": []
    }
    if not os.path.exists(_LEASE_DB_PATH):
        return categories
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, industry_sub, score, final_status, data "
                "FROM past_cases ORDER BY timestamp DESC"
            ).fetchall()
        for r in rows:
            pc_id = str(r["id"])
            try:
                data = json.loads(r["data"] or "{}")
            except Exception:
                data = {}
            
            status = r["final_status"] or data.get("final_status", "審査中")
            if status in ("未登録", "未設定", "", None):
                status = "審査中"
            elif status == "検収":
                status = "検収完了"
            
            if status not in categories:
                status = "審査中"

            industry_major = data.get("industry_major", "")
            industry_sub = r["industry_sub"] or data.get("industry_sub", "")
            company_name = data.get("company_name", "")
            company_no = data.get("company_no", "")
            result_dict = data.get("result", {}) or {}
            judgment = result_dict.get("hantei", data.get("judgment", ""))
            contract_prob = result_dict.get("contract_prob", data.get("contract_prob"))
            
            memo_for_display = json.dumps({
                "company_name": company_name,
                "company_no": company_no,
                "industry_major": industry_major,
                "industry_sub": industry_sub,
                **{k: v for k, v in data.items() if k not in ("result", "data", "inputs")},
            }, ensure_ascii=False, default=str)
            
            categories[status].append({
                "id": pc_id,
                "created_at": r["timestamp"],
                "industry_major": industry_major,
                "industry_sub": industry_sub,
                "customer_type": data.get("customer_type", ""),
                "score": r["score"] or 0.0,
                "judgment": judgment,
                "contract_prob": contract_prob,
                "memo": memo_for_display,
                "status": status,
                "data": data,
                "_source": "past_cases",
            })
    except Exception as _e:
        st.error(f"⚠️ past_cases読み込みエラー: {_e}")
    return categories


def _parse_memo(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _save_workflow_status(record_id, new_status: str, extra: dict = None) -> bool:
    """past_cases の final_status と data JSON を更新する"""
    if not os.path.exists(_LEASE_DB_PATH):
        return False
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (str(record_id),)).fetchone()
            if row is None:
                return False
            data = _parse_memo(row[0])
            data["final_status"] = new_status
            if extra:
                data.update(extra)
            conn.execute(
                "UPDATE past_cases SET final_status = ?, data = ? WHERE id = ?",
                (new_status, json.dumps(data, ensure_ascii=False, default=str), str(record_id)),
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"⚠️ 保存エラー: {e}")
        return False


def render_status_registration():
    """案件結果登録 (進捗パイプライン) タブを描画する"""
    st.title("📝 案件結果登録・進捗管理")
    st.info("審査案件のフェーズ（審査中・見積もり提示・稟議中・成約・失注・検収）を一元管理します。")

    if not os.path.exists(_LEASE_DB_PATH):
        st.error(f"DBファイルが見つかりません: `{_LEASE_DB_PATH}`")
        return

    workflow_data = _load_workflow_cases()
    
    db_total = 0
    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
        db_total = conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0]

    st.info(f"📊 **全体データ件数**: **{db_total} 件**")

    if st.button("♻️ 画面を強制再読み込み", key="btn_reload_status"):
        st.rerun()

    # 絞り込み検索
    search_kw = st.text_input("🔍 絞り込み検索", placeholder="会社名・業種・日付で絞り込み", key="search_register")

    def _filter_list(cases_list):
        if not search_kw:
            return cases_list
        kw = search_kw.lower()
        out = []
        for r in cases_list:
            memo = _parse_memo(r["memo"])
            company = memo.get("company_name", "") or ""
            if (
                kw in company.lower()
                or kw in (r["industry_sub"] or "").lower()
                or kw in (r["industry_major"] or "").lower()
                or kw in (r["created_at"] or "").lower()
            ):
                out.append(r)
        return out

    # タブによるワークフロー分類
    tab_names = ["🆕 審査中", "📄 見積もり提示", "⚖️ 稟議中", "🎉 成約", "❌ 失注", "✅ 検収完了"]
    tabs = st.tabs(tab_names)

    status_keys = ["審査中", "見積もり提示", "稟議中", "成約", "失注", "検収完了"]

    for i, tab in enumerate(tabs):
        with tab:
            status_key = status_keys[i]
            display_list = _filter_list(workflow_data[status_key])
            
            st.write(f"### {tab_names[i]} （{len(display_list)}件）")
            
            if not display_list:
                st.caption("該当する案件はありません。")
                continue
                
            for idx, record in enumerate(display_list):
                rec_id = record["id"]
                memo = _parse_memo(record["memo"])

                company_name = memo.get("company_name", "").strip()
                company_no = memo.get("company_no", "").strip()
                industry = record.get("industry_sub", "") or record.get("industry_major", "") or "業種不明"
                
                display_name = company_name if company_name else industry
                if company_no:
                    display_name = f"[{company_no}] {display_name}"

                score = record.get("score") or 0.0
                judgment = record.get("judgment", "—")
                created = (record.get("created_at") or "")[:16]

                # カード描画
                with st.expander(f"🏢 {display_name}　｜　{created}　｜　スコア: {score:.1f}", expanded=False):
                    col_info, col_actions = st.columns([1, 1])
                    with col_info:
                        st.write(f"**業種**: {industry}")
                        st.write(f"**顧客区分**: {record.get('customer_type', '—')}")
                        if company_name:
                            st.write(f"**会社名**: {company_name}")
                        if company_no:
                            st.write(f"**企業番号**: {company_no}")
                        st.write(f"**判定**: {judgment}")
                        if record.get("contract_prob") is not None:
                            st.write(f"**成約確率**: {record['contract_prob']:.1f}%")

                    with col_actions:
                        st.write("**ステータス更新**")

                        # ── ワークフロー進行ボタン ──
                        if status_key == "審査中":
                            if st.button("📄 見積もり提示へ進める", key=f"to_mitsumori_{rec_id}"):
                                if _save_workflow_status(rec_id, "見積もり提示"):
                                    st.toast("見積もり提示へ更新しました")
                                    time.sleep(0.5)
                                    st.rerun()
                        elif status_key == "見積もり提示":
                            if st.button("⚖️ 稟議中へ進める", key=f"to_ringi_{rec_id}"):
                                if _save_workflow_status(rec_id, "稟議中"):
                                    st.toast("稟議中へ更新しました")
                                    time.sleep(0.5)
                                    st.rerun()
                        elif status_key == "成約":
                            if st.button("✅ 検収完了へ進める", key=f"to_kenshu_{rec_id}", type="primary"):
                                if _save_workflow_status(rec_id, "検収完了"):
                                    st.toast("検収完了に更新しました")
                                    time.sleep(0.5)
                                    st.rerun()

                        # ── 成約/失注は全タブで常時登録可能 ──
                        with st.form(f"finalize_form_{rec_id}"):
                            res_status = st.radio("最終結果", ["成約", "失注"], horizontal=True, key=f"radio_{rec_id}")
                            final_rate = st.number_input("獲得レート (%)", value=0.0, step=0.01, format="%.2f", key=f"rate_{rec_id}")
                            _lost_opts = ["—（選択）", "設備見合わせ", "他社競合", "調達方法変更", "その他"]
                            lost_reason = st.selectbox("失注理由（失注の場合）", _lost_opts, key=f"lost_{rec_id}")
                            competitor_name = st.text_input("競合他社", placeholder="例: ○○リース", key=f"comp_{rec_id}")
                            loan_condition_options = ["本件限度", "親会社保証", "担保あり", "金融機関協調", "その他"]
                            loan_conditions = st.multiselect("承認条件", loan_condition_options, key=f"cond_{rec_id}")

                            submitted = st.form_submit_button("✅ 結果を確定する", type="primary")

                            if submitted:
                                extra = {
                                    "final_rate": final_rate,
                                    "loan_conditions": loan_conditions,
                                    "competitor_name": competitor_name.strip(),
                                }
                                if res_status == "失注":
                                    extra["lost_reason"] = lost_reason if lost_reason != "—（選択）" else ""
                                if _save_workflow_status(rec_id, res_status, extra):
                                    st.toast(f"{res_status} を登録しました")
                                    try:
                                        from shinsa_gunshi import refresh_evidence_weights
                                        refresh_evidence_weights()
                                        from auto_optimizer import run_auto_optimization
                                        run_auto_optimization()
                                    except Exception:
                                        pass
                                    time.sleep(0.5)
                                    st.rerun()

                        # 削除ボタンも設置
                        if st.button("🗑️ 案件レコードを完全に削除", key=f"del_full_{rec_id}", type="secondary"):
                            try:
                                with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
                                    conn.execute("DELETE FROM past_cases WHERE id=?", (str(rec_id),))
                                    conn.commit()
                                st.toast(f"🗑️ 案件を削除しました")
                                time.sleep(0.5)
                                st.rerun()
                            except Exception as e:
                                st.error(f"削除失敗: {e}")


def render_quick_status_widget(context_key: str = "main") -> None:
    """
    サイドバーや任意の場所に埋め込める簡易結果登録ウィジェット。
    どのステータスの案件でも直接 成約/失注 を登録できる。
    st.tabs() を使わないため sidebar でも動作する。
    context_key: 同一画面に複数配置する場合の重複キー防止用。
    """
    if not os.path.exists(_LEASE_DB_PATH):
        st.caption("DBが見つかりません")
        return

    workflow_data = _load_workflow_cases()

    # 検収完了以外の全案件を対象（成約・失注も再登録可）
    all_cases = []
    for status in ["審査中", "見積もり提示", "稟議中", "成約", "失注"]:
        for r in workflow_data[status]:
            all_cases.append((status, r))

    if not all_cases:
        st.caption("案件がありません。")
        return

    # 案件選択
    labels = []
    for status, r in all_cases:
        memo = _parse_memo(r["memo"])
        name = memo.get("company_name", "") or r.get("industry_sub", "") or "案件"
        no = memo.get("company_no", "")
        prefix = f"[{no}] " if no else ""
        labels.append(f"{prefix}{name}（{status}）")

    selected_idx = st.selectbox(
        "案件を選択",
        range(len(labels)),
        format_func=lambda i: labels[i],
        key=f"quick_case_sel_{context_key}",
    )

    cur_status, record = all_cases[selected_idx]
    rec_id = record["id"]
    st.caption(f"現在: **{cur_status}**　スコア: {record.get('score', 0):.1f}")

    # ── 成約/失注 は常に直接登録可能 ──────────────────────────────────────
    with st.form(f"quick_finalize_{rec_id}_{context_key}"):
        res_radio = st.radio(
            "結果", ["成約", "失注"],
            horizontal=True,
            key=f"qr_radio_{rec_id}_{context_key}",
        )
        final_rate = st.number_input(
            "獲得レート (%)", value=0.0, step=0.01, format="%.2f",
            key=f"qr_rate_{rec_id}_{context_key}",
        )
        _ql_opts = ["—（選択）", "設備見合わせ", "他社競合", "調達方法変更", "その他"]
        lost_reason = st.selectbox(
            "失注理由",
            _ql_opts,
            key=f"qr_lost_{rec_id}_{context_key}",
        )
        submitted = st.form_submit_button("✅ 成約/失注を登録", type="primary")
        if submitted:
            extra = {"final_rate": final_rate}
            if res_radio == "失注":
                extra["lost_reason"] = lost_reason if lost_reason != "—（選択）" else ""
            if _save_workflow_status(rec_id, res_radio, extra):
                st.toast(f"{res_radio} を登録しました")
                try:
                    from shinsa_gunshi import refresh_evidence_weights
                    refresh_evidence_weights()
                    from auto_optimizer import run_auto_optimization
                    run_auto_optimization()
                except Exception:
                    pass
                time.sleep(0.4)
                st.rerun()

    # ── ワークフロー進行ボタン（任意） ────────────────────────────────────
    if cur_status == "審査中":
        if st.button("📄 見積もり提示へ", key=f"quick_mitsumori_{rec_id}_{context_key}", use_container_width=True):
            if _save_workflow_status(rec_id, "見積もり提示"):
                st.toast("見積もり提示へ更新")
                time.sleep(0.4)
                st.rerun()
    elif cur_status == "見積もり提示":
        if st.button("⚖️ 稟議中へ", key=f"quick_ringi_{rec_id}_{context_key}", use_container_width=True):
            if _save_workflow_status(rec_id, "稟議中"):
                st.toast("稟議中へ更新")
                time.sleep(0.4)
                st.rerun()
