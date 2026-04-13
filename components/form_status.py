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
_DB_ROOT = os.path.join("/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12", "data")
_LEASE_DB_PATH = os.path.join(_DB_ROOT, "lease_data.db")


def _load_pending() -> list[dict]:
    """final_status = '未登録' の全件を新しい順で返す（past_cases のみ）"""
    results = []
    if not os.path.exists(_LEASE_DB_PATH):
        return results
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, timestamp, industry_sub, score, data "
                "FROM past_cases WHERE final_status = '未登録' ORDER BY timestamp DESC"
            ).fetchall()
        for r in rows:
            pc_id = str(r["id"])
            try:
                data = json.loads(r["data"] or "{}")
            except Exception:
                data = {}
            industry_major = data.get("industry_major", "")
            industry_sub = r["industry_sub"] or data.get("industry_sub", "")
            company_name = data.get("company_name", "")
            result_dict = data.get("result", {}) or {}
            judgment = result_dict.get("hantei", data.get("judgment", ""))
            contract_prob = result_dict.get("contract_prob", data.get("contract_prob"))
            memo_for_display = json.dumps({
                "company_name": company_name,
                "industry_major": industry_major,
                "industry_sub": industry_sub,
                **{k: v for k, v in data.items() if k not in ("result", "data", "inputs")},
            }, ensure_ascii=False, default=str)
            results.append({
                "id": pc_id,
                "created_at": r["timestamp"],
                "industry_major": industry_major,
                "industry_sub": industry_sub,
                "customer_type": data.get("customer_type", ""),
                "score": r["score"] or 0.0,
                "judgment": judgment,
                "contract_prob": contract_prob,
                "memo": memo_for_display,
                "_source": "past_cases",
            })
    except Exception as _e:
        st.error(f"⚠️ past_cases読み込みエラー: {_e}")
    return results


def _parse_memo(raw: str) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _save_final_status(record_id, final_status: str, extra: dict) -> bool:
    """past_cases の final_status と data JSON を更新する"""
    if not os.path.exists(_LEASE_DB_PATH):
        return False
    try:
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            row = conn.execute("SELECT data FROM past_cases WHERE id = ?", (str(record_id),)).fetchone()
            if row is None:
                return False
            data = _parse_memo(row[0])  # row_factory未設定のためインデックスでアクセス
            data["final_status"] = final_status
            data.update(extra)
            conn.execute(
                "UPDATE past_cases SET final_status = ?, data = ? WHERE id = ?",
                (final_status, json.dumps(data, ensure_ascii=False, default=str), str(record_id)),
            )
            conn.commit()
        return True
    except Exception as e:
        st.error(f"⚠️ 保存エラー: {e}")
        return False


def render_status_registration():
    """案件結果登録 (成約/失注) タブを描画する"""
    st.title("📝 案件結果登録")
    st.info("審査案件に対して、最終的な結果（成約・失注）を登録します。")

    if not os.path.exists(_LEASE_DB_PATH):
        st.error(f"DBファイルが見つかりません: `{_LEASE_DB_PATH}`")
        return

    # --- 全件ロード ---
    all_pending = _load_pending()
    total_count = len(all_pending)

    db_total = 0
    if os.path.exists(_LEASE_DB_PATH):
        with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
            db_total = conn.execute("SELECT COUNT(*) FROM past_cases").fetchone()[0]

    st.info(f"📊 **データ状況**  \n"
            f"・past_cases: **{db_total} 件**  \n"
            f"・表示中（未登録）: **{total_count} 件**")

    if st.button("♻️ 画面を強制再読み込み", key="btn_reload_status"):
        st.rerun()

    if total_count == 0:
        st.success("全ての案件が登録済みです！")
        if db_total > 0:
            st.info("💡 登録済みの案件はここには表示されません。「履歴分析」画面等で確認できます。")
        return

    # --- 絞り込みフィルター & 一括削除 ---
    col_filters, col_bulk = st.columns([3, 1])
    with col_filters:
        search_kw = st.text_input("🔍 絞り込み検索", placeholder="会社名・業種・日付で絞り込み", key="search_register")
    with col_bulk:
        st.write("") # 縦位置合わせ
        if st.button("🔥 全件削除", key="btn_bulk_clear", type="secondary", help="表示されている未登録案件をすべてクリアします"):
            if st.session_state.get("confirm_clear_all"):
                try:
                    with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
                        conn.execute("DELETE FROM past_cases WHERE (final_status='未登録' OR final_status IS NULL)")
                        conn.commit()
                    st.success("✅ 未登録データを消去しました")
                    st.session_state.confirm_clear_all = False
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"消去失敗: {e}")
            else:
                st.session_state.confirm_clear_all = True
                st.warning("⚠️ 本当に全削除しますか？もう一度押すと実行します")

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

        # 確実にボタンが見えるように、カードの上に配置
        # 案件カードと削除ボタン
        def _exec_delete(target_id):
            try:
                with closing(sqlite3.connect(_LEASE_DB_PATH)) as conn:
                    conn.execute("DELETE FROM past_cases WHERE id=?", (str(target_id),))
                    conn.commit()
                return True
            except Exception as e:
                st.error(f"削除エラー: {e}")
                return False

        col_card, col_del = st.columns([10, 2])
        with col_del:
            st.write("") # 縦位置調整
            if st.button("🗑️ 案件削除", key=f"del_btn_{rec_id}_{idx}", type="secondary", use_container_width=True):
                if _exec_delete(rec_id):
                    st.toast(f"🗑️ 削除完了: {display_name}")
                    time.sleep(0.3)
                    st.rerun()

        with col_card:
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
                            from shinsa_gunshi import refresh_evidence_weights
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
