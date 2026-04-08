import streamlit as st
import time
from data_cases import load_all_cases, delete_case, update_case

def render_status_registration():
    """案件結果登録 (成約/失注) タブのUIとロジックを描画する"""
    st.title("📝 案件結果登録")
    st.info("過去の審査案件に対して、最終的な結果（成約・失注）を登録します。")
    
    all_cases = load_all_cases()
    if not all_cases:
        st.warning("登録された案件がありません。")
    else:
        st.subheader("未登録の案件")
        pending_cases = [c for c in all_cases if c.get("final_status") == "未登録"]
        
        if not pending_cases:
            st.success("全ての案件が登録済みです！")
        
        for i, case in enumerate(reversed(pending_cases[-5:])): 
            case_id = case.get("id", "")
            res = case.get("result", {})
            score = res.get("score", 0)
            hantei = res.get("hantei", "不明")
            company_no = case.get("company_no") or case.get("inputs", {}).get("company_no") or "NO DATA"
            company_name = case.get("company_name") or case.get("inputs", {}).get("company_name") or ""
            display_name = f"#{company_no} {company_name}".strip()
            
            with st.expander(f"🏢 {display_name} | {case.get('timestamp', '')[:16]} | スコア: {score:.1f}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**判定**: {hantei}")
                    summary = case.get("chat_summary", "")
                    st.caption((summary[:100] + "...") if summary else "サマリなし")
                
                with c2:
                    if st.button("🗑️ この案件を削除", key=f"del_pending_{case_id}", type="secondary", help="未登録のままこの案件を一覧から削除します"):
                        if delete_case(case_id):
                            st.success("削除しました")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("削除に失敗しました。")
                    with st.form(f"status_form_{i}"):
                        res_status = st.radio("結果", ["成約", "失注"], horizontal=True)
                        final_rate = st.number_input("獲得レート (%)", value=0.0, step=0.01, format="%.2f", help="成約した場合の決定金利")
                        past_base_rate = case.get("pricing", {}).get("base_rate", 2.1)
                        base_rate_input = st.number_input("当時の基準金利 (%)", value=past_base_rate, step=0.01, format="%.2f")
                        lost_reason = st.text_input("失注理由 (失注の場合のみ)", placeholder="例: 金利で他社に負けた")
                        loan_condition_options = ["本件限度", "次回決算まで本件限度", "金融機関と協調", "独立・新設向け条件", "親会社等保証", "担保・保全あり", "その他"]
                        loan_conditions = st.multiselect("成約/承認の具体的条件", loan_condition_options, help="成約時に実際に付与された条件を選択してください。これが将来の類似案件の『決め手』として参照されます。")
                        competitor_name = st.text_input("競合他社情報", placeholder="例: 〇〇銀行、〇〇リース")
                        competitor_rate = st.number_input("他社提示金利 (%)", value=0.0, step=0.01, format="%.2f", help="競合の提示条件があれば入力")
                        
                        if st.form_submit_button("登録する"):
                            if res_status == "成約" and final_rate == 0.0:
                                st.warning("💡 獲得レートを入力すると成約分析の精度が向上します")
                            if res_status == "失注" and lost_reason.strip() == "":
                                st.warning("💡 失注理由を入力すると定性分析の精度が向上します")
                            target_id = case.get("id")
                            patches = {
                                "final_status": res_status,
                                "final_rate": final_rate,
                                "base_rate_at_time": base_rate_input,
                                "loan_conditions": loan_conditions,
                                "competitor_name": competitor_name.strip() or "",
                                "competitor_rate": competitor_rate if competitor_rate else None,
                            }
                            if res_status == "成約" and final_rate > 0:
                                patches["winning_spread"] = final_rate - base_rate_input
                            if res_status == "失注":
                                patches["lost_reason"] = lost_reason

                            if update_case(target_id, patches):
                                st.success("登録しました！")
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
                                        _auc_str = f"　AUC: {_auc:.3f}" if _auc else ""
                                        st.success(f"🧠 係数を自動更新しました（{_opt['n_cases']}件{_auc_str}）")
                                    else:
                                        _s = get_training_status()
                                        if _s["phase"] == "waiting":
                                            st.caption(f"📊 初回学習まであと {_s['next_trigger']}件")
                                        elif _s["phase"] == "active":
                                            st.caption(f"📊 次回更新まであと {_s['next_trigger']}件")
                                except Exception:
                                    pass
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("保存に失敗しました。")
