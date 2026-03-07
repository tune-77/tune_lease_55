import streamlit as st
import time
from data_cases import load_all_cases, save_all_cases

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
            
            with st.expander(f"{case.get('timestamp', '')[:16]} - {case.get('industry_sub', '')} (スコア: {score:.0f})"):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**判定**: {hantei}")
                    summary = case.get("chat_summary", "")
                    st.caption((summary[:100] + "...") if summary else "サマリなし")
                
                with c2:
                    if st.button("🗑️ この案件を削除", key=f"del_pending_{case_id}", type="secondary", help="未登録のままこの案件を一覧から削除します"):
                        all_cases = [c for c in load_all_cases() if c.get("id") != case_id]
                        if save_all_cases(all_cases):
                            st.success("削除しました")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("保存に失敗しました。")
                    with st.form(f"status_form_{i}"):
                        res_status = st.radio("結果", ["成約", "失注"], horizontal=True)
                        final_rate = st.number_input("獲得レート (%)", value=0.0, step=0.01, format="%.2f", help="成約した場合の決定金利")
                        past_base_rate = case.get("pricing", {}).get("base_rate", 1.2)
                        base_rate_input = st.number_input("当時の基準金利 (%)", value=past_base_rate, step=0.01, format="%.2f")
                        lost_reason = st.text_input("失注理由 (失注の場合のみ)", placeholder="例: 金利で他社に負けた")
                        loan_condition_options = ["金融機関と協調", "本件限度", "次回格付まで本件限度", "その他"]
                        loan_conditions = st.multiselect("融資条件", loan_condition_options, help="該当する条件を複数選択")
                        competitor_name = st.text_input("競合他社情報", placeholder="例: 〇〇銀行、〇〇リース")
                        competitor_rate = st.number_input("他社提示金利 (%)", value=0.0, step=0.01, format="%.2f", help="競合の提示条件があれば入力")
                        
                        if st.form_submit_button("登録する"):
                            target_id = case.get("id")
                            updated = False
                            for c in all_cases:
                                if c.get("id") == target_id:
                                    c["final_status"] = res_status
                                    c["final_rate"] = final_rate
                                    c["base_rate_at_time"] = base_rate_input
                                    if res_status == "成約" and final_rate > 0:
                                        c["winning_spread"] = final_rate - base_rate_input
                                    if res_status == "失注":
                                        c["lost_reason"] = lost_reason
                                    c["loan_conditions"] = loan_conditions
                                    c["competitor_name"] = competitor_name.strip() or ""
                                    c["competitor_rate"] = competitor_rate if competitor_rate else None
                                    updated = True
                                    break
                            
                            if updated:
                                if save_all_cases(all_cases):
                                    st.success("登録しました！")
                                    # 軍師DBへ自動同期
                                    try:
                                        from components.shinsa_gunshi import sync_from_lease_case
                                        _synced = next((c for c in all_cases if c.get("id") == target_id), None)
                                        if _synced:
                                            sync_from_lease_case(_synced)
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
