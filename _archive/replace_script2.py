import re

file_path = "lease_logic_sumaho12.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

start_marker = '            # ── クイック再入力パネル（全項目） ───────────────────────────\n            with st.expander("✏️ 全項目編集して再判定", expanded=False):\n                st.caption("すべての入力項目をここから変更できます。「🔄 再判定」で即座に再計算します。")'
end_marker = '                    st.session_state["item6_machine"] = _q_machine\n                    st.session_state["item7_other"] = _q_other\n                    st.session_state["net_assets"] = _q_net\n                    st.session_state["total_assets"] = _q_total\n                    # 信用情報\n                    st.session_state["grade"] = _q_grade\n                    st.session_state["bank_credit"] = _q_bank\n                    st.session_state["lease_credit"] = _q_lease\n                    st.session_state["contracts"] = _q_contracts\n                    # 契約条件\n                    st.session_state["customer_type"] = _q_ctype\n                    st.session_state["contract_type"] = _q_contract_type\n                    st.session_state["deal_source"] = _q_deal_source\n                    st.session_state["lease_term"] = _q_lease_term\n                    st.session_state["acceptance_year"] = _q_acceptance_year\n                    st.session_state["acquisition_cost"] = _q_acq\n                    if _q_asset_sel is not None:\n                        st.session_state["selected_asset_index"] = _q_asset_sel\n                    # 定性スコアリング\n                    for _qid, _qval in _q_qual.items():\n                        st.session_state[f"qual_corr_{_qid}"] = _qval\n                    # チャット履歴をリセット（新しい判定なので前の会話を引き継がない）\n                    st.session_state["messages"] = []\n                    st.session_state["debate_history"] = []\n                    # 判定トリガー\n                    st.session_state["_auto_judge"] = True\n                    st.session_state["_nav_pending"] = "📝 審査入力"\n                    st.rerun()'

replacement = """            # ── クイック再入力パネル（全項目） ───────────────────────────
            with st.expander("✏️ 全項目編集して再判定", expanded=False):
                from components.form_apply import render_quick_edit_panel
                quick_res = render_quick_edit_panel(jsic_data)
                
                if quick_res["rejudge_clicked"]:
                    # 業種
                    st.session_state["select_major"] = quick_res["q_major"]
                    st.session_state["select_sub"] = quick_res["q_sub"]
                    # P/L
                    st.session_state["nenshu"] = quick_res["q_nenshu"]
                    st.session_state["item9_gross"] = quick_res["q_gross"]
                    st.session_state["rieki"] = quick_res["q_rieki"]
                    st.session_state["item4_ord_profit"] = quick_res["q_ord"]
                    st.session_state["item5_net_income"] = quick_res["q_net_income"]
                    # 資産・経費
                    st.session_state["item10_dep"] = quick_res["q_dep"]
                    st.session_state["item11_dep_exp"] = quick_res["q_dep_exp"]
                    st.session_state["item8_rent"] = quick_res["q_rent"]
                    st.session_state["item12_rent_exp"] = quick_res["q_rent_exp"]
                    st.session_state["item6_machine"] = quick_res["q_machine"]
                    st.session_state["item7_other"] = quick_res["q_other"]
                    st.session_state["net_assets"] = quick_res["q_net"]
                    st.session_state["total_assets"] = quick_res["q_total"]
                    # 信用情報
                    st.session_state["grade"] = quick_res["q_grade"]
                    st.session_state["bank_credit"] = quick_res["q_bank"]
                    st.session_state["lease_credit"] = quick_res["q_lease"]
                    st.session_state["contracts"] = quick_res["q_contracts"]
                    # 契約条件
                    st.session_state["customer_type"] = quick_res["q_ctype"]
                    st.session_state["contract_type"] = quick_res["q_contract_type"]
                    st.session_state["deal_source"] = quick_res["q_deal_source"]
                    st.session_state["lease_term"] = quick_res["q_lease_term"]
                    st.session_state["acceptance_year"] = quick_res["q_acceptance_year"]
                    st.session_state["acquisition_cost"] = quick_res["q_acq"]
                    if quick_res["q_asset_sel"] is not None:
                        st.session_state["selected_asset_index"] = quick_res["q_asset_sel"]
                    # 定性スコアリング
                    for _qid, _qval in quick_res["q_qual"].items():
                        st.session_state[f"qual_corr_{_qid}"] = _qval
                    # チャット履歴をリセット（新しい判定なので前の会話を引き継がない）
                    st.session_state["messages"] = []
                    st.session_state["debate_history"] = []
                    # 判定トリガー
                    st.session_state["_auto_judge"] = True
                    st.session_state["_nav_pending"] = "📝 審査入力"
                    st.rerun()"""

# 文字列の検索と置換
start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_marker)
    new_content = content[:start_idx] + replacement + content[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("置換成功2")
else:
    print("マーカーが見つかりません。")
    print(f"start_idx: {start_idx}, end_idx: {end_idx}")

