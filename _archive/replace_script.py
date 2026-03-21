import re

file_path = "lease_logic_sumaho12.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 📝 審査入力 の開始から </form> または submit ボタンまでの範囲を探す
# かなり長いブロックなので、特有の文字列から別の文字列までを置換する方針
start_marker = '            if nav_mode == "📝 審査入力":\n                st.header("📝 1. 審査データの入力")'
end_marker = '                    submitted_judge = st.form_submit_button("判定開始", type="primary", use_container_width=True)'

replacement = """            if nav_mode == "📝 審査入力":
                from components.form_apply import render_apply_form
                form_result = render_apply_form(
                    jsic_data, 
                    get_image,
                    get_stats, 
                    scrape_article_text, 
                    is_japanese_text,
                    append_case_news,
                    _fragment_nenshu
                )
                submitted_apply = form_result["submitted_apply"]
                submitted_judge = form_result["submitted_judge"]
                selected_major = form_result["selected_major"]
                selected_sub = form_result["selected_sub"]
                main_bank = form_result["main_bank"]
                competitor = form_result["competitor"]
                item9_gross = form_result["item9_gross"]
                rieki = form_result["rieki"]
                item4_ord_profit = form_result["item4_ord_profit"]
                item5_net_income = form_result["item5_net_income"]
                item10_dep = form_result["item10_dep"]
                item11_dep_exp = form_result["item11_dep_exp"]
                item8_rent = form_result["item8_rent"]
                item12_rent_exp = form_result["item12_rent_exp"]
                item6_machine = form_result["item6_machine"]
                item7_other = form_result["item7_other"]
                net_assets = form_result["net_assets"]
                total_assets = form_result["total_assets"]
                grade = form_result["grade"]
                bank_credit = form_result["bank_credit"]
                lease_credit = form_result["lease_credit"]
                contracts = form_result["contracts"]
                customer_type = form_result["customer_type"]
                contract_type = form_result["contract_type"]
                deal_source = form_result["deal_source"]
                lease_term = form_result["lease_term"]
                acceptance_year = form_result["acceptance_year"]
                acquisition_cost = form_result["acquisition_cost"]
                selected_asset_id = form_result["selected_asset_id"]
                asset_score = form_result["asset_score"]
                asset_name = form_result["asset_name"]"""

# 文字列の検索と置換
start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_marker)
    new_content = content[:start_idx] + replacement + content[end_idx:]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("置換成功")
else:
    print("マーカーが見つかりません。")
    print(f"start_idx: {start_idx}, end_idx: {end_idx}")

