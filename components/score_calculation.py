import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
import time
import math
import datetime

from coeff_definitions import (
    BAYESIAN_PRIOR_EXTRA, STRENGTH_TAG_WEIGHTS, DEFAULT_STRENGTH_WEIGHT,
    COEFFS
)

from charts import _equity_ratio_display
from indicators import calculate_pd
from web_services import search_bankruptcy_trends, get_market_rate, get_stats
from rule_manager import evaluate_custom_rules
from data_cases import (
    get_effective_coeffs, get_score_weights, save_case_log,
    find_similar_past_cases
)
from constants import (
    APPROVAL_LINE, REVIEW_LINE, SCORE_PENALTY_IF_LEARNING_REJECT,
    QUALITATIVE_SCORING_CORRECTION_ITEMS, QUALITATIVE_SCORING_LEVELS, QUALITATIVE_SCORE_RANKS,
    STRENGTH_TAG_OPTIONS
)

def run_scoring(form_result, REQUIRED_FIELDS, benchmarks_data, hints_data, bankruptcy_data, jsic_data, avg_data, _rules, _SCRIPT_DIR):
    # Unpack form_result
    submitted_apply = form_result.get("submitted_apply")
    submitted_judge = form_result.get("submitted_judge")
    selected_major = form_result.get("selected_major")
    selected_sub = form_result.get("selected_sub")
    main_bank = form_result.get("main_bank")
    competitor = form_result.get("competitor")
    item9_gross = form_result.get("item9_gross")
    rieki = form_result.get("rieki")
    item4_ord_profit = form_result.get("item4_ord_profit")
    item5_net_income = form_result.get("item5_net_income")
    item10_dep = form_result.get("item10_dep")
    item11_dep_exp = form_result.get("item11_dep_exp")
    item8_rent = form_result.get("item8_rent")
    item12_rent_exp = form_result.get("item12_rent_exp")
    item6_machine = form_result.get("item6_machine")
    item7_other = form_result.get("item7_other")
    net_assets = form_result.get("net_assets")
    total_assets = form_result.get("total_assets")
    grade = form_result.get("grade")
    bank_credit = form_result.get("bank_credit")
    lease_credit = form_result.get("lease_credit")
    contracts = form_result.get("contracts")
    customer_type = form_result.get("customer_type")
    contract_type = form_result.get("contract_type")
    deal_source = form_result.get("deal_source")
    lease_term = form_result.get("lease_term")
    acceptance_year = form_result.get("acceptance_year")
    acquisition_cost = form_result.get("acquisition_cost")
    selected_asset_id = form_result.get("selected_asset_id")
    asset_score = form_result.get("asset_score", 0)
    asset_name = form_result.get("asset_name", "")
    nenshu = form_result.get("nenshu", 0)
    passion_text = form_result.get("passion_text", "")
    strength_tags = form_result.get("strength_tags", [])

    if submitted_judge or st.session_state.pop("_auto_judge", False):
        try:
            # フラグメント利用時用: session_state の値で上書き（入力ガタつき軽減のため）
            nenshu = st.session_state.get("nenshu", 0)
            item9_gross = st.session_state.get("item9_gross", 0)
            rieki = st.session_state.get("rieki", 0)
            item4_ord_profit = st.session_state.get("item4_ord_profit", 0)
            item5_net_income = st.session_state.get("item5_net_income", 0)
            item10_dep = st.session_state.get("item10_dep", 0)
            item11_dep_exp = st.session_state.get("item11_dep_exp", 0)
            item8_rent = st.session_state.get("item8_rent", 0)
            item12_rent_exp = st.session_state.get("item12_rent_exp", 0)
            item6_machine = st.session_state.get("item6_machine", 0)
            item7_other = st.session_state.get("item7_other", 0)
            net_assets = st.session_state.get("net_assets", 0)
            total_assets = st.session_state.get("total_assets", 0)
            bank_credit = st.session_state.get("bank_credit", 0)
            lease_credit = st.session_state.get("lease_credit", 0)
            contracts = st.session_state.get("contracts", 0)
            lease_term = st.session_state.get("lease_term", 0)
            acquisition_cost = st.session_state.get("acquisition_cost", 0)
            acceptance_year = st.session_state.get("acceptance_year", 2026)
        
            # 変数の再マッピング (None -> 0)
            nenshu = nenshu if nenshu is not None else 0
            item9_gross = item9_gross if item9_gross is not None else 0
            rieki = rieki if rieki is not None else 0
            item4_ord_profit = item4_ord_profit if item4_ord_profit is not None else 0
            item5_net_income = item5_net_income if item5_net_income is not None else 0
            item10_dep = item10_dep if item10_dep is not None else 0
            item11_dep_exp = item11_dep_exp if item11_dep_exp is not None else 0
            item8_rent = item8_rent if item8_rent is not None else 0
            item12_rent_exp = item12_rent_exp if item12_rent_exp is not None else 0
            item6_machine = item6_machine if item6_machine is not None else 0
            item7_other = item7_other if item7_other is not None else 0
            net_assets = net_assets if net_assets is not None else 0
            total_assets = total_assets if total_assets is not None else 0
            bank_credit = bank_credit if bank_credit is not None else 0
            lease_credit = lease_credit if lease_credit is not None else 0
            contracts = contracts if contracts is not None else 0
            lease_term = lease_term if lease_term is not None else 0
            acquisition_cost = acquisition_cost if acquisition_cost is not None else 0
    
            # 必須項目チェック（未入力・不正時は判定をブロック）
            validation_ok = True
            missing = []
            for key, label, cond in REQUIRED_FIELDS:
                val = locals().get(key)
                if not cond(val):
                    missing.append(label)
            if missing:
                validation_ok = False
                st.error(
                    f"**判定には次の必須項目を入力してください。**\n\n"
                    f"- 「{'」「'.join(missing)}」は **1以上** の値を入力してください。\n\n"
                    "売上高は比率計算に、総資産は自己資本比率・学習モデルに必要です。"
                )
            
            if validation_ok:
                # 指標計算
                user_op_margin = (rieki / nenshu * 100) if nenshu > 0 else 0.0
                user_equity_ratio = (net_assets / total_assets * 100) if total_assets > 0 else 0.0
                # 流動比率の簡易算（流動資産≈総資産−固定資産、流動負債≈負債総額）
                liability_total = total_assets - net_assets if (total_assets and net_assets is not None) else 0
                current_assets_approx = max(0, total_assets - item6_machine - item7_other)
                user_current_ratio = (current_assets_approx / liability_total * 100) if liability_total > 0 else 100.0
    
                bench = benchmarks_data.get(selected_sub, {})
                bench_op_margin = bench.get("op_margin", 0.0)
                bench_equity_ratio = _equity_ratio_display(bench.get("equity_ratio")) or 0.0
                bench_comment = bench.get("comment", "")
    
                comp_margin = "高い" if user_op_margin >= bench_op_margin else "低い"
                comp_equity = "高い" if user_equity_ratio >= bench_equity_ratio else "低い"
    
                comparison_text = f"""
                - **営業利益率**: {user_op_margin:.1f}% (業界目安: {bench_op_margin}%) → 平均より{comp_margin}
                - **自己資本比率**: {user_equity_ratio:.1f}% (業界目安: {bench_equity_ratio}%) → 平均より{comp_equity}
                - **業界特性**: {bench_comment}
                ※ **銀行与信・リース与信**は総銀行与信・総リース与信ではなく、**当社（弊社）の与信**である。判定・アドバイスではこの点を踏まえること。
                """
    
                my_hints = hints_data.get(selected_sub, {"subsidies": [], "risks": [], "mandatory": ""})
    
                # 財務ベース倒産確率と業界リスク検索（判定開始時に実行）
                pd_percent = calculate_pd(user_equity_ratio, user_current_ratio, user_op_margin)
                try:
                    network_risk_summary = search_bankruptcy_trends(selected_sub)
                except Exception as _e:
                    network_risk_summary = f"（業界リスクの取得でエラー: {_e}。判定は続行します。）"
    
                # ==========================================================================
                # 🧮 スコア計算ロジック
                # ==========================================================================
    
                # モデル計算用データ (単位調整版)
                data_scoring = {
                    # 対数項用 (千円単位のまま)
                    "nenshu": nenshu,             
                    "bank_credit": bank_credit,   
                    "lease_credit": lease_credit, 
        
                    # 線形項用 (百万円単位に変換) - 係数の桁から推測
                    "op_profit": rieki / 1000,
                    "ord_profit": item4_ord_profit / 1000,
                    "net_income": item5_net_income / 1000,
                    "gross_profit": item9_gross / 1000,
                    "machines": item6_machine / 1000,
                    "other_assets": item7_other / 1000,
                    "rent": item8_rent / 1000,
                    "depreciation": item10_dep / 1000,
                    "dep_expense": item11_dep_exp / 1000,
                    "rent_expense": item12_rent_exp / 1000,
        
                    # その他
                    "contracts": contracts,
                    "grade": grade,
                    "industry_major": selected_major,
                }
    
                # 安全なシグモイド関数 (オーバーフロー対策)
                def safe_sigmoid(x):
                    try:
                        # xが大きすぎる、または小さすぎる場合の対策
                        if x > 700: return 1.0
                        if x < -700: return 0.0
                        return 1 / (1 + math.exp(-x))
                    except OverflowError:
                        return 0.0 if x < 0 else 1.0
    
                def calculate_score_from_coeffs(data, coeff_set):
                    z = coeff_set["intercept"]
        
                    # ダミー変数の適用ロジック
                    major = data["industry_major"]
                    if "医療" in major or "福祉" in major or major.startswith("P"):
                        z += coeff_set.get("ind_medical", 0)
                    elif "運輸" in major or major.startswith("H"):
                        z += coeff_set.get("ind_transport", 0)
                    elif "建設" in major or major.startswith("D"):
                        z += coeff_set.get("ind_construction", 0)
                    elif "製造" in major or major.startswith("E"):
                        z += coeff_set.get("ind_manufacturing", 0)
                    elif "卸売" in major or "小売" in major or "サービス" in major or major[0] in ["I", "K", "M", "R"]:
                         z += coeff_set.get("ind_service", 0)
        
                    # 対数項 (千円単位の値を対数化)
                    if data["nenshu"] > 0: z += np.log1p(data["nenshu"]) * coeff_set.get("sales_log", 0)
                    if data["bank_credit"] > 0: z += np.log1p(data["bank_credit"]) * coeff_set.get("bank_credit_log", 0)
                    if data["lease_credit"] > 0: z += np.log1p(data["lease_credit"]) * coeff_set.get("lease_credit_log", 0)
        
                    # 線形項 (既に百万円単位に変換済みの値を使用)
                    z += data["op_profit"] * coeff_set.get("op_profit", 0)
                    z += data["ord_profit"] * coeff_set.get("ord_profit", 0)
                    z += data["net_income"] * coeff_set.get("net_income", 0)
                    z += data["machines"] * coeff_set.get("machines", 0)
                    z += data["other_assets"] * coeff_set.get("other_assets", 0)
                    z += data["rent"] * coeff_set.get("rent", 0)
                    z += data["gross_profit"] * coeff_set.get("gross_profit", 0)
                    z += data["depreciation"] * coeff_set.get("depreciation", 0)
                    z += data["dep_expense"] * coeff_set.get("dep_expense", 0)
                    z += data["rent_expense"] * coeff_set.get("rent_expense", 0)
        
                    if "4-6" in data["grade"]: z += coeff_set.get("grade_4_6", 0)
                    elif "要注意" in data["grade"]: z += coeff_set.get("grade_watch", 0)
                    elif "無格付" in data["grade"]: z += coeff_set.get("grade_none", 0)
        
                    z += data["contracts"] * coeff_set.get("contracts", 0)
        
                    # 指標モデル用の追加変数 (比率)
                    z += data.get("ratio_op_margin", 0) * coeff_set.get("ratio_op_margin", 0)
                    z += data.get("ratio_gross_margin", 0) * coeff_set.get("ratio_gross_margin", 0)
                    z += data.get("ratio_ord_margin", 0) * coeff_set.get("ratio_ord_margin", 0)
                    z += data.get("ratio_net_margin", 0) * coeff_set.get("ratio_net_margin", 0)
                    z += data.get("ratio_fixed_assets", 0) * coeff_set.get("ratio_fixed_assets", 0)
                    z += data.get("ratio_rent", 0) * coeff_set.get("ratio_rent", 0)
                    z += data.get("ratio_depreciation", 0) * coeff_set.get("ratio_depreciation", 0)
                    z += data.get("ratio_machines", 0) * coeff_set.get("ratio_machines", 0)
        
                    return z
    
                # 1. 全体モデル（成約/失注で更新した係数があればそれを優先）
                z_main = calculate_score_from_coeffs(data_scoring, get_effective_coeffs("全体_既存先"))
                score_prob = safe_sigmoid(z_main)
                score_percent = score_prob * 100
    
                # 2. 指標モデル (比率計算)
                # マッピングロジック更新 (CSV指示に基づく)
                # D, P, H -> 全体(指標)
                # I, K, M, R -> サービス業(指標)
                # E -> 製造業(指標)
    
                bench_key = "全体_指標"
                major_code_bench = selected_major.split(" ")[0]
    
                if major_code_bench == "D":
                    bench_key = "全体_指標"
                elif major_code_bench == "P":
                    bench_key = "医療_指標"
                elif major_code_bench == "H":
                    bench_key = "運送業_指標"
                elif major_code_bench in ["I", "K", "M", "R"]:
                    bench_key = "サービス業_指標"
                elif major_code_bench == "E":
                    bench_key = "製造業_指標"
        
                ratio_data = data_scoring.copy()
    
                # 比率計算のために元の千円単位の値を使う
                raw_nenshu = nenshu if nenshu > 0 else 1.0
    
                raw_op = rieki if rieki is not None else 0
                raw_gross = item9_gross if item9_gross is not None else 0
                raw_ord = item4_ord_profit if item4_ord_profit is not None else 0
                raw_net = item5_net_income if item5_net_income is not None else 0
                raw_fixed = (item6_machine if item6_machine is not None else 0) + (item7_other if item7_other is not None else 0)
                raw_rent = item12_rent_exp if item12_rent_exp is not None else 0
                raw_dep = (item10_dep if item10_dep is not None else 0) + (item11_dep_exp if item11_dep_exp is not None else 0)
                raw_machines = item6_machine if item6_machine is not None else 0
    
                ratio_data["ratio_op_margin"] = raw_op / raw_nenshu
                ratio_data["ratio_gross_margin"] = raw_gross / raw_nenshu
                ratio_data["ratio_ord_margin"] = raw_ord / raw_nenshu
                ratio_data["ratio_net_margin"] = raw_net / raw_nenshu
                ratio_data["ratio_fixed_assets"] = raw_fixed / raw_nenshu
                ratio_data["ratio_rent"] = raw_rent / raw_nenshu
                ratio_data["ratio_depreciation"] = raw_dep / raw_nenshu
                ratio_data["ratio_machines"] = raw_machines / raw_nenshu
    
                # 指標モデル計算（既存先/新規先で更新係数があれば使用）
                bench_key_with_type = f"{bench_key}_{'新規先' if customer_type == '新規先' else '既存先'}"
                bench_coeffs = get_effective_coeffs(bench_key_with_type)
                z_bench = calculate_score_from_coeffs(ratio_data, bench_coeffs)
                score_prob_bench = safe_sigmoid(z_bench)
                score_percent_bench = score_prob_bench * 100
    
                # 3. 業種別モデル (分類ロジックの修正)
                ind_key = "全体_既存先" # デフォルト
    
                major_code = selected_major.split(" ")[0] # "D 建設業" -> "D"
    
                # CSV定義に基づくマッピング
                # H -> 運送業
                # I, K, M, R -> サービス業
                # E -> 製造業
                # D, P -> 全体モデル (既存or新規)
    
                if major_code == "H":
                    ind_key = "運送業_既存先"
                elif major_code == "P":
                    ind_key = "医療_既存先"
                elif major_code in ["I", "K", "M", "R"]:
                    ind_key = "サービス業_既存先"
                elif major_code == "E":
                    ind_key = "製造業_既存先"
                elif major_code == "D":
                    ind_key = "全体_既存先"
    
                # 新規先の場合の切り替え
                if customer_type == "新規先":
                    ind_key = ind_key.replace("既存先", "新規先")
                    # 万が一キーがない場合は全体_新規先へフォールバック
                    if ind_key not in COEFFS: ind_key = "全体_新規先"
    
                ind_coeffs = get_effective_coeffs(ind_key)
                z_ind = calculate_score_from_coeffs(data_scoring, ind_coeffs)
                score_prob_ind = safe_sigmoid(z_ind)
                score_percent_ind = score_prob_ind * 100
    
                gap_val = score_percent - score_percent_bench
                gap_sign = "+" if gap_val >= 0 else ""
                gap_text = f"指標モデル差: {gap_sign}{gap_val:.1f}%"
    
                # ========== 完全版ベイズ初期モデル: 継承＋補完（回帰で更新した係数も反映） ==========
                effective = get_effective_coeffs()  # 成約/失注で更新した係数（既存+追加項目）があれば使用
                # 逆転の鍵は削除済み（定性は定性スコアリングのみ）
                strength_tags = []
                passion_text = ""
                n_strength = 0
                contract_prob = score_percent
                ai_completed_factors = []  # AIが補完した判定要因（表示・バトル用）
    
                # メイン先（係数: 成約/失注で回帰更新されていればその値、なければ既定5）
                # ※ 係数分析・更新モードで回帰分析を実行すると、成約/失注データから自動的に係数が再計算されます
                main_bank_eff = effective.get("main_bank", 5)
                if main_bank == "メイン先":
                    contract_prob += main_bank_eff
                    ai_completed_factors.append({"factor": "メイン取引先", "effect_percent": int(round(main_bank_eff)), "detail": "取引行として優位"})
    
                # 競合: 競合あり=負の係数、競合なし=プラス（成約/失注で回帰更新されていればその値、なければ既定）
                # ※ 係数分析・更新モードで回帰分析を実行すると、成約/失注データから自動的に係数が再計算されます
                comp_present_eff = effective.get("competitor_present", BAYESIAN_PRIOR_EXTRA["competitor_present"])
                comp_none_eff = effective.get("competitor_none", 5)
                comp_effect = comp_present_eff if competitor == "競合あり" else comp_none_eff
                contract_prob += comp_effect
                if competitor == "競合あり":
                    ai_completed_factors.append({"factor": "競合他社の存在", "effect_percent": int(round(comp_effect)), "detail": "他社がいる場合は成約率を下げる補正"})
                else:
                    ai_completed_factors.append({"factor": "競合なし", "effect_percent": int(round(comp_effect)), "detail": "競合優位で成約率を上げる補正"})
    
                # 業界景気動向: Z化（-1,0,1）。係数は更新値 or 既定
                _summary = (network_risk_summary or "").lower()
                if "景気" in _summary or "好調" in _summary or "拡大" in _summary or "堅調" in _summary:
                    industry_z = 1.0
                    ind_label = "業界動向（ポジティブ）"
                elif "倒産" in _summary or "減少" in _summary or "悪化" in _summary or "懸念" in _summary or "低下" in _summary:
                    industry_z = -1.0
                    ind_label = "業界動向（ネガティブ）"
                else:
                    industry_z = 0.0
                    ind_label = "業界動向（中立）"
                ind_coef = effective.get("industry_sentiment_z", BAYESIAN_PRIOR_EXTRA["industry_sentiment_per_z"])
                ind_effect = ind_coef * industry_z
                contract_prob += ind_effect
                if industry_z != 0:
                    ai_completed_factors.append({"factor": ind_label, "effect_percent": int(round(ind_effect)), "detail": "業界の景気動向を成約率に反映"})
    
                # 金利差は y_pred_adjusted 算出後に追加

                # 定性スコア: タグスコア(0-10)と熱意(0/1)。係数は「1ポイントあたり」「熱意ありで」の効果（更新値 or 既定）
                tag_score = min(sum(STRENGTH_TAG_WEIGHTS.get(t, DEFAULT_STRENGTH_WEIGHT) for t in strength_tags), 10)
                tag_coef = effective.get("qualitative_tag_score", 2.0)   # 1ptあたり%効果
                passion_coef = effective.get("qualitative_passion", BAYESIAN_PRIOR_EXTRA["qualitative_passion_bonus"])
                tag_effect = tag_coef * tag_score
                passion_effect = passion_coef if passion_text else 0
                contract_prob += tag_effect + passion_effect
                if n_strength > 0:
                    ai_completed_factors.append({"factor": "定性スコア（強みタグ）", "effect_percent": int(round(tag_effect)), "detail": f"特許・人脈等{n_strength}件を標準重みで加点"})
                if passion_effect > 0:
                    ai_completed_factors.append({"factor": "熱意・裏事情の記述", "effect_percent": int(round(passion_effect)), "detail": "記述ありで加点"})
    
                # 自己資本比率（追加項目）: 係数は「1%あたり」の効果（更新値 or 0）
                equity_coef = effective.get("equity_ratio", 0)
                equity_effect = equity_coef * user_equity_ratio
                contract_prob += equity_effect
                if abs(equity_effect) >= 0.5:
                    ai_completed_factors.append({"factor": "自己資本比率", "effect_percent": int(round(equity_effect)), "detail": f"自己資本比率 {user_equity_ratio:.1f}% を反映"})

                from bayesian_engine import THRESHOLD_APPROVAL
                approval_line = THRESHOLD_APPROVAL * 100
                
                # BNエンジン出力（スコア≤承認ライン かつ BN推論実行済の場合のみ有効）
                _bn_res_sc = st.session_state.get("_bn_s_result")
                if _bn_res_sc:
                    _bn_im_sc  = _bn_res_sc.get("intermediate") or {}
                    _bn_ap_v   = float(_bn_res_sc.get("approval_prob") or 0)
                    _bn_fc_v   = float(_bn_im_sc.get("Financial_Creditworthiness") or 0)
                    _bn_hc_v   = float(_bn_im_sc.get("Hedge_Condition") or 0)
                    _bn_av_v   = float(_bn_im_sc.get("Asset_Value") or 0)
                    _bn_ap_c   = effective.get("bn_approval_prob", 0)
                    _bn_fc_c   = effective.get("bn_fc", 0)
                    _bn_hc_c   = effective.get("bn_hc", 0)
                    _bn_av_c   = effective.get("bn_av", 0)
                    _bn_effect = (_bn_ap_c * _bn_ap_v + _bn_fc_c * _bn_fc_v
                                  + _bn_hc_c * _bn_hc_v + _bn_av_c * _bn_av_v)
                    contract_prob += _bn_effect
                    if abs(_bn_effect) >= 0.5:
                        ai_completed_factors.append({
                            "factor": "BN承認確率",
                            "effect_percent": int(round(_bn_effect)),
                            "detail": (f"BN承認確率{_bn_ap_v:.0%}・"
                                       f"財務{_bn_fc_v:.0%}・"
                                       f"ヘッジ{_bn_hc_v:.0%}・"
                                       f"物件{_bn_av_v:.0%}"),
                        })

                contract_prob = max(0, min(100, contract_prob))
    
                # 利回り予測計算 (簡略化)
                YIELD_COEFFS = {
                    "intercept": -132.213, "item10_dep": -5.2e-07, "item11_dep_exp": -5.9e-07,
                    "item12_rent_exp": -3.3e-07, "grade_1_3": 0.103051, "grade_4_6": 0.115129,
                    "grade_watch": 0.309849, "grade_none": 0.25737, "type_general": 0.032238,
                    "source_bank": 0.062498, "nenshu_log": -0.03134, "bank_credit_log": -0.00841,
                    "lease_credit_log": -0.02849, "term_log": -0.63635, "year": 0.067637,
                    "cost_log": -0.3945, "contracts_log": 0.130446
                }
    
                # 利回り予測モデルには「千円単位の生の数字」を使う (画像の例に従う)
                # ただし、対数項は log1p(千円) を使用
                y_pred = YIELD_COEFFS["intercept"]
                y_pred += item10_dep * YIELD_COEFFS["item10_dep"]
                y_pred += item11_dep_exp * YIELD_COEFFS["item11_dep_exp"]
                y_pred += item12_rent_exp * YIELD_COEFFS["item12_rent_exp"]
    
                if "1-3" in grade: y_pred += YIELD_COEFFS["grade_1_3"]
                elif "4-6" in grade: y_pred += YIELD_COEFFS["grade_4_6"]
                elif "要注意" in grade: y_pred += YIELD_COEFFS["grade_watch"]
                elif "無格付" in grade: y_pred += YIELD_COEFFS["grade_none"]
    
                if contract_type == "一般": y_pred += YIELD_COEFFS["type_general"]
                if deal_source == "銀行紹介": y_pred += YIELD_COEFFS["source_bank"]
    
                if nenshu > 0: y_pred += np.log1p(nenshu) * YIELD_COEFFS["nenshu_log"]
                if bank_credit > 0: y_pred += np.log1p(bank_credit) * YIELD_COEFFS["bank_credit_log"]
                if lease_credit > 0: y_pred += np.log1p(lease_credit) * YIELD_COEFFS["lease_credit_log"]
                if lease_term > 0: y_pred += np.log1p(lease_term) * YIELD_COEFFS["term_log"]
                if contracts > 0: y_pred += np.log1p(contracts) * YIELD_COEFFS["contracts_log"]
    
                val_cost_log = np.log1p(acquisition_cost) if acquisition_cost > 0 else 0
                y_pred += val_cost_log * YIELD_COEFFS["cost_log"]
                y_pred += acceptance_year * YIELD_COEFFS["year"]
    
                # 金利環境補正
                BASE_DATE = "2025-03"
                term_years = lease_term / 12
                base_market_rate = get_market_rate(BASE_DATE, term_years)
                today_str = datetime.date.today().strftime("%Y-%m")
                current_market_rate = get_market_rate(today_str, term_years)
                rate_diff = current_market_rate - base_market_rate
                y_pred_adjusted = y_pred + rate_diff

                # 金利差（競合比）: 係数は更新値 or 既定
                competitor_rate_val = st.session_state.get("competitor_rate")
                if competitor_rate_val is not None and isinstance(competitor_rate_val, (int, float)):
                    rate_diff_pt = float(y_pred_adjusted) - float(competitor_rate_val)
                    rate_z = max(-2, min(2, rate_diff_pt / 5.0))
                    rate_coef = effective.get("rate_diff_z", BAYESIAN_PRIOR_EXTRA["rate_diff_per_z"])
                    rate_effect = rate_coef * (-rate_z)
                    contract_prob += rate_effect
                    ai_completed_factors.append({"factor": "金利差（競合比）", "effect_percent": int(round(rate_effect)), "detail": f"自社が競合より{'有利' if rate_diff_pt < 0 else '不利'}な金利"})
                contract_prob = max(0, min(100, contract_prob))

                # 借手スコア + 物件スコア → 総合スコア（判定に反映）。重みは回帰最適化で変更可能。
                w_borrower, w_asset, w_quant, w_qual = get_score_weights()
                final_score = w_borrower * score_percent + w_asset * asset_score
                st.session_state['current_image'] = "approve" if final_score >= APPROVAL_LINE else "challenge"
        
                # [削除] AIアドバイス (1回目: 入力タブ側)
                # ここにあった ai_question 生成と messages 追加ロジックは削除し、
                # 分析結果タブでのみ参照するようにします。
                # ただし、裏でプロンプト生成だけはしておく必要があるため、セッションステートへの保存は残します。
    
                # 過去の類似案件（同業界・自己資本比率が近い）を最大3件取得
                similar_cases = find_similar_past_cases(selected_sub, user_equity_ratio, max_count=3)
                similar_cases_block = ""
                if similar_cases:
                    similar_cases_block = "【参考：過去の類似案件の結末】\n"
                    for i, sc in enumerate(similar_cases, 1):
                        res = sc.get("result") or {}
                        eq = res.get("user_eq")
                        sc_score = res.get("score")
                        status = sc.get("final_status", "未登録")
                        eq_str = f"{_equity_ratio_display(eq) or 0:.1f}%" if eq is not None else "—"
                        score_str = f"{sc_score:.1f}%" if sc_score is not None else "—"
                        similar_cases_block += f"{i}. 業界: {sc.get('industry_sub', '—')}、自己資本比率: {eq_str}、スコア: {score_str}、結末: {status}\n"
                    similar_cases_block += "\n"
                instruction_past = "過去に似た数値で承認された（または否決された）事例を参考にし、今回の案件との共通点や相違点を踏まえて、より精度の高い最終判定を出してください。\n\n"
    
                ai_question_text = ""
                if similar_cases_block:
                    ai_question_text += similar_cases_block + instruction_past
                # 過去の競合・成約金利をコンテキストとして追加（競合に勝つ対策をAIに促す）
                past_stats = get_stats(selected_sub)
                if past_stats.get("top_competitors_lost") or (past_stats.get("avg_winning_rate") is not None and past_stats["avg_winning_rate"] > 0):
                    ai_question_text += "【過去の競合・成約金利】\n"
                    if past_stats.get("top_competitors_lost"):
                        ai_question_text += "よく負ける競合: " + "、".join(past_stats["top_competitors_lost"][:5]) + "。\n"
                    if past_stats.get("avg_winning_rate") and past_stats["avg_winning_rate"] > 0:
                        ai_question_text += f"同業種の平均成約金利: {past_stats['avg_winning_rate']:.2f}%。\n"
                    ai_question_text += "上記を踏まえ、競合に勝つための対策も考慮してアドバイスしてください。\n\n"
                ai_question_text += "審査お疲れ様です。手元の決算書から、以下の**3点だけ**確認させてください。\n\n"
                from bayesian_engine import THRESHOLD_APPROVAL
                approval_line = THRESHOLD_APPROVAL * 100
                
                questions = []
                if my_hints.get("mandatory"): questions.append(f"🏭 **業界確認**: {my_hints['mandatory']}")
                if score_percent < approval_line: questions.append("💡 **実質利益**: 販管費の内訳に「役員報酬」は十分計上されていますか？")
                elif user_op_margin < bench_op_margin: questions.append("📉 **利益率要因**: 今期の利益率低下は、一過性ですか？")
                if score_percent < approval_line: questions.append("🏦 **資金繰り**: 借入金明細表で、返済が「約定通り」進んでいるか確認してください。")
                if my_hints["risks"]: questions.append(f"⚠️ **業界リスク**: {my_hints['risks'][0]} はクリアしていますか？")
        
                for q in questions[:3]: ai_question_text += f"- {q}\n"
                ai_question_text += "\nこれらがクリアになれば、承認確率80%以上が見込めます。"
                ai_question_text += f"\n\n業界の最新リスク情報も参照済みです。これらを総合して最終的なリスクと承認可否を判断してください。"
    
                # チャット履歴に追加 (表示は分析タブのチャット欄で行う)
                st.session_state.messages = [{"role": "assistant", "content": ai_question_text}]
                st.session_state.debate_history = [] 
    
                # 議論終了・判定プロンプト用に類似案件ブロックを保持
                similar_past_for_prompt = (similar_cases_block + instruction_past) if similar_cases_block else ""
    
                # 定性ワンホット（過去データ・RAG用）
                qualitative_onehot = {tag: 1 for tag in STRENGTH_TAG_OPTIONS if tag in strength_tags}
                qualitative_onehot.update({tag: 0 for tag in STRENGTH_TAG_OPTIONS if tag not in strength_tags})

                # 定性スコアリングの集計（総合×60%＋定性×40%でランクA〜E）
                qual_correction_items = {}
                qual_weight_sum = 0
                qual_weighted_total = 0.0
                for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
                    idx = st.session_state.get(f"qual_corr_{item['id']}", 0)
                    opts = item.get("options") or QUALITATIVE_SCORING_LEVELS
                    val = opts[idx - 1][0] if 1 <= idx <= len(opts) else None
                    level_label = opts[idx - 1][1] if 1 <= idx <= len(opts) else None
                    qual_correction_items[item["id"]] = {
                        "value": val,
                        "label": item["label"],
                        "weight": item["weight"],
                        "level_label": level_label,
                    }
                    if val is not None:
                        qual_weight_sum += item["weight"]
                        qual_weighted_total += (val / 4.0) * 100 * (item["weight"] / 100.0)
                qual_weighted_score = round((qual_weighted_total / qual_weight_sum * 100) if qual_weight_sum > 0 else 0)
                qual_weighted_score = min(100, max(0, qual_weighted_score))
                # ランクA〜Eは総合×重み＋定性×重みに基づく（重みは回帰最適化で変更可能）
                combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                combined_score = min(100, max(0, combined_score))
                qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                qualitative_scoring_correction = None
                if qual_weight_sum > 0:
                    qualitative_scoring_correction = {
                        "items": qual_correction_items,
                        "weighted_score": qual_weighted_score,
                        "combined_score": combined_score,
                        "rank": qual_rank["label"],
                        "rank_text": qual_rank["text"],
                        "rank_desc": qual_rank["desc"],
                    }

                # 学習モデル（業種別ハイブリッド）の予測（総資産・純資産が入力されている場合のみ）
                scoring_result = None
                if (total_assets or 0) > 0 and (net_assets is not None) and (net_assets >= 0):
                    try:
                        _scoring_dir = os.path.join(_SCRIPT_DIR, "scoring")
                        if _scoring_dir not in sys.path:
                            sys.path.insert(0, _SCRIPT_DIR)
                        from scoring.predict_one import predict_one, map_industry_major_to_scoring
                        _base = os.environ.get("LEASE_SCORING_MODELS_DIR", os.path.join(_SCRIPT_DIR, "scoring", "models", "industry_specific"))
                        _industry = map_industry_major_to_scoring(selected_major)
                        scoring_result = predict_one(
                            revenue=(nenshu or 0) * 1000,
                            total_assets=(total_assets or 0) * 1000,
                            equity=(net_assets or 0) * 1000,
                            operating_profit=(rieki or 0) * 1000,
                            net_income=(item5_net_income or 0) * 1000,
                            machinery_equipment=(item6_machine or 0) * 1000,
                            other_fixed_assets=(item7_other or 0) * 1000,
                            depreciation=((item10_dep or 0) + (item11_dep_exp or 0)) * 1000,
                            rent_expense=(item12_rent_exp or 0) * 1000,
                            industry=_industry,
                            base_path=_base,
                        )
                    except Exception:
                        scoring_result = None

                # 学習モデル判定が「否決」のときはすべてのスコアを50%減
                if scoring_result and (scoring_result.get("decision") or "").strip() == "否決":
                    final_score = final_score * SCORE_PENALTY_IF_LEARNING_REJECT
                    contract_prob = contract_prob * SCORE_PENALTY_IF_LEARNING_REJECT
                    score_percent = score_percent * SCORE_PENALTY_IF_LEARNING_REJECT
                    score_percent_bench = (score_percent_bench or 0) * SCORE_PENALTY_IF_LEARNING_REJECT
                    score_percent_ind = (score_percent_ind or 0) * SCORE_PENALTY_IF_LEARNING_REJECT
                    # 定性スコアリングの合計・ランクも否決後の総合で再計算
                    if qualitative_scoring_correction:
                        combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                        combined_score = min(100, max(0, combined_score))
                        qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                        qualitative_scoring_correction["combined_score"] = combined_score
                        qualitative_scoring_correction["rank"] = qual_rank["label"]
                        qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                        qualitative_scoring_correction["rank_desc"] = qual_rank["desc"]

                # デフォルト率50%以上の場合、総合スコアから-50点
                if pd_percent >= 50:
                    final_score = max(0, final_score - 50)
                    if qualitative_scoring_correction:
                        combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                        combined_score = min(100, max(0, combined_score))
                        qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                        qualitative_scoring_correction["combined_score"] = combined_score
                        qualitative_scoring_correction["rank"] = qual_rank["label"]
                        qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                        qualitative_scoring_correction["rank_desc"] = qual_rank["desc"]

                # 新しい審査を実行したのでチャット履歴をリセット
                st.session_state["messages"] = []
                st.session_state["debate_history"] = []

                # カスタムルールの適用
                custom_rules_list = _rules.get("custom_rules", [])
                if custom_rules_list:
                    context_data = {
                        "industry": selected_major,
                        "nenshu": nenshu,
                        "op_profit": rieki,
                        "ord_profit": item4_ord_profit,
                        "net_income": item5_net_income,
                        "net_assets": net_assets,
                        "total_assets": total_assets,
                        "user_eq_ratio": user_equity_ratio,
                        "term": lease_term,
                        "cost": acquisition_cost,
                        "bank_credit": bank_credit,
                        "lease_credit": lease_credit
                    }
                    cr_result = evaluate_custom_rules(custom_rules_list, context_data)
                    
                    # ペナルティ（減点）の適用
                    if cr_result["score_delta"] != 0:
                        delta = cr_result["score_delta"]
                        final_score = max(0, final_score + delta)
                    
                    # リストに理由を追加し、AI用プロンプトにも結合
                    if cr_result.get("applied_reasons"):
                        for reason in cr_result["applied_reasons"]:
                            ai_completed_factors.append({
                                "factor": "社内独自ルール",
                                "effect_percent": int(round(cr_result["score_delta"])),
                                "detail": reason
                            })
                            similar_past_for_prompt += f"\n【重要: 社内カスタムルール発動】\n- {reason}\n※AIは、このルールに抵触したことを踏まえて、顧客への改善アドバイスや留意点をアクションプランに含めてください。\n"
                            
                    # 定性スコアリングの合計等も再計算
                    if qualitative_scoring_correction:
                        combined_score = round(final_score * w_quant + qual_weighted_score * w_qual)
                        combined_score = min(100, max(0, combined_score))
                        qual_rank = next((r for r in QUALITATIVE_SCORE_RANKS if combined_score >= r["min"]), QUALITATIVE_SCORE_RANKS[-1])
                        qualitative_scoring_correction["combined_score"] = combined_score
                        qualitative_scoring_correction["rank"] = qual_rank["label"]
                        qualitative_scoring_correction["rank_text"] = qual_rank["text"]
                    
                    # 強制ステータスの保持（後段の `hantei` トグル等で利用）
                    forced_custom_status = cr_result.get("forced_status")
                else:
                    forced_custom_status = None

                # 将来予測シミュレーション（AI連携用の事前計算）
                future_sim_result = None
                if (nenshu or 0) > 0:
                    try:
                        from future_simulation import run_business_simulation
                        future_sim_result = run_business_simulation(
                            current_sales=nenshu,
                            current_op_profit=rieki,
                            drift=0.01,
                            volatility=0.15,
                            years=5,
                            n_simulations=5000
                        )
                    except Exception:
                        pass

                st.session_state['last_result'] = {
                    "score": final_score, "hantei": "承認圏内" if final_score >= APPROVAL_LINE else "要審議",
                    "score_borrower": score_percent, "asset_score": asset_score, "asset_name": asset_name,
                    "contract_prob": contract_prob, "z": z_main,
                    "ai_completed_factors": ai_completed_factors,
                    "comparison": comparison_text,
                    "user_op": user_op_margin, "bench_op": bench_op_margin,
                    "user_eq": user_equity_ratio, "bench_eq": bench_equity_ratio,
                    "hints": my_hints,
                    "pd_percent": pd_percent,
                    "network_risk_summary": network_risk_summary,
                    "similar_past_cases_prompt": similar_past_for_prompt,
                    "strength_tags": strength_tags,
                    "passion_text": passion_text,
                    "qualitative_onehot": qualitative_onehot,
                    "scoring_result": scoring_result,
                    "future_sim_result": future_sim_result,
                    "qualitative_scoring_correction": qualitative_scoring_correction,
                    "financials": {
                        "nenshu": nenshu,
                        "rieki": rieki,
                        "assets": total_assets,
                        "net_assets": net_assets,
                        "gross_profit": item9_gross,
                        "op_profit": rieki,
                        "ord_profit": item4_ord_profit,
                        "net_income": item5_net_income,
                        "machines": item6_machine,
                        "other_assets": item7_other,
                        "bank_credit": bank_credit,
                        "lease_credit": lease_credit,
                        "depreciation": item10_dep,
                    },
                    "yield_pred": y_pred_adjusted, "yield_base": y_pred, "rate_diff": rate_diff,
                    "gap_text": gap_text, "bench_score": score_percent_bench,
                    "ind_score": score_percent_ind, "ind_name": ind_key,
                    "industry_major": selected_major,
                    "industry_sub": selected_sub,
                    "industry_sentiment_z": industry_z,
                }

                # カスタムルールの影響や強制ステータスの上書き
                if forced_custom_status:
                    st.session_state['last_result']["hantei"] = forced_custom_status
                    st.session_state['current_image'] = "challenge" if forced_custom_status in ["要審議", "否決"] else "approve"
                elif final_score < REVIEW_LINE:
                    st.session_state['last_result']["hantei"] = "否決"
                    st.session_state['current_image'] = "challenge"
                
                # 審査委員会カードバトル用データ（分析タブで表示）
                hp_card = int(min(999, max(1, net_assets / 1000))) if net_assets else int(min(999, max(1, user_equity_ratio * 5)))
                atk_card = int(min(99, max(1, user_op_margin * 2)))
                spd_card = int(min(99, max(1, user_current_ratio / 2)))
                is_approved = final_score >= APPROVAL_LINE
                # 補完要因をスキル・環境効果としてバトルに渡す
                env_effects = [f"{f['factor']}: {f['effect_percent']:+.0f}%" for f in ai_completed_factors]
                st.session_state["battle_data"] = {
                    "hp": hp_card, "atk": atk_card, "spd": spd_card,
                    "is_approved": is_approved,
                    "special_move_name": None, "special_effect": None,
                    "battle_log": [], "dice": None,
                    "score": final_score, "hantei": "承認圏内" if is_approved else "要審議",  # is_approved = (final_score >= APPROVAL_LINE)
                    "environment_effects": env_effects,
                    "ai_completed_factors": ai_completed_factors,
                }
                st.session_state["show_battle"] = False  # 別枠（開発中）のため判定後はダッシュボードへ直行

                # ログ保存 (自動)
                log_payload = {
                    "industry_major": selected_major,
                    "industry_sub": selected_sub,
                    "customer_type": customer_type,
                    "main_bank": main_bank,
                    "competitor": competitor,
                    "competitor_rate": st.session_state.get("competitor_rate"),
                    "inputs": {
                        "nenshu": nenshu,
                        "gross_profit": item9_gross,
                        "op_profit": rieki,
                        "ord_profit": item4_ord_profit,
                        "net_income": item5_net_income,
                        "machines": item6_machine,
                        "other_assets": item7_other,
                        "rent": item8_rent,
                        "depreciation": item10_dep,
                        "dep_expense": item11_dep_exp,
                        "rent_expense": item12_rent_exp,
                        "bank_credit": bank_credit,
                        "lease_credit": lease_credit,
                        "contracts": contracts,
                        "grade": grade,
                        "contract_type": contract_type,
                        "deal_source": deal_source,
                        "lease_term": lease_term,
                        "acceptance_year": acceptance_year,
                        "acquisition_cost": acquisition_cost,
                        "lease_asset_id": selected_asset_id,
                        "lease_asset_name": asset_name,
                        "lease_asset_score": asset_score,
                        "qualitative": {
                            "strength_tags": strength_tags,
                            "passion_text": passion_text,
                            "onehot": qualitative_onehot,
                        },
                        "qualitative_scoring": qualitative_scoring_correction,
                    },
                    "result": st.session_state['last_result'],
                    "pricing": {
                        "base_rate": 1.2,
                        "pred_rate": y_pred_adjusted
                    },
                    "bn_engine": {
                        "evidence": st.session_state.get("_bn_s_evidence"),
                        "approval_prob": st.session_state.get("_bn_s_result", {}).get("approval_prob") if st.session_state.get("_bn_s_result") else None,
                        "decision": st.session_state.get("_bn_s_result", {}).get("decision") if st.session_state.get("_bn_s_result") else None,
                        "intermediate": st.session_state.get("_bn_s_result", {}).get("intermediate") if st.session_state.get("_bn_s_result") else None,
                    } if st.session_state.get("_bn_s_result") else None,
                }
                # 案件ログを保存し、案件IDをセッションに保持しておく
                case_id = save_case_log(log_payload)
                if case_id is None:
                    st.error("ログ保存に失敗しました。")
                else:
                    st.session_state["current_case_id"] = case_id
                    # 戻ったときにクリアされないよう、今回の入力値をすべて保存（訂正で戻ったときに復元）
                    submitted_qual_corr = {f"qual_corr_{item['id']}": st.session_state.get(f"qual_corr_{item['id']}", 0) for item in QUALITATIVE_SCORING_CORRECTION_ITEMS}
                    st.session_state["last_submitted_inputs"] = {
                        "nenshu": nenshu, "item9_gross": item9_gross, "rieki": rieki,
                        "item4_ord_profit": item4_ord_profit, "item5_net_income": item5_net_income,
                        "item10_dep": item10_dep, "item11_dep_exp": item11_dep_exp,
                        "item8_rent": item8_rent, "item12_rent_exp": item12_rent_exp,
                        "item6_machine": item6_machine, "item7_other": item7_other,
                        "net_assets": net_assets, "total_assets": total_assets,
                        "bank_credit": bank_credit, "lease_credit": lease_credit,
                        "contracts": contracts, "lease_term": lease_term,
                        "acquisition_cost": acquisition_cost, "acceptance_year": acceptance_year,
                        "selected_major": selected_major, "selected_sub": selected_sub,
                        "grade": grade, "main_bank": main_bank, "competitor": competitor,
                        "customer_type": customer_type, "contract_type": contract_type,
                        "deal_source": deal_source,
                        "selected_asset_index": st.session_state.get("selected_asset_index", 0),
                        **submitted_qual_corr,
                    }
                    st.session_state["form_restored_from_submit"] = False
                    st.session_state.nav_index = 1  # 1番目（分析結果）に切り替える
                    st.session_state["_jump_to_analysis"] = True  # 判定直後の1回だけ分析結果に飛ぶ
                    # AI自動所見・ワンタップ質問を有効化（ai_consultation.pyで生成）
                    st.session_state["_need_auto_comment"] = True
                    st.session_state["auto_ai_comment"] = None
                    st.rerun()  # 画面を読み込み直して、実際にタブを移動させる
        except Exception as e:
            st.error("判定開始の処理中にエラーが発生しました。入力内容を確認するか、ページを再読み込みして再度お試しください。")
            import traceback
            with st.expander("エラー詳細", expanded=False):
                st.code(traceback.format_exc())
