import streamlit as st
import datetime
import pandas as pd
from data_cases import load_all_cases
from analysis_regression import run_contract_driver_analysis
from report_pdf import build_contract_report_pdf

def render_dashboard():
    """📊 履歴分析・実績ダッシュボード タブのUIとロジックを描画する"""
    st.title("📊 履歴分析・実績ダッシュボード")
    analysis = run_contract_driver_analysis()
    
    if analysis is None:
        st.warning("成約データが5件以上貯まると表示されます。結果登録で「成約」を登録してください。")
    else:
        n = analysis["closed_count"]
        st.success(f"成約 {n} 件を分析しました。")
        try:
            pdf_bytes = build_contract_report_pdf(analysis)
            filename = f"成約の正体レポート_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button("📥 分析結果をPDFでダウンロード", data=pdf_bytes, file_name=filename, mime="application/pdf", key="dl_contract_report_pdf")
        except Exception as e:
            st.caption(f"PDF生成をスキップしました: {e}")
        st.divider()
        
        # ---------- 成約要因分析 ----------
        st.subheader("📈 成約要因分析")
        st.caption("成約した案件だけを抽出し、共通項と成約に効く因子を分析した結果です。")
        st.markdown("**成約に最も寄与している上位3つの因子（ドライバー）**")
        for i, d in enumerate(analysis["top3_drivers"], 1):
            st.markdown(f"**{i}. {d['label']}** … 係数 {d['coef']:.4f}（{d['direction']}に効く）")
        st.divider()
        
        st.subheader("成約案件の平均的な財務数値")
        if analysis["avg_financials"]:
            rows = []
            for k, v in analysis["avg_financials"].items():
                if "自己資本" in k:
                    rows.append({"指標": k, "平均値": f"{v:.1f}%"})
                elif isinstance(v, float) and abs(v) >= 1:
                    rows.append({"指標": k, "平均値": f"{v:,.0f}"})
                else:
                    rows.append({"指標": k, "平均値": f"{v:.4f}"})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        else:
            st.caption("財務データが取得できませんでした。")
        st.divider()
        
        st.subheader("成約案件で頻出する定性タグ（ランキング）")
        if analysis["tag_ranking"]:
            for rank, (tag, count) in enumerate(analysis["tag_ranking"], 1):
                st.markdown(f"{rank}. **{tag}** … {count}件")
        else:
            st.caption("定性タグの登録がありません。")
            
        # 定性スコアリングの集計（成約案件）
        st.divider()
        st.subheader("成約案件の定性スコアリング")
        qs = analysis.get("qualitative_summary")
        if qs and (qs.get("avg_weighted") is not None or qs.get("avg_combined") is not None or qs.get("rank_distribution")):
            n_qual = qs.get("n_with_qual", 0)
            st.caption(f"成約{n}件のうち、定性スコアリングを入力していた案件: **{n_qual}件**")
            if qs.get("avg_weighted") is not None:
                st.metric("定性スコア（加重）の平均", f"{qs['avg_weighted']:.1f} / 100", help="項目別5段階の加重平均")
            if qs.get("avg_combined") is not None:
                st.metric("合計（総合×重み＋定性×重み）の平均", f"{qs['avg_combined']:.1f}", help="ランク算出の元となる合計点")
            if qs.get("rank_distribution"):
                st.markdown("**ランク（A〜E）の分布**")
                for r, cnt in sorted(qs["rank_distribution"].items(), key=lambda x: (-x[1], x[0])):
                    st.caption(f"- **{r}** … {cnt}件")
        else:
            st.caption("定性スコアリングを入力した成約案件がまだありません。審査入力で「定性スコアリング」を選択し、結果登録で成約にするとここに集計が表示されます。")


    all_cases = load_all_cases()

    # ---------------- 営業部ごとの結果集計 ----------------
    st.divider()
    st.subheader("🏢 営業部ごとの結果")
    if all_cases:
        dept_rows = []
        for case in all_cases:
            dept = case.get("sales_dept") or case.get("inputs", {}).get("sales_dept") or "未設定"
            status = case.get("final_status") or "未登録"
            dept_rows.append({"営業部": dept, "結果": status})

        df_dept = pd.DataFrame(dept_rows)
        if not df_dept.empty:
            dept_summary = (
                df_dept
                .pivot_table(index="営業部", columns="結果", aggfunc="size", fill_value=0)
                .reset_index()
            )
            for col in ["成約", "失注", "保留", "未登録"]:
                if col not in dept_summary.columns:
                    dept_summary[col] = 0
            dept_summary["合計"] = dept_summary[["成約", "失注", "保留", "未登録"]].sum(axis=1)
            dept_summary["成約率(%)"] = dept_summary.apply(
                lambda r: (r["成約"] / r["合計"] * 100) if r["合計"] else 0.0,
                axis=1,
            )
            dept_summary = dept_summary[["営業部", "成約", "失注", "保留", "未登録", "合計", "成約率(%)"]]
            dept_summary = dept_summary.sort_values(by=["成約率(%)", "成約"], ascending=[False, False])
            st.dataframe(
                dept_summary.style.format({"成約率(%)": "{:.1f}"}),
                width='stretch',
                hide_index=True,
            )

            st.markdown("#### 📊 営業部ごとの差の検証（業種・成約分布）")
            dept_options = sorted(df_dept["営業部"].dropna().unique().tolist())
            selected_dept = st.selectbox("比較する営業部", dept_options, key="dash_selected_dept")

            df_selected = [
                {
                    "業種": c.get("industry_sub") or "不明",
                    "結果": c.get("final_status") or "未登録",
                }
                for c in all_cases
                if (c.get("sales_dept") or c.get("inputs", {}).get("sales_dept") or "未設定") == selected_dept
            ]
            df_selected = pd.DataFrame(df_selected)

            if not df_selected.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"{selected_dept}の業種分布")
                    industry_dist = (
                        df_selected.groupby("業種").size().reset_index(name="件数").sort_values("件数", ascending=False)
                    )
                    st.bar_chart(industry_dist.set_index("業種")["件数"], height=280)
                with c2:
                    st.caption(f"{selected_dept}の結果分布")
                    result_order = ["成約", "失注", "保留", "未登録"]
                    result_dist = df_selected.groupby("結果").size().reindex(result_order, fill_value=0)
                    st.bar_chart(result_dist, height=280)
            else:
                st.caption("選択した営業部の案件データがありません。")
        else:
            st.caption("営業部集計に使えるデータがありません。")
    else:
        st.caption("まだ案件履歴がありません。")

    # ---------------- 案件履歴一覧 ----------------
    st.divider()
    st.subheader("📋 最新の案件履歴")
    if all_cases:
        for case in reversed(all_cases[-15:]):  # 最新15件を表示
            c_date = case.get('timestamp', '')[:16]
            c_sub = case.get('industry_sub', '不明')
            c_score = case.get('result', {}).get('score', 0)
            c_status = case.get('final_status', '未登録')
            title_emoji = "✅" if c_status == "成約" else "❌" if c_status == "失注" else "📝"
            with st.expander(f"{title_emoji} {c_date} - {c_sub} (スコア: {c_score:.0f}) [{c_status}]"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.write(f"**判定**: {case.get('result', {}).get('hantei', '不明')}")
                    if case.get('chat_summary'):
                        st.caption(case['chat_summary'])
                    if st.button("🔄 このデータを入力に復元", key=f"restore_hist_{case.get('id', c_date)}"):
                        i_data = case.get('inputs', {})
                        st.session_state["last_submitted_inputs"] = {
                            "selected_major": case.get("industry_major", ""),
                            "selected_sub": case.get("industry_sub", ""),
                        }
                        for k, v in i_data.items():
                            if isinstance(v, dict):
                                for sub_k, sub_v in v.items():
                                    st.session_state[sub_k] = sub_v
                            else:
                                st.session_state[k] = v
                        st.session_state["main_mode"] = "📋 審査・分析"
                        st.session_state["nav_index"] = 0
                        st.rerun()
                with c2:
                    if case.get('ai_industry_advice'):
                        st.markdown("##### 📈 AI業界分析アドバイス")
                        st.info(case['ai_industry_advice'])
                    if case.get('ai_byoki'):
                        st.markdown("##### 🤖 AIのぼやき")
                        st.caption(case['ai_byoki'])
    else:
        st.caption("まだ案件履歴がありません。")
