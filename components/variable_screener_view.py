"""
components/variable_screener_view.py
=====================================
SC.3: 未使用変数スクリーニング Streamlit UI。
quantum_screener.py (SC.1) をフロントエンドから実行し、
相関係数の棒グラフと CSV ダウンロードを提供する。
"""
from __future__ import annotations

import streamlit as st


def render_variable_screener_view() -> None:
    st.title("🔬 未使用変数スクリーニング")
    st.caption(
        "現在の量子モデルが使用していない財務変数と失注の相関を分析します。"
        "高い相関係数を持つ変数は新規ペア候補として quantum_config.json に追記できます。"
    )

    st.info(
        "⚠️ データ量の注意: 現状 ~40件。相関係数・p値はあくまで参考値です。"
        "サンプルサイズが小さいため統計的有意性の解釈には注意してください。"
    )

    run_btn = st.button("🔍 未使用変数スクリーニング実行", type="primary")

    if run_btn or st.session_state.get("_screener_ran"):
        st.session_state["_screener_ran"] = True
        _run_screener()


def _run_screener() -> None:
    try:
        from quantum_screener import QuantumScreener
    except ImportError:
        st.error("quantum_screener モジュールが見つかりません。")
        return

    with st.spinner("スクリーニング実行中..."):
        try:
            screener = QuantumScreener()
            df = screener.compute_correlations()
            csv_text = screener.to_csv()
        except Exception as e:
            st.error(f"スクリーニングエラー: {e}")
            return

    if df.empty:
        st.warning("分析できるデータがありません。DB に過去案件を登録してください。")
        return

    n_sample = int(df["n"].max()) if not df.empty else 0
    st.success(f"分析完了（n={n_sample}件、参考値）")

    # ── 棒グラフ ───────────────────────────────────────────────────────────
    st.markdown("### 相関係数（失注との Pearson r）")
    try:
        import plotly.graph_objects as go

        colors = [
            "#dc2626" if abs(r) >= 0.3 else "#f97316" if abs(r) >= 0.15 else "#94a3b8"
            for r in df["correlation"]
        ]
        fig = go.Figure(go.Bar(
            x=df["correlation"],
            y=df["variable"],
            orientation="h",
            marker_color=colors,
            text=[f"r={r:+.3f}  p={p:.3f}" for r, p in zip(df["correlation"], df["p_value"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>r=%{x:.4f}<extra></extra>",
        ))
        fig.add_vline(x=0, line_width=1, line_color="#64748b")
        fig.add_vline(x=0.3, line_width=1, line_dash="dash", line_color="#dc2626",
                      annotation_text="r=0.3", annotation_position="top")
        fig.add_vline(x=-0.3, line_width=1, line_dash="dash", line_color="#dc2626")
        fig.update_layout(
            xaxis_title="相関係数 (失注=1, 成約=0)",
            yaxis=dict(autorange="reversed"),
            height=max(250, 50 + 40 * len(df)),
            margin=dict(l=10, r=100, t=30, b=40),
            plot_bgcolor="#f8fafc",
            paper_bgcolor="#ffffff",
            font=dict(size=11),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df.set_index("variable")["correlation"])

    # ── 詳細テーブル ────────────────────────────────────────────────────────
    with st.expander("📋 詳細テーブル", expanded=False):
        st.caption(f"n={n_sample}件（参考値）。p<0.05 の変数は統計的に有意な傾向あり。")
        st.dataframe(
            df.style.background_gradient(subset=["correlation"], cmap="RdYlGn", vmin=-0.5, vmax=0.5),
            use_container_width=True,
            hide_index=True,
        )

    # ── CSV ダウンロード ─────────────────────────────────────────────────────
    st.download_button(
        "📥 CSV ダウンロード",
        data=csv_text,
        file_name="variable_screening_result.csv",
        mime="text/csv",
    )

    # ── 候補ペア提案 ─────────────────────────────────────────────────────────
    significant = df[df["p_value"] < 0.1]
    if not significant.empty:
        st.markdown("### 💡 新規ペア候補")
        st.caption("p<0.1 の変数を既存の量子変数とペアリングする候補です。")
        candidates = screener.suggest_pairs()
        if candidates:
            import pandas as pd
            cand_df = pd.DataFrame(candidates)[["var_a", "var_b", "weight"]]
            cand_df.columns = ["変数A", "変数B（候補）", "重み"]
            st.dataframe(cand_df, use_container_width=True, hide_index=True)

            if st.button("⚙️ quantum_config.json に候補ペアを追記（SC.5）"):
                try:
                    screener.suggest_pairs()  # suggest_pairs は compute のみ
                    import subprocess, sys
                    subprocess.run(
                        [sys.executable, "quantum_screener.py", "--suggest"],
                        check=True,
                    )
                    st.success("candidate_pairs を quantum_config.json に追記しました。")
                except Exception as e:
                    st.error(f"追記エラー: {e}")
    else:
        st.info("p<0.1 の変数は見つかりませんでした（データ量不足の可能性があります）。")
