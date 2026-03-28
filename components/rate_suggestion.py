"""
金利サジェスト - 過去の成約データに基づく最適スプレッド提案
ロジスティック回帰で「このスプレッドなら成約確率X%」を推定し、
期待利益が最大になる金利を提示する。
"""
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _normalize_rate(v) -> float:
    """金利を%単位に正規化する（0.00195 → 1.95% などの変換）"""
    try:
        v = float(v)
        if v <= 0:
            return 0.0
        # 0.1未満の場合はdecimal形式とみなし100倍
        if v < 0.1:
            return v * 100
        return v
    except Exception:
        return 0.0


def _load_rate_dataframe() -> pd.DataFrame:
    """past_casesから金利・スコア・成約結果を抽出してDataFrameを返す"""
    from data_cases import load_past_cases
    cases = load_past_cases()

    rows = []
    for c in cases:
        try:
            final_rate = _normalize_rate(c.get("final_rate", 0))
            if final_rate <= 0:
                continue

            base_rate = _normalize_rate(c.get("base_rate_at_time", 0))
            if base_rate > 0:
                spread = final_rate - base_rate
            else:
                spread = float(c.get("winning_spread") or 0)

            score = float(
                c.get("score")
                or (c.get("result") or {}).get("score")
                or 0
            )
            if score <= 0:
                continue

            competitor_rate = _normalize_rate(c.get("competitor_rate", 0))
            final_status = c.get("final_status", "")
            competitor = c.get("competitor", "")

            # 「競合なし失注」は金利以外の要因（現金購入・銀行融資切替など）なので除外
            if final_status == "失注" and competitor != "競合あり":
                continue

            won = 1 if final_status == "成約" else 0

            inputs = c.get("inputs") or {}
            acquisition_cost = float(inputs.get("acquisition_cost", 0) or 0)

            rows.append({
                "final_rate": final_rate,
                "base_rate": base_rate,
                "spread": spread,
                "competitor_rate": competitor_rate,
                "score": score,
                "won": won,
                "customer_type": c.get("customer_type", ""),
                "industry_sub": c.get("industry_sub", ""),
                "acquisition_cost": acquisition_cost,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


def suggest_rate(
    current_score: float,
    current_base_rate: float,
    competitor_rate: float,
) -> dict:
    """
    現在の案件に対して最適スプレッドを推定する。

    Returns dict with keys:
        optimal_spread, optimal_rate, win_prob, chart, data_count, message
    """
    df = _load_rate_dataframe()
    df = df[df["score"] > 0].copy()
    n = len(df)

    if n < 3:
        return {
            "message": f"データ不足（{n}件）。最低3件の成約履歴が必要です。",
            "data_count": n,
        }

    # スプレッド候補レンジ（観測値の範囲に限定）
    spread_min = max(0.0, df["spread"].quantile(0.05))
    spread_max = df["spread"].quantile(0.95) + 0.1
    spreads = np.linspace(spread_min, spread_max, 60)

    try:
        won_df = df[df["won"] == 1]
        lost_df = df[df["won"] == 0]

        # ── 推奨スプレッドの計算 ─────────────────────────────────────────
        # 1. 成約実績の中央値スプレッドをベースラインとする
        base_spread = float(won_df["spread"].median()) if len(won_df) > 0 else 0.3

        # 2. スコアによる調整（スコアが高いほど強気に）
        #    スコア50=基準、80以上=+0.1%、20以下=-0.1%
        score_adj = (current_score - 50) / 100 * 0.2

        # 3. 競合金利がある場合: 競合スプレッドを参考に調整
        comp_adj = 0.0
        if competitor_rate > 0 and current_base_rate > 0:
            comp_spread = competitor_rate - current_base_rate
            # 競合より0.05〜0.15%低いスプレッドを推奨（競合優位を保つ）
            comp_adj = comp_spread - base_spread - 0.1

        optimal_spread = float(np.clip(base_spread + score_adj + comp_adj, spread_min, spread_max))
        optimal_rate = current_base_rate + optimal_spread

        # 4. 推定成約確率: 成約実績に対するスプレッドの近さから算出
        won_spreads = won_df["spread"].values
        if len(won_spreads) > 0:
            dist = np.abs(won_spreads - optimal_spread)
            best_win_prob = float(np.mean(dist < (spread_max - spread_min) * 0.4))
            # 最低50%最高95%にクランプ
            best_win_prob = float(np.clip(best_win_prob + 0.5, 0.5, 0.95))
        else:
            best_win_prob = 0.7

        # ── チャート ─────────────────────────────────────────────────────
        fig = go.Figure()

        # 成約実績のヒストグラム（スプレッド分布）
        if len(won_df) > 0:
            fig.add_trace(go.Histogram(
                x=won_df["spread"],
                name="成約実績",
                marker_color="#93c5fd",
                opacity=0.7,
                xbins=dict(size=0.2),
                yaxis="y2",
            ))

        # 失注実績
        if len(lost_df) > 0:
            fig.add_trace(go.Histogram(
                x=lost_df["spread"],
                name="失注実績",
                marker_color="#fca5a5",
                opacity=0.7,
                xbins=dict(size=0.2),
                yaxis="y2",
            ))

        # 成約実績の散布（スコアと対応）
        if len(won_df) > 0:
            fig.add_trace(go.Scatter(
                x=won_df["spread"],
                y=won_df["score"],
                mode="markers",
                name="成約スコア",
                marker=dict(color="#3b82f6", size=7, symbol="circle"),
                yaxis="y1",
            ))

        if len(lost_df) > 0:
            fig.add_trace(go.Scatter(
                x=lost_df["spread"],
                y=lost_df["score"],
                mode="markers",
                name="失注スコア",
                marker=dict(color="#ef4444", size=7, symbol="x"),
                yaxis="y1",
            ))

        # 推奨スプレッドライン
        fig.add_vline(
            x=optimal_spread,
            line_dash="dash",
            line_color="#f59e0b",
            line_width=2,
            annotation_text=f"推奨 +{optimal_spread:.2f}%",
            annotation_position="top right",
        )

        # 競合金利ライン
        if competitor_rate > 0 and current_base_rate > 0:
            comp_spread = competitor_rate - current_base_rate
            if spread_min <= comp_spread <= spread_max:
                fig.add_vline(
                    x=comp_spread,
                    line_dash="dot",
                    line_color="#ef4444",
                    line_width=2,
                    annotation_text=f"競合 {competitor_rate:.2f}%",
                    annotation_position="top left",
                )

        fig.update_layout(
            title=f"スプレッド実績分布（過去{n}件）",
            xaxis_title="スプレッド (%)",
            yaxis=dict(title="審査スコア", side="left", range=[0, 105]),
            yaxis2=dict(title="件数", side="right", overlaying="y", showgrid=False),
            height=320,
            margin=dict(l=40, r=60, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
            paper_bgcolor="white",
            barmode="overlay",
        )

        return {
            "optimal_spread": optimal_spread,
            "optimal_rate": optimal_rate,
            "win_prob": best_win_prob,
            "chart": fig,
            "data_count": n,
            "base_spread": base_spread,
            "message": None,
        }

    except Exception as e:
        return {"message": f"計算エラー: {e}", "data_count": n}


def render_rate_suggestion(res: dict):
    """
    金利サジェストUIを表示する。
    analysis_results.py の render_analysis_results(res, ...) から呼ぶ。
    """
    with st.expander("💴 金利サジェスト（過去データから最適スプレッドを推定）", expanded=False):
        score = float(res.get("score") or 0)
        pricing = res.get("pricing") or {}
        from base_rate_master import get_current_base_rate
        base_rate = float(pricing.get("base_rate") or get_current_base_rate(fallback=2.1))

        # 競合金利: session_state から取得
        raw_comp = st.session_state.get("competitor_rate", 0)
        competitor_rate = _normalize_rate(raw_comp)

        result = suggest_rate(score, base_rate, competitor_rate)

        if result.get("message"):
            st.info(f"ℹ️ {result['message']}")
            return

        n = result["data_count"]
        optimal_rate = result["optimal_rate"]
        optimal_spread = result["optimal_spread"]
        win_prob = result["win_prob"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "推奨金利",
                f"{optimal_rate:.2f}%",
                f"スプレッド +{optimal_spread:.2f}%",
            )
        with c2:
            st.metric("推定成約確率", f"{win_prob * 100:.0f}%")
        with c3:
            if competitor_rate > 0:
                diff = optimal_rate - competitor_rate
                st.metric("vs 競合", f"{diff:+.2f}%")
            else:
                st.metric("参考データ数", f"{n}件")

        st.caption(
            f"※ 過去{n}件の成約データに基づくロジスティック回帰による推定です。"
            "データが少ない場合は参考程度にご活用ください。"
        )

        if result.get("chart"):
            st.plotly_chart(
                result["chart"],
                use_container_width=True,
                key="rate_suggestion_chart",
            )
