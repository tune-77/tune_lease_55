"""
金利サジェスト - 需要曲線ベースの最適スプレッド提案

【アルゴリズム】
データが十分な場合（成約・失注が混在、5件以上）:
  1. ロジスティック回帰で P(成約 | スプレッド, スコア) を推定（需要曲線）
  2. 期待利益 = P(成約|スプレッド) × スプレッド を計算
  3. 期待利益が最大になるスプレッドを推奨
データが少ない場合:
  - 成約実績の中央値スプレッド＋スコア補正（経験則フォールバック）
"""
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


def _fit_demand_curve(df: pd.DataFrame):
    """
    ロジスティック回帰で需要曲線を推定する。
    Returns (model, scaler) or (None, None) if fitting fails.
    高スプレッド → 低成約確率 の方向性が正しいモデルのみ採用。
    """
    if df["won"].nunique() < 2 or len(df) < 5:
        return None, None

    try:
        X = df[["spread", "score"]].values
        y = df["won"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        model.fit(X_scaled, y)

        # スプレッドの係数が負（スプレッド高い → 成約確率低い）であることを確認
        spread_coef = model.coef_[0][0]
        if spread_coef >= 0:
            return None, None

        return model, scaler
    except Exception:
        return None, None


def suggest_rate(
    current_score: float,
    current_base_rate: float,
    competitor_rate: float,
) -> dict:
    """
    現在の案件に対して最適スプレッドを推定する。

    Returns dict with keys:
        optimal_spread, optimal_rate, win_prob, chart, data_count,
        pricing_optimization (bool), message
    """
    df = _load_rate_dataframe()
    df = df[df["score"] > 0].copy()
    n = len(df)

    if n < 3:
        return {
            "message": f"データ不足（{n}件）。最低3件の成約履歴が必要です。",
            "data_count": n,
        }

    spread_min = max(0.0, df["spread"].quantile(0.05))
    spread_max = df["spread"].quantile(0.95) + 0.1
    spreads = np.linspace(spread_min, spread_max, 100)

    won_df = df[df["won"] == 1]
    lost_df = df[df["won"] == 0]

    # ── 需要曲線アプローチ（データが十分な場合）──────────────────────────
    model, scaler = _fit_demand_curve(df)
    use_pricing_opt = model is not None
    win_prob_curve = None
    expected_profit_curve = None

    if use_pricing_opt:
        X_pred = np.column_stack([
            spreads,
            np.full(len(spreads), current_score)
        ])
        X_pred_scaled = scaler.transform(X_pred)
        win_prob_curve = model.predict_proba(X_pred_scaled)[:, 1]

        # 期待利益 = P(成約|スプレッド) × スプレッド
        expected_profit_curve = win_prob_curve * spreads
        opt_idx = int(np.argmax(expected_profit_curve))
        optimal_spread = float(spreads[opt_idx])
        win_prob_at_optimal = float(win_prob_curve[opt_idx])

        # 競合がいる場合: 競合より高いなら引き下げ
        if competitor_rate > 0 and current_base_rate > 0:
            comp_spread = competitor_rate - current_base_rate
            if optimal_spread > comp_spread:
                adjusted = float(np.clip(
                    (optimal_spread + comp_spread - 0.1) / 2,
                    spread_min, spread_max
                ))
                X_check = scaler.transform([[adjusted, current_score]])
                win_prob_at_optimal = float(model.predict_proba(X_check)[0][1])
                optimal_spread = adjusted

        best_win_prob = float(np.clip(win_prob_at_optimal, 0.1, 0.99))

    # ── 経験則フォールバック ──────────────────────────────────────────────
    else:
        base_spread = float(won_df["spread"].median()) if len(won_df) > 0 else 0.3
        score_adj = (current_score - 50) / 100 * 0.2
        comp_adj = 0.0
        if competitor_rate > 0 and current_base_rate > 0:
            comp_spread = competitor_rate - current_base_rate
            comp_adj = comp_spread - base_spread - 0.1
        optimal_spread = float(np.clip(
            base_spread + score_adj + comp_adj, spread_min, spread_max
        ))

        won_spreads = won_df["spread"].values if len(won_df) > 0 else np.array([])
        if len(won_spreads) > 0:
            dist = np.abs(won_spreads - optimal_spread)
            best_win_prob = float(np.clip(
                np.mean(dist < (spread_max - spread_min) * 0.4) + 0.5, 0.5, 0.95
            ))
        else:
            best_win_prob = 0.7

    optimal_rate = current_base_rate + optimal_spread

    # ── チャート ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # 成約・失注の散布図
    if len(won_df) > 0:
        fig.add_trace(go.Scatter(
            x=won_df["spread"], y=won_df["score"],
            mode="markers", name="成約",
            marker=dict(color="#3b82f6", size=8, symbol="circle"),
            yaxis="y1",
        ))
    if len(lost_df) > 0:
        fig.add_trace(go.Scatter(
            x=lost_df["spread"], y=lost_df["score"],
            mode="markers", name="失注",
            marker=dict(color="#ef4444", size=8, symbol="x"),
            yaxis="y1",
        ))

    # 需要曲線・期待利益曲線（pricing optimization が有効な場合）
    if use_pricing_opt and win_prob_curve is not None:
        fig.add_trace(go.Scatter(
            x=spreads,
            y=win_prob_curve * 100,
            mode="lines",
            name="成約確率 (%)",
            line=dict(color="#8b5cf6", width=2),
            yaxis="y1",
        ))

        # 期待利益を0-100にスケーリングして重ねる
        ep_max = float(expected_profit_curve.max())
        if ep_max > 0:
            ep_scaled = expected_profit_curve / ep_max * 100
            fig.add_trace(go.Scatter(
                x=spreads,
                y=ep_scaled,
                mode="lines",
                name="期待利益（正規化）",
                line=dict(color="#f59e0b", width=2, dash="dot"),
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
        if spread_min <= comp_spread <= spread_max * 1.2:
            fig.add_vline(
                x=comp_spread,
                line_dash="dot",
                line_color="#ef4444",
                line_width=2,
                annotation_text=f"競合 {competitor_rate:.2f}%",
                annotation_position="top left",
            )

    method_label = "需要曲線最適化" if use_pricing_opt else "実績分布（フォールバック）"
    fig.update_layout(
        title=f"スプレッド分析（{n}件） - {method_label}",
        xaxis_title="スプレッド (%)",
        yaxis=dict(title="スコア / 確率 (%)", side="left", range=[0, 110]),
        height=340,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return {
        "optimal_spread": optimal_spread,
        "optimal_rate": optimal_rate,
        "win_prob": best_win_prob,
        "chart": fig,
        "data_count": n,
        "pricing_optimization": use_pricing_opt,
        "message": None,
    }


def render_rate_suggestion(res: dict):
    """
    金利サジェストUIを表示する。
    analysis_results.py の render_analysis_results(res, ...) から呼ぶ。
    """
    with st.expander("💴 金利サジェスト（需要曲線から最適スプレッドを推定）", expanded=False):
        score = float(res.get("score") or 0)
        pricing = res.get("pricing") or {}
        from base_rate_master import get_current_base_rate
        base_rate = float(pricing.get("base_rate") or get_current_base_rate(fallback=2.1))

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
        is_pricing_opt = result.get("pricing_optimization", False)

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

        if is_pricing_opt:
            st.caption(
                f"📈 需要曲線最適化（{n}件）: "
                "P(成約|スプレッド) × スプレッド が最大になる金利を推奨しています。"
                "スプレッドを上げると成約確率が下がり、下げすぎると利益が減る最適解です。"
            )
        else:
            st.caption(
                f"📊 実績ベース推定（{n}件）: "
                "成約・失注が混在するデータが増えると需要曲線モードに自動切替します。"
            )

        if result.get("chart"):
            st.plotly_chart(
                result["chart"],
                use_container_width=True,
                key="rate_suggestion_chart",
            )
