"""
金利サジェスト - ボルツマン温度付き3シナリオ提案

【アルゴリズム】
1. ロジスティック回帰で P(成約 | スプレッド, スコア) を推定（需要曲線）
2. 期待利益 = P(成約|スプレッド) × スプレッド を最大化 → 「推奨」スプレッドを算出
3. ボルツマン温度 T = max(0.05, 0.3 / log(n+1)) で探索幅を決定
   - データが少ない（T高）: シナリオ間の幅が広い → 多様な試行を促進
   - データが多い（T低）: シナリオが推奨値に収束 → 実績に基づいた安定提案
4. 守り・推奨・強気の3シナリオを提示。担当者が案件状況に応じて選択。

データ不足時は経験則フォールバック（成約中央値＋スコア補正）。
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


def _boltzmann_temperature(n: int) -> float:
    """
    ボルツマン温度: データが少ないほど高く（探索）、多いほど低く（活用）。
    T = max(0.05, 0.3 / log(n+1))
      n=3  → T≈0.21  幅広い探索
      n=10 → T≈0.13  中程度
      n=50 → T≈0.077 収束傾向
    """
    return max(0.05, 0.3 / np.log(n + 1))


def _win_prob_for_spread(spread: float, score: float, model, scaler, fallback: float) -> float:
    """指定スプレッドの成約確率を返す（モデルなしの場合はfallbackを返す）"""
    if model is None or scaler is None:
        return fallback
    try:
        X = scaler.transform([[spread, score]])
        return float(model.predict_proba(X)[0][1])
    except Exception:
        return fallback


def suggest_rate(
    current_score: float,
    current_base_rate: float,
    competitor_rate: float,
) -> dict:
    """
    現在の案件に対してボルツマン温度付きの3シナリオを推定する。

    Returns dict with keys:
        scenarios: [
            {"label": "守り", "spread": float, "rate": float, "win_prob": float, "expected_profit": float},
            {"label": "推奨", ...},
            {"label": "強気", ...},
        ]
        temperature, optimal_spread, optimal_rate, win_prob,
        chart, data_count, pricing_optimization (bool), message
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

    # ── ボルツマン温度 ────────────────────────────────────────────────────
    T = _boltzmann_temperature(n)
    # シナリオ間のオフセット幅 = T × スプレッドレンジ × 0.4
    offset = T * (spread_max - spread_min) * 0.4

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

    # ── 3シナリオ生成（ボルツマン温度で幅を決定）────────────────────────
    conservative_spread = float(np.clip(optimal_spread - offset, spread_min, spread_max))
    aggressive_spread   = float(np.clip(optimal_spread + offset, spread_min, spread_max))

    def _ep(sp: float, wp: float) -> float:
        return round(wp * sp, 4)

    wp_cons = _win_prob_for_spread(conservative_spread, current_score, model, scaler,
                                   min(best_win_prob + 0.10, 0.97))
    wp_opt  = best_win_prob
    wp_aggr = _win_prob_for_spread(aggressive_spread, current_score, model, scaler,
                                   max(best_win_prob - 0.12, 0.10))

    scenarios = [
        {
            "label": "守り",
            "emoji": "🛡️",
            "spread": conservative_spread,
            "rate": current_base_rate + conservative_spread,
            "win_prob": wp_cons,
            "expected_profit": _ep(conservative_spread, wp_cons),
            "description": "成約優先。確実に取りに行く場合。",
        },
        {
            "label": "推奨",
            "emoji": "⚖️",
            "spread": optimal_spread,
            "rate": optimal_rate,
            "win_prob": wp_opt,
            "expected_profit": _ep(optimal_spread, wp_opt),
            "description": "期待利益が最大の均衡点。",
        },
        {
            "label": "強気",
            "emoji": "⚔️",
            "spread": aggressive_spread,
            "rate": current_base_rate + aggressive_spread,
            "win_prob": wp_aggr,
            "expected_profit": _ep(aggressive_spread, wp_aggr),
            "description": "利幅優先。スコアが高く優位な場合。",
        },
    ]

    # ── チャート ─────────────────────────────────────────────────────────
    fig = go.Figure()

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

    if use_pricing_opt and win_prob_curve is not None:
        fig.add_trace(go.Scatter(
            x=spreads, y=win_prob_curve * 100,
            mode="lines", name="成約確率 (%)",
            line=dict(color="#8b5cf6", width=2),
            yaxis="y1",
        ))
        ep_max = float(expected_profit_curve.max())
        if ep_max > 0:
            ep_scaled = expected_profit_curve / ep_max * 100
            fig.add_trace(go.Scatter(
                x=spreads, y=ep_scaled,
                mode="lines", name="期待利益（正規化）",
                line=dict(color="#f59e0b", width=2, dash="dot"),
                yaxis="y1",
            ))

    # 3シナリオのライン
    scenario_colors = {"守り": "#22c55e", "推奨": "#f59e0b", "強気": "#ef4444"}
    for sc in scenarios:
        fig.add_vline(
            x=sc["spread"],
            line_dash="dash",
            line_color=scenario_colors[sc["label"]],
            line_width=1.5,
            annotation_text=f"{sc['emoji']}{sc['label']} {sc['rate']:.2f}%",
            annotation_position="top right" if sc["label"] == "強気" else "top left",
        )

    if competitor_rate > 0 and current_base_rate > 0:
        comp_spread = competitor_rate - current_base_rate
        if spread_min <= comp_spread <= spread_max * 1.2:
            fig.add_vline(
                x=comp_spread,
                line_dash="dot",
                line_color="#94a3b8",
                line_width=1.5,
                annotation_text=f"競合 {competitor_rate:.2f}%",
                annotation_position="top left",
            )

    method_label = "需要曲線最適化" if use_pricing_opt else "実績分布（フォールバック）"
    fig.update_layout(
        title=f"スプレッド分析（{n}件） - {method_label}　温度T={T:.2f}",
        xaxis_title="スプレッド (%)",
        yaxis=dict(title="スコア / 確率 (%)", side="left", range=[0, 110]),
        height=340,
        margin=dict(l=40, r=60, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    return {
        "scenarios": scenarios,
        "optimal_spread": optimal_spread,
        "optimal_rate": optimal_rate,
        "win_prob": best_win_prob,
        "chart": fig,
        "data_count": n,
        "temperature": T,
        "pricing_optimization": use_pricing_opt,
        "message": None,
    }


def render_rate_suggestion(res: dict):
    """
    金利サジェストUIを表示する。
    analysis_results.py の render_analysis_results(res, ...) から呼ぶ。
    """
    with st.expander("💴 金利サジェスト（3シナリオ提案）", expanded=False):
        score = float(res.get("score") or 0)
        pricing = res.get("pricing") or {}
        from base_rate_master import get_current_base_rate, get_base_rate_by_term
        # リース期間が取得できる場合は期間別基準金利を使用
        _lease_term = int(st.session_state.get("lease_term") or pricing.get("lease_term") or 60)
        _rate_by_term = get_base_rate_by_term(lease_term_months=_lease_term)
        base_rate = float(pricing.get("base_rate") or _rate_by_term or get_current_base_rate(fallback=2.1))

        raw_comp = st.session_state.get("competitor_rate", 0)
        competitor_rate = _normalize_rate(raw_comp)

        result = suggest_rate(score, base_rate, competitor_rate)

        if result.get("message"):
            st.info(f"ℹ️ {result['message']}")
            return

        n = result["data_count"]
        T = result.get("temperature", 0.1)
        is_pricing_opt = result.get("pricing_optimization", False)
        scenarios = result.get("scenarios", [])

        # ── 3シナリオ表示 ────────────────────────────────────────────────
        cols = st.columns(3)
        colors = {"守り": "🟢", "推奨": "🟡", "強気": "🔴"}
        for col, sc in zip(cols, scenarios):
            with col:
                st.markdown(f"**{sc['emoji']} {sc['label']}**")
                st.metric(
                    "金利",
                    f"{sc['rate']:.2f}%",
                    f"spread +{sc['spread']:.2f}%",
                )
                st.caption(
                    f"成約確率: {sc['win_prob']*100:.0f}%　"
                    f"期待利益: {sc['expected_profit']:.3f}"
                )
                st.caption(sc["description"])

        # 競合比較
        if competitor_rate > 0:
            opt_sc = scenarios[1]
            diff = opt_sc["rate"] - competitor_rate
            st.caption(f"推奨金利 vs 競合: **{diff:+.2f}%**")

        # メソッド・温度の説明
        if is_pricing_opt:
            st.caption(
                f"📈 需要曲線最適化 | ボルツマン温度 T={T:.2f}（データ{n}件）　"
                "データが増えると3シナリオが推奨値に収束します。"
            )
        else:
            st.caption(
                f"📊 実績ベース推定（{n}件）| T={T:.2f}　"
                "成約・失注が混在するデータが増えると需要曲線モードに自動切替します。"
            )

        if result.get("chart"):
            st.plotly_chart(
                result["chart"],
                use_container_width=True,
                key="rate_suggestion_chart",
            )
