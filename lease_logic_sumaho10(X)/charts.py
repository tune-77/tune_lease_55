"""
グラフ・チャート描画モジュール（lease_logic_sumaho10）
plot_gauge_plotly, plot_waterfall_plotly 等の Plotly/Matplotlib グラフを提供。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from matplotlib.patches import Patch

sns.set_theme(style="whitegrid", font="sans-serif")

# グラフ共通スタイル（ビジネス向け：ネイビー・グレー・ゴールド/赤アクセント）
CHART_STYLE = {
    "primary": "#1e3a5f",
    "secondary": "#475569",
    "good": "#0d9488",
    "warning": "#b45309",
    "danger": "#b91c1c",
    "accent": "#b45309",
    "bg": "#f8fafc",
    "grid": "#e2e8f0",
    "text": "#334155",
    "text_light": "#64748b",
}
plt.rcParams.update({
    "figure.facecolor": CHART_STYLE["bg"],
    "axes.facecolor": "white",
    "axes.edgecolor": CHART_STYLE["grid"],
    "axes.linewidth": 1.0,
    "grid.alpha": 0.4,
    "grid.color": CHART_STYLE["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})

# 指標の差で「低い方が良い」とする項目名（plot_indicators_* と analyze_indicators_vs_bench で使用）
LOWER_IS_BETTER_NAMES = {"借入金等依存度", "減価償却費/売上高", "固定比率", "負債比率"}


def _equity_ratio_display(val):
    """自己資本比率の表示用。100超の値（例: 2025）は20.25%として解釈する。"""
    if val is None:
        return None
    try:
        v = float(val)
        return (v / 100.0) if v > 100 else v
    except (TypeError, ValueError):
        return val


def plot_gauge(score, title="承認スコア"):
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    if score >= 71:
        color = CHART_STYLE["good"]
    elif score >= 41:
        color = CHART_STYLE["warning"]
    else:
        color = CHART_STYLE["danger"]
    data = [score, 100 - score]
    wedges, _ = ax.pie(data, startangle=90, counterclock=False,
                       colors=[color, "#f1f5f9"],
                       wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"{score:.1f}%", ha="center", va="center", fontsize=22, fontweight="bold", color="#334155")
    ax.set_title(title, fontsize=12, pad=12, color="#334155")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_gauge_plotly(score, title="承認スコア"):
    """Plotly版：ホバー・ズーム可能なゲージ"""
    if score >= 71:
        color = CHART_STYLE["good"]
    elif score >= 41:
        color = CHART_STYLE["warning"]
    else:
        color = CHART_STYLE["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": CHART_STYLE["grid"],
            "steps": [
                {"range": [0, 41], "color": "#f1f5f9"},
                {"range": [41, 71], "color": "#fef3c7"},
                {"range": [71, 100], "color": "#ccfbf1"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor=CHART_STYLE["bg"],
        font={"color": "#334155"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
    )
    return fig


def plot_waterfall(nenshu, gross, op_profit, ord_profit, net_income):
    cost_goods = nenshu - gross
    sga = gross - op_profit
    non_op = ord_profit - op_profit
    tax_extra = net_income - ord_profit
    categories = ["売上高", "売上原価", "販管費", "営業外", "税引前", "当期利益"]
    values = [nenshu, -cost_goods, -sga, non_op, tax_extra, net_income]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    running_total = 0
    c_base = CHART_STYLE["primary"]
    c_pos = CHART_STYLE["good"]
    c_neg = CHART_STYLE["danger"]
    for i, val in enumerate(values):
        if i == 0:
            bottom = 0
            color = c_base
            running_total += val
        elif i == len(values) - 1:
            bottom = 0
            val = running_total
            color = c_pos if val >= 0 else c_neg
        else:
            if val < 0:
                bottom = running_total + val
                running_total += val
                color = c_neg
            else:
                bottom = running_total
                running_total += val
                color = c_pos
        bars = ax.bar(categories[i], abs(val), bottom=bottom, color=color, edgecolor="white", linewidth=1.2, alpha=0.92, width=0.6)
        label_y = bottom + abs(val) + (nenshu * 0.02)
        ax.text(i, label_y, f"{int(val/1000)}k", ha="center", fontsize=9, color="#475569", fontweight="500")
    ax.set_title("利益構造 (単位:千円)", fontsize=12, pad=15, color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color=CHART_STYLE["grid"])
    sns.despine(left=True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_waterfall_plotly(nenshu, gross, op_profit, ord_profit, net_income):
    """Plotly版：ホバー・ズーム可能なウォーターフォール（単位:千円）"""
    cost_goods = nenshu - gross
    sga = gross - op_profit
    non_op = ord_profit - op_profit
    tax_extra = net_income - ord_profit
    categories = ["売上高", "売上原価", "販管費", "営業外", "税引前", "当期利益"]
    values = [nenshu, -cost_goods, -sga, non_op, tax_extra, net_income]
    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    text_vals = [f"{int(v/1000)}k" for v in values]
    colors = [CHART_STYLE["primary"], CHART_STYLE["danger"], CHART_STYLE["danger"],
              CHART_STYLE["good"], CHART_STYLE["good"],
              CHART_STYLE["good"] if net_income >= 0 else CHART_STYLE["danger"]]
    fig = go.Figure(go.Waterfall(
        name="利益構造",
        orientation="v",
        measure=measures,
        x=categories,
        y=values,
        text=text_vals,
        textposition="outside",
        connector={"line": {"color": CHART_STYLE["grid"], "width": 1}},
        increasing={"marker": {"color": CHART_STYLE["good"]}},
        decreasing={"marker": {"color": CHART_STYLE["danger"]}},
        totals={"marker": {"color": CHART_STYLE["good"] if net_income >= 0 else CHART_STYLE["danger"]}},
    ))
    fig.update_layout(
        title="利益構造 (単位:千円)",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=50, b=80, l=50, r=30),
        height=380,
        xaxis_tickangle=-45,
        showlegend=False,
    )
    fig.update_yaxes(gridcolor=CHART_STYLE["grid"], zeroline=True)
    return fig


def plot_benchmark_comparison(user_val, bench_val, metric_name):
    df = pd.DataFrame({
        "対象": ["貴社", "業界平均"],
        "値": [user_val, bench_val]
    })
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    palette = [CHART_STYLE["primary"], CHART_STYLE["secondary"]]
    sns.barplot(data=df, x="対象", y="値", palette=palette, ax=ax, hue="対象", legend=False, width=0.5)
    for i, v in enumerate([user_val, bench_val]):
        ax.text(i, v, f" {v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10, color="#334155")
    ax.set_ylabel(f"{metric_name} (%)", fontsize=10, color="#475569")
    ax.set_xlabel("")
    ax.set_title(f"{metric_name} 比較", fontsize=11, pad=10, color="#334155")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_indicators_gap_analysis(indicators):
    """指標と業界目安の差を横棒図で返す。差 = 貴社 - 業界。緑 = 良い方向、赤 = 要確認。"""
    with_bench = []
    for ind in indicators:
        bench = ind.get("bench")
        if bench is None or (isinstance(bench, float) and (bench != bench)):
            continue
        diff = ind["value"] - bench
        name = ind["name"]
        unit = ind.get("unit", "%")
        is_good = (diff > 0 and name not in LOWER_IS_BETTER_NAMES) or (diff < 0 and name in LOWER_IS_BETTER_NAMES)
        with_bench.append({"name": name, "diff": diff, "unit": unit, "is_good": is_good})
    if not with_bench:
        return None
    names = [x["name"] for x in with_bench]
    diffs = [x["diff"] for x in with_bench]
    colors = [CHART_STYLE["good"] if x["is_good"] else CHART_STYLE["danger"] for x in with_bench]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, len(names) * 0.48)))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    bars = ax.barh(y_pos, diffs, color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.axvline(0, color=CHART_STYLE["secondary"], linewidth=1, linestyle="-", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9, color="#334155")
    ax.set_xlabel("差（貴社 − 業界目安）　← 要確認 | 良い →", fontsize=9, color="#475569")
    ax.set_title("指標と業界目安の差の解釈", fontsize=11, pad=12, color="#334155")
    ax.legend(handles=[
        Patch(facecolor=CHART_STYLE["good"], alpha=0.88, label="業界より良い"),
        Patch(facecolor=CHART_STYLE["danger"], alpha=0.88, label="業界より要確認"),
    ], loc="lower right", fontsize=8, frameon=True, fancybox=True, shadow=True)
    x_range = max(diffs) - min(diffs) or 1
    margin = x_range * 0.03 + 0.01
    for i, (d, w) in enumerate(zip(diffs, with_bench)):
        u = w["unit"]
        s = f"{d:+.1f}{u}"
        ha = "left" if d >= 0 else "right"
        ax.text(d + margin if d >= 0 else d - margin, i, s, va="center", ha=ha, fontsize=8)
    sns.despine(left=True)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_indicators_gap_analysis_plotly(indicators):
    """Plotly版：指標と業界目安の差（横棒・ホバーで数値表示）"""
    with_bench = []
    for ind in indicators:
        bench = ind.get("bench")
        if bench is None or (isinstance(bench, float) and (bench != bench)):
            continue
        diff = ind["value"] - bench
        name = ind["name"]
        unit = ind.get("unit", "%")
        is_good = (diff > 0 and name not in LOWER_IS_BETTER_NAMES) or (diff < 0 and name in LOWER_IS_BETTER_NAMES)
        with_bench.append({"name": name, "diff": diff, "unit": unit, "is_good": is_good})
    if not with_bench:
        return None
    names = [x["name"] for x in with_bench]
    diffs = [x["diff"] for x in with_bench]
    colors = [CHART_STYLE["good"] if x["is_good"] else CHART_STYLE["danger"] for x in with_bench]
    hover_text = [f"{n}<br>差: {d:+.1f}{w['unit']}<br>{'業界より良い' if w['is_good'] else '要確認'}" for n, d, w in zip(names, diffs, with_bench)]
    fig = go.Figure(go.Bar(
        y=names,
        x=diffs,
        orientation="h",
        marker_color=colors,
        text=[f"{d:+.1f}{w['unit']}" for d, w in zip(diffs, with_bench)],
        textposition="outside",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
    ))
    fig.add_vline(x=0, line_dash="solid", line_color=CHART_STYLE["secondary"], line_width=1)
    fig.update_layout(
        title="指標と業界目安の差の解釈",
        xaxis_title="差（貴社 − 業界目安）　← 要確認 | 良い →",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=45, b=45, l=20, r=80),
        height=max(280, len(names) * 36),
        showlegend=False,
        yaxis=dict(autorange="reversed"),
    )
    fig.update_yaxes(gridcolor="white", zeroline=False)
    fig.update_xaxes(gridcolor=CHART_STYLE["grid"], zeroline=True)
    return fig


def plot_indicators_bar(indicators):
    """算出指標を横棒グラフで表示（貴社・業界平均）"""
    if not indicators:
        return None
    names = [x["name"] for x in indicators]
    values = [x["value"] for x in indicators]
    bench_vals = [x["bench"] if x["bench"] is not None else float("nan") for x in indicators]
    units = list({x["unit"] for x in indicators})
    y_label = units[0] if len(units) == 1 else "値"
    x_pos = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, max(3.2, len(names) * 0.42)))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    bars1 = ax.barh(x_pos - width / 2, values, width, label="貴社", color=CHART_STYLE["primary"], alpha=0.9, edgecolor="white", linewidth=0.6)
    has_bench = any(b == b for b in bench_vals)
    if has_bench:
        bars2 = ax.barh(x_pos + width / 2, [b if b == b else 0 for b in bench_vals], width, label="業界目安", color=CHART_STYLE["secondary"], alpha=0.75, edgecolor="white", linewidth=0.6)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=9, color="#334155")
    ax.set_xlabel(y_label, fontsize=10, color="#475569")
    ax.set_title("算出可能指標（貴社 vs 業界目安）", fontsize=11, pad=12, color="#334155")
    if has_bench:
        ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=True, shadow=True)
    for i, v in enumerate(values):
        if not (v != v):
            ax.text(v, i - width / 2, f" {v:.1f}", va="center", fontsize=8, color="#334155", fontweight="500")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    sns.despine(left=True)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_radar_chart(metrics, benchmarks):
    """財務レーダーチャート。metrics/benchmarks: {"収益性": 50, ...}"""
    labels = list(metrics.keys())
    values = list(metrics.values()) + list(metrics.values())[:1]
    bench_values = list(benchmarks.values()) + list(benchmarks.values())[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.2, 4.2), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    ax.plot(angles, values, color=CHART_STYLE["primary"], linewidth=2.2, label="貴社")
    ax.fill(angles, values, color=CHART_STYLE["primary"], alpha=0.22)
    ax.plot(angles, bench_values, color=CHART_STYLE["secondary"], linewidth=2, linestyle="--", label="業界平均")
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.1), frameon=True, fancybox=True, shadow=True)
    ax.set_title("財務バランス分析 (偏差値)", y=1.08, fontsize=12, color="#334155")
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_radar_chart_plotly(metrics, benchmarks):
    """Plotly版：レーダーチャート"""
    labels = list(metrics.keys())
    values = list(metrics.values()) + list(metrics.values())[:1]
    bench_values = list(benchmarks.values()) + list(benchmarks.values())[:1]
    angles = np.linspace(0, 360, len(labels), endpoint=False).tolist() + [0]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels + [labels[0]],
        fill="toself",
        name="貴社",
        line=dict(color=CHART_STYLE["primary"], width=2),
        fillcolor="rgba(30, 58, 95, 0.25)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=bench_values,
        theta=labels + [labels[0]],
        fill="toself",
        name="業界平均",
        line=dict(color=CHART_STYLE["secondary"], width=2, dash="dash"),
        fillcolor="rgba(71, 85, 105, 0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=CHART_STYLE["grid"]),
            angularaxis=dict(gridcolor=CHART_STYLE["grid"], linecolor=CHART_STYLE["grid"]),
            bgcolor="white",
        ),
        title="財務バランス分析 (偏差値)",
        paper_bgcolor=CHART_STYLE["bg"],
        font=dict(color="#334155", size=11),
        margin=dict(t=50, b=30),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def plot_positioning_scatter(current_sales, current_op_margin, past_cases):
    """ポジショニング散布図 (過去案件との比較)"""
    data = []
    for c in past_cases:
        if "financials" in c.get("result", {}):
            fin = c["result"]["financials"]
            s = fin.get("nenshu", 0) / 1000
            p = (fin.get("rieki", 0) / fin.get("nenshu", 1)) * 100 if fin.get("nenshu", 0) > 0 else 0
            res = "承認" if c["result"]["score"] >= 70 else "否決"
            data.append({"売上高(百万円)": s, "営業利益率(%)": p, "Type": res})
    data.append({"売上高(百万円)": current_sales/1000, "営業利益率(%)": current_op_margin, "Type": "★今回"})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    palette = {"承認": CHART_STYLE["primary"], "否決": CHART_STYLE["danger"], "★今回": CHART_STYLE["warning"]}
    sns.scatterplot(data=df, x="売上高(百万円)", y="営業利益率(%)", hue="Type", style="Type",
                    s=120, palette=palette, ax=ax, edgecolor="white", linewidth=1.2)
    current = df[df["Type"] == "★今回"]
    if not current.empty:
        ax.text(current.iloc[0]["売上高(百万円)"], current.iloc[0]["営業利益率(%)"] + 0.5, "YOU",
                ha="center", fontweight="bold", color="#334155", fontsize=10)
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.45, color=CHART_STYLE["grid"])
    ax.set_title("ポジショニング分析 (vs過去案件)", fontsize=12, pad=12, color="#334155")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_3d_analysis(current_data, past_cases):
    """3Dポジショニング分析。current_data: {'sales': 百万円, 'op_margin': %, 'equity_ratio': %}"""
    plot_data = []
    for c in past_cases:
        res = c.get("result", {})
        f = res.get("financials", {})
        if f:
            sales = f.get("nenshu", 0) / 1000
            op_margin = (f.get("rieki", 0) / f.get("nenshu", 1)) * 100 if f.get("nenshu", 0) > 0 else 0
            equity_ratio = _equity_ratio_display(res.get("user_eq", 0)) or 0
            status = "承認済" if res.get("score", 0) >= 70 else "否決"
            plot_data.append({
                "売上(M)": sales, "利益率(%)": op_margin,
                "自己資本比率(%)": equity_ratio, "判定": status, "size": 8
            })
    plot_data.append({
        "売上(M)": current_data["sales"] / 1000,
        "利益率(%)": current_data["op_margin"],
        "自己資本比率(%)": current_data["equity_ratio"],
        "判定": "★今回の案件",
        "size": 15
    })
    df = pd.DataFrame(plot_data)
    if df.empty:
        return None
    fig = px.scatter_3d(
        df, x="売上(M)", y="利益率(%)", z="自己資本比率(%)",
        color="判定", size="size", opacity=0.85,
        color_discrete_map={
            "承認済": CHART_STYLE["primary"],
            "否決": CHART_STYLE["warning"],
            "★今回の案件": CHART_STYLE["danger"]
        },
        hover_data={"size": False}
    )
    fig.update_layout(
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        scene=dict(
            xaxis_title="売上(百万円)",
            yaxis_title="利益率(%)",
            zaxis_title="自己資本比率(%)",
            bgcolor="white",
        ),
        margin=dict(l=0, r=0, b=0, t=28),
        font=dict(color="#334155", size=11),
        legend=dict(bgcolor="white", bordercolor=CHART_STYLE["grid"], borderwidth=1),
    )
    return fig


def plot_score_models_comparison_plotly(res):
    """3モデル（全体・業種・指標ベンチ）のスコア比較＋承認ライン70"""
    models = ["① 全体モデル", "② 業種モデル", "③ 指標ベンチ"]
    scores = [
        res.get("score", 0),
        res.get("ind_score", 0),
        res.get("bench_score", 0),
    ]
    colors = [CHART_STYLE["primary"], CHART_STYLE["secondary"], CHART_STYLE["accent"]]
    fig = go.Figure(go.Bar(
        x=models,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        hovertemplate="%{x}<br>スコア: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color=CHART_STYLE["danger"], line_width=1.5,
                  annotation_text="承認ライン 70%", annotation_position="right")
    fig.update_layout(
        title="スコア内訳（3モデル比較）",
        yaxis_title="スコア (%)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=45, b=50, l=50, r=30),
        height=260,
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="white")
    fig.update_yaxes(gridcolor=CHART_STYLE["grid"], zeroline=True)
    return fig


def plot_contract_prob_factors_plotly(ai_completed_factors):
    """契約期待度への寄与（要因別）を横棒で表示"""
    if not ai_completed_factors:
        return None
    factors = [f.get("factor", "") for f in ai_completed_factors]
    effects = [f.get("effect_percent", 0) for f in ai_completed_factors]
    colors = [CHART_STYLE["good"] if e >= 0 else CHART_STYLE["danger"] for e in effects]
    hover_text = [f"{f.get('factor','')}: {f.get('effect_percent',0):+.0f}%<br>{f.get('detail','')}" for f in ai_completed_factors]
    fig = go.Figure(go.Bar(
        y=factors,
        x=effects,
        orientation="h",
        marker_color=colors,
        text=[f"{e:+.0f}%" for e in effects],
        textposition="outside",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_text,
    ))
    fig.add_vline(x=0, line_dash="solid", line_color=CHART_STYLE["grid"], line_width=1)
    fig.update_layout(
        title="契約期待度への寄与（要因別）",
        xaxis_title="効果 (%)",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=10),
        margin=dict(t=40, b=40, l=20, r=60),
        height=max(220, len(factors) * 28),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor="white")
    fig.update_xaxes(gridcolor=CHART_STYLE["grid"], zeroline=True)
    return fig


def plot_scoring_top5_factors_plotly(scoring_result):
    """判定に効いている指標 Top5 の重要度を横棒で表示"""
    if not scoring_result:
        return None
    top5 = scoring_result.get("top5_reasons") or []
    if not top5:
        return None
    _feat_ja = {
        "ROA": "総資産利益率（ROA）", "ROE": "自己資本利益率（ROE）",
        "operating_margin": "売上高営業利益率", "net_margin": "売上高純利益率",
        "equity_ratio": "自己資本比率", "debt_ratio": "負債比率", "debt_equity_ratio": "負債対自己資本比率",
        "machinery_ratio": "機械設備比率", "fixed_asset_ratio": "固定資産比率",
        "fixed_to_equity": "固定資産対純資産比率", "machinery_equity_coverage": "機械設備の自己資本カバー率",
        "rent_to_revenue": "リース料負担率（対売上高）", "operating_profit_to_rent": "営業利益のリース料カバー率",
        "rent_to_equity": "リース料の純資産負担率", "lease_dependency": "リース依存度",
        "total_fixed_cost_ratio": "総固定費負担率", "depreciation_to_revenue": "減価償却費率（対売上高）",
        "EBITDA_margin": "EBITDAマージン", "depreciation_rate": "設備償却進行度",
        "asset_turnover": "総資産回転率", "fixed_asset_turnover": "固定資産回転率",
        "log_revenue": "売上高（対数）", "log_assets": "総資産（対数）",
        "is_loss": "赤字フラグ", "is_operating_loss": "営業赤字フラグ",
        "low_equity_ratio": "自己資本比率20%未満", "low_ROA": "ROA2%未満",
        "high_rent_burden": "リース負担大", "rent_exceeds_profit": "リース料＞営業利益",
        "industry_encoded": "業種（コード）",
    }
    factors = []
    values = []
    hover_texts = []
    for r in top5:
        if ":" in r:
            _name, _val = r.split(":", 1)
            _label = _feat_ja.get(_name.strip(), _name.strip())
            factors.append(_label)
            values.append(_val.strip())
            hover_texts.append(f"{_label}<br>値: {_val.strip()}")
        else:
            factors.append(r)
            values.append("")
            hover_texts.append(r)
    fig = go.Figure(go.Bar(
        y=factors,
        x=list(range(len(factors), 0, -1)),
        orientation="h",
        marker_color=CHART_STYLE["primary"],
        text=[f"#{i}" for i in range(len(factors), 0, -1)],
        textposition="outside",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    fig.update_layout(
        title="判定に効いている指標 Top5",
        xaxis_title="重要度順位",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=10),
        margin=dict(t=40, b=40, l=20, r=60),
        height=max(220, len(factors) * 35),
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="white", range=[0, 6])
    return fig


def plot_past_scores_histogram_plotly(current_score, past_cases):
    """過去案件のスコア分布＋今回のスコアの位置"""
    scores = []
    for c in (past_cases or []):
        s = c.get("result", {}).get("score")
        if s is not None:
            scores.append(float(s))
    if not scores and current_score is None:
        return None
    fig = go.Figure()
    if scores:
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=min(20, max(5, len(set(scores)))),
            name="過去案件",
            marker_color=CHART_STYLE["secondary"],
            opacity=0.7,
            hovertemplate="スコア: %{x:.1f}%<br>件数: %{y}<extra></extra>",
        ))
    if current_score is not None:
        fig.add_vline(
            x=current_score,
            line_dash="solid",
            line_color=CHART_STYLE["primary"],
            line_width=2.5,
            annotation_text="今回",
            annotation_position="top",
        )
    fig.add_vline(
        x=70,
        line_dash="dash",
        line_color=CHART_STYLE["danger"],
        line_width=1.5,
        annotation_text="承認70",
        annotation_position="bottom",
    )
    fig.update_layout(
        title="過去案件スコア分布 vs 今回",
        xaxis_title="スコア (%)",
        yaxis_title="件数",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=50, b=50, l=50, r=30),
        height=280,
        barmode="overlay",
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 105], gridcolor=CHART_STYLE["grid"])
    fig.update_yaxes(gridcolor=CHART_STYLE["grid"])
    return fig


def plot_balance_sheet_plotly(financials):
    """資産・負債の内訳（積み上げ感覚で比較）"""
    if not financials:
        return None
    total_assets = financials.get("assets") or 0
    net_assets = financials.get("net_assets") or 0
    machines = financials.get("machines") or 0
    other_assets = financials.get("other_assets") or 0
    bank_credit = financials.get("bank_credit") or 0
    lease_credit = financials.get("lease_credit") or 0
    liability = total_assets - net_assets if total_assets else 0
    current_approx = max(0, total_assets - machines - other_assets) if total_assets else 0
    if total_assets <= 0 and liability <= 0:
        return None
    labels = ["流動的資産(近似)", "機械・車両等", "その他固定", "銀行与信", "リース債務", "純資産"]
    values = [current_approx, machines, other_assets, bank_credit, lease_credit, net_assets]
    values_m = [v / 1000 for v in values]
    colors = [
        CHART_STYLE["primary"],
        CHART_STYLE["secondary"],
        CHART_STYLE["accent"],
        CHART_STYLE["danger"],
        CHART_STYLE["warning"],
        CHART_STYLE["good"],
    ]
    fig = go.Figure(go.Bar(
        x=labels,
        y=values_m,
        marker_color=colors,
        text=[f"{v:.1f}M" for v in values_m],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.1f}百万円<extra></extra>",
    ))
    fig.update_layout(
        title="資産・負債内訳 (百万円)",
        yaxis_title="金額 (百万円)",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=45, b=80, l=50, r=30),
        height=320,
        xaxis_tickangle=-35,
        showlegend=False,
    )
    fig.update_yaxes(gridcolor=CHART_STYLE["grid"], zeroline=True)
    return fig


def plot_break_even_point(sales, variable_cost, fixed_cost):
    """損益分岐点グラフ"""
    if sales <= 0:
        return None
    vc_ratio = variable_cost / sales
    bep = fixed_cost / (1 - vc_ratio) if (1 - vc_ratio) > 0 else sales * 2
    max_x = max(sales, bep) * 1.2
    x = np.linspace(0, max_x, 100)
    y_revenue = x
    y_cost = fixed_cost + (x * vc_ratio)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fig.patch.set_facecolor(CHART_STYLE["bg"])
    ax.set_facecolor("white")
    ax.plot(x, y_revenue, label="売上高", color=CHART_STYLE["primary"], linewidth=2.2)
    ax.plot(x, y_cost, label="総費用", color=CHART_STYLE["danger"], linestyle="--", linewidth=2)
    ax.scatter([sales], [sales], color=CHART_STYLE["good"], s=120, zorder=5, label="現在", edgecolor="white", linewidth=1.5)
    ax.vlines(sales, 0, sales, color=CHART_STYLE["good"], linestyle=":", alpha=0.8)
    if bep < max_x:
        ax.scatter([bep], [bep], color=CHART_STYLE["warning"], s=120, zorder=5, label="損益分岐点", edgecolor="white", linewidth=1.5)
        ax.vlines(bep, 0, bep, color=CHART_STYLE["warning"], linestyle=":", alpha=0.8)
        ax.text(bep, 0, f"BEP\n{int(bep/1000)}M", ha="center", va="bottom", fontsize=9, color="#475569", fontweight="500")
    ax.set_xlabel("売上規模", fontsize=10, color="#475569")
    ax.set_ylabel("金額", fontsize=10, color="#475569")
    ax.set_title("損益分岐点分析", fontsize=11, pad=10, color="#334155")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=8)
    ax.grid(True, alpha=0.4, color=CHART_STYLE["grid"], linestyle="--")
    sns.despine()
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_break_even_point_plotly(sales, variable_cost, fixed_cost):
    """Plotly版：損益分岐点（ホバー・ズーム・パン可能）"""
    if sales <= 0:
        return None
    vc_ratio = variable_cost / sales
    bep = fixed_cost / (1 - vc_ratio) if (1 - vc_ratio) > 0 else sales * 2
    max_x = max(sales, bep) * 1.2
    x = np.linspace(0, max_x, 100)
    y_revenue = x
    y_cost = fixed_cost + (x * vc_ratio)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_revenue, mode="lines", name="売上高", line=dict(color=CHART_STYLE["primary"], width=2)))
    fig.add_trace(go.Scatter(x=x, y=y_cost, mode="lines", name="総費用", line=dict(color=CHART_STYLE["danger"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[sales], y=[sales], mode="markers+text", name="現在", marker=dict(size=14, color=CHART_STYLE["good"], line=dict(width=1.5, color="white")), text=["現在"], textposition="top center"))
    if bep < max_x:
        fig.add_trace(go.Scatter(x=[bep], y=[bep], mode="markers+text", name="損益分岐点", marker=dict(size=14, color=CHART_STYLE["warning"], line=dict(width=1.5, color="white")), text=[f"BEP {int(bep/1000)}M"], textposition="top center"))
    fig.update_layout(
        title="損益分岐点分析",
        xaxis_title="売上規模",
        yaxis_title="金額",
        paper_bgcolor=CHART_STYLE["bg"],
        plot_bgcolor="white",
        font=dict(color="#334155", size=11),
        margin=dict(t=45, b=45, l=50, r=30),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=CHART_STYLE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=CHART_STYLE["grid"], zeroline=False)
    return fig
