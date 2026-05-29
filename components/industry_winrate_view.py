"""
業種別成約率ダッシュボード（REV-055/117~119）
past_cases テーブルの実績データから業種別成約率を集計・可視化する。
"""
import streamlit as st
import sqlite3
import os
import plotly.graph_objects as go


_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lease_data.db")
_SUCCESS = {"成約", "検収完了"}
_FAILURE = {"失注"}


def _load_winrate() -> list[dict]:
    if not os.path.exists(_DB_PATH):
        return []
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT industry_sub, final_status, COUNT(*) FROM past_cases "
        "WHERE final_status IS NOT NULL AND final_status != '' "
        "GROUP BY industry_sub, final_status"
    )
    rows = cur.fetchall()
    conn.close()

    agg: dict[str, dict] = {}
    for industry, status, cnt in rows:
        if not industry or industry == "0":
            continue
        d = agg.setdefault(industry, {"won": 0, "lost": 0, "other": 0})
        if status in _SUCCESS:
            d["won"] += cnt
        elif status in _FAILURE:
            d["lost"] += cnt
        else:
            d["other"] += cnt

    result = []
    for industry, d in agg.items():
        total = d["won"] + d["lost"]
        if total == 0:
            continue
        rate = d["won"] / total * 100
        result.append({
            "業種": industry,
            "成約数": d["won"],
            "失注数": d["lost"],
            "合計": total,
            "成約率(%)": round(rate, 1),
        })
    result.sort(key=lambda x: x["合計"], reverse=True)
    return result


def render_industry_winrate_view() -> None:
    st.header("📊 業種別成約率ダッシュボード")
    st.caption("past_cases テーブルの実績データに基づく集計。成約＋検収完了を「成約」、失注を「失注」としてカウント。")

    data = _load_winrate()
    if not data:
        st.warning("集計対象データがありません（past_cases が空 or DB未接続）。")
        return

    total_won = sum(d["成約数"] for d in data)
    total_lost = sum(d["失注数"] for d in data)
    total_all = total_won + total_lost
    overall_rate = total_won / total_all * 100 if total_all > 0 else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("全体成約率", f"{overall_rate:.1f}%")
    c2.metric("累計成約", f"{total_won:,} 件")
    c3.metric("累計失注", f"{total_lost:,} 件")

    st.divider()

    # ─── フィルタ ─────────────────────────────────────────────
    min_cases = st.slider("最低件数（信頼性フィルタ）", 5, 50, 10, step=5,
                          help="件数が少ない業種は統計的信頼性が低いため除外できます。")
    filtered = [d for d in data if d["合計"] >= min_cases]

    if not filtered:
        st.info("フィルタ条件に合う業種がありません。最低件数を下げてください。")
        return

    industries = [d["業種"] for d in filtered]
    rates = [d["成約率(%)"] for d in filtered]
    totals = [d["合計"] for d in filtered]

    bar_colors = ["#16a34a" if r >= 60 else "#d97706" if r >= 45 else "#dc2626" for r in rates]

    fig = go.Figure(go.Bar(
        x=industries,
        y=rates,
        marker_color=bar_colors,
        text=[f"{r}%" for r in rates],
        textposition="outside",
        customdata=[[d["成約数"], d["失注数"], d["合計"]] for d in filtered],
        hovertemplate="<b>%{x}</b><br>成約率: %{y:.1f}%<br>成約: %{customdata[0]}件 / 失注: %{customdata[1]}件 / 合計: %{customdata[2]}件<extra></extra>",
    ))
    fig.update_layout(
        title="業種別 成約率（緑: 60%以上 ／ 黄: 45〜60% ／ 赤: 45%未満）",
        xaxis_title="業種",
        yaxis_title="成約率 (%)",
        yaxis=dict(range=[0, 105]),
        height=420,
        margin=dict(l=10, r=10, t=50, b=120),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font=dict(size=11),
        xaxis=dict(tickangle=-35),
    )
    # 全体平均ライン
    fig.add_hline(y=overall_rate, line_dash="dot", line_color="#6366f1",
                  annotation_text=f"全体平均 {overall_rate:.1f}%", annotation_position="top right")

    st.plotly_chart(fig, use_container_width=True)

    # ─── テーブル ──────────────────────────────────────────────
    st.subheader("📋 業種別成約率 一覧")
    st.caption(f"対象: {len(filtered)} 業種（合計 {min_cases} 件以上）")

    table_rows = ""
    for d in filtered:
        rate = d["成約率(%)"]
        badge_color = "#16a34a" if rate >= 60 else "#d97706" if rate >= 45 else "#dc2626"
        diff = rate - overall_rate
        diff_str = f"+{diff:.1f}pt" if diff >= 0 else f"{diff:.1f}pt"
        diff_color = "#16a34a" if diff >= 0 else "#dc2626"
        table_rows += (
            f"<tr>"
            f"<td>{d['業種']}</td>"
            f"<td style='text-align:center;font-weight:700;color:{badge_color};'>{rate}%</td>"
            f"<td style='text-align:center;'>{d['成約数']}</td>"
            f"<td style='text-align:center;'>{d['失注数']}</td>"
            f"<td style='text-align:center;'>{d['合計']}</td>"
            f"<td style='text-align:center;color:{diff_color};font-weight:600;'>{diff_str}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
        f"<thead><tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:.4rem .6rem;'>業種</th>"
        f"<th>成約率</th><th>成約</th><th>失注</th><th>合計</th><th>全体比</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>",
        unsafe_allow_html=True,
    )

    st.caption("⚠️ データが少ない業種（特に合計20件未満）の成約率は参考値です。営業判断の補助としてご活用ください。")
