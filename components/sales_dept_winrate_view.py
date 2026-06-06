"""
営業部別成約率ダッシュボード（REV-112）
past_cases テーブルの sales_dept カラムを元に営業部別成約率を集計・可視化する。
"""
import streamlit as st
import sqlite3
import os
import plotly.graph_objects as go
from runtime_paths import get_data_path


_DB_PATH = get_data_path("lease_data.db")
_SUCCESS = {"成約", "検収完了"}
_FAILURE = {"失注"}

_DEPT_NOTES: dict[str, str] = {
    "宇都宮営業部": "製造業・建設業案件が多い。スコア重視より関係構築型の成約が多い傾向。",
    "埼玉営業部": "情報通信・サービス業が中心。競合が多く成約率はやや低め。",
    "小山営業部": "農業・物流系が強い。設備投資ニーズが明確な案件が多い。",
    "足利営業部": "中小製造業が多い。初期相談から成約まで期間が長い傾向。",
}


def _load_data() -> list[dict]:
    if not os.path.exists(_DB_PATH):
        return []
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT sales_dept, final_status, COUNT(*), ROUND(AVG(score),1)
        FROM past_cases
        WHERE sales_dept NOT IN ('', '0', '未設定')
          AND final_status IS NOT NULL AND final_status != ''
        GROUP BY sales_dept, final_status
    """)
    rows = cur.fetchall()
    conn.close()

    agg: dict[str, dict] = {}
    for dept, status, cnt, _ in rows:
        d = agg.setdefault(dept, {"won": 0, "lost": 0, "other": 0, "avg_score": 0.0})
        if status in _SUCCESS:
            d["won"] += cnt
        elif status in _FAILURE:
            d["lost"] += cnt
        else:
            d["other"] += cnt

    # avg_score を別途取得
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT sales_dept, ROUND(AVG(score),1)
        FROM past_cases
        WHERE sales_dept NOT IN ('', '0', '未設定')
        GROUP BY sales_dept
    """)
    for dept, avg in cur.fetchall():
        if dept in agg:
            agg[dept]["avg_score"] = avg or 0.0
    conn.close()

    result = []
    for dept, d in agg.items():
        total = d["won"] + d["lost"]
        if total == 0:
            continue
        rate = d["won"] / total * 100
        result.append({
            "営業部": dept,
            "成約数": d["won"],
            "失注数": d["lost"],
            "合計": total,
            "成約率(%)": round(rate, 1),
            "平均スコア": d["avg_score"],
        })
    result.sort(key=lambda x: x["合計"], reverse=True)
    return result


def render_sales_dept_winrate_view() -> None:
    st.header("🏢 営業部別成約率ダッシュボード")
    st.caption("past_cases の sales_dept カラムから集計。成約＋検収完了を「成約」としてカウント。")

    data = _load_data()
    if not data:
        st.warning("集計対象データがありません（past_cases が空 or DB未接続）。")
        return

    total_won = sum(d["成約数"] for d in data)
    total_lost = sum(d["失注数"] for d in data)
    total_all = total_won + total_lost
    overall_rate = total_won / total_all * 100 if total_all > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("全体成約率", f"{overall_rate:.1f}%")
    c2.metric("累計成約", f"{total_won:,} 件")
    c3.metric("累計失注", f"{total_lost:,} 件")

    st.divider()

    depts = [d["営業部"] for d in data]
    rates = [d["成約率(%)"] for d in data]

    bar_colors = ["#16a34a" if r >= 60 else "#d97706" if r >= 45 else "#dc2626" for r in rates]

    fig = go.Figure(go.Bar(
        x=depts,
        y=rates,
        marker_color=bar_colors,
        text=[f"{r}%" for r in rates],
        textposition="outside",
        customdata=[[d["成約数"], d["失注数"], d["合計"], d["平均スコア"]] for d in data],
        hovertemplate=(
            "<b>%{x}</b><br>"
            "成約率: %{y:.1f}%<br>"
            "成約: %{customdata[0]}件 / 失注: %{customdata[1]}件 / 合計: %{customdata[2]}件<br>"
            "平均スコア: %{customdata[3]}pt<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="営業部別 成約率（緑: 60%以上 ／ 黄: 45〜60% ／ 赤: 45%未満）",
        xaxis_title="営業部",
        yaxis_title="成約率 (%)",
        yaxis=dict(range=[0, 105]),
        height=380,
        margin=dict(l=10, r=10, t=50, b=60),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        font=dict(size=12),
    )
    fig.add_hline(
        y=overall_rate, line_dash="dot", line_color="#6366f1",
        annotation_text=f"全体平均 {overall_rate:.1f}%", annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ─── 詳細テーブル ──────────────────────────────────────────────
    st.subheader("📋 営業部別 詳細一覧")

    table_rows = ""
    for d in data:
        rate = d["成約率(%)"]
        badge_color = "#16a34a" if rate >= 60 else "#d97706" if rate >= 45 else "#dc2626"
        diff = rate - overall_rate
        diff_str = f"+{diff:.1f}pt" if diff >= 0 else f"{diff:.1f}pt"
        diff_color = "#16a34a" if diff >= 0 else "#dc2626"
        table_rows += (
            f"<tr>"
            f"<td style='padding:.4rem .6rem;'>{d['営業部']}</td>"
            f"<td style='text-align:center;font-weight:700;color:{badge_color};'>{rate}%</td>"
            f"<td style='text-align:center;color:{diff_color};font-weight:600;'>{diff_str}</td>"
            f"<td style='text-align:center;'>{d['成約数']}</td>"
            f"<td style='text-align:center;'>{d['失注数']}</td>"
            f"<td style='text-align:center;'>{d['合計']}</td>"
            f"<td style='text-align:center;color:#475569;'>{d['平均スコア']}pt</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
        f"<thead><tr style='background:#f1f5f9;'>"
        f"<th style='text-align:left;padding:.4rem .6rem;'>営業部</th>"
        f"<th>成約率</th><th>全体比</th><th>成約</th><th>失注</th><th>合計</th><th>平均スコア</th>"
        f"</tr></thead><tbody>{table_rows}</tbody></table>",
        unsafe_allow_html=True,
    )

    # ─── 営業部特性メモ ────────────────────────────────────────────
    st.divider()
    st.subheader("📝 営業部 特性メモ（REV-110）")
    cols = st.columns(len(data))
    for i, d in enumerate(data):
        note = _DEPT_NOTES.get(d["営業部"], "特性データ未登録。")
        rate = d["成約率(%)"]
        badge_color = "#16a34a" if rate >= 60 else "#d97706" if rate >= 45 else "#dc2626"
        with cols[i]:
            st.markdown(
                f"<div style='background:#f8fafc;border-left:3px solid {badge_color};"
                f"border-radius:6px;padding:.6rem .8rem;font-size:.82rem;'>"
                f"<b>{d['営業部']}</b><br>"
                f"<span style='color:{badge_color};font-weight:700;'>{rate}%</span> / "
                f"平均 {d['平均スコア']}pt<br>"
                f"<span style='color:#64748b;'>{note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.caption("⚠️ 特性メモはサンプルデータです。実態に合わせて随時更新してください。")
