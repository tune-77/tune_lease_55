# -*- coding: utf-8 -*-
"""
審査レポート — 審査・分析結果を見やすくレポート形式で表示するコンポーネント。
"""
import datetime
import streamlit as st

_REPORT_CSS = """
<style>
/* ── レポート全体 ─────────────────────────────── */
.rp-wrap { font-family: 'Noto Sans JP', sans-serif; max-width: 960px; margin: 0 auto; }

/* ── ヘッダーバナー ──────────────────────────── */
.rp-banner {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
}
.rp-banner.approved  { background: linear-gradient(135deg,#0d6e3b 0%,#16a34a 100%); }
.rp-banner.review    { background: linear-gradient(135deg,#92400e 0%,#d97706 100%); }
.rp-banner.rejected  { background: linear-gradient(135deg,#7f1d1d 0%,#b91c1c 100%); }
.rp-banner-left h2   { font-size:1.8rem; font-weight:800; margin:0; letter-spacing:.02em; }
.rp-banner-left p    { margin:.25rem 0 0; font-size:.95rem; opacity:.88; }
.rp-banner-right     { text-align:right; }
.rp-banner-score     { font-size:3rem; font-weight:900; line-height:1; }
.rp-banner-score-lbl { font-size:.8rem; opacity:.85; margin-top:.15rem; }

/* ── セクション共通 ──────────────────────────── */
.rp-section {
    background:#fff;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:1.2rem 1.4rem;
    margin-bottom:1rem;
    box-shadow:0 1px 4px rgba(30,58,95,.06);
}
.rp-section-title {
    font-size:.85rem;
    font-weight:700;
    color:#64748b;
    text-transform:uppercase;
    letter-spacing:.06em;
    margin:0 0 .9rem;
    padding-bottom:.5rem;
    border-bottom:2px solid #f1f5f9;
}

/* ── KPI グリッド ────────────────────────────── */
.rp-kpi-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(140px,1fr));
    gap:.75rem;
}
.rp-kpi {
    background:linear-gradient(145deg,#f8fafc,#f1f5f9);
    border-radius:10px;
    padding:.75rem .9rem;
    border-left:4px solid #2563eb;
    min-width:0;
}
.rp-kpi.good  { border-color:#16a34a; }
.rp-kpi.warn  { border-color:#d97706; }
.rp-kpi.bad   { border-color:#b91c1c; }
.rp-kpi-lbl   { font-size:.72rem; color:#64748b; margin-bottom:.3rem; font-weight:600; }
.rp-kpi-val   { font-size:1.35rem; font-weight:800; color:#1e293b; line-height:1.1; }
.rp-kpi-sub   { font-size:.68rem; color:#94a3b8; margin-top:.2rem; }

/* ── 財務比較テーブル ───────────────────────── */
.rp-table { width:100%; border-collapse:collapse; font-size:.84rem; }
.rp-table th {
    background:#f1f5f9; color:#334155;
    padding:.5rem .7rem; text-align:left;
    font-weight:600; border-bottom:2px solid #e2e8f0;
}
.rp-table td { padding:.45rem .7rem; border-bottom:1px solid #f1f5f9; color:#334155; }
.rp-table tr:last-child td { border-bottom:none; }
.rp-badge-good { background:#dcfce7;color:#166534;border-radius:4px;padding:2px 7px;font-size:.72rem;font-weight:700; }
.rp-badge-warn { background:#fef3c7;color:#92400e;border-radius:4px;padding:2px 7px;font-size:.72rem;font-weight:700; }
.rp-badge-bad  { background:#fee2e2;color:#7f1d1d;border-radius:4px;padding:2px 7px;font-size:.72rem;font-weight:700; }

/* ── 強みタグ ────────────────────────────────── */
.rp-tag {
    display:inline-block;
    background:#eff6ff; color:#1d4ed8;
    border:1px solid #bfdbfe;
    border-radius:20px; padding:3px 11px;
    font-size:.76rem; font-weight:600;
    margin:3px 3px 3px 0;
}
/* ── フッター ────────────────────────────────── */
.rp-footer {
    text-align:center; color:#94a3b8;
    font-size:.75rem; padding:1rem 0 .5rem;
}

/* ══════════════════════════════════════════════
   放射状スコアダッシュボード
   ══════════════════════════════════════════════ */
.rp-radial-wrap {
    display:flex;
    align-items:stretch;
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:18px;
    overflow:hidden;
    margin-bottom:1.6rem;
    box-shadow:0 4px 24px rgba(30,58,95,.10), 0 1px 4px rgba(30,58,95,.06);
}
/* 左：判定パネル */
.rp-radial-left {
    width:210px;
    min-width:210px;
    padding:2rem 1.5rem;
    color:#fff;
    display:flex;
    flex-direction:column;
    justify-content:center;
}
.rp-radial-verdict-icon { font-size:2.6rem; line-height:1; margin-bottom:.45rem; }
.rp-radial-verdict-text {
    font-size:1.5rem; font-weight:900;
    letter-spacing:.02em; margin-bottom:1.1rem;
    text-shadow:0 1px 4px rgba(0,0,0,.2);
}
.rp-radial-meta { font-size:.8rem; opacity:.88; line-height:1.65; }

/* 右：放射状チャートエリア（固定サイズ＝SVG座標系と一致させる） */
.rp-radial-chart-area {
    position:relative;
    width:520px;
    height:480px;
    flex-shrink:0;
}
.rp-radial-svg {
    position:absolute;
    top:0; left:0;
    width:520px; height:480px;
    pointer-events:none;
    overflow:visible;
}

/* 中央スコア円 */
.rp-radial-center {
    position:absolute;
    width:150px; height:150px;
    border-radius:50%;
    transform:translate(-50%,-50%);
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    color:#fff;
    z-index:2;
    border:4px solid rgba(255,255,255,.35);
}
.rp-radial-score-num {
    font-size:3rem; font-weight:900; line-height:1;
    text-shadow:0 2px 8px rgba(0,0,0,.3);
}
.rp-radial-score-unit { font-size:1.1rem; font-weight:700; opacity:.9; }
.rp-radial-score-sub  { font-size:.6rem; opacity:.75; margin-top:.15rem; letter-spacing:.06em; }

/* 放射ノード */
.rp-radial-node {
    position:absolute;
    width:112px;
    padding:.42rem .55rem;
    border-radius:10px;
    border:2px solid;
    text-align:center;
    z-index:3;
    box-shadow:0 2px 8px rgba(0,0,0,.08);
    transition:transform .15s, box-shadow .15s;
}
.rp-radial-node:hover {
    transform:scale(1.07);
    box-shadow:0 4px 16px rgba(0,0,0,.15);
}
.rp-radial-node-lbl {
    font-size:.62rem; font-weight:700;
    letter-spacing:.03em; opacity:.78;
    margin-bottom:.12rem;
}
.rp-radial-node-val { font-size:1.05rem; font-weight:800; line-height:1.1; }

/* レスポンシブ（スマホ）*/
@media (max-width:780px) {
    .rp-radial-wrap { flex-direction:column; }
    .rp-radial-left { width:100%; min-width:unset; flex-direction:row; flex-wrap:wrap; gap:.5rem; }
    .rp-radial-chart-area { transform:scale(0.62); transform-origin:top left; height:300px; }
}
</style>
"""


import math as _math


def _radial_color(val, bench, higher_is_better=True) -> tuple[str, str, str]:
    """(テキスト色, 背景色, ボーダー色) を返す。val/bench が None なら中立色。"""
    if val is None or bench is None:
        return "#64748b", "#f1f5f9", "#cbd5e1"
    diff = val - bench
    if (higher_is_better and diff >= 2) or (not higher_is_better and diff <= -2):
        return "#15803d", "#dcfce7", "#86efac"   # 良好：緑
    if (higher_is_better and diff <= -5) or (not higher_is_better and diff >= 5):
        return "#b91c1c", "#fee2e2", "#fca5a5"   # 危険：赤
    if (higher_is_better and diff <= -2) or (not higher_is_better and diff >= 2):
        return "#b45309", "#fef3c7", "#fcd34d"   # 要注意：黄
    return "#1d4ed8", "#eff6ff", "#93c5fd"       # 標準：青


def _build_radial_dashboard_html(
    score: float,
    hantei: str,
    hantei_icon: str,
    banner_cls: str,
    ts: str,
    industry_sub: str,
    industry_major: str,
    asset_name: str,
    user_op, bench_op,
    user_eq_d, bench_eq_d,
    user_roa, bench_roa,
    user_cr, bench_cr,
    user_dscr,
    qs_rank: str,
    scr_decision: str,
) -> str:
    """放射状スコアダッシュボードの HTML 文字列を生成。"""

    # ── 左パネル背景グラデーション ──
    _grad = {
        "approved": "linear-gradient(150deg,#064e3b 0%,#059669 100%)",
        "review":   "linear-gradient(150deg,#78350f 0%,#d97706 100%)",
        "rejected": "linear-gradient(150deg,#7f1d1d 0%,#dc2626 100%)",
    }
    left_bg = _grad.get(banner_cls, "linear-gradient(150deg,#1e3a5f,#2563eb)")

    # ── 中央円グラデーション＋グロー ──
    if score >= 71:
        c_grad = "linear-gradient(135deg,#065f46 0%,#10b981 100%)"
        c_glow = "rgba(16,185,129,.45)"
    elif score >= 40:
        c_grad = "linear-gradient(135deg,#78350f 0%,#f59e0b 100%)"
        c_glow = "rgba(245,158,11,.45)"
    else:
        c_grad = "linear-gradient(135deg,#7f1d1d 0%,#ef4444 100%)"
        c_glow = "rgba(239,68,68,.45)"

    # ── ノード定義 ──
    # DSCR は bench=1.2 固定
    dscr_txt, dscr_bg, dscr_bd = _radial_color(user_dscr, 1.2)
    # 学習モデル判定は独自色
    if scr_decision == "承認":
        sm_txt, sm_bg, sm_bd = "#15803d", "#dcfce7", "#86efac"
    elif scr_decision == "否決":
        sm_txt, sm_bg, sm_bd = "#b91c1c", "#fee2e2", "#fca5a5"
    else:
        sm_txt, sm_bg, sm_bd = "#64748b", "#f1f5f9", "#cbd5e1"

    def _fv(v, fmt=".1f", sfx=""):
        return f"{v:{fmt}}{sfx}" if v is not None else "—"

    nodes = [
        ("営業利益率",   _fv(user_op,  ".1f", "%"), *_radial_color(user_op,  bench_op)),
        ("自己資本比率", _fv(user_eq_d,".1f", "%"), *_radial_color(user_eq_d,bench_eq_d)),
        ("ROA",          _fv(user_roa, ".1f", "%"), *_radial_color(user_roa, bench_roa)),
        ("流動比率",     _fv(user_cr,  ".0f", "%"), *_radial_color(user_cr,  bench_cr)),
        ("DSCR",         _fv(user_dscr,".2f", "倍"), dscr_txt, dscr_bg, dscr_bd),
        ("定性ランク",   qs_rank or "—",            "#475569", "#f1f5f9", "#cbd5e1"),
        ("学習モデル",   scr_decision or "—",        sm_txt, sm_bg, sm_bd),
    ]

    # ── 放射状座標計算 ──
    CX, CY = 260, 235
    R      = 160
    N      = len(nodes)
    NW, NH = 112, 54   # ノード box サイズ

    svg_lines  = ""
    node_divs  = ""

    for i, (lbl, val, tc, bg, bd) in enumerate(nodes):
        angle_rad = _math.radians(-90 + i * 360 / N)
        nx = CX + R * _math.cos(angle_rad)
        ny = CY + R * _math.sin(angle_rad)

        # SVG: 点線 + 端点ドット
        svg_lines += (
            f'<line x1="{CX}" y1="{CY}" x2="{nx:.1f}" y2="{ny:.1f}" '
            f'stroke="{bd}" stroke-width="1.8" stroke-dasharray="5,3" opacity="0.65"/>'
            f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="4.5" fill="{tc}" opacity="0.85"/>'
        )

        # 絶対配置ノード
        lx = nx - NW / 2
        ly = ny - NH / 2
        node_divs += (
            f'<div class="rp-radial-node" '
            f'style="left:{lx:.1f}px;top:{ly:.1f}px;'
            f'background:{bg};border-color:{bd};color:{tc};">'
            f'<div class="rp-radial-node-lbl">{lbl}</div>'
            f'<div class="rp-radial-node-val" style="color:{tc};">{val}</div>'
            f'</div>'
        )

    return f"""
<div class="rp-radial-wrap">
  <!-- 左：判定パネル -->
  <div class="rp-radial-left" style="background:{left_bg};">
    <div class="rp-radial-verdict-icon">{hantei_icon}</div>
    <div class="rp-radial-verdict-text">{hantei}</div>
    <div class="rp-radial-meta">📋&nbsp;{industry_major}</div>
    <div class="rp-radial-meta" style="padding-left:.9rem;">{industry_sub}</div>
    <div class="rp-radial-meta" style="margin-top:.4rem;">🏭&nbsp;{asset_name}</div>
    <div class="rp-radial-meta" style="margin-top:1rem;font-size:.7rem;opacity:.65;">{ts}</div>
  </div>

  <!-- 右：放射状エリア -->
  <div class="rp-radial-chart-area">
    <!-- 背景リング・接続線 -->
    <svg class="rp-radial-svg" viewBox="0 0 520 480" xmlns="http://www.w3.org/2000/svg">
      <circle cx="{CX}" cy="{CY}" r="{R}" fill="none"
              stroke="#e2e8f0" stroke-width="1.2" stroke-dasharray="6,5"/>
      <circle cx="{CX}" cy="{CY}" r="{int(R*0.5)}" fill="none"
              stroke="#f1f5f9" stroke-width="1" stroke-dasharray="3,4"/>
      {svg_lines}
    </svg>

    <!-- 中央スコア円 -->
    <div class="rp-radial-center"
         style="left:{CX}px;top:{CY}px;background:{c_grad};
                box-shadow:0 0 36px {c_glow},0 0 72px {c_glow.replace('.45','.18')};">
      <div class="rp-radial-score-num">{score:.0f}</div>
      <div class="rp-radial-score-unit">点</div>
      <div class="rp-radial-score-sub">TOTAL SCORE</div>
    </div>

    <!-- ノード群 -->
    {node_divs}
  </div>
</div>
"""


def _badge(val, bench, higher_is_better=True) -> str:
    if val is None or bench is None:
        return ""
    diff = val - bench
    if higher_is_better:
        if diff >= 2:   return '<span class="rp-badge-good">▲ 業界超</span>'
        if diff <= -5:  return '<span class="rp-badge-bad">▼ 要改善</span>'
        if diff <= -2:  return '<span class="rp-badge-warn">△ 要注意</span>'
    else:
        if diff <= -2:  return '<span class="rp-badge-good">▲ 良好</span>'
        if diff >= 5:   return '<span class="rp-badge-bad">▼ 要改善</span>'
        if diff >= 2:   return '<span class="rp-badge-warn">△ 要注意</span>'
    return '<span style="color:#64748b;font-size:.72rem;">― 標準</span>'


def _kpi_color(val, bench, higher_is_better=True) -> str:
    if val is None or bench is None:
        return ""
    diff = val - bench
    if higher_is_better:
        if diff >= 2:   return "good"
        if diff <= -2:  return "bad"
    else:
        if diff <= -2:  return "good"
        if diff >= 2:   return "bad"
    return ""


def _fmt(v, fmt=".1f", suffix="") -> str:
    if v is None:
        return "—"
    try:
        return f"{v:{fmt}}{suffix}"
    except Exception:
        return str(v)


def render_report() -> None:
    res = st.session_state.get("last_result")

    st.markdown(_REPORT_CSS, unsafe_allow_html=True)

    if not res:
        st.info("まだ審査が実行されていません。「審査・分析」から審査を実行してください。")
        return

    # ── データ取り出し ──────────────────────────────────────────────────────
    score        = res.get("score") or 0
    hantei       = res.get("hantei") or "—"
    contract_prob= res.get("contract_prob") or 0
    pd_pct       = res.get("pd_percent")
    industry_sub = res.get("industry_sub") or "—"
    industry_major=res.get("industry_major") or ""
    asset_name   = res.get("asset_name") or "—"
    yield_pred   = res.get("yield_pred")
    user_op      = res.get("user_op")
    bench_op     = res.get("bench_op")
    user_eq      = res.get("user_eq")
    bench_eq     = res.get("bench_eq")
    user_roa     = res.get("user_roa")
    bench_roa    = res.get("bench_roa")
    user_cr      = res.get("user_current_ratio")
    bench_cr     = res.get("bench_current_ratio")
    user_dscr    = res.get("user_dscr")
    fin          = res.get("financials") or {}
    strength_tags= res.get("strength_tags") or []
    passion_text = res.get("passion_text") or ""
    comparison   = res.get("comparison") or ""
    hints        = res.get("hints") or {}
    scoring_result=res.get("scoring_result") or {}
    qs_result    = res.get("qualitative_scoring_correction") or {}
    qs_rank      = qs_result.get("rank_text") or ""
    qs_score     = qs_result.get("weighted_score")

    try:
        from charts import _equity_ratio_display
        user_eq_d  = _equity_ratio_display(user_eq)
        bench_eq_d = _equity_ratio_display(bench_eq)
    except Exception:
        user_eq_d, bench_eq_d = user_eq, bench_eq

    # ── バナークラス決定 ────────────────────────────────────────────────────
    if "承認" in hantei:
        banner_cls, hantei_icon = "approved", "✅"
    elif "否決" in hantei:
        banner_cls, hantei_icon = "rejected", "❌"
    else:
        banner_cls, hantei_icon = "review", "⚠️"

    ts = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # ── ① 放射状スコアダッシュボード ─────────────────────────────────────────
    st.markdown(
        _build_radial_dashboard_html(
            score, hantei, hantei_icon, banner_cls, ts,
            industry_sub, industry_major, asset_name,
            user_op, bench_op, user_eq_d, bench_eq_d,
            user_roa, bench_roa, user_cr, bench_cr,
            user_dscr, qs_rank,
            scoring_result.get("decision") or "—",
        ),
        unsafe_allow_html=True,
    )

    # ── ② ヘッダーバナー ───────────────────────────────────────────────────
    st.markdown(f"""
<div class="rp-wrap">
<div class="rp-banner {banner_cls}">
  <div class="rp-banner-left">
    <h2>{hantei_icon} {hantei}</h2>
    <p>📋 {industry_major} &nbsp;›&nbsp; {industry_sub}</p>
    <p>🏭 物件: {asset_name}</p>
    <p style="font-size:.8rem;opacity:.7;margin-top:.4rem;">作成日時: {ts}</p>
  </div>
  <div class="rp-banner-right">
    <div class="rp-banner-score">{score:.0f}<span style="font-size:1.4rem;">点</span></div>
    <div class="rp-banner-score-lbl">総合スコア（100点満点）</div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── ① チャート行 1: ゲージ ＋ レーダー ─────────────────────────────────
    try:
        from charts import plot_gauge_plotly, plot_radar_chart_plotly
        col_g, col_r = st.columns([1, 2])
        with col_g:
            fig_gauge = plot_gauge_plotly(score)
            if fig_gauge is not None:
                st.plotly_chart(fig_gauge, use_container_width=True)
        with col_r:
            metrics = {
                "営業利益率":    user_op or 0,
                "自己資本比率":  user_eq_d or 0,
                "ROA":           user_roa or 0,
                "流動比率":      user_cr or 0,
                "DSCR×10":       (user_dscr or 0) * 10,
            }
            benchmarks = {
                "営業利益率":    bench_op or 0,
                "自己資本比率":  bench_eq_d or 0,
                "ROA":           bench_roa or 0,
                "流動比率":      bench_cr or 0,
                "DSCR×10":       12,
            }
            fig_radar = plot_radar_chart_plotly(metrics, benchmarks)
            if fig_radar is not None:
                st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.caption(f"⚠️ チャート描画エラー: {e}")

    # ── ① チャート行 2: Top5要因 ＋ バランスシート ──────────────────────────
    try:
        from charts import plot_scoring_top5_factors_plotly, plot_balance_sheet_plotly
        fig_top5 = plot_scoring_top5_factors_plotly(scoring_result)
        fig_bs   = plot_balance_sheet_plotly(fin)
        col_t, col_b = st.columns([1, 1])
        with col_t:
            if fig_top5 is not None:
                st.plotly_chart(fig_top5, use_container_width=True)
            else:
                st.caption("Top5要因データなし")
        with col_b:
            if fig_bs is not None:
                st.plotly_chart(fig_bs, use_container_width=True)
            else:
                st.caption("財務データなし")
    except Exception as e:
        st.caption(f"⚠️ チャート描画エラー: {e}")

    # ── ① チャート行 3: 算出可能指標の業界差（全幅）──────────────────────────
    try:
        from charts import plot_indicators_gap_analysis_plotly
        _gap_indicators = [
            {"name": "営業利益率",   "value": user_op or 0,    "bench": bench_op,   "unit": "%"},
            {"name": "自己資本比率", "value": user_eq_d or 0,  "bench": bench_eq_d, "unit": "%"},
            {"name": "ROA",          "value": user_roa or 0,   "bench": bench_roa,  "unit": "%"},
            {"name": "流動比率",     "value": user_cr or 0,    "bench": bench_cr,   "unit": "%"},
            {"name": "DSCR",         "value": user_dscr or 0,  "bench": 1.2,        "unit": "倍"},
        ]
        # bench が None の指標は除外（関数内でも除外されるが明示的に）
        _gap_indicators = [i for i in _gap_indicators if i["bench"] is not None]
        fig_gap = plot_indicators_gap_analysis_plotly(_gap_indicators)
        if fig_gap is not None:
            st.plotly_chart(fig_gap, use_container_width=True)
    except Exception as e:
        st.caption(f"⚠️ 差分チャート描画エラー: {e}")

    col1, col2 = st.columns([3, 2])

    with col1:
        # ── ② KPI グリッド ──────────────────────────────────────────────────
        kpi_items = [
            ("契約期待度",    f"{contract_prob:.1f}%", "", _kpi_color(contract_prob, 50)),
            ("予測利回り",    _fmt(yield_pred, ".2f", "%"), "", ""),
            ("デフォルト率",  _fmt(pd_pct, ".1f", "%"), "", "bad" if (pd_pct or 0) > 10 else "good" if (pd_pct or 0) < 5 else ""),
            ("営業利益率",    _fmt(user_op, ".1f", "%"), f"業界 {_fmt(bench_op,'.1f','%')}", _kpi_color(user_op, bench_op)),
            ("自己資本比率",  _fmt(user_eq_d, ".1f", "%"), f"業界 {_fmt(bench_eq_d,'.1f','%')}", _kpi_color(user_eq_d, bench_eq_d)),
            ("流動比率",      _fmt(user_cr, ".0f", "%"), f"業界 {_fmt(bench_cr,'.0f','%')}", _kpi_color(user_cr, bench_cr)),
            ("DSCR",          _fmt(user_dscr, ".2f", "倍"), "1.2倍以上が目安", "good" if (user_dscr or 0) >= 1.2 else "bad"),
            ("定性ランク",    qs_rank or "—", f"{_fmt(qs_score,'.0f','点')}" if qs_score else "", ""),
        ]
        kpi_html = '<div class="rp-section"><p class="rp-section-title">📊 主要指標</p><div class="rp-kpi-grid">'
        for lbl, val, sub, cls in kpi_items:
            kpi_html += f'<div class="rp-kpi {cls}"><div class="rp-kpi-lbl">{lbl}</div><div class="rp-kpi-val">{val}</div><div class="rp-kpi-sub">{sub}</div></div>'
        kpi_html += '</div></div>'
        st.markdown(kpi_html, unsafe_allow_html=True)

        # ── ③ 財務比較テーブル ──────────────────────────────────────────────
        nenshu     = fin.get("nenshu") or 0
        op_profit  = fin.get("op_profit") or fin.get("rieki") or 0
        ord_profit = fin.get("ord_profit") or 0
        net_income = fin.get("net_income") or 0
        assets     = fin.get("assets") or 0
        net_assets = fin.get("net_assets") or 0
        dep        = fin.get("depreciation") or 0
        bank_cr    = fin.get("bank_credit") or 0
        lease_cr   = fin.get("lease_credit") or 0

        rows = [
            ("売上高",      f"{nenshu:,.0f} 万円",         "—",                           ""),
            ("営業利益率",  f"{_fmt(user_op,'.1f','%')}",  f"{_fmt(bench_op,'.1f','%')}", _badge(user_op, bench_op)),
            ("自己資本比率",f"{_fmt(user_eq_d,'.1f','%')}", f"{_fmt(bench_eq_d,'.1f','%')}", _badge(user_eq_d, bench_eq_d)),
            ("ROA",         f"{_fmt(user_roa,'.1f','%')}",  f"{_fmt(bench_roa,'.1f','%')}", _badge(user_roa, bench_roa)),
            ("流動比率",    f"{_fmt(user_cr,'.0f','%')}",   f"{_fmt(bench_cr,'.0f','%')}", _badge(user_cr, bench_cr)),
            ("純資産",      f"{net_assets:,.0f} 万円",      "—",                           ""),
            ("総資産",      f"{assets:,.0f} 万円",          "—",                           ""),
            ("EBITDA",      f"{op_profit+dep:,.0f} 百万円", "—",                           ""),
            ("銀行与信残",  f"{bank_cr:,.0f} 百万円",       "—",                           ""),
            ("リース信用残",f"{lease_cr:,.0f} 百万円",      "—",                           ""),
        ]
        tbl = '<div class="rp-section"><p class="rp-section-title">💹 財務データ・業界比較</p>'
        tbl += '<table class="rp-table"><thead><tr><th>指標</th><th>当社</th><th>業界平均</th><th>評価</th></tr></thead><tbody>'
        for lbl, val, bch, badge in rows:
            tbl += f'<tr><td>{lbl}</td><td><b>{val}</b></td><td>{bch}</td><td>{badge}</td></tr>'
        tbl += '</tbody></table></div>'
        st.markdown(tbl, unsafe_allow_html=True)

    with col2:
        # ── ④ 学習モデル判定 ────────────────────────────────────────────────
        scr_decision  = scoring_result.get("decision") or "—"
        scr_prob      = scoring_result.get("prob_approval")
        scr_icon = "✅" if scr_decision == "承認" else "❌" if scr_decision == "否決" else "⚠️"
        scr_html = f"""
<div class="rp-section">
  <p class="rp-section-title">🤖 学習モデル判定</p>
  <div style="font-size:1.5rem;font-weight:800;color:#1e293b;">{scr_icon} {scr_decision}</div>
  {"<div style='font-size:.85rem;color:#64748b;margin-top:.4rem;'>承認確率: <b>" + _fmt(scr_prob*100 if scr_prob else None, ".1f", "%") + "</b></div>" if scr_prob is not None else ""}
</div>"""
        st.markdown(scr_html, unsafe_allow_html=True)

        # ── ⑤ 財務評価コメント ──────────────────────────────────────────────
        if comparison:
            st.markdown(f"""
<div class="rp-section">
  <p class="rp-section-title">📝 財務評価コメント</p>
  <div style="font-size:.84rem;color:#334155;line-height:1.7;">{comparison[:600]}</div>
</div>""", unsafe_allow_html=True)

        # ── ⑥ 強みタグ ──────────────────────────────────────────────────────
        if strength_tags or passion_text:
            tags_html = "".join(f'<span class="rp-tag">✨ {t}</span>' for t in strength_tags)
            passion_html = f'<div style="font-size:.82rem;color:#475569;margin-top:.6rem;line-height:1.65;">{passion_text[:200]}</div>' if passion_text else ""
            st.markdown(f"""
<div class="rp-section">
  <p class="rp-section-title">💪 強み・定性評価</p>
  {tags_html}
  {passion_html}
</div>""", unsafe_allow_html=True)

        # ── ⑦ 審査ヒント ────────────────────────────────────────────────────
        if hints:
            hint_items = ""
            for k, v in list(hints.items())[:5]:
                hint_items += f'<li style="margin-bottom:.4rem;"><b>{k}</b>: {str(v)[:120]}</li>'
            st.markdown(f"""
<div class="rp-section">
  <p class="rp-section-title">💡 審査ヒント</p>
  <ul style="padding-left:1.2rem;font-size:.82rem;color:#334155;line-height:1.7;margin:0;">{hint_items}</ul>
</div>""", unsafe_allow_html=True)

    # ── ⑧ 印刷・PDF ─────────────────────────────────────────────────────────
    st.markdown('<div class="rp-footer">温水式 リース審査AI — 審査レポート</div>', unsafe_allow_html=True)
    st.divider()

    col_dl1, col_dl2, _ = st.columns([1, 1, 3])
    with col_dl1:
        import json
        report_json = json.dumps(res, ensure_ascii=False, default=str, indent=2)
        st.download_button(
            "📥 JSON でダウンロード",
            data=report_json,
            file_name=f"lease_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        if st.button("🖨️ 印刷 / PDF 保存", use_container_width=True, help="ブラウザの印刷機能でPDF保存できます"):
            st.markdown('<script>window.print();</script>', unsafe_allow_html=True)
