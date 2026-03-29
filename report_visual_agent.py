# -*- coding: utf-8 -*-
"""
report_visual_agent.py — ビジュアル審査レポート生成モジュール

機能:
  - collect_report_data  : st.session_state から全データを収集・整形
  - generate_html_report : ダークテーマのスタイリッシュ HTML 生成（自己完結型）
  - generate_pdf_report  : reportlab を使った A4 縦 PDF 生成
"""
from __future__ import annotations

import math
import datetime
import io
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# 1. データ収集
# ══════════════════════════════════════════════════════════════════════════════

def collect_report_data(session_state: Any) -> dict:
    """st.session_state から審査レポート用データを収集・整形する。"""
    res    = session_state.get("last_result") or {}
    inputs = session_state.get("last_submitted_inputs") or {}
    mc_pf  = session_state.get("mc_portfolio_result")

    # 基本スコア
    score          = float(res.get("score",          0) or 0)
    hantei         = str(res.get("hantei",           "—"))
    industry_major = str(res.get("industry_major",   ""))
    industry_sub   = str(res.get("industry_sub",     ""))
    pd_percent     = float(res.get("pd_percent",     0) or 0)

    # 物件・借手スコア（total_scorer_result から）
    tsr            = res.get("total_scorer_result") or {}
    asset_score    = min(100.0, max(0.0, float(tsr.get("asset_score",    0) or 0)))
    borrower_score = min(100.0, max(0.0, float(tsr.get("borrower_score", 0) or 0)))

    # 定性スコア
    qual_result = session_state.get("qualitative_analysis_result") or {}
    qual_score  = min(100.0, max(0.0, float((qual_result or {}).get("total_score", 0) or 0)))

    # 財務指標
    user_eq    = float(res.get("user_eq",    0) or 0)
    user_op    = float(res.get("user_op",    0) or 0)
    bench_eq   = float(res.get("bench_eq",   0) or 0)
    bench_op   = float(res.get("bench_op",   0) or 0)
    contract_p = float(res.get("contract_prob", 0) or 0)
    roa        = float(res.get("roa",        0) or 0)

    # モンテカルロ
    mc_data: dict = {}
    if mc_pf:
        mc_data = {
            "weighted_default_prob": float(getattr(mc_pf, "weighted_default_prob", 0) or 0),
            "concentration_risk":    float(getattr(mc_pf, "concentration_risk",    0) or 0),
            "expected_loss":         float(getattr(mc_pf, "expected_loss",         0) or 0),
            "portfolio_var_95":      float(getattr(mc_pf, "portfolio_var_95",      0) or 0),
        }

    # ベイジアンネットワーク
    bn_result        = session_state.get("_bn_s_result") or {}
    bn_evidence      = session_state.get("_bn_s_evidence") or {}
    bn_approval_prob = float((bn_result or {}).get("承認確率", 0) or 0)

    # AIコメント（軍師テキストのキャッシュがあれば）
    ai_comment = str(session_state.get("_ai_quick_comment_text") or "")

    # 業界ニュース
    news_results = session_state.get("news_results") or []
    news_items: list[dict] = []
    for n in (news_results or [])[:3]:
        if isinstance(n, dict):
            news_items.append({
                "title": str(n.get("title", ""))[:100],
                "body":  str(n.get("body",  ""))[:250],
            })

    # 補助金情報
    subsidy_data = list(session_state.get("_subsidy_results") or [])

    # 企業名・担当者
    company_name = (
        session_state.get("rep_company")
        or (inputs or {}).get("company_name", "")
        or "（企業名未設定）"
    )
    screener = str(session_state.get("rep_screener") or "")

    # SHAP Top5（あれば）
    shap_top5 = list(session_state.get("_shap_top5_items") or [])

    # 入力値から財務数値（千円単位）
    nenshu     = float(inputs.get("nenshu", 0) or 0)
    net_assets = float(inputs.get("net_assets", 0) or 0)
    lease_term = int(inputs.get("lease_term", 0) or 0)
    acq_cost   = float(inputs.get("acquisition_cost", 0) or 0)

    return {
        "company_name":    str(company_name),
        "screener":        screener,
        "date":            datetime.datetime.now().strftime("%Y年%m月%d日"),
        "hantei":          hantei,
        "score":           score,
        "asset_score":     asset_score,
        "borrower_score":  borrower_score,
        "qual_score":      qual_score,
        "pd_percent":      pd_percent,
        "industry_major":  industry_major,
        "industry_sub":    industry_sub,
        "user_eq":         user_eq,
        "user_op":         user_op,
        "bench_eq":        bench_eq,
        "bench_op":        bench_op,
        "contract_prob":   contract_p,
        "roa":             roa,
        "mc_data":         mc_data,
        "bn_approval_prob": bn_approval_prob,
        "bn_evidence":     bn_evidence,
        "ai_comment":      ai_comment,
        "news_items":      news_items,
        "subsidy_data":    subsidy_data,
        "shap_top5":       shap_top5,
        "nenshu":          nenshu,
        "net_assets":      net_assets,
        "lease_term":      lease_term,
        "acq_cost":        acq_cost,
        "inputs":          inputs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. SVG ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def _circular_gauge_svg(score: float, label: str, size: int = 130,
                         color: str = "#f0b429") -> str:
    """円形ゲージ SVG（0-100 スコア表示用）。"""
    pct = max(0.0, min(100.0, float(score)))
    r   = size * 0.34
    cx  = cy = size / 2
    circumference = 2 * math.pi * r
    arc_len = circumference * pct / 100.0
    gap_len = circumference - arc_len
    font_main = int(size * 0.18)
    font_sub  = int(size * 0.065)
    font_lbl  = int(size * 0.07)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none"'
        f' stroke="#1e2a45" stroke-width="9"/>'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none"'
        f' stroke="{color}" stroke-width="9" stroke-linecap="round"'
        f' stroke-dasharray="{arc_len:.2f} {gap_len:.2f}"'
        f' transform="rotate(-90 {cx:.1f} {cy:.1f})"/>'
        f'<text x="{cx:.1f}" y="{cy - 3:.1f}" text-anchor="middle"'
        f' font-size="{font_main}" font-weight="bold" fill="white"'
        f' font-family="\'Noto Sans JP\', sans-serif">{pct:.0f}</text>'
        f'<text x="{cx:.1f}" y="{cy + font_sub + 2:.1f}" text-anchor="middle"'
        f' font-size="{font_sub}" fill="#aabbcc"'
        f' font-family="\'Noto Sans JP\', sans-serif">/ 100</text>'
        f'<text x="{cx:.1f}" y="{size - 5:.1f}" text-anchor="middle"'
        f' font-size="{font_lbl}" fill="#ccd" '
        f' font-family="\'Noto Sans JP\', sans-serif">{label}</text>'
        f'</svg>'
    )


def _radar_svg(values: list, labels: list, size: int = 260) -> str:
    """レーダーチャート SVG（N軸、各値 0-100）。"""
    n = len(values)
    if n < 3:
        return ""
    cx = cy = size / 2
    r  = size * 0.36
    angles = [2 * math.pi * i / n - math.pi / 2 for i in range(n)]

    # 背景グリッド（25 / 50 / 75 / 100%）
    rings = []
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(
            f"{cx + r * frac * math.cos(a):.2f},{cy + r * frac * math.sin(a):.2f}"
            for a in angles
        )
        rings.append(f'<polygon points="{pts}" fill="none" stroke="#1e2a45" stroke-width="1"/>')

    # 軸線
    axes = [
        f'<line x1="{cx:.2f}" y1="{cy:.2f}" '
        f'x2="{cx + r * math.cos(a):.2f}" y2="{cy + r * math.sin(a):.2f}" '
        f'stroke="#1e2a45" stroke-width="1"/>'
        for a in angles
    ]

    # データポリゴン
    data_pts = " ".join(
        f"{cx + r * (max(0.0, min(100.0, v)) / 100.0) * math.cos(a):.2f},"
        f"{cy + r * (max(0.0, min(100.0, v)) / 100.0) * math.sin(a):.2f}"
        for v, a in zip(values, angles)
    )

    # ラベル
    pad = 20
    label_elems = []
    for a, lab in zip(angles, labels):
        lx = cx + (r + pad) * math.cos(a)
        ly = cy + (r + pad) * math.sin(a)
        anchor = "middle"
        if lx < cx - 8:
            anchor = "end"
        elif lx > cx + 8:
            anchor = "start"
        label_elems.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-size="11" fill="#ccd" '
            f'font-family="\'Noto Sans JP\', sans-serif">{lab}</text>'
        )

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        + "".join(rings)
        + "".join(axes)
        + f'<polygon points="{data_pts}" fill="#f0b42930" stroke="#f0b429" stroke-width="2"/>'
        + "".join(label_elems)
        + "</svg>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. HTML レポート生成
# ══════════════════════════════════════════════════════════════════════════════

def _hantei_badge_colors(hantei: str) -> tuple[str, str]:
    """判定バッジの（文字色, 背景色）を返す。"""
    return {
        "承認":     ("#0a0e1a", "#00c853"),
        "条件付き": ("#0a0e1a", "#f0b429"),
        "要審議":   ("#0a0e1a", "#ff9800"),
        "否決":     ("#ffffff", "#d32f2f"),
    }.get(hantei, ("#ffffff", "#546e7a"))


def _esc(text: str) -> str:
    """HTML エスケープ。"""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_html_report(data: dict) -> str:
    """
    収集済みデータからスタイリッシュな HTML レポートを生成する。
    Google Fonts（Noto Sans JP）を使用するため閲覧時にインターネット接続が必要。
    CSS・SVG は完全自己完結。
    """
    company   = _esc(data["company_name"])
    screener  = _esc(data["screener"])
    date      = _esc(data["date"])
    hantei    = _esc(data["hantei"])
    score     = data["score"]
    asset_s   = data["asset_score"]
    borrow_s  = data["borrower_score"]
    qual_s    = data["qual_score"]
    pd_pct    = data["pd_percent"]
    ind_major = _esc(data["industry_major"])
    ind_sub   = _esc(data["industry_sub"])
    user_eq   = data["user_eq"]
    user_op   = data["user_op"]
    mc        = data["mc_data"]
    bn_prob   = data["bn_approval_prob"]
    ai_cmt    = _esc(data["ai_comment"])
    news      = data["news_items"]
    nenshu    = data["nenshu"]
    net_assets = data["net_assets"]
    acq_cost  = data["acq_cost"]
    lease_term = data["lease_term"]

    txt_col, badge_bg = _hantei_badge_colors(data["hantei"])

    # SVG ゲージ
    gauge_total  = _circular_gauge_svg(score,   "総合スコア", 150, "#f0b429")
    gauge_asset  = _circular_gauge_svg(asset_s, "物件スコア", 125, "#4fc3f7")
    gauge_borrow = _circular_gauge_svg(borrow_s,"財務スコア", 125, "#81c784")
    gauge_qual   = _circular_gauge_svg(qual_s,  "定性スコア", 125, "#ce93d8")

    # レーダーチャート
    radar_svg = _radar_svg(
        [score, asset_s, borrow_s, qual_s, min(100.0, bn_prob)],
        ["総合", "物件", "財務", "定性", "BN推定"],
        260,
    )

    # BN Evidence HTML
    bn_ev_html_parts = []
    _EV_LABELS = {
        "Insolvent_Status": "債務超過回避",
        "Main_Bank_Support": "メイン銀行支援",
        "Related_Bank_Status": "関係者銀行取引",
        "Related_Assets": "個人資産保有",
        "Co_Lease": "協調リース",
        "Parent_Guarantor": "親会社保証",
        "Core_Business_Use": "本業不可欠物件",
        "Asset_Liquidity": "物件流動性",
        "Shorter_Lease_Term": "期間短縮",
        "One_Time_Deal": "本件限り",
    }
    bn_evidence = data.get("bn_evidence", {}) or {}
    for k, v in bn_evidence.items():
        if v == 1:
            lbl = _EV_LABELS.get(k, k)
            bn_ev_html_parts.append(
                f'<div class="metric-card"><div class="metric-label">{lbl}</div>'
                f'<div class="metric-value good" style="font-size:14px;">証拠あり</div></div>'
            )
    bn_ev_html = "".join(bn_ev_html_parts) or '<p class="empty" style="grid-column: span 2;">※ 定性的なポジティブ・エビデンスの選択なし</p>'

    # Industry Comparison
    bench_eq = data.get("bench_eq", 0) or 0
    bench_op = data.get("bench_op", 0) or 0
    contract_p = data.get("contract_prob", 0) or 0
    roa = data.get("roa", 0) or 0

    def _cmp(val, bench):
        val = float(val or 0)
        bench = float(bench or 0)
        if val > bench: return "良好", "#81c784"
        if val < bench: return "要注目", "#ff6b6b"
        return "同等", "#8899bb"

    eq_label, eq_color = _cmp(user_eq, bench_eq)
    op_label, op_color = _cmp(user_op, bench_op)

    # ニュース HTML
    news_html_parts = []
    for i, n in enumerate(news):
        news_html_parts.append(
            f'<div class="news-item">'
            f'<div class="news-num">{i+1}</div>'
            f'<div><div class="news-title">{_esc(n["title"])}</div>'
            f'<div class="news-body">{_esc(n["body"])}</div></div>'
            f'</div>'
        )
    news_html = "".join(news_html_parts) or '<p class="empty" style="grid-column: span 2;">業界ニュースデータなし（審査後に再実行してください）</p>'

    # モンテカルロ
    if mc:
        mc_html = (
            '<div class="metric-grid">'
            f'<div class="metric-card"><div class="metric-label">加重平均デフォルト確率</div>'
            f'<div class="metric-value risk">{mc.get("weighted_default_prob", 0)*100:.2f}%</div></div>'
            f'<div class="metric-card"><div class="metric-label">集中リスク</div>'
            f'<div class="metric-value">{mc.get("concentration_risk", 0)*100:.2f}%</div></div>'
            f'<div class="metric-card"><div class="metric-label">期待損失率</div>'
            f'<div class="metric-value risk">{mc.get("expected_loss", 0)*100:.2f}%</div></div>'
            f'<div class="metric-card"><div class="metric-label">VaR (95%)</div>'
            f'<div class="metric-value">{mc.get("portfolio_var_95", 0)*100:.2f}%</div></div>'
            '</div>'
        )
    else:
        mc_html = '<p class="empty">モンテカルロシミュレーションデータなし（審査実行後に自動生成）</p>'

    # AIコメント
    ai_section = (
        f'<blockquote class="ai-quote">{ai_cmt}</blockquote>'
        if ai_cmt
        else '<p class="empty">AIコメントデータなし</p>'
    )

    screener_line = f" &nbsp;|&nbsp; 担当：{screener}" if screener else ""
    ind_display   = ind_sub or ind_major or "業種未設定"

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>審査レポート — {company}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0e1a;--bg2:#111827;--card:#141c2e;--border:#1e2a45;
  --gold:#f0b429;--gold-d:#a07020;--text:#e0e6ef;--dim:#8899bb;
  --risk:#ff6b6b;--good:#81c784;
}}
body{{font-family:'Noto Sans JP',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.75}}
/* ─── ヘッダー ─── */
.hdr{{
  background:linear-gradient(135deg,#0a0e1a 0%,#1a2035 60%,#0d1525 100%);
  border-bottom:2px solid var(--gold);
  padding:32px 48px 24px;
  display:flex;justify-content:space-between;align-items:flex-start;
}}
.hdr-company{{font-size:26px;font-weight:700;color:#fff;letter-spacing:1px}}
.hdr-meta{{font-size:12px;color:var(--dim);margin-top:6px}}
.badge{{
  display:inline-block;padding:8px 22px;border-radius:24px;
  font-size:16px;font-weight:700;letter-spacing:3px;
  background:{badge_bg};color:{txt_col};
}}
/* ─── コンテンツ ─── */
.wrap{{max-width:940px;margin:0 auto;padding:36px 24px 60px}}
.sec-title{{
  font-size:11px;font-weight:700;color:var(--gold);
  border-bottom:1px solid var(--gold-d);padding-bottom:6px;
  margin:36px 0 18px;letter-spacing:3px;text-transform:uppercase;
}}
/* ─── スコアゾーン ─── */
.score-zone{{display:flex;align-items:center;gap:36px;flex-wrap:wrap}}
.score-main-info{{display:flex;flex-direction:column;gap:10px}}
.score-big{{font-size:52px;font-weight:700;color:var(--gold);line-height:1}}
.score-big span{{font-size:16px;color:var(--dim);font-weight:400}}
.kpi-row{{display:flex;gap:24px;flex-wrap:wrap;margin-top:6px}}
.kpi{{display:flex;flex-direction:column}}
.kpi-lbl{{font-size:11px;color:var(--dim)}}
.kpi-val{{font-size:18px;font-weight:700}}
.kpi-val.risk{{color:var(--risk)}}
.kpi-val.good{{color:var(--good)}}
.sub-scores{{display:flex;gap:20px;flex-wrap:wrap;align-items:center}}
/* ─── メトリクス ─── */
.metric-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:12px}}
.metric-card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 18px}}
.metric-label{{font-size:11px;color:var(--dim);margin-bottom:6px}}
.metric-value{{font-size:24px;font-weight:700;color:var(--gold)}}
.metric-value.risk{{color:var(--risk)}}
/* ─── 財務テーブル ─── */
.fin-tbl{{width:100%;border-collapse:collapse}}
.fin-tbl th,.fin-tbl td{{padding:9px 14px;border-bottom:1px solid var(--border);text-align:left}}
.fin-tbl th{{color:var(--gold);font-size:11px;background:var(--card);letter-spacing:1px;text-transform:uppercase}}
.fin-tbl td{{font-size:13px}}
.fin-tbl td:last-child{{font-weight:700;color:var(--text)}}
/* ─── ニュース ─── */
.news-item{{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid var(--border)}}
.news-num{{
  min-width:26px;height:26px;border-radius:50%;
  background:var(--gold);color:#000;font-size:12px;font-weight:700;
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
}}
.news-title{{font-weight:700;font-size:13px;margin-bottom:5px;color:var(--text)}}
.news-body{{font-size:12px;color:var(--dim)}}
/* ─── AI 引用 ─── */
.ai-quote{{
  border-left:3px solid var(--gold);padding:14px 18px;
  background:var(--card);border-radius:0 10px 10px 0;
  color:var(--dim);font-size:13px;white-space:pre-wrap;word-break:break-word;
}}
.empty{{color:var(--dim);font-style:italic;font-size:12px;padding:8px 0}}
/* ─── フッター ─── */
.footer{{
  text-align:center;padding:28px;font-size:11px;color:var(--dim);
  border-top:1px solid var(--border);margin-top:56px;
}}
@media print{{
  body{{background:#fff;color:#000}}
  .hdr{{background:#fff;border-bottom:2px solid #000}}
  .badge,.metric-card,.fin-tbl th{{background:#f5f5f5}}
}}
</style>
</head>
<body>

<!-- ── ヘッダー ── -->
<div class="hdr">
  <div>
    <div class="hdr-company">{company}</div>
    <div class="hdr-meta">審査日：{date}{screener_line} &nbsp;|&nbsp; 業種：{ind_display}</div>
  </div>
  <div class="badge">{hantei}</div>
</div>

<div class="wrap">

<!-- ── Bayesian Approval Evidence ── -->
<div class="sec-title">Bayesian Approval Evidence</div>
<div class="metric-grid" style="margin-bottom: 20px;">
  <div class="metric-card" style="grid-column: span 2;">
    <div class="metric-label">ベイズ推定承認確率</div>
    <div class="metric-value good">{bn_prob:.1f}%</div>
    <div style="font-size:11px; color:var(--dim); margin-top:8px;">
      財務スコアに定性的な証拠（エビデンス）を掛け合わせ、最終的な承認可能性を算出しています。
    </div>
  </div>
  {bn_ev_html}
</div>

<!-- ── Financial Indicators & Industry Comparison ── -->
<div class="sec-title">Financial Indicators & Industry Peer Comparison</div>
<table class="fin-tbl">
  <thead>
    <tr><th>指標</th><th>実績値</th><th>業界平均</th><th>判定</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>自己資本比率</td><td>{user_eq:.1f}%</td><td>{bench_eq:.1f}%</td><td style="color:{eq_color}">{eq_label}</td>
    </tr>
    <tr>
      <td>営業利益率</td><td>{user_op:.1f}%</td><td>{bench_op:.1f}%</td><td style="color:{op_color}">{op_label}</td>
    </tr>
    <tr>
        <td>デフォルト確率 (PD)</td><td>{pd_pct:.2f}%</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>収益性指標 (ROA)</td><td>{roa:.1f}%</td><td>-</td><td>-</td>
    </tr>
  </tbody>
</table>

<!-- ── Executive Summary ── -->
<div class="sec-title">Executive Summary</div>
<div class="score-zone">
  {gauge_total}
  <div class="score-main-info">
    <div class="score-big">{score:.1f}<span> / 100</span></div>
    <div class="kpi-row">
      <div class="kpi">
        <span class="kpi-lbl">成約確率</span>
        <span class="kpi-val good">{contract_p:.1f}%</span>
      </div>
      <div class="kpi">
        <span class="kpi-lbl">年商</span>
        <span class="kpi-val">{nenshu:,.0f}千円</span>
      </div>
      <div class="kpi">
        <span class="kpi-lbl">リース期間</span>
        <span class="kpi-val">{lease_term}ヶ月</span>
      </div>
    </div>
  </div>
  {radar_svg}
</div>

<!-- ── AI Executive Comment ── -->
<div class="sec-title">AI Executive Comment</div>
{ai_section}

<!-- ── Industry Trends ── -->
<div class="sec-title">Industry Trends</div>
{news_html}

</div>

<div class="footer">
  本レポートはAI審査システムにより自動生成されました。最終判断は担当者の責任において行ってください。<br/>
  Generated by リース審査AI &nbsp;|&nbsp; {date}
</div>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 4. PDF レポート生成（reportlab）
# ══════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(data: dict) -> bytes:
    """reportlab を使って A4 縦の PDF レポートを生成する。"""
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable,
    )

    # カラーパレット
    BG    = colors.HexColor("#0a0e1a")
    CARD  = colors.HexColor("#141c2e")
    GOLD  = colors.HexColor("#f0b429")
    LIGHT = colors.HexColor("#e0e6ef")
    DIM   = colors.HexColor("#8899bb")
    RISK  = colors.HexColor("#ff6b6b")
    GOOD  = colors.HexColor("#81c784")
    BLUE  = colors.HexColor("#4fc3f7")
    PURP  = colors.HexColor("#ce93d8")
    WHITE = colors.white

    # 日本語フォント設定
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        fn = "HeiseiKakuGo-W5"
        pdfmetrics.registerFont(UnicodeCIDFont(fn))
    except Exception:
        fn = "Helvetica"

    def esc(text: str) -> str:
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    buf = BytesIO()
    margin = 14 * mm
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )

    styles = getSampleStyleSheet()

    def ps(name: str, **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=styles["Normal"], fontName=fn, **kw)

    title_s  = ps("title",  fontSize=20, textColor=LIGHT, leading=24, spaceAfter=2)
    meta_s   = ps("meta",   fontSize=9,  textColor=DIM,   leading=12)
    badge_s  = ps("badge",  fontSize=14, textColor=GOLD,  leading=18, alignment=TA_RIGHT)
    h2_s     = ps("h2",     fontSize=11, textColor=GOLD,  leading=14, spaceBefore=8, spaceAfter=4,
                  borderPadding=(0, 0, 2, 0))
    body_s   = ps("body",   fontSize=9,  textColor=LIGHT, leading=13)
    body_d   = ps("bodyD",  fontSize=9,  textColor=DIM,   leading=13)
    num_s    = ps("num",    fontSize=22, textColor=GOLD,  leading=26, alignment=TA_CENTER)
    num_b_s  = ps("numB",   fontSize=16, textColor=BLUE,  leading=20, alignment=TA_CENTER)
    num_g_s  = ps("numG",   fontSize=16, textColor=GOOD,  leading=20, alignment=TA_CENTER)
    num_p_s  = ps("numP",   fontSize=16, textColor=PURP,  leading=20, alignment=TA_CENTER)
    lbl_s    = ps("lbl",    fontSize=9,  textColor=DIM,   leading=12, alignment=TA_CENTER)
    small_s  = ps("small",  fontSize=8,  textColor=DIM,   leading=11)

    def hline() -> HRFlowable:
        return HRFlowable(width="100%", thickness=0.5, color=GOLD, spaceAfter=5, spaceBefore=3)

    # データ取得
    company    = data["company_name"]
    screener   = data["screener"]
    date       = data["date"]
    hantei     = data["hantei"]
    score      = data["score"]
    asset_v    = data["asset_score"]
    borrow_v   = data["borrower_score"]
    qual_v     = data["qual_score"]
    pd_pct     = data["pd_percent"]
    user_eq    = data["user_eq"]
    user_op    = data["user_op"]
    mc         = data["mc_data"]
    bn_prob    = data["bn_approval_prob"]
    ai_cmt     = data["ai_comment"]
    news       = data["news_items"]
    nenshu     = data["nenshu"]
    net_assets = data["net_assets"]
    acq_cost   = data["acq_cost"]
    lease_term = data["lease_term"]
    ind_sub    = data["industry_sub"] or data["industry_major"]

    story = []

    # ─── ヘッダーテーブル ───────────────────────────────────────────────────
    meta_text = f"審査日：{esc(date)}　業種：{esc(ind_sub)}"
    if screener:
        meta_text += f"　担当：{esc(screener)}"
    hdr_data = [
        [Paragraph(esc(company), title_s), Paragraph(esc(hantei), badge_s)],
        [Paragraph(meta_text, meta_s), ""],
    ]
    hdr_tbl = Table(hdr_data, colWidths=["68%", "32%"])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BG),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("LINEBELOW",     (0, -1), (-1, -1), 1.5, GOLD),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 7 * mm))

    # ─── Executive Summary ──────────────────────────────────────────────────
    story.append(Paragraph("EXECUTIVE SUMMARY", h2_s))
    story.append(hline())

    # スコアグリッド（4列）
    score_data = [
        [Paragraph("総合スコア", lbl_s), Paragraph("物件スコア", lbl_s),
         Paragraph("財務スコア", lbl_s), Paragraph("定性スコア", lbl_s)],
        [Paragraph(f"{score:.1f}", num_s), Paragraph(f"{asset_v:.1f}", num_b_s),
         Paragraph(f"{borrow_v:.1f}", num_g_s), Paragraph(f"{qual_v:.1f}", num_p_s)],
    ]
    sc_tbl = Table(score_data, colWidths=["25%", "25%", "25%", "25%"])
    sc_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, BG),
        ("FONTNAME",      (0, 0), (-1, -1), fn),
    ]))
    story.append(sc_tbl)
    story.append(Spacer(1, 3 * mm))

    # KPI ライン
    kpi_data = [
        ["デフォルト確率", f"{pd_pct:.2f}%",  "BN承認確率",  f"{bn_prob:.1f}%"],
        ["自己資本比率",   f"{user_eq:.1f}%", "営業利益率",  f"{user_op:.1f}%"],
        ["年商",           f"{nenshu:,.0f}千円", "取得費用",  f"{acq_cost:,.0f}千円"],
        ["純資産",         f"{net_assets:,.0f}千円", "リース期間", f"{lease_term}ヶ月"],
    ]
    kpi_tbl = Table(kpi_data, colWidths=["28%", "22%", "28%", "22%"])
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD),
        ("TEXTCOLOR",     (0, 0), (-1, -1), LIGHT),
        ("FONTNAME",      (0, 0), (-1, -1), fn),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",     (1, 0), (1, 0),   RISK),   # デフォルト確率値
        ("TEXTCOLOR",     (3, 0), (3, 0),   GOOD),   # BN確率値
        ("GRID",          (0, 0), (-1, -1), 0.5, BG),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 6 * mm))

    # ─── Bayesian Approval Evidence ──────────────────────────────────────────
    story.append(Paragraph("BAYESIAN APPROVAL EVIDENCE", h2_s))
    story.append(hline())

    bn_ev_data = [
        [Paragraph(f"ベイズ推定承認確率：<font color='#81c784'>{bn_prob:.1f}%</font>", body_s), ""]
    ]
    
    # 選択されたエビデンスをリスト化
    _EV_LABELS = {
        "Insolvent_Status": "債務超過回避",
        "Main_Bank_Support": "メイン銀行支援",
        "Related_Bank_Status": "関係者銀行取引",
        "Related_Assets": "個人資産保有",
        "Co_Lease": "協調リース",
        "Parent_Guarantor": "親会社保証",
        "Core_Business_Use": "本業不可欠物件",
        "Asset_Liquidity": "物件流動性",
        "Shorter_Lease_Term": "期間短縮",
        "One_Time_Deal": "本件限り",
    }
    bn_evidence = data.get("bn_evidence", {}) or {}
    active_ev = [f"・{_EV_LABELS.get(k, k)}：証拠あり" for k, v in bn_evidence.items() if v == 1]
    
    if active_ev:
        for ev_str in active_ev:
            bn_ev_data.append([Paragraph(ev_str, small_s), ""])
    else:
        bn_ev_data.append([Paragraph("※ 定性的なポジティブ・エビデンスの選択なし", body_d), ""])

    bn_ev_tbl = Table(bn_ev_data, colWidths=["100%"])
    bn_ev_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), CARD),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("GRID",          (0, 0), (-1, -1), 0.5, BG),
    ]))
    story.append(bn_ev_tbl)
    story.append(Spacer(1, 6 * mm))

    # ─── Risk Analysis ──────────────────────────────────────────────────────
    story.append(Paragraph("RISK ANALYSIS — Monte Carlo Simulation", h2_s))
    story.append(hline())
    if mc:
        mc_rows = [
            ["指標", "値"],
            ["加重平均デフォルト確率", f"{mc.get('weighted_default_prob', 0) * 100:.2f}%"],
            ["集中リスク",             f"{mc.get('concentration_risk', 0) * 100:.2f}%"],
            ["期待損失率",             f"{mc.get('expected_loss', 0) * 100:.2f}%"],
            ["VaR (95%)",              f"{mc.get('portfolio_var_95', 0) * 100:.2f}%"],
        ]
        mc_tbl = Table(mc_rows, colWidths=["65%", "35%"])
        mc_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  GOLD),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  BG),
            ("FONTNAME",      (0, 0), (-1, 0),  fn),
            ("FONTSIZE",      (0, 0), (-1, 0),  9),
            ("BACKGROUND",    (0, 1), (-1, -1), CARD),
            ("TEXTCOLOR",     (0, 1), (-1, -1), LIGHT),
            ("FONTNAME",      (0, 1), (-1, -1), fn),
            ("FONTSIZE",      (0, 1), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("GRID",          (0, 0), (-1, -1), 0.5, BG),
        ]))
        story.append(mc_tbl)
    else:
        story.append(Paragraph("モンテカルロシミュレーションデータなし（審査実行後に自動生成）", body_d))
    story.append(Spacer(1, 6 * mm))

    # ─── AI Executive Comment ───────────────────────────────────────────────
    story.append(Paragraph("AI EXECUTIVE COMMENT", h2_s))
    story.append(hline())
    if ai_cmt:
        for line in ai_cmt[:800].split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(esc(line), body_s))
    else:
        story.append(Paragraph("AIコメントデータなし", body_d))
    story.append(Spacer(1, 6 * mm))

    # ─── Industry Trends ────────────────────────────────────────────────────
    story.append(Paragraph("INDUSTRY TRENDS", h2_s))
    story.append(hline())
    if news:
        for i, n in enumerate(news):
            story.append(Paragraph(f"{i+1}. {esc(n['title'])}", body_s))
            if n.get("body"):
                story.append(Paragraph(esc(n["body"]), body_d))
            story.append(Spacer(1, 2 * mm))
    else:
        story.append(Paragraph("業界ニュースデータなし", body_d))

    # ─── フッター ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 10 * mm))
    story.append(hline())
    story.append(Paragraph(
        "本レポートはAI審査システムにより自動生成されました。最終判断は担当者の責任において行ってください。",
        small_s,
    ))

    doc.build(story)
    return buf.getvalue()
