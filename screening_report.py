"""
screening_report.py — 審査報告書 PDF 生成モジュール（ダッシュボード版）

日本語フォント: ReportLab 組み込み CID フォント HeiseiKakuGo-W5 を使用。
外部フォントファイル不要。

build_screening_report_pdf(res, submitted_inputs, extra) -> bytes
  res              : st.session_state["last_result"]
  submitted_inputs : 任意
  extra            : {"company_name": str, "screener": str, "note": str}
"""

from __future__ import annotations
from datetime import datetime
from io import BytesIO
from typing import Optional

from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

_JP  = "HeiseiKakuGo-W5"
pdfmetrics.registerFont(UnicodeCIDFont(_JP))

# ── カラーパレット ─────────────────────────────────────────────────
_NAVY   = (0x1a/255, 0x27/255, 0x44/255)
_STEEL  = (0x2c/255, 0x5f/255, 0x8f/255)
_ACCENT = (0x00/255, 0xb0/255, 0x9b/255)
_WARN   = (1.00,     0.55,     0.00)
_DANGER = (0.84,     0.14,     0.16)
_GREEN  = (0.20,     0.70,     0.30)
_LIGHT  = (0.94,     0.96,     0.98)
_WHITE  = (1.0,      1.0,      1.0)
_BLACK  = (0.0,      0.0,      0.0)
_GRAY   = (0.55,     0.55,     0.55)


def _score_rgb(score: float):
    if score >= 80: return _ACCENT
    if score >= 70: return _GREEN
    if score >= 60: return _WARN
    return _DANGER


def _judge_label(score: float) -> str:
    if score >= 80: return "承認推奨"
    if score >= 70: return "承認圏内"
    if score >= 60: return "条件付承認"
    return "否決圏内"


def _auto_comment(
    score: float, judge_lbl: str,
    user_op: float, bench_op: float,
    user_eq: float, bench_eq: float,
    gross_m: float, net_m: float,
    roa: float, debt_r: float,
    pd_pct: float, yield_pred: float,
    ind_score: float, bench_score: float,
) -> list[str]:
    """
    財務数値・スコアから自動でAIコメント文を生成する。
    外部AI呼び出し不要・事前実行なしでも常に有効なコメントを返す。
    """
    lines = []

    # ── 総合評価 ──────────────────────────────────────────────
    if score >= 80:
        lines.append(
            f"■ 総合評価：スコア {score:.1f}点（{judge_lbl}）。財務健全性は高く、"
            "リース契約の遂行能力に問題はないと判断されます。"
            "優先的な審査承認が推奨されます。"
        )
    elif score >= 70:
        lines.append(
            f"■ 総合評価：スコア {score:.1f}点（{judge_lbl}）。財務指標は概ね良好で、"
            "標準的な審査条件のもとで承認圏内と評価します。"
            "下記の個別指標も参考にしながら最終判断を行ってください。"
        )
    elif score >= 60:
        lines.append(
            f"■ 総合評価：スコア {score:.1f}点（{judge_lbl}）。一部財務指標に懸念があり、"
            "保証条件・担保設定・契約期間の短縮など、リスク軽減措置を検討のうえ"
            "条件付きで承認を検討してください。"
        )
    else:
        lines.append(
            f"■ 総合評価：スコア {score:.1f}点（{judge_lbl}）。財務上のリスクが高く、"
            "現状では否決が妥当な水準です。追加書類の取得や抜本的な与信条件の見直しが"
            "必要です。"
        )

    # ── 収益性コメント ─────────────────────────────────────────
    if user_op > 0 and bench_op > 0:
        diff_op = user_op - bench_op
        if diff_op >= 3:
            lines.append(
                f"■ 収益性：営業利益率 {user_op:.1f}%は業界平均 {bench_op:.1f}%を"
                f"{diff_op:.1f}pt 上回っており、コスト管理・販売力ともに優秀です。"
            )
        elif diff_op >= 0:
            lines.append(
                f"■ 収益性：営業利益率 {user_op:.1f}%は業界平均 {bench_op:.1f}%と"
                "同水準で安定した収益基盤が確認できます。"
            )
        else:
            lines.append(
                f"■ 収益性：営業利益率 {user_op:.1f}%は業界平均 {bench_op:.1f}%を"
                f"{abs(diff_op):.1f}pt 下回っています。"
                "収益改善策の有無・今後の見通しを担当者が確認することを推奨します。"
            )
    elif user_op > 0:
        lvl = "良好" if user_op >= 5 else ("標準的" if user_op >= 2 else "低水準")
        lines.append(f"■ 収益性：営業利益率 {user_op:.1f}%は{lvl}です。")
    else:
        lines.append(
            "■ 収益性：営業損失が発生しています。"
            "損失原因・改善計画・資金繰りへの影響を詳細に確認してください。"
        )

    # ── 財務安全性コメント ─────────────────────────────────────
    if bench_eq > 0:
        diff_eq = user_eq - bench_eq
        if diff_eq >= 5:
            lines.append(
                f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は業界平均 {bench_eq:.1f}%を"
                f"{diff_eq:.1f}pt 上回り、財務基盤は強固です。"
            )
        elif diff_eq >= -3:
            lines.append(
                f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は業界平均 {bench_eq:.1f}%と"
                "同水準で、財務健全性は標準的です。"
            )
        else:
            lines.append(
                f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は業界平均 {bench_eq:.1f}%を"
                f"{abs(diff_eq):.1f}pt 下回っています。"
                "債務超過リスク・借入依存度を精査し、保証条件の強化を検討してください。"
            )
    else:
        if user_eq >= 30:
            lines.append(f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は良好な水準です。")
        elif user_eq >= 10:
            lines.append(
                f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は中程度です。"
                "負債比率 {debt_r:.1f}%と合わせて総合的に判断してください。"
            )
        else:
            lines.append(
                f"■ 財務安全性：自己資本比率 {user_eq:.1f}%は低水準です。"
                "財務基盤の脆弱性について追加確認を行ってください。"
            )

    # ── ROA・資産効率コメント ──────────────────────────────────
    if roa >= 5:
        lines.append(
            f"■ 資産効率：ROA {roa:.1f}%は高水準で、保有資産を効率的に収益化できています。"
        )
    elif roa >= 1:
        lines.append(
            f"■ 資産効率：ROA {roa:.1f}%は一定水準を維持しています。"
        )
    elif roa >= 0:
        lines.append(
            f"■ 資産効率：ROA {roa:.1f}%はほぼゼロです。"
            "資産圧縮・遊休資産の整理等、効率化施策の有無を確認してください。"
        )
    else:
        lines.append(
            f"■ 資産効率：ROAがマイナスです。資産運用の改善が急務です。"
        )

    # ── 信用リスクコメント ─────────────────────────────────────
    if pd_pct < 2:
        lines.append(
            f"■ 信用リスク：PD（デフォルト確率）{pd_pct:.1f}%は低水準で、"
            "短期的な債務不履行リスクは限定的と評価されます。"
        )
    elif pd_pct < 5:
        lines.append(
            f"■ 信用リスク：PD（デフォルト確率）{pd_pct:.1f}%は中程度です。"
            "与信モニタリングを強化し、延滞兆候の早期検知に努めてください。"
        )
    else:
        lines.append(
            f"■ 信用リスク：PD（デフォルト確率）{pd_pct:.1f}%は高水準です。"
            "担保・連帯保証の充実、契約期間短縮またはリース料前払い等の"
            "リスク軽減措置を強く推奨します。"
        )

    # ── 金利・採算コメント ─────────────────────────────────────
    if yield_pred > 0:
        if yield_pred >= 3.5:
            lines.append(
                f"■ 採算性：予測リース金利 {yield_pred:.2f}%は標準的な採算ライン上にあります。"
            )
        elif yield_pred >= 2.0:
            lines.append(
                f"■ 採算性：予測リース金利 {yield_pred:.2f}%は適正範囲内です。"
            )
        else:
            lines.append(
                f"■ 採算性：予測リース金利 {yield_pred:.2f}%は低めです。"
                "資金調達コストとの兼ね合いで採算性を確認してください。"
            )

    return lines


def _arrow(val: float, bench: float) -> str:
    if val > bench: return "▲ 良"
    if val < bench: return "▼ 注意"
    return "－ 同等"


def _C(*rgb):
    from reportlab.lib import colors
    return colors.Color(*rgb)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# グラフ生成ヘルパー（Drawing = ReportLab Flowable として使用可）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_score_donut(score: float, sc_rgb: tuple):
    """スコアドーナツチャート（Drawing を返す）"""
    from reportlab.graphics.shapes import Drawing, String, Circle
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    w, h = 60 * mm, 56 * mm
    d = Drawing(w, h)
    cx = w / 2
    cy = h / 2 + 2 * mm
    r  = 20 * mm

    pie = Pie()
    pie.x      = cx - r
    pie.y      = cy - r
    pie.width  = r * 2
    pie.height = r * 2
    pie.data   = [max(score, 0.5), max(100.0 - score, 0.5)]
    pie.labels = ['', '']  # ラベル非表示
    pie.slices[0].fillColor   = _C(*sc_rgb)
    pie.slices[1].fillColor   = colors.HexColor('#e2e8f0')
    pie.slices[0].strokeColor = colors.white
    pie.slices[1].strokeColor = colors.white
    pie.slices[0].strokeWidth = 2
    pie.slices[1].strokeWidth = 2
    pie.startAngle = 90
    d.add(pie)

    # 中心を白い円で塗りつぶしてドーナツ穴を作る
    d.add(Circle(cx, cy, r * 0.58, fillColor=colors.white, strokeColor=None))

    # スコア数値
    d.add(String(cx, cy + 1 * mm,
                 f"{score:.1f}",
                 fontName=_JP, fontSize=18,
                 fillColor=_C(*sc_rgb), textAnchor='middle'))
    d.add(String(cx, cy - 7 * mm,
                 "\u70b9",   # 点
                 fontName=_JP, fontSize=9,
                 fillColor=_C(*_GRAY), textAnchor='middle'))
    return d


def _make_model_bars(scores: list, labels: list, bar_colors: list):
    """3モデルスコア横棒グラフ（Drawing を返す）"""
    from reportlab.graphics.shapes import Drawing, String, Rect
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    w, h    = 88 * mm, 44 * mm
    d       = Drawing(w, h)
    label_w = 23 * mm
    bar_area = w - label_w - 14 * mm
    n        = len(scores)
    bh       = 9 * mm
    gap      = 2.5 * mm
    total_h  = n * bh + (n - 1) * gap
    sy       = (h - total_h) / 2

    for i, (sv, lbl, bc) in enumerate(zip(scores, labels, bar_colors)):
        y  = sy + (n - 1 - i) * (bh + gap)
        # 背景バー
        d.add(Rect(label_w, y, bar_area, bh,
                   fillColor=colors.HexColor('#e2e8f0'), strokeColor=None))
        # スコアバー
        fw = bar_area * min(max(sv, 0), 100) / 100
        d.add(Rect(label_w, y, fw, bh, fillColor=_C(*bc), strokeColor=None))
        # ラベル
        d.add(String(label_w - 1.5 * mm, y + bh / 2 - 2.5,
                     lbl, fontName=_JP, fontSize=7,
                     fillColor=_C(*_BLACK), textAnchor='end'))
        # スコア値（バー内 or バー外）
        if fw > 14 * mm:
            d.add(String(label_w + fw - 1.5 * mm, y + bh / 2 - 2.5,
                         f"{sv:.1f}\u70b9",
                         fontName=_JP, fontSize=7,
                         fillColor=colors.white, textAnchor='end'))
        else:
            d.add(String(label_w + fw + 1.5 * mm, y + bh / 2 - 2.5,
                         f"{sv:.1f}\u70b9",
                         fontName=_JP, fontSize=7,
                         fillColor=_C(*bc), textAnchor='start'))
    return d


def _make_fin_metrics_chart(items: list, w_total: float, h_total: float):
    """
    財務指標縦棒グラフ（実績 vs 業界）
    items: [(ラベル, 実績%, 業界%orNone, color_rgb), ...]
    """
    from reportlab.graphics.shapes import Drawing, String, Rect, Line
    from reportlab.lib.units import mm
    from reportlab.lib import colors as rl_colors

    d        = Drawing(w_total, h_total)
    n        = len(items)
    if n == 0:
        return d

    col_w    = w_total / n
    label_h  = 8 * mm
    chart_h  = h_total - label_h - 8 * mm   # 上部に値テキスト用余白
    base_y   = label_h
    clamp    = 80.0
    bar_w    = col_w * 0.30

    # ベースライン
    d.add(Line(2 * mm, base_y, w_total - 2 * mm, base_y,
               strokeColor=rl_colors.HexColor('#b0bcc8'), strokeWidth=0.6))

    # 目盛り線
    for pct in [20, 40, 60]:
        gy = base_y + chart_h * pct / clamp
        if gy <= base_y + chart_h:
            d.add(Line(2 * mm, gy, w_total - 2 * mm, gy,
                       strokeColor=rl_colors.HexColor('#dde3ec'), strokeWidth=0.3))
            d.add(String(1.5 * mm, gy - 1.5,
                         str(pct),
                         fontName=_JP, fontSize=4.5,
                         fillColor=rl_colors.HexColor('#aaaaaa'), textAnchor='end'))

    for i, (lbl, uv, bv, c) in enumerate(items):
        cx = col_w * i + col_w / 2

        # 業界棒（薄グレー・後ろ）
        if bv is not None:
            bh2 = chart_h * min(abs(bv), clamp) / clamp
            bx  = cx - bar_w * 0.9
            d.add(Rect(bx, base_y, bar_w * 1.3, bh2,
                       fillColor=rl_colors.HexColor('#c0ccd8'), strokeColor=None))

        # 実績棒（カラー・前）
        uh        = chart_h * min(abs(uv), clamp) / clamp
        ux        = cx - bar_w / 2
        bar_color = _C(*c) if uv >= 0 else _C(*_DANGER)
        d.add(Rect(ux, base_y, bar_w, uh, fillColor=bar_color, strokeColor=None))

        # 実績値テキスト（棒の上）
        top_u  = base_y + uh
        top_bv = base_y + (chart_h * min(abs(bv), clamp) / clamp if bv else uh)
        text_y = max(top_u, top_bv) + 1.5 * mm
        d.add(String(cx, text_y,
                     f"{uv:.1f}%",
                     fontName=_JP, fontSize=6,
                     fillColor=bar_color, textAnchor='middle'))
        if bv is not None:
            d.add(String(cx, text_y + 4.5 * mm,
                         f"({bv:.1f}%)",
                         fontName=_JP, fontSize=5,
                         fillColor=rl_colors.HexColor('#888888'), textAnchor='middle'))

        # 軸ラベル
        d.add(String(cx, 1.5 * mm, lbl,
                     fontName=_JP, fontSize=6.5,
                     fillColor=_C(*_BLACK), textAnchor='middle'))

    # 凡例
    lx = w_total - 30 * mm
    ly = h_total - 4 * mm
    d.add(Rect(lx, ly, 3 * mm, 2.5 * mm,
               fillColor=_C(*_ACCENT), strokeColor=None))
    d.add(String(lx + 3.5 * mm, ly,
                 "\u5b9f\u7e3e",   # 実績
                 fontName=_JP, fontSize=5.5, fillColor=_C(*_BLACK), textAnchor='start'))
    d.add(Rect(lx + 12 * mm, ly, 3 * mm, 2.5 * mm,
               fillColor=rl_colors.HexColor('#c0ccd8'), strokeColor=None))
    d.add(String(lx + 15.5 * mm, ly,
                 "\u696d\u754c",   # 業界
                 fontName=_JP, fontSize=5.5, fillColor=_C(*_BLACK), textAnchor='start'))
    return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# メイン関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_screening_report_pdf(
    res: dict,
    submitted_inputs: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> bytes:
    """審査結果を A4 PDF に変換してバイト列で返す。"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable,
    )

    # ── スタイルファクトリ ────────────────────────────────────
    def ps(name, size, color=_BLACK, align="LEFT", leading=None, sb=0, sa=0):
        return ParagraphStyle(
            name, fontName=_JP, fontSize=size,
            leading=leading or size * 1.55,
            textColor=_C(*color),
            alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}.get(align, 0),
            spaceBefore=sb, spaceAfter=sa,
        )

    S_BODY  = ps("body",  8, _BLACK, leading=12)
    S_SMALL = ps("small", 7, _GRAY,  leading=10)
    S_H2    = ps("h2",   10, _STEEL, sb=3*mm, sa=1*mm)

    # ── テーブルスタイル生成（ヘッダー白・データ黒）──────────
    def make_tbl_style(header_bg=None, row_data=None):
        hbg  = header_bg or _C(*_NAVY)
        base = [
            ("FONTNAME",     (0, 0), (-1, -1), _JP),
            ("FONTSIZE",     (0, 0), (-1, -1), 7),
            ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),   # ヘッダー: 白
            ("TEXTCOLOR",    (0, 1), (-1, -1), colors.black),   # データ行: 黒
            ("BACKGROUND",   (0, 0), (-1,  0), hbg),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("ALIGN",        (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN",        (0, 0), ( 0, -1), "LEFT"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 1.5 * mm),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 1.5 * mm),
            ("LEFTPADDING",  (0, 0), (-1, -1), 2 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ]
        if row_data:
            for i in range(2, len(row_data), 2):
                base.append(("BACKGROUND", (0, i), (-1, i), _C(*_LIGHT)))
        return base

    # ── データ取得 ────────────────────────────────────────────
    extra        = extra or {}
    score        = float(res.get("score",        0))
    ind_score    = float(res.get("ind_score",    0))
    bench_score  = float(res.get("bench_score",  0))
    user_op      = float(res.get("user_op",      0))
    bench_op     = float(res.get("bench_op",     0))
    user_eq      = float(res.get("user_eq",      0))
    bench_eq     = float(res.get("bench_eq",     0))
    pd_pct       = float(res.get("pd_percent",   0))
    yield_pred   = float(res.get("yield_pred",   0))
    contract_p   = float(res.get("contract_prob",0))
    industry_sub = res.get("industry_sub", "")
    asset_name   = res.get("asset_name",   "") or ""
    comparison   = res.get("comparison",   "") or ""
    ai_factors   = res.get("ai_completed_factors", []) or []
    hints        = res.get("hints",               {}) or {}
    net_risk     = res.get("network_risk_summary", "") or ""
    fin          = res.get("financials",           {}) or {}
    nenshu       = float(fin.get("nenshu",       0) or 0)
    rieki        = float(fin.get("rieki",        0) or 0)
    gross        = float(fin.get("gross_profit", 0) or 0)
    ord_profit   = float(fin.get("ord_profit",   0) or 0)
    net_income   = float(fin.get("net_income",   0) or 0)
    assets       = float(fin.get("assets",       0) or 0)
    net_assets   = float(fin.get("net_assets",   0) or 0)
    bank_credit  = float(fin.get("bank_credit",  0) or 0)
    lease_credit = float(fin.get("lease_credit", 0) or 0)

    company_name = extra.get("company_name", "（社名未入力）")
    screener     = extra.get("screener", "")
    note         = extra.get("note", "")
    report_date  = datetime.now().strftime("%Y年%m月%d日  %H:%M")
    sc_rgb       = _score_rgb(score)
    judge_lbl    = _judge_label(score)

    def fm(v):
        if not v: return "—"
        v   = float(v)
        man = v / 10
        if man >= 10000: return f"{man/10000:.2f}億円"
        return f"{man:,.0f}万円"

    def fp(v): return f"{v:.1f}%" if v is not None else "—"

    def _md2rl(text: str) -> str:
        import re
        return re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text or "")

    gross_m = (gross       / nenshu * 100) if nenshu else 0
    ord_m   = (ord_profit  / nenshu * 100) if nenshu else 0
    net_m   = (net_income  / nenshu * 100) if nenshu else 0
    debt_r  = ((assets - net_assets) / assets * 100) if assets else 0
    roa     = (net_income  / assets  * 100) if assets else 0

    # ── ヘッダー描画関数 ──────────────────────────────────────
    def draw_header(canvas, doc):
        W, H = A4
        canvas.setFillColorRGB(*_NAVY)
        canvas.rect(0, H - 36*mm, W, 36*mm, fill=1, stroke=0)
        canvas.setFillColorRGB(*_ACCENT)
        canvas.rect(0, H - 38*mm, W, 2.5*mm, fill=1, stroke=0)
        canvas.setFillColorRGB(*_WHITE)
        canvas.setFont(_JP, 16)
        canvas.drawString(14*mm, H - 13*mm, "リース審査報告書")
        canvas.setFont(_JP, 9)
        canvas.drawString(14*mm, H - 20*mm, f"【 {company_name} 】")
        canvas.setFont(_JP, 7.5)
        canvas.setFillColorRGB(0.78, 0.85, 0.92)
        meta_y = H - 27*mm
        if industry_sub:
            canvas.drawString(14*mm, meta_y, f"業種：{industry_sub}")
        if asset_name:
            canvas.drawString(100*mm, meta_y, f"物件：{asset_name}")
        canvas.drawString(14*mm, H - 32*mm, f"作成日：{report_date}")
        if screener:
            canvas.drawString(100*mm, H - 32*mm, f"担当：{screener}")
        canvas.setFont(_JP, 6.5)
        canvas.setFillColorRGB(*_GRAY)
        canvas.drawString(14*mm, 9*mm,
            "本レポートは審査支援システムによる自動生成資料です。最終判断は担当者・審査委員会の責任において行ってください。")
        canvas.drawRightString(A4[0] - 14*mm, 9*mm, f"- {doc.page} -")

    # ── ドキュメント ──────────────────────────────────────────
    buffer = BytesIO()
    ML     = 14 * mm
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=ML, rightMargin=ML,
        topMargin=42*mm, bottomMargin=16*mm,
        onFirstPage=draw_header, onLaterPages=draw_header,
    )
    story = []

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ① スコアダッシュボード
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 左列: ドーナツチャート + 判定ラベル
    donut_chart = _make_score_donut(score, sc_rgb)
    judge_para  = Paragraph(
        f"<b>{judge_lbl}</b>",
        ps("jl", 10, sc_rgb, "CENTER"),
    )

    # 右列上: KPIバッジ（PD / 予測金利 / 成約確率）
    badge_tbl = Table(
        [[Paragraph("PD（デフォルト率）", S_SMALL),
          Paragraph("予測金利",           S_SMALL),
          Paragraph("成約確率",           S_SMALL)],
         [Paragraph(f"{pd_pct:.1f}%",     ps("b1", 12, _BLACK, "CENTER")),
          Paragraph(f"{yield_pred:.2f}%", ps("b2", 12, _BLACK, "CENTER")),
          Paragraph(f"{contract_p:.0f}%", ps("b3", 12, _BLACK, "CENTER"))]],
        colWidths=[37*mm, 37*mm, 36*mm],
    )
    badge_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, -1), _JP),
        ("FONTSIZE",     (0, 0), (-1, -1), 7),
        ("BACKGROUND",   (0, 0), (-1,  0), _C(*_LIGHT)),
        ("TEXTCOLOR",    (0, 0), (-1,  0), _C(*_GRAY)),
        ("TEXTCOLOR",    (0, 1), (-1, -1), colors.black),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 2 * mm),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2 * mm),
    ]))

    # 右列下: スコア横棒グラフ
    model_chart = _make_model_bars(
        [score, ind_score, bench_score],
        ["① 総合", "② 業種", "③ 指標比較"],
        [sc_rgb, _STEEL, _GRAY],
    )

    left_col  = [donut_chart, judge_para]
    right_col = [
        Paragraph("■ スコア内訳", S_H2),
        badge_tbl,
        Spacer(1, 3 * mm),
        model_chart,
    ]
    top_layout = Table([[left_col, right_col]], colWidths=[63*mm, 114*mm])
    top_layout.setStyle(TableStyle([
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("ALIGN",        (0, 0), ( 0, -1), "CENTER"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING",   (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 0),
    ]))
    story.append(top_layout)
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=0.8, color=_C(*_ACCENT)))
    story.append(Spacer(1, 2 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ② 財務指標グラフ（ダッシュボード）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(Paragraph("■ 財務指標グラフ", S_H2))

    chart_items = [
        ("粗利率",    gross_m, None,      _STEEL),
        ("営業利益率", user_op, bench_op, _ACCENT),
        ("純利益率",   net_m,  None,      _GREEN),
        ("自己資本比", user_eq, bench_eq, _WARN),
        ("ROA",        roa,    None,      _NAVY),
    ]
    fin_chart = _make_fin_metrics_chart(chart_items, 177 * mm, 54 * mm)
    story.append(fin_chart)
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=0.3, color=_C(*_LIGHT)))
    story.append(Spacer(1, 1 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ③ AI自動分析コメント（事前実行不要・常に生成）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ai_lines = _auto_comment(
        score=score, judge_lbl=judge_lbl,
        user_op=user_op, bench_op=bench_op,
        user_eq=user_eq, bench_eq=bench_eq,
        gross_m=gross_m, net_m=net_m,
        roa=roa, debt_r=debt_r,
        pd_pct=pd_pct, yield_pred=yield_pred,
        ind_score=ind_score, bench_score=bench_score,
    )

    # コメントをテーブルカードで表示
    comment_rows = []
    for line in ai_lines:
        # "■ タイトル：" と本文を分離してスタイルを変える
        if "：" in line and line.startswith("■"):
            parts = line.split("：", 1)
            tag_text  = parts[0].lstrip("■ ").strip()  # タイトル部
            body_text = parts[1].strip() if len(parts) > 1 else ""
            # タイトルの色はスコアに応じたものか STEEL
            if "信用リスク" in tag_text and pd_pct >= 5:
                tag_color = _DANGER
            elif "収益性" in tag_text and user_op < 0:
                tag_color = _DANGER
            elif "総合評価" in tag_text:
                tag_color = sc_rgb
            else:
                tag_color = _STEEL
            comment_rows.append([
                Paragraph(f"<b>{tag_text}</b>", ps("ctag", 7, tag_color, "CENTER")),
                Paragraph(_md2rl(body_text), ps("cbody", 7.5, _BLACK, leading=11)),
            ])
        else:
            comment_rows.append([
                Paragraph("", ps("ctag2", 7, _STEEL)),
                Paragraph(_md2rl(line), ps("cbody2", 7.5, _BLACK, leading=11)),
            ])

    if comment_rows:
        story.append(Paragraph("■ AI自動分析コメント", S_H2))
        ai_card = Table(comment_rows, colWidths=[26*mm, 151*mm])
        ai_card.setStyle(TableStyle([
            ("FONTNAME",     (0, 0), (-1, -1), _JP),
            ("FONTSIZE",     (0, 0), (-1, -1), 7),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("ALIGN",        (0, 0), ( 0, -1), "CENTER"),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("TOPPADDING",   (0, 0), (-1, -1), 2 * mm),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 2 * mm),
            ("LEFTPADDING",  (0, 0), (-1, -1), 2 * mm),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ("ROWBACKGROUNDS",(0, 0),(-1, -1),
             [colors.white, _C(*_LIGHT)] * (len(comment_rows) // 2 + 1)),
        ]))
        story.append(ai_card)
        story.append(Spacer(1, 4 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ④ 財務サマリー（改ページ）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    story.append(Paragraph("■ 財務サマリー", S_H2))

    fin_data = [
        ["項目",               "実績値",         "業界目安",   "評価"],
        ["売上高",             fm(nenshu),         "—",         "—"],
        ["売上総利益",         fm(gross),           "—",         "—"],
        ["営業利益",           fm(rieki),           "—",         "—"],
        ["経常利益",           fm(ord_profit),      "—",         "—"],
        ["当期純利益",         fm(net_income),      "—",         "—"],
        ["総資産",             fm(assets),          "—",         "—"],
        ["純資産",             fm(net_assets),      "—",         "—"],
        ["銀行与信（残高）",   fm(bank_credit),     "※残高",     "—"],
        ["リース与信（残高）", fm(lease_credit),    "※残高",     "—"],
        ["売上総利益率",       fp(gross_m),          "—",        _arrow(gross_m, 0)],
        ["営業利益率",         fp(user_op),          fp(bench_op),_arrow(user_op, bench_op)],
        ["自己資本比率",       fp(user_eq),          fp(bench_eq),_arrow(user_eq, bench_eq)],
        ["負債比率",           fp(debt_r),           "—",        _arrow(-debt_r, 0)],
        ["ROA",                fp(roa),              "—",        _arrow(roa, 0)],
    ]
    fin_tbl = Table(fin_data, colWidths=[40*mm, 35*mm, 30*mm, 22*mm])
    fin_tbl.setStyle(TableStyle(make_tbl_style(_C(*_NAVY), fin_data)))
    story.append(fin_tbl)
    story.append(Spacer(1, 4 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ④ AI補完要因
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ai_factors:
        story.append(Paragraph("■ AI補完要因（スコア調整）", S_H2))
        fac_data = [["要因", "効果", "評価根拠"]]
        for f in ai_factors[:6]:
            fac_data.append([
                str(f.get("factor", "")),
                f"{f.get('effect_percent', 0):+.0f}%",
                str(f.get("reason", "") or "")[:55],
            ])
        fac_tbl = Table(fac_data, colWidths=[45*mm, 16*mm, 66*mm])
        fac_tbl.setStyle(TableStyle(make_tbl_style(_C(*_STEEL), fac_data)))
        story.append(fac_tbl)
        story.append(Spacer(1, 4 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⑤ 業界比較コメント
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(Paragraph("■ 業界比較コメント", S_H2))
    if comparison.strip():
        for line in comparison.split("\n")[:10]:
            line = line.strip()
            if not line:
                continue
            story.append(Paragraph(_md2rl(line), S_BODY))
    else:
        story.append(Paragraph("（業界比較データなし）", S_SMALL))
    story.append(Spacer(1, 3 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⑥ 審査ポイント（業種別ヒント）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    risks     = hints.get("risks",     []) or []
    subsidies = hints.get("subsidies", []) or []
    story.append(Paragraph("■ 審査ポイント（業種別ヒント）", S_H2))
    if risks or subsidies:
        hint_data = [["区分", "内容"]]
        for r in risks[:4]:
            hint_data.append(["リスク", str(r)[:85]])
        for s in subsidies[:3]:
            nm = s.get("name", str(s)) if isinstance(s, dict) else str(s)
            hint_data.append(["補助金", nm[:85]])
        ht = Table(hint_data, colWidths=[20*mm, 107*mm])
        ht.setStyle(TableStyle(make_tbl_style(_C(*_ACCENT), hint_data)))
        story.append(ht)
    else:
        story.append(Paragraph(
            f"業種「{industry_sub}」の個別ヒントデータなし。審査マニュアル標準基準を適用してください。",
            S_SMALL))
    story.append(Spacer(1, 3 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⑦ 業界リスク情報
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(Paragraph("■ 業界リスク情報", S_H2))
    _risk_ok = (net_risk
                and "取得できません" not in net_risk
                and "検索エラー"     not in net_risk
                and "エラー"         not in net_risk)
    if _risk_ok:
        story.append(Paragraph(
            net_risk[:350] + ("…" if len(net_risk) > 350 else ""), S_SMALL))
    else:
        story.append(Paragraph(
            "（業界リスク情報の取得なし。Web検索機能を使用して事前に確認してください。）",
            S_SMALL))
    story.append(Spacer(1, 3 * mm))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⑧ 担当者メモ
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if note.strip():
        story.append(HRFlowable(width="100%", thickness=0.5, color=_C(*_LIGHT)))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph("■ 担当者メモ", S_H2))
        story.append(Paragraph(note, S_BODY))

    doc.build(story)
    return buffer.getvalue()
