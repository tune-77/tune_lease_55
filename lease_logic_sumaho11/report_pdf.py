"""
report_pdf.py — 成約の正体レポートPDF生成モジュール。

含む機能:
- build_contract_report_pdf : 分析結果を A4 1枚の PDF バイト列として返す
"""

import os
from config import BASE_DIR


def build_contract_report_pdf(analysis: dict) -> bytes:
    """
    成約の正体レポートの分析結果をPDFバイト列で返す。A4 1枚に収まるようレイアウト。
    日本語表示のためリポジトリルートの IPAexGothic.ttf を使用（無ければ Helvetica で代替）。
    """
    from io import BytesIO
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = BytesIO()
    margin = 12 * mm
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
    )
    styles = getSampleStyleSheet()

    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        font_path = os.path.join(BASE_DIR, "IPAexGothic.ttf")
        if os.path.isfile(font_path):
            pdfmetrics.registerFont(TTFont("JP", font_path))
            font_name = "JP"
        else:
            font_name = "Helvetica"
    except Exception:
        font_name = "Helvetica"

    def safe_text(text):
        return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    body_style  = ParagraphStyle("BodyJP",      parent=styles["Normal"],   fontName=font_name, fontSize=8,  leading=10)
    title_style = ParagraphStyle("CustomTitle", parent=styles["Heading1"], fontName=font_name, fontSize=14, leading=16)
    h2_style    = ParagraphStyle("CustomH2",    parent=styles["Heading2"], fontName=font_name, fontSize=10, leading=12)
    thin = 1.5 * mm

    story = []
    story.append(Paragraph(safe_text("成約の正体レポート"), title_style))
    n = analysis["closed_count"]
    story.append(Paragraph(safe_text(f"成約 {n} 件を分析しました。"), body_style))
    story.append(Spacer(1, thin))

    story.append(Paragraph(safe_text("【成約要因】上位3因子"), h2_style))
    for i, d in enumerate(analysis["top3_drivers"], 1):
        story.append(Paragraph(safe_text(f"{i}. {d['label']} 係数{d['coef']:.4f}（{d['direction']}）"), body_style))
    story.append(Spacer(1, thin))

    story.append(Paragraph(safe_text("【成約案件の平均財務】"), h2_style))
    if analysis["avg_financials"]:
        rows = [["指標", "平均値"]]
        for k, v in analysis["avg_financials"].items():
            if "自己資本" in k:
                rows.append([k, f"{v:.1f}%"])
            elif isinstance(v, float) and abs(v) >= 1:
                rows.append([k, f"{v:,.0f}"])
            else:
                rows.append([k, f"{v:.4f}"])
        t = Table(rows, colWidths=[75 * mm, 50 * mm])
        t.setStyle(TableStyle([
            ("FONTNAME",     (0, 0), (-1, -1), font_name),
            ("FONTSIZE",     (0, 0), (-1, -1), 7),
            ("LEFTPADDING",  (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING",   (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
            ("BACKGROUND",   (0, 0), (-1, 0), colors.lightgrey),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(t)
    else:
        story.append(Paragraph(safe_text("財務データなし"), body_style))
    story.append(Spacer(1, thin))

    story.append(Paragraph(safe_text("【定性タグ ランキング】"), h2_style))
    if analysis["tag_ranking"]:
        parts = [f"{rank}.{tag}({count})" for rank, (tag, count) in enumerate(analysis["tag_ranking"][:10], 1)]
        story.append(Paragraph(safe_text(" ".join(parts)), body_style))
    else:
        story.append(Paragraph(safe_text("定性タグなし"), body_style))
    story.append(Spacer(1, thin))

    story.append(Paragraph(safe_text("【定性スコアリング】"), h2_style))
    qs = analysis.get("qualitative_summary")
    if qs and (qs.get("avg_weighted") is not None or qs.get("avg_combined") is not None or qs.get("rank_distribution")):
        n_qual = qs.get("n_with_qual", 0)
        line = f"入力{n_qual}件"
        if qs.get("avg_weighted") is not None:
            line += f" 加重平均{qs['avg_weighted']:.1f}/100"
        if qs.get("avg_combined") is not None:
            line += f" 合計平均{qs['avg_combined']:.1f}"
        if qs.get("rank_distribution"):
            dist = " ".join(f"{r}:{c}件" for r, c in sorted(qs["rank_distribution"].items(), key=lambda x: (-x[1], x[0])))
            line += f" ランク分布 {dist}"
        story.append(Paragraph(safe_text(line), body_style))
    else:
        story.append(Paragraph(safe_text("定性スコア入力案件なし"), body_style))

    doc.build(story)
    return buffer.getvalue()
