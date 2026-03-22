"""
審査レポートのPDFエクスポート（reportlab使用）
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io, os, datetime

# 日本語フォント登録（システムフォントを使用）
def _register_font():
    candidates = [
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("JpFont", path))
                return "JpFont"
            except Exception:
                continue
    return "Helvetica"


def generate_report_pdf(res: dict) -> bytes:
    """
    審査結果 dict から PDF バイト列を生成して返す。
    """
    font_name = _register_font()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle("Title",  fontName=font_name, fontSize=18, leading=24, textColor=colors.HexColor("#1565c0"), spaceAfter=6)
    h2_style     = ParagraphStyle("H2",     fontName=font_name, fontSize=13, leading=18, textColor=colors.HexColor("#333333"), spaceBefore=12, spaceAfter=4)
    body_style   = ParagraphStyle("Body",   fontName=font_name, fontSize=10, leading=15, textColor=colors.HexColor("#444444"))
    small_style  = ParagraphStyle("Small",  fontName=font_name, fontSize=8,  leading=12, textColor=colors.HexColor("#888888"))

    score    = res.get("score") or 0
    hantei   = res.get("hantei") or "—"
    industry = res.get("industry_sub") or res.get("industry_major") or "—"
    asset    = res.get("asset_name") or "—"
    ts       = datetime.datetime.now().strftime("%Y年%m月%d日 %H:%M")

    story = []

    # タイトル
    story.append(Paragraph("リース審査レポート", title_style))
    story.append(Paragraph(f"作成日時: {ts}", small_style))
    story.append(Spacer(1, 6*mm))

    # スコア・判定
    story.append(Paragraph("■ 審査結果", h2_style))
    result_data = [
        ["業種", industry],
        ["物件", asset],
        ["総合スコア", f"{score:.0f} 点 / 100"],
        ["判定", hantei],
    ]
    t = Table(result_data, colWidths=[40*mm, 130*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), font_name),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("BACKGROUND",  (0,0), (0,-1), colors.HexColor("#e3f2fd")),
        ("TEXTCOLOR",   (0,0), (0,-1), colors.HexColor("#1565c0")),
        ("FONTNAME",    (0,0), (0,-1), font_name),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 6*mm))

    # 財務指標
    story.append(Paragraph("■ 財務指標", h2_style))
    fin_data = [
        ["指標", "企業値", "業界平均"],
        ["営業利益率",   f"{res.get('user_op') or 0:.1f}%",  f"{res.get('bench_op') or 0:.1f}%"],
        ["自己資本比率", f"{res.get('user_eq') or 0:.1f}%",  f"{res.get('bench_eq') or 0:.1f}%"],
        ["ROA",          f"{res.get('user_roa') or 0:.1f}%", f"{res.get('bench_roa') or 0:.1f}%"],
        ["流動比率",     f"{res.get('user_current_ratio') or 0:.0f}%", f"{res.get('bench_current_ratio') or 0:.0f}%"],
    ]
    t2 = Table(fin_data, colWidths=[60*mm, 55*mm, 55*mm])
    t2.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), font_name),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#1565c0")),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("GRID",         (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("ALIGN",        (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6*mm))

    # 免責
    story.append(Paragraph("※ 本レポートはAIによる自動審査結果です。最終判断は担当者の責任において行ってください。", small_style))

    doc.build(story)
    return buf.getvalue()
