"""
高精度モンテカルロ リース審査システム（拡張版）

機能:
1. 業種別ボラティリティ（製造業・小売業・建設業など）
2. リース期間中の財務悪化を時系列で予測
3. 複数企業のポートフォリオリスク分析
4. 審査結果のPDF レポート出力
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# PDF生成
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)

# ============================================================
# 1. 業種別ボラティリティ設定
# ============================================================

INDUSTRY_VOLATILITY = {
    "製造業": {
        "revenue_vol": 0.12, "margin_vol": 0.04, "equity_vol": 0.02,
        "debt_vol": 0.08, "revenue_drift": 0.02,
        "description": "景気変動の影響を受けやすいが比較的安定"
    },
    "小売業": {
        "revenue_vol": 0.18, "margin_vol": 0.06, "equity_vol": 0.03,
        "debt_vol": 0.12, "revenue_drift": 0.01,
        "description": "消費動向に敏感・季節性あり"
    },
    "建設業": {
        "revenue_vol": 0.20, "margin_vol": 0.07, "equity_vol": 0.03,
        "debt_vol": 0.15, "revenue_drift": 0.01,
        "description": "受注の波が大きく資金繰りリスクあり"
    },
    "IT・情報通信": {
        "revenue_vol": 0.15, "margin_vol": 0.05, "equity_vol": 0.025,
        "debt_vol": 0.10, "revenue_drift": 0.04,
        "description": "成長性高いが競争激しい"
    },
    "飲食・サービス": {
        "revenue_vol": 0.25, "margin_vol": 0.08, "equity_vol": 0.04,
        "debt_vol": 0.18, "revenue_drift": 0.00,
        "description": "外部環境（景気・疫病等）の影響が最大"
    },
    "卸売業": {
        "revenue_vol": 0.14, "margin_vol": 0.04, "equity_vol": 0.02,
        "debt_vol": 0.10, "revenue_drift": 0.015,
        "description": "薄利多売・在庫リスクあり"
    },
    "不動産": {
        "revenue_vol": 0.10, "margin_vol": 0.03, "equity_vol": 0.02,
        "debt_vol": 0.07, "revenue_drift": 0.02,
        "description": "安定収益だが金利感応度高い"
    },
    "運輸・物流": {
        "revenue_vol": 0.13, "margin_vol": 0.045, "equity_vol": 0.025,
        "debt_vol": 0.09, "revenue_drift": 0.02,
        "description": "燃料費・人件費の影響を受ける"
    },
}

DEFAULT_VOLATILITY = {
    "revenue_vol": 0.15, "margin_vol": 0.05, "equity_vol": 0.03,
    "debt_vol": 0.10, "revenue_drift": 0.02,
    "description": "標準的なボラティリティ設定"
}

# 既存システムの industry_major コード → モンテカルロ業種名マッピング
INDUSTRY_MAJOR_MAP: Dict[str, str] = {
    "A": "飲食・サービス",   # 農業・林業
    "B": "飲食・サービス",   # 漁業
    "C": "製造業",           # 鉱業・採石業
    "D": "建設業",
    "E": "製造業",
    "F": "製造業",           # 電気・ガス
    "G": "運輸・物流",       # 情報通信
    "H": "運輸・物流",
    "I": "卸売業",
    "J": "小売業",
    "K": "飲食・サービス",   # 金融・保険
    "L": "不動産",
    "M": "IT・情報通信",     # 学術・専門
    "N": "飲食・サービス",   # 宿泊・飲食
    "O": "飲食・サービス",   # 生活関連
    "P": "飲食・サービス",   # 教育
    "Q": "飲食・サービス",   # 医療・福祉
    "R": "飲食・サービス",   # 複合サービス
    "S": "飲食・サービス",   # その他サービス
    "T": "飲食・サービス",   # 公務
}

def map_industry_from_major(industry_major: str) -> str:
    """既存システムの industry_major（例: 'D 建設業'）をモンテカルロ業種名に変換。"""
    code = (industry_major or "").split(" ")[0].strip().upper()
    return INDUSTRY_MAJOR_MAP.get(code, "製造業")

# ============================================================
# 2. データクラス
# ============================================================

@dataclass
class CompanyData:
    """企業財務データ"""
    name: str
    industry: str
    revenue: float           # 売上高（円）
    operating_margin: float  # 営業利益率（小数）
    equity_ratio: float      # 自己資本比率（小数）
    total_debt: float        # 借入金残高（円）
    lease_amount: float = 0  # リース希望額（円）
    lease_months: int = 36   # リース期間（月）


@dataclass
class SimResult:
    """1社分のシミュレーション結果"""
    company: CompanyData
    default_prob: float
    risk_level: str
    score_median: float
    score_p5: float
    score_p95: float
    time_series_default_prob: np.ndarray
    score_paths: np.ndarray
    revenue_paths: np.ndarray
    sensitivity: Dict[str, float]
    var_95: float


@dataclass
class PortfolioResult:
    """ポートフォリオ全体の分析結果"""
    results: List[SimResult]
    total_exposure: float
    weighted_default_prob: float
    concentration_risk: float
    expected_loss: float
    portfolio_var_95: float


# ============================================================
# 3. モンテカルロエンジン
# ============================================================

class AdvancedMonteCarloEngine:

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_sim = n_simulations
        np.random.seed(seed)

    def _get_vol(self, industry: str) -> dict:
        return INDUSTRY_VOLATILITY.get(industry, DEFAULT_VOLATILITY)

    def _gbm_paths(self, S0, mu, sigma, n_periods, dt=1/12) -> np.ndarray:
        paths = np.zeros((self.n_sim, n_periods + 1))
        paths[:, 0] = S0
        for t in range(1, n_periods + 1):
            z = np.random.standard_normal(self.n_sim)
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )
        return paths

    def _score(self, revenue, margin, equity, debt) -> np.ndarray:
        rev_s  = np.clip(np.log1p(revenue) / np.log1p(1e10) * 25, 0, 25)
        mar_s  = np.clip((margin + 0.10) / 0.30 * 25, 0, 25)
        eq_s   = np.clip(equity / 0.60 * 25, 0, 25)
        debt_s = np.clip((1 - np.log1p(debt) / np.log1p(1e10)) * 25, 0, 25)
        return rev_s + mar_s + eq_s + debt_s

    def _default_mask(self, equity, margin, debt, revenue,
                      eq_thresh=0.10, mar_thresh=-0.05, cov_thresh=8.0):
        op_profit = revenue * margin
        coverage = np.where(op_profit > 0, debt / op_profit, np.inf)
        return (equity < eq_thresh) | (margin < mar_thresh) | (coverage > cov_thresh)

    def simulate_company(self, company: CompanyData) -> SimResult:
        vol = self._get_vol(company.industry)
        T = company.lease_months
        dt = 1 / 12

        rev_paths = self._gbm_paths(company.revenue, vol["revenue_drift"], vol["revenue_vol"], T, dt)
        mar_paths = self._gbm_paths(company.operating_margin, 0.0, vol["margin_vol"], T, dt)
        mar_paths = np.clip(mar_paths, -0.30, 0.50)
        eq_paths  = self._gbm_paths(company.equity_ratio, 0.005, vol["equity_vol"], T, dt)
        eq_paths  = np.clip(eq_paths, 0.01, 0.99)
        debt_paths = self._gbm_paths(company.total_debt, -0.02, vol["debt_vol"], T, dt)
        debt_paths = np.clip(debt_paths, 0, None)

        ts_default = np.zeros(T + 1)
        ever_defaulted = np.zeros(self.n_sim, dtype=bool)
        for t in range(T + 1):
            mask = self._default_mask(eq_paths[:, t], mar_paths[:, t], debt_paths[:, t], rev_paths[:, t])
            ever_defaulted |= mask
            ts_default[t] = ever_defaulted.mean()

        scores = self._score(rev_paths[:, -1], mar_paths[:, -1], eq_paths[:, -1], debt_paths[:, -1])
        sensitivity = self._sensitivity(company)
        base_score = self._score(
            np.array([company.revenue]), np.array([company.operating_margin]),
            np.array([company.equity_ratio]), np.array([company.total_debt])
        )[0]
        var_95 = base_score - float(np.percentile(scores, 5))

        return SimResult(
            company=company,
            default_prob=float(ts_default[-1]),
            risk_level=self._risk_level(float(ts_default[-1])),
            score_median=float(np.median(scores)),
            score_p5=float(np.percentile(scores, 5)),
            score_p95=float(np.percentile(scores, 95)),
            time_series_default_prob=ts_default,
            score_paths=scores[:200],
            revenue_paths=rev_paths[:100],
            sensitivity=sensitivity,
            var_95=var_95,
        )

    def _sensitivity(self, c: CompanyData) -> Dict[str, float]:
        base = self._score(
            np.array([c.revenue]), np.array([c.operating_margin]),
            np.array([c.equity_ratio]), np.array([c.total_debt])
        )[0]
        tests = {
            "売上+10%":       dict(revenue=c.revenue * 1.1),
            "売上-10%":       dict(revenue=c.revenue * 0.9),
            "営業利益率+3pt": dict(operating_margin=c.operating_margin + 0.03),
            "営業利益率-3pt": dict(operating_margin=c.operating_margin - 0.03),
            "自己資本比率+5pt": dict(equity_ratio=min(c.equity_ratio + 0.05, 0.99)),
            "自己資本比率-5pt": dict(equity_ratio=max(c.equity_ratio - 0.05, 0.01)),
            "借入金+20%":     dict(total_debt=c.total_debt * 1.2),
            "借入金-20%":     dict(total_debt=c.total_debt * 0.8),
        }
        out = {}
        for label, override in tests.items():
            d = {"revenue": c.revenue, "operating_margin": c.operating_margin,
                 "equity_ratio": c.equity_ratio, "total_debt": c.total_debt}
            d.update(override)
            s = self._score(
                np.array([d["revenue"]]), np.array([d["operating_margin"]]),
                np.array([d["equity_ratio"]]), np.array([d["total_debt"]])
            )[0]
            out[label] = float(s - base)
        return out

    def _risk_level(self, prob: float) -> str:
        if prob < 0.05:  return "低リスク"
        if prob < 0.15:  return "中リスク"
        if prob < 0.30:  return "高リスク"
        return "極高リスク"

    def analyze_portfolio(self, companies: List[CompanyData]) -> PortfolioResult:
        results = [self.simulate_company(c) for c in companies]
        total_exp = sum(c.lease_amount for c in companies)
        weights = [c.lease_amount / total_exp if total_exp > 0 else 1/len(companies)
                   for c in companies]
        w_def_prob = sum(w * r.default_prob for w, r in zip(weights, results))
        sorted_w = sorted(weights, reverse=True)
        conc = sum(sorted_w[:3]) if len(sorted_w) >= 3 else 1.0
        exp_loss = sum(c.lease_amount * r.default_prob * 0.4 for c, r in zip(companies, results))
        p_var = sum(w * r.var_95 for w, r in zip(weights, results))
        return PortfolioResult(
            results=results, total_exposure=total_exp,
            weighted_default_prob=w_def_prob, concentration_risk=conc,
            expected_loss=exp_loss, portfolio_var_95=p_var,
        )


# ============================================================
# 4. グラフ生成
# ============================================================

RISK_COLORS = {
    "低リスク":   "#27ae60",
    "中リスク":   "#f39c12",
    "高リスク":   "#e74c3c",
    "極高リスク": "#8e44ad",
}


def make_company_chart(result: SimResult) -> bytes:
    """1社分の詳細チャート（PNG bytes）"""
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#f8f9fa')
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    col = RISK_COLORS.get(result.risk_level, "#95a5a6")

    # 1. 売上シミュレーションパス
    ax1 = fig.add_subplot(gs[0, 0])
    months = np.arange(result.revenue_paths.shape[1])
    for path in result.revenue_paths:
        ax1.plot(months, path / 1e6, alpha=0.08, color='steelblue', lw=0.7)
    mean_path = result.revenue_paths.mean(axis=0)
    ax1.plot(months, mean_path / 1e6, color='navy', lw=2, label='Mean')
    ax1.fill_between(months,
        np.percentile(result.revenue_paths, 5, axis=0) / 1e6,
        np.percentile(result.revenue_paths, 95, axis=0) / 1e6,
        alpha=0.15, color='steelblue', label='5-95%ile')
    ax1.set_title('Revenue Simulation Paths', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Month'); ax1.set_ylabel('Revenue (M JPY)')
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    # 2. 累積デフォルト確率
    ax2 = fig.add_subplot(gs[0, 1])
    ts = result.time_series_default_prob
    x = np.arange(len(ts))
    ax2.fill_between(x, ts * 100, alpha=0.3, color=col)
    ax2.plot(x, ts * 100, color=col, lw=2.5)
    ax2.axhline(15, color='orange', ls='--', lw=1, alpha=0.7, label='15% warn')
    ax2.axhline(30, color='red', ls='--', lw=1, alpha=0.7, label='30% danger')
    ax2.set_title('Cumulative Default Probability', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Month'); ax2.set_ylabel('Prob (%)')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(100, max(ts * 100) * 1.2 + 5))

    # 3. スコア分布
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(result.score_paths, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(result.score_median, color='navy', lw=2, label=f'Median: {result.score_median:.1f}')
    ax3.axvline(result.score_p5, color='red', ls='--', lw=1.5, label=f'5%ile: {result.score_p5:.1f}')
    ax3.axvline(result.score_p95, color='green', ls='--', lw=1.5, label=f'95%ile: {result.score_p95:.1f}')
    ax3.set_title('Score Distribution', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Score (0-100)'); ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    # 4. 感度分析
    ax4 = fig.add_subplot(gs[1, :2])
    labels = list(result.sensitivity.keys())
    vals = list(result.sensitivity.values())
    bar_colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals]
    bars = ax4.barh(labels, vals, color=bar_colors, alpha=0.8, height=0.6)
    ax4.axvline(0, color='black', lw=0.8)
    for bar, val in zip(bars, vals):
        sign = '+' if val >= 0 else ''
        ax4.text(val + (0.05 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2,
                 f'{sign}{val:.2f}', va='center',
                 ha='left' if val >= 0 else 'right', fontsize=8)
    ax4.set_title('Sensitivity Analysis (Score Change)', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Score Change'); ax4.grid(True, alpha=0.3, axis='x')

    # 5. リスクサマリーテーブル
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    summary = [
        ["Item", "Value"],
        ["Risk Level", result.risk_level],
        ["Default Prob", f"{result.default_prob:.1%}"],
        ["Score (Median)", f"{result.score_median:.1f}"],
        ["Score (5%ile)", f"{result.score_p5:.1f}"],
        ["Score (95%ile)", f"{result.score_p95:.1f}"],
        ["VaR (95%)", f"{result.var_95:.1f}pt"],
        ["Industry", result.company.industry],
        ["Lease Period", f"{result.company.lease_months}mo"],
    ]
    tbl = ax5.table(cellText=summary[1:], colLabels=summary[0],
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50'); cell.set_text_props(color='white', fontweight='bold')
        elif r == 1:
            cell.set_facecolor(col + '40')
        else:
            cell.set_facecolor('#ecf0f1' if r % 2 == 0 else 'white')

    fig.suptitle(f'Monte Carlo Lease Assessment: {result.company.name}',
                 fontsize=13, fontweight='bold', y=0.98)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_portfolio_chart(portfolio: PortfolioResult) -> bytes:
    """ポートフォリオ全体のチャート（PNG bytes）"""
    results = portfolio.results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # バブルチャート
    ax1 = axes[0]
    for r in results:
        size = max(r.company.lease_amount / 1e6 * 10, 50)
        col = RISK_COLORS.get(r.risk_level, '#95a5a6')
        ax1.scatter(r.score_median, r.default_prob * 100, s=size, color=col,
                    alpha=0.7, edgecolors='white', lw=1.5)
        ax1.annotate(r.company.name, (r.score_median, r.default_prob * 100),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax1.axhline(15, color='orange', ls='--', lw=1, alpha=0.6)
    ax1.axhline(30, color='red', ls='--', lw=1, alpha=0.6)
    ax1.set_xlabel('Score (Median)'); ax1.set_ylabel('Default Prob (%)')
    ax1.set_title('Risk Map\n(bubble = lease amount)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # リスクレベル別 金額
    ax2 = axes[1]
    levels = ["低リスク", "中リスク", "高リスク", "極高リスク"]
    risk_amounts = {lv: 0 for lv in levels}
    risk_counts  = {lv: 0 for lv in levels}
    for r in results:
        lv = r.risk_level
        risk_amounts[lv] += r.company.lease_amount
        risk_counts[lv]  += 1
    amounts = [risk_amounts[lv] / 1e6 for lv in levels]
    cols = [RISK_COLORS[lv] for lv in levels]
    bars = ax2.bar(range(len(levels)), amounts, color=cols, alpha=0.8)
    for bar, lv in zip(bars, levels):
        cnt = risk_counts[lv]
        if cnt > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{cnt}件', ha='center', fontsize=9, fontweight='bold')
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(["Low", "Mid", "High", "Very\nHigh"], fontsize=9)
    ax2.set_ylabel('Lease Amount (M JPY)')
    ax2.set_title('Portfolio by Risk Level', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 期待損失ランキング
    ax3 = axes[2]
    el_list = sorted(
        [(r.company.name, r.company.lease_amount * r.default_prob * 0.4 / 1e4) for r in results],
        key=lambda x: x[1], reverse=True
    )
    el_names = [e[0] for e in el_list]
    el_vals  = [e[1] for e in el_list]
    bar_cols = [RISK_COLORS.get(
        next((r.risk_level for r in results if r.company.name == n), "中リスク"), '#95a5a6')
        for n in el_names]
    ax3.barh(el_names, el_vals, color=bar_cols, alpha=0.8)
    ax3.set_xlabel('Expected Loss (10K JPY)')
    ax3.set_title('Expected Loss Ranking\n(LGD=40%)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Portfolio Risk Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ============================================================
# 5. PDFレポート生成
# ============================================================

def _risk_color_rl(risk_level: str):
    mapping = {
        "低リスク":   colors.HexColor("#27ae60"),
        "中リスク":   colors.HexColor("#f39c12"),
        "高リスク":   colors.HexColor("#e74c3c"),
        "極高リスク": colors.HexColor("#8e44ad"),
    }
    return mapping.get(risk_level, colors.gray)


def _pf_eval_text(prob: float) -> str:
    if prob < 0.05:  return "良好"
    if prob < 0.15:  return "要注意"
    return "要精査"


def _generate_comment(result: SimResult) -> str:
    c = result.company
    vol = INDUSTRY_VOLATILITY.get(c.industry, DEFAULT_VOLATILITY)
    comments = []
    if result.risk_level == "低リスク":
        comments.append(
            f"{c.name}は財務基盤が安定しており、"
            f"{c.lease_months}ヶ月のリース期間中のデフォルト確率は"
            f"{result.default_prob:.1%}と低水準です。"
        )
    elif result.risk_level == "中リスク":
        comments.append(
            f"{c.name}は現状の財務指標は許容範囲内ですが、"
            f"シミュレーション上のデフォルト確率は{result.default_prob:.1%}と"
            f"やや高めです。定期的な財務モニタリングを推奨します。"
        )
    else:
        comments.append(
            f"{c.name}はリース期間中のデフォルト確率が{result.default_prob:.1%}と"
            f"高く、慎重な審査が必要です。追加担保や保証人の確保を検討してください。"
        )
    comments.append(
        f"業種（{c.industry}）の特性: {vol['description']}。"
        f"売上高ボラティリティ{vol['revenue_vol']:.0%}、"
        f"営業利益率ボラティリティ{vol['margin_vol']:.0%}を考慮したシミュレーションを実施。"
    )
    top_neg = min(result.sensitivity.items(), key=lambda x: x[1])
    top_pos = max(result.sensitivity.items(), key=lambda x: x[1])
    comments.append(
        f"感度分析によると、「{top_neg[0]}」が最もスコアを引き下げるリスク要因（{top_neg[1]:.2f}pt）、"
        f"「{top_pos[0]}」が最も改善に寄与する要因（+{top_pos[1]:.2f}pt）です。"
    )
    return " ".join(comments)


def generate_pdf_bytes(portfolio: PortfolioResult) -> bytes:
    """
    PDFレポートをメモリ上（bytes）で生成して返す。
    Streamlit の st.download_button に直接渡せる。
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor('#2c3e50'), alignment=TA_CENTER)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=14, textColor=colors.HexColor('#2c3e50'), spaceBefore=12, spaceAfter=6)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=11, textColor=colors.HexColor('#34495e'), spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9, leading=14, textColor=colors.HexColor('#2c3e50'))
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#7f8c8d'))

    story = []
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # 表紙
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("Monte Carlo Lease Assessment Report", title_style))
    story.append(Paragraph("モンテカルロ リース審査レポート",
        ParagraphStyle('SubTitle', parent=styles['Normal'],
            fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor('#7f8c8d'), spaceAfter=4)))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2c3e50')))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(f"作成日時: {now}",
        ParagraphStyle('Date', parent=styles['Normal'],
            fontSize=10, alignment=TA_RIGHT, textColor=colors.HexColor('#95a5a6'))))
    story.append(Paragraph(
        f"対象企業数: {len(portfolio.results)}社  |  "
        f"総リース額: {portfolio.total_exposure/1e6:,.1f}百万円",
        ParagraphStyle('Summary', parent=styles['Normal'],
            fontSize=11, alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50'), spaceBefore=8)))
    story.append(Spacer(1, 10*mm))

    # ポートフォリオサマリー
    story.append(Paragraph("1. ポートフォリオ サマリー", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7')))
    story.append(Spacer(1, 3*mm))
    pf_data = [
        ["指標", "値", "評価"],
        ["加重平均デフォルト確率", f"{portfolio.weighted_default_prob:.2%}", _pf_eval_text(portfolio.weighted_default_prob)],
        ["上位3社への集中度", f"{portfolio.concentration_risk:.1%}", "要注意" if portfolio.concentration_risk > 0.6 else "適切"],
        ["期待損失額（LGD=40%）", f"{portfolio.expected_loss/1e4:,.0f}万円", ""],
        ["ポートフォリオ VaR（95%）", f"{portfolio.portfolio_var_95:.1f}pt スコア下落", ""],
    ]
    pf_tbl = Table(pf_data, colWidths=[65*mm, 55*mm, 45*mm])
    pf_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ALIGN',      (1, 0), (-1, -1), 'CENTER'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ecf0f1'), colors.white]),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(pf_tbl)
    story.append(Spacer(1, 5*mm))

    pf_chart_bytes = make_portfolio_chart(portfolio)
    story.append(Image(io.BytesIO(pf_chart_bytes), width=175*mm, height=58*mm))
    story.append(Spacer(1, 5*mm))

    # 企業一覧
    story.append(Paragraph("企業別 審査結果一覧", h2_style))
    header = ["企業名", "業種", "スコア", "デフォルト確率", "リスクレベル", "リース額(万円)"]
    rows = [header]
    for r in portfolio.results:
        rows.append([r.company.name, r.company.industry, f"{r.score_median:.1f}",
                     f"{r.default_prob:.2%}", r.risk_level, f"{r.company.lease_amount/1e4:,.0f}"])
    list_tbl = Table(rows, colWidths=[38*mm, 30*mm, 20*mm, 28*mm, 24*mm, 25*mm])
    ts = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ALIGN',      (2, 0), (-1, -1), 'CENTER'),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ecf0f1'), colors.white]),
    ]
    for i, r in enumerate(portfolio.results, 1):
        col_r = _risk_color_rl(r.risk_level)
        ts.append(('TEXTCOLOR', (4, i), (4, i), col_r))
    list_tbl.setStyle(TableStyle(ts))
    story.append(list_tbl)

    # 個社別詳細
    story.append(PageBreak())
    story.append(Paragraph("2. 個社別 詳細分析", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7')))
    for i, result in enumerate(portfolio.results):
        story.append(Spacer(1, 4*mm))
        c = result.company
        rc = _risk_color_rl(result.risk_level)
        story.append(Paragraph(f"2-{i+1}. {c.name}",
            ParagraphStyle('CompName', parent=styles['Heading2'],
                fontSize=12, textColor=colors.HexColor('#2c3e50'), spaceBefore=6)))
        story.append(Paragraph(
            f"[{result.risk_level}]  デフォルト確率: {result.default_prob:.2%}  |  "
            f"スコア: {result.score_median:.1f} / 100  |  VaR: {result.var_95:.1f}pt",
            ParagraphStyle('Badge', parent=styles['Normal'],
                fontSize=11, textColor=rc, fontName='Helvetica-Bold')))
        story.append(Spacer(1, 2*mm))
        fin_data = [
            ["項目", "現在値", "項目", "現在値"],
            ["業種", c.industry, "リース希望額", f"{c.lease_amount/1e4:,.0f}万円"],
            ["売上高", f"{c.revenue/1e6:,.1f}百万円", "リース期間", f"{c.lease_months}ヶ月"],
            ["営業利益率", f"{c.operating_margin:.1%}", "スコア(5%ile)", f"{result.score_p5:.1f}"],
            ["自己資本比率", f"{c.equity_ratio:.1%}", "スコア(95%ile)", f"{result.score_p95:.1f}"],
            ["借入金残高", f"{c.total_debt/1e6:,.1f}百万円",
             "業種特性", INDUSTRY_VOLATILITY.get(c.industry, DEFAULT_VOLATILITY)["description"][:12]],
        ]
        fin_tbl = Table(fin_data, colWidths=[35*mm, 45*mm, 35*mm, 50*mm])
        fin_tbl.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
            ('FONTSIZE',    (0, 0), (-1, -1), 8),
            ('ALIGN',       (1, 0), (1, -1), 'RIGHT'),
            ('ALIGN',       (3, 0), (3, -1), 'RIGHT'),
            ('GRID',        (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ecf0f1'), colors.white]),
            ('TOPPADDING',  (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(fin_tbl)
        story.append(Spacer(1, 3*mm))
        chart_bytes = make_company_chart(result)
        story.append(Image(io.BytesIO(chart_bytes), width=175*mm, height=100*mm))
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("審査コメント", h2_style))
        story.append(Paragraph(_generate_comment(result), body_style))
        if i < len(portfolio.results) - 1:
            story.append(PageBreak())

    # 免責事項
    story.append(PageBreak())
    story.append(Paragraph("3. 注意事項・免責事項", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7')))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "本レポートはモンテカルロシミュレーションによる統計的分析結果であり、"
        "将来の財務状況や信用リスクを保証するものではありません。"
        "審査判断は本レポートを参考情報として、担当者の総合的な判断のもとで行ってください。"
        "また、業種別ボラティリティの設定値は一般的な傾向に基づくものであり、"
        "個社の実態と異なる場合があります。",
        small_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ============================================================
# 既存システムの res dict → CompanyData 変換ヘルパー
# ============================================================

def res_to_company_data(res: dict, company_name: str = "審査対象",
                        lease_amount_man: float = 0,
                        lease_months: int = 36) -> CompanyData:
    """
    lease_logic_sumaho11 の審査結果 dict から CompanyData を生成する。

    Args:
        res:              審査結果dict（score, user_op, user_eq, financials 等）
        company_name:     企業名（表示用）
        lease_amount_man: リース希望額（万円）
        lease_months:     リース期間（月）
    """
    fin = res.get("financials") or {}
    nenshu     = (fin.get("nenshu", 0) or 0) * 10_000          # 万円 → 円
    op_margin  = (res.get("user_op", 0) or 0) / 100            # % → 小数
    eq_ratio   = max((res.get("user_eq", 0) or 0) / 100, 0.01) # % → 小数、最低1%
    bank_c     = (fin.get("bank_credit",  0) or 0) * 1_000_000 # 百万円 → 円
    lease_c    = (fin.get("lease_credit", 0) or 0) * 1_000_000
    total_debt = bank_c + lease_c
    industry   = map_industry_from_major(res.get("industry_major", ""))

    return CompanyData(
        name=company_name,
        industry=industry,
        revenue=max(nenshu, 1),
        operating_margin=op_margin,
        equity_ratio=eq_ratio,
        total_debt=max(total_debt, 0),
        lease_amount=lease_amount_man * 10_000,
        lease_months=lease_months,
    )
