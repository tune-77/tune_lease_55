"""
高精度モンテカルロ リース審査システム(拡張版)

機能:
1. 業種別ボラティリティ(製造業・小売業・建設業など)
2. リース期間中の財務悪化を時系列で予測
3. 複数企業のポートフォリオリスク分析
4. 審査結果のPDF レポート出力
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import io
import os
import platform
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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ── 日本語フォント設定 ────────────────────────────────────────────────────────

# ReportLab: CIDフォント(PDFビューア内蔵・フォントファイル不要)
_RL_JP_FONT = "HeiseiKakuGo-W5"
_RL_JP_FONT_BOLD = "HeiseiKakuGo-W5"
try:
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
    _RL_JP_FONT_BOLD = "HeiseiKakuGo-W5"
except Exception:
    _RL_JP_FONT = "Helvetica"
    _RL_JP_FONT_BOLD = "Helvetica-Bold"

# Matplotlib: OS別に日本語フォントを自動検出
def _detect_jp_font_path() -> Optional[str]:
    """macOS / Linux / Windows で日本語TTFを探して返す。見つからなければ None。"""
    candidates: List[str] = []
    sys = platform.system()
    if sys == "Darwin":
        candidates = [
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode MS.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        ]
    elif sys == "Linux":
        candidates = [
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/truetype/ipafont/ipagp.ttf",
            "/usr/share/fonts/truetype/ipafont-gothic/ipagp.ttf",
        ]
    elif sys == "Windows":
        candidates = [
            "C:/Windows/Fonts/meiryo.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def _setup_matplotlib_jp():
    """matplotlibに日本語フォントを設定する。"""
    import matplotlib.font_manager as fm
    font_path = _detect_jp_font_path()
    if font_path:
        try:
            fe = fm.FontEntry(fname=font_path, name="JpFont")
            fm.fontManager.ttflist.append(fe)
            plt.rcParams["font.family"] = "sans-serif"
            current = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "JpFont"] + current
        except Exception:
            pass

_setup_matplotlib_jp()

# 日本語テキスト用 FontProperties(直接指定用)
def _get_jp_font_prop(size: float = 10):
    """日本語フォントの FontProperties を返す。フォントが見つからなければ None。"""
    from matplotlib.font_manager import FontProperties
    fp = _detect_jp_font_path()
    if fp:
        return FontProperties(fname=fp, size=size)
    return None

def _w(text: str) -> str:
    """ASCII英数記号を全角に変換(日本語フォントで混在テキストを表示するため)。"""
    _TBL = str.maketrans(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789+-.,:%()[]",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789+-.,:%()[]"
    )
    return text.translate(_TBL)

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
        "description": "外部環境(景気・疫病等)の影響が最大"
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
    """既存システムの industry_major(例: 'D 建設業')をモンテカルロ業種名に変換。"""
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
    revenue: float           # 売上高(円)
    operating_margin: float  # 営業利益率(小数)
    equity_ratio: float      # 自己資本比率(小数)
    total_debt: float        # 借入金残高(円)
    lease_amount: float = 0  # リース希望額(円)
    lease_months: int = 36   # リース期間(月)
    subsidy_amount: float = 0  # 適用補助金額(円)。負債の初期値を圧縮する形でモデル化


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
        # 補助金は初期借入残高を圧縮する形でモデル化(補助分だけ調達不要になる)
        effective_debt = max(company.total_debt - company.subsidy_amount, 0.0)
        debt_paths = self._gbm_paths(effective_debt, -0.02, vol["debt_vol"], T, dt)
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
            "売上+10%":         dict(revenue=c.revenue * 1.1),
            "売上-10%":         dict(revenue=c.revenue * 0.9),
            "営業利益率+3pt":   dict(operating_margin=c.operating_margin + 0.03),
            "営業利益率-3pt":   dict(operating_margin=c.operating_margin - 0.03),
            "自己資本比率+5pt": dict(equity_ratio=min(c.equity_ratio + 0.05, 0.99)),
            "自己資本比率-5pt": dict(equity_ratio=max(c.equity_ratio - 0.05, 0.01)),
            "借入金+20%":       dict(total_debt=c.total_debt * 1.2),
            "借入金-20%":       dict(total_debt=c.total_debt * 0.8),
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
    """1社分の詳細チャート(PNG bytes)— 3行2列レイアウト"""
    jp_fp    = _get_jp_font_prop(size=10)
    jp_fp_lg = _get_jp_font_prop(size=13)

    fig = plt.figure(figsize=(18, 17))
    fig.patch.set_facecolor('#f8f9fa')
    # 3行2列: 上段(売上/デフォルト) 中段(スコア分布/サマリー表) 下段(感度分析・全幅)
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                  height_ratios=[1, 1, 0.85])
    col = RISK_COLORS.get(result.risk_level, "#95a5a6")

    # ── 1. 売上シミュレーションパス(左上)────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    months = np.arange(result.revenue_paths.shape[1])
    for path in result.revenue_paths:
        ax1.plot(months, path / 1e6, alpha=0.07, color='steelblue', lw=0.6)
    mean_path = result.revenue_paths.mean(axis=0)
    ax1.plot(months, mean_path / 1e6, color='navy', lw=2.5, label='平均')
    ax1.fill_between(months,
        np.percentile(result.revenue_paths, 5,  axis=0) / 1e6,
        np.percentile(result.revenue_paths, 95, axis=0) / 1e6,
        alpha=0.18, color='steelblue', label='5〜95%ile')
    ax1.set_title('売上シミュレーションパス', fontsize=12, fontweight='bold',
                  **({'fontproperties': jp_fp} if jp_fp else {}))
    ax1.set_xlabel('Month', fontsize=10)
    ax1.set_ylabel('Revenue (M JPY)', fontsize=10)
    ax1.legend(fontsize=9,
               prop=jp_fp if jp_fp else None)
    ax1.grid(True, alpha=0.3)

    # ── 2. 累積デフォルト確率(右上)────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ts = result.time_series_default_prob
    x  = np.arange(len(ts))
    ax2.fill_between(x, ts * 100, alpha=0.25, color=col)
    ax2.plot(x, ts * 100, color=col, lw=2.5)
    ax2.axhline(15, color='orange', ls='--', lw=1.5, alpha=0.8, label='15% 警告')
    ax2.axhline(30, color='red',    ls='--', lw=1.5, alpha=0.8, label='30% 危険')
    ax2.set_title('累積デフォルト確率', fontsize=12, fontweight='bold',
                  **({'fontproperties': jp_fp} if jp_fp else {}))
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('確率 (%)', fontsize=10,
                   **({'fontproperties': jp_fp} if jp_fp else {}))
    ax2.legend(fontsize=9, prop=jp_fp if jp_fp else None)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, min(100, max(ts * 100) * 1.25 + 5))

    # ── 3. スコア分布ヒストグラム(左中)────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(result.score_paths, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax3.axvline(result.score_median, color='navy',  lw=2.5,
                label=f'中央値: {result.score_median:.1f}')
    ax3.axvline(result.score_p5,     color='red',   ls='--', lw=2,
                label=f'5%ile: {result.score_p5:.1f}')
    ax3.axvline(result.score_p95,    color='green', ls='--', lw=2,
                label=f'95%ile: {result.score_p95:.1f}')
    ax3.set_title('スコア分布', fontsize=12, fontweight='bold',
                  **({'fontproperties': jp_fp} if jp_fp else {}))
    ax3.set_xlabel('Score (0-100)', fontsize=10)
    ax3.set_ylabel('頻度', fontsize=10,
                   **({'fontproperties': jp_fp} if jp_fp else {}))
    ax3.legend(fontsize=9, prop=jp_fp if jp_fp else None)
    ax3.grid(True, alpha=0.3)

    # ── 4. リスクサマリーテーブル(右中)────────────────────────
    _RISK_JP = {"低リスク": "低リスク", "中リスク": "中リスク",
                "高リスク": "高リスク", "極高リスク": "極高リスク"}
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    summary = [
        ["項目",          "値"],
        ["リスクレベル",   _RISK_JP.get(result.risk_level, result.risk_level)],
        ["デフォルト確率", f"{result.default_prob:.1%}"],
        ["スコア中央値",   f"{result.score_median:.1f}"],
        ["スコア 5%ile",  f"{result.score_p5:.1f}"],
        ["スコア 95%ile", f"{result.score_p95:.1f}"],
        ["VaR (95%)",     f"{result.var_95:.1f}pt"],
        ["業種",           result.company.industry],
        ["リース期間",     f"{result.company.lease_months}ヶ月"],
        ["リース金額",     f"{result.company.lease_amount/1e4:,.0f}万円"],
    ]
    tbl = ax5.table(cellText=summary[1:], colLabels=summary[0],
                    cellLoc='center', loc='center', bbox=[0.05, 0, 0.95, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold',
                                **({'fontproperties': jp_fp} if jp_fp else {}))
        elif r == 1:
            cell.set_facecolor(col + '55')
            if jp_fp:
                cell.set_text_props(fontproperties=jp_fp)
        else:
            cell.set_facecolor('#ecf0f1' if r % 2 == 0 else 'white')
            if jp_fp:
                cell.set_text_props(fontproperties=jp_fp)
        cell.set_height(0.095)
    ax5.set_title('リスクサマリー', fontsize=12, fontweight='bold', pad=10,
                  **({'fontproperties': jp_fp} if jp_fp else {}))

    # ── 5. 感度分析(下段・全幅)────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    labels = list(result.sensitivity.keys())
    vals   = list(result.sensitivity.values())
    bar_colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in vals]
    y_pos = range(len(labels))
    bars  = ax4.barh(list(y_pos), vals, color=bar_colors, alpha=0.85, height=0.55)
    ax4.set_yticks(list(y_pos))
    ax4.set_yticklabels([""] * len(labels))
    ax4.axvline(0, color='black', lw=1)
    max_abs = max(abs(v) for v in vals) if vals else 1
    for i, (bar, val, lbl) in enumerate(zip(bars, vals, labels)):
        sign = '+' if val >= 0 else ''
        ax4.text(val + (max_abs * 0.02 if val >= 0 else -max_abs * 0.02),
                 bar.get_y() + bar.get_height() / 2,
                 f'{sign}{val:.2f}', va='center',
                 ha='left' if val >= 0 else 'right', fontsize=10, fontweight='bold')
        kw = {'fontproperties': jp_fp} if jp_fp else {'fontsize': 10}
        ax4.text(-max_abs * 1.08, bar.get_y() + bar.get_height() / 2,
                 lbl, va='center', ha='right', fontsize=10, **kw)
    ax4.set_title('感度分析(スコアへの影響)', fontsize=12, fontweight='bold',
                  **({'fontproperties': jp_fp} if jp_fp else {}))
    ax4.set_xlabel('スコア変化量', fontsize=10,
                   **({'fontproperties': jp_fp} if jp_fp else {}))
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.margins(y=0.12)

    # ── タイトル ─────────────────────────────────────────────────
    if jp_fp_lg:
        fig.suptitle(f'モンテカルロ リース審査: {result.company.name}',
                     fontsize=14, fontweight='bold', y=1.005,
                     fontproperties=jp_fp_lg)
    else:
        fig.suptitle(f'Monte Carlo Lease Assessment: {result.company.name}',
                     fontsize=14, fontweight='bold', y=1.005)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_portfolio_chart(portfolio: PortfolioResult) -> bytes:
    """ポートフォリオ全体のチャート(PNG bytes)"""
    jp_fp = _get_jp_font_prop(size=7)
    jp_fp9 = _get_jp_font_prop(size=9)

    results = portfolio.results
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#f8f9fa')
    gs_pf = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # バブルチャート(左上)
    ax1 = fig.add_subplot(gs_pf[0, 0])
    for r in results:
        size = max(r.company.lease_amount / 1e6 * 10, 50)
        col = RISK_COLORS.get(r.risk_level, '#95a5a6')
        ax1.scatter(r.score_median, r.default_prob * 100, s=size, color=col,
                    alpha=0.7, edgecolors='white', lw=1.5)
        ann_kw = {'fontproperties': jp_fp} if jp_fp else {'fontsize': 7}
        ax1.annotate(r.company.name, (r.score_median, r.default_prob * 100),
                     textcoords="offset points", xytext=(5, 5),
                     **(ann_kw if jp_fp else {'fontsize': 7}))
    ax1.axhline(15, color='orange', ls='--', lw=1, alpha=0.6)
    ax1.axhline(30, color='red', ls='--', lw=1, alpha=0.6)
    ax1.set_xlabel('Score (Median)'); ax1.set_ylabel('Default Prob (%)')
    ax1.set_title('Risk Map\n(bubble = lease amount)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # リスクレベル別 金額(右上)
    ax2 = fig.add_subplot(gs_pf[0, 1])
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
            # jp_fp使用時は全角数字にしないと□になる
            cnt_str = _w(str(cnt)) + '件' if jp_fp9 else f'{cnt}cases'
            kw = {'fontproperties': jp_fp9} if jp_fp9 else {'fontsize': 9, 'fontweight': 'bold'}
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     cnt_str, ha='center', **kw)
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(["Low", "Mid", "High", "Very\nHigh"], fontsize=9)
    ax2.set_ylabel('Lease Amount (M JPY)')
    ax2.set_title('Portfolio by Risk Level', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 期待損失ランキング(下段・横幅全体)
    ax3 = fig.add_subplot(gs_pf[1, :])
    el_list = sorted(
        [(r.company.name, r.company.lease_amount * r.default_prob * 0.4 / 1e4) for r in results],
        key=lambda x: x[1], reverse=True
    )
    el_names = [e[0] for e in el_list]
    el_vals  = [e[1] for e in el_list]
    bar_cols = [RISK_COLORS.get(
        next((r.risk_level for r in results if r.company.name == n), "中リスク"), '#95a5a6')
        for n in el_names]
    x_pos = range(len(el_names))
    ax3.bar(x_pos, el_vals, color=bar_cols, alpha=0.8, width=0.6)
    ax3.set_xticks(list(x_pos))
    ax3.set_xticklabels([""] * len(el_names))
    for i, (name, val) in enumerate(zip(el_names, el_vals)):
        kw = {'fontproperties': jp_fp9} if jp_fp9 else {'fontsize': 9}
        ax3.text(i, -max(el_vals) * 0.04 if el_vals else 0, name,
                 va='top', ha='center', **kw)
        ax3.text(i, val + max(el_vals) * 0.01 if el_vals else val,
                 f"{val:,.0f}", va='bottom', ha='center', fontsize=9, fontweight='bold')
    ax3.set_ylabel('Expected Loss (10K JPY)', fontsize=10)
    ax3.set_title('Expected Loss Ranking (LGD=40%)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.margins(x=0.05)

    fig.suptitle('Portfolio Risk Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
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
        f"業種({c.industry})の特性: {vol['description']}。"
        f"売上高ボラティリティ{vol['revenue_vol']:.0%}、"
        f"営業利益率ボラティリティ{vol['margin_vol']:.0%}を考慮したシミュレーションを実施。"
    )
    top_neg = min(result.sensitivity.items(), key=lambda x: x[1])
    top_pos = max(result.sensitivity.items(), key=lambda x: x[1])
    comments.append(
        f"感度分析によると、「{top_neg[0]}」が最もスコアを引き下げるリスク要因({top_neg[1]:.2f}pt)、"
        f"「{top_pos[0]}」が最も改善に寄与する要因(+{top_pos[1]:.2f}pt)です。"
    )
    return " ".join(comments)


def generate_pdf_bytes(portfolio: PortfolioResult) -> bytes:
    """
    PDFレポートをメモリ上(bytes)で生成して返す。
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
        fontSize=20, spaceAfter=6, textColor=colors.HexColor('#2c3e50'),
        alignment=TA_CENTER, fontName=_RL_JP_FONT)
    h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
        fontSize=14, textColor=colors.HexColor('#2c3e50'),
        spaceBefore=12, spaceAfter=6, fontName=_RL_JP_FONT)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=11, textColor=colors.HexColor('#34495e'),
        spaceBefore=8, spaceAfter=4, fontName=_RL_JP_FONT)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=9, leading=14, textColor=colors.HexColor('#2c3e50'), fontName=_RL_JP_FONT)
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
        fontSize=8, textColor=colors.HexColor('#7f8c8d'), fontName=_RL_JP_FONT)

    story = []
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # 表紙
    story.append(Spacer(1, 20*mm))
    story.append(Paragraph("Monte Carlo Lease Assessment Report", title_style))
    story.append(Paragraph("モンテカルロ リース審査レポート",
        ParagraphStyle('SubTitle', parent=styles['Normal'],
            fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor('#7f8c8d'),
            spaceAfter=4, fontName=_RL_JP_FONT)))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2c3e50')))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(f"作成日時: {now}",
        ParagraphStyle('Date', parent=styles['Normal'],
            fontSize=10, alignment=TA_RIGHT, textColor=colors.HexColor('#95a5a6'),
            fontName=_RL_JP_FONT)))
    story.append(Paragraph(
        f"対象企業数: {len(portfolio.results)}社  |  "
        f"総リース額: {portfolio.total_exposure/1e6:,.1f}百万円",
        ParagraphStyle('Summary', parent=styles['Normal'],
            fontSize=11, alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50'), spaceBefore=8, fontName=_RL_JP_FONT)))
    story.append(Spacer(1, 10*mm))

    # ポートフォリオサマリー
    story.append(Paragraph("1. ポートフォリオ サマリー", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7')))
    story.append(Spacer(1, 3*mm))
    pf_data = [
        ["指標", "値", "評価"],
        ["加重平均デフォルト確率", f"{portfolio.weighted_default_prob:.2%}", _pf_eval_text(portfolio.weighted_default_prob)],
        ["上位3社への集中度", f"{portfolio.concentration_risk:.1%}", "要注意" if portfolio.concentration_risk > 0.6 else "適切"],
        ["期待損失額(LGD=40%)", f"{portfolio.expected_loss/1e4:,.0f}万円", ""],
        ["ポートフォリオ VaR(95%)", f"{portfolio.portfolio_var_95:.1f}pt スコア下落", ""],
    ]
    pf_tbl = Table(pf_data, colWidths=[65*mm, 55*mm, 45*mm])
    pf_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, -1), _RL_JP_FONT),
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
        ('FONTNAME',   (0, 0), (-1, -1), _RL_JP_FONT),
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
                fontSize=12, textColor=colors.HexColor('#2c3e50'),
                spaceBefore=6, fontName=_RL_JP_FONT)))
        story.append(Paragraph(
            f"[{result.risk_level}]  デフォルト確率: {result.default_prob:.2%}  |  "
            f"スコア: {result.score_median:.1f} / 100  |  VaR: {result.var_95:.1f}pt",
            ParagraphStyle('Badge', parent=styles['Normal'],
                fontSize=11, textColor=rc, fontName=_RL_JP_FONT_BOLD)))
        story.append(Spacer(1, 2*mm))
        fin_data = [
            ["項目", "現在値", "項目", "現在値"],
            ["業種", c.industry, "リース希望額", f"{c.lease_amount/1e4:,.0f}万円"],
            ["売上高", f"{c.revenue/1e6:,.1f}百万円", "リース期間", f"{c.lease_months}ヶ月"],
            ["営業利益率", f"{c.operating_margin:.1%}", "スコア(5%ile)", f"{result.score_p5:.1f}"],
            ["自己資本比率", f"{c.equity_ratio:.1%}", "スコア(95%ile)", f"{result.score_p95:.1f}"],
            ["負債合計", f"{c.total_debt/1e6:,.1f}百万円",
             "業種特性", INDUSTRY_VOLATILITY.get(c.industry, DEFAULT_VOLATILITY)["description"][:12]],
        ]
        fin_tbl = Table(fin_data, colWidths=[35*mm, 45*mm, 35*mm, 50*mm])
        fin_tbl.setStyle(TableStyle([
            ('BACKGROUND',  (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR',   (0, 0), (-1, 0), colors.white),
            ('FONTNAME',    (0, 0), (-1, -1), _RL_JP_FONT),
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
        res:              審査結果dict(score, user_op, user_eq, financials 等)
        company_name:     企業名(表示用)
        lease_amount_man: リース希望額(万円)
        lease_months:     リース期間(月)
    """
    fin = res.get("financials") or {}
    nenshu       = (fin.get("nenshu", 0) or 0) * 1_000           # 千円 → 円
    op_margin    = (res.get("user_op", 0) or 0) / 100            # % → 小数
    eq_ratio     = max((res.get("user_eq", 0) or 0) / 100, 0.01) # % → 小数、最低1%
    # 借入金残高 = 総資産 - 純資産(負債合計)。bank_credit/lease_creditは当社与信残高なので使わない
    total_assets = (fin.get("assets",     0) or 0) * 1_000       # 千円 → 円
    net_assets_v = (fin.get("net_assets", 0) or 0) * 1_000       # 千円 → 円
    total_debt   = max(total_assets - net_assets_v, 0)
    industry     = map_industry_from_major(res.get("industry_major", ""))

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
