#!/usr/bin/env python3
"""Fetch and analyze Cabinet Office machinery orders from the e-Stat API."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "external" / "estat_machinery_orders"
REPORT_DIR = ROOT / "reports"
STATS_DATA_ID = "0003355222"
START_TIME = "2023000303"
END_TIME = "2026000303"
API_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
ESTAT_URL = f"https://www.e-stat.go.jp/dbview?sid={STATS_DATA_ID}"
CAO_URL = "https://www.esri.cao.go.jp/jp/stat/juchu/menu_juchu.html"

DEMAND_NAMES = {
    "100": "受注額合計",
    "120": "海外需要",
    "130": "官公需計",
    "140": "民間需要合計",
    "160": "民間需要（船舶・電力を除く）",
    "170": "民間需要_製造業計",
    "200": "民間需要_非製造業（船舶・電力を除く）",
}
SERIES_NAMES = {"100": "季節調整系列", "110": "原系列"}
CORE_NAME = DEMAND_NAMES["160"]


def fetch_data() -> dict:
    app_id = os.environ.get("ESTAT_APP_ID", "").strip()
    if not app_id:
        raise RuntimeError("ESTAT_APP_ID environment variable is required")

    params = {
        "appId": app_id,
        "statsDataId": STATS_DATA_ID,
        "cdCat01": ",".join(DEMAND_NAMES),
        "cdCat02": ",".join(SERIES_NAMES),
        "cdTimeFrom": START_TIME,
        "cdTimeTo": END_TIME,
        "metaGetFlg": "N",
        "cntGetFlg": "N",
        "lang": "J",
        "limit": "10000",
    }
    request = urllib.request.Request(
        API_URL + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": "tunelease/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    result = payload.get("GET_STATS_DATA", {}).get("RESULT", {})
    if result.get("STATUS") != 0:
        raise RuntimeError(f"e-Stat API error: {result}")
    return payload


def parse_payload(payload: dict) -> pd.DataFrame:
    values = (
        payload["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"].get("VALUE", [])
    )
    if isinstance(values, dict):
        values = [values]

    rows = []
    for value in values:
        time_code = value["@time"]
        year = int(time_code[:4])
        month = int(time_code[-2:])
        rows.append(
            {
                "date": pd.Timestamp(year=year, month=month, day=1),
                "time_code": time_code,
                "demand_code": value["@cat01"],
                "demand": DEMAND_NAMES[value["@cat01"]],
                "series_code": value["@cat02"],
                "series": SERIES_NAMES[value["@cat02"]],
                "value_million_yen": float(value["$"]),
                "value_100m_yen": float(value["$"]) / 100.0,
                "unit": value.get("@unit", "100万円"),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["date", "series_code", "demand_code"]
    )
    expected = 37 * len(DEMAND_NAMES) * len(SERIES_NAMES)
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected} observations, got {len(frame)}")
    return frame


def calculate_analysis(frame: pd.DataFrame) -> dict:
    sa = frame[frame["series_code"] == "100"].pivot(
        index="date", columns="demand", values="value_100m_yen"
    )
    original = frame[frame["series_code"] == "110"].pivot(
        index="date", columns="demand", values="value_100m_yen"
    )
    core = sa[CORE_NAME]
    core_original = original[CORE_NAME]
    latest_date = core.index.max()

    mom = core.pct_change() * 100
    yoy = core_original.pct_change(12) * 100
    rolling3 = core.rolling(3).mean()
    rolling12_mean = core.rolling(12).mean()
    rolling12_std = core.rolling(12).std()
    trend_change = (
        rolling3.iloc[-1] / rolling3.iloc[-4] - 1
        if len(rolling3.dropna()) >= 4
        else np.nan
    ) * 100

    x = np.arange(len(core))
    slope = float(np.polyfit(x, core.values, 1)[0])
    corr = sa.pct_change().corr()[CORE_NAME].drop(CORE_NAME).sort_values(
        ascending=False
    )

    signal_score = 0
    signal_score += 1 if rolling3.iloc[-1] > rolling12_mean.iloc[-1] else -1
    signal_score += 1 if mom.iloc[-1] > 0 else -1
    signal_score += 1 if yoy.iloc[-1] > 0 else -1
    signal = (
        "改善"
        if signal_score >= 2
        else "減速"
        if signal_score <= -2
        else "中立"
    )

    return {
        "metadata": {
            "stats_data_id": STATS_DATA_ID,
            "period": "2023-03 to 2026-03",
            "latest_month": latest_date.strftime("%Y-%m"),
            "observations": int(len(frame)),
            "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "unit": "100 million yen",
            "estat_url": ESTAT_URL,
            "cabinet_office_url": CAO_URL,
        },
        "core_indicator": {
            "name": CORE_NAME,
            "latest_value_100m_yen": round(float(core.iloc[-1]), 1),
            "latest_mom_pct": round(float(mom.iloc[-1]), 1),
            "latest_yoy_pct_original": round(float(yoy.iloc[-1]), 1),
            "latest_3m_average_100m_yen": round(float(rolling3.iloc[-1]), 1),
            "latest_12m_average_100m_yen": round(float(rolling12_mean.iloc[-1]), 1),
            "latest_12m_volatility_100m_yen": round(float(rolling12_std.iloc[-1]), 1),
            "three_month_average_change_vs_3m_ago_pct": round(
                float(trend_change), 1
            ),
            "monthly_linear_trend_100m_yen": round(slope, 1),
            "period_min_100m_yen": round(float(core.min()), 1),
            "period_min_month": core.idxmin().strftime("%Y-%m"),
            "period_max_100m_yen": round(float(core.max()), 1),
            "period_max_month": core.idxmax().strftime("%Y-%m"),
            "macro_signal": signal,
            "macro_signal_score": signal_score,
        },
        "latest_values_100m_yen": {
            name: round(float(sa[name].iloc[-1]), 1) for name in sa.columns
        },
        "change_correlations_with_core": {
            name: round(float(value), 3) for name, value in corr.items()
        },
        "lease_system_recommendation": {
            "usable": True,
            "recommended_role": "macro overlay / industry outlook feature",
            "do_not_use_as": "standalone borrower credit decision or default label",
            "features": [
                "machinery_orders_core_mom",
                "machinery_orders_core_yoy",
                "machinery_orders_core_3m_vs_12m",
                "machinery_orders_manufacturing_mom",
                "machinery_orders_nonmanufacturing_mom",
                "machinery_orders_overseas_mom",
                "machinery_orders_macro_signal",
            ],
            "governance": [
                "Use only data that was publicly available at the screening date.",
                "Store publication date and vintage to prevent look-ahead bias.",
                "Treat annual seasonal-adjustment revisions as versioned updates.",
                "Validate incremental predictive value with out-of-time tests.",
            ],
        },
    }


def configure_japanese_font() -> None:
    candidates = [
        "Hiragino Sans",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next(
        (font for font in candidates if font in installed), "DejaVu Sans"
    )
    plt.rcParams["axes.unicode_minus"] = False


def create_chart(frame: pd.DataFrame, analysis: dict, output: Path) -> None:
    configure_japanese_font()
    sa = frame[frame["series_code"] == "100"].pivot(
        index="date", columns="demand", values="value_100m_yen"
    )
    core = sa[CORE_NAME]
    core_3m = core.rolling(3).mean()

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    ax = axes[0]
    ax.plot(core.index, core, color="#155e75", linewidth=1.8, label="除船・電 民需")
    ax.plot(
        core_3m.index,
        core_3m,
        color="#dc2626",
        linewidth=2.6,
        label="3か月移動平均",
    )
    ax.axhline(
        core.mean(),
        color="#64748b",
        linestyle="--",
        linewidth=1,
        label="期間平均",
    )
    ax.scatter(core.index[-1], core.iloc[-1], color="#dc2626", zorder=5)
    ax.annotate(
        f"{core.index[-1]:%Y-%m}\n{core.iloc[-1]:,.0f}億円",
        (core.index[-1], core.iloc[-1]),
        xytext=(-70, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#334155"},
        fontsize=10,
    )
    ax.set_title("機械受注統計: 民間設備投資の先行指標", fontsize=15, weight="bold")
    ax.set_ylabel("季節調整値（億円）")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3)

    ax = axes[1]
    comparison = {
        "製造業": sa["民間需要_製造業計"],
        "非製造業\n(除船・電)": sa["民間需要_非製造業（船舶・電力を除く）"],
        "海外需要": sa["海外需要"],
    }
    colors = ["#2563eb", "#16a34a", "#9333ea"]
    for (label, series), color in zip(comparison.items(), colors):
        index_series = series / series.iloc[0] * 100
        ax.plot(index_series.index, index_series, label=label, color=color, linewidth=2)
    ax.axhline(100, color="#64748b", linestyle="--", linewidth=1)
    ax.set_title("需要部門別指数（2023年3月=100）", fontsize=14, weight="bold")
    ax.set_ylabel("指数")
    ax.set_xlabel("出所: e-Stat / 内閣府 機械受注統計調査")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=3)

    signal = analysis["core_indicator"]["macro_signal"]
    fig.suptitle(
        f"2023年3月〜2026年3月 | 最新マクロ判定: {signal}",
        fontsize=12,
        color="#334155",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(analysis: dict, output: Path) -> None:
    core = analysis["core_indicator"]
    latest = analysis["latest_values_100m_yen"]
    correlations = analysis["change_correlations_with_core"]
    corr_text = "\n".join(
        f"- {name}: {value:.3f}" for name, value in correlations.items()
    )
    text = f"""# 機械受注統計分析（2023年3月〜2026年3月）

## 結論

内閣府が民間設備投資の先行指標として扱う「船舶・電力を除く民需」は、2026年3月に
**{core['latest_value_100m_yen']:,.1f}億円**。前月比は
**{core['latest_mom_pct']:+.1f}%**、原系列の前年同月比は
**{core['latest_yoy_pct_original']:+.1f}%**、3か月平均の3か月前比は
**{core['three_month_average_change_vs_3m_ago_pct']:+.1f}%**で、機械受注のマクロ判定は
**{core['macro_signal']}**。

リースシステムには利用可能。ただし、個社の信用力を直接判定する変数ではなく、
設備投資環境を示すマクロ補助指標として使うべきである。

## 最新値

- 受注額合計: {latest['受注額合計']:,.1f}億円
- 船舶・電力を除く民需: {latest[CORE_NAME]:,.1f}億円
- 製造業: {latest['民間需要_製造業計']:,.1f}億円
- 非製造業（船舶・電力を除く）: {latest['民間需要_非製造業（船舶・電力を除く）']:,.1f}億円
- 海外需要: {latest['海外需要']:,.1f}億円
- 官公需: {latest['官公需計']:,.1f}億円

## 期間分析

- 期間最小: {core['period_min_month']}の{core['period_min_100m_yen']:,.1f}億円
- 期間最大: {core['period_max_month']}の{core['period_max_100m_yen']:,.1f}億円
- 最新3か月平均: {core['latest_3m_average_100m_yen']:,.1f}億円
- 最新12か月平均: {core['latest_12m_average_100m_yen']:,.1f}億円
- 最新12か月標準偏差: {core['latest_12m_volatility_100m_yen']:,.1f}億円
- 期間の線形傾向: 1か月当たり{core['monthly_linear_trend_100m_yen']:+.1f}億円

月次変化率と「船舶・電力を除く民需」の相関:

{corr_text}

## リースシステムへの利用案

1. **審査時点の景況補正**: 3か月平均と12か月平均の乖離を、設備投資環境の追い風・向かい風として表示する。
2. **業種別警戒**: 製造業と非製造業を分け、顧客業種に合う系列の前月比・前年同月比を審査コメントへ反映する。
3. **資産価値の補助判断**: 機械需要が減速する局面では、中古流通性・再販価格・再リース余地を通常より厳しく確認する。
4. **営業機会検知**: 上昇局面は設備投資需要の増加候補として、案件発掘・案件量予測に使う。
5. **モデル特徴量**: `core_mom`、`core_yoy`、`3m_vs_12m`、製造業・非製造業・外需の変化率を候補とし、時系列外部検証で追加効果を確認する。

## 制約とガバナンス

- 個社の返済能力、資金繰り、担保価値を直接示す統計ではないため、単独で承認・否決に使わない。
- 公表には対象月から時間差がある。審査日より後に公表された値を学習へ混入させない。
- 季節調整値は毎年改訂されるため、取得日時とデータ版を保存する。
- 2023年3月〜2026年3月の37か月はモデル学習には短い。まずルール表示・モニタリング用途から導入する。
- 予測力を検証する場合は、案件の審査日ベースで時点整合した特徴量を作り、out-of-time検証を行う。

## データ

- e-Stat統計表ID: `{STATS_DATA_ID}`
- 観測数: {analysis['metadata']['observations']}（7項目 × 2系列 × 37か月）
- 単位: 億円へ換算（API原単位は100万円）
- e-Stat: {ESTAT_URL}
- 内閣府: {CAO_URL}
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    payload = fetch_data()
    frame = parse_payload(payload)
    analysis = calculate_analysis(frame)

    raw_output = OUTPUT_DIR / "machinery_orders_202303_202603_raw.json"
    long_output = OUTPUT_DIR / "machinery_orders_202303_202603_long.csv"
    wide_output = OUTPUT_DIR / "machinery_orders_202303_202603_wide.csv"
    analysis_output = OUTPUT_DIR / "machinery_orders_analysis.json"
    chart_output = REPORT_DIR / "machinery_orders_202303_202603.png"
    report_output = REPORT_DIR / "machinery_orders_202303_202603.md"

    raw_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    frame.to_csv(long_output, index=False, encoding="utf-8-sig")
    wide = frame.pivot(
        index="date", columns=["series", "demand"], values="value_100m_yen"
    )
    wide.to_csv(wide_output, encoding="utf-8-sig")
    analysis_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    create_chart(frame, analysis, chart_output)
    write_report(analysis, report_output)

    print(json.dumps(analysis["core_indicator"], ensure_ascii=False, indent=2))
    for path in [
        raw_output,
        long_output,
        wide_output,
        analysis_output,
        chart_output,
        report_output,
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
