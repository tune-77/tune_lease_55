#!/usr/bin/env python3
"""
FinceptTerminal 由来の Python スクリプト群（AkShare ラッパー）を活用して
日本マクロ経済データを取得し、tune_lease_55 のスコアリングに使える形式で保存する。

参照元: https://github.com/Fincept-Corporation/FinceptTerminal
        fincept-qt/scripts/akshare_macro.py, boj_fetcher.py, estat_japan_api.py

取得データ:
  - 日銀政策金利 (macro_japan_bank_rate)
  - コアCPI / 総合CPI (macro_japan_core_cpi_yearly / macro_japan_cpi_yearly)
  - 景気先行指数 (macro_japan_head_indicator)
  - 失業率 (macro_japan_unemployment_rate)

出力:
  static_data/macro_context.json  … スコアリングで industry_sentiment_z 等に利用可
  static_data/jgb_rates.json      … 既存の静的JGB金利データを最新値で追記更新

依存: pip install akshare
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static_data"

try:
    import akshare as ak
    import pandas as pd
except ImportError:
    print("Error: akshare が未インストールです。以下を実行してください:")
    print("  pip install akshare")
    sys.exit(1)


def _latest_row(df: pd.DataFrame, value_col: str) -> dict:
    """最新の有効値（NaN でない）行を返す。"""
    valid = df[df[value_col].notna()]
    if valid.empty:
        return {}
    row = valid.iloc[-1]
    return row.to_dict()


def fetch_boj_policy_rate() -> dict:
    """日銀政策金利 (%) を取得する。"""
    try:
        df = ak.macro_japan_bank_rate()
        # Columns: 时间, 前值, 现值, 发布日期
        row = _latest_row(df, "现值")
        return {
            "value": float(row.get("现值", 0)),
            "prev": float(row.get("前值", 0)),
            "period": str(row.get("时间", "")),
            "published": str(row.get("发布日期", "")),
            "source": "akshare.macro_japan_bank_rate",
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_core_cpi() -> dict:
    """日本コアCPI（前年比 %）を取得する。"""
    try:
        df = ak.macro_japan_core_cpi_yearly()
        row = _latest_row(df, "现值")
        return {
            "value": float(row.get("现值", 0)),
            "prev": float(row.get("前值", 0)),
            "period": str(row.get("时间", "")),
            "published": str(row.get("发布日期", "")),
            "source": "akshare.macro_japan_core_cpi_yearly",
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_cpi() -> dict:
    """日本総合CPI（前年比 %）を取得する。"""
    try:
        df = ak.macro_japan_cpi_yearly()
        row = _latest_row(df, "现值")
        return {
            "value": float(row.get("现值", 0)),
            "prev": float(row.get("前值", 0)),
            "period": str(row.get("时间", "")),
            "published": str(row.get("发布日期", "")),
            "source": "akshare.macro_japan_cpi_yearly",
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_leading_indicator() -> dict:
    """日本景気先行指数（CI先行指数）を取得する。"""
    try:
        df = ak.macro_japan_head_indicator()
        row = _latest_row(df, "现值")
        return {
            "value": float(row.get("现值", 0)),
            "prev": float(row.get("前值", 0)),
            "period": str(row.get("时间", "")),
            "published": str(row.get("发布日期", "")),
            "source": "akshare.macro_japan_head_indicator",
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_unemployment() -> dict:
    """日本完全失業率（%）を取得する。"""
    try:
        df = ak.macro_japan_unemployment_rate()
        row = _latest_row(df, "现值")
        return {
            "value": float(row.get("现值", 0)),
            "prev": float(row.get("前值", 0)),
            "period": str(row.get("时间", "")),
            "published": str(row.get("发布日期", "")),
            "source": "akshare.macro_japan_unemployment_rate",
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_macro_score(data: dict) -> dict:
    """
    マクロ環境スコアを計算する。
    tune_lease_55 の industry_sentiment_z や環境リスク判定に使う。

    出力:
      macro_risk_score: 0〜100（高いほど環境リスク大）
      macro_env: "良好" / "中立" / "注意" / "警戒"
      factors: 個別評価
    """
    factors = {}

    # 政策金利: 高いほどリース調達コスト上昇 → リスク大
    rate = data.get("boj_policy_rate", {}).get("value")
    if rate is not None and not isinstance(rate, str):
        if rate >= 1.0:
            factors["policy_rate"] = {"score": 70, "comment": f"政策金利 {rate:.2f}% 高水準: 調達コスト要注意"}
        elif rate >= 0.5:
            factors["policy_rate"] = {"score": 40, "comment": f"政策金利 {rate:.2f}%: 緩やかな上昇局面"}
        else:
            factors["policy_rate"] = {"score": 15, "comment": f"政策金利 {rate:.2f}%: 低金利環境継続"}

    # コアCPI: 高インフレは設備投資コスト上昇
    cpi = data.get("core_cpi", {}).get("value")
    if cpi is not None and not isinstance(cpi, str):
        if cpi >= 3.0:
            factors["core_cpi"] = {"score": 65, "comment": f"コアCPI {cpi:.1f}%: 高インフレで物件単価上昇"}
        elif cpi >= 1.5:
            factors["core_cpi"] = {"score": 35, "comment": f"コアCPI {cpi:.1f}%: 適度なインフレ"}
        else:
            factors["core_cpi"] = {"score": 20, "comment": f"コアCPI {cpi:.1f}%: 低インフレ環境"}

    # 景気先行指数: 100超=拡張、100未満=後退
    cli = data.get("leading_indicator", {}).get("value")
    if cli is not None and not isinstance(cli, str):
        if cli >= 110:
            factors["leading_indicator"] = {"score": 10, "comment": f"景気先行指数 {cli:.1f}: 強い拡張局面"}
        elif cli >= 100:
            factors["leading_indicator"] = {"score": 25, "comment": f"景気先行指数 {cli:.1f}: 緩やかな拡張"}
        elif cli >= 95:
            factors["leading_indicator"] = {"score": 50, "comment": f"景気先行指数 {cli:.1f}: 中立〜やや軟化"}
        else:
            factors["leading_indicator"] = {"score": 75, "comment": f"景気先行指数 {cli:.1f}: 後退局面のリスク"}

    # 失業率: 高いほど法人の経営環境悪化リスク
    unemp = data.get("unemployment", {}).get("value")
    if unemp is not None and not isinstance(unemp, str):
        if unemp <= 2.5:
            factors["unemployment"] = {"score": 10, "comment": f"失業率 {unemp:.1f}%: 完全雇用水準"}
        elif unemp <= 3.5:
            factors["unemployment"] = {"score": 25, "comment": f"失業率 {unemp:.1f}%: 安定的"}
        else:
            factors["unemployment"] = {"score": 60, "comment": f"失業率 {unemp:.1f}%: 労働市場軟化"}

    if not factors:
        return {"macro_risk_score": 50, "macro_env": "中立", "factors": {}}

    avg_score = sum(f["score"] for f in factors.values()) / len(factors)

    if avg_score < 25:
        env = "良好"
    elif avg_score < 45:
        env = "中立"
    elif avg_score < 65:
        env = "注意"
    else:
        env = "警戒"

    return {
        "macro_risk_score": round(avg_score, 1),
        "macro_env": env,
        "factors": factors,
    }


def build_macro_context() -> dict:
    """全マクロデータを取得して macro_context を構築する。"""
    print("日本マクロ経済データを取得中...")

    ctx = {
        "fetched_at": datetime.now().isoformat(),
        "source": "FinceptTerminal/AkShare (https://github.com/Fincept-Corporation/FinceptTerminal)",
    }

    print("  [1/4] 日銀政策金利...")
    ctx["boj_policy_rate"] = fetch_boj_policy_rate()

    print("  [2/4] コアCPI...")
    ctx["core_cpi"] = fetch_core_cpi()
    ctx["cpi"] = fetch_cpi()

    print("  [3/4] 景気先行指数...")
    ctx["leading_indicator"] = fetch_leading_indicator()

    print("  [4/4] 完全失業率...")
    ctx["unemployment"] = fetch_unemployment()

    ctx["assessment"] = _compute_macro_score(ctx)

    return ctx


def update_jgb_rates(ctx: dict) -> None:
    """
    jgb_rates.json の最新月を BOJ 政策金利から推計更新する。
    本来は MOF/BOJ の時系列データが必要だが、PoC として政策金利ベースの推計を追記する。
    """
    jgb_path = STATIC_DIR / "jgb_rates.json"
    if not jgb_path.exists():
        print(f"  警告: {jgb_path} が存在しないためスキップ")
        return

    with open(jgb_path, encoding="utf-8") as f:
        jgb = json.load(f)

    policy_rate = ctx.get("boj_policy_rate", {}).get("value")
    if policy_rate is None or isinstance(policy_rate, str):
        print("  警告: 政策金利が取得できなかったため jgb_rates.json を更新しません")
        return

    # 現在月
    ym = datetime.now().strftime("%Y-%m")

    # 政策金利から JGB 利回りを簡易推計（実際の市場データではない）
    # 5年: 政策金利 + スプレッド 0.25〜0.50%
    # 10年: 政策金利 + スプレッド 0.50〜0.80%
    estimated_5y = round(policy_rate + 0.35, 3)
    estimated_10y = round(policy_rate + 0.65, 3)

    if ym not in jgb:
        jgb[ym] = {
            "5y": estimated_5y,
            "10y": estimated_10y,
            "note": f"estimated from BOJ policy rate {policy_rate}% (fetched by fetch_fincept_data.py)",
        }
        with open(jgb_path, "w", encoding="utf-8") as f:
            json.dump(jgb, f, ensure_ascii=False, indent=4)
        print(f"  jgb_rates.json に {ym} を追記しました (5y={estimated_5y}%, 10y={estimated_10y}%)")
    else:
        print(f"  jgb_rates.json: {ym} は既存のためスキップ")


def save_macro_context(ctx: dict) -> None:
    """macro_context.json として保存する。"""
    out_path = STATIC_DIR / "macro_context.json"
    STATIC_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存完了: {out_path}")


def print_summary(ctx: dict) -> None:
    assess = ctx.get("assessment", {})
    print("\n" + "=" * 55)
    print("  日本マクロ経済サマリー")
    print("=" * 55)

    rate = ctx.get("boj_policy_rate", {})
    cpi = ctx.get("core_cpi", {})
    cli = ctx.get("leading_indicator", {})
    unemp = ctx.get("unemployment", {})

    def _fmt(d, key="value", unit=""):
        v = d.get(key)
        if v is None or isinstance(v, str):
            return "取得失敗"
        return f"{v:.2f}{unit}"

    print(f"  日銀政策金利 : {_fmt(rate, unit='%')} ({rate.get('period', '')})")
    print(f"  コアCPI      : {_fmt(cpi, unit='%')} ({cpi.get('period', '')})")
    print(f"  景気先行指数  : {_fmt(cli)} ({cli.get('period', '')})")
    print(f"  失業率       : {_fmt(unemp, unit='%')} ({unemp.get('period', '')})")
    print()
    print(f"  マクロリスクスコア : {assess.get('macro_risk_score', 'N/A')} / 100")
    print(f"  マクロ環境評価    : {assess.get('macro_env', 'N/A')}")

    for name, factor in assess.get("factors", {}).items():
        print(f"    [{name}] {factor.get('comment', '')}")

    print("=" * 55)
    print(f"  取得日時: {ctx.get('fetched_at', '')}")
    print("=" * 55)


def main() -> None:
    ctx = build_macro_context()
    save_macro_context(ctx)
    update_jgb_rates(ctx)
    print_summary(ctx)


if __name__ == "__main__":
    main()
