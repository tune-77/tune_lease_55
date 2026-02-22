"""
indicators.py — 財務指標の計算・業界目安比較・倒産確率。

含む機能:
- compute_financial_indicators  : 財務データから指標リストを算出
- analyze_indicators_vs_bench   : 指標と業界目安の差を分析してテキスト生成
- get_indicator_analysis_for_advice : AI相談用の指標分析サマリを生成
- calculate_pd                  : 簡易倒産確率の計算
"""

import os
import json

from charts import _equity_ratio_display, LOWER_IS_BETTER_NAMES
from web_services import _WEB_BENCH_KEYS, _load_web_benchmarks_cache
from config import BASE_DIR


# ─── ローカルデータの遅延ロード ───────────────────────────────────────────────

_benchmarks_data: dict | None = None
_avg_data: dict | None = None


def _load_json(filename: str) -> dict:
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_benchmarks() -> dict:
    global _benchmarks_data
    if _benchmarks_data is None:
        _benchmarks_data = _load_json("industry_benchmarks.json")
    return _benchmarks_data


def _get_avg_data() -> dict:
    global _avg_data
    if _avg_data is None:
        _avg_data = _load_json("industry_averages.json")
    return _avg_data


# ─── 財務指標の計算 ───────────────────────────────────────────────────────────

def compute_financial_indicators(fin: dict, bench: dict | None = None) -> list:
    """
    入力済み財務データから算出可能な指標のリストを返す。
    fin  : last_result["financials"]（千円単位）
    bench: industry_benchmarks の当該業種エントリ（op_margin, equity_ratio 等）
    返却 : [{"name": str, "value": float, "bench": float|None, "unit": str}]
    """
    n       = fin.get("nenshu") or 0
    total   = fin.get("assets") or 0
    net_a   = fin.get("net_assets")
    gross   = fin.get("gross_profit") or 0
    op      = fin.get("op_profit") or fin.get("rieki") or 0
    ord_p   = fin.get("ord_profit") or 0
    net     = fin.get("net_income") or 0
    machines = fin.get("machines") or 0
    other_a  = fin.get("other_assets") or 0
    bank    = fin.get("bank_credit") or 0
    lease   = fin.get("lease_credit") or 0
    dep     = fin.get("depreciation") or 0
    fixed_a    = machines + other_a   # 固定資産（機械＋その他資産）
    debt_total = bank + lease          # 借入金等

    indicators = []

    # ── 売上高ベースの利益率（売上高 > 0 で算出可能） ──
    if n > 0:
        indicators.append({"name": "売上高総利益率",      "value": gross / n * 100, "bench": bench.get("gross_margin")          if bench else None, "unit": "%"})
        indicators.append({"name": "営業利益率",          "value": op    / n * 100, "bench": bench.get("op_margin")             if bench else None, "unit": "%"})
        indicators.append({"name": "経常利益率",          "value": ord_p / n * 100, "bench": bench.get("ord_margin")            if bench else None, "unit": "%"})
        indicators.append({"name": "当期純利益率",        "value": net   / n * 100, "bench": bench.get("net_margin")            if bench else None, "unit": "%"})
        if dep > 0:
            indicators.append({"name": "減価償却費/売上高", "value": dep / n * 100, "bench": bench.get("dep_ratio")             if bench else None, "unit": "%"})
        if fixed_a > 0:
            indicators.append({"name": "固定資産回転率",  "value": n / fixed_a,     "bench": bench.get("fixed_asset_turnover")  if bench else None, "unit": "回"})

    # ── 総資産・純資産ベース（total > 0 で算出可能） ──
    if total > 0:
        if net_a is not None and net_a > 0:
            indicators.append({"name": "自己資本比率",         "value": net_a / total * 100,         "bench": _equity_ratio_display(bench.get("equity_ratio")) if bench else None, "unit": "%"})
            indicators.append({"name": "ROE(自己資本利益率)",  "value": net   / net_a * 100,         "bench": bench.get("roe")               if bench else None, "unit": "%"})
            indicators.append({"name": "固定比率",             "value": fixed_a / net_a * 100,       "bench": bench.get("fixed_to_equity")   if bench else None, "unit": "%"})
            indicators.append({"name": "負債比率",             "value": (total - net_a) / net_a * 100, "bench": bench.get("debt_to_equity")  if bench else None, "unit": "%"})
        indicators.append({"name": "ROA(総資産利益率)",        "value": net / total * 100,            "bench": bench.get("roa")               if bench else None, "unit": "%"})
        indicators.append({"name": "総資産回転率",             "value": n   / total if n > 0 else 0,  "bench": bench.get("asset_turnover")    if bench else None, "unit": "回"})
        if fixed_a > 0:
            indicators.append({"name": "固定資産比率",         "value": fixed_a / total * 100,       "bench": bench.get("fixed_ratio")       if bench else None, "unit": "%"})
        indicators.append({"name": "流動資産比率(総資産比)",   "value": (total - fixed_a) / total * 100, "bench": bench.get("current_asset_ratio") if bench else None, "unit": "%"})
        if debt_total > 0:
            indicators.append({"name": "借入金等依存度",       "value": debt_total / total * 100,    "bench": bench.get("debt_ratio")        if bench else None, "unit": "%"})

    return indicators


# ─── 業界目安との比較・分析テキスト ─────────────────────────────────────────

def analyze_indicators_vs_bench(indicators: list) -> tuple:
    """
    指標と業界目安の差を見て分析文を返す。
    返却: (要約1行: str, 詳細マークダウン: str)
    """
    above, below = [], []
    for ind in indicators:
        bench = ind.get("bench")
        if bench is None or (isinstance(bench, float) and bench != bench):
            continue
        name  = ind["name"]
        value = ind["value"]
        unit  = ind.get("unit", "%")
        diff  = value - bench
        if name in LOWER_IS_BETTER_NAMES:
            if value < bench:
                above.append((name, value, bench, diff, unit))
            else:
                below.append((name, value, bench, diff, unit))
        else:
            if diff > 0:
                above.append((name, value, bench, diff, unit))
            elif diff < 0:
                below.append((name, value, bench, diff, unit))

    lines = []
    if above:
        parts = [f"**{n}**（貴社 {v:.1f}{u} / 業界目安 {b:.1f}{u}、差 {d:+.1f}{u}）" for n, v, b, d, u in above]
        lines.append("**業界目安を上回っている指標**\n- " + "\n- ".join(parts))
    if below:
        parts = [f"**{n}**（貴社 {v:.1f}{u} / 業界目安 {b:.1f}{u}、差 {d:+.1f}{u}）" for n, v, b, d, u in below]
        lines.append("**業界目安を下回っている指標**\n- " + "\n- ".join(parts))
    if not lines:
        return ("業界目安と比較できる指標がありません。",
                "業界目安が登録されている指標がひとつもないため、差の分析は行えません。")

    detail = "\n\n".join(lines)
    if any(n == "借入金等依存度" for n, *_ in above):
        detail += "\n\n※ 借入金等依存度は「業界より低い」＝負債が相対的に少なく健全と解釈しています。"
    elif any(n == "借入金等依存度" for n, *_ in below):
        detail += "\n\n※ 借入金等依存度は業界より高く出ています。返済余力・担保とのバランスを確認してください。"

    n_above, n_below = len(above), len(below)
    if n_below == 0:
        summary = "算出指標はおおむね業界目安を上回っており、財務面は良好です。"
    elif n_above == 0:
        summary = "算出指標の多くが業界目安を下回っています。利益率・効率性・負債水準の改善余地を検討してください。"
    else:
        summary = (f"業界目安を上回っている指標が{n_above}件、下回っている指標が{n_below}件あります。"
                   "強みを維持しつつ、下回っている項目の要因確認をおすすめします。")
    return summary, detail


# ─── AI相談用の指標分析サマリ ─────────────────────────────────────────────────

def get_indicator_analysis_for_advice(last_result: dict) -> tuple:
    """
    last_result から業界目安を組み立て、指標の差の分析（要約・内訳）と指標一覧テキストを返す。
    AI相談で「指標の分析と改善アドバイス」に使う。
    返却: (summary: str, detail: str, indicators_text: str)。データ不足時は ("", "", "")。
    """
    if not last_result:
        return "", "", ""
    fin = last_result.get("financials", {})
    if not fin:
        return "", "", ""

    selected_sub = last_result.get("industry_sub", "")
    major        = last_result.get("industry_major", "")
    benchmarks   = _get_benchmarks()
    avg_data     = _get_avg_data()

    bench = dict(benchmarks.get(selected_sub, {}))
    cache = _load_web_benchmarks_cache()
    for k in _WEB_BENCH_KEYS:
        v = cache.get(selected_sub, {}).get(k)
        if v is not None:
            bench[k] = v

    bench_ext = dict(bench)
    if major and avg_data and major in avg_data:
        avg = avg_data[major]
        an  = avg.get("nenshu") or 0
        if an > 0:
            for key, src in [("gross_margin", "gross_profit"), ("ord_margin", "ord_profit"),
                             ("net_margin", "net_income"), ("dep_ratio", "depreciation")]:
                if bench_ext.get(key) is None:
                    bench_ext[key] = (avg.get(src) or 0) / an * 100
        total_avg = sum(avg.get(k) or 0 for k in ["machines", "other_assets", "bank_credit", "lease_credit"])
        if total_avg > 0:
            if bench_ext.get("roa")            is None: bench_ext["roa"]            = (avg.get("net_income") or 0) / total_avg * 100
            if bench_ext.get("asset_turnover") is None: bench_ext["asset_turnover"] = an / total_avg
            if bench_ext.get("fixed_ratio")    is None: bench_ext["fixed_ratio"]    = ((avg.get("machines") or 0) + (avg.get("other_assets") or 0)) / total_avg * 100
            if bench_ext.get("debt_ratio")     is None: bench_ext["debt_ratio"]     = ((avg.get("bank_credit") or 0) + (avg.get("lease_credit") or 0)) / total_avg * 100

    indicators = compute_financial_indicators(fin, bench_ext)
    if not indicators:
        return "", "", ""

    summary, detail = analyze_indicators_vs_bench(indicators)
    lines = [
        f"- {ind['name']}: 貴社 {ind['value']:.1f}{ind.get('unit','%')}"
        + (f" / 業界目安 {ind['bench']:.1f}{ind.get('unit','%')}" if ind.get("bench") is not None else "")
        for ind in indicators
    ]
    return summary, detail, "\n".join(lines)


# ─── 簡易倒産確率 ─────────────────────────────────────────────────────────────

def calculate_pd(equity: float, current: float, profit: float | None) -> float:
    """
    財務指標に基づく簡易倒産確率（%）を計算する。
    equity : 自己資本比率（%）
    current: 流動比率（%）
    profit : 営業利益率（%）、None 可
    """
    risk = 0.0
    if equity < 10:
        risk += 25.0
    elif equity < 20:
        risk += 12.0
    elif equity < 30:
        risk += 5.0
    if current < 100:
        risk += 20.0
    elif current < 120:
        risk += 8.0
    elif current < 150:
        risk += 3.0
    if profit is not None:
        if profit < 0:
            risk += 30.0
        elif profit < 2:
            risk += 10.0
        elif profit < 5:
            risk += 4.0
    return min(100.0, max(0.0, risk))
