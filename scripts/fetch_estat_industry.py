#!/usr/bin/env python3
"""
e-Stat API から業種別財務指標を取得し industry_benchmarks.json を動的更新する。

環境変数 ESTAT_APP_ID が未設定の場合はスキップ（graceful skip）。
取得したデータは static_data/industry_estat_cache.json にキャッシュ保存。

e-Stat API 公式: https://www.e-stat.go.jp/api/
使用統計: 法人企業統計調査（財務省）statsDataId = 0003060791
  cat01 = 調査項目 (045=売上高, 048=営業利益, 069=動産・不動産賃借料, 122=自己資本比率)
  cat02 = 業種 (数値コード)
  cat03 = 規模 (26=全規模)
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import urlencode

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static_data"
BENCHMARKS_PATH = STATIC_DIR / "industry_benchmarks.json"
CACHE_PATH = STATIC_DIR / "industry_estat_cache.json"

# e-Stat API エンドポイント
ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"

# 法人企業統計（財務省）- 業種別財務指標
# statsDataId=0003060791: 時系列データ 金融業、保険業以外の業種(原数値)
STATS_DATA_IDS = {
    "法人企業統計_年報": "0003060791",
}

# 取得する調査項目コード
ITEM_SALES = "045"       # 売上高（百万円）
ITEM_OP_PROFIT = "048"   # 営業利益（百万円）
ITEM_LEASE_RENT = "069"  # 動産・不動産賃借料（百万円）
ITEM_EQUITY_RATIO = "122"  # 自己資本比率（%）

# e-Stat 業種コード(cat02) → industry_benchmarks.json キーのマッピング
INDUSTRY_CODE_MAP: dict[str, str] = {
    "109": "09 食料品製造業",
    "120": "21 金属製品製造業",
    "121": "24 生産用機械器具製造業",
    "145": "26 情報通信機械器具製造業",
    "107": "06 総合工事業",
    "127": "50-55 各種卸売業",
    "128": "56-61 各種小売業",
    "131": "44 道路貨物運送業",
    "156": "76 飲食店",
    "130": "68 不動産代理・仲介",
    "150": "70 物品賃貸業(リース・レンタル)",
    "160": "91 職業紹介・労働者派遣業",
    "152": "83 医療業(病院・診療所)",
}


def _get_app_id() -> Optional[str]:
    return os.environ.get("ESTAT_APP_ID")


def _fetch_json(url: str, params: dict, retries: int = 3) -> Optional[dict]:
    query = urlencode(params)
    full_url = f"{url}?{query}"
    for attempt in range(retries):
        try:
            with urlopen(full_url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, json.JSONDecodeError) as e:
            print(f"  警告: API呼び出し失敗 (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def fetch_corporate_enterprise_stats(app_id: str) -> Optional[dict]:
    """法人企業統計から業種別財務指標を取得する。"""
    params = {
        "appId": app_id,
        "statsDataId": STATS_DATA_IDS["法人企業統計_年報"],
        "metaGetFlg": "Y",
        "cntGetFlg": "N",
        "sectionHeaderFlg": "1",
        "cdCat01": f"{ITEM_SALES},{ITEM_OP_PROFIT},{ITEM_LEASE_RENT},{ITEM_EQUITY_RATIO}",
        "cdCat03": "26",  # 26=全規模
    }
    print(f"  e-Stat API: 法人企業統計取得中 (statsDataId={params['statsDataId']})...")
    data = _fetch_json(ESTAT_BASE_URL, params)
    if data is None:
        return None

    status = data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("STATUS", -1)
    if status != 0:
        msg = data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("ERROR_MSG", "不明なエラー")
        print(f"  警告: e-Stat API エラー (status={status}): {msg}")
        return None

    return data


def _parse_metrics(raw: dict) -> dict[str, dict[str, float]]:
    """APIレスポンスから業種コード→財務指標辞書を抽出する。

    法人企業統計(0003060791) の VALUE 配列:
      @cat01 = 調査項目コード
      @cat02 = 業種コード (数値)
      @cat03 = 規模コード (26=全規模)
      @time  = 年次, $ = 値

    返り値: {industry_code: {op_margin, equity_ratio, lease_cost_ratio}}
    """
    try:
        values = raw["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    except KeyError:
        return {}

    if isinstance(values, dict):
        values = [values]

    # {industry_code: {item_code: {year: value}}}
    raw_data: dict[str, dict[str, dict[str, float]]] = {}

    for v in values:
        cat01 = v.get("@cat01", "")
        cat02 = v.get("@cat02", "")
        cat03 = v.get("@cat03", "")
        year = v.get("@time", "")
        raw_val = v.get("$", "")
        if cat03 != "26":
            continue
        if not raw_val or raw_val in ("-", "…", "x", "X"):
            continue
        try:
            val = float(str(raw_val).replace(",", ""))
        except ValueError:
            continue
        raw_data.setdefault(cat02, {}).setdefault(cat01, {})[year] = val

    result: dict[str, dict[str, float]] = {}
    for code, items in raw_data.items():
        sales = items.get(ITEM_SALES, {})
        op_profit = items.get(ITEM_OP_PROFIT, {})
        lease_rent = items.get(ITEM_LEASE_RENT, {})
        equity_ratio_vals = items.get(ITEM_EQUITY_RATIO, {})

        entry: dict[str, float] = {}

        # op_margin = 営業利益 / 売上高 × 100
        common_years = set(sales) & set(op_profit)
        if common_years:
            yr = max(common_years)
            if sales[yr] > 0:
                entry["op_margin"] = round(op_profit[yr] / sales[yr] * 100, 2)

        # equity_ratio: 直接値（単位 %）
        if equity_ratio_vals:
            yr = max(equity_ratio_vals)
            entry["equity_ratio"] = round(equity_ratio_vals[yr], 1)

        # lease_cost_ratio = 動産・不動産賃借料 / 売上高 × 100
        common_years_lr = set(sales) & set(lease_rent)
        if common_years_lr:
            yr = max(common_years_lr)
            if sales[yr] > 0:
                entry["lease_cost_ratio"] = round(lease_rent[yr] / sales[yr] * 100, 2)

        if entry:
            result[code] = entry

    return result


def update_benchmarks(metrics: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """industry_benchmarks.json の財務指標を更新し、変更サマリを返す。"""
    with open(BENCHMARKS_PATH, encoding="utf-8") as f:
        benchmarks: dict = json.load(f)

    today = datetime.now().strftime("%Y-%m-%d")
    changes: dict[str, list[str]] = {}

    for estat_code, bench_key in INDUSTRY_CODE_MAP.items():
        if estat_code not in metrics or bench_key not in benchmarks:
            continue
        entry = metrics[estat_code]
        bench = benchmarks[bench_key]
        diffs: list[str] = []

        if "op_margin" in entry:
            old = bench.get("op_margin")
            new = entry["op_margin"]
            if old != new:
                diffs.append(f"op_margin {old}→{new}")
                bench["op_margin"] = new
                bench["op_margin_source"] = "e-Stat法人企業統計"
                bench["op_margin_updated"] = today

        if "equity_ratio" in entry:
            old = bench.get("equity_ratio")
            new = entry["equity_ratio"]
            if old != new:
                diffs.append(f"equity_ratio {old}→{new}")
                bench["equity_ratio"] = new
                bench["equity_ratio_source"] = "e-Stat法人企業統計"
                bench["equity_ratio_updated"] = today

        if "lease_cost_ratio" in entry:
            old = bench.get("lease_cost_ratio")
            new = entry["lease_cost_ratio"]
            if old != new:
                diffs.append(f"lease_cost_ratio {old}→{new}")
                bench["lease_cost_ratio"] = new
                bench["lease_cost_ratio_source"] = "e-Stat法人企業統計"
                bench["lease_cost_ratio_updated"] = today

        if diffs:
            changes[bench_key] = diffs

    benchmarks["_last_updated"] = datetime.now().isoformat()

    with open(BENCHMARKS_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, ensure_ascii=False, indent=4)

    return changes


def update_from_cache() -> bool:
    """ESTAT_APP_ID 未設定時にキャッシュから industry_benchmarks.json を更新する。

    industry_estat_cache.json が存在し metrics_by_estat_code が空でない場合のみ実行。
    """
    if not CACHE_PATH.exists():
        return False
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False

    metrics: dict[str, dict[str, float]] = cache.get("metrics_by_estat_code", {})
    if not metrics:
        return False

    fetched_at = cache.get("fetched_at", "不明")
    print(f"  キャッシュからロード (fetched_at={fetched_at}, 業種数={len(metrics)})")
    changes = update_benchmarks(metrics)
    if changes:
        print(f"  industry_benchmarks.json 更新（キャッシュ）: {len(changes)} 業種")
        for k, diffs in changes.items():
            print(f"    {k}: {', '.join(diffs)}")
    else:
        print("  変更なし（キャッシュ値と既存値が同一）")
    return True


def save_cache(raw: dict, metrics: dict[str, dict[str, float]]) -> None:
    cache = {
        "fetched_at": datetime.now().isoformat(),
        "source": "e-Stat法人企業統計",
        "stats_data_id": STATS_DATA_IDS["法人企業統計_年報"],
        "metrics_by_estat_code": metrics,
        "raw_summary": {
            "total_values": len(
                raw.get("GET_STATS_DATA", {})
                .get("STATISTICAL_DATA", {})
                .get("DATA_INF", {})
                .get("VALUE", [])
            )
        },
    }
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"  キャッシュ保存: {CACHE_PATH}")


def main() -> int:
    app_id = _get_app_id()
    if not app_id:
        print("[fetch_estat_industry] ESTAT_APP_ID 未設定 → キャッシュから更新を試みます")
        if update_from_cache():
            print("[fetch_estat_industry] キャッシュ反映完了")
        else:
            print("[fetch_estat_industry] キャッシュなし → スキップ")
        return 0

    print("[fetch_estat_industry] e-Stat API 業種別統計更新開始...")

    raw = fetch_corporate_enterprise_stats(app_id)
    if raw is None:
        print("[fetch_estat_industry] データ取得失敗 → スキップ（既存データを維持）")
        return 0

    metrics = _parse_metrics(raw)
    print(f"  取得した業種数: {len(metrics)}")

    save_cache(raw, metrics)

    if not metrics:
        print("[fetch_estat_industry] 解析可能な指標なし → benchmarks 更新スキップ")
        return 0

    changes = update_benchmarks(metrics)
    if changes:
        print(f"  industry_benchmarks.json 更新: {len(changes)} 業種")
        for k, diffs in changes.items():
            print(f"    {k}: {', '.join(diffs)}")
    else:
        print("  変更なし（既存値と同一）")

    print("[fetch_estat_industry] 完了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
