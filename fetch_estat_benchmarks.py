"""
e-Stat API から法人企業統計調査の業種別財務指標を取得し、
data/industry_benchmarks.json を更新するスクリプト。

使い方:
    python3 fetch_estat_benchmarks.py

取得先: 法人企業統計調査（金融業・保険業以外）statsDataId=0003060191
AppID: 5d55cf528a66dc1ded12484f09cfe9e62a1522c7
"""

import urllib.request
import urllib.parse
import json
import os
import time
from datetime import datetime

# ─── 設定 ───────────────────────────────────────────────────────────────
APP_ID = "5d55cf528a66dc1ded12484f09cfe9e62a1522c7"
STATS_DATA_ID = "0003060191"   # 時系列データ 金融業、保険業以外の業種（原数値）

# リース審査に使う財務比率の調査項目コード
TARGET_INDICATORS = {
    "136": "op_margin",          # 売上高営業利益率
    "154": "ord_income_margin",  # 売上高経常利益率
    "143": "equity_ratio",       # 自己資本比率
    "144": "current_ratio",      # 流動比率
    "151": "debt_ratio",         # 負債比率
    "140": "roa",                # 総資本営業利益率（ROA代替）
    "153": "roa_ordinary",       # 総資本経常利益率
    "141": "asset_turnover",     # 総資本回転率
    "152": "quick_ratio",        # 当座比率
    "148": "interest_burden",    # 売上高利子率
}

# e-Stat 業種コード → システムの JSIC 中分類名 マッピング
# ※ e-Stat は大分類レベル。系統が合う中分類グループに適用する。
ESTAT_TO_JSIC_MAP = {
    # 建設業 (107)
    "107": [
        "06 総合工事業",
        "07 職別工事業(大工・とび等)",
        "08 設備工事業(電気・管工事)",
    ],
    # 製造業 (108)
    "108": [
        "09 食料品製造業",
        "10 飲料・飼料製造業",
        "11 繊維工業",
        "12 木材・木製品製造業",
        "13 家具・装備品製造業",
        "14 パルプ・紙製造業",
        "15 印刷業",
        "16 化学工業",
        "17 プラスチック製品製造業",
        "18 ゴム製品製造業",
        "19 窯業・土石製品製造業",
        "20 鉄鋼業",
        "21 非鉄金属製造業",
        "22 金属製品製造業",
        "23 一般機械器具製造業",
        "24 電気機械器具製造業",
        "25 情報通信機械器具製造業",
        "26 電子部品・デバイス製造業",
        "27 輸送用機械器具製造業",
        "28 精密機械器具製造業",
        "29 その他の製造業",
    ],
    # 食料品製造業 (109) - より詳細な上書き用
    "109": ["09 食料品製造業", "10 飲料・飼料製造業"],
    # 電気機械 (122)
    "122": ["24 電気機械器具製造業"],
    # 情報通信機械 (145)
    "145": ["25 情報通信機械器具製造業", "26 電子部品・デバイス製造業"],
    # 輸送用機械 (146)
    "146": ["27 輸送用機械器具製造業"],
    # 生産用機械 (121)
    "121": ["23 一般機械器具製造業"],
    # 卸売業 (127)
    "127": [
        "50 各種商品卸売業",
        "51 繊維品卸売業",
        "52 衣服・身回品卸売業",
        "53 農畜産物卸売業",
        "54 食料品卸売業",
        "55 建築材料卸売業",
        "56 化学製品卸売業",
        "57 機械器具卸売業",
        "58 その他の卸売業",
    ],
    # 小売業 (128)
    "128": [
        "58 各種商品小売業",
        "59 織物・衣服・身回品小売業",
        "60 飲食料品小売業",
        "61 機械器具小売業",
        "62 医薬品・化粧品小売業",
        "63 その他の小売業",
        "64 無店舗小売業",
    ],
    # 陸運業 (131)
    "131": [
        "41 道路貨物運送業",
        "42 道路旅客運送業",
        "43 鉄道業",
    ],
    # 情報通信業 (142)
    "142": [
        "37 通信業",
        "38 放送業",
        "39 情報サービス業",
        "40 インターネット附随サービス業",
        "41 映像・音声・文字情報制作業",
    ],
    # 不動産業 (130)
    "130": [
        "68 不動産取引業",
        "69 不動産賃貸業・管理業",
    ],
    # 物品賃貸業＝リース業 (149)
    "149": [
        "70 各種物品賃貸業",
        "71 産業用機械器具賃貸業",
        "72 事務用機械器具賃貸業",
        "73 自動車賃貸業",
        "74 スポーツ・娯楽用品賃貸業",
        "75 その他の物品賃貸業",
    ],
    # サービス業 (137)
    "137": [
        "75 廃棄物処理業",
        "76 自動車整備業",
        "77 機械等修理業",
        "78 職業紹介・労働者派遣業",
        "79 その他の事業サービス業",
        "80 政治・経済・文化団体",
        "81 宗教",
        "82 その他のサービス業",
        "83 外国公務",
    ],
    # 宿泊・飲食 (156)
    "156": [
        "75 宿泊業",
        "76 飲食店",
        "77 持ち帰り・配達飲食サービス業",
    ],
    # 医療・福祉 (152)
    "152": [
        "83 医療業",
        "84 保健衛生",
        "85 社会保険・社会福祉・介護事業",
    ],
    # 農林水産業 (105)
    "105": [
        "01 農業",
        "02 林業",
        "03 漁業",
        "04 水産養殖業",
    ],
    # 電気・ガス (135/136)
    "135": ["33 電気業"],
    "136": ["34 ガス業", "35 熱供給業", "36 水道業"],
    # 運輸業計 (134)
    "134": [
        "41 道路貨物運送業",
        "42 道路旅客運送業",
        "43 鉄道業",
        "44 水運業",
        "45 航空運輸業",
        "46 倉庫業",
        "47 運輸に附帯するサービス業",
    ],
}

# 規模コード（中小企業重視）
# 19 = 1千万円以上1億円未満, 26 = 全規模
SIZE_CODE = "19"   # 中小企業規模を基準にする
SIZE_CODE_ALL = "26"

# 取得する最新期（2025年10-12月）
LATEST_TIME = "20254"
# 一期前（2025年7-9月）フォールバック
PREV_TIME = "20253"

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "industry_benchmarks.json")


def fetch_stats(cat01_codes: list[str], cat02_codes: list[str],
                cat03_code: str, time_code: str) -> dict:
    """e-Stat API からデータを取得して {業種コード: {指標コード: 値}} を返す。"""
    base = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
    params = {
        "appId": APP_ID,
        "statsDataId": STATS_DATA_ID,
        "cdCat01": ",".join(cat01_codes),
        "cdCat02": ",".join(cat02_codes),
        "cdCat03": cat03_code,
        "cdTime": time_code,
        "metaGetFlg": "N",
        "cntGetFlg": "N",
        "lang": "J",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as res:
            data = json.loads(res.read())
    except Exception as e:
        print(f"  [ERROR] fetch failed: {e}")
        return {}

    values = (data.get("GET_STATS_DATA", {})
                  .get("STATISTICAL_DATA", {})
                  .get("DATA_INF", {})
                  .get("VALUE", []))
    if isinstance(values, dict):
        values = [values]

    result: dict[str, dict[str, float]] = {}
    for v in values:
        ind_code = v.get("@cat01", "")
        ind_name_code = v.get("@cat02", "")
        raw = v.get("$", "")
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if ind_name_code not in result:
            result[ind_name_code] = {}
        result[ind_name_code][ind_code] = val

    return result


def build_benchmarks() -> dict:
    """全業種・全指標を取得してベンチマーク辞書を構築する。"""
    industry_codes = list(ESTAT_TO_JSIC_MAP.keys())
    indicator_codes = list(TARGET_INDICATORS.keys())

    print(f"e-Stat API から {len(indicator_codes)} 指標 × {len(industry_codes)} 業種を取得中...")

    # 一括取得（上限に注意: 100,000件/リクエスト）
    raw = fetch_stats(indicator_codes, industry_codes, SIZE_CODE, LATEST_TIME)

    if not raw:
        print("  最新期データなし。一期前にフォールバック...")
        raw = fetch_stats(indicator_codes, industry_codes, SIZE_CODE, PREV_TIME)

    if not raw:
        print("  中小企業規模データなし。全規模で再取得...")
        raw = fetch_stats(indicator_codes, industry_codes, SIZE_CODE_ALL, LATEST_TIME)

    if not raw:
        print("  [WARN] データを取得できませんでした。既存ファイルを維持します。")
        return {}

    print(f"  取得業種数: {len(raw)}")

    # JSIC 中分類へ展開
    benchmarks: dict[str, dict] = {}
    for estat_code, jsic_names in ESTAT_TO_JSIC_MAP.items():
        if estat_code not in raw:
            print(f"  [SKIP] e-Stat業種コード {estat_code} のデータなし")
            continue

        indicator_data = raw[estat_code]
        bench_entry: dict = {}
        for ind_code, field_name in TARGET_INDICATORS.items():
            if ind_code in indicator_data:
                bench_entry[field_name] = round(indicator_data[ind_code], 2)

        if not bench_entry:
            continue

        # op_marginをコメント生成に使用
        op_m = bench_entry.get("op_margin", 0.0)
        eq_r = bench_entry.get("equity_ratio", 0.0)
        bench_entry["comment"] = (
            f"e-Stat法人企業統計（{datetime.now().strftime('%Y-%m')}取得）。"
            f"業種平均 営業利益率{op_m:.1f}%・自己資本比率{eq_r:.1f}%。"
        )
        bench_entry["source"] = "e-stat"
        bench_entry["updated"] = datetime.now().strftime("%Y-%m-%d")

        for jsic_name in jsic_names:
            # 既存コメントがあれば維持、なければ上書き
            benchmarks[jsic_name] = bench_entry.copy()

    return benchmarks


def merge_with_existing(new_data: dict) -> dict:
    """既存 JSON と新データをマージ。新データ優先、既存のみ項目は維持。"""
    existing: dict = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    merged = dict(existing)
    for jsic_name, bench in new_data.items():
        merged[jsic_name] = bench

    return merged


def main():
    print("=" * 60)
    print("e-Stat ベンチマーク取得スクリプト")
    print(f"statsDataId: {STATS_DATA_ID}")
    print(f"出力先: {OUTPUT_PATH}")
    print("=" * 60)

    new_benchmarks = build_benchmarks()
    if not new_benchmarks:
        print("取得データなし。終了します。")
        return

    merged = merge_with_existing(new_benchmarks)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(new_benchmarks)} 業種を更新（全{len(merged)}業種）")
    print(f"保存先: {OUTPUT_PATH}")

    # サンプル表示
    sample_keys = list(new_benchmarks.keys())[:3]
    print("\n--- 取得サンプル ---")
    for k in sample_keys:
        print(f"{k}:")
        for field, val in new_benchmarks[k].items():
            if field != "comment":
                print(f"  {field}: {val}")
        print()


if __name__ == "__main__":
    main()
