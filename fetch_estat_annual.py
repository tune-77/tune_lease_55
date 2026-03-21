"""
e-Stat API から法人企業統計調査（年度版）のリース・設備投資関連データを取得し、
data/industry_capex_lease.json を生成するスクリプト。

使い方:
    python3 fetch_estat_annual.py

取得先: 法人企業統計調査（年度版）statsDataId=0003060791
        金融業・保険業以外の業種（年度原数値）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
★ リースシステムでの活用方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. lease_burden_rate（リース料負担率）= 動産・不動産賃借料 ÷ 売上高 × 100 (%)
   → 審査先の「リース料/売上高」が業種平均と比べて
     大幅に上回る場合は過剰レバレッジのリスクを示す。
   → 申込リース料を審査先の売上で割った値と業種平均を比較するスコアリングに使用。

2. capex_to_sales（設備投資率）= ソフトウェア除く設備投資 ÷ 売上高 × 100 (%)
   → 設備投資依存度が高い業種ほど装置産業→リース需要が高い。
   → 業種ごとの「リース需要スコア」の根拠として活用。

3. depreciation_to_sales（減価償却率）= 減価償却費 ÷ 売上高 × 100 (%)
   → 有形固定資産の回転スピードを示す。
   → 短期リースや機器の陳腐化リスク評価に利用。

4. lease_to_capex（リース/設備投資比率）= 動産・不動産賃借料 ÷ ソフトウェア除く設備投資 × 100 (%)
   → 業種として設備投資の何割をリースで賄っているかの目安。
   → 高いほど当該業種ではリースが主流 → 回収リスク低。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import urllib.request
import urllib.parse
import json
import os
from datetime import datetime

# ─── 設定 ─────────────────────────────────────────────────────────────────
APP_ID = "5d55cf528a66dc1ded12484f09cfe9e62a1522c7"
STATS_DATA_ID = "0003060791"   # 法人企業統計調査（年度版）金融業・保険業以外

# 取得する調査項目コード
TARGET_ITEMS = {
    "045": "sales",            # 売上高（分母）
    "062": "depreciation",     # 減価償却費
    "069": "lease_rent",       # 動産・不動産賃借料 ← リース料の業種平均
    "086": "capex",            # ソフトウェアを除く設備投資
    "051": "ordinary_profit",  # 経常利益（参考）
    "122": "equity_ratio",     # 自己資本比率
    "125": "roe",              # 自己資本経常利益率
    "126": "op_margin",        # 売上高営業利益率
    "127": "ord_income_margin",# 売上高経常利益率
    "136": "depreciation_rate",# 減価償却率（財務比率）
    "143": "capex_efficiency", # 設備投資効率
}

# e-Stat 業種コード（年度版0003060791）→ システムの JSIC 中分類名 マッピング
ESTAT_ANNUAL_TO_JSIC = {
    # 農林水産業
    "105": [
        "01 農業",
        "02 林業",
        "03 漁業",
        "04 水産養殖業",
    ],
    # 建設業
    "107": [
        "06 総合工事業",
        "07 職別工事業(大工・とび等)",
        "08 設備工事業(電気・管工事)",
    ],
    # 製造業（全体）
    "108": [
        "29 その他の製造業",  # デフォルト製造業（個別コードなし場合）
    ],
    # 食料品製造業
    "109": [
        "09 食料品製造業",
        "10 飲料・飼料製造業",
    ],
    # 繊維工業
    "110": [
        "11 繊維工業",
    ],
    # 木材・木製品
    "112": [
        "12 木材・木製品製造業",
        "13 家具・装備品製造業",
    ],
    # パルプ・紙
    "113": [
        "14 パルプ・紙製造業",
    ],
    # 印刷
    "114": [
        "15 印刷業",
    ],
    # 化学
    "115": [
        "16 化学工業",
        "17 プラスチック製品製造業",
        "18 ゴム製品製造業",
    ],
    # 窯業・土石
    "117": [
        "19 窯業・土石製品製造業",
    ],
    # 鉄鋼
    "118": [
        "20 鉄鋼業",
    ],
    # 非鉄金属
    "119": [
        "21 非鉄金属製造業",
    ],
    # 金属製品
    "120": [
        "22 金属製品製造業",
    ],
    # 生産用機械
    "121": [
        "23 一般機械器具製造業",
    ],
    # 電気機械
    "122": [
        "24 電気機械器具製造業",
    ],
    # 情報通信機械
    "145": [
        "25 情報通信機械器具製造業",
        "26 電子部品・デバイス製造業",
    ],
    # 輸送用機械
    "146": [
        "27 輸送用機械器具製造業",
    ],
    # 精密・その他製造業
    "124": [
        "28 精密機械器具製造業",
    ],
    # 電気業
    "135": [
        "33 電気業",
    ],
    # ガス・熱供給・水道
    "136": [
        "34 ガス業",
        "35 熱供給業",
        "36 水道業",
    ],
    # 情報通信業
    "142": [
        "37 通信業",
        "38 放送業",
        "39 情報サービス業",
        "40 インターネット附随サービス業",
        "41 映像・音声・文字情報制作業",
    ],
    # 陸運業（運輸業全体 134 の一部）
    "131": [
        "41 道路貨物運送業",
        "42 道路旅客運送業",
        "43 鉄道業",
    ],
    # 運輸業全体（水運・航空・倉庫含む）
    "134": [
        "44 水運業",
        "45 航空運輸業",
        "46 倉庫業",
        "47 運輸に附帯するサービス業",
    ],
    # 卸売業
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
    # 小売業
    "128": [
        "58 各種商品小売業",
        "59 織物・衣服・身回品小売業",
        "60 飲食料品小売業",
        "61 機械器具小売業",
        "62 医薬品・化粧品小売業",
        "63 その他の小売業",
        "64 無店舗小売業",
    ],
    # 不動産業
    "130": [
        "68 不動産取引業",
        "69 不動産賃貸業・管理業",
    ],
    # 物品賃貸業（リース業含む）
    "149": [
        "70 各種物品賃貸業",
        "71 産業用機械器具賃貸業",
        "72 事務用機械器具賃貸業",
        "73 自動車賃貸業",
        "74 スポーツ・娯楽用品賃貸業",
        "75 その他の物品賃貸業",
    ],
    # 宿泊・飲食
    "156": [
        "75 宿泊業",
        "76 飲食店",
        "77 持ち帰り・配達飲食サービス業",
    ],
    # 医療・福祉
    "152": [
        "83 医療業",
        "84 保健衛生",
        "85 社会保険・社会福祉・介護事業",
    ],
    # サービス業（その他）
    "137": [
        "75 廃棄物処理業",
        "76 自動車整備業",
        "77 機械等修理業",
        "78 職業紹介・労働者派遣業",
        "79 その他の事業サービス業",
        "82 その他のサービス業",
    ],
}

# 規模コード: 19=中小企業（1千万以上1億未満）, 26=全規模
SIZE_CODE = "19"
SIZE_CODE_ALL = "26"

# 最新年度
LATEST_YEAR = "20240"  # 2024年度
PREV_YEAR = "20230"    # 2023年度（フォールバック）

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "industry_capex_lease.json")


def fetch_annual(item_codes: list[str], industry_codes: list[str],
                 size_code: str, time_code: str) -> dict:
    """
    e-Stat API から年度データを取得。
    返却: {業種コード: {指標コード: 値（百万円 or %）}}
    """
    base = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
    params = {
        "appId": APP_ID,
        "statsDataId": STATS_DATA_ID,
        "cdCat01": ",".join(item_codes),
        "cdCat02": ",".join(industry_codes),
        "cdCat03": size_code,
        "cdTime": time_code,
        "metaGetFlg": "N",
        "cntGetFlg": "N",
        "lang": "J",
    }
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as res:
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

    # {業種コード: {指標コード: 値}} に変換
    result: dict[str, dict[str, float]] = {}
    for v in values:
        item_code = v.get("@cat01", "")
        industry_code = v.get("@cat02", "")
        raw = v.get("$", "")
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if industry_code not in result:
            result[industry_code] = {}
        result[industry_code][item_code] = val

    return result


def build_capex_lease_data() -> dict:
    """全業種の設備投資・リース料データを取得して辞書を構築。"""
    industry_codes = list(ESTAT_ANNUAL_TO_JSIC.keys())
    item_codes = list(TARGET_ITEMS.keys())

    print(f"e-Stat 年度版から {len(item_codes)} 項目 × {len(industry_codes)} 業種を取得中...")

    # まず中小企業規模で最新年度
    raw = fetch_annual(item_codes, industry_codes, SIZE_CODE, LATEST_YEAR)

    if not raw:
        print("  最新年度データなし。前年度にフォールバック...")
        raw = fetch_annual(item_codes, industry_codes, SIZE_CODE, PREV_YEAR)

    if not raw:
        print("  中小規模データなし。全規模で取得...")
        raw = fetch_annual(item_codes, industry_codes, SIZE_CODE_ALL, LATEST_YEAR)

    if not raw:
        print("  [WARN] データ取得失敗。終了します。")
        return {}

    print(f"  取得業種数: {len(raw)}")

    benchmarks: dict[str, dict] = {}
    for estat_code, jsic_names in ESTAT_ANNUAL_TO_JSIC.items():
        if estat_code not in raw:
            print(f"  [SKIP] e-Stat業種コード {estat_code} のデータなし")
            continue

        ind = raw[estat_code]

        # 基本数値（百万円）
        sales         = ind.get("045", None)
        depreciation  = ind.get("062", None)
        lease_rent    = ind.get("069", None)
        capex         = ind.get("086", None)

        # 財務比率（%）
        op_margin          = ind.get("126", None)
        ord_income_margin  = ind.get("127", None)
        equity_ratio       = ind.get("122", None)
        roe                = ind.get("125", None)
        depr_rate          = ind.get("136", None)  # 減価償却率（財務比率）
        capex_eff          = ind.get("143", None)  # 設備投資効率

        # 計算指標（売上高がないと計算不能）
        entry: dict = {}

        # ── 金額データ（百万円）──
        if sales is not None:
            entry["sales_million"] = round(sales, 0)
        if lease_rent is not None:
            entry["lease_rent_million"] = round(lease_rent, 0)
        if capex is not None:
            entry["capex_million"] = round(capex, 0)
        if depreciation is not None:
            entry["depreciation_million"] = round(depreciation, 0)

        # ── 計算比率（%）──
        if sales and sales > 0:
            if lease_rent is not None:
                entry["lease_burden_rate"] = round(lease_rent / sales * 100, 3)
            if capex is not None:
                entry["capex_to_sales"] = round(capex / sales * 100, 3)
            if depreciation is not None:
                entry["depreciation_to_sales"] = round(depreciation / sales * 100, 3)

        if capex and capex > 0 and lease_rent is not None:
            entry["lease_to_capex"] = round(lease_rent / capex * 100, 1)

        # ── 財務比率（e-Stat 計算済み）──
        for field, code in [
            ("op_margin", "126"),
            ("ord_income_margin", "127"),
            ("equity_ratio", "122"),
            ("roe", "125"),
            ("depreciation_rate", "136"),
            ("capex_efficiency", "143"),
        ]:
            val = ind.get(code)
            if val is not None:
                entry[field] = round(val, 2)

        if not entry:
            continue

        # コメント生成
        lb = entry.get("lease_burden_rate")
        cs = entry.get("capex_to_sales")
        lb_str = f"{lb:.2f}%" if lb is not None else "N/A"
        cs_str = f"{cs:.2f}%" if cs is not None else "N/A"
        entry["comment"] = (
            f"e-Stat法人企業統計年度版（{datetime.now().strftime('%Y-%m')}取得）。"
            f"業種平均 リース料/売上高={lb_str}・設備投資率={cs_str}。"
        )
        entry["source"] = "e-stat-0003060791"
        entry["updated"] = datetime.now().strftime("%Y-%m-%d")
        entry["time_period"] = LATEST_YEAR

        for jsic_name in jsic_names:
            benchmarks[jsic_name] = entry.copy()

    return benchmarks


def merge_with_existing(new_data: dict) -> dict:
    """既存 JSON とマージ。新データ優先。"""
    existing: dict = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
    merged = dict(existing)
    for k, v in new_data.items():
        merged[k] = v
    return merged


def print_usage_guide(benchmarks: dict) -> None:
    """リースシステムでの活用方法を表示。"""
    print("\n" + "=" * 60)
    print("★ リースシステムでの活用方法")
    print("=" * 60)
    print("""
■ 活用場面①：リース料適正性チェック（申込審査時）
  申込リース料 ÷ 審査先売上高 = 申込リース料負担率
  → 業種平均 lease_burden_rate と比較
  → 申込率が業種平均の2倍超 → リスク警告
  → 申込率が業種平均以下   → 問題なし

■ 活用場面②：業種別設備投資需要スコア
  capex_to_sales が高い業種 → 装置産業 → リース需要が高い
  lease_to_capex が高い業種 → リースが主流 → 回収リスク低
  → スコアリングの加点要素として使用

■ 活用場面③：業種別キャッシュフロー余力推定
  (depreciation_to_sales + op_margin) ≈ EBITDA/売上高
  → リース料返済余力の業種標準値として参照

■ 活用場面④：財務比率の二重チェック
  同一の e-Stat 年度版から equity_ratio, op_margin を取得。
  四半期版(0003060191)と年度版で数値が若干異なる場合は
  年度版を正式値として採用することを検討。

■ ファイル出力先: data/industry_capex_lease.json
  業種キーは既存 industry_benchmarks.json と同じ JSIC 名。
  score_calculation.py から同名キーで参照可能。
""")

    # サンプル表示
    sample_keys = list(benchmarks.keys())[:5]
    print("─── 取得サンプル ───")
    for k in sample_keys:
        d = benchmarks[k]
        lb = d.get("lease_burden_rate", "N/A")
        cs = d.get("capex_to_sales", "N/A")
        lc = d.get("lease_to_capex", "N/A")
        op = d.get("op_margin", "N/A")
        print(f"  {k}")
        print(f"    リース料負担率: {lb}%  設備投資率: {cs}%  "
              f"リース/設備投資: {lc}%  営業利益率: {op}%")


def main():
    print("=" * 60)
    print("e-Stat 年度版ベンチマーク取得スクリプト")
    print(f"statsDataId : {STATS_DATA_ID}")
    print(f"出力先      : {OUTPUT_PATH}")
    print("=" * 60)

    new_data = build_capex_lease_data()
    if not new_data:
        print("取得データなし。終了します。")
        return

    merged = merge_with_existing(new_data)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(new_data)} 業種を更新（全{len(merged)}業種）")
    print(f"保存先: {OUTPUT_PATH}")

    print_usage_guide(new_data)


if __name__ == "__main__":
    main()
