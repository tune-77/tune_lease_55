"""
業種別倒産率ベンチマーク（外部参照データ）

出典: TDB「倒産集計2025年度報」× 中小企業庁「中小企業の企業数・事業者数」2021年
対象: 負債1,000万円以上の法的整理ベース（休廃業・解散は含まない）
注意: 業種全体の母集団倒産率であり、個社の倒産確率（PD）ではない。
     TDB業種分類と中小企業庁産業分類は完全一致しないため概算値。
更新: 2026-05-23
"""

from __future__ import annotations

_BENCH: dict[str, dict] = {
    "卸売業": {
        "rate": 0.56,
        "per_10k": 56.2,
        "bankruptcies_2025": 1_138,
        "sme_count_2021": 202_432,
        "risk_level": "高",
        "risk_stars": 4,
        "note": "",
    },
    "建設業": {
        "rate": 0.48,
        "per_10k": 48.0,
        "bankruptcies_2025": 2_041,
        "sme_count_2021": 424_976,
        "risk_level": "中高",
        "risk_stars": 3,
        "note": "",
    },
    "小売業": {
        "rate": 0.42,
        "per_10k": 42.4,
        "bankruptcies_2025": 2_233,
        "sme_count_2021": 527_138,
        "risk_level": "中高",
        "risk_stars": 3,
        "note": "飲食料品小売は細分類で高リスク出現率39.8%",
    },
    "運輸・通信業": {
        "rate": 0.38,
        "per_10k": 38.1,
        "bankruptcies_2025": 457,
        "sme_count_2021": 120_060,
        "risk_level": "中",
        "risk_stars": 2,
        "note": "",
    },
    "製造業": {
        "rate": 0.35,
        "per_10k": 34.5,
        "bankruptcies_2025": 1_159,
        "sme_count_2021": 335_552,
        "risk_level": "中",
        "risk_stars": 2,
        "note": "出版・印刷業は細分類で高リスク出現率41.7%",
    },
    "サービス業": {
        "rate": 0.19,
        "per_10k": 19.2,
        "bankruptcies_2025": 2_677,
        "sme_count_2021": 1_396_616,
        "risk_level": "低",
        "risk_stars": 1,
        "note": "飲食店は細分類で高リスク出現率43.0%（最高水準）",
    },
    "不動産・物品賃貸業": {
        "rate": 0.10,
        "per_10k": 9.5,
        "bankruptcies_2025": 309,
        "sme_count_2021": 324_197,
        "risk_level": "低",
        "risk_stars": 1,
        "note": "",
    },
}

# 全業種単純平均（参考値）
OVERALL_AVG_RATE: float = round(
    sum(v["rate"] for v in _BENCH.values()) / len(_BENCH), 2
)  # ≈ 0.35

# industry_major ("D 建設業" 等) → _BENCH キーへのマッピング
_MAJOR_MAP: dict[str, str] = {
    "建設":     "建設業",
    "製造":     "製造業",
    "卸売":     "卸売業",
    "小売":     "小売業",
    "運輸":     "運輸・通信業",
    "情報通信": "サービス業",
    "宿泊":     "サービス業",
    "飲食":     "サービス業",
    "生活関連": "サービス業",
    "娯楽":     "サービス業",
    "教育":     "サービス業",
    "医療":     "サービス業",
    "福祉":     "サービス業",
    "学術":     "サービス業",
    "複合":     "サービス業",
    "不動産":   "不動産・物品賃貸業",
    "物品賃貸": "不動産・物品賃貸業",
    "金融":     "サービス業",
    "保険":     "サービス業",
    "農業":     "サービス業",
    "漁業":     "サービス業",
    "鉱業":     "製造業",
    "電気":     "サービス業",
    "ガス":     "サービス業",
    "水道":     "サービス業",
}


def get_bankruptcy_bench(industry_major: str) -> dict | None:
    """
    industry_major ("D 建設業" 等) から倒産率ベンチマークを返す。
    マッチしない場合は None。
    """
    if not industry_major:
        return None
    label = industry_major.strip()
    # "D 建設業" → "建設業" のようにプレフィックスを除去
    parts = label.split(" ", 1)
    name = parts[-1] if len(parts) > 1 else label

    # 直接キーマッチ
    if name in _BENCH:
        bench = dict(_BENCH[name])
        bench["matched_category"] = name
        return bench

    # キーワードマッチ
    for keyword, category in _MAJOR_MAP.items():
        if keyword in name or keyword in label:
            bench = dict(_BENCH[category])
            bench["matched_category"] = category
            return bench

    return None


def get_relative_risk(rate: float) -> str:
    """全業種平均との比較ラベルを返す。"""
    if rate >= OVERALL_AVG_RATE * 1.3:
        return "業界平均より高め"
    if rate <= OVERALL_AVG_RATE * 0.7:
        return "業界平均より低め"
    return "業界平均並み"
