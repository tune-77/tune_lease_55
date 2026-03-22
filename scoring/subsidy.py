# scoring/subsidy.py
# 補助金マスタ管理・マッチングロジック

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
DB_PATH = os.path.join(_REPO_ROOT, "data", "lease_data.db")

# 初期補助金マスタデータ
_INITIAL_SUBSIDIES: list[dict[str, Any]] = [
    {
        "id": "monodukuri",
        "name": "ものづくり補助金",
        "category": "設備投資",
        "max_amount": 1250,
        "rate": 0.5,
        "industries": ["製造", "加工"],
        "asset_keywords": ["機械", "工作機械", "設備", "製造設備", "加工機"],
        "regions": ["全国"],
        "certainty": "high",
        "deadline": "通年",
        "url": "https://portal.monodukuri-hojo.jp/",
        "note": "中小企業・小規模事業者の革新的サービス開発・試作品開発・生産プロセス改善を支援",
        "active": 1,
    },
    {
        "id": "it_donyu",
        "name": "IT導入補助金",
        "category": "IT導入",
        "max_amount": 450,
        "rate": 0.5,
        "industries": [],
        "asset_keywords": ["IT", "ソフトウェア", "システム", "ツール", "クラウド", "SaaS"],
        "regions": ["全国"],
        "certainty": "high",
        "deadline": "通年",
        "url": "https://www.it-hojo.jp/",
        "note": "中小企業・小規模事業者のITツール導入を支援。A類型・B類型あり",
        "active": 1,
    },
    {
        "id": "shoene_sii",
        "name": "省エネ補助金 SII",
        "category": "省エネ",
        "max_amount": 10000,
        "rate": 0.3333,
        "industries": [],
        "asset_keywords": ["省エネ", "空調", "照明", "LED", "冷凍冷蔵", "ボイラー", "コンプレッサー", "変圧器"],
        "regions": ["全国"],
        "certainty": "medium",
        "deadline": "通年",
        "url": "https://sii.or.jp/",
        "note": "省エネルギー投資促進支援事業。設備更新の1/3補助（最大1億円）",
        "active": 1,
    },
    {
        "id": "jigyou_saikouchiku",
        "name": "事業再構築補助金",
        "category": "設備投資",
        "max_amount": 1500,
        "rate": 0.6667,
        "industries": [],
        "asset_keywords": ["設備", "機械", "建屋", "内装", "改装", "新事業"],
        "regions": ["全国"],
        "certainty": "medium",
        "deadline": "通年",
        "url": "https://jigyou-saikouchiku.go.jp/",
        "note": "コロナ禍からの事業再構築を支援。通常枠・大規模賃金引上枠等あり",
        "active": 1,
    },
    {
        "id": "jizokuka",
        "name": "小規模事業者持続化補助金",
        "category": "設備投資",
        "max_amount": 250,
        "rate": 0.6667,
        "industries": [],
        "asset_keywords": ["設備", "機械", "広告", "展示会", "店舗", "改装"],
        "regions": ["全国"],
        "certainty": "high",
        "deadline": "通年",
        "url": "https://r3.jizokukahojokin.info/",
        "note": "小規模事業者の販路開拓・業務効率化を支援。商工会議所・商工会が窓口",
        "active": 1,
    },
]


def _get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_subsidy_master() -> None:
    """初期補助金データをDBに投入（既存データがあればスキップ）。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subsidies")
        count = cursor.fetchone()[0]
        if count > 0:
            return  # 既存データあり → スキップ

        for s in _INITIAL_SUBSIDIES:
            cursor.execute(
                """
                INSERT OR IGNORE INTO subsidies
                    (id, name, category, max_amount, rate,
                     industries, asset_keywords, regions,
                     certainty, deadline, url, note, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s["id"],
                    s["name"],
                    s["category"],
                    s["max_amount"],
                    s["rate"],
                    json.dumps(s["industries"], ensure_ascii=False),
                    json.dumps(s["asset_keywords"], ensure_ascii=False),
                    json.dumps(s["regions"], ensure_ascii=False),
                    s["certainty"],
                    s["deadline"],
                    s["url"],
                    s["note"],
                    s["active"],
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: tuple, cursor: sqlite3.Cursor) -> dict:
    cols = [d[0] for d in cursor.description]
    d = dict(zip(cols, row))
    for key in ("industries", "asset_keywords", "regions"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = []
    return d


def match_subsidies(
    industry_sub: str,
    asset_name: str,
    region: str = "全国",
) -> list[dict]:
    """
    業種・設備名・地域から利用可能な補助金を返す。

    Returns:
        [{"id": ..., "name": ..., "max_amount": ..., "rate": ..., "certainty": ..., ...}]
        補助金効果（max_amount * rate）の降順でソートされる。
    """
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM subsidies WHERE active = 1")
        rows = cursor.fetchall()
        all_subsidies = [_row_to_dict(r, cursor) for r in rows]
    finally:
        conn.close()

    matched: list[dict] = []
    industry_lower = industry_sub.lower()
    asset_lower = asset_name.lower()
    region_lower = region.lower()

    for s in all_subsidies:
        # 地域マッチング: "全国" は常にマッチ
        regions: list[str] = s.get("regions") or []
        region_ok = any(
            r == "全国" or r.lower() in region_lower or region_lower in r.lower()
            for r in regions
        ) if regions else True

        if not region_ok:
            continue

        # 業種マッチング: industries が空リストなら全業種対象
        industries: list[str] = s.get("industries") or []
        if industries:
            industry_ok = any(kw.lower() in industry_lower for kw in industries)
        else:
            industry_ok = True

        # 設備キーワードマッチング: asset_keywords が空リストなら全設備対象
        asset_keywords: list[str] = s.get("asset_keywords") or []
        if asset_keywords:
            asset_ok = any(kw.lower() in asset_lower for kw in asset_keywords)
        else:
            asset_ok = True

        if industry_ok and asset_ok:
            matched.append(s)

    # 補助金効果（max_amount * rate）の降順でソート
    matched.sort(
        key=lambda x: (x.get("max_amount") or 0) * (x.get("rate") or 0),
        reverse=True,
    )
    return matched


def calc_subsidy_effect(
    lease_amount_man: float,
    lease_term_months: int,
    subsidies: list[dict],
) -> dict:
    """
    補助金効果を計算する。

    Args:
        lease_amount_man: リース申込額（万円）
        lease_term_months: リース期間（ヶ月）
        subsidies: match_subsidies の戻り値

    Returns:
        {
            "best_subsidy": dict,           最も効果的な補助金
            "subsidy_amount": float,        補助金額（万円）
            "monthly_reduction": float,     月額軽減額（万円）
            "effective_monthly": float,     補助金適用後の実質月額（万円）
            "score_bonus": float,           スコア加点（0〜5点）
        }
    """
    no_effect: dict = {
        "best_subsidy": {},
        "subsidy_amount": 0.0,
        "monthly_reduction": 0.0,
        "effective_monthly": round(lease_amount_man / max(lease_term_months, 1), 4),
        "score_bonus": 0.0,
    }

    if not subsidies or lease_amount_man <= 0 or lease_term_months <= 0:
        return no_effect

    # 最も効果的な補助金（match_subsidies は既に降順ソート済み）
    best = subsidies[0]
    max_amount: float = float(best.get("max_amount") or 0)
    rate: float = float(best.get("rate") or 0)

    # 補助金額 = min(申込額 × 補助率, 最大補助額)
    subsidy_amount = min(lease_amount_man * rate, max_amount)
    monthly_reduction = subsidy_amount / lease_term_months
    base_monthly = lease_amount_man / lease_term_months
    effective_monthly = max(0.0, base_monthly - monthly_reduction)

    # スコア加点: 補助率・確度・補助額規模に応じて 0〜5 点
    certainty_bonus = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(
        best.get("certainty", "low"), 0.0
    )
    coverage_ratio = subsidy_amount / lease_amount_man if lease_amount_man > 0 else 0.0
    score_bonus = round(min(5.0, coverage_ratio * 4.0 + certainty_bonus), 2)

    return {
        "best_subsidy": best,
        "subsidy_amount": round(subsidy_amount, 2),
        "monthly_reduction": round(monthly_reduction, 4),
        "effective_monthly": round(effective_monthly, 4),
        "score_bonus": score_bonus,
    }


def get_all_subsidies(active_only: bool = True) -> list[dict]:
    """全補助金マスタを返す。"""
    conn = _get_conn()
    try:
        cursor = conn.cursor()
        if active_only:
            cursor.execute("SELECT * FROM subsidies WHERE active = 1 ORDER BY max_amount DESC")
        else:
            cursor.execute("SELECT * FROM subsidies ORDER BY max_amount DESC")
        rows = cursor.fetchall()
        return [_row_to_dict(r, cursor) for r in rows]
    finally:
        conn.close()
