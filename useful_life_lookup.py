"""
法定耐用年数ルックアップ（別表第二優先・別表第一フォールバック）

減価償却資産の耐用年数等に関する省令（昭和40年大蔵省令第15号）に基づく。

優先順位:
  1. 別表第二（業種別設備）: industry_sub が特定でき、かつ useful_life_by_industry.json に
     asset × industry のマッチがある場合
  2. 別表第一（物件種別）: マッチしない場合は FALLBACK_MAPPING へフォールバック
"""
from __future__ import annotations

import json
import os

_DIR = os.path.dirname(os.path.abspath(__file__))
_BY_INDUSTRY_PATH = os.path.join(_DIR, "static_data", "useful_life_by_industry.json")

# モジュールレベルキャッシュ（複数呼び出しでファイル読み込みを繰り返さない）
_industry_entries_cache: list[dict] | None = None


def _load_industry_entries() -> list[dict]:
    global _industry_entries_cache
    if _industry_entries_cache is None:
        try:
            with open(_BY_INDUSTRY_PATH, encoding="utf-8") as f:
                _industry_entries_cache = json.load(f).get("entries", [])
        except Exception:
            _industry_entries_cache = []
    return _industry_entries_cache


# 別表第一ベースのフォールバックマッピング（業種不明または非対象品目）
# キー: 物件名に含まれるキーワード群 / 値: 耐用年数（年）
_FALLBACK_MAPPING: list[tuple[list[str], int, str]] = [
    # (キーワードリスト, 耐用年数, 法令根拠)
    # クレーン系をトラックより先に置く（「トラッククレーン」が「トラック」5年に誤マッチするのを防ぐ）
    (["クレーン"],
     8, "別表第二55号・自走式作業用機械設備（据置型は業種別別表第二を優先）"),
    (["ブルドーザー", "建設機械", "重機"],
     8, "別表第二55号・ブルドーザー、パワーショベルその他の自走式作業用機械設備"),
    (["ショベル", "バックホウ", "ユンボ"],
     8, "別表第二55号・パワーショベルその他の自走式作業用機械設備"),
    (["フォークリフト"],
     4, "別表第一・車両及び運搬具・前掲のもの以外のもの・フォークリフト"),
    (["トラック", "スーパーグレード", "プロフィア", "キャリイ", "elf", "エルフ",
      "キャンター", "デュトロ", "ダイナ", "レンジャー"],
     5, "別表第一・車両及び運搬具・貨物自動車（その他）"),
    (["医療", "介護", "mri", "ct", "レントゲン", "透析"],
     6, "別表第一・器具及び備品・医療機器・レントゲン等（固定式）その他"),
    (["複合機", "コピー機", "oa", "プリンタ"],
     5, "別表第一・器具及び備品・事務機器・複写機等"),
    (["防犯カメラ", "監視カメラ", "セキュリティカメラ"],
     6, "別表第一・器具及び備品（参考値）"),
    (["pc", "パソコン", "ソフト", "サーバ", "ネットワーク",
      "タブレット", "スイッチ", "ルータ"],
     4, "別表第一・器具及び備品・電子計算機（パソコン）"),
    (["工作機械", "旋盤", "マシニング", "プレス", "射出成形",
      "製造機", "溶接機"],
     10, "別表第二16-18号・製造業用設備（金属加工機械等）"),
]

_DEFAULT_YEARS = 7  # 上記どれにもマッチしない場合


def get_legal_useful_life(asset_type: str, industry_sub: str = "") -> int:
    """
    物件種別・借主業種から法定耐用年数（年）を返す。

    Args:
        asset_type: 物件名または物件種別文字列（部分一致で判定）
        industry_sub: 借主の業種小分類（例: "44 道路貨物運送業"）。
                      空文字の場合は別表第一フォールバックのみ。

    Returns:
        法定耐用年数（年）。マッチしない場合は 7 を返す。

    優先順位:
        別表第二（業種別設備） > 別表第一（物件種別）
    """
    asset_lower = (asset_type or "").lower()
    industry_lower = (industry_sub or "").lower()

    # ── 別表第二：業種 × 物件のマッチング ──────────────────────────
    if industry_lower:
        for entry in _load_industry_entries():
            ind_kws: list[str] = entry.get("industry_keywords", [])
            asset_kws: list[str] = entry.get("asset_keywords", [])
            if any(kw.lower() in industry_lower for kw in ind_kws):
                if any(kw in asset_lower for kw in asset_kws):
                    return int(entry["years"])

    # ── 別表第一フォールバック ──────────────────────────────────────
    for keywords, years, _ in _FALLBACK_MAPPING:
        if any(kw in asset_lower for kw in keywords):
            return years

    return _DEFAULT_YEARS
