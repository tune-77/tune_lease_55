# -*- coding: utf-8 -*-
"""
customer_db.py
==============
匿名化された審査実績データベース（SQLite）。

保存ポリシー:
  - 会社名・担当者名・住所など特定情報は一切保存しない
  - 数値は単位を丸めて匿名性を高める
      年商/総資産/純資産 → 10百万円単位
      利益              → 5百万円単位
      リース額          → 5百万円単位
      リース期間        → 6ヶ月単位
      自己資本比率      → 5%単位
  - 保存できる情報: 業種・規模帯・スコア・判定・成約確率・顧客区分
"""

import sqlite3
import os
import datetime
from contextlib import closing
from typing import Optional

# DB は lease_logic_sumaho12/ フォルダ内に保存
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "screening_db.sqlite")


# ─────────────────────────────────────────────
# 丸め処理
# ─────────────────────────────────────────────

def _round_m(val_yen, unit_m: int = 10) -> Optional[int]:
    """円 → 百万円に変換し unit_m 百万円単位で丸める"""
    if val_yen is None:
        return None
    try:
        val_m = float(val_yen) / 1_000_000
        return int(round(val_m / unit_m) * unit_m)
    except (TypeError, ValueError):
        return None


def _round_m_from_k(val_k, unit_m: int = 10) -> Optional[int]:
    """千円 → 百万円に変換し unit_m 百万円単位で丸める"""
    if val_k is None:
        return None
    try:
        val_m = float(val_k) / 1_000
        return int(round(val_m / unit_m) * unit_m)
    except (TypeError, ValueError):
        return None


def _round_term(months) -> Optional[int]:
    """リース期間を 6 ヶ月単位に丸める"""
    if months is None:
        return None
    try:
        m = int(months)
        return max(6, round(m / 6) * 6)
    except (TypeError, ValueError):
        return None


def _round_ratio(val, unit: int = 5) -> Optional[int]:
    """比率を unit% 単位に丸める"""
    if val is None:
        return None
    try:
        return int(round(float(val) / unit) * unit)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# DB 初期化
# ─────────────────────────────────────────────

def init_db():
    """テーブルが存在しなければ作成する"""
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS screening_records (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                industry_major  TEXT,
                industry_sub    TEXT,
                customer_type   TEXT,
                contract_type   TEXT,
                revenue_m       INTEGER,   -- 年商（百万円、10M丸め）
                op_profit_m     INTEGER,   -- 経常利益（百万円、5M丸め）
                total_assets_m  INTEGER,   -- 総資産（百万円、10M丸め）
                net_assets_m    INTEGER,   -- 純資産（百万円、10M丸め）
                equity_ratio    INTEGER,   -- 自己資本比率（%、5%丸め）
                lease_amount_m  INTEGER,   -- リース額（百万円、5M丸め）
                lease_term      INTEGER,   -- リース期間（月、6ヶ月丸め）
                score           REAL,      -- 審査スコア（小数第1位）
                judgment        TEXT,      -- 判定
                contract_prob   REAL,      -- 成約確率
                memo            TEXT       -- 任意メモ（200文字以内）
            )
        """)
        # 後方互換: 列が足りない場合は追加
        existing = {row[1] for row in c.execute("PRAGMA table_info(screening_records)")}
        for col, typedef in [
            ("op_profit_m",    "INTEGER"),
            ("contract_type",  "TEXT"),
            ("contract_prob",  "REAL"),
            ("memo",           "TEXT"),
        ]:
            if col not in existing:
                c.execute(f"ALTER TABLE screening_records ADD COLUMN {col} {typedef}")
        conn.commit()


# ─────────────────────────────────────────────
# 保存
# ─────────────────────────────────────────────

def save_record(result: dict, inputs: dict, memo: str = "") -> int:
    """
    審査結果を匿名化して保存。

    Parameters
    ----------
    result : スコアリング結果 dict（last_result）
    inputs : 送信入力 dict（last_submitted_inputs）。値は千円単位。
    memo   : 任意メモ（200文字以内）

    Returns
    -------
    int : 新しいレコードID
    """
    init_db()

    # 入力値（千円単位）
    nenshu_k      = float(inputs.get("nenshu",      0) or 0)
    ord_profit_k  = float(inputs.get("item4_ord_profit", 0) or
                          inputs.get("ord_profit", 0) or
                          inputs.get("rieki", 0) or 0)
    total_assets_k = float(inputs.get("total_assets", 0) or 0)
    net_assets_k   = float(inputs.get("net_assets",   0) or 0)
    lease_credit_k = float(inputs.get("lease_credit", 0) or 0)
    lease_term     = inputs.get("lease_term", 36)

    record = {
        "created_at":     datetime.datetime.now().isoformat(timespec="seconds"),
        "industry_major": result.get("industry_major", ""),
        "industry_sub":   result.get("industry_sub", ""),
        "customer_type":  inputs.get("customer_type", ""),
        "contract_type":  inputs.get("contract_type", ""),
        # 数値（千円 → 百万円 → 丸め）
        "revenue_m":      _round_m_from_k(nenshu_k,      unit_m=10),
        "op_profit_m":    _round_m_from_k(ord_profit_k,  unit_m=5),
        "total_assets_m": _round_m_from_k(total_assets_k, unit_m=10),
        "net_assets_m":   _round_m_from_k(net_assets_k,   unit_m=10),
        "equity_ratio":   _round_ratio(result.get("user_equity_ratio"), unit=5),
        "lease_amount_m": _round_m_from_k(lease_credit_k, unit_m=5),
        "lease_term":     _round_term(lease_term),
        # スコア・判定
        "score":          round(float(result.get("score", 0)), 1),
        "judgment":       result.get("hantei", ""),
        "contract_prob":  round(float(result.get("contract_prob", 0) or 0), 1),
        "memo":           str(memo)[:200],
    }

    with closing(sqlite3.connect(_DB_PATH)) as conn:
        c = conn.cursor()
        c.execute("""
            INSERT INTO screening_records
            (created_at, industry_major, industry_sub, customer_type, contract_type,
             revenue_m, op_profit_m, total_assets_m, net_assets_m, equity_ratio,
             lease_amount_m, lease_term, score, judgment, contract_prob, memo)
            VALUES
            (:created_at, :industry_major, :industry_sub, :customer_type, :contract_type,
             :revenue_m, :op_profit_m, :total_assets_m, :net_assets_m, :equity_ratio,
             :lease_amount_m, :lease_term, :score, :judgment, :contract_prob, :memo)
        """, record)
        new_id = c.lastrowid
        conn.commit()
    return new_id


# ─────────────────────────────────────────────
# 取得
# ─────────────────────────────────────────────

def get_records(
    industry_major: str = "",
    judgment: str = "",
    score_min: float = 0,
    score_max: float = 100,
    limit: int = 200,
    offset: int = 0,
) -> list:
    """レコード一覧を取得（フィルタ付き）"""
    init_db()
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        where, params = ["score >= ?", "score <= ?"], [score_min, score_max]
        if industry_major:
            where.append("industry_major = ?")
            params.append(industry_major)
        if judgment:
            where.append("judgment = ?")
            params.append(judgment)

        sql = ("SELECT * FROM screening_records WHERE "
               + " AND ".join(where)
               + " ORDER BY created_at DESC LIMIT ? OFFSET ?")
        params += [limit, offset]
        c.execute(sql, params)
        rows = [dict(r) for r in c.fetchall()]
    return rows


def get_total_count() -> int:
    """総レコード数"""
    init_db()
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM screening_records").fetchone()
    return row[0] if row else 0


def get_stats() -> dict:
    """全体集計統計"""
    init_db()
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM screening_records")
        total = c.fetchone()[0]
        if total == 0:
            return {"total": 0}

        c.execute("SELECT AVG(score), MIN(score), MAX(score) FROM screening_records")
        score_avg, score_min, score_max = c.fetchone()

        c.execute("""
            SELECT judgment, COUNT(*) FROM screening_records
            GROUP BY judgment ORDER BY COUNT(*) DESC
        """)
        judgment_counts = dict(c.fetchall())

        c.execute("""
            SELECT industry_major, COUNT(*) AS cnt, AVG(score) AS avg_sc
            FROM screening_records
            WHERE industry_major != ''
            GROUP BY industry_major ORDER BY cnt DESC LIMIT 12
        """)
        by_industry = [
            {"industry": r[0], "count": r[1], "avg_score": round(r[2], 1)}
            for r in c.fetchall()
        ]

        c.execute("""
            SELECT
                AVG(revenue_m), AVG(total_assets_m), AVG(net_assets_m),
                AVG(equity_ratio), AVG(lease_amount_m), AVG(lease_term)
            FROM screening_records
        """)
        avgs = c.fetchone()

        c.execute("""
            SELECT
                CASE
                    WHEN score < 40  THEN '〜40'
                    WHEN score < 50  THEN '40〜50'
                    WHEN score < 60  THEN '50〜60'
                    WHEN score < 70  THEN '60〜70'
                    WHEN score < 80  THEN '70〜80'
                    ELSE '80〜'
                END AS band,
                COUNT(*) AS cnt
            FROM screening_records
            GROUP BY band ORDER BY band
        """)
        score_dist = dict(c.fetchall())

    return {
        "total":          total,
        "score_avg":      round(score_avg, 1) if score_avg else None,
        "score_min":      round(score_min, 1) if score_min else None,
        "score_max":      round(score_max, 1) if score_max else None,
        "judgment_counts": judgment_counts,
        "by_industry":    by_industry,
        "avg_revenue_m":  round(avgs[0], 0) if avgs[0] else None,
        "avg_assets_m":   round(avgs[1], 0) if avgs[1] else None,
        "avg_equity":     round(avgs[3], 1) if avgs[3] else None,
        "avg_lease_m":    round(avgs[4], 0) if avgs[4] else None,
        "avg_lease_term": round(avgs[5], 0) if avgs[5] else None,
        "score_dist":     score_dist,
    }


def get_industry_list() -> list:
    """DB に存在する業種大分類一覧"""
    init_db()
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT DISTINCT industry_major FROM screening_records "
            "WHERE industry_major != '' ORDER BY industry_major"
        ).fetchall()
    return [r[0] for r in rows]


def delete_record(record_id: int):
    """レコードを削除"""
    init_db()
    with closing(sqlite3.connect(_DB_PATH)) as conn:
        conn.execute("DELETE FROM screening_records WHERE id = ?", (record_id,))
        conn.commit()


def get_db_path() -> str:
    return _DB_PATH
