#!/usr/bin/env python3
"""
edinet_companies テーブルのデータを screening_records 形式に変換して追加。
hantei は NULL（ラベルなし）で格納する。
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "screening_db.sqlite"

# industry_name → screening_records.industry_major マッピング
INDUSTRY_MAJOR_MAP = {
    "製造業": "製造業",
    "卸売業": "卸売業",
    "建設業": "建設業",
}

# industry_name → industry_sub デフォルト
INDUSTRY_SUB_MAP = {
    "製造業": "一般製造",
    "卸売業": "一般卸売",
    "建設業": "一般建設",
}


def convert(conn: sqlite3.Connection) -> int:
    """edinet_companies → screening_records に変換して挿入。挿入件数を返す。"""
    cur = conn.cursor()

    # すでに取り込み済みのedinet_codeをスキップ (memo列でトラック)
    cur.execute("SELECT DISTINCT memo FROM screening_records WHERE memo LIKE 'edinet:%'")
    already = {row[0] for row in cur.fetchall()}

    cur.execute("""
        SELECT
            id, company_name, industry_name, industry_code,
            revenue, op_profit, total_assets, net_assets,
            equity_ratio, employees, edinet_code, period_end
        FROM edinet_companies
        WHERE revenue IS NOT NULL
           OR total_assets IS NOT NULL
    """)
    rows = cur.fetchall()

    inserted = 0
    skipped = 0

    for row in rows:
        (ec_id, company_name, industry_name, industry_code,
         revenue, op_profit, total_assets, net_assets,
         equity_ratio, employees, edinet_code, period_end) = row

        memo_key = f"edinet:{edinet_code or ec_id}"
        if memo_key in already:
            skipped += 1
            continue

        industry_major = INDUSTRY_MAJOR_MAP.get(industry_name, industry_name or "その他")
        industry_sub   = INDUSTRY_SUB_MAP.get(industry_name, "その他")

        # 単位: EDINET は円 → 百万円に変換
        def to_m(val):
            if val is None:
                return None
            return int(val / 1_000_000)

        revenue_m   = to_m(revenue)
        op_profit_m = to_m(op_profit)
        assets_m    = to_m(total_assets)
        net_assets_m = to_m(net_assets)

        # equity_ratio は % 単位のまま (小数 → int)
        eq_ratio_int = int(round(equity_ratio)) if equity_ratio is not None else None

        conn.execute("""
            INSERT INTO screening_records (
                created_at, industry_major, industry_sub, customer_type,
                contract_type, revenue_m, op_profit_m, total_assets_m, net_assets_m,
                equity_ratio, lease_amount_m, lease_term, score, judgment,
                contract_prob, memo
            ) VALUES (
                datetime('now','localtime'),
                ?, ?, 'EDINETインポート',
                'ファイナンスリース', ?, ?, ?, ?,
                ?, NULL, NULL, NULL, NULL,
                NULL, ?
            )
        """, (
            industry_major, industry_sub,
            revenue_m, op_profit_m, assets_m, net_assets_m,
            eq_ratio_int,
            memo_key,
        ))
        already.add(memo_key)
        inserted += 1

    conn.commit()
    log.info("変換完了: 挿入=%d, スキップ=%d", inserted, skipped)
    return inserted


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        n = convert(conn)
        log.info("screening_records に %d 件追加しました", n)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
