import os
import sqlite3

DB_PATH = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db"

def run_migration():
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH))

    print(f"Connecting to SQLite DB: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 物件市場相場履歴テーブルの作成
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS asset_price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id TEXT NOT NULL,            -- past_cases.id へのリレーション (TEXT型)
            inspected_at TEXT NOT NULL,           -- 調査日時 (YYYY-MM-DD HH:MM:SS)
            current_market_price INTEGER,         -- 現在の中古市場価格 (千円)
            residual_debt INTEGER,                -- その時点でのリース残債 (千円)
            profit_margin INTEGER,                -- 含み益 (市場価格 - リース残債)
            is_alert_triggered BOOLEAN DEFAULT 0  -- アラート発動フラグ
        )
    """)
    conn.commit()
    print("✅ Table 'asset_price_history' successfully created (or already exists).")
    conn.close()

if __name__ == "__main__":
    run_migration()
