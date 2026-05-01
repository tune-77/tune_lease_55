"""
payment_history_migration.py
============================
payment_history テーブルを lease_data.db に追加するマイグレーションスクリプト。

実行方法:
    python payment_history_migration.py

lease_logic_sumaho12.py の init_db ブロックからも呼ばれる。
"""
import os
import sqlite3

DB_PATH = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db"


def run_migration() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS payment_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_id     TEXT    NOT NULL,           -- past_cases.id へのリレーション
            check_date      TEXT    NOT NULL,           -- 記録日 (YYYY-MM-DD)
            payment_status  TEXT    NOT NULL,           -- 正常/延滞/デフォルト/完済
            overdue_amount  INTEGER DEFAULT 0,          -- 延滞金額（千円）、0=正常
            model_version   TEXT    DEFAULT '',         -- 審査時のモデルバージョン
            screening_score REAL,                       -- 審査スコアのスナップショット
            notes           TEXT    DEFAULT '',         -- 自由記述（最大500文字）
            created_at      TEXT    NOT NULL            -- レコード作成日時
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_contract_id "
        "ON payment_history(contract_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_check_date "
        "ON payment_history(check_date)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_ph_payment_status "
        "ON payment_history(payment_status)"
    )

    conn.commit()
    conn.close()
    print("✅ payment_history テーブルの作成（または確認）完了。")


if __name__ == "__main__":
    run_migration()
