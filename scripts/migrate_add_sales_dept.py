"""
past_cases テーブルに sales_dept カラムを追加するマイグレーションスクリプト。
既に存在する場合はスキップし、既存レコードの data JSON に sales_dept: "未設定" をバックフィルする。
"""
import os
import sys
import sqlite3
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

from data_cases import DB_PATH  # noqa: E402  absolute path defined there


def migrate():
    if not os.path.exists(DB_PATH):
        print(f"DB not found: {DB_PATH}")
        print("マイグレーション対象の DB が存在しないためスキップします。")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # カラムが既に存在するか確認
    cursor.execute("PRAGMA table_info(past_cases)")
    columns = [row[1] for row in cursor.fetchall()]

    if "sales_dept" in columns:
        print("sales_dept カラムは既に存在します。スキップします。")
    else:
        cursor.execute("ALTER TABLE past_cases ADD COLUMN sales_dept TEXT DEFAULT '未設定'")
        conn.commit()
        print("sales_dept カラムを追加しました。")

    # 既存レコードの data JSON に sales_dept をバックフィル
    cursor.execute("SELECT id, data FROM past_cases WHERE sales_dept IS NULL OR sales_dept = ''")
    rows = cursor.fetchall()
    updated = 0
    for row_id, data_str in rows:
        try:
            data = json.loads(data_str) if data_str else {}
        except (json.JSONDecodeError, TypeError):
            data = {}
        if "sales_dept" not in data:
            data["sales_dept"] = "未設定"
            new_json = json.dumps(data, ensure_ascii=False)
            cursor.execute(
                "UPDATE past_cases SET sales_dept = '未設定', data = ? WHERE id = ?",
                (new_json, row_id),
            )
            updated += 1

    conn.commit()
    conn.close()
    print(f"バックフィル完了: {updated} 件のレコードを更新しました。")


if __name__ == "__main__":
    migrate()
