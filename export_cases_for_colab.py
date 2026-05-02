"""
Colab 学習用データエクスポートスクリプト
-------------------------------------
このスクリプトをローカルで実行すると、
past_cases テーブルの成約・失注データを
data/cases_for_colab.json として書き出します。

実行方法:
  python export_cases_for_colab.py
"""
import json
import os
import sqlite3
from contextlib import closing

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(_SCRIPT_DIR, "data", "lease_data.db")
OUT_PATH    = os.path.join(_SCRIPT_DIR, "data", "cases_for_colab.json")


def main():
    if not os.path.exists(DB_PATH):
        print(f"DB が見つかりません: {DB_PATH}")
        return

    cases = []
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT data FROM past_cases ORDER BY timestamp ASC")
        for row in cursor.fetchall():
            try:
                d = json.loads(row[0])
            except json.JSONDecodeError:
                continue
            if not d.get("id"):
                continue
            # 検収 → 成約に正規化
            if d.get("final_status") in ("検収", "検収完了"):
                d["final_status"] = "成約"
            if d.get("final_status") not in ("成約", "失注"):
                continue
            cases.append(d)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    n_pos = sum(1 for c in cases if c["final_status"] == "成約")
    n_neg = len(cases) - n_pos
    print(f"エクスポート完了: {len(cases)} 件（成約 {n_pos} / 失注 {n_neg}）")
    print(f"出力先: {OUT_PATH}")
    print("次のステップ: cases_for_colab.json を Google Colab にアップロードしてください。")


if __name__ == "__main__":
    main()
