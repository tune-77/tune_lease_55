#!/usr/bin/env python3
"""
past_cases の実データで利回り推測を試し、Before/After を比較表示する。

使い方:
  python scripts/demo_predict_yield.py              # 最新の1件を使う
  python scripts/demo_predict_yield.py --all        # 全件を処理
  python scripts/demo_predict_yield.py --case-id <id>
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# ルートを sys.path に追加（どこから実行しても動くように）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from lease_yield_predictor import predict_yield

DB_PATH = Path(__file__).parent.parent.parent / "lease_logic_sumaho12" / "data" / "lease_data.db"


def _extract_inputs(data: dict, year_month: str) -> dict:
    inputs_raw = data.get("inputs", {})
    result = data.get("result", {})
    return {
        "year_month":        year_month,
        "lease_term_months": inputs_raw.get("lease_term", 60),
        "lease_asset_id":    inputs_raw.get("lease_asset_id", "other"),
        "grade":             inputs_raw.get("grade", ""),
        "borrower_score":    result.get("score_borrower", 70.0),
    }


def _show(case_id: str, data: dict, conn: sqlite3.Connection) -> None:
    # 審査年月を timestamp から取得（YYYY-MM 形式に変換）
    ts = data.get("timestamp", "")
    year_month = ts[:7] if ts else "2026-04"

    old_base = data.get("pricing", {}).get("base_rate")
    old_pred = data.get("pricing", {}).get("pred_rate")

    try:
        inputs = _extract_inputs(data, year_month)
        result = predict_yield(conn, inputs)
    except ValueError as e:
        print(f"  [スキップ] {e}")
        return

    bk = result["breakdown"]
    print(f"\n案件ID: {case_id}")
    print(f"  資産: {inputs['lease_asset_id']}  格付: {inputs['grade']}  "
          f"期間: {inputs['lease_term_months']}ヶ月 → {result['term_years_used']}年で計算")
    print(f"  借手スコア: {inputs['borrower_score']:.1f}")
    if result["fallback_used"]:
        print(f"  ※ {result['fallback_note']}")
    print()
    print(f"  【旧方式】  base_rate={old_base}%  pred_rate={old_pred:.4f}")
    print(f"  【新方式】  推測利回り = {result['predicted_yield']:.2f}%")
    print(f"             内訳: 調達金利 {bk['base']:.2f}%"
          f" + 資産 {bk['asset']:.2f}%"
          f" + 格付 {bk['grade']:.2f}%"
          f" + リスク {bk['risk']:+.2f}%"
          f" = {result['predicted_yield']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="利回り推測のBefore/Afterを確認")
    parser.add_argument("--case-id", help="特定の案件IDを指定")
    parser.add_argument("--all", action="store_true", help="全件を処理")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.case_id:
            row = conn.execute(
                "SELECT id, data FROM past_cases WHERE id=?", (args.case_id,)
            ).fetchone()
            if not row:
                sys.exit(f"案件ID {args.case_id} が見つかりません")
            rows = [row]
        elif args.all:
            rows = conn.execute("SELECT id, data FROM past_cases").fetchall()
        else:
            row = conn.execute(
                "SELECT id, data FROM past_cases ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                sys.exit("past_cases にデータがありません")
            rows = [row]

        for case_id, data_json in rows:
            data = json.loads(data_json)
            _show(case_id, data, conn)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
