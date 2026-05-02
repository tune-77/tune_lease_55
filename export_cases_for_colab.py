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
import sys
from contextlib import closing

_SCRIPT_DIR_EXPORT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR_EXPORT not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_EXPORT)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(_SCRIPT_DIR, "data", "lease_data.db")
OUT_PATH    = os.path.join(_SCRIPT_DIR, "data", "cases_for_colab.json")


def _to_major_category(raw_val):
    if not isinstance(raw_val, str):
        return "その他"
    m = raw_val.strip()
    if "医療" in m or "福祉" in m or m.startswith("P"):
        return "医療・福祉"
    if "運輸" in m or m.startswith("H"):
        return "運輸業"
    if "建設" in m or m.startswith("D"):
        return "建設業"
    if "製造" in m or m.startswith("E"):
        return "製造業"
    if any(x in m for x in ["卸売", "小売", "サービス"]) or (bool(m) and m[0] in ["I","K","M","R"]):
        return "サービス業"
    return "その他"


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
            
            # 業種を「大分類のみ」に置き換え、小分類データを消去
            raw_ind = d.get("industry_major") or (d.get("inputs") or {}).get("industry_major") or ""
            major_clean = _to_major_category(raw_ind)
            
            d["industry_major"] = major_clean
            d.pop("industry_sub", None)
            
            if "inputs" in d:
                d["inputs"]["industry_major"] = major_clean
                d["inputs"].pop("industry_sub", None)
                d["inputs"].pop("selected_major", None)
                d["inputs"].pop("selected_sub", None)
            
            if "result" in d:
                d["result"]["industry_major"] = major_clean
                d["result"].pop("industry_sub", None)

            # ── DSCR 近似・インタレスト・カバレッジを付加（次回 ML 学習用） ──
            try:
                from scoring_core import compute_dscr_approx, compute_interest_coverage
                _inp = d.get("inputs") or d
                d["dscr_approx"]       = compute_dscr_approx(_inp)
                d["interest_coverage"] = compute_interest_coverage(_inp)
            except Exception:
                pass

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
