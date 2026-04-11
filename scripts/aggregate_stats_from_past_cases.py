"""
統計集計バッチ: past_cases → screening_records

past_cases（業務ソース）の全件を screening_records（統計ビュー）へ再構築する。
実行のたびに screening_records を全クリアしてから全件 INSERT するため冪等。

使用方法:
    python scripts/aggregate_stats_from_past_cases.py
"""
import json
import os
import sqlite3
import sys
import time
from contextlib import closing

# プロジェクトルートをパスに追加
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

_LEASE_DB = os.path.join(_REPO_ROOT, "data", "lease_data.db")
_SCREENING_DB = os.path.join(_REPO_ROOT, "data", "screening_db.sqlite")


def aggregate(verbose: bool = True) -> dict:
    """
    past_cases を全件読み込み → screening_records を再構築する。

    Returns
    -------
    dict: {"processed": int, "skipped": int, "elapsed_s": float}
    """
    from customer_db import save_record, init_db

    if not os.path.exists(_LEASE_DB):
        msg = f"[aggregate] lease_data.db が見つかりません: {_LEASE_DB}"
        print(msg, file=sys.stderr)
        return {"processed": 0, "skipped": 0, "elapsed_s": 0.0, "error": msg}

    t0 = time.time()

    # 1. past_cases を全件読み込む
    cases = []
    with closing(sqlite3.connect(_LEASE_DB)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, data, final_status FROM past_cases ORDER BY timestamp ASC"
        ).fetchall()
        for r in rows:
            try:
                d = json.loads(r["data"] or "{}")
                d["_past_case_id"] = str(r["id"])
                d["final_status"] = r["final_status"] or "未登録"
                cases.append(d)
            except json.JSONDecodeError:
                continue

    if verbose:
        print(f"[aggregate] past_cases 読み込み: {len(cases)} 件")

    # 2. screening_records を全クリア（ダミーデータ含む全て削除）
    init_db()
    with closing(sqlite3.connect(_SCREENING_DB)) as conn:
        conn.execute("DELETE FROM screening_records")
        conn.commit()

    if verbose:
        print("[aggregate] screening_records をクリアしました")

    # 3. past_cases から screening_records へ変換・INSERT
    processed = 0
    skipped = 0
    for case in cases:
        try:
            result = case.get("result", {}) or {}
            inputs = case.get("inputs", {}) or {}

            # result に必要なキーを補完
            if not result.get("industry_major") and case.get("industry_major"):
                result["industry_major"] = case["industry_major"]
            if not result.get("industry_sub") and case.get("industry_sub"):
                result["industry_sub"] = case["industry_sub"]
            if not result.get("score") and case.get("score"):
                result["score"] = case["score"]
            if not result.get("hantei") and case.get("hantei"):
                result["hantei"] = case["hantei"]

            # inputs に必要なキーを補完
            if not inputs.get("customer_type") and case.get("customer_type"):
                inputs["customer_type"] = case["customer_type"]
            if not inputs.get("company_name") and case.get("company_name"):
                inputs["company_name"] = case["company_name"]
            if not inputs.get("company_no") and case.get("company_no"):
                inputs["company_no"] = case["company_no"]

            # final_status と _past_case_id を memo に埋め込む
            memo_dict = {
                "_past_case_id": case["_past_case_id"],
                "final_status": case["final_status"],
            }
            if case.get("company_name"):
                memo_dict["company_name"] = case["company_name"]

            save_record(result, inputs, json.dumps(memo_dict, ensure_ascii=False))
            processed += 1
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[aggregate] スキップ id={case.get('_past_case_id')}: {e}", file=sys.stderr)

    elapsed = round(time.time() - t0, 2)
    if verbose:
        print(f"[aggregate] 完了: {processed} 件処理, {skipped} 件スキップ, {elapsed}s")

    return {"processed": processed, "skipped": skipped, "elapsed_s": elapsed}


if __name__ == "__main__":
    result = aggregate(verbose=True)
    sys.exit(0 if result.get("error") is None else 1)
