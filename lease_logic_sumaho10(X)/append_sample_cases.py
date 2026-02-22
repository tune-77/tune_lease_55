"""
サンプル案件を約150件、**別ファイル** past_cases_sample.jsonl に出力する。
本体の past_cases.jsonl には一切書き込まない（本番データと混在させない）。
開発・デモ用。実行: リポジトリルートで python lease_logic_sumaho10/append_sample_cases.py
"""
import os
import sys
import json
import random
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# 本体は触らない。サンプル専用ファイルにのみ出力する。
SAMPLE_CASES_FILE = os.path.join(_REPO_ROOT, "past_cases_sample.jsonl")

# 業種中分類 → 大分類の簡易マッピング
def _sub_to_major(industry_sub):
    if not industry_sub:
        return "D 建設業"
    code = industry_sub.split()[0] if " " in industry_sub else industry_sub[:2]
    if code.startswith("0"):
        return "D 建設業"
    if code in ("09", "21", "24", "26"):
        return "E 製造業"
    if code in ("43", "44"):
        return "H 運輸業"
    if "卸売" in industry_sub or "小売" in industry_sub or code in ("50", "56"):
        return "I 卸売業・小売業"
    if code in ("68", "70", "75", "76", "89", "91"):
        return "M その他サービス業"
    if code == "83" or "医療" in industry_sub or "福祉" in industry_sub or "介護" in industry_sub:
        return "P 医療・福祉"
    return "D 建設業"


def main():
    # 業種一覧を読み込み
    bench_path = os.path.join(_REPO_ROOT, "industry_benchmarks.json")
    if not os.path.exists(bench_path):
        print("industry_benchmarks.json が見つかりません。")
        sys.exit(1)
    with open(bench_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    industry_subs = list(benchmarks.keys())

    # サンプル用ファイルの既存件数（あれば）
    existing = 0
    if os.path.exists(SAMPLE_CASES_FILE):
        with open(SAMPLE_CASES_FILE, "r", encoding="utf-8") as f:
            existing = sum(1 for line in f if line.strip())
    print(f"サンプル用ファイル既存: {existing} 件（本体 past_cases.jsonl は書き換えません）")

    n_add = 150
    base_time = datetime.now() - timedelta(days=400)
    statuses = ["成約", "成約", "成約", "失注", "失注", "未登録"]

    for i in range(n_add):
        industry_sub = random.choice(industry_subs)
        industry_major = _sub_to_major(industry_sub)
        # スコアは 40〜92 の範囲でランダム（承認ライン71前後を多めに）
        score = round(random.gauss(68, 12), 1)
        score = max(30, min(98, score))
        # 自己資本比率 10〜60%
        user_eq = round(random.gauss(32, 12), 1)
        user_eq = max(5, min(65, user_eq))
        # 営業利益率
        user_op = round(random.gauss(4, 3), 1)
        user_op = max(-2, min(12, user_op))

        nenshu = random.randint(30000, 2000000)  # 千円
        total_assets = int(nenshu * random.uniform(0.8, 2.5))
        net_assets = int(total_assets * (user_eq / 100))

        ts = base_time + timedelta(days=random.randint(0, 400), hours=random.randint(0, 23))
        case_id = ts.strftime("%Y%m%d%H%M%S") + f"{i:04d}"

        final_status = random.choice(statuses)
        if final_status == "成約" and score < 65:
            final_status = "失注"
        elif final_status == "失注" and score >= 78:
            final_status = "成約"

        record = {
            "id": case_id,
            "timestamp": ts.isoformat(),
            "industry_major": industry_major,
            "industry_sub": industry_sub,
            "customer_type": random.choice(["既存先", "既存先", "新規先"]),
            "main_bank": random.choice(["メイン先", "メイン先", "非メイン先"]),
            "competitor": random.choice(["競合なし", "競合あり"]),
            "competitor_rate": None,
            "inputs": {
                "nenshu": nenshu,
                "gross_profit": int(nenshu * 0.18),
                "op_profit": int(nenshu * user_op / 100),
                "ord_profit": int(nenshu * 0.03),
                "net_income": int(nenshu * 0.02),
                "machines": int(nenshu * 0.15),
                "other_assets": int(nenshu * 0.05),
                "rent": 0,
                "depreciation": int(nenshu * 0.03),
                "dep_expense": int(nenshu * 0.01),
                "rent_expense": int(nenshu * 0.005),
                "bank_credit": int(nenshu * 0.2),
                "lease_credit": int(nenshu * 0.05),
                "contracts": random.randint(1, 10),
                "grade": random.choice(["1-3", "1-3", "4-6", "要注意"]),
                "contract_type": "一般",
                "deal_source": "銀行紹介",
                "lease_term": 60,
                "acceptance_year": ts.year,
                "acquisition_cost": int(nenshu * 0.02),
                "lease_asset_id": None,
                "lease_asset_name": "",
                "lease_asset_score": 50,
                "qualitative": {},
                "qualitative_scoring": None,
            },
            "result": {
                "score": score,
                "hantei": "承認圏内" if score >= 71 else "要審議",
                "user_eq": user_eq,
                "user_op": user_op,
                "comparison": f"営業利益率{user_op:.1f}% 自己資本比率{user_eq:.1f}%",
                "financials": {
                    "nenshu": nenshu,
                    "rieki": int(nenshu * user_op / 100),
                    "assets": total_assets,
                    "net_assets": net_assets,
                },
                "industry_major": industry_major,
                "industry_sub": industry_sub,
            },
            "pricing": {"base_rate": 1.2, "pred_rate": 2.5},
            "final_status": final_status,
        }

        with open(SAMPLE_CASES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"追加: {n_add} 件 → 合計: {existing + n_add} 件")
    print(f"保存先: {SAMPLE_CASES_FILE}")


if __name__ == "__main__":
    main()
