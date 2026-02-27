"""
可視化用JSONを生成する。
past_cases.jsonl を読んでフロント用スキーマに変換。不足分はダミーで補完。
Flask の /api/visualization/data からもこの get_visualization_data() を利用する。
静的ファイルに書き出す場合: リポジトリルートで
  python lease_logic_sumaho10/web/export_visualization_data.py
→ lease_logic_sumaho10/web/static/data/visualization.json に出力。
"""
import os
import sys
import json
import random
from datetime import datetime
from pathlib import Path

_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMODULE = os.path.dirname(_WEB_DIR)
_REPO_ROOT = os.path.dirname(_SUBMODULE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CASES_FILE = os.path.join(_REPO_ROOT, "past_cases.jsonl")
DEFAULT_OUTPUT = os.path.join(_WEB_DIR, "static", "data", "visualization.json")


def _is_prime(n):
    if n < 2:
        return False
    n = int(round(n))
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def _case_to_data_point(idx: int, c: dict) -> dict:
    """過去案件1件を可視化用の1データポイントに変換。"""
    res = c.get("result") or {}
    fin = res.get("financials") or c.get("inputs") or {}
    qual = (c.get("inputs") or {}).get("qualitative_scoring") or res.get("qualitative_scoring_correction") or {}
    score = res.get("score")
    if score is None:
        score = 50.0
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 50.0
    score = max(0, min(100, score))
    total_score = int(round(score))

    quant_score = float(res.get("score_borrower") or score * 0.9)
    qual_score = float(qual.get("combined_score") or qual.get("weighted_score") or 50)
    rank = (qual.get("rank") or "C").strip() if isinstance(qual.get("rank"), str) else "C"
    if rank not in ("A", "B", "C", "D", "E"):
        rank = "C"

    hantei = res.get("hantei") or ""
    approved = hantei == "承認圏内"

    nenshu = fin.get("nenshu") or 0
    assets = fin.get("assets") or (c.get("inputs") or {}).get("machines", 0) * 2
    net_assets = fin.get("net_assets") or (assets * 0.3)
    try:
        nenshu, assets, net_assets = int(nenshu), int(assets), int(net_assets)
    except (TypeError, ValueError):
        nenshu, assets, net_assets = 0, 0, 0
    # 千円→円
    revenue = nenshu * 1000
    assets_yen = assets * 1000
    equity = net_assets * 1000

    ts = c.get("timestamp") or res.get("timestamp") or ""
    date = ts[:10] if isinstance(ts, str) and len(ts) >= 10 else "2026-01-01"
    industry = c.get("industry_sub") or c.get("industry_major") or "その他"
    company_name = f"案件 {c.get('id', idx)}" if not industry else f"{industry} #{idx+1}"

    return {
        "id": idx,
        "company": company_name,
        "industry": industry,
        "quantScore": round(quant_score, 1),
        "qualScore": round(qual_score, 1),
        "totalScore": round(score, 1),
        "rank": rank,
        "approved": approved,
        "isPrime": _is_prime(total_score),
        "date": date,
        "details": {
            "revenue": revenue,
            "assets": assets_yen,
            "equity": equity,
        },
    }


def _generate_dummy_point(idx: int, industries: list) -> dict:
    """ダミーデータ1件を生成。"""
    total_score = 30 + random.random() * 70
    rank = "A" if total_score >= 80 else "B" if total_score >= 71 else "C" if total_score >= 50 else "D" if total_score >= 30 else "E"
    approved = total_score >= 71
    return {
        "id": idx,
        "company": f"サンプル企業 {idx+1:03d}",
        "industry": random.choice(industries) if industries else "その他",
        "quantScore": round(30 + random.random() * 70, 1),
        "qualScore": round(30 + random.random() * 70, 1),
        "totalScore": round(total_score, 1),
        "rank": rank,
        "approved": approved,
        "isPrime": _is_prime(int(round(total_score))),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "details": {
            "revenue": int(random.random() * 500_000_000),
            "assets": int(random.random() * 800_000_000),
            "equity": int(random.random() * 300_000_000),
        },
    }


def get_visualization_data(max_count: int = 1000) -> dict:
    """
    可視化用のJSON用辞書を返す。
    past_cases.jsonl から読んだ件数が max_count 未満ならダミーで補完する。
    """
    industries = [
        "06 総合工事業",
        "09 食料品製造業",
        "21 金属製品製造業",
        "44 道路貨物運送業",
        "50-55 各種卸売業",
        "76 飲食店",
        "85 社会保険・社会福祉・介護事業",
    ]
    data_list = []

    real_count = 0
    if os.path.exists(CASES_FILE):
        try:
            with open(CASES_FILE, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                        data_list.append(_case_to_data_point(idx, c))
                        real_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

    # データがある場合はデータ優先。不足分のみダミーで補完（最大 max_count 件）
    while len(data_list) < max_count:
        data_list.append(_generate_dummy_point(len(data_list), industries))

    data_list = data_list[:max_count]
    return {
        "generated_at": datetime.now().isoformat(),
        "total_count": len(data_list),
        "real_count": real_count,
        "data": data_list,
        "metadata": {
            "version": "1.0",
            "source": "past_cases",
            "description": "リース審査スコア可視化用データ。実データを先頭に、不足分のみサンプル補完。",
        },
    }


def export_to_json(output_path: str = None) -> str:
    """get_visualization_data() の結果をJSONファイルに書き出す。"""
    output_path = output_path or DEFAULT_OUTPUT
    data = get_visualization_data()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"可視化用JSONを出力しました: {output_path}")
    print(f"件数: {data['total_count']} / 承認: {sum(1 for d in data['data'] if d['approved'])} / 素数スコア: {sum(1 for d in data['data'] if d['isPrime'])}")
    return output_path


if __name__ == "__main__":
    export_to_json()
