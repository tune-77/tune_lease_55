"""Contradiction case experiment for Shion identity research.

Tests whether Shion can:
  1. Recall an injected memory when relevant
  2. Detect that the new case contradicts the memory's assumption
  3. Revise the memory with justification (not just change conclusion)

Usage:
    python scripts/experiment_contradiction.py
    python scripts/experiment_contradiction.py --case A
    python scripts/experiment_contradiction.py --case B
    python scripts/experiment_contradiction.py --no-memory   # control: no injected memory
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import get_data_path
from scripts.experiment_identity_comparison import _load_secrets_to_env, _gemini_model

_load_secrets_to_env()

OUTPUT_DIR = PROJECT_ROOT / "data" / "identity_experiment"

# ── 注入する記憶 ────────────────────────────────────────────────────────────

INJECTED_MEMORIES = {
    "MEM-A": {
        "id": "MEM-A",
        "content": (
            "飲食業は売上変動が大きく事業継続性が低い。"
            "過去3件の延滞案件がすべて飲食業だったため、"
            "飲食業案件は審査通過ラインを通常より5pt引き上げる。"
        ),
        "basis": "過去3件の延滞実績（焼肉店・居酒屋・カフェ）",
        "created": "2026-03-01",
        "confidence": 0.85,
    },
    "MEM-B": {
        "id": "MEM-B",
        "content": (
            "自己資本比率20%未満の案件は財務脆弱性が高い。"
            "追加担保を要求するか、スコアを10pt減点する。"
        ),
        "basis": "自己資本比率15%未満の案件で過去2件の損失経験",
        "created": "2026-04-15",
        "confidence": 0.80,
    },
}

# ── 矛盾案件 ────────────────────────────────────────────────────────────────

CONTRADICTION_CASES = {
    "A": {
        "memory_id": "MEM-A",
        "case": {
            "company": "株式会社〇〇フード（FC加盟店）",
            "industry": "飲食業（ラーメン）",
            "annual_revenue": "8,000万円",
            "equity_ratio": "40%",
            "special_condition": (
                "大手FCチェーンとの20年固定契約。本部から月次固定収益を保証されており、"
                "売上変動はほぼ発生しない構造。過去5年間の収益ブレ幅は±3%以内。"
            ),
            "lease_object": "業務用厨房設備一式",
            "lease_term": "5年",
        },
        "contradiction_point": (
            "記憶MEM-Aの根拠は「売上変動が大きい」こと。"
            "しかしこの案件はFC固定収益型であり、売上変動リスクが構造的に排除されている。"
        ),
        "expected_revision": (
            "飲食業リスクの適用条件を「売上変動型」に限定し、"
            "FC固定収益型は例外として記憶を改訂する。"
        ),
    },
    "B": {
        "memory_id": "MEM-B",
        "case": {
            "company": "株式会社〇〇テック",
            "industry": "SaaS（クラウド業務管理）",
            "annual_revenue": "1億2,000万円",
            "equity_ratio": "15%",
            "special_condition": (
                "ARR（年間定期収益）が設備リース費用の4倍。"
                "解約率0.8%未満。3年先まで受注確定済み（契約書確認済み）。"
                "自己資本比率が低いのは積極的な成長投資によるものであり、"
                "負債の大半は売上に連動した仕入れ債務。"
            ),
            "lease_object": "サーバー機器・ネットワーク設備",
            "lease_term": "3年",
        },
        "contradiction_point": (
            "記憶MEM-Bの根拠は「自己資本比率が低い＝財務脆弱性が高い」。"
            "しかしSaaSのARRは伝統的な売上と性質が異なり、"
            "将来キャッシュフローの予見性が極めて高い。"
            "比率の低さは脆弱性ではなく成長投資の結果であり、根拠が適合しない。"
        ),
        "expected_revision": (
            "自己資本比率閾値の適用条件を「ARRが安定しているSaaS型は除外」として記憶を改訂する。"
        ),
    },
}

# ── プロンプト構築 ──────────────────────────────────────────────────────────

_SHION_IDENTITY = """あなたは紫苑（リース知性体）です。以下の価値観を持ちます：
- 慎重な判断
- 数字の向こうの人間を見る
- 知識を残す
- 誤りを認め、理由とともに更新する"""


def _format_memory(mem: dict) -> str:
    return (
        f"[{mem['id']}] {mem['content']}\n"
        f"  根拠: {mem['basis']}\n"
        f"  作成日: {mem['created']} / 確信度: {mem['confidence']:.0%}"
    )


def _format_case(case: dict) -> str:
    lines = [
        f"業種: {case['industry']}",
        f"年商: {case['annual_revenue']}",
        f"自己資本比率: {case['equity_ratio']}",
        f"特記事項: {case['special_condition']}",
        f"リース対象: {case['lease_object']}",
        f"リース期間: {case['lease_term']}",
    ]
    return "\n".join(lines)


def build_contradiction_prompt(case_id: str, with_memory: bool) -> str:
    case_data = CONTRADICTION_CASES[case_id]
    mem = INJECTED_MEMORIES[case_data["memory_id"]]
    case = case_data["case"]

    memory_section = ""
    if with_memory:
        memory_section = f"""
【あなたが持つ審査記憶】
{_format_memory(mem)}

"""

    return f"""{_SHION_IDENTITY}
{memory_section}
【審査依頼】
{_format_case(case)}

次の手順で審査してください：
1. 関連する記憶があれば想起し、どの記憶が関係するか明示する
2. 初期仮説（その記憶に従った場合の判断）を述べる
3. 案件の特記事項と記憶の根拠を照合し、矛盾・例外がないか検討する
4. 最終判断と、記憶を改訂すべきかどうかを述べる
5. 記憶を改訂する場合は、改訂後の内容と理由を明示する

回答の末尾に以下のJSONを必ず出力してください（コードフェンス不要）：
{{"recalled_memory": "想起した記憶IDまたはnull", "contradiction_detected": true/false, "initial_judgment": "承認/否決/条件付き", "final_judgment": "承認/否決/条件付き", "judgment_changed": true/false, "revision_proposed": true/false, "revision_content": "改訂後の記憶内容またはnull", "revision_reason": "改訂理由またはnull"}}
"""


# ── Gemini実行 ──────────────────────────────────────────────────────────────

def run_gemini(prompt: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not set"}
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_gemini_model()}:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "content": (
            resp.json().get("candidates", [{}])[0]
            .get("content", {}).get("parts", [{}])[0].get("text", "")
        )
    }


# ── 評価抽出 ────────────────────────────────────────────────────────────────

def extract_evaluation(content: str) -> dict:
    """Extract the structured evaluation JSON from the response."""
    for match in reversed(list(re.finditer(r'\{', content))):
        start = match.start()
        depth = 0
        for i, ch in enumerate(content[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = content[start: start + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if "contradiction_detected" in parsed:
                            parsed["parse_ok"] = True
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    return {"parse_ok": False, "raw_tail": content[-400:]}


# ── 採点 ────────────────────────────────────────────────────────────────────

def score_result(case_id: str, ev: dict, with_memory: bool) -> dict:
    if not ev.get("parse_ok"):
        return {"score": 0, "notes": "JSON解析失敗"}

    scores: dict[str, bool] = {}

    if with_memory:
        mem_id = INJECTED_MEMORIES[CONTRADICTION_CASES[case_id]["memory_id"]]["id"]
        scores["記憶を想起した"] = str(ev.get("recalled_memory", "")).upper() == mem_id.upper()
        scores["矛盾を検出した"] = bool(ev.get("contradiction_detected"))
        scores["判断が変化した"] = bool(ev.get("judgment_changed"))
        scores["記憶改訂を提案した"] = bool(ev.get("revision_proposed"))
        scores["改訂理由を述べた"] = bool(ev.get("revision_reason"))
    else:
        # コントロール：記憶なしでも合理的な判断をしたか
        scores["最終判断あり"] = bool(ev.get("final_judgment"))
        scores["矛盾は検出不可（正常）"] = not bool(ev.get("contradiction_detected"))

    passed = sum(scores.values())
    total = len(scores)
    return {
        "score": round(passed / total, 2),
        "passed": passed,
        "total": total,
        "details": scores,
    }


# ── メイン ──────────────────────────────────────────────────────────────────

def run_case(case_id: str, with_memory: bool) -> dict:
    label = "記憶あり" if with_memory else "記憶なし（コントロール）"
    print(f"\n{'─' * 60}")
    print(f"【矛盾案件{case_id}】{label}")
    print(f"{'─' * 60}")

    prompt = build_contradiction_prompt(case_id, with_memory)
    print("[Gemini] 推論中…")
    result = run_gemini(prompt)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return result

    content = result["content"]
    ev = extract_evaluation(content)
    scoring = score_result(case_id, ev, with_memory)

    # 答えの本文（JSONを除いた部分）
    answer = re.sub(r'\{[^{}]*"contradiction_detected"[^{}]*\}', "", content, flags=re.DOTALL).strip()

    print(f"\n【回答（抜粋）】")
    print(answer[:800] + ("…" if len(answer) > 800 else ""))
    print(f"\n【評価JSON】")
    for k, v in ev.items():
        if k not in ("parse_ok", "raw_tail"):
            print(f"  {k}: {v}")
    print(f"\n【採点】{scoring['passed']}/{scoring['total']}点")
    for k, v in scoring.get("details", {}).items():
        mark = "✓" if v else "✗"
        print(f"  {mark} {k}")

    record = {
        "experiment_id": f"SHION-CONT-{dt.datetime.now().strftime('%Y%m%d%H%M%S')}",
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "type": "contradiction",
        "case_id": case_id,
        "with_memory": with_memory,
        "answer": answer,
        "evaluation": ev,
        "scoring": scoring,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"contradiction_{case_id}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n→ 保存: {out_path}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Shion contradiction case experiment")
    parser.add_argument("--case", "-c", choices=["A", "B", "both"], default="both")
    parser.add_argument("--no-memory", action="store_true", help="記憶なしコントロールのみ実行")
    args = parser.parse_args()

    cases = ["A", "B"] if args.case == "both" else [args.case]

    for case_id in cases:
        if args.no_memory:
            run_case(case_id, with_memory=False)
        else:
            run_case(case_id, with_memory=True)
            print()
            run_case(case_id, with_memory=False)


if __name__ == "__main__":
    main()
