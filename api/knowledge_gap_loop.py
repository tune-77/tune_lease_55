"""
ナレッジ穴探しループ（Knowledge Gap Loop Engineering）。

Observe   : data/case_memory_usage_log.jsonl（/api/chat 等が記録する、
            質問ごとの knowledge_refs 参照件数ログ）
Aggregate : knowledge_refs が空だった質問（＝知識ソースが薄いまま
            答えた質問）を集める
Propose   : 頻出する話題をGeminiに分析させ、外部調査器官
            （/api/research-organ/run）へ回すべき調査トピックを提案させる
Persist   : data/knowledge_gap_proposals.jsonl

安全設計: このループはObsidian Vaultへの書き込みや外部調査の自動実行は
行わない。提案は「何を調べるとよいか」の候補どまりで、実行は人間が
/research-organ 画面から判断する。
"""
from __future__ import annotations

from typing import Any

from api.loop_engineering_common import DATA_DIR, append_jsonl, call_gemini_json, load_jsonl

_USAGE_LOG_PATH = DATA_DIR / "case_memory_usage_log.jsonl"
_PROPOSALS_PATH = DATA_DIR / "knowledge_gap_proposals.jsonl"


def aggregate_gaps(limit_examples: int = 20) -> dict[str, Any]:
    entries = load_jsonl(_USAGE_LOG_PATH, newest_first=True)
    weak_questions: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        refs = entry.get("knowledge_refs") or []
        if refs:
            continue
        question = str(entry.get("question_preview") or "").strip()
        if not question or question in seen:
            continue
        seen.add(question)
        weak_questions.append(question)
        if len(weak_questions) >= limit_examples:
            break
    return {
        "total_logged": len(entries),
        "weak_coverage_count": sum(1 for e in entries if not (e.get("knowledge_refs") or [])),
        "weak_questions": weak_questions,
    }


def _build_prompt(aggregate: dict[str, Any]) -> str:
    questions_lines = "\n".join(f"- {q}" for q in aggregate["weak_questions"]) or "（該当データなし）"

    return f"""あなたはリース審査AIシステム「紫苑」です。ユーザーからの質問のうち、
Obsidianナレッジの参照が0件のまま回答した（＝知識ソースが薄いまま答えた）
質問の一覧を分析し、Obsidian Vaultに知識として補うべきトピックを考えるのが
役目です。

【知識参照0件だった質問の一覧（重複除去済み・最大20件）】
{questions_lines}

これらに共通する話題・分野を分析し、外部調査器官（Google検索経由でResearch
ノートを作る仕組み）で調べるべきトピックを2〜4件、以下のJSON配列形式のみで
返してください（前後の説明テキストは不要）:

[
  {{
    "topic": "調査すべきトピック（30字以内）",
    "reason": "なぜこのトピックの知識が不足していると考えたか（100字程度、質問例を含めてよい）",
    "search_hint": "調査器官に渡すとよい検索キーワード案"
  }}
]"""


def generate_proposals() -> dict[str, Any]:
    aggregate = aggregate_gaps()
    if aggregate["total_logged"] == 0:
        return {"generated": False, "reason": "利用ログデータがまだありません", "proposals": []}
    if not aggregate["weak_questions"]:
        return {
            "generated": False,
            "reason": "知識参照0件の質問は見つかりませんでした（良好）",
            "proposals": [],
        }

    prompt = _build_prompt(aggregate)
    try:
        proposals = call_gemini_json(prompt)
        if not isinstance(proposals, list):
            raise ValueError("Gemini応答がリストではありません")
    except Exception as exc:
        return {"generated": False, "reason": f"Gemini生成に失敗: {exc}", "proposals": []}

    import datetime as dt

    generated_at = dt.datetime.now().isoformat(timespec="seconds")
    saved: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict) or not str(item.get("topic") or "").strip():
            continue
        entry = {
            "title": str(item.get("topic") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
            "search_hint": str(item.get("search_hint") or "").strip(),
            "generated_at": generated_at,
            "status": "needs_human_review",
        }
        append_jsonl(_PROPOSALS_PATH, entry)
        saved.append(entry)

    return {"generated": True, "aggregate": aggregate, "proposals": saved}


def load_proposals(limit: int = 20) -> list[dict[str, Any]]:
    return load_jsonl(_PROPOSALS_PATH, limit=limit, newest_first=True)
