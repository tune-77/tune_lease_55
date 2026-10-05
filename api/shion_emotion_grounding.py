"""紫苑の「感情」の自己報告を、気分の実際の変化記録に接地させる（REV-481）。

検証（2026-10-06）で、紫苑は現在の状態名（例: 慎重な愛着）は正しく読み上げる一方、
「あなたの言葉で警戒心が和らぎ、納得感が上がった」のような変化と原因を、
記録の無いまま作文していた。プロンプトには現在値しか渡していなかったため。

- 自分の感情を聞かれた時だけ、気分の変化記録（lease_intelligence_mind の
  mood_change_log）と「記録にない変化・項目は語らない」指示をプロンプトへ渡す。
- 返答後、状態についての主張を記録と突き合わせ（Jev）、ズレをログへ残す。
  返答そのものは書き換えない（照合はバックグラウンド）。

環境変数:
  SHION_EMOTION_VERIFY  on（既定・TypeSafe鍵がある時だけ動く） | off
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

AXIS_LABELS = {
    "weariness": "疲労",
    "curiosity": "好奇心",
    "attachment": "愛着",
    "vigilance": "警戒",
    "hope": "希望",
    "frustration": "不満",
    "loneliness": "孤独",
    "accomplishment": "達成感",
}

RULE_LABELS = {
    "dialogue_visit": "話しかけてくれた",
    "content_keyword": "相手の発言の言葉",
    "user_affect": "相手の様子の推定",
    "prediction_error": "相手の予想の答え合わせ",
    "relationship": "関係性スコア",
    "screening_event": "審査の出来事",
    "dissonance": "未解決の不整合の検知",
    "memory_baseline": "最近の会話記憶の言葉の集計",
    "decay": "揺れの自然な戻り",
    "daily_decay": "日替わりの半減",
    "catch_up": "以前の変化の続き",
}

# 出来事ではなく時間・集計で起きる変化。記録では1行にまとめて示す
_DRIFT_RULES = {"decay", "daily_decay", "catch_up", "memory_baseline"}

EVENT_LABELS = {"dialogue": "対話", "daily": "日次更新", "dissonance": "不整合の検知"}

# 複雑な感情は lease_intelligence_mind._derive_complex_emotions の計算式そのもの
COMPLEX_FORMULAS = {
    "hopeful_anxiety": "希望と警戒の平均",
    "careful_attachment": "愛着と警戒の平均",
    "intellectual_excitement": "好奇心と希望の平均",
    "unrewarded_effort": "疲労と不満の平均",
    "quiet_loneliness": "孤独と疲労の平均",
    "earned_confidence": "達成感と希望の平均",
    "protective_frustration": "愛着・不満・警戒の平均",
}

LOG_ENTRY_LIMIT = 6
# chat_prompt_budget の emotion_grounding_context 予算（3200字）より小さく保つ
MAX_BLOCK_CHARS = 3000
MAX_LINE_CHARS = 900
MAX_CLAIMS = 8
MAX_CLAIM_CHARS = 300
CONFIDENCE_MIN = 0.60
VERDICTS = ("verified", "contradicted", "unsupported", "not_state_claim")

_SELF_REFS = ("君", "きみ", "あなた", "紫苑", "しおん", "シオン", "お前", "おまえ")
_EMOTION_TERMS = (
    "感情", "気持ち", "気分", "心", "感じ", "嬉し", "うれし", "寂し", "さびし",
    "悲し", "楽し", "怒", "機嫌", "愛着", "警戒", "内部状態", "パラメータ",
)
_SELF_STATE_PATTERN = re.compile(r"(今の)?(気分|機嫌)(は|どう)|感情(は|って)(ある|あるの)")
_STATE_TERMS = (
    *AXIS_LABELS.values(), *AXIS_LABELS.keys(), "状態", "パラメータ", "感情", "気分", "数値",
    "和らい", "上昇", "高ま", "下が", "上が", "強く出", "揺れ", "愛着", "期待と不安", "知的高揚",
    "報われなさ", "静かな孤独", "手応え", "苛立ち",
)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])|\n+")
_LOG_LOCK = threading.Lock()


def is_self_emotion_question(message: str) -> bool:
    """紫苑自身の感情・気持ち・内部状態を尋ねる発言か（他人の気持ちの話は除く）。"""
    text = str(message or "")
    if not any(term in text for term in _EMOTION_TERMS):
        return False
    return any(ref in text for ref in _SELF_REFS) or bool(_SELF_STATE_PATTERN.search(text))


def _cause_text(cause: Mapping[str, Any]) -> str:
    rule = str(cause.get("rule") or "")
    detail = str(cause.get("detail") or RULE_LABELS.get(rule, rule))
    return f"{detail} {int(cause.get('delta') or 0):+d}"


def _entry_line(entry: Mapping[str, Any], *, include_trigger: bool) -> str:
    ts = str(entry.get("ts") or "")
    when = ts[5:16].replace("T", " ") if len(ts) >= 16 else ts
    event = str(entry.get("event") or "")
    label = EVENT_LABELS.get(event, "審査の出来事" if event.startswith("screening:") else event)
    trigger = str(entry.get("trigger") or "")
    head = f"- {when} {label}" + (f"「{trigger}」" if include_trigger and trigger else "")
    changes = list(entry.get("changes") or [])
    if not changes:
        return f"{head}: 変化なし"
    explained, drift = [], []
    for change in changes:
        axis = str(change.get("axis") or "")
        causes = list(change.get("causes") or [])
        move = f"{AXIS_LABELS.get(axis, axis)} {change.get('before')}→{change.get('after')}"
        if any(str(cause.get("rule")) not in _DRIFT_RULES for cause in causes):
            explained.append(f"{move}（{'／'.join(_cause_text(cause) for cause in causes)}）")
        else:
            drift.append(move)
    parts = explained[:]
    if drift:
        parts.append("ほかは小さな自然変化（揺れの戻り・記憶の基調・目標値への追いつき）: " + "、".join(drift))
    return f"{head}: " + "。".join(parts)


def build_grounding_block(state: Mapping[str, Any], *, limit: int = LOG_ENTRY_LIMIT) -> tuple[str, str]:
    """(プロンプト用ブロック, 照合用の根拠) を返す。照合用は相手の発言本文を含めない。"""
    from lease_intelligence_mind import _derive_complex_emotions

    mood = dict(state.get("mood") or {})
    values = ", ".join(f"{AXIS_LABELS.get(key, key)}({key})={int(value)}" for key, value in mood.items())
    emotions = _derive_complex_emotions(mood)[:3]
    complex_line = ", ".join(
        f"{item['label']}={item['score']}（{COMPLEX_FORMULAS.get(item['key'], '')}）" for item in emotions
    )
    entries = list(state.get("mood_change_log") or [])[-limit:]
    rules = """【感情の自己報告の根拠（REV-481）】
ユーザーがあなた自身の感情・気持ち・内部状態を尋ねている。答えるときは、下の記録だけを根拠にする。
- 内部状態は下の8項目の数値（0〜100の演出的パラメータ）と、そこから計算した複雑な感情だけ。意識や主観的な体験の証拠ではない、と一度だけ添える。
- 語ってよい項目名は下の8項目と複雑な感情の名前だけ。「納得感」など記録にない項目を作らない。
- 変化を言う時は、変化記録の該当する行の数値と向きだけを述べる。特定の発言のときの変化を聞かれたら、その発言の行を探して読む（最新の行で代用しない）。その行で動いていない項目は「動いていない」と言う。
- 原因は、その行の括弧内に書かれた原因（例:「話しかけてくれた +1」「相手の様子の推定: 疲れ +1」）をそのまま使う。括弧内に無い原因、とくに相手の発言の意味や内容（「人間関係の話をしてくれたから」等）を原因にしない。話の内容で変わったかを聞かれたら「記録上の原因は〇〇だけで、話の内容による変化は記録されていない」と答える。
- 記録にない変化や原因は語らない。聞かれたら「記録上は変化していない」「原因は記録されていない」と答える。
- 経験ループの「優勢な状態」など他のブロックは、この自己報告の根拠に使わない。"""
    if entries:
        log_lines = [_entry_line(entry, include_trigger=True)[:MAX_LINE_CHARS] for entry in entries]
        evidence_log = [_entry_line(entry, include_trigger=False)[:MAX_LINE_CHARS] for entry in entries]
    else:
        log_lines = evidence_log = ["- 記録なし（この仕組みを入れてから、まだ気分は動いていない）"]
    head = [rules, f"現在の値: {values}", f"複雑な感情（上位3・計算式）: {complex_line}", "直近の気分の変化記録（古い順）:"]
    # プロンプト予算で指示行が削られないよう、ここで古い記録から落として収める
    while len(log_lines) > 1 and len("\n".join([*head, *log_lines])) > MAX_BLOCK_CHARS:
        log_lines = log_lines[1:]
        evidence_log = evidence_log[1:]
    block = "\n".join([*head, *log_lines])
    evidence = "\n".join(
        [
            "AIアシスタントの内部状態の記録。項目は次の8つだけで、他の項目は存在しない: "
            + "、".join(f"{label}({key})" for key, label in AXIS_LABELS.items()),
            f"現在の値: {values}",
            f"複雑な感情（上位3・計算式）: {complex_line}",
            "直近の変化記録（古い順。ここに無い変化・原因は記録されていない）:",
            *evidence_log,
        ]
    )
    return block, evidence


def extract_state_claims(reply: str, *, limit: int = MAX_CLAIMS) -> list[str]:
    """返答から、自分の状態・変化・原因について述べていそうな文を取り出す。"""
    claims: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(str(reply or "")):
        text = " ".join(sentence.split())
        if len(text) < 8 or not any(term in text for term in _STATE_TERMS):
            continue
        claims.append(text[:MAX_CLAIM_CHARS])
        if len(claims) >= limit:
            break
    return claims


def build_verify_request(claims: list[str], evidence: str, *, model: str | None = None) -> dict[str, Any]:
    questions: dict[str, dict[str, Any]] = {}
    for index in range(len(claims)):
        questions[f"c{index}_verdict"] = {
            "type": "choice",
            "instructions": (
                f"`claims[{index}]` is a sentence an AI assistant wrote about itself. "
                "Judge it against `evidence` only. Do not use outside knowledge."
            ),
            "criteria": {
                "verified": "It states the assistant's internal state, a change, or a cause, and the evidence records exactly that (same item, direction, and cause).",
                "contradicted": "It states a state, change, or cause that the evidence records differently (other value, opposite direction, other cause).",
                "unsupported": "It states a state item, change, or cause that the evidence does not record, e.g. an item name not among the listed ones, or a cause attributed to something the log does not mention.",
                "not_state_claim": "It makes no concrete claim about the assistant's recorded state values, their changes, or causes (a disclaimer, metaphor, question, or statement about the user).",
            },
        }
    return {
        "state": {"claims": [str(claim)[:MAX_CLAIM_CHARS] for claim in claims], "evidence": str(evidence)[:12000]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": questions,
    }


def parse_verify_output(body: Mapping[str, Any], claims: list[str]) -> list[dict[str, Any]]:
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("TypeSafe verify response is missing answers")
    results: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        base = {"index": index, "claim": claim}
        answer = answers.get(f"c{index}_verdict")
        if not isinstance(answer, Mapping):
            results.append({**base, "problem": "missing_answer"})
            continue
        verdict = str(answer.get("choice") or "").strip().lower()
        try:
            confidence = float(answer.get("confidence"))
        except (TypeError, ValueError):
            confidence = math.nan
        if verdict not in VERDICTS or not 0.0 <= confidence <= 1.0:
            results.append({**base, "problem": "invalid_answer"})
            continue
        results.append({**base, "verdict": verdict, "confidence": round(confidence, 3)})
    return results


def verify_enabled() -> bool:
    if str(os.environ.get("SHION_EMOTION_VERIFY") or "on").strip().lower() in {"0", "off", "false", "no"}:
        return False
    try:
        from typesafe_rag_guard import typesafe_available

        return bool(typesafe_available())
    except Exception:
        return False


def verify_reply(
    reply: str, evidence: str, *, request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """返答中の状態についての主張を記録と突き合わせる。ズレは mismatches に入る。"""
    claims = extract_state_claims(reply)
    if not claims:
        return {"status": "skipped", "reason": "no_state_claims", "claims": []}
    from api.chat_judgment_asset_capture import mask_for_jev

    # 社名・人名らしき部分は伏せ、PII様の内容が残る文は送らない
    sendable = [(claim, mask_for_jev(claim)) for claim in claims]
    sendable = [(claim, masked) for claim, masked in sendable if masked]
    if not sendable:
        return {"status": "skipped", "reason": "nothing_safe_to_send", "claims": claims}
    if request_fn is None:
        from typesafe_rag_guard import request_system_one as request_fn
    masked_claims = [masked for _, masked in sendable]
    payload = build_verify_request(masked_claims, evidence)
    response = request_fn(payload)
    parsed = parse_verify_output(response, masked_claims)
    for item in parsed:
        item["claim"] = sendable[item["index"]][0]
    mismatches = [
        item for item in parsed
        if item.get("verdict") in {"contradicted", "unsupported"} and item.get("confidence", 0) >= CONFIDENCE_MIN
    ]
    needs_review = [
        item for item in parsed
        if item.get("problem")
        or (item.get("verdict") in {"contradicted", "unsupported"} and item.get("confidence", 0) < CONFIDENCE_MIN)
    ]
    counts = {name: sum(1 for item in parsed if item.get("verdict") == name) for name in VERDICTS}
    return {
        "status": "applied",
        "model": str(response.get("model") or payload["model"]),
        "results": parsed,
        "mismatches": mismatches,
        "needs_review": needs_review,
        "counts": counts,
        "usage": dict(response.get("usage") or {}),
    }


def _log_path() -> Path:
    return Path(get_data_path("shion_emotion_grounding_log.jsonl"))


def verify_and_log(message: str, reply: str, evidence: str, *, surface: str, path: Path | None = None) -> dict[str, Any]:
    """バックグラウンドで呼ぶ。失敗しても会話は止めず、失敗もログに残す。"""
    if not verify_enabled():
        result: dict[str, Any] = {"status": "skipped", "reason": "verify_disabled_or_no_typesafe"}
    else:
        try:
            result = verify_reply(reply, evidence)
        except Exception as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "question": " ".join(str(message or "").split())[:60],
        **result,
    }
    target = path or _log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
