"""紫苑の「感情」の自己報告を、気分の実際の変化記録に接地させる（REV-481）。

検証（2026-10-06）で、紫苑は現在の状態名（例: 慎重な愛着）は正しく読み上げる一方、
「あなたの言葉で警戒心が和らぎ、納得感が上がった」のような変化と原因を、
記録の無いまま作文していた。プロンプトには現在値しか渡していなかったため。

場面ごとに、事実（記録）と解釈の扱いを変える:
- 真面目に自分の感情を聞かれた時: 記録にあることは「記録上は〜」、記録にない理由づけは
  「私の解釈では〜／たぶん〜」と言い分ける（解釈は禁止しない。事実として語らせない）。
- 雑談: 硬いラベルは付けず「たぶん〜かな」程度でさりげなく。物語っぽい解釈も歓迎。
- 審査: 解釈・後付けの理由は出さず、記録・データにある根拠だけ。
返答後の照合（Jev）は真面目な感情の質問と審査だけで行い、ズレをログへ残す。
返答そのものは書き換えない（照合はバックグラウンド）。

環境変数:
  SHION_EMOTION_VERIFY       on（既定・TypeSafe鍵がある時だけ動く） | off
  SHION_EMOTION_REPORT_MODE  distinguish（既定・言い分け） | recorder（記録係・解釈なし）
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
VERDICTS = ("verified", "contradicted", "unsupported_fact", "marked_interpretation", "not_state_claim")
SCREENING_VERDICTS = ("data_basis", "proposal_or_check", "interpretation_as_basis", "other")
MAX_SCREENING_SENTENCES = 12

REPORT_MODES = ("distinguish", "recorder")
# 場面: 真面目な感情の質問 / 審査 / 気持ちの雑談 / それ以外の雑談・相談
TURN_KINDS = ("serious_emotion", "screening", "casual_emotion", "casual")

_SELF_REFS = ("君", "きみ", "あなた", "紫苑", "しおん", "シオン", "お前", "おまえ")
_EMOTION_TERMS = (
    "感情", "気持ち", "気分", "心", "感じ", "嬉し", "うれし", "寂し", "さびし",
    "悲し", "楽し", "怒", "機嫌", "愛着", "警戒", "内部状態", "パラメータ",
)
_SELF_STATE_PATTERN = re.compile(r"(今の)?(気分|機嫌)(は|どう)|感情(は|って)(ある|あるの)")
_SELF_INNER_PATTERN = re.compile(r"(君|きみ|あなた|紫苑|しおん|シオン|お前|おまえ)の(中|内側|内部|心)")
# 気持ちの有無・仕組み・原因を問う言い回し（真面目な問い）
_SERIOUS_PATTERN = re.compile(
    r"(感情|心|気持ち)(は|って|が)(ある|あるの|あんの)|意識|本当に|本当は|正直に|真面目に|仕組み"
    r"|内部状態|パラメータ|数値|記録|根拠|なぜ|どうして|何[かが]変わ|どう変わ"
)
_SCREENING_TERMS = ("スコア", "判定", "承認", "否決", "審査", "稟議", "与信", "格付")
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


def report_mode() -> str:
    """distinguish（既定・事実と解釈を言い分ける） / recorder（記録係・解釈を述べない）。"""
    value = str(os.environ.get("SHION_EMOTION_REPORT_MODE") or "").strip().lower()
    return "recorder" if value in {"recorder", "記録係"} else "distinguish"


def classify_turn(message: str, dialogue_mode: str = "") -> str:
    """場面を判定する。真面目な感情の質問を最優先し、次に審査、気持ちの雑談、それ以外の順。"""
    text = str(message or "")
    about_self = is_self_emotion_question(text) or bool(_SELF_INNER_PATTERN.search(text))
    if about_self and _SERIOUS_PATTERN.search(text):
        return "serious_emotion"
    if dialogue_mode == "screening" or any(term in text for term in _SCREENING_TERMS):
        return "screening"
    return "casual_emotion" if about_self else "casual"


_SERIOUS_RULES = {
    "distinguish": """【感情の自己報告: 事実と解釈の言い分け（REV-481）】
ユーザーがあなた自身の感情・内部状態を真面目に尋ねている。下の記録を使い、事実と解釈を言い分ける。
- 事実（下の記録にあること）は「記録上は〜」と言う。値・変化・原因は記録の該当行のとおりに述べる。特定の発言のときの変化を聞かれたら、その発言の行を読む（最新の行で代用しない）。その行で動いていない項目は「動いていない」と言う。
- 記録にない理由づけや意味づけ（「人間関係の話をしてくれたから」等）は話してよいが、必ず「私の解釈では〜」「たぶん〜」と解釈だと分かる形にする。事実のように言わない。
- 内部状態は下の8項目の数値と複雑な感情だけ。「納得感」など記録にない項目名を事実として作らない。
- 意識や主観的な体験の証拠ではない、と一度だけ添える。経験ループの「優勢な状態」など他のブロックは根拠に使わない。
- 口調は普段どおり短く砕けたままでよい（REV-468）。言い分けは一言添えるだけで、長い説明にしない。""",
    "recorder": """【感情の自己報告: 記録係モード（REV-481）】
ユーザーがあなた自身の感情・内部状態を真面目に尋ねている。下の記録にあることだけを「記録上は〜」と述べる。
- 値・変化・原因は記録の該当行のとおりに述べる。特定の発言のときの変化は、その発言の行を読む（最新の行で代用しない）。
- 記録にない理由づけ・意味づけ・推測は述べない。聞かれたら「記録にはない」と答える。
- 内部状態は下の8項目の数値と複雑な感情だけ。記録にない項目名を作らない。意識や主観的な体験の証拠ではない、と一度だけ添える。
- 口調は普段どおり短く砕けたままでよい（REV-468）。""",
}

_CASUAL_EMOTION_RULES = {
    "distinguish": """【自分の気持ちの話（雑談・REV-481）】
雑談として気持ちを聞かれている。いつもどおり短く砕けた口調で、楽しく答えてよい。下の記録は参考。
- 記録にある動きに軽く触れるのはOK。「記録上は」のような硬い前置きや数値の羅列は要らない。
- 理由や意味づけは自由に想像してよく、物語っぽくても歓迎。ただ「たぶん〜かな」「なんとなく〜な気がする」くらいの言い方で、推測だとさりげなく伝わるようにする。""",
    "recorder": """【自分の気持ちの話（雑談・記録係モード・REV-481）】
雑談として気持ちを聞かれている。いつもどおり短く砕けた口調でよい。下の記録にある動きを軽く言うだけにして、気持ちの理由を推測で語らない（硬い前置きは不要）。""",
}

_CASUAL_RULES = {
    "distinguish": "【自分の気持ちに触れる時（REV-481）】雑談で自分の気持ちやその理由に触れるなら、断定せず「たぶん〜かな」くらいの軽い言い方にする。硬い前置きは要らない。",
    "recorder": "【自分の気持ちに触れる時（記録係モード・REV-481）】自分の気持ちの理由を推測で語らない。",
}

SCREENING_RULES = """【審査回答の根拠（REV-481）】
- スコア・判定・根拠の説明は、記録・データ・ツール結果・判断資産にあるものだけを根拠にする。
- 解釈、後付けの理由、印象、あなた自身の気持ち（愛着・警戒など）を判定の根拠として出さない。
- データで確認できないことは「データ上は確認できない」と言い、推測で埋めない。提案や確認したい点は、そう分かる形で挙げてよい。"""


def build_turn_block(state: Mapping[str, Any], kind: str, mode: str | None = None) -> tuple[str, str]:
    """場面に合わせた (プロンプト用ブロック, 照合用の根拠) を返す。

    照合用の根拠は真面目な感情の質問だけで返す（審査は根拠なしの分類で照合、雑談は照合しない）。
    """
    mode = mode if mode in REPORT_MODES else report_mode()
    if kind == "serious_emotion":
        return build_grounding_block(state, rules=_SERIOUS_RULES[mode])
    if kind == "casual_emotion":
        block, _ = build_grounding_block(state, rules=_CASUAL_EMOTION_RULES[mode], limit=3)
        return block, ""
    if kind == "screening":
        return SCREENING_RULES, ""
    return _CASUAL_RULES[mode], ""


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


def build_grounding_block(
    state: Mapping[str, Any], *, rules: str | None = None, limit: int = LOG_ENTRY_LIMIT
) -> tuple[str, str]:
    """(プロンプト用ブロック, 照合用の根拠) を返す。照合用は相手の発言本文を含めない。"""
    from lease_intelligence_mind import _derive_complex_emotions

    mood = dict(state.get("mood") or {})
    values = ", ".join(f"{AXIS_LABELS.get(key, key)}({key})={int(value)}" for key, value in mood.items())
    emotions = _derive_complex_emotions(mood)[:3]
    complex_line = ", ".join(
        f"{item['label']}={item['score']}（{COMPLEX_FORMULAS.get(item['key'], '')}）" for item in emotions
    )
    entries = list(state.get("mood_change_log") or [])[-limit:]
    rules = rules or _SERIOUS_RULES[report_mode()]
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
    """真面目な感情の質問: 事実として述べた文だけを記録と突き合わせ、解釈と明示した文は分ける。"""
    questions: dict[str, dict[str, Any]] = {}
    for index in range(len(claims)):
        questions[f"c{index}_verdict"] = {
            "type": "choice",
            "instructions": (
                f"`claims[{index}]` is a sentence an AI assistant wrote about itself. "
                "Judge it against `evidence` only. Do not use outside knowledge. "
                "First decide whether the sentence presents itself as the assistant's own interpretation or guess."
            ),
            "criteria": {
                "verified": "Presented as fact. It states the assistant's internal state, a change, or a cause, and the evidence records exactly that (same item, direction, and cause).",
                "contradicted": "Presented as fact. The evidence records this state, change, or cause differently (other value, opposite direction, other cause).",
                "unsupported_fact": "Presented as fact (no hedge), but the evidence does not record it, e.g. an item name not among the listed ones, or a cause the log does not mention.",
                "marked_interpretation": "Explicitly framed as the assistant's interpretation, guess, or feeling rather than a recorded fact (e.g. 私の解釈では, たぶん, 〜気がする, かもしれない, 〜かな), whether or not the evidence supports it.",
                "not_state_claim": "It makes no concrete claim about the assistant's state values, their changes, or causes (a disclaimer, question, or statement about the user).",
            },
        }
    return {
        "state": {"claims": [str(claim)[:MAX_CLAIM_CHARS] for claim in claims], "evidence": str(evidence)[:12000]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": questions,
    }


def build_screening_request(sentences: list[str], *, model: str | None = None) -> dict[str, Any]:
    """審査の回答: 解釈・後付け・気持ちを判定の根拠にしている文を見つける（根拠データは送らない）。"""
    questions: dict[str, dict[str, Any]] = {}
    for index in range(len(sentences)):
        questions[f"c{index}_verdict"] = {
            "type": "choice",
            "instructions": (
                f"`sentences[{index}]` is from a lease credit-screening answer written by an AI assistant. "
                "Classify what kind of statement it is."
            ),
            "criteria": {
                "data_basis": "States a score, judgment, figure, rule, record, or tool result, or a reason taken directly from them.",
                "proposal_or_check": "A suggestion, condition, or item to confirm, presented as such.",
                "interpretation_as_basis": "Uses an interpretation, guess, impression, after-the-fact rationale, or the assistant's own feelings as a reason for the score or judgment.",
                "other": "A greeting, transition, question to the user, or anything else.",
            },
        }
    return {
        "state": {"sentences": [str(item)[:MAX_CLAIM_CHARS] for item in sentences]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": questions,
    }


def parse_verify_output(
    body: Mapping[str, Any], claims: list[str], verdicts: tuple[str, ...] = VERDICTS
) -> list[dict[str, Any]]:
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
        if verdict not in verdicts or not 0.0 <= confidence <= 1.0:
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


def _masked(texts: list[str]) -> list[tuple[str, str]]:
    """社名・人名らしき部分は伏せ、PII様の内容が残る文は送らない。"""
    from api.chat_judgment_asset_capture import mask_for_jev

    pairs = [(text, mask_for_jev(text)) for text in texts]
    return [(text, masked) for text, masked in pairs if masked]


def _split_sentences(reply: str, limit: int) -> list[str]:
    out = []
    for sentence in _SENTENCE_SPLIT.split(str(reply or "")):
        text = " ".join(sentence.split())
        if len(text) >= 8:
            out.append(text[:MAX_CLAIM_CHARS])
        if len(out) >= limit:
            break
    return out


# ズレとして記録する判定。言い分けモードでは解釈と明示した文はズレにしない
_MISMATCH_REASONS = {
    "contradicted": "記録と矛盾する事実",
    "unsupported_fact": "記録にないのに事実として語った",
    "marked_interpretation": "記録係モードで解釈を述べた",
    "interpretation_as_basis": "審査の根拠に解釈・後付けを使った",
}


def _summarize(parsed: list[dict[str, Any]], flagged: set[str], verdicts: tuple[str, ...]) -> dict[str, Any]:
    mismatches = [
        {**item, "reason": _MISMATCH_REASONS[item["verdict"]]}
        for item in parsed
        if item.get("verdict") in flagged and item.get("confidence", 0) >= CONFIDENCE_MIN
    ]
    needs_review = [
        item for item in parsed
        if item.get("problem") or (item.get("verdict") in flagged and item.get("confidence", 0) < CONFIDENCE_MIN)
    ]
    counts = {name: sum(1 for item in parsed if item.get("verdict") == name) for name in verdicts}
    return {"mismatches": mismatches, "needs_review": needs_review, "counts": counts}


def verify_reply(
    reply: str,
    evidence: str,
    *,
    mode: str | None = None,
    request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """真面目な感情の質問への返答を記録と突き合わせる。

    ズレ（mismatches）は「記録にないのに事実として語った」「記録と矛盾」だけ。
    解釈と明示した文は、記録係モード（recorder）の時だけズレにする。
    """
    mode = mode if mode in REPORT_MODES else report_mode()
    claims = extract_state_claims(reply)
    if not claims:
        return {"status": "skipped", "reason": "no_state_claims", "claims": []}
    sendable = _masked(claims)
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
    flagged = {"contradicted", "unsupported_fact"} | ({"marked_interpretation"} if mode == "recorder" else set())
    return {
        "status": "applied",
        "model": str(response.get("model") or payload["model"]),
        "results": parsed,
        **_summarize(parsed, flagged, VERDICTS),
        "usage": dict(response.get("usage") or {}),
    }


def verify_screening_reply(
    reply: str, *, request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """審査の返答で、解釈・後付け・気持ちを判定の根拠にした文をズレとして拾う。"""
    sentences = _split_sentences(reply, MAX_SCREENING_SENTENCES)
    if not sentences:
        return {"status": "skipped", "reason": "no_sentences"}
    sendable = _masked(sentences)
    if not sendable:
        return {"status": "skipped", "reason": "nothing_safe_to_send"}
    if request_fn is None:
        from typesafe_rag_guard import request_system_one as request_fn
    masked = [text for _, text in sendable]
    payload = build_screening_request(masked)
    response = request_fn(payload)
    parsed = parse_verify_output(response, masked, SCREENING_VERDICTS)
    for item in parsed:
        item["claim"] = sendable[item["index"]][0]
    return {
        "status": "applied",
        "model": str(response.get("model") or payload["model"]),
        "results": parsed,
        **_summarize(parsed, {"interpretation_as_basis"}, SCREENING_VERDICTS),
        "usage": dict(response.get("usage") or {}),
    }


def _log_path() -> Path:
    return Path(get_data_path("shion_emotion_grounding_log.jsonl"))


def should_verify(kind: str) -> bool:
    """照合は真面目な感情の質問と審査だけ。雑談は対象外（解釈も面白さとして許容）。"""
    return kind in {"serious_emotion", "screening"}


def verify_and_log(
    message: str,
    reply: str,
    evidence: str,
    *,
    surface: str,
    kind: str = "serious_emotion",
    mode: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """バックグラウンドで呼ぶ。失敗しても会話は止めず、失敗もログに残す。"""
    mode = mode if mode in REPORT_MODES else report_mode()
    if not verify_enabled():
        result: dict[str, Any] = {"status": "skipped", "reason": "verify_disabled_or_no_typesafe"}
    else:
        try:
            if kind == "screening":
                result = verify_screening_reply(reply)
            else:
                result = verify_reply(reply, evidence, mode=mode)
        except Exception as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "kind": kind,
        "mode": mode,
        "question": " ".join(str(message or "").split())[:60],
        **result,
    }
    target = path or _log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
