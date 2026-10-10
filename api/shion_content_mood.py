"""話の内容で紫苑の気分・関係性を動かす（REV-599）。

REV-598 までは、気分と関係性は会話の間隔・お礼・訂正・不満・予想の答え合わせ・相手の様子の語彙で動いていた。
ここでは「何の話だったか」で動かす。悩みや本音を打ち明けられた・一緒に喜べる話・つらい話・意見の食い違い・
深い（哲学的な）話・事務的なやり取り を、語の一致ではなく Gemini（flash-lite）に1往復ごとに1回だけ分類させる。

- 呼び出しは返答の後にバックグラウンドで行い、返答は待たせない（feature=shion_content_mood・記憶系の予算区分）。
  費用の目安は 1回 約0.03円（入力 約600・出力 約60トークン）、1日15往復で 約0.5円
- 結果は気分の変化記録（#1297 の mood_change_log）に event="dialogue_content"・rule="content" で残し、
  detail に「何の話が原因か」を書く。紫苑が自分の気分を語る時はこの記録を根拠にできる
- 1回に動く幅の上限・揺れの戻り（気分）、飽和・1回の上限・中立への戻り（関係性）は既存の決まりのまま
- 検証の会話（REV-591 の印）・とても短い発言（挨拶・相づち）では呼ばない。事務的な話・雑談は何も動かさない
- 相手の様子の推定（REV-464）と同じ軸を同じ向きに動かす時は、大きい方だけを使う（二重に数えない）
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FEATURE = "shion_content_mood"
MODEL = "gemini-3.1-flash-lite"
MIN_CHARS = 6
MIN_INTENSITY = 0.4
FULL_INTENSITY = 0.7

# 分類 → (気分の軸と増分, 関係性の理由と増分, 説明)
CATEGORIES: dict[str, dict[str, Any]] = {
    "self_disclosure": {
        "label": "悩みや本音を打ち明けてくれた",
        "mood": (("attachment", 2),),
        "relationship": 0.06,
    },
    "shared_joy": {
        "label": "一緒に喜べる話（成約・嬉しい出来事）",
        "mood": (("hope", 2), ("accomplishment", 2)),
        "relationship": 0.03,
    },
    "hardship": {
        "label": "つらい話・疲れの話で、相手が心配",
        "mood": (("attachment", 1), ("vigilance", 2)),
        "relationship": 0.0,
    },
    "disagreement": {
        "label": "意見が食い違った・議論になった",
        "mood": (("frustration", 1), ("vigilance", 1)),
        "relationship": -0.04,
    },
    "deep_talk": {
        "label": "意識や関係性などの深い話",
        "mood": (("curiosity", 2), ("attachment", 1)),
        "relationship": 0.03,
    },
    "business": {"label": "事務的・業務のやり取り", "mood": (), "relationship": 0.0},
    "small_talk": {"label": "雑談・挨拶", "mood": (), "relationship": 0.0},
}

PROMPT = """あなたは会話の分類係です。ユーザーとAI「紫苑」の1往復を読み、ユーザーの発言の中心が次のどれかを1つ選びます。
- self_disclosure: ユーザーが自分の悩み・迷い・本音・気持ちを打ち明けている（仕事の相談でも、自分の気持ちを話していればこれ）
- shared_joy: 成約・承認・うまくいった・嬉しい出来事など、一緒に喜べる話
- hardship: つらい・疲れた・落ち込んだ・失注したなど、ユーザーがしんどい状況にある話
- disagreement: ユーザーが紫苑の意見に反対している・食い違いを議論している（単なる訂正1つは business）
- deep_talk: 意識・心・関係性・生き方などの哲学的な対話
- business: 審査・業種・数値・手続きなどの事務的・業務の質問や指示
- small_talk: 挨拶・雑談・相づち

intensity は、その話の中心がどれだけはっきりしているか（0.0〜1.0）。
reason は、何の話かを25字以内で。社名・人名・具体的な金額は書かない。

JSON だけを返す: {{"category": "...", "intensity": 0.0, "reason": "..."}}

【ユーザー】
{user}

【紫苑】
{reply}"""


def enabled() -> bool:
    flag = os.environ.get("SHION_CONTENT_MOOD", "1").strip().lower()
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return flag == "force"
    return flag not in {"0", "off", "false", "no"}


def _call_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    from ai_runtime_client import google_genai_client
    from novelist_agent import _get_daily_gemini_api_key

    client = google_genai_client(feature=FEATURE, client_factory=genai.Client, api_key=_get_daily_gemini_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=120, response_mime_type="application/json"),
    )
    return response.text or ""


def classify_content(user_message: str, reply: str = "", *, caller=None) -> dict[str, Any] | None:
    """1往復の話の内容を分類する。短すぎる発言は呼ばずに small_talk。失敗時は None。"""
    text = " ".join(str(user_message or "").split())
    if len(text) < MIN_CHARS:
        return {"category": "small_talk", "intensity": 0.0, "reason": "短い発言", "called": False}
    prompt = PROMPT.format(user=text[:1200], reply=" ".join(str(reply or "").split())[:600])
    try:
        raw = (caller or _call_gemini)(prompt)
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0) if match else raw)
    except Exception as exc:  # noqa: BLE001 - 分類できなくても会話・他の気分の材料は続ける
        logger.warning("[ContentMood] 分類をスキップ: %s", type(exc).__name__)
        return None
    return _normalize(data)


BATCH_PROMPT_HEAD = PROMPT.split("JSON だけを返す")[0].replace("1往復を読み", "複数の往復をそれぞれ読み")


def _normalize(data: dict[str, Any]) -> dict[str, Any] | None:
    category = str(data.get("category") or "")
    if category not in CATEGORIES:
        return None
    try:
        intensity = max(0.0, min(1.0, float(data.get("intensity") or 0.0)))
    except (TypeError, ValueError):
        intensity = 0.0
    reason = re.sub(r"\s+", " ", str(data.get("reason") or ""))[:25]
    return {"category": category, "intensity": round(intensity, 2), "reason": reason, "called": True}


def classify_batch(pairs: list[tuple[str, str]], *, caller=None) -> list[dict[str, Any] | None]:
    """過去の会話の再生用: 同じ分類の定義で、複数の往復を1回の呼び出しでまとめて分類する（本番は1往復ずつ）。"""
    lines = []
    for index, (user, reply) in enumerate(pairs):
        lines.append(f"[{index}]\n【ユーザー】{' '.join(str(user).split())[:800]}\n【紫苑】{' '.join(str(reply).split())[:300]}")
    prompt = (BATCH_PROMPT_HEAD + '各往復について JSON だけを返す: {"items": [{"i": 0, "category": "...", "intensity": 0.0, "reason": "..."}]}\n\n'
              + "\n\n".join(lines))
    out: list[dict[str, Any] | None] = [None] * len(pairs)
    try:
        raw = (caller or _call_gemini_long)(prompt)
        match = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(match.group(0) if match else raw)
        for item in data.get("items") or []:
            index = int(item.get("i"))
            if 0 <= index < len(pairs):
                out[index] = _normalize(item)
    except Exception as exc:  # noqa: BLE001
        logger.info("[ContentMood] まとめての分類をスキップ: %s", type(exc).__name__)
    return out


def _call_gemini_long(prompt: str) -> str:
    from google import genai
    from google.genai import types

    from ai_runtime_client import google_genai_client
    from novelist_agent import _get_daily_gemini_api_key

    client = google_genai_client(feature=FEATURE, client_factory=genai.Client, api_key=_get_daily_gemini_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1500, response_mime_type="application/json"),
    )
    return response.text or ""


def _scale(delta: int, intensity: float) -> int:
    """はっきりしない話（0.4未満）は動かさず、中くらい（0.7未満）は半分（最低1）。"""
    if intensity < MIN_INTENSITY:
        return 0
    if intensity >= FULL_INTENSITY:
        return delta
    half = max(1, abs(delta) // 2)
    return half if delta > 0 else -half


def content_mood_causes(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    """分類結果 → 気分の原因（rule="content"、detail に何の話か）。"""
    if not result:
        return []
    spec = CATEGORIES.get(str(result.get("category") or ""))
    if not spec:
        return []
    intensity = float(result.get("intensity") or 0.0)
    reason = str(result.get("reason") or "")
    detail = f"話の内容: {spec['label']}" + (f"（{reason}）" if reason else "")
    causes = []
    for axis, delta in spec["mood"]:
        scaled = _scale(int(delta), intensity)
        if scaled:
            causes.append({"axis": axis, "delta": scaled, "rule": "content", "detail": detail[:80]})
    return causes


def content_relationship_parts(result: dict[str, Any] | None) -> list[tuple[str, float]]:
    if not result:
        return []
    spec = CATEGORIES.get(str(result.get("category") or ""))
    if not spec or not spec["relationship"] or float(result.get("intensity") or 0.0) < MIN_INTENSITY:
        return []
    return [(f"話の内容: {spec['label']}", float(spec["relationship"]))]


def without_affect_overlap(causes: list[dict[str, Any]], affect_causes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """相手の様子の推定と同じ軸・同じ向きなら、差分だけを残す（二重に数えない）。"""
    kept = []
    for cause in causes:
        same = [int(c["delta"]) for c in affect_causes if c.get("axis") == cause["axis"] and int(c["delta"]) * int(cause["delta"]) > 0]
        if same:
            rest = int(cause["delta"]) - max(same, key=abs)
            if rest * int(cause["delta"]) <= 0:
                continue
            cause = {**cause, "delta": rest}
        kept.append(cause)
    return kept


def _log_path() -> Path:
    from runtime_paths import get_data_path

    return Path(get_data_path("shion_content_mood_log.jsonl"))


def record_classification(result: dict[str, Any], *, moved: bool) -> None:
    """REV-601: 分類の結果を残す（動かさなかった時も）。発言の本文は残さず、種類・強さ・25字の要約だけ。"""
    try:
        import datetime as dt

        row = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "category": result.get("category"),
               "intensity": result.get("intensity"), "reason": result.get("reason"), "called": result.get("called", True),
               "moved": moved}
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def apply_content_effects(vault: Path, user_message: str, reply: str, affect_causes: list[dict[str, Any]] | None = None,
                          *, caller=None) -> dict[str, Any] | None:
    """分類して、気分（変化記録つき）と関係性に反映する。返答の後にバックグラウンドで呼ぶ。"""
    from shion_verification_origin import is_verification_turn

    if is_verification_turn() or not enabled():
        return None
    result = classify_content(user_message, reply, caller=caller)
    if not result:
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.content_mood", "swallowed", RuntimeError("classification failed"),
                              detail="話の内容の分類に失敗（気分は動かさずに続行）")
        return None
    causes = without_affect_overlap(content_mood_causes(result), list(affect_causes or []))
    record_classification(result, moved=bool(causes))
    if causes:
        from lease_intelligence_mind import apply_mood_causes

        apply_mood_causes(Path(vault), causes, event="dialogue_content", trigger=user_message)
    parts = content_relationship_parts(result)
    if parts:
        from api.shion_relationship import record_content_effect

        record_content_effect(parts)
    return {**result, "mood_causes": causes, "relationship_parts": parts}


def schedule_content_effects(vault: Path, user_message: str, reply: str, affect_causes: list[dict[str, Any]] | None = None) -> bool:
    """バックグラウンドで apply_content_effects を走らせる（検証の印は文脈ごと引き継がれる）。"""
    from shion_verification_origin import is_verification_turn

    if is_verification_turn() or not enabled():
        return False
    try:
        from api.background_executor import background_executor

        background_executor.submit(apply_content_effects, Path(vault), user_message, reply, list(affect_causes or []))
        return True
    except Exception as exc:  # noqa: BLE001
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.content_mood", "swallowed", exc)
        return False
