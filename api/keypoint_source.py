"""会話キーポイントの出所（user / shion / unknown）を決める（REV-544）。

キーポイントは対話室のやり取り（ユーザー発言＋紫苑の返答）から抜き出すため、
紫苑自身の言ったことが「以前教わった〜」として次の回答に戻っていた（自己強化）。
判断資産の出所（REV-498 / #1317 の content_source）と同じ考え方で、要点にも出所を残す。
AI 呼び出しはせず、要点の文字の並び（2文字単位）がどちらの発言に多く含まれるかで決める。
決めきれないものは unknown（不明）にする。
"""

from __future__ import annotations

import re
import unicodedata

USER = "user"
SHION = "shion"
UNKNOWN = "unknown"

# 出所ごとの表示。user は従来どおり（教わった知識として扱ってよい）
SOURCE_LABELS = {SHION: "紫苑の発言", UNKNOWN: "出所不明"}
NOT_TAUGHT_NOTE = (
    "（「紫苑の発言」「出所不明」と付いた要点は、ユーザーから教わった知識ではない。"
    "「以前教わった」「教えてもらった」と言わず、使う時は「前に私が整理した見方では」のように言う）"
)

_USER_MIN = 0.5
_SHION_MIN = 0.5
_USER_MAX_FOR_SHION = 0.35


def _bigrams(text: str) -> set[str]:
    norm = re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", str(text or ""))).lower()
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


def _coverage(point: set[str], text: str) -> float:
    if not point:
        return 0.0
    return len(point & _bigrams(text)) / len(point)


def keypoint_content_source(point: str, user_message: str, reply: str) -> str:
    """要点がユーザーの発言由来か、紫苑の返答由来かを決める。決めきれなければ unknown。"""
    grams = _bigrams(point)
    if len(grams) < 3:
        return UNKNOWN
    from_user = _coverage(grams, user_message)
    from_shion = _coverage(grams, reply)
    if from_user >= _USER_MIN:
        return USER
    if from_shion >= _SHION_MIN and from_user < _USER_MAX_FOR_SHION:
        return SHION
    return UNKNOWN


def normalized_source(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in {USER, SHION} else UNKNOWN


def source_suffix(value: object) -> str:
    """Memory ノート・想起メモで要点の後ろに付ける出所の表示（user は付けない）。"""
    label = SOURCE_LABELS.get(normalized_source(value))
    return f"（{label}）" if label else ""
