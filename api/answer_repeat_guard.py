"""同じ質問への丸写しを防ぐ（REV-544）。

2026-10-09 の調査（Obsidian「紫苑の仕組み_2026-10/13_答えが固まる原因調査」）で、
同じ質問を聞き直すと、紫苑自身の前回の答えが (1) 会話履歴 (2) RAG に入った対話室の会話ログ
の2経路でプロンプトに戻り、Gemini が一字一句写していた。記憶は削らず、渡し方だけを変える。

- A: 会話履歴に同じ（ほぼ同じ）質問がある時だけ、繰り返さない指示を足す
- B: 対話室の会話ログが RAG に当たったら「過去の紫苑自身の回答・写さない」の印を付け、
     会話履歴にある答えと同じ中身なら本文を省く。開発用ノートは参照ナレッジに入れない
AI 呼び出しはしない。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

# B: 紫苑自身の回答が書かれるフォルダ（lease_intelligence_dialogue.append_dialogue_note の書き先）
SELF_ANSWER_PATH_MARKERS = ("Lease Intelligence/Dialogue/",)
# B: 開発用ノート（仕組みの説明・調査記録）。紫苑の回答の材料にしない
DEV_NOTE_PATH_MARKERS = ("tune_lease_55/紫苑の仕組み_",)

SELF_ANSWER_START = "〔過去の紫苑自身の回答・写さない〕"
SELF_ANSWER_END = "〔過去の回答ここまで〕"
SELF_ANSWER_OMITTED = "〔過去の紫苑自身の回答: 会話履歴にある答えと同じ中身のため省略〕"
# 参照ナレッジ側の切り詰め（_append_rag_hits は ref 込み600字）でも終わりの印が残る長さ
_SELF_ANSWER_BODY_CHARS = 480

REPEAT_QUESTION_BLOCK = (
    "【同じ質問の聞き直し】\n"
    "この質問（またはほぼ同じ質問）には、会話履歴の中ですでに答えている。"
    "前回の答えの文面・構成・見出しをそのまま繰り返さない。"
    "前回の要点は必要なら一言でまとめ、今回は新しい観点・参考メモ（最近のニュース等）・前回から変わった情報を中心に答える。"
    "結論が前回と同じでよい時も、言い回しと例を変える。"
)

_REPEAT_MIN_CHARS = 4
_REPEAT_RATIO = 0.85
_OVERLAP_WINDOW = 30


def _path_text(path: str) -> str:
    return unicodedata.normalize("NFC", str(path or "").replace("\\", "/"))


def is_dev_note_path(path: str) -> bool:
    text = _path_text(path)
    return any(marker in text for marker in DEV_NOTE_PATH_MARKERS)


def is_self_answer_path(path: str) -> bool:
    text = _path_text(path)
    return any(marker in text for marker in SELF_ANSWER_PATH_MARKERS)


def mark_self_answer_text(text: str) -> str:
    """会話ログのチャンクに始まりと終わりの印を付ける（2回付けない）。"""
    body = str(text or "")
    if SELF_ANSWER_START in body:
        return body
    return f"{SELF_ANSWER_START}{body.strip()[:_SELF_ANSWER_BODY_CHARS]}{SELF_ANSWER_END}"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"[\s\W_]+", "", text).lower()


def find_repeated_question(message: str, history: list[dict[str, Any]] | None) -> str:
    """履歴のユーザー発言に今回とほぼ同じものがあれば、その発言を返す。無ければ空文字。"""
    current = _normalize(message)
    if len(current) < _REPEAT_MIN_CHARS:
        return ""
    for item in reversed(list(history or [])):
        if str(item.get("role") or "") != "user":
            continue
        previous = _normalize(str(item.get("content") or ""))
        if not previous:
            continue
        if previous == current or difflib.SequenceMatcher(None, previous, current).ratio() >= _REPEAT_RATIO:
            return str(item.get("content") or "")
    return ""


def _overlaps_history(body: str, assistant_text: str) -> bool:
    """会話ログの答えの部分が、履歴の紫苑の発言に含まれているか（30字の窓で照合）。"""
    answer = body.split("リース知性体", 1)[-1]
    norm = _normalize(answer)
    if len(norm) < _OVERLAP_WINDOW or not assistant_text:
        return False
    step = max(1, (len(norm) - _OVERLAP_WINDOW) // 3)
    windows = [norm[i : i + _OVERLAP_WINDOW] for i in range(0, len(norm) - _OVERLAP_WINDOW + 1, step)][:4]
    return any(window in assistant_text for window in windows)


def drop_self_answers_in_history(system_prompt: str, history: list[dict[str, Any]] | None) -> tuple[str, int]:
    """印の付いた会話ログのうち、履歴の紫苑の発言と同じ中身のものの本文を省く。"""
    text = str(system_prompt or "")
    if SELF_ANSWER_START not in text:
        return text, 0
    assistant_text = "".join(
        _normalize(str(m.get("content") or "")) for m in (history or []) if str(m.get("role") or "") != "user"
    )
    pattern = re.compile(re.escape(SELF_ANSWER_START) + r"(.*?)" + re.escape(SELF_ANSWER_END), re.S)
    dropped = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal dropped
        if _overlaps_history(match.group(1), assistant_text):
            dropped += 1
            return SELF_ANSWER_OMITTED
        return match.group(0)

    return pattern.sub(_replace, text), dropped


def apply_repeat_guard(system_prompt: str, history: list[dict[str, Any]] | None, message: str) -> str:
    """Gemini へ送る直前のシステムプロンプトに A・B（履歴との重複）を当てる。"""
    prompt, _dropped = drop_self_answers_in_history(system_prompt, history)
    if find_repeated_question(message, history):
        prompt = f"{prompt.rstrip()}\n\n{REPEAT_QUESTION_BLOCK}"
    return prompt
