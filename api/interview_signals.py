"""面談メモ・現場メモから経営者の定性シグナルを推定する（REV-470）。

REV-464 の気持ち推定（api/user_affect.py）と同じく、語彙と否定の手がかりだけで決定的に推定する
（追加の API 呼び出し・待ち時間なし）。対象は審査入力の「現場メモ」（passion_text）。

- 結果は審査画面と紫苑の審査コメントの「参考情報」専用。スコア・審査ロジック・判定には一切混ぜない。
- 各シグナルには根拠としてメモの該当文を引用し、推測であることを明示する。
- 年齢・性別・国籍・健康など偏見につながる属性に触れる文は、手がかりにも引用にも使わない。
- 環境変数 SHION_INTERVIEW_SIGNALS_ENABLED=0 で無効化できる（既定は有効）。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

DISCLAIMER = (
    "メモの言葉づかいから機械的に推測した参考情報です。外れることがあり、"
    "スコア・審査判定には反映していません。確認論点のきっかけとしてだけ使ってください。"
)

# シグナル → (表示名, 語と重み)。語はメモを書いた担当者が経営者の様子を描写する表現を想定する。
_LEXICON: dict[str, tuple[str, tuple[tuple[str, float], ...]]] = {
    "anxiety": ("不安", (
        ("不安", 1.0), ("心配", 1.0), ("気にして", 0.8), ("懸念", 0.8), ("迷って", 0.7), ("悩んで", 0.7),
        ("自信がな", 1.0), ("自信なさ", 1.0), ("弱気", 1.0), ("表情が硬", 0.8), ("言葉に詰ま", 0.9),
        ("先行きが見え", 0.9), ("資金繰りを気に", 1.0), ("落ち着かな", 0.8), ("焦りが見え", 0.6),
    )),
    "confidence": ("自信", (
        ("自信", 1.0), ("即答", 1.0), ("堂々", 1.0), ("明確", 0.8), ("具体的", 0.7), ("前向き", 0.7),
        ("意欲", 0.7), ("熱意", 0.7), ("手応え", 0.8), ("見通しが立", 0.9), ("計画どおり", 0.7),
        ("計画通り", 0.7), ("数字で説明", 1.0), ("積極的", 0.6), ("受注が決ま", 0.8), ("確信", 1.0),
    )),
    "urgency": ("切迫感", (
        ("急ぎ", 1.0), ("至急", 1.2), ("早急", 1.0), ("急いで", 1.0), ("今月中", 0.9), ("今週中", 1.0),
        ("月末まで", 0.8), ("すぐに", 0.6), ("一刻も", 1.2), ("間に合", 0.8), ("期限", 0.5), ("納期", 0.5),
        ("支払いが迫", 1.2), ("資金がショート", 1.2), ("決済を急", 1.0), ("早く", 0.5), ("焦って", 1.0),
    )),
    "vagueness": ("説明の曖昧さ", (
        ("曖昧", 1.0), ("あいまい", 1.0), ("はっきりしな", 1.0), ("明言を避", 1.2), ("言葉を濁", 1.2),
        ("濁して", 1.0), ("はぐらか", 1.2), ("回答を避", 1.2), ("答えられな", 1.0), ("説明できな", 1.0),
        ("よく分からな", 0.8), ("よくわからな", 0.8), ("具体性に欠", 1.0), ("ざっくり", 0.6),
        ("なんとなく", 0.6), ("未定", 0.5), ("確認中", 0.4), ("たぶん", 0.4), ("多分", 0.4), ("らしい", 0.3),
        ("後日回答", 0.6), ("資料が出な", 1.0), ("資料を出さな", 1.0),
    )),
    "inconsistency": ("説明の食い違い", (
        ("矛盾", 1.2), ("食い違", 1.2), ("前回と違", 1.2), ("話が変わ", 1.0), ("説明が変わ", 1.0),
        ("二転三転", 1.5), ("辻褄が合わ", 1.2), ("整合しな", 1.0), ("言い直", 0.6), ("訂正", 0.5),
        ("言っていることが違", 1.2), ("資料と違", 1.0), ("数字が合わ", 1.0),
    )),
    "consistency": ("説明の一貫性", (
        ("一貫", 1.0), ("説明が一致", 1.0), ("資料と一致", 1.0), ("整合して", 1.0), ("ぶれな", 1.0),
        ("ブレな", 1.0), ("前回と同じ説明", 1.0), ("裏付け", 0.6),
    )),
}

# 打ち消し: 語の直後の否定（「不安はない」「自信がない」）・直前の否定接頭辞（「不明確」「未具体的」）
_NEGATION_RE = re.compile(r"^.{0,4}?(ない|なく|ません|じゃな|ではな|無い|無く)")
_NEGATION_PREFIX = "不未非無"
# 否定を含む語自体（「自信がな」等）は打ち消し判定をしない
_SELF_NEGATED = re.compile(r"(な|ない|避|欠|濁|わ)$")

# 偏見につながる属性に触れる文は丸ごと使わない
_PROTECTED_RE = re.compile(
    r"\d+\s*(歳|才|代)|年齢|高齢|年配|若い|若手|老齢|女性|男性|女社長|性別|国籍|外国人|外国籍|人種|民族|"
    r"宗教|信仰|障害|障がい|病気|持病|妊娠|出身地|出身国|既婚|未婚|離婚|独身|子ども|子供"
)
_SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")
_MIN_SCORE = 0.6
_MAX_QUOTES = 3
_QUOTE_LEN = 80


def interview_signals_enabled() -> bool:
    return os.environ.get("SHION_INTERVIEW_SIGNALS_ENABLED", "1") != "0"


@dataclass
class InterviewSignal:
    key: str
    label: str
    score: float = 0.0
    evidence: list[dict[str, str]] = field(default_factory=list)

    @property
    def level(self) -> str:
        return "強" if self.score >= 2.0 else "中" if self.score >= 1.0 else "弱"

    def to_payload(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "level": self.level,
                "score": round(self.score, 2), "evidence": self.evidence[:_MAX_QUOTES]}


def _hits(sentence: str, word: str) -> int:
    hits, start = 0, 0
    while (idx := sentence.find(word, start)) >= 0:
        start = idx + len(word)
        if not _SELF_NEGATED.search(word):
            if idx > 0 and sentence[idx - 1] in _NEGATION_PREFIX:
                continue
            tail = sentence[start:start + 8]
            if _NEGATION_RE.match(tail) and not tail.startswith("しかな"):
                continue
        hits += 1
    return hits


def _quote(sentence: str, word: str) -> str:
    s = sentence.strip()
    if len(s) <= _QUOTE_LEN:
        return s
    idx = max(0, s.find(word) - _QUOTE_LEN // 2)
    return ("…" if idx else "") + s[idx: idx + _QUOTE_LEN] + "…"


def extract_interview_signals(text: str) -> dict[str, Any]:
    """メモから定性シグナルを推定し、根拠の引用つきで返す。手がかりが弱いシグナルは返さない。"""
    memo = str(text or "")[:4000]
    sentences = [s for s in (m.group(0).strip() for m in _SENTENCE_RE.finditer(memo)) if s]
    usable = [s for s in sentences if not _PROTECTED_RE.search(s)]
    signals = {key: InterviewSignal(key, label) for key, (label, _) in _LEXICON.items()}
    for sentence in usable:
        for key, (_, entries) in _LEXICON.items():
            cues = [(w, wt, n) for w, wt in entries if (n := _hits(sentence, w))]
            if not cues:
                continue
            sig = signals[key]
            sig.score += sum(wt * min(n, 2) for _, wt, n in cues)
            sig.evidence.append({"quote": _quote(sentence, cues[0][0]), "cue": "・".join(w for w, _, _ in cues[:3])})
    found = sorted((s for s in signals.values() if s.score >= _MIN_SCORE), key=lambda s: -s.score)
    return {
        "signals": [s.to_payload() for s in found],
        "sentence_count": len(sentences),
        "excluded_sentence_count": len(sentences) - len(usable),
        "disclaimer": DISCLAIMER,
        "affects_score": False,
    }


def build_interview_signals_prompt_block(payload: dict[str, Any]) -> str:
    """紫苑の審査コメント用の参考ブロック。シグナルがなければ空文字。"""
    signals = payload.get("signals") or []
    if not signals:
        return ""
    lines = ["【現場メモからの定性シグナル（言葉づかいからの推測・参考情報）】"]
    for s in signals:
        quotes = " / ".join(f"「{e['quote']}」" for e in s.get("evidence", [])[:2])
        lines.append(f"- {s['label']}（{s['level']}）根拠: {quotes}")
    lines.append(
        "- これはスコア・判定に入れていない推測。採否やスコアの根拠にせず、触れる場合はメモの該当箇所を引用して"
        "「〜という記述から〜の可能性（推測）」と書き、面談で確かめる確認論点として扱う。人柄や属性の評価に広げない。"
    )
    return "\n".join(lines)
