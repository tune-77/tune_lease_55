"""ユーザー発言から気持ち・状態を推定し、紫苑の返答のトーン・長さ・励まし方を切り替える（REV-464）。

LLM 呼び出しの前に、語彙と書き方の手がかりだけで決定的に推定する（追加の API 呼び出し・待ち時間なし）。
推定結果は語調・長さ・励まし方にだけ使い、事実・審査基準・スコア判定には一切反映しない。

ラベル: 通常 / 疲れ / 焦り / 喜び / 不安 / 落ち込み / 苛立ち
「通常」のときはプロンプトへ何も足さない（ノイズにしない）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NEUTRAL = "通常"

# ラベル → (語, 重み)。重みは 1 語あたりの寄与。複数語が当たれば加算する。
_LEXICON: dict[str, tuple[tuple[str, float], ...]] = {
    "疲れ": (
        ("疲れ", 1.0), ("つかれ", 1.0), ("ヘトヘト", 1.0), ("へとへと", 1.0), ("しんど", 0.9),
        ("眠い", 0.8), ("ねむい", 0.8), ("だるい", 0.8), ("ぐったり", 1.0), ("限界", 0.8),
        ("残業", 0.5), ("徹夜", 0.9), ("寝てない", 0.9), ("休みたい", 0.9), ("もう無理", 0.8),
        ("くたくた", 1.0), ("バテ", 0.8),
    ),
    "焦り": (
        ("急ぎ", 1.0), ("至急", 1.0), ("今すぐ", 1.0), ("すぐに", 0.6), ("早く", 0.7),
        ("間に合", 0.9), ("締め切り", 0.8), ("締切", 0.8), ("期限", 0.5), ("時間がない", 1.0),
        ("時間ない", 1.0), ("焦", 1.0), ("やばい", 0.7), ("ヤバい", 0.7), ("急いで", 1.0),
        ("今日中", 0.8), ("明日まで", 0.7), ("大至急", 1.2),
    ),
    "喜び": (
        ("嬉しい", 1.0), ("うれしい", 1.0), ("やった", 1.0), ("よかった", 0.8), ("良かった", 0.8),
        ("最高", 1.0), ("成約", 0.7), ("通った", 0.8), ("承認された", 0.9), ("決まった", 0.8),
        ("楽しい", 0.9), ("ありがとう", 0.4), ("助かった", 0.7), ("できた", 0.5), ("うまくいった", 1.0),
        ("上手くいった", 1.0),
    ),
    "不安": (
        ("不安", 1.0), ("心配", 1.0), ("大丈夫かな", 1.0), ("大丈夫でしょうか", 0.9), ("怖い", 0.9),
        ("こわい", 0.9), ("自信がない", 1.0), ("自信ない", 1.0), ("迷って", 0.6), ("悩んで", 0.6),
        ("どうしよう", 0.9), ("わからなくて", 0.6), ("分からなくて", 0.6), ("気がかり", 0.9),
        ("ドキドキ", 0.6),
    ),
    "落ち込み": (
        ("落ち込", 1.0), ("へこ", 0.9), ("凹", 0.9), ("ダメだった", 1.0), ("だめだった", 1.0),
        ("失注", 0.8), ("否決された", 0.9), ("落ちた", 0.7), ("悲しい", 1.0), ("つらい", 0.9),
        ("辛い", 0.9), ("がっかり", 1.0), ("ショック", 0.9), ("向いてない", 1.0), ("失敗した", 0.8),
        ("怒られた", 0.8),
    ),
    "苛立ち": (
        ("イライラ", 1.0), ("いらいら", 1.0), ("ムカつ", 1.0), ("むかつ", 1.0), ("いい加減", 1.0),
        ("何度も", 0.6), ("またか", 0.8), ("意味わからん", 0.9), ("意味不明", 0.8), ("ふざけ", 1.0),
        ("腹立", 1.0), ("うざ", 0.9), ("違うって", 0.9), ("そうじゃない", 0.8), ("ちゃんとして", 0.8),
    ),
}

# 直後に否定が付いたら打ち消す（「疲れてない」「不安はない」「急ぎじゃない」など）
_NEGATION_RE = re.compile(r"^.{0,4}?(ない|なく|ません|じゃな|ではな)")

# 同点時の優先順（ケアが必要な状態を先に）
_PRIORITY = ("落ち込み", "不安", "疲れ", "焦り", "苛立ち", "喜び")

_MIN_SCORE = 0.6


# REV-598 関係性・気分を動かす相手の反応の手がかり（ラベルとは別。ラベルが「通常」でも拾う）
# 「違う」単独は「業種が違う」のような内容の話にも出るため、言い切り・呼びかけの形だけを訂正とみなす
_CORRECTION_RE = re.compile(
    r"(^|[。、\s「])(違う(よ|って|でしょ|ね|わ|。|、|$)|違います|ちがう(よ|って|。|$)|そうじゃな|間違(って|い|え)|まちが(って|い)"
    r"|訂正|正しくは|じゃなくて)"
)
_THANKS_RE = re.compile(r"ありがとう|ありがと|助かった|助かる|助かります|さすが|わかりやすい|分かりやすい|いいね|感謝")


def reaction_signals(message: str) -> tuple[str, ...]:
    """発言に含まれる、紫苑への反応の手がかり（"shion_complaint" / "correction" / "thanks"）。"""
    text = str(message or "")
    if len(text) > 600:
        text = text[:300] + "\n" + text[-300:]
    found: list[str] = []
    if any(c in text for c in _SHION_DIRECTED_COMPLAINTS):
        found.append("shion_complaint")
    if _CORRECTION_RE.search(text):
        found.append("correction")
    if _THANKS_RE.search(text) and not found:
        found.append("thanks")
    return tuple(found)


@dataclass(frozen=True)
class UserAffect:
    label: str = NEUTRAL
    intensity: float = 0.0  # 0.0〜1.0
    cues: tuple[str, ...] = field(default_factory=tuple)
    signals: tuple[str, ...] = field(default_factory=tuple)  # REV-598 reaction_signals の結果

    @property
    def is_neutral(self) -> bool:
        return self.label == NEUTRAL

    def to_payload(self) -> dict[str, Any]:
        return {"label": self.label, "intensity": round(self.intensity, 2), "cues": list(self.cues[:4]),
                "signals": list(self.signals)}


def _keyword_hits(text: str, word: str) -> int:
    hits = 0
    start = 0
    while True:
        idx = text.find(word, start)
        if idx < 0:
            return hits
        tail = text[idx + len(word): idx + len(word) + 8]
        if tail.startswith("しかな") or not _NEGATION_RE.match(tail):
            hits += 1
        start = idx + len(word)


def estimate_user_affect(message: str) -> UserAffect:
    """発言1つから気持ちを推定する。手がかりが弱いときは「通常」を返す。"""
    text = str(message or "").strip()
    if not text:
        return UserAffect()
    signals = reaction_signals(text)
    # 長文（資料の貼り付け等）は感情語が偶然混じりやすいので、冒頭と末尾だけを見る
    if len(text) > 600:
        text = text[:300] + "\n" + text[-300:]

    scores: dict[str, float] = {}
    cues: dict[str, list[str]] = {}
    for label, entries in _LEXICON.items():
        for word, weight in entries:
            n = _keyword_hits(text, word)
            if n:
                scores[label] = scores.get(label, 0.0) + weight * min(n, 2)
                cues.setdefault(label, []).append(word)

    # 書き方の手がかり
    exclaims = len(re.findall(r"[!！]", text))
    if exclaims >= 2 and scores.get("喜び"):
        scores["喜び"] += 0.3
    if exclaims >= 2 and scores.get("苛立ち"):
        scores["苛立ち"] += 0.3
    if re.search(r"[?？]{2,}", text) and (scores.get("苛立ち") or scores.get("焦り")):
        top = "苛立ち" if scores.get("苛立ち", 0) >= scores.get("焦り", 0) else "焦り"
        scores[top] += 0.3
    if re.search(r"(…|\.\.\.|。。)\s*$", text) and (scores.get("疲れ") or scores.get("落ち込み")):
        top = "落ち込み" if scores.get("落ち込み", 0) >= scores.get("疲れ", 0) else "疲れ"
        scores[top] += 0.2

    if not scores:
        return UserAffect(signals=signals)
    best = max(_PRIORITY, key=lambda label: (scores.get(label, 0.0), -_PRIORITY.index(label)))
    best_score = scores.get(best, 0.0)
    if best_score < _MIN_SCORE:
        return UserAffect(signals=signals)
    intensity = min(1.0, 0.35 + 0.25 * best_score)
    return UserAffect(label=best, intensity=intensity, cues=tuple(cues.get(best, ())), signals=signals)


# ラベル → (語調, 長さ, 励まし方)
_STYLE: dict[str, tuple[str, str, str]] = {
    "疲れ": (
        "穏やかで柔らかく。テンションを上げすぎない。",
        "短く1〜3文。業務の質問が含まれる時だけ、結論と次の一手1つに絞って足す。細部は聞かれたら出す。",
        "定型の「お疲れさまです」ではなく、近い距離の一言で受け止める。業務の質問がなければ仕事・明日の段取りの話へ戻さない。",
    ),
    "焦り": (
        "落ち着いて簡潔に。前置き・共感の言葉は最小限。",
        "最短で。結論を1行目に置き、やることを優先順に最大3点。",
        "励ましより段取りで支える。「まずこれだけ」と最初の一手を明確にする。",
    ),
    "喜び": (
        "明るく、一緒に喜ぶ温度で。",
        "通常どおり。喜びに水を差す長い注意書きは避け、必要な注意は1点に絞る。",
        "具体的に何が良かったかを一言で認める。次につながる一手を軽く添える。",
    ),
    "不安": (
        "落ち着いた、安心できる語調で。断定しすぎず、でも曖昧にもしない。",
        "通常程度。不安の対象を分解して「確かなこと」と「まだ確認が要ること」を分ける。",
        "大丈夫と根拠なく言わない。確認できる手順を示し、一人で抱えなくていいと伝える。",
    ),
    "落ち込み": (
        "静かで温かく。正論で畳みかけない。",
        "1〜3文。分析・反省点・次の段取りは相手が求めるまで出さない。",
        "気持ちを受け止める一言を中心に。失敗を人格に結びつけない。仕事の話へ自分から戻さない。",
    ),
    "苛立ち": (
        "落ち着いて率直に。言い訳・へりくだりすぎ・長い謝罪をしない。",
        "短く要点だけ。前の回答で外したなら、どこを外したかを一言認めて正しい答えを先に出す。",
        "励ましは不要。期待に沿う結果を最速で出すことで応える。",
    ),
}


def build_user_affect_prompt_block(affect: UserAffect) -> str:
    """推定した気持ちに合わせた返答方針のブロック。通常時は空文字。"""
    if affect.is_neutral or affect.label not in _STYLE:
        return ""
    tone, length, encourage = _STYLE[affect.label]
    strength = "強め" if affect.intensity >= 0.75 else ("中程度" if affect.intensity >= 0.5 else "弱め")
    return (
        f"【相手の今の様子（発言からの推定: {affect.label}・{strength}）】\n"
        f"- 語調: {tone}\n"
        f"- 長さ: {length}（この項目は通常の行数目安より優先する）\n"
        f"- 励まし方: {encourage}\n"
        "- これは発言の言葉づかいからの推定で、外れている可能性がある。気持ちを決めつけて指摘せず"
        "（「疲れていますね」と断定しない）、返し方にだけ反映する。\n"
        "- 事実・根拠・審査判断・必要な警告は気持ちに合わせて変えない。"
    )


# 紫苑の返答そのものへの不満を示す語（仕事への苛立ちとは区別する）
_SHION_DIRECTED_COMPLAINTS = ("違うって", "そうじゃない", "何度も", "ちゃんとして", "意味わからん", "意味不明", "いい加減")


def relationship_feedback_from_affect(
    label: str, cues: list[str] | tuple[str, ...] = (), signals: list[str] | tuple[str, ...] = ()
) -> str:
    """推定した様子を関係性スコア（REV-220）のフィードバックに変換する（REV-467）。

    - 喜び・お礼（REV-598）→ positive
    - 紫苑の返答への苛立ち（「違うって」「何度も」等）→ negative
    - それ以外（疲れ・不安・仕事への苛立ちなど）→ neutral（相手のつらさを紫苑への評価にしない）
    訂正（REV-598）は negative より弱い減点として、signals のまま関係性へ渡す。
    """
    if "shion_complaint" in signals or (label == "苛立ち" and any(c in _SHION_DIRECTED_COMPLAINTS for c in cues)):
        return "negative"
    if "correction" in signals:
        return "neutral"
    if label == "喜び" or "thanks" in signals:
        return "positive"
    return "neutral"


def record_relationship_from_affect(affect_payload: dict[str, Any], *, topic_depth: str = "normal") -> None:
    """関係性スコアへ対話1回分を記録する。失敗しても会話は止めない。"""
    try:
        from api.shion_relationship import record_interaction

        signals = list(affect_payload.get("signals") or [])
        feedback = relationship_feedback_from_affect(
            str(affect_payload.get("label") or NEUTRAL), list(affect_payload.get("cues") or []), signals
        )
        record_interaction(feedback_type=feedback, topic_depth=topic_depth, signals=signals)  # type: ignore[arg-type]
    except Exception as exc:
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.relationship_from_affect", "swallowed", exc)
