"""参照ナレッジのプロンプト文字列に信頼度マーカーが載ることの単体テスト（REV-008）。

従来 confidence_for_hit() の結果はフロントのバッジ用 payload にしか流れず、
LLM に渡る【参照ナレッジ】ブロックは低信頼ヒットと高信頼ヒットが同一形式だった。
低・中のヒットにだけ印を付け、モデルが断定を避けられるようにする。
"""
import time

from api.chat_retrieval import RAG_CONFIDENCE_MARKERS, _append_rag_hits
from api.knowledge.vector_store import confidence_for_hit


def _days_ago(days: float) -> float:
    return time.time() - days * 86_400


# confidence = 0.7 * rank + 0.3 * recency（high >= 0.8 / medium >= 0.5 / low < 0.5）
_HIGH_HIT = {"rank_score": 0.95, "metadata": {"mtime": _days_ago(1)}}
_MEDIUM_HIT = {"rank_score": 0.6, "metadata": {"mtime": _days_ago(120)}}
_LOW_HIT = {"rank_score": 0.2, "metadata": {"mtime": _days_ago(700)}}


def _render(hit: dict) -> str:
    """1件のヒットをプロンプト用文字列に変換する。"""
    return _append_rag_hits(
        [dict(hit)],
        rag_refs=[],
        rag_knowledge_refs=[],
    )


def test_fixture_hits_produce_the_intended_levels():
    """マーカー検証の前提となるレベル分けが崩れていないこと。"""
    assert confidence_for_hit(_HIGH_HIT)[1] == "high"
    assert confidence_for_hit(_MEDIUM_HIT)[1] == "medium"
    assert confidence_for_hit(_LOW_HIT)[1] == "low"


def test_low_confidence_hit_is_marked_in_prompt():
    context = _render({**_LOW_HIT, "ref": "古いノート", "text": "旧耐用年数は8年"})
    assert "[信頼度:低]" in context
    assert "[信頼度:低] 古いノート: 旧耐用年数は8年" in context


def test_medium_confidence_hit_is_marked_in_prompt():
    context = _render({**_MEDIUM_HIT, "ref": "中程度ノート", "text": "審査所見"})
    assert "[信頼度:中] 中程度ノート: 審査所見" in context


def test_high_confidence_hit_is_not_marked():
    context = _render({**_HIGH_HIT, "ref": "新しいノート", "text": "最新の承認ライン"})
    assert "信頼度:" not in context
    assert "新しいノート: 最新の承認ライン" in context


def test_marker_does_not_consume_the_600_char_text_budget():
    """マーカーは本文予算の外側に付く（低信頼ヒットの本文が削られない）。"""
    body = "あ" * 1000
    context = _render({**_LOW_HIT, "ref": "", "text": body})
    rendered = context.split("【参照ナレッジ】\n", 1)[1]
    marker = RAG_CONFIDENCE_MARKERS["low"]
    assert rendered.startswith(marker)
    assert rendered[len(marker):] == "あ" * 600


def test_knowledge_refs_payload_still_carries_confidence():
    """フロント向けバッジ payload の既存挙動を壊していないこと。"""
    refs: list[dict] = []
    _append_rag_hits(
        [{**_LOW_HIT, "ref": "古いノート", "text": "本文", "doc_id": "d1"}],
        rag_refs=[],
        rag_knowledge_refs=refs,
    )
    assert len(refs) == 1
    assert refs[0]["doc_id"] == "d1"
    assert refs[0]["confidence_level"] == "low"
    assert 0.0 <= refs[0]["confidence"] < 0.5


def test_marker_map_covers_every_confidence_level():
    """confidence_for_hit が返しうるレベルがすべてマーカー表にあること。"""
    levels = {
        confidence_for_hit(hit)[1] for hit in (_HIGH_HIT, _MEDIUM_HIT, _LOW_HIT)
    }
    assert levels == set(RAG_CONFIDENCE_MARKERS)


def test_empty_hits_still_return_empty_context():
    assert _append_rag_hits([], rag_refs=[], rag_knowledge_refs=[]) == ""
