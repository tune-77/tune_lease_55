import pytest

from api.chat_prompt_budget import SPECS, SHAPE
from api.user_affect import NEUTRAL, build_user_affect_prompt_block, estimate_user_affect


@pytest.mark.parametrize(
    ("message", "label"),
    [
        ("今日はもう疲れた…", "疲れ"),
        ("残業続きでしんどい", "疲れ"),
        ("至急！今日中に稟議を出さないと間に合わない", "焦り"),
        ("やった！承認された！", "喜び"),
        ("この案件、通るか不安で…大丈夫かな", "不安"),
        ("失注しました。落ち込んでます", "落ち込み"),
        ("何度も言ってるけど違うって。いい加減にして", "苛立ち"),
    ],
)
def test_estimate_user_affect_labels(message, label):
    affect = estimate_user_affect(message)
    assert affect.label == label
    assert 0.0 < affect.intensity <= 1.0
    assert affect.cues


@pytest.mark.parametrize(
    "message",
    [
        "",
        "ファイナンスリースとオペレーティングリースの違いは？",
        "ありがとう",
        "全然疲れてないよ",
        "急ぎではないので、時間があるときに",
        "特に不安はないです",
    ],
)
def test_estimate_user_affect_neutral(message):
    assert estimate_user_affect(message).label == NEUTRAL


def test_negation_exception_for_shika_nai():
    assert estimate_user_affect("もう不安しかない").label == "不安"


def test_prompt_block_empty_for_neutral():
    assert build_user_affect_prompt_block(estimate_user_affect("償却期間を教えて")) == ""


def test_prompt_block_changes_length_and_keeps_facts():
    block = build_user_affect_prompt_block(estimate_user_affect("疲れた…手短に教えて"))
    assert "疲れ" in block
    assert "短く" in block
    assert "審査判断" in block  # 事実・判断は変えないガード
    assert "断定しない" in block


def test_long_pasted_text_only_checks_edges():
    body = "決算書の注記: 不安定な受注と心配な資金繰りの記載がある。" * 10
    message = "要約して\n" + ("あ" * 400) + body[:20] + ("い" * 400)
    assert estimate_user_affect(message).label == NEUTRAL


def test_budget_spec_registered_as_shape():
    assert SPECS["user_affect_context"].tier == SHAPE


def test_payload_shape():
    payload = estimate_user_affect("至急お願いします").to_payload()
    assert payload["label"] == "焦り"
    assert set(payload) == {"label", "intensity", "cues"}
