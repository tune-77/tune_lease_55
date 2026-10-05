from datetime import datetime, timedelta

from api.chat_prompt_budget import MEMORY, SPECS
from api.user_affect_memory import (
    MAX_OBSERVATIONS,
    build_user_affect_memory_block,
    recall_user_affect,
    record_user_affect,
    remember_and_build_block,
)

T0 = datetime(2026, 10, 5, 22, 0)


def test_records_per_user_and_recalls_on_next_session(tmp_path):
    path = tmp_path / "affect.json"
    record_user_affect("alice", "疲れ", 0.8, now=T0, path=path)
    record_user_affect("bob", "喜び", 0.8, now=T0, path=path)

    recall = recall_user_affect("alice", now=T0 + timedelta(hours=10), path=path)
    assert recall.label == "疲れ"
    block = build_user_affect_memory_block(recall)
    assert "疲れ" in block and "気遣って" in block
    assert recall_user_affect("bob", now=T0 + timedelta(hours=10), path=path).label == "喜び"
    assert recall_user_affect("carol", path=path).label == "通常"


def test_old_state_decays_away(tmp_path):
    path = tmp_path / "affect.json"
    record_user_affect("alice", "不安", 0.6, now=T0, path=path)
    # 半減期24h: 0.6 -> 3日後 0.075 < 0.2
    recall = recall_user_affect("alice", now=T0 + timedelta(days=3), path=path)
    assert recall.label == "通常"
    assert build_user_affect_memory_block(recall) == ""


def test_calmed_down_after_neutral_turns(tmp_path):
    path = tmp_path / "affect.json"
    record_user_affect("alice", "落ち込み", 0.9, now=T0, path=path)
    record_user_affect("alice", "通常", 0.0, now=T0 + timedelta(minutes=5), path=path)
    record_user_affect("alice", "通常", 0.0, now=T0 + timedelta(minutes=10), path=path)
    recall = recall_user_affect("alice", now=T0 + timedelta(hours=8), path=path)
    assert recall.calmed_down
    assert "蒸し返さず" in build_user_affect_memory_block(recall)


def test_recurrent_pattern_is_noted(tmp_path):
    path = tmp_path / "affect.json"
    for d in range(3):
        record_user_affect("alice", "焦り", 0.7, now=T0 + timedelta(days=d), path=path)
    recall = recall_user_affect("alice", now=T0 + timedelta(days=4), path=path)
    assert recall.recurrent == "焦り"
    assert "繰り返し" in build_user_affect_memory_block(recall)


def test_remember_and_build_does_not_count_current_turn(tmp_path):
    path = tmp_path / "affect.json"
    block, payload = remember_and_build_block("alice", "疲れ", 0.9, surface="t", now=T0, path=path)
    assert block == "" and payload["label"] == "通常"
    block2, payload2 = remember_and_build_block(
        "alice", "通常", 0.0, surface="t", now=T0 + timedelta(hours=12), path=path
    )
    assert payload2["label"] == "疲れ"
    assert "疲れ" in block2


def test_store_has_no_message_text_and_is_bounded(tmp_path):
    path = tmp_path / "affect.json"
    for i in range(MAX_OBSERVATIONS + 10):
        record_user_affect("alice", "通常", 0.0, now=T0 + timedelta(minutes=i), path=path)
    record_user_affect("alice", "疲れ", 0.5, now=T0 - timedelta(days=30), path=path)  # 古すぎる
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    obs = data["users"]["alice"]["observations"]
    assert len(obs) <= MAX_OBSERVATIONS
    assert set(obs[-1]) == {"label", "intensity", "at", "surface"}


def test_budget_spec():
    assert SPECS["user_affect_memory_context"].tier == MEMORY
