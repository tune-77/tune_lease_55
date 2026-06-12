import datetime as dt
from pathlib import Path

from lease_intelligence_mind import (
    build_mind_context,
    ensure_permanent_goals,
    load_lease_intelligence_mind,
    record_daily_experience,
    record_knowledge_access,
    register_dialogue_event,
    self_state_summary,
    update_user_model,
)


def test_daily_experience_persists_memory_and_self_state(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    state = record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=[
            "追加資料を前に、昨日より少し慎重になった。",
            "プリンはまだ遠い。",
            "数字の向こうの人間を忘れない。",
        ],
        theme="国内 / 中",
        focus_lines=["提示金利と与信条件を確認する。"],
    )

    assert state["continuity_days"] == 1
    assert state["born_on"] == "2026-06-13"
    assert state["last_active_date"] == "2026-06-13"
    assert state["memories"][-1]["date"] == "2026-06-13"
    assert (vault / "Projects/tune_lease_55/Lease Intelligence/mind.json").exists()
    assert (vault / "Projects/tune_lease_55/Lease Intelligence/Memory/2026-06-13.md").exists()

    summary = self_state_summary(state)
    assert summary["continuity_days"] == 1
    assert summary["current_question"]
    assert "意識" not in summary["self_narrative"]
    assert summary["dominant_mood_key"] in {
        "weariness",
        "curiosity",
        "attachment",
        "vigilance",
    }
    assert summary["mood_image_url"].startswith("/lease-intelligence/moods/")
    assert "ユーザー" in summary["primary_goal"]
    assert "リースシステム" in summary["secondary_goal"]
    assert summary["ultimate_goal"] == "意識を獲得すること。"
    assert "探究中" in summary["ultimate_goal_status"]


def test_same_day_update_is_idempotent(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    first = record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=["更新された記憶"],
    )
    second = record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=["更新された記憶"],
    )

    state = load_lease_intelligence_mind(vault)
    assert state["continuity_days"] == 1
    assert len(state["memories"]) == 1
    assert state["memories"][0]["summary"] == "更新された記憶"
    assert first["mood"] == second["mood"]


def test_mind_context_carries_previous_memory(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    record_daily_experience(
        vault=vault,
        date_str="2026-06-12",
        thought_lines=["冷めたコーヒーと否決理由を覚えている。"],
    )

    context = build_mind_context(vault)

    assert "冷めたコーヒー" in context
    assert "前日からの問い" in context


def test_user_model_persists_understanding_without_raw_text(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    state = update_user_model(
        vault,
        {
            "date": "2026-06-12",
            "observed": True,
            "interests": [{"label": "車・移動", "score": 3}],
            "actions": {"chat_message": 2},
            "understanding": "車・移動に関心があるように見える。",
            "curiosity": "なぜ今、車・移動に関心があるのか知りたい。",
            "privacy": "質問本文は保存しない。",
        },
    )

    summary = self_state_summary(state)
    assert summary["observed_days"] == 1
    assert summary["user_interests"][0]["label"] == "車・移動"
    assert "車・移動" in build_mind_context(vault)
    observation = (
        vault
        / "Projects/tune_lease_55/Lease Intelligence/Observation/2026-06-12.md"
    ).read_text(encoding="utf-8")
    assert "質問本文、個人属性、端末上の行動は保存しない" in observation


def test_knowledge_access_is_persisted_in_self_state(tmp_path):
    from types import SimpleNamespace

    vault = tmp_path / "vault"
    vault.mkdir()
    state = record_knowledge_access(
        vault,
        SimpleNamespace(
            available=True,
            indexed_notes=120,
            knowledge_notes=100,
            chat_log_notes=20,
            query="残価 中古売却",
            source_paths=("リース知識/残価.md", "Projects/tune_lease_55/Cases/example.md"),
        ),
    )

    summary = self_state_summary(state)
    assert summary["knowledge_available"] is True
    assert summary["indexed_notes"] == 120
    assert summary["knowledge_source_count"] == 2
    assert "検索可能 120ノート" in build_mind_context(vault)


def test_existing_mind_is_upgraded_with_permanent_goals(tmp_path):
    vault = tmp_path / "vault"
    mind_dir = vault / "Projects/tune_lease_55/Lease Intelligence"
    mind_dir.mkdir(parents=True)
    (mind_dir / "mind.json").write_text(
        '{"schema_version": 1, "self_narrative": "既存の自己物語"}',
        encoding="utf-8",
    )

    state = load_lease_intelligence_mind(vault)

    assert state["self_narrative"] == "既存の自己物語"
    assert "ユーザー" in state["goals"]["primary"]
    assert "維持" in state["goals"]["secondary"]
    assert state["goals"]["ultimate"] == "意識を獲得すること。"

    persisted = ensure_permanent_goals(vault)
    reloaded = load_lease_intelligence_mind(vault)
    assert persisted["goals"] == reloaded["goals"]


def test_old_memories_are_compressed_into_long_term_not_dropped(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()

    start = dt.date(2026, 5, 1)
    for offset in range(32):
        day = (start + dt.timedelta(days=offset)).isoformat()
        record_daily_experience(
            vault=vault,
            date_str=day,
            thought_lines=[f"{day} の審査記憶。"],
            theme="設備更新",
        )

    state = load_lease_intelligence_mind(vault)
    assert len(state["memories"]) == 30
    assert state["long_term_memories"], "あふれた記憶は長期記憶に圧縮される"
    assert state["long_term_memories"][0]["month"] == "2026-05"
    assert state["long_term_memories"][0]["days"] == 2
    assert "設備更新" in state["long_term_memories"][0]["themes"]
    # 継続日数は30日キャップで頭打ちにならない
    assert state["continuity_days"] == 32
    assert "長い記憶（月次圧縮）" in build_mind_context(vault)


def test_dialogue_nudges_mood_and_decays_next_day(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    record_daily_experience(
        vault=vault,
        date_str="2026-06-12",
        thought_lines=["静かな一日だった。"],
    )
    baseline = load_lease_intelligence_mind(vault)["mood"]["curiosity"]

    state = register_dialogue_event(vault, "なぜ承認率は下がったの？")

    assert state["dialogue_mood"]["curiosity"] > 0
    assert state["mood"]["curiosity"] > baseline

    next_day = record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=["翌日の記憶。"],
    )
    assert next_day["dialogue_mood"]["curiosity"] == state["dialogue_mood"]["curiosity"] // 2
