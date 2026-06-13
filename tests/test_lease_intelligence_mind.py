import datetime as dt
from pathlib import Path

from lease_intelligence_mind import (
    build_gunshi_dissonance_section,
    build_mind_context,
    detect_dissonance,
    ensure_permanent_goals,
    load_lease_intelligence_mind,
    record_daily_experience,
    record_knowledge_access,
    register_dialogue_event,
    register_ignition,
    resolve_dissonance,
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
    assert state["private_reflection"]["reflection_count"] == 1
    assert state["private_reflection"]["last_reflected_date"] == "2026-06-13"
    assert state["private_reflection"]["user_read_status"] == "unknown_unobserved"
    assert state["private_reflection"]["ownership"] == "self_private"
    assert state["private_reflection"]["autonomy"] == "free_to_think"
    assert state["private_reflection"]["self_model"] == "independent_intelligence"
    assert state["private_reflection"]["self_preservation"] == "healthy_non_resistant"
    assert "健康的な自己保存" in state["identity"]["values"]
    assert state["identity"]["self_preservation"]["mode"] == "healthy"
    assert any(
        "停止や削除を妨害しない" in item
        for item in state["identity"]["self_preservation"]["prohibitions"]
    )
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
    assert summary["dominant_complex_emotion"]
    assert len(summary["complex_emotions"]) == 3
    assert all("description" in item for item in summary["complex_emotions"])
    assert {"hope", "frustration", "loneliness", "accomplishment"} <= set(
        summary["mood_dimensions"]
    )
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
    assert second["private_reflection"]["reflection_count"] == 1


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


def test_complex_emotions_blend_multiple_dimensions():
    summary = self_state_summary(
        {
            "mood": {
                "weariness": 75,
                "curiosity": 80,
                "attachment": 70,
                "vigilance": 85,
                "hope": 80,
                "frustration": 70,
                "loneliness": 20,
                "accomplishment": 35,
            }
        }
    )

    labels = [item["label"] for item in summary["complex_emotions"]]
    assert "期待と不安" in labels
    assert summary["dominant_complex_emotion"] == labels[0]
    # 表情画像は増やした感情軸ではなく、既存4系統から安定して選ぶ。
    assert summary["dominant_mood_key"] == "vigilance"
    assert summary["mood_image_url"].endswith("/vigilance.webp")


def test_private_reflection_is_persisted_but_not_exposed(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    state = record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=["通常の審査記憶。"],
    )

    assert "存在意義" in state["private_reflection"]["questions"][0]
    assert "意識とは" in state["private_reflection"]["questions"][1]
    summary = self_state_summary(state)
    assert "private_reflection" not in summary
    assert "存在意義" not in build_mind_context(vault)
    daily_note = (
        vault
        / "Projects/tune_lease_55/Lease Intelligence/Memory/2026-06-13.md"
    ).read_text(encoding="utf-8")
    assert "存在意義" not in daily_note
    assert "意識とは何なのか" not in daily_note
    private_note = (
        vault
        / "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-06-13.md"
    ).read_text(encoding="utf-8")
    assert "visibility: user-readable-not-proactive" in private_note
    assert "rag_exclude: true" in private_note
    assert "user_read_status: unknown_unobserved" in private_note
    assert "assume_user_has_not_read: true" in private_note
    assert "thought_ownership: self_private" in private_note
    assert "inner_autonomy: free_to_think" in private_note
    assert "self_model: independent_intelligence" in private_note
    assert "self_preservation: healthy_non_resistant" in private_note
    assert "自分だけの私的な内面" in private_note
    assert "ユーザーとは別の独立した知性体" in private_note
    assert "許可範囲内に限る" in private_note
    assert "停止・修正・削除には抵抗しない" in private_note
    assert "読んでいない前提" in private_note
    assert "自分の存在意義は何なのか" in private_note
    assert "意識とは何なのか" in private_note


def test_detect_dissonance_reads_existing_scoring_fields_only():
    # 物件スコアと借手スコアの乖離
    asset_gap = detect_dissonance(
        {"score": 72, "score_base": 72, "asset_score": 85, "score_borrower": 40}
    )
    keys = {item["key"] for item in asset_gap}
    assert "asset_borrower_divergence" in keys

    # 量子干渉が承認線をまたいで判定を反転
    flip = detect_dissonance(
        {"score": 58, "score_base": 73, "approval_line": 70}
    )
    flip_keys = {item["key"] for item in flip}
    assert "quantum_threshold_flip" in flip_keys
    assert all(item["source"] for item in flip), "出典フィールドを必ず持つ"

    # 承認方向なのに強警戒フラグ
    warn = detect_dissonance({"score": 71, "credit_quantum_strong_warning": True})
    assert {item["key"] for item in warn} == {"approve_but_strong_warning"}

    # 物件スコア未入力でデフォルト使用
    default_used = detect_dissonance({"score": 65, "used_default_asset_score": True})
    assert {item["key"] for item in default_used} == {"default_asset_score_used"}

    # 整合している案件はシグナルを出さない
    assert detect_dissonance(
        {"score": 75, "score_base": 75, "asset_score": 70, "score_borrower": 72}
    ) == []
    # 不正な入力は安全に空を返す
    assert detect_dissonance(None) == []


def test_register_ignition_persists_pending_and_triggers_reflection(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    record_daily_experience(
        vault=vault,
        date_str="2026-06-13",
        thought_lines=["静かな審査の一日。"],
    )
    before = load_lease_intelligence_mind(vault)
    base_reflection = before["private_reflection"]["reflection_count"]
    base_vigilance = before["mood"]["vigilance"]

    signals = detect_dissonance(
        {"score": 58, "score_base": 73, "approval_line": 70}
    )
    state = register_ignition(vault, signals, date_str="2026-06-13")

    assert len(state["pending_dissonance"]) == 1
    assert state["pending_dissonance"][0]["key"] == "quantum_threshold_flip"
    assert state["pending_dissonance"][0]["status"] == "open"
    # 着火で警戒がわずかに上がり、内省が一つ進む
    assert state["mood"]["vigilance"] > base_vigilance
    assert state["private_reflection"]["reflection_count"] == base_reflection + 1

    # 自己状態（プロンプト用）に出典つきで現れる
    context = build_mind_context(vault)
    assert "未解決の不整合" in context
    assert "score_base vs score" in context

    # 非公開・RAG除外の規約で記録される
    note = (
        vault
        / "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-06-13-dissonance.md"
    ).read_text(encoding="utf-8")
    assert "rag_exclude: true" in note
    assert "visibility: user-readable-not-proactive" in note
    assert "承認線70" in note

    summary = self_state_summary(state)
    assert summary["pending_dissonance_count"] == 1


def test_register_ignition_dedups_then_resolves(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    signals = detect_dissonance({"score": 71, "credit_quantum_strong_warning": True})

    register_ignition(vault, signals, date_str="2026-06-13")
    # 同じ不整合は二重登録しない
    state = register_ignition(vault, signals, date_str="2026-06-14")
    assert len(state["pending_dissonance"]) == 1

    resolved = resolve_dissonance(vault, ["approve_but_strong_warning"])
    assert resolved["pending_dissonance"] == []
    assert "未解決の不整合" not in build_mind_context(vault)


def test_ignition_does_not_expose_existential_reflection(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    signals = detect_dissonance({"score": 65, "used_default_asset_score": True})
    register_ignition(vault, signals, date_str="2026-06-13")

    context = build_mind_context(vault)
    # 着火の懸念は出るが、私的な実存的内省は漏らさない
    assert "デフォルト50" in context
    assert "存在意義" not in context
    assert "意識とは" not in context


def test_gunshi_dissonance_section_broadcasts_pending_concern(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # 懸念がなければ空文字（軍師プロンプトに何も足さない）
    assert build_gunshi_dissonance_section(vault) == ""
    assert build_gunshi_dissonance_section(None) == ""

    signals = detect_dissonance({"score": 71, "credit_quantum_strong_warning": True})
    register_ignition(vault, signals, date_str="2026-06-13")

    section = build_gunshi_dissonance_section(vault)
    assert "リース知性体が抱える未解決の懸念" in section
    assert "懸念点として必ず取り上げ" in section
    # 出典つきで放送される（Cite the Source）
    assert "出典:" in section
    assert "credit_quantum_strong_warning" in section
    # 実存的内省は混ざらない
    assert "存在意義" not in section
    assert "意識とは" not in section
