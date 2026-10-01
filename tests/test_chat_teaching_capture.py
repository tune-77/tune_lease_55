"""チャットで教わったノウハウの判定・保存・想起・正直さ・指標の回帰テスト。"""

from __future__ import annotations

import json

import pytest

import api.chat_teaching_capture as capture
from memory_promotion_policy import classify_lease_teaching, classify_memory_destination


@pytest.mark.parametrize(
    "text",
    [
        "新車登録が5年以内で走行距離が200,000キロ位だったらばリースでもやっちゃう。なぜならトラックは1,000,000キロ位までは平気で走るからだ。",
        "中古自動車の見積書は要注意。金額の入れ方が特殊なところが多い。また業者によってもいろいろ違うので、間違いやすい判断資産に入れて",
        "車の希望ナンバー登録をするユーザーは見積書が変わるので要注意",
        "銀行と取引のない企業とは付き合わない　資料がすぐ出てこない企業とも付き合わない　延滞がちな企業にもリース増額しない",
        "省力化に対する機械リースも増えてきたから一緒に検討しよう",
        # 旧判定は「とは」「何」の部分一致で落としていた
        "債務超過ということは、銀行との協調リースでないと取り扱わない",
        # 旧判定は「説明」「提案」を改善要望に誤分類していた
        "資金使途の説明が付かない設備リースは、販売店に直接確認してから進める",
        "金利が上がる局面では固定金利のリースを提案すると通りやすい",
    ],
)
def test_user_teachings_are_captured(text):
    assert classify_lease_teaching(text)[0] is True
    assert classify_memory_destination(text) == "judgment_asset_candidate"


@pytest.mark.parametrize(
    "text",
    [
        "所有権移転リースとはなんだ",
        "車の再リースはどうやるの",
        "医療機器のリースをするときの注意点を教えてくれ",
        "判断資産グラフの表示を改善してほしい",
        "判断資産は記憶システムと関係あるのか？",
        "元気？今日は暑いね",
    ],
)
def test_questions_requests_and_system_talk_are_not_teachings(text):
    assert classify_lease_teaching(text)[0] is False


def test_anaphoric_instruction_uses_previous_user_message():
    claim = capture.resolve_teaching_claim("そうだね　判断資産にしておいて", "銀行の今後の支援が明確ならば取り扱う")
    assert claim.startswith("銀行の今後の支援が明確ならば取り扱う")


def _vault(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    monkeypatch.setattr("lease_intelligence_mind.mind_directory", lambda v: vault / "Lease Intelligence")
    monkeypatch.setattr(capture, "_funnel_path", lambda: tmp_path / "funnel.jsonl")
    return vault


def test_save_writes_knowledge_and_candidate_and_dedupes(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    saved_claims: list[str] = []

    def saver(claim):
        saved_claims.append(claim)
        return {"captured": True, "duplicate": len(saved_claims) > 1, "candidate": {"id": "cand-1"}}

    text = "車の希望ナンバー登録をするユーザーは見積書が変わるので要注意"
    first = capture.save_lease_teaching(text, vault=vault, surface="t", candidate_saver=saver, date_str="2026-10-02")
    second = capture.save_lease_teaching(text, vault=vault, surface="t", candidate_saver=saver, date_str="2026-10-02")

    assert first["saved"] is True
    assert first["knowledge_path"].startswith("Lease Intelligence/Knowledge/")
    assert first["candidate_id"] == "cand-1"
    notes = list((vault / "Lease Intelligence" / "Knowledge").glob("*.md"))
    assert len(notes) == 1
    assert "source_type: \"chat_teaching\"" in notes[0].read_text(encoding="utf-8")
    assert second["knowledge_duplicate"] is True

    summary = capture.funnel_summary("2026-10-02", path=tmp_path / "funnel.jsonl")
    assert summary["total"]["taught"] == 2
    assert summary["total"]["saved"] == 2


def test_non_teaching_is_not_saved(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    result = capture.save_lease_teaching(
        "元気？", vault=vault, surface="t", candidate_saver=lambda claim: pytest.fail("must not save")
    )
    assert result == {"is_teaching": False, "saved": False, "reason": "length"}


def test_recall_finds_taught_knowledge(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    capture.save_lease_teaching(
        "車の希望ナンバー登録をするユーザーは見積書が変わるので要注意",
        vault=vault,
        surface="t",
        candidate_saver=lambda claim: {"captured": True, "candidate": {"id": "c"}},
        date_str="2026-10-02",
    )
    items = capture.recall_taught_knowledge(vault, "希望ナンバーの車の見積書はどう見る？")
    assert items and items[0]["user_taught"] is True
    block = capture.build_recall_prompt_block(items)
    assert "ユーザーが教えた知識" in block
    assert capture.used_in_reply(items[0], "以前教わった通り、希望ナンバー登録をするユーザーは見積書が変わるので要注意です。")


def test_honesty_removes_unbacked_promises():
    reply = "なるほど、重要な視点ですね。これは判断資産として登録します。次も聞かせてください。"
    fixed = capture.enforce_save_honesty(reply, {"is_teaching": False, "saved": False})
    assert "登録します" not in fixed
    assert "まだ保存していません" in fixed


def test_honesty_appends_save_location_when_saved():
    fixed = capture.enforce_save_honesty(
        "承知しました。",
        {"saved": True, "knowledge_path": "Lease Intelligence/Knowledge/x_2026-10-02.md", "candidate_id": "c"},
    )
    assert "保存先: `Lease Intelligence/Knowledge/x_2026-10-02.md`・判断資産候補（要確認）" in fixed


def test_chat_candidates_go_to_review(tmp_path):
    from types import SimpleNamespace

    from api.chat_judgment_asset_capture import create_manual_judgment_asset_candidate

    row = create_manual_judgment_asset_candidate(
        SimpleNamespace(
            claim="車の希望ナンバー登録をするユーザーは見積書が変わるので要注意",
            candidate_type="caution",
            research_topic="chat_judgment_teaching",
            case_id="chat:u",
            review_id=None,
            research_date="2026-08-24",
        ),
        candidates_jsonl=tmp_path / "c.jsonl",
        candidate_state_json=tmp_path / "s.json",
    )
    assert row["promotion_status"] == "needs_review_quality"
    assert row["research_date"] == "2026-08-24"
    stored = json.loads((tmp_path / "c.jsonl").read_text(encoding="utf-8"))
    assert stored["promotion_status"] == "needs_review_quality"


def test_reflection_boilerplate_does_not_crowd_out_user_keypoints():
    from lease_intelligence_mind import (
        SYSTEM_KEYPOINT_LIMIT,
        _dedupe_conversation_keypoints,
        _limit_conversation_keypoints,
    )

    items = []
    for day in range(1, 31):
        for text in ("Private Reflectionからの学び: A", "Private Reflectionからの学び: B"):
            items.append({"date": f"2026-09-{day:02d}", "session_id": "private_reflection_feedback_loop", "content": text})
    for n in range(10):
        items.append({"date": "2026-10-01", "session_id": "lease-intelligence-dialogue", "content": f"ユーザー発{n}"})

    deduped = _dedupe_conversation_keypoints(items)
    system = [i for i in deduped if i["session_id"] == "private_reflection_feedback_loop"]
    assert len(system) == 2
    assert {i["date"] for i in system} == {"2026-09-30"}

    many_system = [
        {"date": "2026-09-01", "session_id": "private_reflection_feedback_loop", "content": f"定型{n}"}
        for n in range(50)
    ]
    users = [{"date": "2026-10-01", "session_id": "u", "content": f"ユーザー{n}"} for n in range(15)]
    limited = _limit_conversation_keypoints(users[:5] + many_system + users[5:], 30)
    assert sum(1 for i in limited if i["session_id"] == "u") == 10
    assert sum(1 for i in limited if i["session_id"] != "u") == SYSTEM_KEYPOINT_LIMIT


def test_memory_index_reads_vault_mind_and_knowledge(tmp_path, monkeypatch):
    from scripts import build_shion_memory_index as idx

    li = tmp_path / "vault" / "Lease Intelligence"
    (li / "Knowledge").mkdir(parents=True)
    (li / "mind.json").write_text(
        json.dumps({"conversation_keypoints": [{"content": "銀行と取引のない企業とは付き合わない", "session_id": "u"}]}),
        encoding="utf-8",
    )
    (li / "Knowledge" / "希望ナンバー_2026-08-24.md").write_text(
        "---\nsource_type: \"chat_teaching\"\n---\n# 希望ナンバー\n\n- 車の希望ナンバー登録をするユーザーは見積書が変わるので要注意\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SHION_MEMORY_INDEX_VAULT", "on")
    monkeypatch.setattr(idx, "REPO_ROOT", tmp_path / "repo")
    monkeypatch.setattr(idx, "_vault_root", lambda: tmp_path / "vault")
    monkeypatch.setattr(idx, "_vault_lease_intelligence_dir", lambda: li)

    records = idx.build_index(tmp_path / "missing.json")["records"]
    contents = {r["content"]: r for r in records}
    assert "銀行と取引のない企業とは付き合わない" in contents
    knowledge = [r for r in records if r["source"] == "lease_intelligence.knowledge"]
    assert knowledge and knowledge[0]["user_taught"] is True
    assert knowledge[0]["source_path"].startswith("vault:")

    demo = idx.build_index(tmp_path / "missing.json", demo_safe=True)["records"]
    assert not [r for r in demo if str(r.get("source_path", "")).startswith("vault:")]
