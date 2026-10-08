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
    items = capture.recall_taught_knowledge(vault, "希望ナンバーの車の見積書はどう見る？", candidates=[])
    assert items and items[0]["user_taught"] is True
    block = capture.build_recall_prompt_block(items)
    assert "ユーザーが教えた知識" in block
    assert capture.used_in_reply(items[0], "以前教わった通り、希望ナンバー登録をするユーザーは見積書が変わるので要注意です。")


def test_honesty_removes_unbacked_promises():
    reply = "なるほど、重要な視点ですね。これは判断資産として登録します。次も聞かせてください。"
    fixed = capture.enforce_save_honesty(reply, {"is_teaching": False, "saved": False})
    assert "登録します" not in fixed
    assert "まだ保存していません" in fixed


def test_saved_reply_shows_no_save_notice():
    """保存しても本文に処理の通知を出さない（2026-10-09 ユーザー方針）。保存先は teaching_save と指標ログで確かめる。"""
    saved = {"saved": True, "knowledge_path": "Lease Intelligence/Knowledge/x_2026-10-02.md", "candidate_id": "c"}
    assert capture.enforce_save_honesty("承知しました。", saved) == "承知しました。"
    incident = (
        "経営者の離婚は、経営権と連帯保証の変化として見る。\n\n"
        "Knowledgeノート `Projects/tune_lease_55/Lease Intelligence/Knowledge/経営者の離婚とリース審査_2026-10-09.md`"
        " に保存し、判断資産候補として反映した。\n\n"
        "保存先: `Lease Intelligence/Knowledge/x_2026-10-02.md`・判断資産候補（要確認）"
    )
    fixed = capture.enforce_save_honesty(incident, saved)
    assert fixed == "経営者の離婚は、経営権と連帯保証の変化として見る。"
    block = capture.build_save_result_prompt_block({"is_teaching": True, **saved})
    assert "回答に書かない" in block and "添えてよい" not in block


def test_knowledge_title_skips_chitchat_and_uses_title_maker():
    answer = "友人の話、それは胸が痛むね。急な環境の変化は大きな衝撃だ。リース審査では経営権と連帯保証の変化を見る。"
    assert capture.teaching_topic(answer).startswith("リース審査では")
    assert capture.teaching_topic(answer, lambda text: "**「経営者の離婚とリース審査」**\n") == "経営者の離婚とリース審査"
    # 題名づけが失敗・空なら文から作る
    assert capture.teaching_topic(answer, lambda text: 1 / 0).startswith("リース審査では")
    assert capture.teaching_topic(answer, lambda text: "").startswith("リース審査では")
    # 見出しのある回答は見出しを使い、題名づけを呼ばない
    assert capture.answer_topic("### 稟議の型\n本文", lambda text: pytest.fail("見出しがあれば呼ばない")) == "稟議の型"


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


def test_recall_matches_short_note_by_shared_domain_terms(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    capture.save_lease_teaching(
        "車の希望ナンバー登録をするユーザーは見積書が変わるので要注意",
        vault=vault,
        surface="t",
        candidate_saver=lambda claim: {"captured": True, "candidate": {"id": "c"}},
        date_str="2026-10-02",
    )
    items = capture.recall_taught_knowledge(
        vault, "中古車のリースで見積書をチェックするとき、気をつけることは？", candidates=[]
    )
    assert items and "希望ナンバー" in items[0]["snippet"]
    assert capture.recall_taught_knowledge(vault, "リース審査で大事なことは？", candidates=[]) == []


def test_recall_includes_chat_taught_candidates(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    items = capture.recall_taught_knowledge(
        vault,
        "中古トラックのリースは走行距離どこまで見ればいい？",
        candidates=[
            {
                "id": "c1",
                "research_date": "2026-08-04",
                "claim": "新車登録が5年以内で走行距離が200,000キロ位だったらばリースでもやっちゃう。なぜならトラックは1,000,000キロ位までは平気で走るからだ。",
            }
        ],
    )
    assert items and items[0]["path"] == "judgment_candidate:c1"
    assert "要確認" in items[0]["topic"]


def test_teaching_turn_saves_then_next_question_recalls_and_counts_use(tmp_path, monkeypatch):
    """通常チャットと対話室で共通の1ターン処理: 教える→保存→次の問いで想起→回答で使用。"""
    vault = _vault(tmp_path, monkeypatch)
    saver_calls: list[str] = []

    def saver(claim):
        saver_calls.append(claim)
        return {"captured": True, "candidate": {"id": "cand-x"}}

    taught = capture.prepare_teaching_turn(
        "車の希望ナンバー登録をするユーザーは見積書が変わるので要注意",
        vault=vault,
        surface="next_chat_rag",
        candidate_saver=saver,
        rag_search=lambda q: pytest.fail("教示の発言では想起しない"),
    )
    assert taught.save["saved"] is True and saver_calls
    assert "保存済み" in taught.save_context and taught.recall_context == ""
    assert taught.finalize("承知しました。") == "承知しました。"

    monkeypatch.setattr(capture, "_chat_taught_candidates", lambda: [])
    asked = capture.prepare_teaching_turn(
        "中古車のリースで見積書をチェックするとき、気をつけることは？",
        vault=vault,
        surface="next_chat_general",
        candidate_saver=lambda claim: pytest.fail("問いは保存しない"),
        rag_search=lambda q: [{"text": "見積書の確認手順", "source": "rag.md"}],
    )
    assert asked.recall_items and "希望ナンバー" in asked.recall_context
    assert "参照ナレッジ（rag.md）" in asked.recall_context
    reply = asked.finalize("以前教わった通り、希望ナンバー登録をするユーザーは見積書が変わるので要注意です。")
    assert "保存先" not in reply
    assert asked.response_extra()["pre_recall"][0]["user_taught"] is True

    summary = capture.funnel_summary(capture._today(), path=tmp_path / "funnel.jsonl")
    assert summary["day"] == {"taught": 1, "saved": 1, "recalled": 1, "used": 1}
    assert summary["day_by_surface"]["通常チャット"]["used"] == 1


def test_teaching_turn_survives_saver_and_search_failures(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)

    def broken(_):
        raise RuntimeError("down")

    turn = capture.prepare_teaching_turn(
        "リース審査で銀行借入の多い会社を見るときの注意点は？",
        vault=vault,
        surface="next_chat_general",
        candidate_saver=broken,
        rag_search=broken,
    )
    assert turn.rag_hits == []
    fixed = turn.finalize("覚えておきます。借入の返済原資を確認しましょう。")
    assert "覚えておきます" not in fixed and "まだ保存していません" in fixed


def test_funnel_summary_splits_dialogue_and_normal_chat(tmp_path):
    path = tmp_path / "f.jsonl"
    rows = [
        {"date": "2026-10-02", "event": "taught", "surface": "lease_intelligence_dialogue"},
        {"date": "2026-10-02", "event": "taught", "surface": "next_chat_rag"},
        {"date": "2026-10-02", "event": "saved", "surface": "next_chat_general"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    by = capture.funnel_summary("2026-10-02", path=path)["day_by_surface"]
    assert by["対話室"]["taught"] == 1
    assert by["通常チャット"] == {"taught": 1, "saved": 1, "recalled": 0, "used": 0}


# 2026-10-04 対話室: 保存していないのに「テンプレート集として永続化します」「いつでも呼び出せます」と言った回答。
INCIDENT_REPLY = """承知いたしました。

### 記録内容：異業種多角化案件の審査コメントテンプレート
- **内容**:
    1. 資産転用の妥当性（本業とのシナジー vs 専用設備リスク）
    2. 収益構造の独立性（景気サイクルの分散、販売ルートの確実性）
    3. リスク管理体制（採算管理の独立性、撤退基準の有無）
    4. 総合判断（本業の強みの延長 vs 新規投資リスク）

上記を、リース審査システム内の「稟議コメント・テンプレート集」として永続化します。

---
これで、いつでもこの判断軸を呼び出せます。

次に、このテンプレートを実際に使ってみる場面を想定して、何か準備しておきますか？"""

TEMPLATE_ANSWER = """承知いたしました。

### 稟議コメントテンプレート：異業種多角化案件

**【異業種多角化案件の審査意見】**
1. **資産転用の妥当性:** 本業の既存設備・人員の活用状況について、[本業とのシナジー/専用設備によるリスク]を確認。
2. **収益構造の独立性:** 本業と異なる景気サイクルへの対応、および販売ルートの確実性について、[確保済み/未定]と判断。
3. **リスク管理体制:** 採算管理の独立性および撤退基準の有無について、[明示あり/なし]を確認。
4. **総合判断:** 上記より、本件は[本業の強みの延長/新規投資によるリスク大]と判断し、[承認/条件付き承認/否決]とする。

今回の保存で、多角化案件に関する判断資産の構築はひとまず完了です。他に、稟議フローで気になる点はありますか？

（この発言はまだ保存していません。残したい場合は「判断資産に入れて」と送ってください。）"""


def test_honesty_removes_incident_persist_and_recall_claims():
    fixed = capture.enforce_save_honesty(INCIDENT_REPLY, {"is_teaching": False, "saved": False})
    for claim in ("永続化します", "テンプレート集", "いつでもこの判断軸を呼び出せます"):
        assert claim not in fixed
    assert "資産転用の妥当性" in fixed and "まだ保存していません" in fixed


@pytest.mark.parametrize(
    "sentence",
    [
        "判断資産に入れておきます。", "Knowledgeに登録しておきました。", "記録に残しておきます。", "テンプレート集に追加します。",
        "保存は完了しました。", "今回の保存で、判断資産の構築はひとまず完了です。", "ナレッジへ入れておきました。",
        "いつでも参照できます。", "永続化しました。",
    ],
)
def test_save_claims_are_detected(sentence):
    assert capture.SAVE_CLAIM_RE.search(sentence)
    assert "まだ保存していません" in capture.enforce_save_honesty(sentence, {"saved": False})


@pytest.mark.parametrize(
    "sentence",
    [
        "このテンプレートは保存が必要です。", "審査記録として残すべきです。", "稟議書に保存しておくと良いでしょう。",
        "保存しますか？", "保存しておきますか。", "保存する必要があります。", "残価を保証する契約です。",
        "条件付き承認では保証人を追加します。", "記録を残すことをお勧めします。", "保存してください。",
    ],
)
def test_suggestions_and_general_talk_are_kept(sentence):
    assert capture.enforce_save_honesty(sentence, {"saved": False}) == sentence


def test_save_request_saves_previous_shion_answer(tmp_path, monkeypatch):
    vault = _vault(tmp_path, monkeypatch)
    calls: list[tuple[str, dict]] = []

    def saver(claim, **kw):
        calls.append((claim, kw))
        return {"captured": True, "candidate": {"id": "cand-tpl"}}

    turn = capture.prepare_teaching_turn(
        "作ったテンプレートを保存する必要があるな",
        vault=vault,
        surface="lease_intelligence_dialogue",
        candidate_saver=saver,
        previous_user_message="稟議コメントのテンプレートとして保存　そこまで頻度はないだろう",
        previous_assistant_message=TEMPLATE_ANSWER,
    )
    save = turn.save
    assert save["saved"] and save["source"] == "shion_answer"
    assert save["topic"] == "稟議コメントテンプレート：異業種多角化案件"
    claim, kw = calls[0]
    assert kw == {"user_requested": True}
    assert "資産転用の妥当性" in claim and "まだ保存していません" not in claim
    assert "今回の保存" not in claim and "気になる点はありますか" not in claim
    note = (vault / "Lease Intelligence" / "Knowledge").glob("*.md")
    assert any("撤退基準" in p.read_text(encoding="utf-8") for p in note)
    assert "直前の紫苑の回答「稟議コメントテンプレート：異業種多角化案件」" in turn.save_context

    fixed = turn.finalize(INCIDENT_REPLY)
    assert "テンプレート集" not in fixed and "永続化します" not in fixed  # 保存しても無い保存先は言わせない
    assert "保存したもの" not in fixed and "判断資産候補" not in fixed and "保存先" not in fixed

    # 次に多角化案件を聞いたら、保存したテンプレートが想起される
    monkeypatch.setattr(capture, "_chat_taught_candidates", lambda: [])
    asked = capture.prepare_teaching_turn(
        "異業種への多角化案件の稟議コメントはどう書けばいい？",
        vault=vault,
        surface="next_chat_general",
        candidate_saver=lambda claim, **kw: pytest.fail("問いは保存しない"),
    )
    assert asked.recall_items and "多角化" in asked.recall_context


def test_bare_save_instruction_still_saves_users_own_teaching():
    """「判断資産にしておいて」だけなら、直前のユーザーの教示を保存する（回答は保存しない）。"""
    assert capture.save_request_target(
        "そうだね　判断資産にしておいて", "銀行と取引のない企業とは付き合わない", TEMPLATE_ANSWER
    ) == ""
    assert capture.save_request_target("作ったテンプレートを保存して", "", TEMPLATE_ANSWER) == TEMPLATE_ANSWER
    assert capture.save_request_target("保存が必要か考えてる", "", "元気です") == ""


@pytest.mark.parametrize(
    "query",
    ["本業と異なる新規事業に参入する案件の稟議はどう書く？", "多角化案件の審査コメントのテンプレートある？"],
)
def test_diversification_template_is_recalled_by_paraphrase(query):
    # 実際に保存した形（見出し＋本文）
    claim = f"異業種多角化案件の審査コメントテンプレート\n\n{capture._answer_body(TEMPLATE_ANSWER)}"
    row = {"id": "tpl", "research_date": "2026-10-04", "claim": claim}
    items = capture.recall_taught_knowledge(None, query, candidates=[row])
    assert items and items[0]["path"] == "judgment_candidate:tpl"
    assert capture.recall_taught_knowledge(None, "中古トラックの走行距離はどこまで見る？", candidates=[row]) == []
