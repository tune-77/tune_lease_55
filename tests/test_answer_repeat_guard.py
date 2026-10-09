"""REV-544: 同じ質問への丸写しを防ぐ（聞き直しの指示・RAG の自己回答の印・開発用ノートの除外）。"""
from __future__ import annotations

from api.answer_repeat_guard import (
    REPEAT_QUESTION_BLOCK,
    SELF_ANSWER_END,
    SELF_ANSWER_OMITTED,
    SELF_ANSWER_START,
    apply_repeat_guard,
    drop_self_answers_in_history,
    find_repeated_question,
    is_dev_note_path,
    is_self_answer_path,
    mark_self_answer_text,
)

Q = "運送業や建設業の倒産が増えてるけど、リース審査で気をつけることは？"
ANSWER = (
    "運送業や建設業の倒産増加は、単なる「販売不振」ではなく、**「収益構造の限界」**を示唆している。"
    "特に燃料や人件費の高騰を価格転嫁できない企業は、どれほど稼働していてもキャッシュフローが枯渇しやすい。"
)


def _history(*pairs: tuple[str, str]) -> list[dict]:
    out: list[dict] = []
    for user, assistant in pairs:
        out += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return out


def test_find_repeated_question_matches_same_and_near_same():
    history = _history(("こんにちは", "こんにちは。"), (Q, ANSWER))
    assert find_repeated_question(Q, history) == Q
    assert find_repeated_question("運送業や建設業の倒産が増えているけど、リース審査で気をつけることは？", history) == Q
    assert find_repeated_question("フォークリフトの耐用年数は？", history) == ""


def test_find_repeated_question_ignores_assistant_turns_and_short_messages():
    assert find_repeated_question(Q, [{"role": "assistant", "content": Q}]) == ""
    assert find_repeated_question("はい", _history(("はい", "了解"))) == ""


def test_small_talk_repeat_is_detected():
    assert find_repeated_question("今日ちょっと疲れた", _history(("今日ちょっと疲れた", "お疲れさま。"))) == "今日ちょっと疲れた"


def test_apply_repeat_guard_adds_block_only_for_repeats():
    prompt = "システム"
    assert apply_repeat_guard(prompt, _history((Q, ANSWER)), Q).endswith(REPEAT_QUESTION_BLOCK)
    assert apply_repeat_guard(prompt, _history(("別の話", "はい")), Q) == prompt
    assert apply_repeat_guard(prompt, [], Q) == prompt


def test_path_markers():
    assert is_self_answer_path("/x/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-10-09.md")
    assert not is_self_answer_path("Projects/tune_lease_55/Lease Intelligence/Knowledge/a.md")
    assert is_dev_note_path("Projects/tune_lease_55/紫苑の仕組み_2026-10/11_ニュース.md")
    assert not is_dev_note_path("Projects/tune_lease_55/Research/a.md")


def test_mark_self_answer_text_is_idempotent_and_bounded():
    marked = mark_self_answer_text("あ" * 2000)
    assert marked.startswith(SELF_ANSWER_START) and marked.endswith(SELF_ANSWER_END)
    assert len(marked) < 600
    assert mark_self_answer_text(marked) == marked


def test_drop_self_answers_in_history_omits_only_duplicates():
    chunk = mark_self_answer_text(f"**ユーザー**\n\n{Q}\n\n**リース知性体**\n\n{ANSWER}")
    other = mark_self_answer_text("**ユーザー**\n\n別の質問\n\n**リース知性体**\n\n" + "まったく別の内容の答えで、履歴には出てこない文章が続く。" * 3)
    prompt = f"【参照ナレッジ】\n[[2026-10-09#07:22:10]]: {chunk}\n---\n{other}"
    dropped_prompt, dropped = drop_self_answers_in_history(prompt, _history((Q, ANSWER)))
    assert dropped == 1
    assert SELF_ANSWER_OMITTED in dropped_prompt
    assert "収益構造の限界" not in dropped_prompt
    assert "まったく別の内容" in dropped_prompt  # 履歴に無い過去回答は印付きで残す
    unchanged, count = drop_self_answers_in_history(prompt, [])
    assert count == 0 and unchanged == prompt


def test_append_rag_hits_labels_self_answers_and_skips_dev_notes():
    from api.chat_retrieval import _append_rag_hits

    hits = [
        {"text": "開発メモ", "ref": "11_ニュース", "file_path": "/v/Obsidian Vault/Projects/tune_lease_55/紫苑の仕組み_2026-10/11.md"},
        {"text": f"**リース知性体** {ANSWER}", "ref": "2026-10-09#07:22:10",
         "file_path": "/v/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-10-09.md"},
    ]
    refs: list[str] = []
    context = _append_rag_hits(hits, rag_refs=refs, rag_knowledge_refs=[])
    assert "開発メモ" not in context
    assert SELF_ANSWER_START in context and SELF_ANSWER_END in context


def test_vector_store_rerank_marks_self_answers_and_drops_dev_notes():
    from api.knowledge.vector_store import KnowledgeVectorStore

    store = KnowledgeVectorStore(chroma_dir="/tmp/unused-rev544", ranking_config={})
    base = {"wikilinks": "", "metadata": {}, "score": 10, "source": "keyword", "section": "s"}
    hits = [
        {**base, "doc_id": "a", "text": f"倒産 審査 {ANSWER}", "ref": "d", "file_name": "2026-10-09.md",
         "file_path": "/v/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-10-09.md"},
        {**base, "doc_id": "b", "text": "倒産 審査 開発の記録", "ref": "e", "file_name": "11.md",
         "file_path": "/v/Obsidian Vault/Projects/tune_lease_55/紫苑の仕組み_2026-10/11.md"},
    ]
    ranked = store._rerank_hits("倒産 審査", hits, top_k=5)
    assert [h["doc_id"] for h in ranked] == ["a"]
    assert ranked[0]["self_answer"] is True and ranked[0]["text"].startswith(SELF_ANSWER_START)


def test_teaching_recall_block_does_not_present_self_answer_as_taught():
    from api.chat_teaching_capture import build_recall_prompt_block

    hit = {"text": mark_self_answer_text(ANSWER), "self_answer": True, "source": "vector"}
    block = build_recall_prompt_block([], [hit])
    assert "紫苑自身の過去の回答（教わった知識ではない。写さない）" in block
    assert block.count(SELF_ANSWER_START) == 1 and block.count(SELF_ANSWER_END) == 1
    deduped, dropped = drop_self_answers_in_history(block, _history((Q, ANSWER)))
    assert dropped == 1 and "収益構造の限界" not in deduped


def test_call_gemini_chat_sends_guarded_prompt(monkeypatch):
    import api.chat_memory as cm

    sent: dict = {}

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "回答"}]}, "finishReason": "STOP"}]}

    monkeypatch.setattr(cm, "_get_gemini_api_key", lambda: "k")
    monkeypatch.setattr(cm, "tracked_ai_http_call", lambda fn, **_kw: fn())
    monkeypatch.setattr(cm.requests, "post", lambda url, json, headers, timeout: sent.update(json) or _Resp())
    assert cm.call_gemini_chat("システム", _history((Q, ANSWER)), Q) == "回答"
    assert REPEAT_QUESTION_BLOCK in sent["system_instruction"]["parts"][0]["text"]
