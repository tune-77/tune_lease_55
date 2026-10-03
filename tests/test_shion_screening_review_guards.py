"""紫苑レビュー（審査分析画面）の遅延・漏れ・方針の扱いの検査（2026-10-03）.

- 依頼文（約3,000字）で RAG の再ランクが候補文書ごとに分かち書きをやり直し、1件2〜18分かかっていた
- 依頼文（企業名・営業メモ入り）が伏字なしで蒸留ノートに残り、Vault → Vertex 同期に載っていた
- 社内方針が案件と語の一致で拾われず、判断資産として「変形して応用」されていた
"""
from __future__ import annotations

import json

from api.knowledge import vector_store
from api.routers import feedback_loop
from api.vertex_query_mask import mask_for_vertex

REVIEW_PROMPT_HEAD = (
    "【審査分析画面からの紫苑レビュー依頼】\n前提:\n・企業名: サンプル精機\n・業種: 金属製品製造業\n"
    "・営業メモ: 山田部長と昔から付き合いがあり、多少高くても受注\n・取得価額: 5百万円"
)


def test_query_terms_are_split_once_per_query():
    vector_store._cached_query_terms.cache_clear()
    query = REVIEW_PROMPT_HEAD * 20
    first = vector_store.KnowledgeVectorStore._query_terms(query)
    for _ in range(50):
        assert vector_store.KnowledgeVectorStore._query_terms(query) == first
    info = vector_store._cached_query_terms.cache_info()
    assert info.misses == 1 and info.hits == 50


def test_mask_hides_review_prompt_company_and_memo_lines():
    masked = mask_for_vertex(REVIEW_PROMPT_HEAD)
    assert "サンプル精機" not in masked and "山田" not in masked and "多少高くても" not in masked
    assert "企業名:〈伏字〉" in masked and "営業メモ:〈伏字〉" in masked
    assert "金属製品製造業" in masked  # 審査概念は残す


def test_distilled_note_is_masked_before_it_reaches_the_vault(tmp_path):
    from api.vertex_distillation import capture_vertex_distillation

    vault = tmp_path / "vault"
    vault.mkdir()
    result = capture_vertex_distillation(
        REVIEW_PROMPT_HEAD,
        {"used": True, "status": "ok", "answer_text": "サンプル精機株式会社の山田部長への確認が要る", "refs": []},
        vault_path=vault,
        state_path=tmp_path / "state.json",
    )
    note = (vault / result["note_path"]).read_text(encoding="utf-8")
    assert "サンプル精機" not in note and "山田" not in note and "多少高くても" not in note


def test_active_policy_block_lists_every_active_policy(tmp_path):
    from api.judgment_policy import POLICY_HEADER
    from api.shion_memory_recall import active_policy_block

    index = tmp_path / "index.json"
    index.write_text(json.dumps({"records": [
        {"source": "canonical_judgment_rules", "status": "active", "knowledge_kind": "policy", "judgment_asset_id": "93d1b22c",
         "created_at": "2026-10-02", "content": "銀行と取引のない企業とは付き合わない"},
        {"source": "canonical_judgment_rules", "status": "active", "knowledge_kind": "policy", "judgment_asset_id": "8932df45",
         "created_at": "2026-10-02", "content": "メイン銀行の借り入れよりもリースの借り入れが増えないようにする"},
        {"source": "canonical_judgment_rules", "status": "merged", "knowledge_kind": "policy", "judgment_asset_id": "old", "content": "統合済みの古い方針"},
        {"source": "canonical_judgment_rules", "status": "active", "knowledge_kind": "insight", "judgment_asset_id": "x", "content": "資金繰り表を確認する"},
    ]}, ensure_ascii=False))
    block = active_policy_block(index_path=index)
    assert block.startswith(POLICY_HEADER)
    assert "判断資産 93d1b22c" in block and "判断資産 8932df45" in block
    assert "統合済み" not in block and "資金繰り表" not in block


def _asset(asset_id: str, claim: str, kind: str) -> dict:
    return {
        "id": asset_id, "source": "canonical_judgment_rules", "candidate_type": "application_rule",
        "research_topic": "chat_judgment_teaching", "claim": claim, "effective_claim": claim,
        "promotion_status": "active", "verified_status": "canonical", "knowledge_kind": kind,
        "use_count": 0, "useful_count": 0, "edit_count": 0, "rejected_count": 0,
    }


def test_policies_come_first_and_do_not_use_up_the_insight_limit(monkeypatch):
    policy = _asset("cr-93d1b22ceac4e274", "銀行と取引のない企業とは付き合わない", "policy")
    insights = [
        {**_asset(f"cr-{i}0000000000000000", f"油圧ショベルの更新では稼働時間と保守記録を確認する{i}", "insight"), "candidate_type": kind}
        for i, kind in enumerate(("application_rule", "condition_signal", "confirmation_question", "application_rule"))
    ]
    monkeypatch.setattr(feedback_loop, "_load_canonical_judgment_asset_candidates", lambda *a, **k: [*insights, policy])
    monkeypatch.setattr(feedback_loop, "_load_autoresearch_judgment_asset_candidates", lambda *a, **k: [])
    monkeypatch.setattr(feedback_loop, "_load_news_judgment_signals", lambda *a, **k: [])

    selected = feedback_loop._select_screening_judgment_asset_candidates(
        industry_major="建設業", industry_sub="土木", asset_name="油圧ショベル", asset_purpose="更新", hantei="承認", limit=3,
    )
    assert selected[0]["id"] == policy["id"] and selected[0]["knowledge_kind"] == "policy"
    assert [item["knowledge_kind"] for item in selected[1:]] == ["insight"] * 3
