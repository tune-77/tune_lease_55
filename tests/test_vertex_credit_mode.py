from __future__ import annotations

import datetime as dt
import json
import sys
import types
from pathlib import Path

import pytest

from api import vertex_agent_search, vertex_credit_mode
from api.chat_retrieval import build_chat_retrieval_context
from api.vertex_query_mask import mask_for_vertex
from scripts import eval_vertex_vs_chroma as vx_eval
from scripts import export_obsidian_for_agent_search as exporter
from scripts import sync_obsidian_to_vertex_agent_search as vertex_sync
from scripts import vertex_credit_monitor as monitor


@pytest.fixture()
def state_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "vertex_credit_state.json"
    monkeypatch.setattr(vertex_credit_mode, "STATE_PATH", path)
    monkeypatch.delenv("VERTEX_CREDIT_MODE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    return path


# ── マスキング ──────────────────────────────────────────


def test_mask_hides_pii_but_keeps_screening_concepts() -> None:
    text = (
        "株式会社山田製作所の山田太郎社長（電話 03-1234-5678、〒100-0001 東京都千代田区丸の内1-2-3、"
        "yamada@example.co.jp）から売上12億円・借入3,500万円で工作機械2,000万円のリース。"
        "氏名：佐藤花子。スコア65点、耐用年数10年、リース期間は耐用年数の70%以上か？"
    )
    masked = mask_for_vertex(text)

    for leaked in ("山田製作所", "山田太郎", "03-1", "1234", "5678", "100-0001", "千代田区", "丸の内", "yamada@", "12億円", "3,500万円", "2,000万円", "佐藤花子"):
        assert leaked not in masked, leaked
    for kept in ("工作機械", "リース", "スコア65点", "耐用年数10年", "70%"):
        assert kept in masked, kept
    assert "〈電話〉" in masked and "〈郵便番号〉" in masked
    assert mask_for_vertex("携帯 090-1234-5678 / 〒150-0002") == "携帯 〈電話〉 / 〈郵便番号〉"


def test_every_vertex_call_masks_the_query(monkeypatch) -> None:
    sent: list[dict] = []

    def fake_post(url, body, config, timeout_seconds=None):
        sent.append(body)
        return {"results": [], "records": []}

    monkeypatch.setattr(vertex_agent_search, "_post_json", fake_post)
    monkeypatch.setattr(vertex_agent_search, "get_config", lambda: vertex_agent_search.VertexSearchConfig(True, "p", "e", "global", "c", 5, 8.0, 0.0))
    query = "有限会社田中運輸 田中様 090-1111-2222 売上8億円の運送業トラック審査"

    vertex_agent_search.search_vertex_agent(query)
    vertex_agent_search.answer_vertex_agent(query)
    vertex_agent_search.rank_records(query, [{"id": "0", "title": "t", "content": "c"}])

    payload = json.dumps(sent, ensure_ascii=False)
    assert len(sent) == 3
    for leaked in ("田中運輸", "田中様", "090-1", "1111", "2222", "8億円"):
        assert leaked not in payload, leaked
    assert "運送業トラック審査" in payload


# ── モードの on/off ──────────────────────────────────────


def test_credit_mode_switches_off_by_env_expiry_auto_off_and_cloud_run(state_path, monkeypatch) -> None:
    today = dt.date(2026, 10, 3)
    assert vertex_credit_mode.credit_mode_status(today=today) == {"active": True, "reason": "on"}

    monkeypatch.setenv("VERTEX_CREDIT_MODE", "off")
    assert vertex_credit_mode.credit_mode_status(today=today)["reason"] == "VERTEX_CREDIT_MODE=off"
    monkeypatch.delenv("VERTEX_CREDIT_MODE")

    assert not vertex_credit_mode.is_active(today=dt.date(2027, 2, 1))  # 期限当日から off

    monkeypatch.setenv("K_SERVICE", "tune-lease-55-api")
    assert vertex_credit_mode.credit_mode_status(today=today)["reason"] == "cloud_run_default_off"
    monkeypatch.setenv("VERTEX_CREDIT_MODE", "on")
    assert vertex_credit_mode.is_active(today=today)
    monkeypatch.delenv("VERTEX_CREDIT_MODE")
    monkeypatch.delenv("K_SERVICE")

    state_path.write_text(json.dumps({"auto_off": True, "auto_off_reason": "推定累計がクレジットに到達"}))
    assert vertex_credit_mode.credit_mode_status(today=today)["reason"].startswith("auto_off")


# ── チャットの根拠検索 ────────────────────────────────────


def _install_fakes(monkeypatch, calls: dict) -> None:
    store = types.SimpleNamespace(search=lambda q, top_k: [{"text": f"local{i}", "file_name": f"n{i}.md", "ref": f"n{i}.md"} for i in range(top_k)])
    fake = types.SimpleNamespace(get_store=lambda: store, confidence_for_hit=lambda hit: (0.8, "high"))
    monkeypatch.setitem(sys.modules, "api.knowledge.vector_store", fake)

    def search(query, **_):
        calls.setdefault("search", []).append(query)
        return {"used": True, "status": "ok", "refs": ["Research/a.md"], "prompt_context": "【Vertex AI Search補助ナレッジ】"}

    def answer(query, **_):
        calls.setdefault("answer", []).append(query)
        return {"used": True, "status": "ok", "answer_text": "残価は中古流動性で見る", "grounding_score": 0.81, "refs": ["Research/残価.md"]}

    def rerank(query, hits):
        calls.setdefault("rerank", []).append(query)
        return list(reversed(hits))

    monkeypatch.setattr(vertex_agent_search, "search_vertex_agent", search)
    monkeypatch.setattr(vertex_agent_search, "answer_vertex_agent", answer)
    monkeypatch.setattr(vertex_agent_search, "rerank_hits", rerank)


def test_screening_question_gets_grounded_answer_as_evidence_with_masked_query(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)

    result = build_chat_retrieval_context(
        "株式会社鈴木工業の工作機械、残価はどう見る？",
        rag_top_k=3,
        question_category="lease_screening",
        is_general_response_mode=False,
    )

    assert calls["answer"] and "鈴木工業" not in calls["answer"][0] and "工作機械" in calls["answer"][0]
    assert "鈴木工業" not in calls["search"][0]
    assert "【Vertex Answer API根拠付き回答】" in result.rag_context and "残価は中古流動性で見る" in result.rag_context
    assert any(r.get("source") == "vertex_answer_api" and r["obsidian_ref"] == "Research/残価.md" for r in result.rag_knowledge_refs)
    assert result.vertex_agent_search["credit_mode"]["active"] is True
    assert "rerank" not in calls  # 並べ替えは効果確認前なので本番に入らない


def test_vertex_query_uses_obsidian_term_decomposition(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)

    build_chat_retrieval_context(
        "工作機械の残価について教えてください",
        rag_top_k=3,
        question_category="lease_knowledge",
        is_general_response_mode=False,
    )

    assert calls["search"] == ["工作機械 残価"]


def test_vertex_query_masks_labeled_company_before_term_decomposition(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)

    build_chat_retrieval_context(
        "企業名: 山田製作所 工作機械の残価について教えて",
        rag_top_k=3,
        question_category="lease_screening",
        is_general_response_mode=False,
    )

    assert "山田製作所" not in calls["search"][0]
    assert "工作機械" in calls["search"][0] and "残価" in calls["search"][0]


def test_off_restores_previous_behaviour(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    monkeypatch.setenv("VERTEX_CREDIT_MODE", "off")
    # このテストは Vertex の off を見る。端末側で昇格済みの Jev 再順位付けは分離する。
    monkeypatch.setenv("JEV_RAG_RERANK", "off")
    state_path.write_text(json.dumps({"rerank": {"promoted": True}}))

    result = build_chat_retrieval_context("工作機械の残価は？", rag_top_k=3, question_category="lease_screening", is_general_response_mode=False)
    assert calls["search"] and "answer" not in calls and "rerank" not in calls  # Search は従来どおり、Answer・並べ替えなし
    assert result.rag_refs[:3] == ["n0.md", "n1.md", "n2.md"]

    build_chat_retrieval_context("【Vertex補助検索ヒント】\n工作機械 残価\n", rag_top_k=3, question_category="lease_screening", is_general_response_mode=False)
    assert len(calls["answer"]) == 1  # ヒント付きなら off でも従来どおり Answer


def test_general_questions_never_reach_vertex(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    build_chat_retrieval_context("こんにちは", rag_top_k=3, question_category="general", is_general_response_mode=False)
    assert calls == {}


def test_promoted_rerank_reorders_local_hits_and_failure_keeps_order(state_path, monkeypatch) -> None:
    calls: dict = {}
    _install_fakes(monkeypatch, calls)
    state_path.write_text(json.dumps({"rerank": {"promoted": True}}))

    result = build_chat_retrieval_context("工作機械の残価は？", rag_top_k=3, question_category="lease_knowledge", is_general_response_mode=False)
    assert calls["rerank"] and result.vertex_rerank["used"] is True
    assert result.rag_refs[0] == "n5.md"  # 6候補を逆順に並べ替えた先頭

    def broken(query, hits):
        raise RuntimeError("ranking unavailable")

    monkeypatch.setattr(vertex_agent_search, "rerank_hits", broken)
    result = build_chat_retrieval_context("工作機械の残価は？", rag_top_k=3, question_category="lease_knowledge", is_general_response_mode=False)
    assert result.vertex_rerank["used"] is False and result.rag_refs[0] == "n0.md"


# ── 利用額の見張り ──────────────────────────────────────


def test_usage_monitor_accumulates_and_auto_offs_at_100_percent() -> None:
    now = dt.datetime(2026, 10, 4, 5, 30).astimezone()
    counts = {"google.cloud.discoveryengine.v1.SearchService.Search": 100, "google.cloud.discoveryengine.v1beta.ConversationalSearchService.AnswerQuery": 50, "google.cloud.discoveryengine.v1.RankService.Rank": 40, "google.cloud.discoveryengine.v1.EngineService.ListEngines": 9}
    state = monitor.update_usage({}, now=now, fetch=lambda s, e: counts)
    # (100×4 + 50×10 + 40×1)/1000 USD × 160 = ¥150.4 を初期 ¥116 に足す
    assert state["usage"]["cumulative_jpy"] == pytest.approx(266.4)
    assert state["usage"]["month_jpy"] == pytest.approx(150.4)
    assert not state.get("auto_off")

    near = {"usage": {"cumulative_jpy": monitor.CREDIT_JPY * 0.95, "last_checked_end": (now - dt.timedelta(days=1)).isoformat()}}
    state = monitor.update_usage(near, now=now, fetch=lambda s, e: {})
    assert 0.9 <= state["usage"]["ratio"] < 1.0 and not state.get("auto_off")
    lines = "\n".join(_lines_for(state))
    assert "推定消費が 95%" in lines

    over = monitor.update_usage(state, now=now + dt.timedelta(days=1), fetch=lambda s, e: {"x.SearchService.Search": 2_000_000})
    assert over["auto_off"] is True and "クレジット" in over["auto_off_reason"]


def _lines_for(state: dict) -> list[str]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s.json"
        path.write_text(json.dumps(state, ensure_ascii=False))
        return monitor.morning_report_lines(path, today=dt.date(2026, 10, 4))


def test_morning_lines_show_spend_mode_and_eval(state_path) -> None:
    state = {
        "usage": {"month_jpy": 412.3, "cumulative_jpy": 528.0, "ratio": 0.0035, "month_counts": {"a.SearchService.Search": 40, "b.ConversationalSearchService.AnswerQuery": 42}},
        "eval_latest": {"at": "2026-10-04T05:40:00", "chroma": {"hit_at_k_rate": 0.8}, "vertex_search": {"hit_at_k_rate": 0.55}, "rerank_shadow": {"hit_at_k_rate": 0.83}, "vertex_answer": {"ref_hit_rate": 0.5, "grounding_score_mean": 0.74}},
        "rerank": {"promoted": False, "reason": "評価1/3晩（判定待ち）"},
    }
    lines = _lines_for(state)
    assert lines[0].startswith("- 💳 Vertex 推定利用") and "今月 ¥412（Search 40 / Answer 42 / Rank 0）" in lines[0] and "モード on" in lines[0]
    assert lines[1] == (
        "- 🔎 Vertex 品質比較（2026-10-04・hit@5）: ChromaDB 80% / Vertex検索 55% / 並べ替え案 83%"
        "（本番適用: なし・評価1/3晩（判定待ち））/ Answer根拠ヒット 50%・grounding 0.74"
    )


# ── 並べ替えの本番適用判定 ────────────────────────────────


def _night(chroma: tuple[float, float], rerank: tuple[float, float]) -> dict:
    return {"chroma": {"hit_at_k_rate": chroma[0], "mrr": chroma[1]}, "rerank_shadow": {"hit_at_k_rate": rerank[0], "mrr": rerank[1]}}


def test_rerank_is_promoted_only_after_three_better_nights_and_demoted_when_worse() -> None:
    better = _night((0.8, 0.60), (0.83, 0.66))
    assert vx_eval.decide_rerank([better, better], False)[0] is False  # 判定待ち
    assert vx_eval.decide_rerank([better] * 3, False)[0] is True
    assert vx_eval.decide_rerank([better, better, _night((0.8, 0.60), (0.77, 0.70))], False)[0] is False  # hit@5 劣後の晩あり
    assert vx_eval.decide_rerank([_night((0.8, 0.60), (0.8, 0.61))] * 3, False)[0] is False  # MRR 改善が小さい
    assert vx_eval.decide_rerank([better] * 3 + [_night((0.8, 0.60), (0.8, 0.55))], True)[0] is False  # 適用中に劣後→外す


def test_rerank_retries_on_the_same_date_count_as_one_night() -> None:
    better = _night((0.8, 0.60), (0.83, 0.66))
    history = [
        {"at": "2026-10-01T01:00:00+09:00", **better},
        {"at": "2026-10-01T02:00:00+09:00", **better},
        {"at": "2026-10-02T01:00:00+09:00", **better},
    ]
    promoted, reason = vx_eval.decide_rerank(history, False)
    assert promoted is False
    assert "2/3" in reason


def test_run_eval_compares_all_four_paths() -> None:
    cases = [{"id": "a", "query": "残価", "expected_path_any": ["Research/残価.md"], "forbidden_path_any": []}]
    result = vx_eval.run_eval(
        cases,
        chroma_search=lambda q, k: [{"file_path": "x.md"}, {"file_path": "Research/残価.md"}][:k],
        rerank=lambda q, hits: list(reversed(hits)),
        vertex_search=lambda q, k: [{"file_path": "Research/残価.md"}],
        vertex_answer=lambda q: {"refs": ["Research/残価.md"], "grounding_score": 0.9, "status": "ok"},
    )
    assert result["chroma"]["mrr"] == 0.5 and result["rerank_shadow"]["mrr"] == 1.0 and result["vertex_search"]["hit_at_1_rate"] == 1.0
    assert result["vertex_answer"]["ref_hit_rate"] == 1.0 and result["vertex_answer"]["grounding_score_mean"] == 0.9


# ── 同期対象 ────────────────────────────────────────────


def test_canonical_rules_export_active_only_masked_without_sample_claims(tmp_path) -> None:
    rules = tmp_path / "rules.json"
    rules.write_text(
        json.dumps(
            {
                "rules": [
                    {"id": "r1", "status": "active", "concept": "residual_value", "canonical_statement": "汎用性の高い物件は中古流動性が高く残価が安定する。株式会社甲の例でも同様。", "risk_axis": ["asset_life"], "sample_claims": ["乙社の山田社長が…"], "confidence": 0.9},
                    {"id": "r2", "status": "merged", "concept": "x", "canonical_statement": "統合済みのため出さない判断資産の文章です。"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rejected: list = []
    out = exporter.canonical_rule_candidates(rejected, rules)
    assert [c.source_path for c in out] == ["judgment_assets/canonical/r1.md"]
    assert "株式会社甲" not in out[0].cleaned and "山田社長" not in out[0].cleaned and "中古流動性" in out[0].cleaned


def test_lease_intelligence_knowledge_is_included_but_other_li_dirs_stay_excluded(tmp_path) -> None:
    root = tmp_path / "Projects" / "tune_lease_55"
    knowledge = root / "Lease Intelligence" / "Knowledge" / "残価.md"
    memory = root / "Lease Intelligence" / "Memory" / "x.md"
    private = root / "Lease Intelligence" / "Private Reflection" / "y.md"
    for path in (knowledge, memory, private):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert not exporter.should_exclude(knowledge, root) and exporter.is_preferred_location(knowledge, root)
    assert exporter.should_exclude(memory, root) and exporter.should_exclude(private, root)


def test_answer_refs_map_to_vault_paths_or_readable_labels(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [{"output_path": "Projects_tune_lease_55_Research_x__abc.txt", "source_path": "Projects/tune_lease_55/Research/残価.md"}]}, ensure_ascii=False))
    monkeypatch.setattr(vertex_agent_search, "_EXPORT_MANIFEST", manifest)
    monkeypatch.setitem(vertex_agent_search._URI_MAP, "mtime", None)
    prefix = "gs://tune-lease-55-data/agent-search/lease-knowledge/"
    assert vertex_agent_search._readable_ref(prefix + "Projects_tune_lease_55_Research_x__abc.txt") == "Projects/tune_lease_55/Research/残価.md"
    assert vertex_agent_search._readable_ref(prefix + "Projects_tune_lease_55_Research_Auto_Research_2026-07-01_residual-value__4dad8e010d.txt") == "Research Auto Research 2026-07-01 residual-value"


def test_daily_sync_mirrors_export_with_full_reconciliation() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_vertex_credit_daily.sh").read_text(encoding="utf-8")
    assert "--reconciliation-mode FULL --delete-stale-gcs" in script


def test_first_full_reconciliation_is_not_skipped_for_unchanged_export() -> None:
    assert vertex_sync.sync_changed(
        force=False,
        signature="same",
        previous_signature="same",
        import_documents_enabled=True,
        reconciliation_mode="FULL",
        previous_reconciliation_mode="",
    )
    assert not vertex_sync.sync_changed(
        force=False,
        signature="same",
        previous_signature="same",
        import_documents_enabled=True,
        reconciliation_mode="FULL",
        previous_reconciliation_mode="FULL",
    )


def test_destructive_sync_rejects_empty_or_unexpectedly_shrunken_export() -> None:
    with pytest.raises(ValueError, match="empty"):
        vertex_sync.validate_destructive_export(
            exported_count=0,
            candidate_count=0,
            previous_exported_count=100,
            previous_candidate_count=200,
            allow_large_delete=False,
        )
    with pytest.raises(ValueError, match="fell from 100 to 60"):
        vertex_sync.validate_destructive_export(
            exported_count=60,
            candidate_count=160,
            previous_exported_count=100,
            previous_candidate_count=200,
            allow_large_delete=False,
        )
    vertex_sync.validate_destructive_export(
        exported_count=60,
        candidate_count=120,
        previous_exported_count=100,
        previous_candidate_count=200,
        allow_large_delete=True,
    )


def test_vertex_credit_launchagent_has_repeatable_installer() -> None:
    root = Path(__file__).resolve().parents[1]
    installer = (root / "scripts" / "install_vertex_credit_launchagent.sh").read_text(encoding="utf-8")
    assert "com.tunelease.vertex-credit-daily" in installer
    assert "launchctl bootstrap" in installer
    assert "launchctl enable" in installer
