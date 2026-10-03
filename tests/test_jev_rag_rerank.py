from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from api import jev_rag_rerank as jr
from api.chat_retrieval import build_chat_retrieval_context
from scripts import eval_vertex_vs_chroma as vx_eval


def _answers(probs: list[float]) -> dict:
    return {"answers": {f"p{i}_useful": {"type": "noul", "noul": p} for i, p in enumerate(probs)}, "usage": {"input_tokens": 1200}, "model": "jev-latest"}


def test_request_is_masked_and_hits_are_sorted_by_probability(monkeypatch) -> None:
    logged: list = []
    monkeypatch.setattr(jr, "_log_judgments", lambda payload, probs, model: logged.append((payload, probs)))
    sent: list = []
    hits = [
        {"text": "株式会社青木運輸（03-1111-2222）の売上3億円", "file_name": "a.md"},
        {"text": "工作機械の残価と中古流動性", "file_name": "b.md"},
        {"text": "補助金の公募要領", "file_name": "c.md"},
    ]

    def fake(payload):
        sent.append(payload)
        return _answers([0.2, 0.9, 0.2])

    ordered, meta = jr.jev_rerank("有限会社山本製作所の山本社長、残価は？", hits, request_fn=fake)

    body = json.dumps(sent, ensure_ascii=False)
    for leaked in ("青木運輸", "03-1111-2222", "3億円", "山本製作所", "山本社長"):
        assert leaked not in body, leaked
    assert "a.md" not in body or sent[0]["state"]["passages"][0]["title"] == "a.md"  # 題名はファイル名のみ・パスは送らない
    assert [h["file_name"] for h in ordered] == ["b.md", "a.md", "c.md"]  # 同点は元の順位
    assert meta["input_tokens"] == 1200 and meta["calls"] == 1 and logged


def test_invalid_answer_raises_so_callers_keep_chroma_order() -> None:
    with pytest.raises(ValueError):
        jr.jev_rerank("q", [{"text": "x"}], request_fn=lambda p: {"answers": {}}, log=False)


def test_judgments_go_to_rev424_log_as_shadow(monkeypatch, tmp_path) -> None:
    import jev_judgment_log

    path = tmp_path / "log.jsonl"
    monkeypatch.setattr(jev_judgment_log, "log_path", lambda environ=None: path)
    jr.jev_rerank("残価は？", [{"text": "残価"}, {"text": "補助金"}], request_fn=lambda p: _answers([0.8, 0.1]))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["guard"] for r in rows} == {"rag_jev_rerank"} and {r["mode"] for r in rows} == {"shadow"}
    assert [r["probability"] for r in rows] == [0.8, 0.1] and all("残価" not in json.dumps(r, ensure_ascii=False) for r in rows)


def test_production_switch_is_independent_of_vertex_credit_mode(monkeypatch, tmp_path) -> None:
    state = tmp_path / "jev.json"
    monkeypatch.delenv("JEV_RAG_RERANK", raising=False)
    assert jr.production_enabled(state) is False
    state.write_text(json.dumps({"promoted": True}))
    monkeypatch.setenv("VERTEX_CREDIT_MODE", "off")
    assert jr.production_enabled(state) is True  # Vertex を止めても Jev は別スイッチ
    monkeypatch.setenv("JEV_RAG_RERANK", "off")
    assert jr.production_enabled(state) is False


def _fake_store(monkeypatch) -> None:
    store = types.SimpleNamespace(search=lambda q, top_k: [{"text": f"local{i}", "file_name": f"n{i}.md", "ref": f"n{i}.md"} for i in range(top_k)])
    monkeypatch.setitem(sys.modules, "api.knowledge.vector_store", types.SimpleNamespace(get_store=lambda: store, confidence_for_hit=lambda hit: (0.8, "high")))


def test_chat_uses_jev_only_when_promoted_and_falls_back_when_down(monkeypatch, tmp_path) -> None:
    _fake_store(monkeypatch)
    monkeypatch.setenv("VERTEX_CREDIT_MODE", "off")
    monkeypatch.setattr(jr, "STATE_PATH", tmp_path / "jev.json")
    calls: list = []
    monkeypatch.setattr(jr, "jev_rerank", lambda q, hits: (calls.append(q), (list(reversed(hits)), {"input_tokens": 10}))[1])

    r = build_chat_retrieval_context("残価の見方は？", rag_top_k=3, question_category="general", is_general_response_mode=False)
    assert calls == [] and r.rag_refs[0] == "n0.md"  # 未昇格なら使わない

    (tmp_path / "jev.json").write_text(json.dumps({"promoted": True}))
    r = build_chat_retrieval_context("残価の見方は？", rag_top_k=3, question_category="general", is_general_response_mode=False)
    assert calls and r.jev_rerank["used"] is True and r.rag_refs[0] == "n5.md"

    def down(q, hits):
        raise RuntimeError("jev down")

    monkeypatch.setattr(jr, "jev_rerank", down)
    r = build_chat_retrieval_context("残価の見方は？", rag_top_k=3, question_category="general", is_general_response_mode=False)
    assert r.jev_rerank["status"] == "error" and r.rag_refs[0] == "n0.md"


def test_eval_scores_jev_mechanically_with_fallback_and_covered_subset() -> None:
    cases = [
        {"id": "a", "query": "残価", "expected_path_any": ["Research/残価.md"], "forbidden_path_any": []},
        {"id": "b", "query": "補助金", "expected_path_any": ["リース知識/補助金.md"], "forbidden_path_any": []},
    ]
    chroma = {"残価": [{"file_path": "x.md"}, {"file_path": "Research/残価.md"}], "補助金": [{"file_path": "y.md"}, {"file_path": "リース知識/補助金.md"}]}

    def jev(query, hits):
        if query == "補助金":
            raise RuntimeError("down")
        return list(reversed(hits)), {"calls": 1, "input_tokens": 2_000_000}

    result = vx_eval.run_eval(
        cases,
        chroma_search=lambda q, k: chroma[q][:k],
        rerank=lambda q, hits: hits,
        vertex_search=lambda q, k: [{"file_path": "Research/残価.md"}] if q == "残価" else [],
        vertex_answer=lambda q: {"refs": [], "status": "ok"},
        jev_rerank=jev,
        covered_ids={"a"},
    )
    assert result["chroma"]["mrr"] == 0.5
    assert result["jev_rerank"]["mrr"] == pytest.approx(0.75)  # a は1位へ、b は Jev 不通で元の2位のまま
    assert result["jev_rerank"]["calls"] == 1 and result["jev_rerank"]["failures"] == 1
    assert result["jev_rerank"]["estimated_jpy"] == pytest.approx(2_000_000 * 42 / 1e9 * 160)
    assert result["vertex_covered"]["total"] == 1 and result["vertex_covered"]["vertex_search"]["hit_at_k_rate"] == 1.0


def test_jev_promotion_is_tracked_separately(tmp_path) -> None:
    path = tmp_path / "jev.json"
    night = {"chroma": {"hit_at_k_rate": 0.55, "mrr": 0.40}, "jev_rerank": {"hit_at_k_rate": 0.60, "mrr": 0.50}}
    for i in range(3):
        promoted, reason = vx_eval.update_jev_state(night, f"2026-10-0{i + 3}T05:40", path)
    assert promoted is True and json.loads(path.read_text())["promoted"] is True
    worse = {"chroma": {"hit_at_k_rate": 0.55, "mrr": 0.40}, "jev_rerank": {"hit_at_k_rate": 0.50, "mrr": 0.42}}
    assert vx_eval.update_jev_state(worse, "2026-10-06T05:40", path)[0] is False


def test_vertex_covered_ids_reads_export_manifest(tmp_path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"documents": [{"source_path": "Projects/tune_lease_55/Research/残価.md"}]}, ensure_ascii=False))
    cases = [{"id": "a", "expected_path_any": ["Research/残価.md"]}, {"id": "b", "expected_path_any": ["リース知識/x.md"]}]
    assert vx_eval.vertex_covered_ids(cases, manifest) == {"a"}
