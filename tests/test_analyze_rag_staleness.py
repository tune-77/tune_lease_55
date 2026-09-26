import json
import sys
import types
from datetime import timezone

from scripts import analyze_rag_staleness as staleness


def test_shadow_log_jev_relevance_noop_when_mode_off(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_STALENESS_MODE", raising=False)
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(staleness, "STALENESS_SHADOW_LOG", log_path)

    staleness.shadow_log_jev_relevance(
        [{"obsidian_ref": "a", "category": "stale"}], [], {}
    )

    assert not log_path.exists()


def test_shadow_log_jev_relevance_noop_without_shared_context_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_STALENESS_MODE", "shadow")
    monkeypatch.delenv("TYPESAFE_ALLOW_SHARED_CONTEXT", raising=False)
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(staleness, "STALENESS_SHADOW_LOG", log_path)

    staleness.shadow_log_jev_relevance(
        [{"obsidian_ref": "a", "category": "stale"}], [], {}
    )

    assert not log_path.exists()


def test_shadow_log_jev_relevance_logs_agreement_with_heuristic(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_STALENESS_MODE", "shadow")
    monkeypatch.setenv("TYPESAFE_ALLOW_SHARED_CONTEXT", "1")
    log_path = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(staleness, "STALENESS_SHADOW_LOG", log_path)

    fake_module = types.ModuleType("typesafe_rag_guard")
    fake_module.typesafe_rag_enabled = lambda: True
    fake_module.build_passage_request = lambda query, hits: {
        "state": {"query": query, "passages": hits},
        "questions": {},
    }
    fake_module.request_system_one = lambda payload: {
        "model": "jev-test",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "answers": {
            "p0_relevant": {"type": "noul", "noul": 0.9},
            "p0_evidence": {"type": "noul", "noul": 0.8},
            "p0_contradicts": {"type": "noul", "noul": 0.1},
            "p0_injection": {"type": "noul", "noul": 0.1},
        },
    }
    fake_module.route_passage = lambda judgment: "include"
    monkeypatch.setitem(sys.modules, "typesafe_rag_guard", fake_module)

    important_nodes = [{"obsidian_ref": "[[重要#概要]]", "category": "important_but_unused"}]
    staleness.shadow_log_jev_relevance([], important_nodes, {})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["obsidian_ref"] == "[[重要#概要]]"
    assert record["jev_route"] == "include"
    assert record["agreement"] is True


def test_collect_last_access_reads_nested_search_results(tmp_path):
    log_path = tmp_path / "rag_search_log.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "ts": "2026-08-09T01:02:03+00:00",
                "query": "工作機械の再販リスク",
                "results": [
                    {"obsidian_ref": "[[工作機械 中古相場・残価・再販リスク#概要]]"},
                    {"ref": "[[物件残存価値・処分実務#概要]]"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = staleness.collect_last_access_from_jsonl(log_path)

    assert result["[[工作機械 中古相場・残価・再販リスク#概要]]"].tzinfo == timezone.utc
    assert result["[[物件残存価値・処分実務#概要]]"].tzinfo == timezone.utc


def test_collect_refs_and_meta_uses_current_vector_store_collection(monkeypatch):
    class FakeCollection:
        def get(self, include):
            assert include == ["metadatas"]
            return {
                "metadatas": [
                    {"obsidian_ref": "[[重要ノート#概要]]", "file_name": "重要ノート.md"},
                    {"obsidian_ref": "", "file_name": "empty.md"},
                ]
            }

    class FakeStore:
        def __init__(self):
            self._collection = None

        def _ensure_collection(self):
            self._collection = FakeCollection()

    fake_module = types.ModuleType("api.knowledge.vector_store")
    fake_module.KnowledgeVectorStore = FakeStore
    monkeypatch.setitem(sys.modules, "api.knowledge.vector_store", fake_module)

    result = staleness.collect_refs_and_meta_from_chroma()

    assert result == {
        "[[重要ノート#概要]]": {
            "obsidian_ref": "[[重要ノート#概要]]",
            "file_name": "重要ノート.md",
        }
    }
