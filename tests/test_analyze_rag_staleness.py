import json
import sys
import types
from datetime import timezone

from scripts import analyze_rag_staleness as staleness


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
