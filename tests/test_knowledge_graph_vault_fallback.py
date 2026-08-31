import json
from pathlib import Path

from api.routers import pipeline_misc


def _reset_cache() -> None:
    pipeline_misc._knowledge_graph_vault_cache.update(
        {"vault": "", "loaded_at": 0.0, "notes": {}}
    )


def test_knowledge_graph_display_path_normalizes_cloud_run_vault(monkeypatch):
    monkeypatch.setenv("GCS_VAULT_LOCAL_DIR", "/tmp/gcs_vault")

    assert pipeline_misc._knowledge_graph_display_path(
        "/tmp/gcs_vault/Projects/tune_lease_55/Research/rate.md"
    ) == "Projects/tune_lease_55/Research/rate.md"


def test_vault_fallback_builds_note_metadata(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    research = vault / "Projects" / "tune_lease_55" / "Research"
    research.mkdir(parents=True)
    (research / "金利.md").write_text(
        "# 金利\n\n## 判断\n金利上昇を確認する。[[残価]]\n\n## 反証\n固定金利なら影響は限定的。",
        encoding="utf-8",
    )
    private = vault / "Private Reflection"
    private.mkdir()
    (private / "秘密.md").write_text("## 非公開\n表示しない", encoding="utf-8")

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    _reset_cache()

    notes = pipeline_misc._load_knowledge_graph_vault_notes()

    note = notes["Projects/tune_lease_55/Research/金利.md"]
    assert note["category"] == "research"
    assert note["sections"] == {"判断", "反証"}
    assert note["wikilinks"] == {"残価"}
    assert note["chunk_count"] == 2
    assert "Private Reflection/秘密.md" not in notes


def test_graph_endpoint_fills_sparse_chroma_from_vault(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    wiki = vault / "リース知識"
    wiki.mkdir(parents=True)
    for index in range(35):
        (wiki / f"note-{index}.md").write_text(
            f"## 要点\n知識ノート {index} の本文です。",
            encoding="utf-8",
        )

    class FakeCollection:
        def count(self):
            return 1

        def get(self, include):
            return {
                "metadatas": [{
                    "file_name": "note-0.md",
                    "file_path": str(wiki / "note-0.md"),
                    "section": "要点",
                    "wikilinks": "",
                    "mtime": 1,
                }]
            }

    class FakeStore:
        def __init__(self):
            self._collection = FakeCollection()
            self._client = object()

        def _ensure_collection(self):
            return None

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "empty-data"))
    monkeypatch.setenv("CLOUDRUN_BUNDLE_DIR", str(tmp_path / "empty-bundle"))
    monkeypatch.setattr("api.knowledge.vector_store.get_store", lambda: FakeStore())
    _reset_cache()
    pipeline_misc._knowledge_graph_snapshot_cache.update(
        {"path": "", "mtime": 0.0, "notes": {}}
    )

    payload = pipeline_misc.get_knowledge_graph(limit=30)

    assert payload["summary"]["indexed_chunks"] == 1
    assert payload["summary"]["notes"] == 35
    assert payload["summary"]["vault_fallback_used"] is True
    assert payload["summary"]["graph_source"] == "chroma+vault"


def test_snapshot_fallback_uses_bundled_retrieval_graph(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot = {
        "nodes": [{
            "path": "Projects/tune_lease_55/Research/金利.md",
            "key": "金利",
            "headings": ["判断", "反証"],
            "links": ["残価"],
            "mtime": 123,
        }]
    }
    (data_dir / "obsidian_retrieval_graph.json").write_text(
        json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    pipeline_misc._knowledge_graph_snapshot_cache.update(
        {"path": "", "mtime": 0.0, "notes": {}}
    )

    notes = pipeline_misc._load_knowledge_graph_snapshot_notes()

    note = notes["Projects/tune_lease_55/Research/金利.md"]
    assert note["category"] == "research"
    assert note["sections"] == {"判断", "反証"}
    assert note["wikilinks"] == {"残価"}
