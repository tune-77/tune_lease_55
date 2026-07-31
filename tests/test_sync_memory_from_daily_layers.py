import json

import scripts.sync_memory_from_daily as syncer


def test_sync_memory_routes_persistent_promotions(tmp_path, monkeypatch):
    repo = tmp_path
    memory_dir = repo / "memory"
    memory_dir.mkdir()
    (repo / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (repo / "PERSISTENT_MEMORY.md").write_text("# Persistent\n", encoding="utf-8")
    (memory_dir / "2026-07-31.md").write_text(
        "\n".join(
            [
                "## Promotable Items",
                "- 永続記憶として、紫苑の運用原則は記憶の寿命と役割を分けて扱うこと。",
                "- 今後も、判断軸は採用・保留・却下の結果で更新する。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(syncer, "PROJECT_ROOT", repo)
    monkeypatch.setattr(syncer, "MEMORY_DIR", memory_dir)
    monkeypatch.setattr(syncer, "MEMORY_FILE", repo / "MEMORY.md")
    monkeypatch.setattr(syncer, "PERSISTENT_MEMORY_FILE", repo / "PERSISTENT_MEMORY.md")
    monkeypatch.setattr(syncer, "STATE_FILE", memory_dir / "memory_sync_state.json")

    result = syncer.sync_memory()

    assert result["promoted"] == 2
    assert result["persistent_promoted"] == 1
    assert result["long_term_promoted"] == 1
    assert "永続記憶として" in (repo / "PERSISTENT_MEMORY.md").read_text(encoding="utf-8")
    assert "今後も" in (repo / "MEMORY.md").read_text(encoding="utf-8")
    state = json.loads((memory_dir / "memory_sync_state.json").read_text(encoding="utf-8"))
    assert len(state["promoted_keys"]) == 2
