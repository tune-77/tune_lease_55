from scripts import build_cloud_chat_memory_pack as pack


def test_collect_mid_term_memory_uses_timeline_delta(monkeypatch):
    monkeypatch.setattr(
        "scripts.build_shion_timeline_delta.build_timeline_delta",
        lambda memory_dir, target_day, days=4: {
            "memory_layers": {
                "mid_term": {
                    "signals": {
                        "repeated_terms": ["前回", "判断"],
                        "continued_terms": ["圧点"],
                    },
                    "items": ["同じ不満が続く時だけ応答方針を少し変える。"],
                }
            }
        },
    )

    items = pack.collect_mid_term_memory(limit=3)

    assert items[0].startswith("最近の継続論点:")
    assert "前回" in items[0]
    assert "同じ不満" in items[1]


def test_collect_long_term_memory_includes_persistent_memory(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "MEMORY.md").write_text(
        "# Memory\n- 長期記憶は判断軸を守り、次回の応答でぶれないようにする。\n",
        encoding="utf-8",
    )
    (repo / "PERSISTENT_MEMORY.md").write_text(
        "# Persistent Memory\n- 永続記憶は人格と運用原則を支え、短期の気分では書き換えない。\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(pack, "PROJECT_ROOT", repo)
    monkeypatch.setattr(pack, "LONG_TERM_MEMORY", repo / "MEMORY.md")
    monkeypatch.setattr(pack, "PERSISTENT_MEMORY", repo / "PERSISTENT_MEMORY.md")

    items = pack.collect_long_term_memory(limit=10)

    assert any("永続記憶" in item for item in items)
    assert any("長期記憶" in item for item in items)
