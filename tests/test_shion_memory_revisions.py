import json

from scripts.revise_shion_memory import append_revision, apply_revisions, load_revisions
import scripts.build_shion_memory_index as builder


def _index():
    return {
        "records": [
            {
                "id": "mem_old",
                "content": "コンテナの法定耐用年数は6年。",
                "memory_type": "factual_memory",
                "status": "active",
                "supersedes": [],
            },
            {
                "id": "mem_other",
                "content": "境界案件は条件付き承認を検討する。",
                "memory_type": "judgment_memory",
                "status": "active",
                "supersedes": [],
            },
        ]
    }


def test_apply_revisions_marks_revised_and_creates_successor():
    index = _index()
    revisions = [
        {
            "old_id": "mem_old",
            "new_content": "コンテナの法定耐用年数は7年（2026-07改訂）。",
            "reason": "制度改定",
        }
    ]

    summary = apply_revisions(index, revisions)

    by_id = {r["id"]: r for r in index["records"]}
    assert by_id["mem_old"]["status"] == "revised"
    successors = [r for r in index["records"] if "mem_old" in (r.get("supersedes") or [])]
    assert len(successors) == 1
    assert "7年" in successors[0]["content"]
    assert summary["revised"] == 1
    assert summary["superseded_created"] == 1

    # 再適用しても二重に増えない（冪等）
    summary2 = apply_revisions(index, revisions)
    assert summary2["revised"] == 0
    assert summary2["superseded_created"] == 0
    assert len(index["records"]) == 3


def test_apply_revisions_links_existing_successor():
    index = _index()
    revisions = [{"old_id": "mem_old", "new_id": "mem_other", "reason": "統合"}]

    summary = apply_revisions(index, revisions)

    by_id = {r["id"]: r for r in index["records"]}
    assert by_id["mem_old"]["status"] == "revised"
    assert by_id["mem_other"]["supersedes"] == ["mem_old"]
    assert summary["superseded_linked"] == 1


def test_append_and_load_revisions_roundtrip(tmp_path):
    path = tmp_path / "revisions.jsonl"
    append_revision(old_id="mem_a", reason="test", new_content="新しい結論", path=path)

    revisions = load_revisions(path)

    assert len(revisions) == 1
    assert revisions[0]["old_id"] == "mem_a"
    assert revisions[0]["new_content"] == "新しい結論"


def test_build_index_applies_revisions(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "memory").mkdir()
    (repo / "data").mkdir()
    (repo / "MEMORY.md").write_text("- コンテナの法定耐用年数は6年と記録した。\n", encoding="utf-8")

    monkeypatch.setattr(builder, "REPO_ROOT", repo)
    base_index = builder.build_index()
    old_id = base_index["records"][0]["id"]

    (repo / "data" / "shion_memory_revisions.jsonl").write_text(
        json.dumps({"old_id": old_id, "new_content": "コンテナの法定耐用年数は7年（改訂）。", "reason": "制度改定"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    index = builder.build_index()

    by_id = {r["id"]: r for r in index["records"]}
    assert by_id[old_id]["status"] == "revised"
    assert any(old_id in (r.get("supersedes") or []) for r in index["records"])
