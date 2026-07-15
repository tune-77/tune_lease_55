import json
from pathlib import Path

from scripts import import_obsidian_knowledge_to_judgment_assets as importer


def test_extract_candidates_from_asset_knowledge_note(tmp_path):
    vault = tmp_path / "vault"
    note = vault / "Projects/tune_lease_55/Asset Knowledge/工作機械.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """# 工作機械

## 稟議で使えそうな根拠
- 保守契約、定期点検、精度測定記録、主軸交換履歴があると残価リスクの緩和材料になる。

## 雑談
- これは短い。
""",
        encoding="utf-8",
    )

    rows = importer.extract_candidates_from_note(note, vault)

    assert len(rows) == 1
    assert rows[0]["research_topic"] == "obsidian_knowledge"
    assert rows[0]["candidate_type"] == "caution"
    assert "精度測定記録" in rows[0]["claim"]
    assert rows[0]["promotion_status"] == "not_promoted"
    assert rows[0]["requires_human_use_feedback"] is True


def test_import_knowledge_candidates_appends_and_dedupes(tmp_path):
    vault = tmp_path / "vault"
    rel_dir = Path("Projects/tune_lease_55/Lease Intelligence/Knowledge")
    note = vault / rel_dir / "運用ルール.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        """# 運用ルール

## 参照優先順位
1. 個別案件・最新運用・ユーザー訂正は、Obsidian知識、業務メモ、現行コード仕様を優先。
""",
        encoding="utf-8",
    )
    output = tmp_path / "candidates.jsonl"

    first = importer.import_knowledge_candidates(
        vault=vault,
        source_dirs=[rel_dir],
        output_jsonl=output,
    )
    second = importer.import_knowledge_candidates(
        vault=vault,
        source_dirs=[rel_dir],
        output_jsonl=output,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert first["imported"] == 1
    assert second["imported"] == 0
    assert len(rows) == 1
    assert rows[0]["import_source"] == "obsidian_knowledge"


def test_render_markdown_reports_import_counts():
    markdown = importer.render_markdown(
        {
            "scanned_notes": 2,
            "extracted": 3,
            "imported": 1,
            "imported_items": [
                {
                    "id": "abc",
                    "candidate_type": "application_rule",
                    "claim": "稟議では契約と入金時期を確認する。",
                    "evidence_path": "Knowledge/a.md",
                    "source_section": "稟議",
                }
            ],
        }
    )

    assert "Scanned notes: 2" in markdown
    assert "Imported new candidates: 1" in markdown
    assert "稟議では契約と入金時期" in markdown
