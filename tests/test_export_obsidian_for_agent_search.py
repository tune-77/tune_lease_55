import json
from pathlib import Path

from scripts import export_obsidian_for_agent_search as exporter


def _write_note(vault: Path, rel: str, body: str) -> Path:
    path = vault / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_export_filters_private_notes_and_writes_quality_metadata(tmp_path):
    vault = tmp_path / "vault"
    project = "Projects/tune_lease_55"
    output = tmp_path / "export"

    _write_note(
        vault,
        f"{project}/Research/補助金 総合レポート.md",
        """---
canonical_topic: 補助金活用
---
# 補助金 総合レポート

## 確認論点
リース審査では補助金の対象経費、交付決定、自己資金、返済余力、設備の必要性を確認する。
判断資産として、条件、リスク、根拠、反証材料、更新トリガーを残す。
出典と適用条件を分け、稟議では補助金が遅れた場合の資金繰りも確認する。
設備導入後の売上寄与、対象設備の用途、補助対象外費用、交付前発注の可否も分けて確認する。
審査コメントでは、補助金が入らない場合でも返済できるかを別シナリオで残す。
""",
    )
    _write_note(
        vault,
        f"{project}/Research/非公開メモ.md",
        """---
public: false
---
# 非公開メモ

リース審査の判断、条件、リスク、根拠、出典を含むが外へ出さない。
""",
    )

    exported = exporter.export_notes(vault, output, 10, "gs://bucket/prefix")

    assert len(exported) == 1
    note = exported[0]
    assert note.canonical_topic == "補助金活用"
    assert note.source_bucket == "research"
    assert note.quality_score >= 5

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rejected_summary"]["private_marker"] == 1
    assert manifest["quality_gate"]["dedupe"] == "canonical_topic"

    records = [
        json.loads(line)
        for line in (output / "documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["structData"]["canonical_topic"] == "補助金活用"
    assert records[0]["structData"]["quality_score"] == note.quality_score


def test_export_deduplicates_by_canonical_topic_preferring_better_source(tmp_path):
    vault = tmp_path / "vault"
    project = "Projects/tune_lease_55"
    output = tmp_path / "export"

    _write_note(
        vault,
        f"{project}/Research/返済余力確認_下書き.md",
        """---
canonical_topic: 返済余力確認
---
# 返済余力確認 下書き

## 確認論点
リース審査では資金繰り、返済余力、条件、リスク、根拠、出典、更新を確認する。
稟議では短期借入と営業キャッシュフローも見る。
売上入金の季節性、既存借入の返済予定、設備導入後の資金固定化も確認する。
反証材料、適用条件、注意点、質問事項も残し、審査コメントの再利用に備える。
""",
    )
    _write_note(
        vault,
        f"{project}/Judgment Assets/返済余力確認.md",
        """---
canonical_topic: 返済余力確認
---
# 返済余力確認

## 判断資産
リース審査では返済余力、資金繰り、短期借入、営業キャッシュフローを確認する。
条件、リスク、根拠、反証、出典、適用、更新トリガーを分けて稟議へ残す。
補助金や設備投資のタイミングがずれる場合は、資金ショートの兆候を質問する。
月商比の借入水準、仕入債務の増減、入金サイトの長期化、代表者借入への依存も確認する。
""",
    )

    exported = exporter.export_notes(vault, output, 10, "gs://bucket/prefix")

    assert len(exported) == 1
    assert exported[0].source_path.endswith("Judgment Assets/返済余力確認.md")
    assert exported[0].source_bucket == "judgment_assets"

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rejected_summary"]["duplicate_canonical_topic"] == 1


def test_quality_gate_rejects_thin_keyword_only_note(tmp_path):
    vault = tmp_path / "vault"
    project = "Projects/tune_lease_55"
    output = tmp_path / "export"

    _write_note(
        vault,
        f"{project}/Research/薄いメモ.md",
        "# 薄いメモ\n\nリース 審査 リスク 補助金 与信 物件 これは単語だけのメモです。" * 4,
    )

    exported = exporter.export_notes(vault, output, 10, "gs://bucket/prefix")

    assert exported == []
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rejected_summary"]["quality_gate"] == 1
