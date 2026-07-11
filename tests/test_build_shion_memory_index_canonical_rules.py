import json

from scripts import build_shion_memory_index as builder


def test_canonical_judgment_rules_become_judgment_memory_records(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    data_dir = repo / "data"
    data_dir.mkdir(parents=True)
    rules_path = data_dir / "canonical_judgment_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "id": "rule_1",
                        "status": "active",
                        "private": False,
                        "concept": "support_specificity",
                        "canonical_statement": "銀行支援は対象リースへの直接性を確認する。",
                        "confidence": 0.9,
                        "evidence_count": 3,
                        "user_evidence_count": 1,
                        "evidence_paths": ["dialogue.md"],
                    },
                    {
                        "id": "rule_2",
                        "status": "candidate",
                        "private": False,
                        "canonical_statement": "candidateは入れない。",
                    },
                    {
                        "id": "rule_3",
                        "status": "active",
                        "private": True,
                        "canonical_statement": "privateは入れない。",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "REPO_ROOT", repo)

    records = builder._canonical_judgment_rule_records(rules_path)

    assert len(records) == 1
    assert records[0]["memory_type"] == "judgment_memory"
    assert records[0]["source"] == "canonical_judgment_rules"
    assert records[0]["topic"] == "support_specificity"
    assert records[0]["evidence_count"] == 3
    assert records[0]["user_evidence_count"] == 1
