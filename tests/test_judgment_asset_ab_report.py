from scripts import build_judgment_asset_ab_report as ab


def test_build_ab_report_pairs_same_concept_rules():
    canonical = {
        "rules": [
            {"id": "a", "status": "active", "concept": "asset_life", "canonical_statement": "期間を見る", "confidence": 0.8},
            {"id": "b", "status": "active", "concept": "asset_life", "canonical_statement": "出口を見る", "confidence": 0.7},
            {"id": "c", "status": "candidate", "concept": "asset_life", "canonical_statement": "候補は除外"},
        ]
    }
    feedback = [{"rule_id": "b", "outcome": "helped"}]

    payload = ab.build_report(canonical, feedback)

    assert payload["summary"]["candidate_pairs"] == 1
    pair = payload["pairs"][0]
    assert pair["concept"] == "asset_life"
    assert {pair["asset_a"]["rule_id"], pair["asset_b"]["rule_id"]} == {"a", "b"}
    assert "自動で勝者確定しない" in pair["winner_policy"]
