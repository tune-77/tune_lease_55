import datetime as dt
import json

import pytest

from scripts import judgment_asset_dedup as dedup


def _rule(rid, text, **extra):
    return {
        "id": rid,
        "status": "active",
        "concept": "chat_judgment_teaching",
        "material_type": "judgment_rule",
        "canonical_statement": text,
        "evidence_count": 1,
        "user_evidence_count": 1,
        "confidence": 0.8,
        "evidence_paths": [f"path/{rid}"],
        "private": False,
        **extra,
    }


def _no_embedding(texts):
    return None


def test_normalize_ignores_user_fillers_and_punctuation():
    assert dedup.normalize_statement("水害が増えた 動産保険ついているから（そうだね 判断資産にしておいて）") == dedup.normalize_statement(
        "水害が増えた、動産保険ついているから"
    )


def test_detect_auto_merges_only_identical_text_and_skips_different_numbers():
    rules = [
        _rule("a", "水害が増えたせいでリースにしたい企業が増えると思う 動産保険ついているから"),
        _rule("b", "水害が増えたせいでリースにしたい企業が増えると思う 動産保険ついているから（そうだね 判断資産にしておいて）"),
        _rule("c", "新車登録が5年以内で走行距離が20万キロ位ならリースでもやる"),
        _rule("d", "新車登録が3年以内で走行距離が20万キロ位ならリースでもやる"),
        _rule("e", "中古自動車の見積書は要注意、業者によって金額の入れ方が違う", status="merged"),
    ]
    found = dedup.detect(rules, similarity=None)
    assert found["auto_clusters"] == [{"representative_id": "a", "source_ids": ["b"]}]
    assert all({c["representative_id"], c["source_id"]} != {"c", "d"} for c in found["candidates"])
    assert all("e" not in (c["representative_id"], c["source_id"]) for c in found["candidates"])


def test_merge_into_keeps_user_words_and_records_provenance():
    rules = [_rule("rep", "銀行と取引のない企業とは付き合わない"), _rule("src", "銀行との取引がないと取扱いしない")]
    result = dedup.merge_into(rules, "rep", ["src"], reason="test", now="2026-10-04T00:30:00")
    rep, src = rules
    assert rep["canonical_statement"] == "銀行と取引のない企業とは付き合わない（同旨: 銀行との取引がないと取扱いしない）"
    assert rep["pre_merge_statement"] == "銀行と取引のない企業とは付き合わない"
    assert rep["merged_from"][0]["canonical_statement"] == "銀行との取引がないと取扱いしない"
    assert rep["merged_from"][0]["origin"] == "user_chat"
    assert rep["evidence_count"] == 2
    assert src["status"] == "merged" and src["merged_into"] == "rep"
    assert result["source_ids"] == ["src"]


def test_merge_into_rejects_inactive_rules():
    rules = [_rule("rep", "返済原資を確認する"), _rule("src", "返済原資を確かめる", status="merged")]
    with pytest.raises(ValueError):
        dedup.merge_into(rules, "rep", ["src"], reason="test")


def _write_store(path, rules):
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")


def test_weekly_run_backs_up_merges_and_queues_candidates(tmp_path):
    canonical = tmp_path / "canonical_judgment_rules.json"
    _write_store(
        canonical,
        [
            _rule("a", "3月と9月にサプライヤーの売り込みがありリース契約が増加する傾向がある 覚えておいて"),
            _rule("b", "3月と9月にサプライヤーの売り込みがありリース契約が増加する傾向がある"),
            _rule("c", "ゼロゼロ融資の返済計画は現実的か、資金繰り表で返済余力を確認する"),
            _rule("d", "ゼロゼロ融資の返済計画と、資金繰りに与える影響を詳細に確認する"),
        ],
    )
    paths = dict(
        canonical_path=canonical,
        candidates_path=tmp_path / "merge_candidates.json",
        history_path=tmp_path / "history.jsonl",
        latest_path=tmp_path / "latest.json",
        vault_dir=tmp_path / "vault",
        similarity_fn=_no_embedding,
        today=dt.date(2026, 10, 4),
    )

    dry = dedup.run(dry_run=True, **paths)
    assert dry["auto_merged_sources"] == 1
    assert json.loads(canonical.read_text(encoding="utf-8"))["rules"][0]["status"] == "active"
    assert not (tmp_path / "latest.json").exists()

    report = dedup.run(dry_run=False, jev_fn=lambda pairs: [0.9] * len(pairs), jev_drop_below=0.2, **paths)
    store = json.loads(canonical.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in store["rules"]}
    assert report["before_active"] == 4 and report["after_active"] == 3
    assert by_id["a"]["status"] == "merged" and by_id["a"]["merged_into"] == "b"
    assert list((tmp_path / "backups").glob("canonical_judgment_rules.before_weekly_dedup_*.json"))
    pending = dedup.pending_candidates(canonical_path=canonical, candidates_path=tmp_path / "merge_candidates.json")
    assert [(c["representative_id"], c["source_id"]) for c in pending] in ([("c", "d")], [("d", "c")])
    assert pending[0]["jev_same_asset"] == 0.9
    assert (tmp_path / "vault" / "判断資産 重複整理 週次 2026-10-04.md").exists()
    assert "active 4→3件" in dedup.morning_report_line(tmp_path / "latest.json")

    # 2回目は同じペアを再判定・再登録しない
    again = dedup.run(dry_run=False, jev_fn=lambda pairs: pytest.fail("should not re-judge"), **paths)
    assert again["new_candidates"] == 0 and again["auto_merged_sources"] == 0


def test_weekly_run_drops_low_jev_pairs_but_remembers_them(tmp_path):
    canonical = tmp_path / "canonical_judgment_rules.json"
    _write_store(canonical, [_rule("c", "ゼロゼロ融資の返済計画は現実的か確認する"), _rule("d", "ゼロゼロ融資の返済負担が大きい場合は要注意")])
    queue = tmp_path / "merge_candidates.json"
    report = dedup.run(
        dry_run=False,
        canonical_path=canonical,
        candidates_path=queue,
        history_path=tmp_path / "h.jsonl",
        latest_path=tmp_path / "latest.json",
        vault_dir=tmp_path / "vault",
        similarity_fn=_no_embedding,
        jev_fn=lambda pairs: [0.05] * len(pairs),
        jev_drop_below=0.2,
    )
    assert report["new_candidates"] == 0 and report["jev"]["dropped"] == 1
    stored = json.loads(queue.read_text(encoding="utf-8"))["candidates"]
    assert [c["status"] for c in stored.values()] == ["dropped_by_jev"]


def test_jev_failure_falls_back_to_strong_similarity_only(tmp_path):
    canonical = tmp_path / "canonical_judgment_rules.json"
    _write_store(canonical, [_rule("c", "ゼロゼロ融資の返済計画は現実的か、資金繰り表で返済余力を確認する"), _rule("d", "ゼロゼロ融資の返済計画は現実的か、資金繰り表で返済懸念を確認する")])

    def broken(pairs):
        raise RuntimeError("down")

    report = dedup.run(
        dry_run=False,
        canonical_path=canonical,
        candidates_path=tmp_path / "q.json",
        history_path=tmp_path / "h.jsonl",
        latest_path=tmp_path / "latest.json",
        vault_dir=tmp_path / "vault",
        similarity_fn=_no_embedding,
        jev_fn=broken,
    )
    assert report["jev"]["status"] == "failed"
    assert report["new_candidates"] == 1


def test_merge_and_reject_candidate(tmp_path):
    canonical = tmp_path / "canonical_judgment_rules.json"
    _write_store(canonical, [_rule("rep", "日銀の金利が上がりそうだ するとリースの金利も上がる"), _rule("src", "金利は影響するだろうね 実際に契約する金利は上がってきた"), _rule("x", "別の資産")])
    queue = tmp_path / "q.json"
    queue.write_text(
        json.dumps(
            {
                "candidates": {
                    "m1": {"id": "m1", "representative_id": "rep", "source_id": "src", "status": "pending", "jaccard": 0.2},
                    "m2": {"id": "m2", "representative_id": "rep", "source_id": "x", "status": "pending", "jaccard": 0.1},
                }
            }
        ),
        encoding="utf-8",
    )
    result = dedup.merge_candidate("m1", canonical_path=canonical, candidates_path=queue)
    assert result["active_rules"] == 2
    assert dedup.reject_candidate("m2", candidates_path=queue)["status"] == "rejected"
    with pytest.raises(KeyError):
        dedup.merge_candidate("m1", canonical_path=canonical, candidates_path=queue)
    assert dedup.pending_candidates(canonical_path=canonical, candidates_path=queue) == []


def test_jev_request_and_answer_parsing():
    payload = dedup.build_jev_pair_request([("a", "b"), ("c", "d")])
    assert set(payload["questions"]) == {"pair0_same_asset", "pair1_same_asset"}
    body = {"answers": {"pair0_same_asset": {"type": "noul", "noul": 0.7}, "pair1_same_asset": {"type": "noul", "noul": 0.1}}}
    assert dedup.parse_jev_answers(body, 2) == [0.7, 0.1]


def test_morning_report_line_without_runs(tmp_path):
    assert "まだ実行されていません" in dedup.morning_report_line(tmp_path / "missing.json")
