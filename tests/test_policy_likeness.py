import json

from api import policy_likeness as pl


def _canonical(tmp_path, rules):
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps({"rules": rules}, ensure_ascii=False), encoding="utf-8")
    return path


def test_tier_is_conservative_and_human_decision_wins():
    assert pl.tier_for(0.88, rule_kind="insight") == "guideline"  # Jev 単独では方針にしない（現状の最大値）
    assert pl.tier_for(0.95, rule_kind="insight") == "policy"
    assert pl.tier_for(0.5, rule_kind="insight") == "insight"
    assert pl.tier_for(None, rule_kind="insight") == "insight"
    assert pl.tier_for(0.1, rule_kind="policy") == "policy"  # 決定的ルールの方針は従来どおり
    assert pl.tier_for(0.7, rule_kind="insight", decision=True) == "policy"
    assert pl.tier_for(0.7, rule_kind="policy", decision=False) == "insight"


def test_score_review_and_decide_roundtrip(tmp_path, monkeypatch):
    log = tmp_path / "jev_log.jsonl"
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", str(log))
    canonical = _canonical(
        tmp_path,
        [
            {"id": "a", "status": "active", "canonical_statement": "ダクト付きの空調機はリースできない。（同旨: 別文）"},
            {"id": "b", "status": "active", "canonical_statement": "市場金利は上がっている。"},
            {"id": "c", "status": "merged", "canonical_statement": "統合済み"},
        ],
    )
    queue = tmp_path / "queue.json"
    sent = []

    def judge(texts):
        sent.extend(texts)
        return [0.8 if "空調" in t else 0.2 for t in texts]

    result = pl.score_canonical_rules(canonical_path=canonical, queue_path=queue, judge=judge)
    assert result == {"targets": 2, "scored": 2, "failed": 0, "queued": 2}
    assert sent[0] == "ダクト付きの空調機はリースできない。"  # 同旨の追記は送らない

    review = pl.review_candidates(queue)
    assert [item["id"] for item in review["candidates"]] == ["a"]

    # 本文が変わらなければ再採点しない
    assert pl.score_canonical_rules(canonical_path=canonical, queue_path=queue, judge=judge)["targets"] == 0

    row = pl.record_decision("a", True, queue)
    assert row["tier"] == "policy" and row["decision"] is True
    assert pl.review_candidates(queue)["total_count"] == 0
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    judgments = [r for r in rows if r["record_type"] == "judgment"]
    assert {r["route"] for r in judgments} == {"guideline", "insight"}
    assert all(r["guard"] == "knowledge_kind_policy" and r["mode"] == "shadow" for r in judgments)
    labels = [r for r in rows if r["record_type"] == "label"]
    assert labels and labels[-1]["label"] is True and labels[-1]["label_source"] == "human:judgment_review"


def test_jev_failure_leaves_items_unscored_for_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", "off")
    canonical = _canonical(tmp_path, [{"id": "a", "status": "active", "canonical_statement": "要注意先には慎重になる"}])
    queue = tmp_path / "queue.json"
    result = pl.score_canonical_rules(canonical_path=canonical, queue_path=queue, judge=lambda texts: [None] * len(texts))
    assert result["scored"] == 0 and result["failed"] == 1
    assert pl.load_queue(queue) == {}


def test_changed_statement_drops_previous_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_JUDGMENT_LOG_PATH", "off")
    queue = tmp_path / "queue.json"
    canonical = _canonical(tmp_path, [{"id": "a", "status": "active", "canonical_statement": "旧本文の取扱い"}])
    pl.score_canonical_rules(canonical_path=canonical, queue_path=queue, judge=lambda texts: [0.7] * len(texts))
    pl.record_decision("a", True, queue)
    canonical = _canonical(tmp_path, [{"id": "a", "status": "active", "canonical_statement": "新本文の取扱い"}])
    pl.score_canonical_rules(canonical_path=canonical, queue_path=queue, judge=lambda texts: [0.7] * len(texts))
    row = pl.load_queue(queue)["a"]
    assert row["decision"] is None and row["tier"] == "guideline"


def test_decision_endpoint_rejects_on_cloud_run(monkeypatch):
    import pytest
    from fastapi import HTTPException

    from api.routers import feedback_loop

    monkeypatch.setenv("K_SERVICE", "lease")
    with pytest.raises(HTTPException) as excinfo:
        feedback_loop.post_policy_likeness_decision("a", feedback_loop.PolicyLikenessDecisionRequest(is_policy=True))
    assert excinfo.value.status_code == 409


def test_judge_with_jev_defaults_keychain_service(monkeypatch):
    # REV-496: 改善パイプラインの launchd には鍵の受け渡しが無く、毎回 TYPESAFE_API_KEY is not configured だった
    import typesafe_dedup_guard as transport

    monkeypatch.delenv("TYPESAFE_API_KEYCHAIN_SERVICE", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    seen: dict[str, str] = {}

    def fake_request(payload):
        import os

        seen["service"] = os.environ.get("TYPESAFE_API_KEYCHAIN_SERVICE", "")
        return {"answers": {"item0": {"noul": 0.8}}}

    monkeypatch.setattr(transport, "_default_request", fake_request)
    monkeypatch.setattr(transport, "_noul", lambda answers, key: answers[key]["noul"])

    assert pl.judge_with_jev(["設備の稼働率と返済原資を確認してから条件を決める方針。"]) == [0.8]
    assert seen["service"] == "typesafe-api-key"
