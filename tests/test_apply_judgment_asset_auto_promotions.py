import json

from scripts import apply_judgment_asset_auto_promotions as auto_apply


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_is_enabled_reads_env_var(monkeypatch):
    monkeypatch.delenv(auto_apply.ENABLE_ENV_VAR, raising=False)
    assert auto_apply.is_enabled() is False

    monkeypatch.setenv(auto_apply.ENABLE_ENV_VAR, "1")
    assert auto_apply.is_enabled() is True

    monkeypatch.setenv(auto_apply.ENABLE_ENV_VAR, "true")
    assert auto_apply.is_enabled() is True

    monkeypatch.setenv(auto_apply.ENABLE_ENV_VAR, "0")
    assert auto_apply.is_enabled() is False


def test_run_promotes_only_auto_apply_eligible_candidate(tmp_path):
    candidates_jsonl = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates_jsonl,
        [
            {"id": "cand-eligible", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-a"},
            {"id": "cand-low-score", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-b"},
        ],
    )
    state_json = tmp_path / "state.json"
    _write_json(
        state_json,
        {
            # score = 2*4 + 1*3 = 11 (>=8), evidence = 2+1+0 = 3 (>=3) -> eligible
            "cand-eligible": {"useful_count": 2, "edit_count": 1, "rejected_count": 0},
            # score = 1*4 = 4 (<8) -> ready_for_review だが auto_apply対象外
            "cand-low-score": {"useful_count": 1, "rejected_count": 0},
        },
    )
    canonical_json = tmp_path / "canonical.json"
    _write_json(canonical_json, {"rules": []})
    log_jsonl = tmp_path / "log.jsonl"

    calls = []

    def fake_promote_fn(candidate_id, *, promoted_by):
        calls.append((candidate_id, promoted_by))
        return {"status": "promoted", "rule": {"id": f"rule-{candidate_id}"}}

    result = auto_apply.run(
        candidates_jsonl=candidates_jsonl,
        state_json=state_json,
        canonical_json=canonical_json,
        log_jsonl=log_jsonl,
        target_date="2026-09-06",
        promote_fn=fake_promote_fn,
    )

    assert calls == [("cand-eligible", "auto_apply_pilot")]
    assert result["eligible_count"] == 1
    assert result["promoted_count"] == 1
    assert result["error_count"] == 0
    assert result["promoted_ids"] == ["cand-eligible"]

    log_lines = log_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    log_entry = json.loads(log_lines[0])
    assert log_entry["candidate_id"] == "cand-eligible"
    assert log_entry["rule_id"] == "rule-cand-eligible"
    assert log_entry["promoted_by"] == "auto_apply_pilot"


def test_run_records_error_without_stopping_other_candidates(tmp_path):
    candidates_jsonl = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates_jsonl,
        [
            {"id": "cand-fails", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-a"},
            {"id": "cand-ok", "claim": "十分に長い別の判断資産候補の文面です。", "research_topic": "topic-b"},
        ],
    )
    state_json = tmp_path / "state.json"
    _write_json(
        state_json,
        {
            "cand-fails": {"useful_count": 3, "edit_count": 0, "rejected_count": 0},
            "cand-ok": {"useful_count": 3, "edit_count": 0, "rejected_count": 0},
        },
    )
    canonical_json = tmp_path / "canonical.json"
    _write_json(canonical_json, {"rules": []})
    log_jsonl = tmp_path / "log.jsonl"

    def flaky_promote_fn(candidate_id, *, promoted_by):
        if candidate_id == "cand-fails":
            raise RuntimeError("boom")
        return {"status": "promoted", "rule": {"id": "rule-ok"}}

    result = auto_apply.run(
        candidates_jsonl=candidates_jsonl,
        state_json=state_json,
        canonical_json=canonical_json,
        log_jsonl=log_jsonl,
        target_date="2026-09-06",
        promote_fn=flaky_promote_fn,
    )

    assert result["eligible_count"] == 2
    assert result["promoted_count"] == 1
    assert result["error_count"] == 1
    assert result["errors"][0]["id"] == "cand-fails"
    assert result["promoted_ids"] == ["cand-ok"]


def test_run_with_no_eligible_candidates_writes_no_log(tmp_path):
    candidates_jsonl = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates_jsonl,
        [{"id": "cand-1", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-a"}],
    )
    state_json = tmp_path / "state.json"
    _write_json(state_json, {"cand-1": {}})
    canonical_json = tmp_path / "canonical.json"
    _write_json(canonical_json, {"rules": []})
    log_jsonl = tmp_path / "log.jsonl"

    def promote_fn(candidate_id, *, promoted_by):
        raise AssertionError("should not be called")

    result = auto_apply.run(
        candidates_jsonl=candidates_jsonl,
        state_json=state_json,
        canonical_json=canonical_json,
        log_jsonl=log_jsonl,
        target_date="2026-09-06",
        promote_fn=promote_fn,
    )

    assert result["eligible_count"] == 0
    assert result["promoted_count"] == 0
    assert not log_jsonl.exists()


def test_main_is_noop_when_disabled(monkeypatch, capsys):
    monkeypatch.delenv(auto_apply.ENABLE_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "sys.argv",
        ["apply_judgment_asset_auto_promotions.py"],
    )

    exit_code = auto_apply.main()

    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["enabled"] is False
