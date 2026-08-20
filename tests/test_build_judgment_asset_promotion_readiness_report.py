import json

from scripts import build_judgment_asset_promotion_readiness_report as readiness


def _canonical_with_active_statement():
    return {
        "rules": [
            {
                "id": "existing-rule",
                "status": "active",
                "canonical_statement": "既存の正規判断資産の文面です。",
            }
        ]
    }


def test_ready_for_review_when_score_positive_and_no_rejections():
    candidates = [
        {"id": "cand-1", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-a"},
    ]
    state = {"cand-1": {"useful_count": 2, "use_count": 1, "rejected_count": 0}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["ready_for_review"] == 1
    entry = payload["buckets"]["ready_for_review"][0]
    assert entry["id"] == "cand-1"
    assert entry["score"] == 2 * 4 + 1  # useful*4 + use*1


def test_caution_mixed_signal_when_score_positive_but_rejected():
    candidates = [
        {"id": "cand-2", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-b"},
    ]
    state = {"cand-2": {"useful_count": 3, "rejected_count": 1}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["caution_mixed_signal"] == 1
    entry = payload["buckets"]["caution_mixed_signal"][0]
    assert entry["rejected_count"] == 1


def test_insufficient_evidence_when_score_not_positive_and_not_manual():
    candidates = [
        {"id": "cand-3", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-c"},
    ]
    state = {"cand-3": {}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["insufficient_evidence"] == 1
    entry = payload["buckets"]["insufficient_evidence"][0]
    assert entry["score"] == 0


def test_manual_candidate_is_ready_even_with_zero_score():
    candidates = [
        {"id": "cand-4", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "manual_input"},
    ]
    state = {"cand-4": {}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["ready_for_review"] == 1
    assert payload["buckets"]["ready_for_review"][0]["is_manual"] is True


def test_not_eligible_when_claim_too_short():
    candidates = [{"id": "cand-5", "claim": "短い", "research_topic": "topic-e"}]

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state={}, canonical={},
    )

    assert payload["summary"]["not_eligible"] == 1


def test_held_candidate_is_never_reported_as_ready_even_with_high_score():
    # 人間が「保留」を選んだ候補は、その後usefulが積まれてscoreが正になっても
    # 人間の判断を勝手に覆してready_for_reviewへ戻してはいけない。
    candidates = [
        {"id": "cand-held", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-held"},
    ]
    state = {"cand-held": {"promotion_status": "held", "useful_count": 10, "rejected_count": 0}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["held_by_human"] == 1
    assert payload["summary"]["ready_for_review"] == 0
    assert payload["buckets"]["held_by_human"][0]["id"] == "cand-held"


def test_duplicate_no_new_evidence_when_matches_active_statement_without_signal():
    candidates = [
        {"id": "cand-6", "claim": "既存の正規判断資産の文面です。", "research_topic": "topic-f"},
    ]

    payload = readiness.build_report(
        target_date="2026-08-20",
        candidates=candidates,
        state={},
        canonical=_canonical_with_active_statement(),
    )

    assert payload["summary"]["duplicate_no_new_evidence"] == 1


def test_duplicate_statement_with_new_evidence_is_still_evaluated_normally():
    candidates = [
        {"id": "cand-7", "claim": "既存の正規判断資産の文面です。", "research_topic": "topic-g"},
    ]
    state = {"cand-7": {"useful_count": 5}}

    payload = readiness.build_report(
        target_date="2026-08-20",
        candidates=candidates,
        state=state,
        canonical=_canonical_with_active_statement(),
    )

    assert payload["summary"]["duplicate_no_new_evidence"] == 0
    assert payload["summary"]["ready_for_review"] == 1


def test_already_promoted_and_already_rejected_are_separated():
    candidates = [
        {"id": "cand-8", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-h"},
        {"id": "cand-9", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-i"},
    ]
    state = {
        "cand-8": {"promotion_status": "active"},
        "cand-9": {"promotion_status": "rejected"},
    }

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )

    assert payload["summary"]["already_promoted"] == 1
    assert payload["summary"]["already_rejected"] == 1


def test_markdown_lists_criteria_and_candidate_reasons():
    candidates = [
        {"id": "cand-10", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "topic-j"},
    ]
    state = {"cand-10": {"useful_count": 1}}

    payload = readiness.build_report(
        target_date="2026-08-20", candidates=candidates, state=state, canonical={},
    )
    markdown = readiness.build_markdown(payload)

    assert "Judgment Asset Promotion Readiness" in markdown
    assert readiness.SCORE_FORMULA in markdown
    assert "cand-10" in markdown


def test_main_writes_json_and_markdown(tmp_path):
    candidates_jsonl = tmp_path / "candidates.jsonl"
    candidates_jsonl.write_text(
        json.dumps({"id": "cand-11", "claim": "十分に長い判断資産候補の文面です。", "research_topic": "t"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    state_json = tmp_path / "state.json"
    state_json.write_text(json.dumps({"cand-11": {"useful_count": 1}}, ensure_ascii=False), encoding="utf-8")
    canonical_json = tmp_path / "canonical.json"
    canonical_json.write_text(json.dumps({"rules": []}, ensure_ascii=False), encoding="utf-8")
    output_json = tmp_path / "out.json"
    output_md = tmp_path / "out.md"

    payload = readiness.build_report(
        target_date="2026-08-20",
        candidates=readiness._read_jsonl(candidates_jsonl),
        state=readiness._read_json(state_json),
        canonical=readiness._read_json(canonical_json),
    )
    readiness._write_json(output_json, payload)
    output_md.write_text(readiness.build_markdown(payload), encoding="utf-8")

    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["ready_for_review"] == 1
    assert "Judgment Asset Promotion Readiness" in output_md.read_text(encoding="utf-8")
