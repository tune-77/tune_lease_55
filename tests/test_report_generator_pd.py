from __future__ import annotations

from report_generator import generate_full_report_from_res, generate_reference_info


def _session_state() -> dict:
    return {
        "rep_company": "テスト製作所",
        "last_submitted_inputs": {
            "nenshu": 1000,
            "rieki": 20,
            "acquisition_cost": 100,
            "lease_term": 60,
            "bank_credit": 200,
        },
    }


def test_report_does_not_fabricate_pd_from_score_when_pd_missing():
    report = generate_full_report_from_res(
        {"score": 56.5, "industry_major": "E 製造業"},
        _session_state(),
    )

    assert "デフォルト確率（PD）: 未算出" in report
    assert "スコアからの推定表示は行いません" in report
    assert "8.00%" not in report
    assert "PDは未算出です" in report


def test_report_uses_pd_percent_only_when_present():
    report = generate_full_report_from_res(
        {"score": 56.5, "industry_major": "E 製造業", "pd_percent": 3.25},
        _session_state(),
    )

    assert "デフォルト確率（PD）: 3.25%" in report
    assert "未算出" not in report


def test_reference_info_empty_when_no_hits(monkeypatch):
    from mobile_app import obsidian_bridge

    monkeypatch.setattr(obsidian_bridge, "search_notes", lambda *a, **k: [])

    assert generate_reference_info("製造業") == ""


def test_reference_info_empty_when_search_fails(monkeypatch):
    from mobile_app import obsidian_bridge

    def _raise(*a, **k):
        raise RuntimeError("vault unavailable")

    monkeypatch.setattr(obsidian_bridge, "search_notes", _raise)

    assert generate_reference_info("製造業") == ""


def test_reference_info_lists_hits_without_touching_score(monkeypatch):
    from mobile_app import obsidian_bridge

    monkeypatch.setattr(
        obsidian_bridge,
        "search_notes",
        lambda *a, **k: [
            {"path": "Projects/tune_lease_55/Research/補助金メモ.md", "snippet": "ものづくり補助金の対象要件"},
            {"path": "Projects/tune_lease_55/Research/規制メモ.md", "snippet": ""},
        ],
    )

    info = generate_reference_info("製造業")

    assert "【参考情報：補助金・規制】" in info
    assert "スコア・承認判定には反映していません" in info
    assert "補助金メモ.md: ものづくり補助金の対象要件" in info
    assert "規制メモ.md" in info


def test_full_report_includes_reference_info_when_available(monkeypatch):
    from mobile_app import obsidian_bridge

    monkeypatch.setattr(
        obsidian_bridge,
        "search_notes",
        lambda *a, **k: [{"path": "Projects/tune_lease_55/Research/補助金メモ.md", "snippet": "対象要件"}],
    )

    report = generate_full_report_from_res(
        {"score": 56.5, "industry_major": "E 製造業"},
        _session_state(),
    )

    assert "【参考情報：補助金・規制】" in report
    assert "補助金メモ.md" in report
