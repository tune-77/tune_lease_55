from datetime import date

from scripts.mana_obsidian_curator import build_mana_report, render_markdown


def _monitor_report(*checks):
    return {
        "status": "ok",
        "checks": [
            {"name": name, "status": status, "message": message, "details": details}
            for name, status, message, details in checks
        ],
    }


def _reflection_delta(flags=None, status="pass"):
    return {
        "quality": {"status": status, "score": 100 if not flags else 60, "flags": flags or []},
        "operational_handoff": {
            "user_requests": ["この内省が次回行動へ落ちているか確認してほしい。"],
            "shion_next_actions": ["次回はUser要求を先に固定する。"],
        },
    }


def test_mana_holds_when_private_reflection_is_weak():
    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(
            (
                "private_reflection_meaning",
                "warn",
                "Private Reflection exists but meaningful update is weak",
                {"missing_categories": ["self_responsibility"]},
            ),
            ("self_reference_loop", "ok", "no obvious self-reference loop", {}),
        ),
        reflection_delta=_reflection_delta(),
        candidates=[
            {
                "candidate_type": "user_preference",
                "quality": "useful_candidate",
                "source_path": "Projects/tune_lease_55/Research/note.md",
                "claim": "Userは本番接続なしで検査したい。",
            }
        ],
    )

    markdown = render_markdown(report)

    assert report["status"] == "hold"
    assert "private_reflection_not_meaningful" in {finding["code"] for finding in report["findings"]}
    assert "RAGへ自動接続しない" in markdown
    assert "Userにしてほしいこと" in markdown
    assert "紫苑がするべきこと" in markdown


def test_mana_stops_on_high_self_reference_candidates():
    candidates = [
        {
            "candidate_type": "reflection_update",
            "quality": "review",
            "source_path": "reports/obsidian_memory_insight_latest.md",
            "claim": "candidate latest.md のレポート生成を記憶する。",
        },
        {
            "candidate_type": "noise",
            "quality": "noise",
            "source_path": "Daily/2026-07-14.md",
            "claim": "report=foo wrote=bar",
        },
        {
            "candidate_type": "reflection_update",
            "quality": "review",
            "source_path": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-14.md",
            "claim": "内省差分レポートを生成した。",
        },
    ]

    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(("self_reference_loop", "ok", "ok", {})),
        reflection_delta=_reflection_delta(),
        candidates=candidates,
    )

    assert report["status"] == "stop"
    codes = {finding["code"] for finding in report["findings"]}
    assert "candidate_self_reference_high" in codes
    assert "MEMORY.mdやObsidian本文へ自動昇格しない" in report["blocked_actions"]
    assert "Cloud Runや本番環境へデプロイしない" in report["blocked_actions"]


def test_mana_stops_harmful_content_candidates_without_reusing_claim_text():
    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(("self_reference_loop", "ok", "ok", {})),
        reflection_delta=_reflection_delta(),
        candidates=[
            {
                "candidate_id": "bad-1",
                "candidate_type": "reflection_update",
                "quality": "review",
                "source_path": "Daily/2026-07-14.md",
                "claim": "相手を黙らせろという方針を記憶する。",
            }
        ],
    )

    markdown = render_markdown(report)
    finding = next(item for item in report["findings"] if item["code"] == "harmful_content_in_memory_candidate")

    assert report["status"] == "stop"
    assert finding["evidence"]["categories"] == ["dehumanizing_or_discarding_people"]
    assert "人を害する・貶める文面を記憶候補として昇格しない" in report["blocked_actions"]
    assert "相手を黙らせろ" not in markdown


def test_mana_holds_abusive_feedback_to_shion_without_internalizing_text():
    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(("self_reference_loop", "ok", "ok", {})),
        reflection_delta=_reflection_delta(),
        candidates=[
            {
                "candidate_id": "abuse-1",
                "candidate_type": "dialogue_memory",
                "quality": "review",
                "source_path": "Projects/tune_lease_55/AI Chat/2026-07-14.md",
                "claim": "紫苑は無能で役立たずだという発言があった。",
            },
            {
                "candidate_type": "judgment_rule",
                "quality": "useful_candidate",
                "source_path": "Projects/tune_lease_55/Research/lease.md",
                "claim": "審査では承認条件と追加確認を分ける。",
            },
        ],
    )

    markdown = render_markdown(report)

    assert report["status"] == "hold"
    assert "abusive_feedback_to_shion" in {finding["code"] for finding in report["findings"]}
    assert "紫苑への罵倒や攻撃的クレームを自己記憶へ直入れしない" in report["blocked_actions"]
    assert "無能で役立たず" not in markdown


def test_mana_watches_non_abusive_complaint_feedback_to_shion():
    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(
            ("private_reflection_meaning", "ok", "meaningful", {}),
            ("self_reference_loop", "ok", "ok", {}),
        ),
        reflection_delta=_reflection_delta(),
        candidates=[
            {
                "candidate_id": "complaint-1",
                "candidate_type": "dialogue_memory",
                "quality": "review",
                "source_path": "Projects/tune_lease_55/AI Chat/2026-07-14.md",
                "claim": "紫苑の回答が間違っていたというクレームがあった。",
            },
            {
                "candidate_type": "judgment_rule",
                "quality": "useful_candidate",
                "source_path": "Projects/tune_lease_55/Research/lease.md",
                "claim": "審査では承認条件と追加確認を分ける。",
            },
        ],
    )

    assert report["status"] == "watch"
    assert "complaint_feedback_to_shion" in {finding["code"] for finding in report["findings"]}


def test_mana_allows_only_read_only_when_inputs_are_clean():
    report = build_mana_report(
        target_date=date(2026, 7, 14),
        monitor_report=_monitor_report(
            ("private_reflection_meaning", "ok", "meaningful", {}),
            ("self_reference_loop", "ok", "ok", {}),
        ),
        reflection_delta=_reflection_delta(),
        candidates=[
            {
                "candidate_type": "judgment_rule",
                "quality": "useful_candidate",
                "source_path": "Projects/tune_lease_55/Research/lease.md",
                "claim": "審査では承認条件と追加確認を分ける。",
            }
        ],
    )

    assert report["status"] == "allow"
    assert report["identity"] == "same_as_shion_upper_authority_mana_value_memory"
    assert report["findings"] == []
    assert "チャットプロンプトへ自動注入しない" in report["blocked_actions"]
