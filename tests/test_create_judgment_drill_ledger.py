import csv
from datetime import date

from scripts import create_judgment_drill_ledger as ledger


def test_build_rows_creates_unique_pending_cases():
    rows = ledger.build_rows(1000, date(2026, 7, 25))

    assert len(rows) == 1000
    assert len({row["case_id"] for row in rows}) == 1000
    assert rows[0]["case_id"] == "CASE-JD-20260725-0001"
    assert rows[-1]["case_id"] == "CASE-JD-20260725-1000"
    assert {row["status"] for row in rows} == {"pending_user_judgment"}
    assert {row["source_kind"] for row in rows} == {"demo_user_judgment_drill"}
    assert {row["bank_is_main"] for row in rows} == {"メイン", "非メイン"}
    assert {row["lease_customer_status"] for row in rows} == {"既存先", "新規先"}


def test_user_judgment_fields_start_blank():
    row = ledger.build_rows(1, date(2026, 7, 25))[0]

    for field in [
        "credit_score_20",
        "repayment_source_score_20",
        "asset_exit_score_20",
        "plan_specificity_score_20",
        "uncertainty_control_score_20",
        "total_score_100",
        "user_decision",
        "heaviest_issue",
        "additional_checks",
        "ringi_sentence",
        "ai_feedback_outcome",
        "ai_feedback_note",
        "reviewer",
        "judged_at",
    ]:
        assert row[field] == ""


def test_new_customer_has_zero_existing_lease_credit_balance():
    rows = ledger.build_rows(20, date(2026, 7, 25))

    for row in rows:
        if row["lease_customer_status"] == "新規先":
            assert row["existing_lease_credit_balance_million_yen"] == "0"
        assert row["equity_ratio_percent"]
        assert row["bank_credit_balance_million_yen"]


def test_write_csv_uses_declared_field_order(tmp_path):
    rows = ledger.build_rows(3, date(2026, 7, 25))
    path = tmp_path / "ledger.csv"

    ledger.write_csv(path, rows)

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)

    assert reader.fieldnames == ledger.FIELDNAMES
    assert [row["case_id"] for row in loaded] == [
        "CASE-JD-20260725-0001",
        "CASE-JD-20260725-0002",
        "CASE-JD-20260725-0003",
    ]
