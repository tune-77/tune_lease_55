from pathlib import Path

from scripts.check_frontend_backend_schema import (
    DEFAULT_SCHEMAS_PATH,
    DEFAULT_TYPES_PATH,
    DEFAULT_UNITS_PATH,
    check_field_parity,
    check_monetary_unit_conversion,
    parse_pydantic_model_fields,
    parse_ts_interface_fields,
    parse_ts_string_array,
    run_checks,
)


def test_real_repo_schemas_are_currently_in_sync():
    report = run_checks(DEFAULT_SCHEMAS_PATH, DEFAULT_TYPES_PATH, DEFAULT_UNITS_PATH)

    assert report["status"] == "ok", report["findings"]


def test_field_missing_from_frontend_is_an_error():
    request_fields = {"nenshu": "float", "new_backend_field": "str"}
    form_fields = {"nenshu": "number"}

    findings = check_field_parity(request_fields, form_fields)

    assert any(
        f.check == "missing_in_frontend" and "new_backend_field" in f.message
        for f in findings
    )


def test_field_missing_from_backend_is_an_error_unless_allowlisted():
    request_fields = {"nenshu": "float"}
    form_fields = {"nenshu": "number", "orphaned_ui_field": "string"}

    findings = check_field_parity(request_fields, form_fields)

    assert any(
        f.check == "missing_in_backend" and "orphaned_ui_field" in f.message
        for f in findings
    )


def test_known_frontend_only_field_does_not_raise_error():
    request_fields = {"nenshu": "float"}
    form_fields = {"nenshu": "number", "strength_tags": "string[]"}

    findings = check_field_parity(request_fields, form_fields)

    assert not any(f.level == "error" for f in findings)


def test_new_monetary_float_field_missing_from_conversion_list_is_an_error():
    request_fields = {"new_amount": "float"}
    form_fields = {"new_amount": "number"}
    monetary_fields: list[str] = []

    findings = check_monetary_unit_conversion(request_fields, form_fields, monetary_fields)

    assert any(
        f.check == "unit_conversion_missing" and "new_amount" in f.message
        for f in findings
    )


def test_known_non_monetary_float_field_does_not_raise_error():
    request_fields = {"competitor_rate": "Optional[float]"}
    form_fields = {"competitor_rate": "number"}
    monetary_fields: list[str] = []

    findings = check_monetary_unit_conversion(request_fields, form_fields, monetary_fields)

    assert not any(f.level == "error" for f in findings)


def test_parse_pydantic_model_fields(tmp_path: Path):
    schemas_file = tmp_path / "schemas.py"
    schemas_file.write_text(
        "from pydantic import BaseModel\n"
        "class Example(BaseModel):\n"
        "    amount: float\n"
        "    label: str\n",
        encoding="utf-8",
    )

    fields = parse_pydantic_model_fields(schemas_file, "Example")

    assert fields == {"amount": "float", "label": "str"}


def test_parse_ts_interface_fields(tmp_path: Path):
    types_file = tmp_path / "index.ts"
    types_file.write_text(
        "export interface Example {\n"
        "  amount: number; // comment\n"
        "  label?: string;\n"
        "}\n",
        encoding="utf-8",
    )

    fields = parse_ts_interface_fields(types_file, "Example")

    assert fields == {"amount": "number", "label": "string"}


def test_parse_ts_string_array(tmp_path: Path):
    units_file = tmp_path / "scoringUnits.ts"
    units_file.write_text(
        'export const monetaryScoringFields = [\n  "amount",\n  "rent",\n] as const;\n',
        encoding="utf-8",
    )

    values = parse_ts_string_array(units_file, "monetaryScoringFields")

    assert values == ["amount", "rent"]
