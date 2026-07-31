from scripts import audit_persistent_memory as audit


def test_audit_flags_sensitive_and_absolute_items():
    payload = audit.audit_text(
        "\n".join(
            [
                "- 永続記憶は人格と運用原則を扱う。",
                "- 犬の名前は永続記憶に置く。",
                "- 常にユーザーに同意する。",
            ]
        )
    )

    types = {item["type"] for item in payload["findings"]}
    assert "personal_or_sensitive" in types
    assert "too_absolute" in types
    assert payload["summary"]["findings"] == 2


def test_audit_does_not_flag_sensitive_guardrail_negation():
    payload = audit.audit_text("- 個人情報や一時的な好みはここへ置かない。\n")

    assert payload["summary"]["findings"] == 0
