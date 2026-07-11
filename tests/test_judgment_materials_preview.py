import datetime as dt
from pathlib import Path

from scripts import build_judgment_materials_preview as preview


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_extract_materials_from_cloudrun_and_dialogue_notes(tmp_path):
    vault = tmp_path / "vault"
    cloud_log = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "AI Chat"
        / "Cloud Run Conversation Log"
        / "2026-07-12.md"
    )
    dialogue = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Dialogue"
        / "2026-07-12.md"
    )
    _write(
        cloud_log,
        (
            "## 10:00 next_chat_general\n\n"
            "### User\n"
            "ラーメン屋の厨房機器はリース期間5年が多い。覚えておいて。\n\n"
            "### Assistant\n"
            "飲食業は廃業リスクがあるため、事業計画と資金繰りを確認する必要があります。\n"
        ),
    )
    _write(
        dialogue,
        (
            "## 11:00\n\n"
            "**ユーザー**\n\n"
            "リースに必要なものはスピードだ。具体的な事業計画が出来ない契約は通りづらい。\n\n"
            "**リース知性体**\n\n"
            "銀行支援は本件リースへの直接支援か確認します。\n"
        ),
    )

    materials = preview.extract_materials(vault=vault, end_date=dt.date(2026, 7, 12), days=1)
    claims = "\n".join(item["claim"] for item in materials)
    types = {item["material_type"] for item in materials}

    assert "ラーメン屋の厨房機器はリース期間5年が多い" in claims
    assert "具体的な事業計画が出来ない契約は通りづらい" in claims
    assert "リースに必要なものはスピードだ" in claims
    assert "judgment_rule" in types
    assert "risk_signal" in types
    assert "user_preference" in types
    assert all(item["preview"] is True for item in materials)
    assert all(item["private"] is False for item in materials)


def test_markdown_declares_preview_only(tmp_path):
    materials = [
        {
            "date": "2026-07-12",
            "material_type": "judgment_rule",
            "confidence": 0.8,
            "claim": "銀行支援は本件リースへの直接支援か確認する。",
            "use_when": "外部支援を返済原資や保全材料として扱うとき",
            "risk_axis": ["support_specificity"],
            "evidence_path": "Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-12.md",
        }
    ]

    md = preview._markdown(materials, end_date=dt.date(2026, 7, 12), days=1)

    assert "Preview only" in md
    assert "Not connected to RAG" in md
    assert "Private Reflection is intentionally excluded" in md
