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


def test_extract_materials_rejects_meta_ops_and_chatter_without_lease_decision(tmp_path):
    vault = tmp_path / "vault"
    cloud_log = (
        vault
        / "Projects"
        / "tune_lease_55"
        / "AI Chat"
        / "Cloud Run Conversation Log"
        / "2026-07-25.md"
    )
    _write(
        cloud_log,
        (
            "## 10:00 next_chat_general\n\n"
            "### User\n"
            "判断資産グラフの画面が白紙だから修正して。\n"
            "犬の名前を覚えているかが、なぜAIへの信頼に関係するの。\n"
            "紫苑 頑張ってね 2次審査は通ったみたい。\n"
            "リース期間・残価判断では、物件の使用状況と換金性も確認する。\n\n"
            "### Assistant\n"
            "今日も一日、リース審査の判断資産を育むために共に歩んでいきましょう。\n"
            "特に日次改善パイプラインの異常は見落としがないよう注意深く監視します。\n"
            "厨房機器は耐用年数だけでなく、満了後の出口と再販可能性を見る必要があります。\n"
        ),
    )

    materials = preview.extract_materials(vault=vault, end_date=dt.date(2026, 7, 25), days=1)
    claims = "\n".join(item["claim"] for item in materials)

    assert "判断資産グラフ" not in claims
    assert "犬の名前" not in claims
    assert "頑張ってね" not in claims
    assert "今日も一日" not in claims
    assert "日次改善パイプライン" not in claims
    assert "物件の使用状況と換金性" in claims
    assert "厨房機器は耐用年数" in claims
    assert {item["domain"] for item in materials} == {"lease_screening"}


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
