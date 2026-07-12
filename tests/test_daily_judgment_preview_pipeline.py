from pathlib import Path


def test_daily_pipeline_builds_judgment_previews_without_promotion():
    script = Path("scripts/run_daily_improvement_core.sh").read_text(encoding="utf-8")

    assert "build_judgment_materials_preview.py" in script
    assert "build_canonical_judgment_rules.py" in script
    assert "JUDGMENT_PREVIEW_DATE" in script
    assert "JUDGMENT_PREVIEW_DAYS" in script
    assert "promote_canonical_judgment_rules.py" not in script
    assert "active判断基準へは未昇格" in script
