"""requirements-pipeline.txt が akshare を宣言していることを保証する（REV-034a 再発防止）。

fetch_fincept_data.py は健全スキップ（exit 0）するため akshare 未導入でもパイプラインは
止まらないが、マクロ更新自体が止まったままになる「サイレント劣化」を防ぐには、
日次パイプライン用 venv に akshare を再現可能にインストールできる宣言が要る。
requirements.txt（Cloud Run本番）には混ぜない（fetch_fincept_data は Mac日次専用）。
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def test_requirements_pipeline_file_declares_akshare():
    p = _REPO / "requirements-pipeline.txt"
    assert p.exists(), "requirements-pipeline.txt が存在しない"
    text = p.read_text(encoding="utf-8")
    assert "akshare" in text.lower()


def test_akshare_not_mixed_into_production_requirements():
    # Cloud Run 本番（Dockerfile / Dockerfile.api が読む）requirements.txt には
    # 混ぜない。fetch_fincept_data.py は run_daily_improvement_core.sh 専用。
    text = (_REPO / "requirements.txt").read_text(encoding="utf-8")
    assert "akshare" not in text.lower()
