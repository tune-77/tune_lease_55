"""pipeline_runner.py の終了コード規約のテスト（REV-024a 再発防止）。

「適用するものがなかった」状態を失敗扱いすると analyze_pipeline_health が
障害として誤検出する。FAILED だけが exit 1 になることを保証する。
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / ".agents" / "skills" / "auto-improvement-pipeline" / "pipeline_runner.py"

_spec = importlib.util.spec_from_file_location("pipeline_runner", RUNNER_PATH)
pipeline_runner = importlib.util.module_from_spec(_spec)
sys.modules["pipeline_runner"] = pipeline_runner
_spec.loader.exec_module(pipeline_runner)


def test_no_work_statuses_are_success():
    for status in ("COMPLETED", "NO_APPLIED", "DRY_RUN_COMPLETE", "NO_IMPROVEMENTS", "NO_NEW_IMPROVEMENTS"):
        assert status in pipeline_runner.SUCCESS_STATUSES, status


def test_failed_is_not_success():
    assert "FAILED" not in pipeline_runner.SUCCESS_STATUSES


def test_cli_exits_zero_when_no_improvements_found(tmp_path):
    """改善タグのないログを渡した実行が exit 0 で終わること（E2E）。"""
    chat_log = tmp_path / "empty_chat.txt"
    chat_log.write_text("今日は雑談だけでした。改善要望はありません。\n", encoding="utf-8")
    output = tmp_path / "result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER_PATH),
            str(chat_log),
            "--dry-run",
            "--output",
            str(output),
            "--workspace",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout[-800:]}\nstderr={proc.stderr[-800:]}"
