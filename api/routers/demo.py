"""デモパイプライン用エンドポイント（REV-141）"""
import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_DIR = _REPO_ROOT / "scripts" / "demo"
_LOG_FILE = _DEMO_DIR / "demo_agent_log.jsonl"
_SHELL_SCRIPT = _DEMO_DIR / "run_demo_pipeline.sh"


@router.get("/demo/agent-log")
def get_agent_log(since: int = Query(0, ge=0)) -> List[dict]:
    """demo_agent_log.jsonl の since 行目以降を返す。"""
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    result = []
    for line in lines[since:]:
        line = line.strip()
        if not line:
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


@router.get("/demo/apply-summary")
def get_apply_summary() -> dict:
    """demo_apply_summary.json を返す。"""
    summary_file = _DEMO_DIR / "demo_apply_summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="demo_apply_summary.json が存在しません")
    import json as _json
    return _json.loads(summary_file.read_text(encoding="utf-8"))


@router.post("/demo/run")
def run_demo_pipeline() -> dict:
    """demo_agent_log.jsonl をリセットしてデモパイプラインをバックグラウンド実行する。"""
    if not _SHELL_SCRIPT.exists():
        raise HTTPException(status_code=404, detail="run_demo_pipeline.sh が見つかりません")

    # ログファイルをリセット
    _LOG_FILE.write_text("", encoding="utf-8")

    # バックグラウンド実行
    subprocess.Popen(
        ["bash", str(_SHELL_SCRIPT)],
        cwd=str(_REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"status": "started"}
