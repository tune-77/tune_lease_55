#!/usr/bin/env python3
"""Execute the Codex auto-improvement queue by running claude for each safe item."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_status(root: Path, rev_id: str, status: str, detail: str = "") -> None:
    recorder = root / "scripts" / "record_codex_auto_status.py"
    if not recorder.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(recorder),
            rev_id,
            "--status", status,
            "--detail", detail,
            "--source", "execute_codex_queue",
        ],
        capture_output=True,
    )


def _get_gemini_api_key(root: Path) -> str:
    """環境変数 → .streamlit/secrets.toml の順で GEMINI_API_KEY を取得する."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    secrets_path = root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                if "GEMINI_API_KEY" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def _try_gemini(prompt: str, api_key: str) -> tuple[int, str, str]:
    """gemini-2.0-flash で prompt を実行し (exit_code, stdout, stderr) を返す."""
    try:
        import google.generativeai as genai  # type: ignore[import-untyped]
    except ImportError:
        return -1, "", "google-generativeai パッケージ未インストール"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        text = response.text or ""
        if not text.strip():
            return -1, "", "Gemini returned empty response"
        return 0, text, ""
    except Exception as exc:
        return -1, "", f"Gemini error: {exc}"


def run_item(item: dict[str, Any], gemini_api_key: str = "") -> dict[str, Any]:
    prompt = str(item.get("prompt") or "")
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    # 1. claude --print を試みる
    try:
        proc = subprocess.run(
            ["claude", "--print", prompt],
            capture_output=True,
            text=True,
            timeout=300,
        )
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        exit_code = -1
        stdout = ""
        stderr = "timeout after 300s"
    except Exception as exc:
        exit_code = -1
        stdout = ""
        stderr = str(exc)

    backend = "claude"

    # 2. claude 失敗時は Gemini にフォールバック
    if exit_code != 0:
        if gemini_api_key:
            claude_stderr = stderr
            gem_exit, gem_stdout, gem_stderr = _try_gemini(prompt, gemini_api_key)
            if gem_exit == 0:
                exit_code = 0
                stdout = gem_stdout
                stderr = ""
                backend = "gemini"
            else:
                stderr = f"claude: {claude_stderr} | gemini: {gem_stderr}"
                backend = "none"
        else:
            backend = "none"

    return {
        "id": str(item.get("id") or ""),
        "title": item.get("title"),
        "exit_code": exit_code,
        "stdout": stdout[:4000],
        "stderr": stderr[:2000],
        "backend": backend,
        "started_at": started_at,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True, help="Queue JSON (codex_auto_queue_YYYYMMDD.json)")
    parser.add_argument("--output", type=Path, default=None, help="Result JSON path (default: reports/codex_queue_result_YYYYMMDD.json)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    queue = load_json(args.queue)
    items: list[dict[str, Any]] = [i for i in (queue.get("items") or []) if isinstance(i, dict)]
    if not items:
        print("Queue is empty, nothing to execute.")
        return

    date_tag = dt.date.today().strftime("%Y%m%d")
    output_path = args.output or root / "reports" / f"codex_queue_result_{date_tag}.json"

    gemini_api_key = _get_gemini_api_key(root)
    if gemini_api_key:
        print("[execute_codex_queue] Gemini fallback: enabled (gemini-2.5-flash)")
    else:
        print("[execute_codex_queue] Gemini fallback: disabled (GEMINI_API_KEY not set)")

    results: list[dict[str, Any]] = []
    any_failure = False

    for item in items:
        rev_id = str(item.get("id") or "")
        print(f"[execute_codex_queue] {rev_id}: {item.get('title')}")

        if args.dry_run:
            print(f"  [dry-run] prompt: {item.get('prompt')}")
            results.append({"id": rev_id, "title": item.get("title"), "exit_code": 0, "dry_run": True})
            continue

        record_status(root, rev_id, "running")
        entry = run_item(item, gemini_api_key=gemini_api_key)
        results.append(entry)

        if entry["exit_code"] == 0:
            print(f"  -> OK (backend={entry['backend']})")
            record_status(root, rev_id, "completed_pending_review", detail=entry["stdout"][:200])
        else:
            print(f"  -> FAILED (backend={entry['backend']}, exit={entry['exit_code']}): {entry['stderr'][:100]}")
            record_status(root, rev_id, "failed", detail=entry["stderr"][:200])
            any_failure = True

    report: dict[str, Any] = {
        "executed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "queue_file": str(args.queue),
        "total": len(results),
        "succeeded": sum(1 for r in results if r.get("exit_code") == 0),
        "failed": sum(1 for r in results if r.get("exit_code") != 0),
        "results": results,
    }

    dump_json(output_path, report)
    print(f"Result written: {output_path}")

    if any_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
