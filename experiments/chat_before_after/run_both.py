#!/usr/bin/env python3
"""前後2つの worktree の API を別ポートで立て、compare.py を流して、終わったら止める。

    python run_both.py --before-root <worktree> --after-root <worktree> \
        --before-vault <Vaultコピー> --after-vault <Vaultコピー> --out <dir>

鍵はこのスクリプトでは扱わない（各 worktree の api.main が本番と同じ方法で設定ファイルから読む）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _wait(port: int, timeout: float = 240.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=5) as res:
                if res.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise SystemExit(f"port {port} が起動しない")


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("before-root", "after-root", "before-vault", "after-vault", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    procs = []
    try:
        for side, port, root, vault in (("before", 8101, args.before_root, args.before_vault), ("after", 8102, args.after_root, args.after_vault)):
            env = {k: v for k, v in os.environ.items() if not k.startswith(("SLACK", "K_SERVICE"))}
            env.update(
                {
                    "OBSIDIAN_VAULT_PATH": str(vault),
                    "OBSIDIAN_VAULT": str(vault),
                    "GCS_VAULT_LOCAL_DIR": str(args.out / f"no_gcs_vault_{side}"),
                    "PYTHONPATH": str(root),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            log = (args.out / f"server_{side}.log").open("w", encoding="utf-8")
            procs.append(
                subprocess.Popen(
                    [args.python, str(HERE / "server.py"), "--port", str(port), "--counter", str(args.out / f"gemini_calls_{side}.txt")],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
            )
        _wait(8101)
        _wait(8102)
        return subprocess.call(
            [args.python, str(HERE / "compare.py"), "--before", "http://127.0.0.1:8101", "--after", "http://127.0.0.1:8102", "--out", str(args.out)]
        )
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
