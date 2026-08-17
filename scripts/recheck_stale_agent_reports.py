#!/usr/bin/env python3
"""Re-run the agents whose sidecar reports have gone stale (GAP-003 recheck queue).

`.claude/reports/` は書き手（エージェント）が居ないと更新されない。日次
パイプラインが回していたのは蒸留役の `agent_sidecar_reader.py` だけで、
エージェント本体を再実行する経路が無かったため、全レポートが数ヶ月単位で
STALE になっていた。

このスクリプトは古い順に少数だけを選んで再実行する「再確認キュー」であり、
毎日14体を回すものではない。既定の1日3件なら約5日で全周し、
`agent_sidecar_reader.STALE_AFTER_DAYS`（30日）を余裕で下回る。

安全弁は `scripts/execute_codex_queue.py` と同じ形を踏襲する
（キルスイッチ・日次上限・連続失敗停止）。新しい仕組みは足さない。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_sidecar_reader import STALE_AFTER_DAYS, _is_stale, _parse_frontmatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / ".claude" / "agents"
REPORT_ROOT = PROJECT_ROOT / ".claude" / "reports"
RESULT_DIR = PROJECT_ROOT / "reports"

# エージェント名とレポートディレクトリ名は一致しない（scoring-auditor →
# scoring-audit 等）。正は .claude/AGENTS.md の読み書き表。
AGENT_REPORT_DIRS: dict[str, str] = {
    "file-searcher": "file-searcher",
    "change-impact-analyzer": "impact-analysis",
    "code-reviewer": "code-review",
    "security-checker": "security",
    "build-runner": "build",
    "test-runner": "test-results",
    "test-result-analyzer": "test-results",
    "log-file-analyzer": "log-analysis",
    "scoring-auditor": "scoring-audit",
    "data-quality-checker": "data-quality",
    "rule-validator": "rule-validation",
    "api-health-checker": "api-health",
    "report-stylist": "report-stylist",
    "migration-validator": "migration",
}

# 再実行しないエージェント。上流レポートを前提にする協調役は単独で回しても
# 意味が無いか、他エージェントの出力を上書きしてしまう。
SKIP_AGENTS = frozenset({"test-result-analyzer"})

CLAUDE_TIMEOUT_SEC = 900


# ── 自律実行ガード（execute_codex_queue.py と同じ規約）──────────────────────


def _guard_disabled() -> bool:
    return str(os.environ.get("AGENT_RECHECK_DISABLED") or "").strip() in {"1", "true", "yes"}


def _guard_daily_limit() -> int:
    try:
        return max(0, int(os.environ.get("AGENT_RECHECK_DAILY_LIMIT", "3")))
    except ValueError:
        return 3


def _guard_max_consecutive_failures() -> int:
    try:
        return max(1, int(os.environ.get("AGENT_RECHECK_MAX_CONSECUTIVE_FAILURES", "2")))
    except ValueError:
        return 2


def _permission_mode() -> str:
    """非対話実行の権限モード。

    `claude --print` には許可を尋ねる相手が居ないため、既定のままでは
    レポートの書き込みが通らない。`acceptEdits` はファイル編集を自動承認する。
    Bash を多用するエージェントが止まる場合は `dontAsk` を検討する
    （`bypassPermissions` は全チェックを外すので既定にはしない）。
    """
    mode = str(os.environ.get("AGENT_RECHECK_PERMISSION_MODE") or "").strip()
    allowed = {"acceptEdits", "auto", "dontAsk", "bypassPermissions", "plan", "manual"}
    return mode if mode in allowed else "acceptEdits"


def build_command(candidate: dict[str, Any], prompt: str) -> list[str]:
    """`claude` の起動コマンド。

    エージェント選択はプロンプト頼みではなく `--agent` で明示する。
    """
    return [
        "claude",
        "--print",
        "--agent",
        str(candidate["agent"]),
        "--permission-mode",
        _permission_mode(),
        prompt,
    ]


def count_executed_today(date_tag: str, result_dir: Path | None = None) -> int:
    """今日すでに再実行した件数（dry-run は除く）。"""
    root = result_dir if result_dir is not None else RESULT_DIR
    executed = 0
    for path in root.glob(f"agent_recheck_result_{date_tag}*.json"):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entry in report.get("results") or []:
            if isinstance(entry, dict) and not entry.get("dry_run"):
                executed += 1
    return executed


# ── 対象選定 ────────────────────────────────────────────────────────────────


def defined_agents(agent_dir: Path | None = None) -> list[str]:
    """`.claude/agents/` に定義されたエージェント名。定義が正。"""
    root = agent_dir if agent_dir is not None else AGENT_DIR
    return sorted(p.stem for p in root.glob("*.md"))


def _report_timestamp(report_path: Path) -> str:
    try:
        meta, _ = _parse_frontmatter(report_path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""
    return meta.get("timestamp", "")


def select_stale_agents(
    limit: int,
    agent_dir: Path | None = None,
    report_root: Path | None = None,
) -> list[dict[str, Any]]:
    """再確認が必要なエージェントを古い順に最大 limit 件返す。

    レポートが存在しないものを最優先（一度も走っていない）、次に古い順。
    """
    reports = report_root if report_root is not None else REPORT_ROOT
    candidates: list[dict[str, Any]] = []

    for name in defined_agents(agent_dir):
        if name in SKIP_AGENTS:
            continue
        report_dir = AGENT_REPORT_DIRS.get(name)
        if report_dir is None:
            # 定義が増えたのに対応表が追随していない。黙って落とさず表に出す。
            candidates.append(
                {"agent": name, "report": "", "timestamp": "", "reason": "unmapped"}
            )
            continue
        path = reports / report_dir / "latest.md"
        if not path.exists():
            candidates.append(
                {
                    "agent": name,
                    "report": f".claude/reports/{report_dir}/latest.md",
                    "timestamp": "",
                    "reason": "missing",
                }
            )
            continue
        timestamp = _report_timestamp(path)
        if _is_stale(timestamp):
            candidates.append(
                {
                    "agent": name,
                    "report": f".claude/reports/{report_dir}/latest.md",
                    "timestamp": timestamp,
                    "reason": "stale",
                }
            )

    # timestamp 空（missing / unmapped / 日付不明）を先頭へ、以降は古い順。
    candidates.sort(key=lambda c: (c["timestamp"] != "", c["timestamp"]))
    return candidates[:limit]


def build_prompt(candidate: dict[str, Any]) -> str:
    agent = candidate["agent"]
    report = candidate["report"] or f".claude/reports/<{agent}>/latest.md"
    # エージェント自体は `--agent` で選択済み（build_command 参照）。
    return (
        f"あなたの担当範囲（{agent}）を、現在のコードに対して再監査してください。\n\n"
        f"結果は `{report}` に上書き保存してください。書式は "
        "`.claude/reports/REPORT_SCHEMA.md` に従い、frontmatter の "
        f"timestamp には実行時刻（{dt.datetime.now():%Y-%m-%d %H:%M}）を入れてください。\n\n"
        "制約:\n"
        "- 調査とレポート作成のみ。アプリのコード・DB・モデルは変更しないこと。\n"
        "- 前回のレポートは古い可能性がある。現在のコードで検証し直すこと。\n"
        "- 事実に基づかない指摘を書かないこと。確認できない場合はその旨を書くこと。\n"
    )


# ── 実行 ────────────────────────────────────────────────────────────────────


def run_candidate(candidate: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "agent": candidate["agent"],
        "report": candidate["report"],
        "previous_timestamp": candidate["timestamp"],
        "reason": candidate["reason"],
        "started_at": started_at,
        "dry_run": dry_run,
    }
    if dry_run:
        result.update({"exit_code": 0, "stdout": "", "stderr": ""})
        return result

    prompt = build_prompt(candidate)
    try:
        proc = subprocess.run(
            build_command(candidate, prompt),
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT_SEC,
        )
        exit_code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        exit_code, stdout, stderr = -1, "", f"timeout after {CLAUDE_TIMEOUT_SEC}s"
    except Exception as exc:  # noqa: BLE001 - 失敗理由を結果に残す
        exit_code, stdout, stderr = -1, "", str(exc)

    # 正常終了でも出力が空なら失敗扱い（execute_codex_queue.py と同じ判定）。
    if exit_code == 0 and not stdout.strip():
        exit_code = -1
        stderr = stderr or "claude returned empty output"

    result.update(
        {
            "exit_code": exit_code,
            "stdout": stdout[:4000],
            "stderr": stderr[:2000],
            "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="今回の再実行上限（既定: 環境変数）")
    parser.add_argument("--dry-run", action="store_true", help="選定だけ行い claude を起動しない")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if _guard_disabled():
        print("[guard] AGENT_RECHECK_DISABLED が設定されているため、再実行を停止します（キルスイッチ）")
        return 0

    date_tag = dt.datetime.now().strftime("%Y%m%d")
    daily_limit = _guard_daily_limit()
    if daily_limit == 0:
        print("[guard] AGENT_RECHECK_DAILY_LIMIT=0 のため、再実行しません")
        return 0

    already = count_executed_today(date_tag)
    remaining = max(0, daily_limit - already)
    if remaining == 0:
        print(f"[guard] 本日の上限に到達済み（{already}/{daily_limit}）。再実行しません")
        return 0

    limit = remaining if args.limit is None else max(0, min(args.limit, remaining))
    candidates = select_stale_agents(limit)
    if not candidates:
        print(f"[recheck] STALE({STALE_AFTER_DAYS}日超)なレポートはありません。再実行不要")
        return 0

    print(f"[recheck] 対象 {len(candidates)} 件（本日 {already}/{daily_limit} 実行済み）")

    results: list[dict[str, Any]] = []
    consecutive_failures = 0
    max_failures = _guard_max_consecutive_failures()
    for candidate in candidates:
        print(f"  - {candidate['agent']} ({candidate['reason']}, 前回: {candidate['timestamp'] or '不明'})")
        result = run_candidate(candidate, dry_run=args.dry_run)
        results.append(result)
        if result["exit_code"] == 0:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"    失敗: {result['stderr'][:200]}")
            if consecutive_failures >= max_failures:
                print(f"[guard] 連続{consecutive_failures}件失敗のため中断します")
                break

    out = args.output or (RESULT_DIR / f"agent_recheck_result_{date_tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "daily_limit": daily_limit,
                "already_executed_today": already,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    ok = sum(1 for r in results if r["exit_code"] == 0)
    print(f"[recheck] 完了: 成功 {ok}/{len(results)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
