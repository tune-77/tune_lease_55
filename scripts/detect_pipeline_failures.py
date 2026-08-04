#!/usr/bin/env python3
"""
パイプライン失敗検出とリカバリー管理
改善パイプラインの健全性を監視し、失敗パターンを記録・分析する
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_log_dir() -> Path:
    log_dir = Path.home() / "Library" / "Logs" / "tunelease"
    return log_dir


def get_pipeline_health_db() -> Path:
    log_dir = get_log_dir()
    return log_dir / "pipeline_health.jsonl"


def load_health_entries(db_path: Path) -> list[dict]:
    """健全性レコードを読み込む"""
    entries = []
    if not db_path.exists():
        return entries

    try:
        for line in db_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except (OSError, IOError):
        pass

    return entries


def save_health_entry(db_path: Path, entry: dict) -> None:
    """健全性レコードを追記"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except (OSError, IOError) as e:
        print(f"警告: 健全性ログの書き込みに失敗: {e}", file=sys.stderr)


def parse_pipeline_log(log_path: Path) -> dict:
    """パイプラインログを解析"""
    if not log_path.exists():
        return {}

    result = {
        "log_path": str(log_path),
        "timestamp": log_path.stat().st_mtime,
        "failed_steps": [],
        "critical_errors": [],
        "warnings": [],
        "exit_code": None,
    }

    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, IOError) as e:
        result["parse_error"] = str(e)
        return result

    lines = content.splitlines()

    # Exit code を探す
    for line in reversed(lines):
        match = re.search(r"終了コード:\s*(\d+)", line)
        if match:
            result["exit_code"] = int(match.group(1))
            break

    # 失敗ステップを探す
    for line in lines:
        if "エラー:" in line or "ERROR:" in line:
            result["critical_errors"].append(line.strip())
        if "警告:" in line or "WARNING:" in line:
            result["warnings"].append(line.strip())
        if re.search(r"\[(post|core)\].*失敗", line):
            match = re.search(r"\[([^\]]+)\]", line)
            if match:
                result["failed_steps"].append(match.group(1))

    return result


def detect_failure_pattern(entries: list[dict], days: int = 7) -> dict:
    """失敗パターンを検出"""
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_timestamp = cutoff.timestamp()

    recent = [
        e for e in entries
        if e.get("timestamp", 0) >= cutoff_timestamp
    ]

    if not recent:
        return {"status": "no_data", "recent_runs": 0}

    failed = [e for e in recent if e.get("exit_code", 0) != 0]
    failure_rate = len(failed) / len(recent) if recent else 0

    # ステップごとの失敗カウント
    step_failures: dict[str, int] = {}
    for entry in failed:
        for step in entry.get("failed_steps", []):
            step_failures[step] = step_failures.get(step, 0) + 1

    return {
        "status": "detected" if failure_rate > 0.3 else "healthy",
        "recent_runs": len(recent),
        "failed_runs": len(failed),
        "failure_rate": round(failure_rate * 100, 1),
        "most_common_failures": sorted(
            step_failures.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5],
    }


def record_pipeline_event(event_type: str, details: Optional[dict] = None) -> None:
    """パイプラインイベントを記録"""
    db_path = get_pipeline_health_db()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details or {},
    }
    save_health_entry(db_path, entry)


def get_latest_log_date(log_dir: Path) -> Optional[str]:
    """最新のパイプラインログの日付を取得"""
    logs = sorted(log_dir.glob("improvement_*.log"))
    if not logs:
        return None
    # improvement_YYYYMMDD.log 形式から日付を抽出
    match = re.search(r"improvement_(\d{8})\.log", logs[-1].name)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze pipeline failure patterns over recent runs",
    )
    parser.add_argument(
        "--record-failure",
        metavar="LOG_PATH",
        type=Path,
        help="Record a pipeline failure from a log file",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    log_dir = get_log_dir()
    db_path = get_pipeline_health_db()

    if args.record_failure:
        analysis = parse_pipeline_log(args.record_failure)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "failure_detected",
            "log_path": str(args.record_failure),
            "analysis": analysis,
        }
        save_health_entry(db_path, entry)
        if args.json:
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        else:
            print(f"Recorded failure from {args.record_failure}")
        return

    if args.analyze:
        entries = load_health_entries(db_path)
        pattern = detect_failure_pattern(entries, days=args.days)
        if args.json:
            print(json.dumps(pattern, ensure_ascii=False, indent=2))
        else:
            print(f"Pipeline Health Analysis (last {args.days} days):")
            print(f"  Status: {pattern.get('status', 'unknown')}")
            print(f"  Recent runs: {pattern.get('recent_runs', 0)}")
            print(f"  Failed runs: {pattern.get('failed_runs', 0)}")
            print(f"  Failure rate: {pattern.get('failure_rate', 0)}%")
            if pattern.get("most_common_failures"):
                print("  Most common failures:")
                for step, count in pattern["most_common_failures"]:
                    print(f"    - {step}: {count} times")
        return

    # Default: list recent health status
    entries = load_health_entries(db_path)
    pattern = detect_failure_pattern(entries, days=args.days)
    print(json.dumps(pattern, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
