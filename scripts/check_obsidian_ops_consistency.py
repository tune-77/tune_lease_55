#!/usr/bin/env python3
"""Obsidian 運用（launchd ジョブ + Vault パス解決）の整合性を検査する。

検査のみで、plist もノートも書き換えない。

背景:
    Vault パスの解決契約が複数箇所に散らばり、
    ``OBSIDIAN_VAULT_PATH`` 優先の実装と ``OBSIDIAN_VAULT`` 優先の実装が併存していた。
    両者が別の値を指すと「書き込み先」と「RAG 索引先」が静かに分裂する。
    このスクリプトはその分裂と、夜間ジョブの発火時刻の衝突を検出する。

使い方:
    python3 scripts/check_obsidian_ops_consistency.py
    python3 scripts/check_obsidian_ops_consistency.py --json reports/obsidian_ops_consistency.json

終了コード:
    0 = error なし（warning は許容）
    1 = error あり
"""
from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import (  # noqa: E402
    LEGACY_OBSIDIAN_VAULT,
    OBSIDIAN_VAULT_ENV_VARS,
    describe_obsidian_vault_resolution,
)

DEFAULT_LAUNCHD_DIR = REPO_ROOT / "launchd"

# Vault を読む/書くジョブ。ここに載るジョブは Vault の env をすべて宣言していなければならない。
# 新しく Vault を触る launchd ジョブを足したらここにも追加する。
VAULT_JOB_LABELS = frozenset(
    {
        "com.tunelease.obsidian-reindex",
        "com.tunelease.obsidian-backup",
        "com.tunelease.aurion-core-midnight",
        "com.tunelease.aurion-core-morning-report",
        "com.tunelease.lease-news-collector",
        "com.tunelease.lease-judgment-autoresearch",
        "com.tunelease.daily-knowledge-feed",
        "com.tunelease.prompt-feedback-monthly",
        "com.tunelease.weekly-health-check",
    }
)

# 同時刻に走ると Vault の読み書きが並走するため、この間隔以内は衝突とみなす。
MIN_GAP_MINUTES = 10

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class Finding:
    level: str
    check: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "check": self.check, "message": self.message}


@dataclass
class Job:
    label: str
    path: Path
    env: dict[str, str] = field(default_factory=dict)
    program_arguments: list[str] = field(default_factory=list)
    calendar: dict[str, int] = field(default_factory=dict)

    @property
    def touches_vault(self) -> bool:
        return self.label in VAULT_JOB_LABELS

    @property
    def minute_of_day(self) -> int | None:
        if not self.calendar:
            return None
        return int(self.calendar.get("Hour", 0)) * 60 + int(self.calendar.get("Minute", 0))

    @property
    def interpreter(self) -> str | None:
        """実際に Python を起動しているパス（zsh ラッパー経由なら PYTHON_BIN）。"""
        for arg in self.program_arguments:
            if "python" in Path(arg).name:
                return arg
        return self.env.get("PYTHON_BIN")


def load_jobs(launchd_dir: Path) -> list[Job]:
    jobs: list[Job] = []
    for path in sorted(launchd_dir.glob("*.plist")):
        with path.open("rb") as handle:
            data = plistlib.load(handle)
        jobs.append(
            Job(
                label=str(data.get("Label") or path.stem),
                path=path,
                env=dict(data.get("EnvironmentVariables") or {}),
                program_arguments=[str(a) for a in (data.get("ProgramArguments") or [])],
                calendar=dict(data.get("StartCalendarInterval") or {}),
            )
        )
    return jobs


def check_vault_env(jobs: Iterable[Job]) -> list[Finding]:
    """Vault ジョブが env をすべて、同じ値で宣言しているか。"""
    findings: list[Finding] = []
    declared_values: dict[str, str] = {}

    for job in jobs:
        present = {name: job.env[name] for name in OBSIDIAN_VAULT_ENV_VARS if name in job.env}

        if not job.touches_vault:
            if present:
                findings.append(
                    Finding(
                        WARNING,
                        "vault_env",
                        f"{job.label}: Vault を触らない想定なのに {sorted(present)} が設定されています。"
                        " VAULT_JOB_LABELS に追加するか env を外してください",
                    )
                )
            continue

        missing = [name for name in OBSIDIAN_VAULT_ENV_VARS if name not in present]
        if missing:
            findings.append(
                Finding(
                    ERROR,
                    "vault_env",
                    f"{job.label}: Vault の env が不足しています（不足: {missing}）。"
                    " 参照側の実装ごとに読む変数名が違うため、両方を同じ値で設定してください",
                )
            )

        distinct = {v.rstrip("/") for v in present.values()}
        if len(distinct) > 1:
            findings.append(
                Finding(
                    ERROR,
                    "vault_env",
                    f"{job.label}: Vault の env が食い違っています（{present}）。"
                    " 書き込み先と RAG 索引先が分裂します",
                )
            )

        for value in present.values():
            declared_values[job.label] = value.rstrip("/")

    distinct_across_jobs = set(declared_values.values())
    if len(distinct_across_jobs) > 1:
        findings.append(
            Finding(
                ERROR,
                "vault_env",
                "launchd ジョブ間で Vault パスが一致していません: "
                + json.dumps(declared_values, ensure_ascii=False),
            )
        )

    return findings


def _can_coincide(a: Job, b: Job) -> bool:
    """2つのジョブが同じ日に発火しうるか（曜日/日付が別なら重ならない）。"""
    for key in ("Weekday", "Day", "Month"):
        av, bv = a.calendar.get(key), b.calendar.get(key)
        if av is not None and bv is not None and av != bv:
            return False
    return True


def check_schedule_collisions(
    jobs: Iterable[Job], min_gap_minutes: int = MIN_GAP_MINUTES
) -> list[Finding]:
    """Vault を触るジョブ同士が近すぎる時刻で発火しないか。"""
    findings: list[Finding] = []
    scheduled = [j for j in jobs if j.minute_of_day is not None]

    for i, a in enumerate(scheduled):
        for b in scheduled[i + 1 :]:
            if not (a.touches_vault and b.touches_vault):
                continue
            if not _can_coincide(a, b):
                continue
            gap = abs((a.minute_of_day or 0) - (b.minute_of_day or 0))
            if gap < min_gap_minutes:
                level = ERROR if gap == 0 else WARNING
                findings.append(
                    Finding(
                        level,
                        "schedule",
                        f"{a.label} と {b.label} の発火間隔が {gap} 分です"
                        f"（{min_gap_minutes} 分未満）。Vault の読み書きが並走します",
                    )
                )
    return findings


CANONICAL_INTERPRETER_SUFFIX = "/.venv/bin/python"


def check_interpreters(jobs: Iterable[Job]) -> list[Finding]:
    """Python 実行系がジョブ間で統一されているか。

    実行系が割れていると、あるジョブにだけ入っているパッケージに依存したコードが
    別のジョブでは ImportError になる。全ジョブがプロジェクトの venv を使うのが規約。
    """
    interpreters: dict[str, list[str]] = {}
    for job in jobs:
        interp = job.interpreter
        if interp:
            interpreters.setdefault(interp, []).append(job.label)

    findings: list[Finding] = []

    if len(interpreters) > 1:
        detail = json.dumps(interpreters, ensure_ascii=False, indent=2)
        findings.append(
            Finding(
                ERROR,
                "interpreter",
                f"Python 実行系が {len(interpreters)} 種類に分かれています。"
                f"インストール済みパッケージが揃わずジョブごとに失敗します:\n{detail}",
            )
        )

    for interp, labels in interpreters.items():
        if not interp.endswith(CANONICAL_INTERPRETER_SUFFIX):
            findings.append(
                Finding(
                    WARNING,
                    "interpreter",
                    f"{sorted(labels)} がプロジェクトの venv 以外の Python を使っています: {interp}"
                    f"（規約: *{CANONICAL_INTERPRETER_SUFFIX}）",
                )
            )

    return findings


def check_vault_resolution() -> list[Finding]:
    """現在の環境で runtime_paths がどこを指すか。"""
    resolution = describe_obsidian_vault_resolution()
    findings = [
        Finding(
            INFO,
            "vault_resolution",
            f"runtime_paths の解決結果: {resolution.path}（解決元: {resolution.source}）",
        )
    ]
    findings.extend(
        Finding(WARNING, "vault_resolution", warning) for warning in resolution.warnings
    )
    return findings


def scan_hardcoded_vault_paths(repo_root: Path = REPO_ROOT) -> list[Finding]:
    """runtime_paths を経由せず Vault パスを直書きしているファイルを検出する。

    集約は完了済みなので、新たな直書きが入ったら error として止める。
    テストは「iCloud 配下であること」を確かめる目的で文字列を持つため除外する。
    """
    pattern = re.compile(r"iCloud~md~obsidian")
    allowlist = {
        repo_root / "runtime_paths.py",  # 解決窓口そのもの
        Path(__file__).resolve(),  # この検査スクリプト
    }
    hits: list[str] = []
    for suffix in ("*.py", "*.sh"):
        for path in repo_root.rglob(suffix):
            if path.resolve() in allowlist:
                continue
            if any(part in {".git", "node_modules", "_archive", ".venv"} for part in path.parts):
                continue
            if path.name.startswith("test_") or "tests" in path.parts:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern.search(text):
                hits.append(str(path.relative_to(repo_root)))

    if not hits:
        return []
    listed = "\n".join(f"  - {h}" for h in sorted(hits))
    return [
        Finding(
            ERROR,
            "hardcoded_paths",
            f"Vault パスを直書きしているファイルが {len(hits)} 件あります。"
            f"runtime_paths の解決窓口を使ってください:\n{listed}",
        )
    ]


def run_checks(launchd_dir: Path = DEFAULT_LAUNCHD_DIR) -> dict[str, Any]:
    jobs = load_jobs(launchd_dir)
    findings: list[Finding] = []
    findings.extend(check_vault_env(jobs))
    findings.extend(check_schedule_collisions(jobs))
    findings.extend(check_interpreters(jobs))
    findings.extend(check_vault_resolution())
    findings.extend(scan_hardcoded_vault_paths())

    counts = {
        level: sum(1 for f in findings if f.level == level)
        for level in (ERROR, WARNING, INFO)
    }
    return {
        "launchd_dir": str(launchd_dir),
        "job_count": len(jobs),
        "vault_job_count": sum(1 for j in jobs if j.touches_vault),
        "canonical_env_vars": list(OBSIDIAN_VAULT_ENV_VARS),
        "legacy_vault": str(LEGACY_OBSIDIAN_VAULT),
        "counts": counts,
        "status": "error" if counts[ERROR] else ("warning" if counts[WARNING] else "ok"),
        "findings": [f.as_dict() for f in findings],
    }


def format_report(report: dict[str, Any]) -> str:
    icons = {ERROR: "❌", WARNING: "⚠️ ", INFO: "ℹ️ "}
    lines = [
        "Obsidian 運用整合性チェック",
        f"  ジョブ数: {report['job_count']}（うち Vault 使用: {report['vault_job_count']}）",
        f"  総合: {report['status'].upper()}"
        f"  error={report['counts']['error']}"
        f" warning={report['counts']['warning']}",
        "",
    ]
    for finding in report["findings"]:
        lines.append(f"{icons.get(finding['level'], '')} [{finding['check']}] {finding['message']}")
    if not report["findings"]:
        lines.append("✅ 指摘なし")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launchd-dir", type=Path, default=DEFAULT_LAUNCHD_DIR)
    parser.add_argument("--json", type=Path, help="レポートの JSON 出力先")
    args = parser.parse_args(argv)

    report = run_checks(args.launchd_dir)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON: {args.json}")

    return 1 if report["counts"][ERROR] else 0


if __name__ == "__main__":
    raise SystemExit(main())
