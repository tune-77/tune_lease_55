#!/usr/bin/env python3
"""launchd ジョブの Python 実行系に、必要な依存が揃っているか実機で検証する。

launchd の実行系を anaconda から プロジェクトの venv に統一したため、
「venv には入っていないパッケージに依存しているジョブ」が朝になって初めて
ImportError で落ちる、という事故が起こりうる。このスクリプトは各ジョブの
エントリスクリプトのトップレベル import を静的に読み、その実行系で実際に
import できるかを確認する。

**Mac の実機で流すことを想定している。** venv が存在しない環境では
その旨を報告して終了コード 0 を返す（CI を落とさない）。

使い方:
    python3 scripts/check_launchd_python_env.py
    python3 scripts/check_launchd_python_env.py --interpreter /path/to/python
    python3 scripts/check_launchd_python_env.py --json reports/launchd_python_env.json

終了コード:
    0 = 全ジョブ import 可能、または実行系が存在せず検証をスキップ
    1 = import できない依存があった
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from check_obsidian_ops_consistency import Job, load_jobs  # noqa: E402

DEFAULT_LAUNCHD_DIR = REPO_ROOT / "launchd"

# import 名 != パッケージ名 のものは、import 名でしか検証できないのでそのまま使う。
# ここに挙げるのは「import できなくても実害がない」オプショナル依存。
OPTIONAL_MODULES = frozenset({"chromadb", "google", "timesfm", "umap"})

_PY_IN_SHELL = re.compile(r"([A-Za-z0-9_./$\{\}\-]+\.py)")
_MODULE_IN_SHELL = re.compile(r"-m\s+([A-Za-z_][\w.]*)")


def _script_for_job(job: Job) -> Path | None:
    """ジョブが最終的に起動する .py を割り出す（zsh ラッパー経由も追う）。"""
    for arg in job.program_arguments:
        if arg.endswith(".py"):
            return _localize(arg)

    # -m モジュール指定
    if "-m" in job.program_arguments:
        idx = job.program_arguments.index("-m")
        if idx + 1 < len(job.program_arguments):
            module = job.program_arguments[idx + 1]
            candidate = REPO_ROOT / (module.replace(".", "/") + ".py")
            if candidate.exists():
                return candidate

    for arg in job.program_arguments:
        if arg.endswith(".sh"):
            shell = _localize(arg)
            if not shell or not shell.exists():
                continue
            text = shell.read_text(encoding="utf-8")
            for match in _PY_IN_SHELL.findall(text):
                candidate = _localize(match)
                if candidate and candidate.exists():
                    return candidate
            # `python -m mobile_app.rag_daily_maintenance` 形式
            for module in _MODULE_IN_SHELL.findall(text):
                candidate = REPO_ROOT / (module.replace(".", "/") + ".py")
                if candidate.exists():
                    return candidate
    return None


def _localize(mac_path: str) -> Path | None:
    """plist / シェル内のパスを、このリポジトリ上の位置に読み替える。"""
    marker = "tune_lease_55/"
    if marker in mac_path:
        return REPO_ROOT / mac_path.split(marker, 1)[1]

    # "$SCRIPT_DIR/foo.py" のようなシェル変数付きはベース名で引き当てる
    if "$" in mac_path:
        name = Path(mac_path).name
        for base in (REPO_ROOT / "scripts", REPO_ROOT):
            candidate = base / name
            if candidate.exists():
                return candidate
        return None

    path = Path(mac_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _module_names(nodes: Iterable[ast.AST]) -> set[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _parse(script: Path) -> ast.Module | None:
    try:
        return ast.parse(script.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return None


def top_level_imports(script: Path) -> set[str]:
    """モジュール直下の import。失敗するとジョブが起動時点で死ぬ。"""
    tree = _parse(script)
    return _module_names(tree.body) if tree else set()


def deferred_imports(script: Path) -> set[str]:
    """関数内などの遅延 import。失敗してもその経路だけが死ぬ。

    このリポジトリは重い依存を関数内で import する書き方が多く、
    トップレベルだけ見ると検証がほぼ素通りしてしまうため別枠で拾う。
    """
    tree = _parse(script)
    if not tree:
        return set()
    top = set(tree.body)
    nested = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and n not in top]
    return _module_names(nested) - _module_names(tree.body)


def _is_local(name: str) -> bool:
    """リポジトリ内のモジュールか（scripts/ 直下も sys.path に載る運用のため含める）。"""
    for base in (REPO_ROOT, REPO_ROOT / "scripts"):
        if (base / f"{name}.py").exists() or (base / name / "__init__.py").exists():
            return True
    return False


def _external(names: Iterable[str]) -> set[str]:
    stdlib = set(sys.stdlib_module_names)
    return {
        name
        for name in names
        if name not in stdlib and not _is_local(name) and not name.startswith("_")
    }


def verify_imports(interpreter: Path, modules: Iterable[str]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for module in sorted(modules):
        proc = subprocess.run(
            [str(interpreter), "-c", f"import {module}"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        results[module] = proc.returncode == 0
    return results


def run(launchd_dir: Path, interpreter_override: str | None = None) -> dict:
    jobs = load_jobs(launchd_dir)

    interpreters = {j.interpreter for j in jobs if j.interpreter}
    if interpreter_override:
        interpreter = Path(interpreter_override)
    elif len(interpreters) == 1:
        interpreter = _localize(next(iter(interpreters)))
    else:
        return {
            "status": "error",
            "reason": f"実行系が統一されていません: {sorted(interpreters)}",
            "jobs": [],
        }

    if not interpreter or not interpreter.exists():
        return {
            "status": "skipped",
            "reason": f"実行系が存在しません（Mac 実機で流してください）: {interpreter}",
            "interpreter": str(interpreter),
            "jobs": [],
        }

    job_reports = []
    missing_total = 0
    for job in jobs:
        script = _script_for_job(job)
        if script is None or not script.exists():
            job_reports.append(
                {"label": job.label, "status": "skipped", "reason": "エントリスクリプト不明"}
            )
            continue

        required = _external(top_level_imports(script))
        deferred = _external(deferred_imports(script)) - required

        results = verify_imports(interpreter, required | deferred)
        missing_required = [m for m in sorted(required) if not results[m]]
        missing_deferred = [m for m in sorted(deferred) if not results[m]]

        # トップレベルの import が欠けるとジョブは起動時点で落ちる = blocking。
        # 遅延 import は該当経路だけの問題なので報告に留める。
        blocking = [m for m in missing_required if m not in OPTIONAL_MODULES]
        missing_total += len(blocking)

        job_reports.append(
            {
                "label": job.label,
                "script": str(script.relative_to(REPO_ROOT)),
                "required": sorted(required),
                "deferred": sorted(deferred),
                "missing_required": missing_required,
                "missing_deferred": missing_deferred,
                "blocking": blocking,
                "status": "error" if blocking else "ok",
            }
        )

    return {
        "status": "error" if missing_total else "ok",
        "interpreter": str(interpreter),
        "blocking_count": missing_total,
        "jobs": job_reports,
    }


def format_report(report: dict) -> str:
    if report["status"] == "skipped":
        return f"⏭️  スキップ: {report['reason']}"

    lines = [
        "launchd Python 実行系チェック",
        f"  実行系: {report.get('interpreter', '?')}",
        f"  総合: {report['status'].upper()}",
        "",
    ]
    for job in report["jobs"]:
        if job["status"] == "skipped":
            lines.append(f"⏭️  {job['label']}: {job['reason']}")
            continue

        count = len(job["required"]) + len(job["deferred"])
        if job["status"] == "ok":
            lines.append(f"✅ {job['label']}: {count} 件検証（{job['script']}）")
        else:
            lines.append(
                f"❌ {job['label']}: トップレベル import 不可 → {job['blocking']}"
                f"（{job['script']}）"
            )
        if job["missing_deferred"]:
            lines.append(f"   ⚠️  遅延 import が未導入: {job['missing_deferred']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launchd-dir", type=Path, default=DEFAULT_LAUNCHD_DIR)
    parser.add_argument("--interpreter", help="検証に使う Python（既定: plist の実行系）")
    parser.add_argument("--json", type=Path, help="レポートの JSON 出力先")
    args = parser.parse_args(argv)

    report = run(args.launchd_dir, args.interpreter)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {args.json}")

    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
