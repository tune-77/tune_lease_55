"""Read-only compatibility check for committed model pickle/joblib artifacts.

Loads every model file under data/, models/, mobile_app/ with the currently
installed dependencies (scikit-learn, lightgbm, etc.) and reports which ones
fail. It does not retrain, modify, or delete anything.

Background: requirements.txt and requirements-dev.txt once pinned different
scikit-learn versions, which silently broke several committed model pickles
(mobile_app/simple_model.pkl, data/ml_rf_v4.pkl, data/spread_predictor_v2.pkl)
without crashing the app — the loaders already had try/except fallbacks, so
the failures only showed up as log lines nobody was watching. This check
surfaces that class of drift on a recurring basis instead of by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# コミット済みpickleにはプロジェクト固有クラス（MahalanobisScorer等）が含まれるため、
# unpickle 時にそれらのモジュールをimportできるようリポジトリルートをパスへ追加する。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_JSON = PROJECT_ROOT / "reports" / "model_pickle_compat_latest.json"
DEFAULT_MD = PROJECT_ROOT / "reports" / "model_pickle_compat_latest.md"

DEFAULT_SCAN_DIRS = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "mobile_app",
]

MODEL_SUFFIXES = {".pkl", ".joblib"}
# 明らかにモデルではないバックアップ・一時ファイルは対象外
SKIP_NAME_SUFFIXES = (".bak.pkl",)


def _discover_model_files(scan_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for base in scan_dirs:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix not in MODEL_SUFFIXES:
                continue
            if any(path.name.endswith(suffix) for suffix in SKIP_NAME_SUFFIXES):
                continue
            found.append(path)
    return found


def _check_one(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(PROJECT_ROOT))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import joblib

            obj = joblib.load(path)
            version_warnings = [
                str(w.message) for w in caught if "version" in str(w.message).lower()
            ]
            return {
                "path": rel,
                "status": "ok",
                "type": type(obj).__name__,
                "version_warnings": version_warnings,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - report every failure mode, not just known ones
            return {
                "path": rel,
                "status": "fail",
                "type": "",
                "version_warnings": [],
                "error": f"{type(exc).__name__}: {exc}",
            }


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("sklearn", "numpy", "lightgbm", "joblib"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = "not_installed"
    return versions


def build_report(scan_dirs: list[Path]) -> dict[str, Any]:
    files = _discover_model_files(scan_dirs)
    results = [_check_one(path) for path in files]
    failed = [r for r in results if r["status"] == "fail"]
    with_warnings = [r for r in results if r["status"] == "ok" and r["version_warnings"]]
    status = "warn" if failed else "ok"
    return {
        "label": "Model Pickle Compatibility Check",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "installed_versions": _installed_versions(),
        "scanned_count": len(results),
        "failed_count": len(failed),
        "version_warning_count": len(with_warnings),
        "results": results,
        "guardrail": "read_only_no_retrain_no_delete",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Model Pickle Compatibility Check",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- status: `{report['status']}`",
        f"- scanned: `{report['scanned_count']}` / failed: `{report['failed_count']}` "
        f"/ version-mismatch warnings: `{report['version_warning_count']}`",
        f"- guardrail: `{report['guardrail']}`",
        "",
        "## Installed versions",
        "",
    ]
    for name, version in report["installed_versions"].items():
        lines.append(f"- `{name}`: `{version}`")
    lines.extend(["", "## Failed files", ""])
    failed = [r for r in report["results"] if r["status"] == "fail"]
    if not failed:
        lines.append("- none")
    for item in failed:
        lines.append(f"- `{item['path']}` — {item['error']}")
    lines.extend(["", "## Loaded with version-mismatch warnings", ""])
    warned = [r for r in report["results"] if r["status"] == "ok" and r["version_warnings"]]
    if not warned:
        lines.append("- none")
    for item in warned:
        for message in item["version_warnings"]:
            lines.append(f"- `{item['path']}` — {message}")
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], *, output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check committed model pickle/joblib files load under the installed dependencies."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_MD)
    parser.add_argument("--scan-dir", type=Path, action="append")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    report = build_report(args.scan_dir or DEFAULT_SCAN_DIRS)
    write_outputs(report, output_json=args.json, output_md=args.report)
    if args.print:
        print(render_markdown(report))
    else:
        print(f"report={args.report}")
        print(f"status={report['status']}")
        print(f"scanned={report['scanned_count']}")
        print(f"failed={report['failed_count']}")
        for item in report["results"]:
            if item["status"] == "fail":
                print(f"  FAIL {item['path']}: {item['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
