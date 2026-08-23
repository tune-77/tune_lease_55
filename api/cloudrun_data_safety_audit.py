"""Cloud Run実データ保護の読み取り専用監査。

#863/#864/#865 周辺は、審査データ登録、APIアクセス制御、Cloud Run再起動時の
SQLite永続化に触れるため、通常の構文チェックだけでは足りない。このモジュールは
CI/ADK/UI から同じ観点を読めるよう、リポジトリ内のコードとテスト配線だけを静的に
確認する。GCS、DB、本番APIへの接続や書き込みは行わない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_TESTS = [
    "tests/test_cloudrun_db_snapshot.py",
    "tests/test_api_key_auth.py",
    "tests/test_promote_cloudrun_return_data.py",
    "tests/test_cloudrun_data_safety_audit.py",
]


def _read_text(path: str) -> str:
    try:
        return (_REPO_ROOT / path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _check(name: str, ok: bool, message: str, severity: str = "critical") -> dict[str, Any]:
    return {
        "name": name,
        "status": "ok" if ok else "fail",
        "severity": severity,
        "message": message,
    }


def audit_cloudrun_data_safety() -> dict[str, Any]:
    """Cloud Run実データ保護の静的監査を実行する。

    Returns:
        status, issue_count, checks, guardrail を含む辞書。読み取り専用で、DB/GCS/
        GitHub設定は変更しない。
    """
    workflow = _read_text(".github/workflows/pr-checks.yml")
    snapshot = _read_text("api/cloudrun_db_snapshot.py")
    restore = _read_text("scripts/restore_lease_db_snapshot.py")
    api_auth = _read_text("api/api_key_auth.py")
    deploy_api = _read_text("scripts/deploy_cloud_run_api.sh")
    promote = _read_text("scripts/promote_cloudrun_return_data.py")
    pending = _read_text("api/cloudrun_pending_cases.py")
    agent_tools = _read_text("api/shion_agent_tools.py")
    judgment_router = _read_text("api/routers/judgment_assets.py")

    checks = [
        _check(
            "ci_job_present",
            "cloudrun-data-safety:" in workflow,
            "PR Checks に cloudrun-data-safety ジョブがある",
        ),
        _check(
            "ci_runs_required_tests",
            all(test in workflow for test in _REQUIRED_TESTS),
            "cloudrun-data-safety ジョブが危険域テストを実行する",
        ),
        _check(
            "api_access_key_fail_closed",
            "api_access_key_required" in api_auth
            and "K_SERVICE" in api_auth
            and "API_ACCESS_KEY is required in this runtime" in api_auth,
            "Cloud Run系実行環境で API_ACCESS_KEY 未設定時に /api/* を fail-closed する",
        ),
        _check(
            "deploy_refuses_missing_access_key",
            "Refusing to deploy non-demo" in deploy_api
            and "API_ACCESS_KEY" in deploy_api
            and "exit 1" in deploy_api,
            "非demoデプロイ時に API_ACCESS_KEY Secret 不在ならデプロイを止める",
        ),
        _check(
            "snapshot_disabled_outside_runtime",
            "K_SERVICE" in snapshot and 'mode == "demo"' in snapshot and "if mode:" in snapshot,
            "GCS DBスナップショットはdemo無効、明示非demoまたはCloud Run実行時だけ有効",
        ),
        _check(
            "restore_never_overwrites_existing_db",
            "target.exists()" in restore
            and "target.stat().st_size > 0" in restore
            and "os.replace" in restore
            and ".downloading" in restore,
            "GCS復元は既存DBを上書きせず、一時ファイルから成功時のみ置換する",
        ),
        _check(
            "score_input_dedupes_return_registered_case_id",
            "return_registered_case_id" in promote
            and "skipped_already_registered" in promote
            and "already registered via /api/cases/register" in promote,
            "Cloud Run score_input 昇格は register_case_result 済み案件を二重登録しない",
        ),
        _check(
            "main_db_requires_explicit_allow",
            "allow_main_db" in promote
            and "Refusing to promote Cloud Run return data into data/lease_data.db" in promote,
            "lease_data.db への昇格は --allow-main-db/allow_main_db=True を明示要求する",
        ),
        _check(
            "redacted_company_name_placeholder",
            "Cloud Run審査入力（企業名要確認" in pending and "[REDACTED]" in pending,
            "マスキング済み企業名は推測せず、人間補完用プレースホルダーにする",
            severity="high",
        ),
        _check(
            "adk_read_only_tool_registered",
            "CLOUDRUN_DATA_SAFETY_AUDIT_TOOLS" in agent_tools,
            "紫苑ADKの読み取り専用ツールとして Cloud Runデータ安全監査が登録されている",
            severity="high",
        ),
        _check(
            "ui_api_endpoint_present",
            "cloudrun-data-safety-audit" in judgment_router,
            "improvement-log から読める監査APIがある",
            severity="high",
        ),
    ]
    failed = [check for check in checks if check["status"] != "ok"]
    critical_failed = [check for check in failed if check["severity"] == "critical"]
    status = "ok" if not failed else "critical" if critical_failed else "warn"
    return {
        "mode": "cloudrun_data_safety_audit",
        "status": status,
        "issue_count": len(failed),
        "critical_issue_count": len(critical_failed),
        "checks": checks,
        "guardrail": "read_only_no_db_write_no_gcs_access_no_github_mutation",
    }


CLOUDRUN_DATA_SAFETY_AUDIT_TOOLS = [
    audit_cloudrun_data_safety,
]
