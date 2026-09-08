"""scripts/setup_artifact_registry_cleanup_policy.sh の配線テスト。

高頻度push（deploy.yml が api/**・frontend/** 変更のたびにビルドする）で
Artifact Registry に溜まり続けるイメージのストレージ課金を、GCP標準の
クリーンアップポリシー（サーバー側で継続適用される宣言的設定）で止める
（2026-09のCloud Runコスト急増インシデント振り返り、PR #978フォローアップ）。
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts/setup_artifact_registry_cleanup_policy.sh").read_text(encoding="utf-8")


def test_sets_cleanup_policy_on_the_deploy_repository() -> None:
    assert "gcloud artifacts repositories set-cleanup-policies" in SCRIPT
    assert 'REPOSITORY="${REPOSITORY:-cloud-run-source-deploy}"' in SCRIPT


def test_never_deletes_the_most_recent_versions() -> None:
    # Keepポリシーが常にDeleteより優先して評価されるため、直近デプロイ・
    # ロールバック候補は誤って消えない。
    assert '"action": {"type": "Keep"}' in SCRIPT
    assert "mostRecentVersions" in SCRIPT
    assert 'KEEP_COUNT="${KEEP_COUNT:-15}"' in SCRIPT


def test_only_deletes_versions_older_than_threshold() -> None:
    assert '"action": {"type": "Delete"}' in SCRIPT
    assert '"condition": {"olderThan":' in SCRIPT
    assert 'DELETE_AFTER_DAYS="${DELETE_AFTER_DAYS:-30}"' in SCRIPT


def test_requires_project_id() -> None:
    assert 'PROJECT_ID is required.' in SCRIPT
    assert "exit 1" in SCRIPT
