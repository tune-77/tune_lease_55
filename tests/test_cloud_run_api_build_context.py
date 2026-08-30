from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_api_deploy_uses_dedicated_ignore_file() -> None:
    deploy_script = (ROOT / "scripts/deploy_cloud_run_api.sh").read_text(
        encoding="utf-8"
    )

    assert 'API_IGNORE_FILE="${API_IGNORE_FILE:-$ROOT_DIR/.gcloudignore.api}"' in deploy_script
    assert '--ignore-file "$API_IGNORE_FILE"' in deploy_script
    assert 'if [[ ! -f "$API_IGNORE_FILE" ]]' in deploy_script


def test_api_ignore_excludes_non_runtime_trees_but_keeps_bundle() -> None:
    ignore_text = (ROOT / ".gcloudignore.api").read_text(encoding="utf-8")

    for pattern in ("frontend/", "mebuki/", "tests/", "docs/", "artifacts/", "graft/"):
        assert pattern in ignore_text
    assert ".cloudrun_bundle/" not in ignore_text


def test_api_dockerfile_copies_build_context_once() -> None:
    dockerfile = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")

    assert "COPY . ." in dockerfile
    assert "COPY .cloudrun_bundle/ /app/.cloudrun_bundle/" not in dockerfile
