from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SECRET_CLEAR_BLOCK = """if (( has_replacement_secrets == 0 )); then
  deploy_args+=(--clear-secrets)
fi"""


def test_cloud_run_deploy_scripts_do_not_attach_cloud_sql() -> None:
    for relative_path in (
        "scripts/deploy_cloud_run.sh",
        "scripts/deploy_cloud_run_api.sh",
    ):
        script = (ROOT / relative_path).read_text(encoding="utf-8")

        assert 'DATABASE_SECRET_NAME=' not in script
        assert 'CLOUDSQL_INSTANCE=' not in script
        assert '--set-secrets "DATABASE_URL=' not in script
        assert '--add-cloudsql-instances' not in script
        assert '--remove-secrets=DATABASE_URL' not in script
        assert '--clear-cloudsql-instances' in script


def test_cloud_run_deploy_clears_secrets_only_without_replacements() -> None:
    expected_replacement_counts = {
        "scripts/deploy_cloud_run.sh": 3,
        "scripts/deploy_cloud_run_api.sh": 3,
    }

    for relative_path, expected_count in expected_replacement_counts.items():
        script = (ROOT / relative_path).read_text(encoding="utf-8")

        assert "has_replacement_secrets=0" in script
        assert script.count("has_replacement_secrets=1") == expected_count
        assert SECRET_CLEAR_BLOCK in script
        assert script.index(SECRET_CLEAR_BLOCK) < script.index("--clear-cloudsql-instances")
        assert "--set-secrets" in script
        assert "--remove-secrets" not in script


def test_public_cloud_run_deployments_require_api_access_key() -> None:
    combined_script = (ROOT / "scripts/deploy_cloud_run.sh").read_text(encoding="utf-8")
    api_script = (ROOT / "scripts/deploy_cloud_run_api.sh").read_text(encoding="utf-8")
    web_script = (ROOT / "scripts/deploy_cloud_run_web.sh").read_text(encoding="utf-8")

    assert "REQUIRE_API_ACCESS_KEY=1" in combined_script
    assert "Refusing to deploy a public service without an access key" in combined_script
    assert "REQUIRE_API_ACCESS_KEY=1" in api_script
    assert "Demo mode stays unauthenticated" not in api_script
    assert "Refusing to deploy a public API without an access key" in api_script
    assert "Refusing to deploy Web without the API proxy key" in web_script
