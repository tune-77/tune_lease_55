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
        "scripts/deploy_cloud_run.sh": 2,
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
