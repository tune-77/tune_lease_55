from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_cloud_run_deploy_does_not_mix_secret_set_and_remove_flags() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_api.sh").read_text(encoding="utf-8")

    assert "--set-secrets" in script
    assert "--remove-secrets" not in script
