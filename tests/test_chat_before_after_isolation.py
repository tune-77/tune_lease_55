from pathlib import Path

from experiments.chat_before_after import run_both


def test_sandbox_env_removes_production_database_and_cloud_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://production")
    monkeypatch.setenv("CLOUDRUN_DATA_MODE", "production")
    monkeypatch.setenv("K_SERVICE", "prod")
    root, vault, out = tmp_path / "repo", tmp_path / "vault", tmp_path / "out"
    env = run_both.sandbox_env(root, vault, out, "before")
    assert "DATABASE_URL" not in env and "K_SERVICE" not in env
    assert env["DATA_DIR"] == str(root / "data")
    assert env["DB_PATH"] == str(root / "data" / "lease_data.db")
    assert env["USE_GCS_VAULT"] == "false"


def test_compare_uses_a_distinct_user_id_per_side_and_question():
    source = (Path(__file__).parents[1] / "experiments" / "chat_before_after" / "compare.py").read_text(encoding="utf-8")
    assert "before_after_compare_{run_id}_{side}_{question['id']}" in source
