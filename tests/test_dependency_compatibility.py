from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tokenizers_constraint_stays_compatible_with_transformers() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"tokenizers>=0.22.0,<0.23.0"' in pyproject
    assert '"tokenizers>=0.22.0,<0.24.0"' not in pyproject
