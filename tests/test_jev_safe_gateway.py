from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jev_safe_gateway import GatewayInputError, prepare_gateway_request


def test_abstract_projection_keeps_mapping_local_and_writes_content_free_audit(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    result = prepare_gateway_request(
        {
            "mode": "abstract",
            "purpose": "変更候補を依頼との関連度で分類する",
            "items": [
                {
                    "source_id": "private-file-17",
                    "artifact_kind": "application code",
                    "change_kind": "feature",
                    "relation": "direct",
                    "effect": "既存画面に利用者向けの操作を追加する",
                    "verification_status": "focused tests passed",
                }
            ],
        },
        audit_path=audit_path,
    )

    assert result["status"] == "allowed"
    assert result["local_mapping"] == {"item_1": "private-file-17"}
    assert result["outbound"]["items"][0]["id"] == "item_1"
    assert "private-file-17" not in json.dumps(result["outbound"], ensure_ascii=False)

    audit_text = audit_path.read_text(encoding="utf-8")
    assert "private-file-17" not in audit_text
    assert "既存画面" not in audit_text
    assert json.loads(audit_text)["content_logged"] is False


@pytest.mark.parametrize(
    "effect,reason",
    [
        ("/Users/example/private/repo/app.py を変更", "private_path"),
        ("/workspace/acme/service を変更", "private_path"),
        ("/srv/internal/repository を変更", "private_path"),
        ("```python\ndef secret():\n    pass\n```", "raw_code_or_diff"),
        ("src/private_module.py の処理を変更", "raw_code_or_diff"),
        ("src/internal/service の処理を変更", "raw_code_or_diff"),
        ("`InternalService` の処理を変更", "raw_code_or_diff"),
        ("API_KEY=super-secret-value", "secret_like_content"),
        ("申込者: 山田太郎", "pii_like_content"),
    ],
)
def test_abstract_projection_blocks_sensitive_or_raw_content(effect, reason):
    result = prepare_gateway_request(
        {
            "mode": "abstract",
            "purpose": "変更候補を分類する",
            "items": [{"source_id": "local", "effect": effect}],
        }
    )

    assert result["status"] == "blocked"
    assert result["outbound"] is None
    assert reason in result["audit"]["reason_codes"]


def test_public_excerpt_requires_an_approved_public_https_url():
    with pytest.raises(GatewayInputError, match="approved HTTPS host"):
        prepare_gateway_request(
            {
                "mode": "public_excerpt",
                "purpose": "公開コードの変更種別を分類する",
                "items": [
                    {
                        "source_id": "local",
                        "visibility": "public",
                        "public_source_url": "https://git.example.internal/source.py",
                        "content": "def add(a, b): return a + b",
                    }
                ],
            }
        )


def test_public_excerpt_rejects_credentials_or_query_in_public_url():
    with pytest.raises(GatewayInputError, match="must not contain credentials"):
        prepare_gateway_request(
            {
                "mode": "public_excerpt",
                "purpose": "公開コードを分類する",
                "items": [
                    {
                        "visibility": "public",
                        "public_source_url": "https://token@github.com/example/project?key=secret",
                        "content": "safe public content",
                    }
                ],
            }
        )


def test_public_excerpt_allows_clean_bounded_public_code():
    result = prepare_gateway_request(
        {
            "mode": "public_excerpt",
            "purpose": "公開コードの変更種別を分類する",
            "items": [
                {
                    "source_id": "local",
                    "visibility": "public",
                    "public_source_url": "https://github.com/example/project/blob/main/add.py",
                    "artifact_kind": "code",
                    "content": "def add(a, b):\n    return a + b\n",
                }
            ],
        }
    )

    assert result["status"] == "allowed"
    assert result["outbound"]["items"][0]["content"] == "def add(a, b):\n    return a + b\n"


def test_public_excerpt_still_blocks_a_secret():
    result = prepare_gateway_request(
        {
            "mode": "public_excerpt",
            "purpose": "公開コードを分類する",
            "items": [
                {
                    "visibility": "public",
                    "public_source_url": "https://github.com/example/project/blob/main/config.py",
                    "content": "token = 'ghp_abcdefghijklmnopqrstuvwxyz123456'",
                }
            ],
        }
    )

    assert result["status"] == "blocked"
    assert "secret_like_content" in result["audit"]["reason_codes"]


def test_public_excerpt_blocks_fine_grained_github_token():
    result = prepare_gateway_request(
        {
            "mode": "public_excerpt",
            "purpose": "公開コードを分類する",
            "items": [
                {
                    "visibility": "public",
                    "public_source_url": "https://github.com/example/project/blob/main/config.py",
                    "content": "github_pat_11AA22BB33CC44DD55EE66FF77GG88HH",
                }
            ],
        }
    )

    assert result["status"] == "blocked"
    assert "secret_like_content" in result["audit"]["reason_codes"]


def test_aggregate_accepts_buckets_and_flags_but_rejects_raw_numbers():
    allowed = prepare_gateway_request(
        {
            "mode": "aggregate",
            "purpose": "classify_review_route",
            "items": [
                {
                    "source_id": "case-raw-id",
                    "industry_bucket": "manufacturing",
                    "size_bucket": "medium",
                    "risk_flags": ["short_history", "high_concentration"],
                    "boolean_signals": {"has_guarantor": True},
                }
            ],
        }
    )
    assert allowed["status"] == "allowed"
    assert "case-raw-id" not in json.dumps(allowed["outbound"])

    with pytest.raises(GatewayInputError, match="must be a string"):
        prepare_gateway_request(
            {
                "mode": "aggregate",
                "purpose": "classify_review_route",
                "items": [{"size_bucket": 12345678}],
            }
        )

    for raw_value in ("12345678円", "年商 12345678円", "annual revenue 12345678 yen"):
        with pytest.raises(GatewayInputError, match="unknown value"):
            prepare_gateway_request(
                {
                    "mode": "aggregate",
                    "purpose": "classify_review_route",
                    "items": [{"size_bucket": raw_value}],
                }
            )

    with pytest.raises(GatewayInputError, match="unknown key"):
        prepare_gateway_request(
            {
                "mode": "aggregate",
                "purpose": "classify_review_route",
                "items": [{"boolean_signals": {"annual revenue 12345678 yen": True}}],
            }
        )


def test_aggregate_rejects_free_text_and_untrusted_purpose():
    with pytest.raises(GatewayInputError, match="unknown value"):
        prepare_gateway_request(
            {
                "mode": "aggregate",
                "purpose": "classify_review_route",
                "items": [{"industry_bucket": "株式会社秘密商事"}],
            }
        )

    with pytest.raises(GatewayInputError, match="predefined value"):
        prepare_gateway_request(
            {
                "mode": "aggregate",
                "purpose": "classify annual revenue 12345678 yen",
                "items": [{"size_bucket": "medium"}],
            }
        )


@pytest.mark.parametrize(
    "effect",
    ["internal_config.goを変更", "README.mdを変更", "Dockerfileを変更", "Makefile を変更"],
)
def test_abstract_projection_blocks_generic_filenames(effect):
    result = prepare_gateway_request(
        {
            "mode": "abstract",
            "purpose": "変更候補を分類する",
            "items": [{"effect": effect}],
        }
    )

    assert result["status"] == "blocked"
    assert "raw_code_or_diff" in result["audit"]["reason_codes"]


def test_unknown_fields_are_rejected_instead_of_silently_forwarded():
    with pytest.raises(GatewayInputError, match="unsupported fields: raw_diff"):
        prepare_gateway_request(
            {
                "mode": "abstract",
                "purpose": "変更候補を分類する",
                "items": [{"effect": "一般化した変更", "raw_diff": "+ secret"}],
            }
        )


def test_non_string_mode_is_reported_as_invalid_input():
    with pytest.raises(GatewayInputError, match="mode must be"):
        prepare_gateway_request({"mode": [], "purpose": "分類する", "items": [{}]})


def test_cli_imports_root_module_when_project_root_is_already_on_pythonpath(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    payload = json.dumps(
        {
            "mode": "abstract",
            "purpose": "変更候補を分類する",
            "items": [{"effect": "既存画面へ操作を追加する"}],
        },
        ensure_ascii=False,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "jev_safe_gateway.py"),
            "--input",
            "-",
            "--audit",
            str(tmp_path / "audit.jsonl"),
        ],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "allowed"
