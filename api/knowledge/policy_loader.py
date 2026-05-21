"""
Obsidian Vault の 審査方針.md を読み込み、全エージェントのシステムプロンプトに注入するためのローダー。
frontmatter に active: false が設定されている場合は無視する。
"""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_DEFAULT_VAULT_PATH = (
    "/Users/kobayashiisaoryou/Documents/Obsidian Vault/Projects/tune_lease_55/"
)
_VAULT_PATH = os.environ.get("OBSIDIAN_VAULT_PATH", _DEFAULT_VAULT_PATH)
_POLICY_FILENAME = "審査方針.md"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """frontmatter をパースしてメタデータと本文を返す。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text

    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()

    body = text[m.end():]
    return meta, body


def load_policy(vault_path: str = _VAULT_PATH) -> str:
    """
    {vault_path}/審査方針.md を読み込んでテキストを返す。

    - ファイルが存在しない場合は空文字を返す
    - frontmatter に active: false が設定されている場合は空文字を返す
    """
    policy_path = os.path.join(vault_path, _POLICY_FILENAME)
    if not os.path.exists(policy_path):
        logger.debug(f"[PolicyLoader] 審査方針.md not found: {policy_path}")
        return ""

    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.warning(f"[PolicyLoader] failed to read 審査方針.md: {e}")
        return ""

    meta, body = _parse_frontmatter(content)

    active = meta.get("active", "true").lower()
    if active == "false":
        logger.debug("[PolicyLoader] 審査方針.md is inactive (active: false), skipping")
        return ""

    text = body.strip()
    logger.info(f"[PolicyLoader] loaded policy ({len(text)} chars)")
    return text
