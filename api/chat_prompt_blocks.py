"""Small prompt assembly helpers for `/api/chat`.

The chat route still decides which blocks are active. This module only keeps
string joining consistent and easy to audit.
"""

from __future__ import annotations

from typing import Iterable


def join_prompt_blocks(blocks: Iterable[str]) -> str:
    return "".join(str(block or "") for block in blocks)


def block_with_spacing(block: str) -> str:
    text = str(block or "")
    return f"\n\n{text}" if text else ""


def append_optional_block(prompt: str, block: str) -> str:
    return str(prompt or "") + block_with_spacing(block)
