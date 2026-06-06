"""
合成したパターンを Obsidian の Generated/ サブディレクトリに書き出す。
"""
from __future__ import annotations

import datetime
import json
import os
import re

from api.crystallizer.anomaly_extractor import AnomalyCase
from runtime_paths import get_obsidian_vault_path

_VAULT_ROOT = get_obsidian_vault_path()
_GENERATED_DIR = "Generated"


def _safe_filename(text: str, max_len: int = 40) -> str:
    """ファイル名に使えない文字を除去して安全なファイル名を生成する。"""
    cleaned = re.sub(r'[\\/:*?"<>|\n\r\t]', "_", text)
    cleaned = cleaned.strip("_").strip()
    return cleaned[:max_len] if cleaned else "pattern"


def write_pattern_to_obsidian(
    pattern_text: str,
    cases: list[AnomalyCase],
    vault_root: str = _VAULT_ROOT,
    generated_at: datetime.date | None = None,
) -> str | None:
    """
    パターンテキストを Obsidian の Generated/ ディレクトリに書き出す。

    Args:
        pattern_text:  pattern_synthesizer.synthesize_pattern() の出力
        cases:         根拠となった案件リスト
        vault_root:    Obsidian Vault のルートパス
        generated_at:  生成日（None なら今日）

    Returns:
        書き出したファイルパス（失敗時は None）
    """
    if not vault_root or not os.path.isdir(vault_root):
        return None

    generated_at = generated_at or datetime.date.today()
    out_dir = os.path.join(vault_root, _GENERATED_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # 業種リスト（重複排除）
    industries = list(dict.fromkeys(c.industry for c in cases if c.industry != "不明"))
    industry_str = industries[0] if industries else "一般"

    # 証拠案件 ID リスト
    evidence_ids = [c.case_id for c in cases]

    # パターン名を抽出（"パターン名: ..." が含まれる場合）
    pattern_name = "自動合成パターン"
    for line in pattern_text.splitlines():
        if line.startswith("パターン名:") or line.startswith("パターン名："):
            name_candidate = line.split(":", 1)[-1].strip().split("：", 1)[-1].strip()
            if name_candidate:
                pattern_name = name_candidate
            break

    # ファイル名: YYYY-MM-DD_パターン名.md
    date_str = generated_at.strftime("%Y-%m-%d")
    fname = f"{date_str}_{_safe_filename(pattern_name)}.md"
    fpath = os.path.join(out_dir, fname)

    # frontmatter + 本文（evidence_records はYAML準拠のリスト形式）
    evidence_yaml = "[" + ", ".join(json.dumps(str(eid)) for eid in evidence_ids) + "]"
    content = f"""---
generated_by: crystallizer
generated_at: {date_str}
evidence_records: {evidence_yaml}
industry: {industry_str}
---

# {pattern_name}

{pattern_text}

## 根拠案件

| ID | 業種 | スコア | 判定 | 理由 |
|----|------|--------|------|------|
"""
    for c in cases:
        content += f"| {c.case_id} | {c.industry} | {c.score:.1f} | {c.judgment} | {c.reason} |\n"

    try:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        return fpath
    except Exception:
        return None
