"""改善台帳の REV 番号採番に関する共通ユーティリティ。

台帳ファイルは複数存在する（JSON配列形式の api/rule_engine/ledger_rules.json、
JSONL形式の scripts/improvement_ledger.jsonl 等）。呼び出し元は自分の台帳を
読み込んで list[dict] にした上で max_rev_number() に渡すこと。

これまで analyze_error_logs.py / analyze_pipeline_health.py /
analyze_rag_staleness.py / analyze_report_quality.py / analyze_wizard_inputs.py
の5スクリプトが同じロジックを個別にコピペしており、新規スクリプト
（step1_extract_and_structure.py）がこの慣習に従わず REV 番号を毎回1から
振り直して既存 REV と衝突する事故を起こした。再発防止のため一本化する。
"""

from __future__ import annotations

import re

_REV_ID_RE = re.compile(r"REV-(\d+)")


def max_rev_number(entries: list[dict], fields: tuple[str, ...] = ("rev_id",)) -> int:
    """entries から REV-NNN 形式の最大番号を返す（無ければ0）。

    Args:
        entries: 台帳エントリの一覧（各要素は dict）。
        fields: REV番号を探すフィールド名（複数指定可）。
    """
    max_num = 0
    for entry in entries:
        for field in fields:
            m = _REV_ID_RE.search(str(entry.get(field) or ""))
            if m:
                max_num = max(max_num, int(m.group(1)))
    return max_num
