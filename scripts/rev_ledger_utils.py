"""改善台帳の REV 番号採番に関する共通ユーティリティ。

台帳ファイルは複数存在する（JSON配列形式の api/rule_engine/ledger_rules.json、
JSONL形式の scripts/improvement_ledger.jsonl 等）。この2ファイルは互いの
存在を知らずに運用されており、片方にしか記録されていない番号を「空き番号」と
誤認して衝突する事故が発生した（REV-292 が異なる2つの改善案に重複発行された
ケースなど）。採番時は load_all_rev_sources() で両方を横断して読み込んだ上で
max_rev_number() に渡すこと。個別に自分の台帳だけを読んで +1 してはならない。

これまで analyze_error_logs.py / analyze_pipeline_health.py /
analyze_rag_staleness.py / analyze_report_quality.py / analyze_wizard_inputs.py
の5スクリプトが ledger_rules.json だけを見て同じロジックを個別にコピペしており、
新規スクリプト（step1_extract_and_structure.py）は improvement_ledger.jsonl
だけを見ていた。6スクリプトが同じ REV 番号空間を2つの独立した台帳から
別々に採番していたため、両者を横断しない限り衝突は避けられない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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


def load_all_rev_sources(repo_root: Path | None = None) -> list[dict]:
    """REV番号を発行しうる全台帳（ledger_rules.json + improvement_ledger.jsonl）を
    横断して読み込み、1つのリストに合算して返す。

    採番前の唯一の窓口として、6スクリプト全てがここを経由する。片方の台帳しか
    見ずに max_rev_number() を呼ぶと、もう片方に記録済みの番号と衝突する。
    """
    root = repo_root or Path(__file__).resolve().parents[1]
    entries: list[dict] = []

    ledger_rules_path = root / "api" / "rule_engine" / "ledger_rules.json"
    if ledger_rules_path.exists():
        try:
            data = json.loads(ledger_rules_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = []
        if isinstance(data, list):
            entries.extend(e for e in data if isinstance(e, dict))

    improvement_ledger_path = root / "scripts" / "improvement_ledger.jsonl"
    if improvement_ledger_path.exists():
        for line in improvement_ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)

    return entries
