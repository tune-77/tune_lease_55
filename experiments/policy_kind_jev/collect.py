#!/usr/bin/env python3
"""ユーザーが教えた判断資産・Knowledge を文単位に分けて、方針/知見ラベル付け用の一覧を作る。

入力（いずれもコミットしない data/・Vault）:
  data/canonical_judgment_rules.json の status=active（全件ユーザー根拠つき・レビューゲート通過）の canonical_statement
  Vault の Lease Intelligence/Knowledge/*.md のうち source_type: "chat_teaching" の本文
  data/autoresearch_judgment_asset_candidates.jsonl の research_topic=chat_judgment_teaching
出力: data/policy_kind_items_20261004.json（label は null。ラベルは手で付ける）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime_paths import get_data_dir, resolve_obsidian_vault  # noqa: E402

DATA = get_data_dir()  # worktree から本番の data/ を読む時は DATA_DIR を渡す
OUT = DATA / "policy_kind_items_20261004.json"
_SAME_RE = re.compile(r"[（(]同旨[:：]\s*(.*?)[）)]\s*$", re.S)


def _sentences(text: str) -> list[str]:
    return [s.strip(" 　") for s in re.split(r"[。！？\n]", text) if len(s.strip(" 　")) >= 6]


def _chat_clauses(text: str) -> list[str]:
    """句読点のない口頭の教示を空白で節に分け、短い断片（「車両や重機」等）は次の節へつなぐ。"""
    out: list[str] = []
    carry = ""
    for part in re.split(r"[\s　。！？]+", text):
        part = (carry + " " + part).strip() if carry else part
        if len(part) < 8:
            carry = part
            continue
        out.append(part)
        carry = ""
    if carry:
        if out:
            out[-1] += " " + carry
        else:
            out.append(carry)
    return out


def collect() -> list[dict]:
    items: list[dict] = []
    rules = json.loads((DATA / "canonical_judgment_rules.json").read_text(encoding="utf-8"))["rules"]
    for rule in rules:
        if rule.get("status") != "active":
            continue
        statement = str(rule.get("canonical_statement") or "")
        same = _SAME_RE.search(statement)
        parts = [statement[: same.start()], same.group(1)] if same else [statement]
        for part in parts:
            for s in _sentences(part):
                items.append({"source": "canonical", "source_id": rule["id"], "current_kind": rule.get("knowledge_kind"), "text": s})
    knowledge = resolve_obsidian_vault() / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Knowledge"
    for path in sorted(knowledge.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        if 'source_type: "chat_teaching"' not in raw:
            continue
        body = raw.split("---", 2)[-1]
        body = "\n".join(l for l in body.splitlines() if l.strip() and not l.startswith(("#", ">")))
        for s in _chat_clauses(body):
            items.append({"source": "knowledge_chat", "source_id": path.name, "current_kind": None, "text": s})
    for line in (DATA / "autoresearch_judgment_asset_candidates.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("research_topic") == "chat_judgment_teaching":
            for s in _chat_clauses(str(row.get("edited_claim") or row.get("claim") or "")):
                items.append({"source": "candidate_chat", "source_id": row["id"], "current_kind": row.get("knowledge_kind"), "text": s})
    seen: set[str] = set()
    unique = []
    for item in items:
        key = re.sub(r"\s+", "", item["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append({"n": len(unique) + 1, **item, "label": None})
    return unique


if __name__ == "__main__":
    items = collect()
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(len(items), OUT)
