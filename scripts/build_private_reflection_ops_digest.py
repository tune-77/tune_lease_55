#!/usr/bin/env python3
"""Build an AI Agent Ops digest from Private Reflection notes.

The digest is intentionally not a raw export. It extracts the serious
reflection protocol and converts it into a safer, public-demo-friendly evidence
layer:

Raw Private Reflection -> Digest -> Public Demo Evidence
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
PRIVATE_REFLECTION_REL = (
    Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Private Reflection"
)

PROTOCOL_HEADING = "本格内省プロトコル"
REUSE_HEADING = "差分と再利用"
HARANMARU_HEADING = "波乱丸式の私室メモ"

PROTOCOL_FIELDS = [
    "事前の思い込み",
    "破られた前提",
    "私の責任",
    "まだ逃げていること",
    "更新する信念",
    "次回の検証方法",
]


def _vault_path() -> Path:
    raw = os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH") or str(DEFAULT_VAULT)
    return Path(raw).expanduser()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()


def _section(text: str, heading: str) -> str:
    match = re.search(rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _bullet_items(section_text: str) -> list[str]:
    items: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _field_items(section_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for item in _bullet_items(section_text):
        if ":" not in item and "：" not in item:
            continue
        label, value = re.split(r"[:：]", item, maxsplit=1)
        label = label.strip()
        value = value.strip()
        if label:
            fields[label] = value
    return fields


def _redact_public(text: str, limit: int = 220) -> str:
    clean = str(text or "").strip()
    clean = re.sub(r"`([^`]+)`", r"\1", clean)
    clean = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", clean)
    clean = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", clean)
    clean = re.sub(r"\b\d{6,}\b", "[number]", clean)
    clean = clean.replace("User", "ユーザー")
    clean = re.sub(r"\s+", " ", clean)
    if len(clean) > limit:
        return clean[: limit - 1] + "…"
    return clean


def _reflection_path(vault: Path, day: dt.date) -> Path:
    return vault / PRIVATE_REFLECTION_REL / f"{day.isoformat()}.md"


def _date_range(end_date: dt.date, days: int) -> list[dt.date]:
    return [end_date - dt.timedelta(days=offset) for offset in range(days)]


def _parse_quality(reuse_section: str) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    for item in _bullet_items(reuse_section):
        if item.startswith("品質ゲート:"):
            quality["quality_gate"] = item.split(":", 1)[1].strip()
        elif item.startswith("品質ゲート理由:"):
            quality["quality_reasons"] = item.split(":", 1)[1].strip()
        elif item.startswith("前日との差分類似度:"):
            raw = item.split(":", 1)[1].strip()
            try:
                quality["similarity_to_previous"] = float(raw)
            except ValueError:
                quality["similarity_to_previous"] = raw
        elif item.startswith("次回対話へ戻すこと:"):
            quality["next_context"] = item.split(":", 1)[1].strip()
    return quality


def _parse_reflection(path: Path, day: dt.date) -> dict[str, Any] | None:
    text = _strip_frontmatter(_read_text(path))
    if not text:
        return None
    protocol = _field_items(_section(text, PROTOCOL_HEADING))
    haranmaru = _field_items(_section(text, HARANMARU_HEADING))
    reuse = _section(text, REUSE_HEADING)
    quality = _parse_quality(reuse)
    missing = [field for field in PROTOCOL_FIELDS if not protocol.get(field)]
    return {
        "date": day.isoformat(),
        "path": str(path),
        "has_private_reflection": True,
        "protocol_complete": not missing,
        "missing_protocol_fields": missing,
        "protocol": {field: _redact_public(protocol.get(field, "")) for field in PROTOCOL_FIELDS},
        "haranmaru_lens": {
            key: _redact_public(value)
            for key, value in haranmaru.items()
            if key in {"場面", "摩擦", "ぼやき", "次の一手", "残す芯"}
        },
        "quality": quality,
    }


def build_digest(*, vault: Path, end_date: dt.date, days: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for day in _date_range(end_date, days):
        path = _reflection_path(vault, day)
        parsed = _parse_reflection(path, day)
        if parsed:
            records.append(parsed)
        else:
            records.append(
                {
                    "date": day.isoformat(),
                    "path": str(path),
                    "has_private_reflection": False,
                    "protocol_complete": False,
                    "missing_protocol_fields": PROTOCOL_FIELDS,
                    "protocol": {},
                    "haranmaru_lens": {},
                    "quality": {},
                }
            )
    records.sort(key=lambda item: item["date"])
    complete = sum(1 for record in records if record.get("protocol_complete"))
    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "start_date": records[0]["date"] if records else "",
        "end_date": records[-1]["date"] if records else "",
        "days": days,
        "records_total": len(records),
        "protocol_complete_count": complete,
        "protocol_complete_rate": round(complete / len(records), 4) if records else 0.0,
        "records": records,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _markdown(digest: dict[str, Any]) -> str:
    lines = [
        f"# Private Reflection Agent Ops Digest ({digest.get('start_date')} to {digest.get('end_date')})",
        "",
        "## Summary",
        "",
        f"- Records: {digest.get('records_total', 0)}",
        f"- Complete serious protocols: {digest.get('protocol_complete_count', 0)}",
        f"- Completion rate: {digest.get('protocol_complete_rate', 0)}",
        "",
        "## Public Demo Evidence",
        "",
        "This digest does not expose raw Private Reflection text. It extracts only the operational loop: prior assumption, broken premise, self-responsibility, updated belief, and verification method.",
        "",
    ]
    for record in digest.get("records", []):
        date = record.get("date", "")
        lines += [
            f"### {date}",
            "",
            f"- Protocol complete: {record.get('protocol_complete')}",
        ]
        quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}
        if quality.get("quality_gate"):
            lines.append(f"- Quality gate: {quality.get('quality_gate')}")
        protocol = record.get("protocol") if isinstance(record.get("protocol"), dict) else {}
        if protocol:
            lines += [
                f"- Prior assumption: {protocol.get('事前の思い込み', '')}",
                f"- Broken premise: {protocol.get('破られた前提', '')}",
                f"- Self-responsibility: {protocol.get('私の責任', '')}",
                f"- Updated belief: {protocol.get('更新する信念', '')}",
                f"- Verification: {protocol.get('次回の検証方法', '')}",
            ]
        haranmaru = record.get("haranmaru_lens") if isinstance(record.get("haranmaru_lens"), dict) else {}
        if haranmaru:
            lines += [
                f"- Friction lens: {haranmaru.get('摩擦', '')}",
                f"- Next move: {haranmaru.get('次の一手', '')}",
            ]
        missing = record.get("missing_protocol_fields") or []
        if missing:
            lines.append(f"- Missing: {', '.join(missing)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _public_demo_markdown(digest: dict[str, Any]) -> str:
    complete_records = [record for record in digest.get("records", []) if record.get("protocol_complete")]
    latest = complete_records[-1] if complete_records else {}
    protocol = latest.get("protocol") if isinstance(latest.get("protocol"), dict) else {}
    haranmaru = latest.get("haranmaru_lens") if isinstance(latest.get("haranmaru_lens"), dict) else {}
    lines = [
        "# AI Agent Ops Evidence: Reflection Loop",
        "",
        "## What This Shows",
        "",
        "This system does not only answer user questions. It records a private operational reflection, extracts a public-safe digest, and checks whether the next behavior should change.",
        "",
        "## Latest Loop",
        "",
        f"- Date: {latest.get('date', '')}",
        f"- Prior assumption: {protocol.get('事前の思い込み', '')}",
        f"- Broken premise: {protocol.get('破られた前提', '')}",
        f"- Self-responsibility: {protocol.get('私の責任', '')}",
        f"- Updated belief: {protocol.get('更新する信念', '')}",
        f"- Verification method: {protocol.get('次回の検証方法', '')}",
        "",
        "## Narrative Lens",
        "",
        f"- Friction: {haranmaru.get('摩擦', '')}",
        f"- Next move: {haranmaru.get('次の一手', '')}",
        "",
        "## Why It Matters",
        "",
        "For lease screening, the most valuable moment is often quiet: a human can explain a decision faster, with clearer risks and next checks. The reflection loop turns missed assumptions into the next operating rule.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_reports(digest: dict[str, Any], date_str: str) -> dict[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"private_reflection_ops_digest_{date_str.replace('-', '')}.json"
    md_path = REPORTS_DIR / f"private_reflection_ops_digest_{date_str.replace('-', '')}.md"
    public_path = REPORTS_DIR / f"private_reflection_ops_public_{date_str.replace('-', '')}.md"
    latest_json = REPORTS_DIR / "private_reflection_ops_digest_latest.json"
    latest_md = REPORTS_DIR / "private_reflection_ops_digest_latest.md"
    latest_public = REPORTS_DIR / "private_reflection_ops_public_latest.md"
    _write_json(json_path, digest)
    _write_json(latest_json, digest)
    md = _markdown(digest)
    public_md = _public_demo_markdown(digest)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")
    public_path.write_text(public_md, encoding="utf-8")
    latest_public.write_text(public_md, encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "public_markdown": str(public_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
        "latest_public_markdown": str(latest_public),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Private Reflection Agent Ops digest")
    parser.add_argument("--date", default=dt.date.today().isoformat(), help="End date YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--vault", default="", help="Obsidian Vault path")
    args = parser.parse_args()

    end_date = dt.date.fromisoformat(args.date)
    vault = Path(args.vault).expanduser() if args.vault else _vault_path()
    digest = build_digest(vault=vault, end_date=end_date, days=max(1, args.days))
    paths = write_reports(digest, args.date)
    print(json.dumps({"summary": {k: digest[k] for k in ("records_total", "protocol_complete_count", "protocol_complete_rate")}, "paths": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
