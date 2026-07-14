#!/usr/bin/env python3
"""Monitor the local Obsidian environment for morning operations.

This is an inspection-only monitor. It writes reports under this repository and
does not edit Obsidian notes, RAG stores, prompts, scoring, or Cloud Run state.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
DEFAULT_JSON = REPO_ROOT / "reports" / "obsidian_environment_monitor_latest.json"
DEFAULT_MD = REPO_ROOT / "reports" / "obsidian_environment_monitor_latest.md"

KEY_PATHS = {
    "daily": Path("Daily"),
    "private_reflection": Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Private Reflection",
    "dialogue": Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Dialogue",
    "cloudrun_conversation": Path("Projects") / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log",
    "research": Path("Projects") / "tune_lease_55" / "Research",
    "news": Path("Projects") / "tune_lease_55" / "News",
}
TECH_NOISE_RE = re.compile(
    r"(pytest|npm run|python -m|wrote=|report=|Traceback|ERROR|git\s+|curl\s+|node_modules|\.py:\d+)"
)
WIKILINK_RE = re.compile(r"\[\[([^\]#|]+)(?:[#|][^\]]*)?\]\]")
REINDEX_DONE_RE = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*\[reindex\] 完了.*total_in_db=(\d+)")
MAINTENANCE_DONE_RE = re.compile(
    r"実行時刻:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?ChromaDB:\s*success",
    re.DOTALL,
)
MAINTENANCE_DB_COUNT_RE = re.compile(r"LocalVectorDB:\s*(\d+)\s*件")
PRIVATE_REFLECTION_REQUIRED_TERMS = {
    "user_expectation": ("ユーザーが何を望", "ユーザーは何を求め", "User", "望んだ", "求めた"),
    "misread": ("すり替え", "誤読", "読み違え", "逃げ", "見落とし"),
    "self_responsibility": ("私の責任", "私の見落とし", "自分", "浅かった", "足りない"),
    "next_behavior": ("次回", "次に", "禁止", "検証", "変える", "更新する信念"),
}
SELF_REFERENCE_SOURCE_TERMS = (
    "Private Reflection",
    "Daily/",
    "Codex Work Log",
    "Claude Work Log",
    "obsidian_memory_insight",
    "shion_reflection_delta",
    "recursive_self_improvement",
)
SELF_REFERENCE_META_TERMS = (
    "品質ゲート",
    "監視",
    "レポート",
    "生成",
    "candidate",
    "latest.md",
    "latest.json",
    "score=",
    "wrote=",
    "report=",
    "内省差分",
    "Obsidian Memory Insight",
)
WIKILINK_MONITOR_SOURCE_EXCLUDE = (
    "Projects/tune_lease_55/Asset Knowledge/Promoted Knowledge.md",
    "Projects/tune_lease_55/検索語インデックス.md",
    "Projects/tune_lease_55/tune_lease_55 Wiki.md",
    "Projects/tune_lease_55/2026-05-13_all_file_ingest_index.md",
    "Projects/tune_lease_55/2026-05-13_リース審査AI_知識分解.md",
)


@dataclass
class MonitorCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any] | None = None


def _now() -> datetime:
    return datetime.now().astimezone()


def _vault_path(value: str | None = None) -> Path:
    raw = value or os.environ.get("OBSIDIAN_VAULT") or os.environ.get("OBSIDIAN_VAULT_PATH")
    return Path(raw).expanduser() if raw else DEFAULT_VAULT


def _age_hours(path: Path) -> float:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=_now().tzinfo)
    return (_now() - modified).total_seconds() / 3600


def _read_text(path: Path, limit: int = 100_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "ok"
    return "warn" if warn else "fail"


def check_vault(vault: Path) -> MonitorCheck:
    if not vault.exists():
        return MonitorCheck("vault", "fail", f"Vault not found: {vault}")
    if not (vault / ".obsidian").exists():
        return MonitorCheck("vault", "warn", f"Vault exists but .obsidian is missing: {vault}")
    md_count = sum(1 for _ in vault.rglob("*.md"))
    return MonitorCheck("vault", "ok", f"Vault reachable, markdown files={md_count}", {"md_count": md_count})


def check_key_paths(vault: Path) -> MonitorCheck:
    missing = [name for name, rel in KEY_PATHS.items() if not (vault / rel).exists()]
    status = "ok" if not missing else "warn"
    message = "all key paths exist" if not missing else f"missing key paths: {', '.join(missing)}"
    return MonitorCheck("key_paths", status, message, {"missing": missing})


def check_daily_notes(vault: Path, target: date) -> MonitorCheck:
    daily_dir = vault / "Daily"
    today = daily_dir / f"{target.isoformat()}.md"
    yesterday = daily_dir / f"{(target - timedelta(days=1)).isoformat()}.md"
    missing = [str(path.name) for path in (today, yesterday) if not path.exists()]
    status = "ok" if not missing else "warn"
    message = "today/yesterday daily notes exist" if not missing else f"missing daily notes: {', '.join(missing)}"
    return MonitorCheck("daily_notes", status, message, {"today": today.exists(), "yesterday": yesterday.exists()})


def check_surface_freshness(vault: Path, target: date, max_age_hours: int) -> MonitorCheck:
    results: dict[str, Any] = {}
    stale: list[str] = []
    for name in ("private_reflection", "cloudrun_conversation", "dialogue"):
        root = vault / KEY_PATHS[name]
        path = root / f"{target.isoformat()}.md"
        if not path.exists():
            path = root / f"{(target - timedelta(days=1)).isoformat()}.md"
        if not path.exists():
            results[name] = {"exists": False, "age_hours": None}
            stale.append(name)
            continue
        age = round(_age_hours(path), 1)
        results[name] = {"exists": True, "path": str(path), "age_hours": age}
        if age > max_age_hours:
            stale.append(name)
    status = "ok" if not stale else "warn"
    msg = "dialogue/reflection surfaces fresh" if not stale else f"stale or missing surfaces: {', '.join(stale)}"
    return MonitorCheck("surface_freshness", status, msg, results)


def _reflection_path(vault: Path, day: date) -> Path:
    return vault / KEY_PATHS["private_reflection"] / f"{day.isoformat()}.md"


def _reflection_body(text: str) -> str:
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL).strip()
    match = re.search(r"##\s*今日の対話について\s*\n(.*?)(?=\n##\s*差分と再利用|\Z)", text, flags=re.DOTALL)
    return (match.group(1).strip() if match else text)[:30_000]


def _reflection_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    normalized_left = re.sub(r"\s+", "", left)
    normalized_right = re.sub(r"\s+", "", right)
    return round(SequenceMatcher(None, normalized_left, normalized_right).ratio(), 3)


def check_private_reflection_meaning(vault: Path, target: date) -> MonitorCheck:
    today_path = _reflection_path(vault, target)
    yesterday_path = _reflection_path(vault, target - timedelta(days=1))
    if not today_path.exists():
        return MonitorCheck(
            "private_reflection_meaning",
            "warn",
            f"today Private Reflection missing: {today_path.name}",
            {"today_exists": False},
        )

    today_body = _reflection_body(_read_text(today_path, 80_000))
    yesterday_body = _reflection_body(_read_text(yesterday_path, 80_000)) if yesterday_path.exists() else ""
    similarity = _reflection_similarity(today_body, yesterday_body)
    missing_categories = [
        name
        for name, terms in PRIVATE_REFLECTION_REQUIRED_TERMS.items()
        if not any(term in today_body for term in terms)
    ]
    labels = (
        "今日の観察:",
        "私の見落とし:",
        "仮説の更新:",
        "次回の小さな実験:",
        "私の責任:",
        "更新する信念:",
        "次回の検証方法:",
    )
    matched_labels = [label for label in labels if label in today_body]
    problems: list[str] = []
    if len(today_body) < 500:
        problems.append("too_short")
    if similarity >= 0.82:
        problems.append(f"too_similar_to_yesterday:{similarity}")
    if missing_categories:
        problems.append(f"missing_meaning_categories:{','.join(missing_categories)}")
    if len(matched_labels) < 4:
        problems.append("reflection_protocol_labels_missing")

    status = "ok" if not problems else "warn"
    if status == "ok":
        message = "Private Reflection has meaningful update signals"
    else:
        message = "Private Reflection exists but meaningful update is weak: " + "; ".join(problems)
    return MonitorCheck(
        "private_reflection_meaning",
        status,
        message,
        {
            "today_path": str(today_path),
            "yesterday_path": str(yesterday_path),
            "today_length": len(today_body),
            "similarity_to_yesterday": similarity,
            "matched_labels": matched_labels,
            "missing_categories": missing_categories,
            "required_categories": sorted(PRIVATE_REFLECTION_REQUIRED_TERMS),
        },
    )


def check_reindex_and_chroma(max_age_hours: int) -> MonitorCheck:
    log_path = Path.home() / "Library" / "Logs" / "tune_lease_55_obsidian_reindex.out.log"
    chroma = REPO_ROOT / "api" / "chroma_db" / "chroma.sqlite3"
    details: dict[str, Any] = {"reindex_log": str(log_path), "chroma_db": str(chroma)}
    problems: list[str] = []
    if not log_path.exists():
        problems.append("missing reindex log")
    else:
        text = _read_text(log_path, 500_000)
        completions: list[tuple[datetime, str, int | None]] = []
        for ts_text, total_text in REINDEX_DONE_RE.findall(text):
            completions.append((
                datetime.fromisoformat(ts_text).replace(tzinfo=_now().tzinfo),
                "reindex_log",
                int(total_text),
            ))
        db_counts = MAINTENANCE_DB_COUNT_RE.findall(text)
        maintenance_total = int(db_counts[-1]) if db_counts else None
        for ts_text in MAINTENANCE_DONE_RE.findall(text):
            completions.append((
                datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_now().tzinfo),
                "rag_daily_maintenance",
                maintenance_total,
            ))
        if not completions:
            problems.append("no successful reindex completion")
        else:
            completed, source, total = max(completions, key=lambda item: item[0])
            details["completion_source"] = source
            if total is not None:
                details["total_in_db"] = total
            age = round((_now() - completed).total_seconds() / 3600, 1)
            details["last_reindex_age_hours"] = age
            if age > max_age_hours:
                problems.append(f"reindex stale: {age}h")
    if not chroma.exists():
        problems.append("missing chroma sqlite")
    else:
        age = round(_age_hours(chroma), 1)
        details["chroma_age_hours"] = age
        details["chroma_size"] = chroma.stat().st_size
        if age > max_age_hours:
            problems.append(f"chroma stale: {age}h")
        if chroma.stat().st_size <= 0:
            problems.append("chroma sqlite empty")
    status = "ok" if not problems else "warn"
    return MonitorCheck("rag_index", status, "RAG index fresh" if not problems else "; ".join(problems), details)


def check_memory_insight_reports(max_age_hours: int) -> MonitorCheck:
    paths = {
        "reflection_delta": REPO_ROOT / "reports" / "shion_reflection_delta_latest.md",
        "memory_insight": REPO_ROOT / "reports" / "obsidian_memory_insight_latest.md",
        "promotion_queue": REPO_ROOT / "reports" / "shion_memory_promotion_queue_latest.md",
    }
    stale: list[str] = []
    details: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.exists():
            stale.append(name)
            details[name] = {"exists": False}
            continue
        age = round(_age_hours(path), 1)
        details[name] = {"exists": True, "age_hours": age, "path": str(path)}
        if age > max_age_hours:
            stale.append(name)
    status = "ok" if not stale else "warn"
    message = "memory insight sidecars fresh" if not stale else f"stale or missing sidecars: {', '.join(stale)}"
    return MonitorCheck("memory_insight_reports", status, message, details)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _normalized_claim(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or "").lower())
    text = re.sub(r"\d+", "0", text)
    return text[:120]


def check_self_reference_loop() -> MonitorCheck:
    candidates_path = REPO_ROOT / "data" / "obsidian_memory_insight_candidates.jsonl"
    rows = _load_jsonl(candidates_path)
    if not rows:
        return MonitorCheck(
            "self_reference_loop",
            "warn",
            "memory insight candidates are missing; worm guard cannot evaluate",
            {"candidate_path": str(candidates_path), "candidate_count": 0},
        )

    total = len(rows)
    source_hits: list[dict[str, str]] = []
    meta_hits: list[dict[str, str]] = []
    claim_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_path") or "")
        claim = str(row.get("claim") or "")
        ctype = str(row.get("candidate_type") or "unknown")
        source_type_counts[ctype] = source_type_counts.get(ctype, 0) + 1
        if any(term in source for term in SELF_REFERENCE_SOURCE_TERMS):
            source_hits.append({"source": source, "claim": claim[:120]})
        if any(term in claim or term in source for term in SELF_REFERENCE_META_TERMS):
            meta_hits.append({"source": source, "claim": claim[:120]})
        key = _normalized_claim(claim)
        if key:
            claim_counts[key] = claim_counts.get(key, 0) + 1

    repeated = sum(1 for count in claim_counts.values() if count >= 3)
    source_ratio = round(len(source_hits) / total, 4) if total else 0.0
    meta_ratio = round(len(meta_hits) / total, 4) if total else 0.0
    problems: list[str] = []
    if source_ratio >= 0.35:
        problems.append(f"self_generated_source_ratio_high:{source_ratio}")
    if meta_ratio >= 0.12:
        problems.append(f"meta_term_ratio_high:{meta_ratio}")
    if repeated >= 3:
        problems.append(f"repeated_claim_groups:{repeated}")
    if source_type_counts.get("noise", 0) / total >= 0.35:
        problems.append("noise_candidates_high")

    status = "ok" if not problems else "warn"
    message = "no obvious self-reference loop in memory candidates" if not problems else "possible self-reference loop: " + "; ".join(problems)
    return MonitorCheck(
        "self_reference_loop",
        status,
        message,
        {
            "candidate_path": str(candidates_path),
            "candidate_count": total,
            "self_generated_source_ratio": source_ratio,
            "meta_term_ratio": meta_ratio,
            "repeated_claim_groups": repeated,
            "candidate_type_counts": source_type_counts,
            "source_hit_sample": source_hits[:8],
            "meta_hit_sample": meta_hits[:8],
        },
    )


def recent_notes(vault: Path, max_age_hours: int, limit: int = 120) -> list[Path]:
    cutoff = _now() - timedelta(hours=max_age_hours)
    notes: list[Path] = []
    for path in vault.rglob("*.md"):
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=_now().tzinfo)
        except OSError:
            continue
        if modified >= cutoff:
            notes.append(path)
    notes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return notes[:limit]


def check_recent_note_noise(vault: Path) -> MonitorCheck:
    notes = recent_notes(vault, 72)
    total_lines = 0
    noisy_lines = 0
    noisy_files: list[str] = []
    for path in notes:
        text = _read_text(path, 80_000)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        file_noisy = sum(1 for line in lines if TECH_NOISE_RE.search(line))
        total_lines += len(lines)
        noisy_lines += file_noisy
        if file_noisy >= 3:
            try:
                noisy_files.append(str(path.relative_to(vault)))
            except ValueError:
                noisy_files.append(str(path))
    ratio = round(noisy_lines / total_lines, 4) if total_lines else 0.0
    warn = ratio > 0.08 or len(noisy_files) >= 5
    msg = f"recent note technical-noise ratio={ratio}, noisy_files={len(noisy_files)}"
    return MonitorCheck("recent_note_noise", "warn" if warn else "ok", msg, {"ratio": ratio, "noisy_files": noisy_files[:12]})


def _normalize_link_target(value: str) -> str:
    cleaned = value.strip().replace("\\", "/").strip("/")
    return re.sub(r"/+", "/", cleaned)


def _link_candidates(value: str) -> set[str]:
    cleaned = _normalize_link_target(value)
    if not cleaned:
        return {""}
    candidates = {cleaned}
    if cleaned.lower().endswith(".md"):
        candidates.add(cleaned[:-3])
    else:
        candidates.add(f"{cleaned}.md")
    return candidates


def _existing_wikilink_targets(vault: Path) -> set[str]:
    targets: set[str] = set()
    for path in vault.rglob("*"):
        try:
            rel = path.relative_to(vault)
        except ValueError:
            continue
        rel_text = _normalize_link_target(str(rel))
        if not rel_text:
            continue
        targets.add(rel_text)
        targets.add(path.name)
        if path.is_dir():
            targets.add(f"{rel_text}/")
            continue
        targets.add(path.stem)
        if path.suffix.lower() == ".md":
            targets.add(_normalize_link_target(str(rel.with_suffix(""))))
    return targets


def _iter_wikilink_targets(text: str) -> list[str]:
    """Extract wikilink targets using bracket delimiters instead of a fragile regex."""
    targets: list[str] = []
    pos = 0
    while True:
        start = text.find("[[", pos)
        if start < 0:
            break
        end = text.find("]]", start + 2)
        if end < 0:
            break
        raw = text[start + 2:end]
        target = re.split(r"[#|]", raw, maxsplit=1)[0].strip()
        if target:
            targets.append(target)
        pos = end + 2
    return targets


def _wikilink_exists(vault: Path, source_path: Path, target: str, existing_targets: set[str]) -> bool:
    cleaned = _normalize_link_target(target)
    if not cleaned:
        return True
    candidates = _link_candidates(cleaned)
    try:
        source_rel_parent = source_path.relative_to(vault).parent
        for value in list(candidates):
            candidates.update(_link_candidates(str(source_rel_parent / value)))
    except ValueError:
        pass
    for candidate in candidates:
        if candidate in existing_targets:
            return True
    return False


def check_wikilinks(vault: Path) -> MonitorCheck:
    notes = recent_notes(vault, 7 * 24)
    existing_targets = _existing_wikilink_targets(vault)
    unresolved: list[dict[str, str]] = []
    link_count = 0
    for path in notes:
        try:
            source_rel = str(path.relative_to(vault))
        except ValueError:
            source_rel = str(path)
        if source_rel in WIKILINK_MONITOR_SOURCE_EXCLUDE:
            continue
        text = _read_text(path, 80_000)
        for target in _iter_wikilink_targets(text):
            cleaned = target.strip()
            if not cleaned:
                continue
            link_count += 1
            if not _wikilink_exists(vault, path, cleaned, existing_targets):
                unresolved.append({"source": source_rel, "target": cleaned})
                if len(unresolved) >= 20:
                    break
        if len(unresolved) >= 20:
            break
    status = "ok" if len(unresolved) < 10 else "warn"
    msg = f"recent wikilinks={link_count}, unresolved_sample={len(unresolved)}"
    return MonitorCheck("wikilinks", status, msg, {"link_count": link_count, "unresolved_sample": unresolved})


def run_monitor(vault: Path, target: date) -> dict[str, Any]:
    checks = [
        check_vault(vault),
        check_key_paths(vault),
        check_daily_notes(vault, target),
        check_surface_freshness(vault, target, max_age_hours=48),
        check_private_reflection_meaning(vault, target),
        check_reindex_and_chroma(max_age_hours=36),
        check_memory_insight_reports(max_age_hours=36),
        check_self_reference_loop(),
        check_recent_note_noise(vault),
        check_wikilinks(vault),
    ]
    status_order = {"fail": 2, "warn": 1, "ok": 0}
    worst = max((check.status for check in checks), key=lambda value: status_order[value], default="ok")
    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "target_date": target.isoformat(),
        "vault": str(vault),
        "status": worst,
        "checks": [asdict(check) for check in checks],
        "guardrail": "monitor_only_no_obsidian_write_no_rag_no_prompt_no_cloudrun",
        "recommended_viewpoints": [
            "鮮度: 今日/昨日のDaily・対話・Private Reflectionが揃っているか",
            "内省品質: Private Reflectionが昨日と違い、User要求・誤読・自己責任・次回行動を含むか",
            "同期: Cloud Run会話ログがObsidianへ戻っているか",
            "検索性: reindex/ChromaDBが古くないか",
            "記憶形成: 内省差分・記憶候補・Obsidian insightが生成されているか",
            "ワーム化防止: 自分のレポート・内省・Daily作業ログを材料に候補が増殖していないか",
            "ノイズ: 技術ログや一時出力が知識ノートを汚していないか",
            "リンク: 直近ノートのwikilinkが解決できるか",
            "安全性: 監視は読み取り専用で、本番・Cloud Run・RAGに接続しない",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Obsidian Environment Monitor",
        "",
        "## Summary",
        f"- generated_at: `{report['generated_at']}`",
        f"- target_date: `{report['target_date']}`",
        f"- status: `{report['status']}`",
        f"- guardrail: `{report['guardrail']}`",
        "",
        "## Viewpoints",
        *[f"- {item}" for item in report.get("recommended_viewpoints", [])],
        "",
        "## Checks",
    ]
    for check in report.get("checks", []):
        lines.append(f"### {check['name']}")
        lines.append(f"- status: `{check['status']}`")
        lines.append(f"- message: {check['message']}")
        details = check.get("details")
        if details:
            compact = json.dumps(details, ensure_ascii=False, sort_keys=True)
            if len(compact) > 900:
                compact = compact[:897] + "..."
            lines.append(f"- details: `{compact}`")
        lines.append("")
    lines.extend(
        [
            "## Next Safe Action",
            "- `warn` が出た項目だけ手動で確認する。",
            "- 監視結果をRAGやチャットへ自動注入しない。まず3日分を比較して、警告が実際に役立つか見る。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")


def _parse_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor Obsidian environment for morning reporting.")
    parser.add_argument("--vault", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = run_monitor(_vault_path(args.vault), _parse_date(args.date))
    if args.dry_run:
        print(render_markdown(report))
        return 0
    write_reports(report, args.json, args.report)
    print(f"json={args.json}")
    print(f"report={args.report}")
    print(f"status={report['status']}")
    return 0 if report["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
