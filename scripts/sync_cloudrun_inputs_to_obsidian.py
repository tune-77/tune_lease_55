#!/usr/bin/env python3
"""Materialize Cloud Run input events into the normal Obsidian Vault.

Cloud Run cannot write to the local iCloud Vault directly.  The intended flow is:

1. Cloud Run appends redacted events to GCS.
2. scripts/sync_cloudrun_inputs_from_gcs.py downloads them to data/cloudrun_inputs/.
3. This script writes a compact daily summary into the local Obsidian Vault.

The script is intentionally summary-first. It avoids copying full payloads into
Obsidian and uses event IDs for traceability back to the local JSONL archive.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_INPUT_DIR = Path(os.environ.get("LOCAL_CLOUDRUN_INPUT_DIR", PROJECT_ROOT / "data" / "cloudrun_inputs"))
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
OBSIDIAN_VAULT = Path(os.environ.get("OBSIDIAN_VAULT", str(DEFAULT_VAULT))).expanduser()
OUTPUT_SUBDIR = Path("Projects") / "tune_lease_55" / "Cloud Run Inputs"
IMPROVEMENT_LOG_SUBDIR = Path("Projects") / "tune_lease_55" / "AI Chat" / "Improvement Log"
CHAT_LOG_SUBDIR = Path("Projects") / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log"
DIALOGUE_LOG_SUBDIR = Path("Projects") / "tune_lease_55" / "Lease Intelligence" / "Dialogue"
DAILY_SUBDIR = Path("Daily")
DAILY_SECTION_START = "<!-- cloudrun-input-sync:start -->"
DAILY_SECTION_END = "<!-- cloudrun-input-sync:end -->"
IMPROVEMENT_EVENT_MARKER_PREFIX = "cloudrun-improvement-event:"
CHAT_EVENT_MARKER_PREFIX = "cloudrun-chat-event:"
DIALOGUE_EVENT_MARKER_PREFIX = "cloudrun-dialogue-event:"
JST = timezone(timedelta(hours=9))


EVENT_LABELS = {
    "score_calculated": "スコア計算",
    "score_full_calculated": "フルスコア計算",
    "case_result_registered": "案件結果登録",
    "rag_feedback": "RAGフィードバック",
    "improvement_note": "改善メモ",
    "chat_exchange": "会話ログ",
    "shion_memory_usage": "紫苑メモリ利用",
    "shion_screening_review": "紫苑審査レビュー",
    "shion_screening_review_feedback": "紫苑審査レビューFB",
    "screening_loop_feedback": "審査ループFB",
    "lease_news_judgment_change": "ニュース起点の判断変更",
    "judgment_feedback_created": "判断フィードバック",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _date_range(days: int) -> Iterable[date]:
    today = datetime.now().date()
    for offset in range(max(1, days)):
        yield today - timedelta(days=offset)


def _local_event_file(day: date) -> Path:
    return LOCAL_INPUT_DIR / f"{day.isoformat()}.jsonl"


def _parse_event_ts(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _event_jst_date(event: dict[str, Any], fallback: date) -> date:
    parsed = _parse_event_ts(event.get("ts"))
    if not parsed:
        return fallback
    return parsed.astimezone(JST).date()


def _event_jst_datetime(event: dict[str, Any], fallback: date) -> datetime:
    parsed = _parse_event_ts(event.get("ts"))
    if not parsed:
        return datetime.combine(fallback, datetime.min.time(), JST)
    return parsed.astimezone(JST)


def _load_events_by_jst_day(scan_days: Iterable[date]) -> dict[date, list[dict[str, Any]]]:
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for file_day in scan_days:
        for event in _load_jsonl(_local_event_file(file_day)):
            key = str(event.get("event_id") or json.dumps(event, ensure_ascii=False, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            grouped[_event_jst_date(event, file_day)].append(event)
    for events in grouped.values():
        events.sort(key=lambda event: str(event.get("ts") or ""))
    return grouped


def _safe_text(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _payload_summary(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    event_type = str(event.get("event_type") or "")

    if event_type == "score_calculated":
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        score = payload.get("score") or result.get("score")
        fields = [k for k, v in inputs.items() if v and str(v) != "[REDACTED]"]
        return f"入力項目 {len(fields)} 件" + (f" / score={score}" if score is not None else "")

    if event_type == "rag_feedback":
        ref = payload.get("obsidian_ref") or payload.get("doc_id") or ""
        rating = payload.get("rating") or ""
        return f"rating={_safe_text(rating, 40)} / ref={_safe_text(ref, 80)}"

    if event_type == "improvement_note":
        title = payload.get("title") or "改善メモ"
        body = str(payload.get("body") or "")
        first_line = next((line.strip() for line in body.splitlines() if line.strip()), "")
        return f"{_safe_text(title, 60)} / {_safe_text(first_line, 120)}"

    if event_type == "chat_exchange":
        user_msg = payload.get("user_message") or ""
        category = payload.get("category") or event.get("surface") or ""
        return f"{_safe_text(category, 40)} / {_safe_text(user_msg, 120)}"

    if event_type == "case_result_registered":
        case_id = payload.get("case_id") or payload.get("id") or ""
        status = payload.get("status") or payload.get("result") or ""
        return f"case={_safe_text(case_id, 60)} / result={_safe_text(status, 60)}"

    if event_type == "lease_news_judgment_change":
        action = payload.get("action") or payload.get("judgment_change") or payload.get("reason") or ""
        return _safe_text(action, 120)

    if event_type == "judgment_feedback_created":
        rating = payload.get("rating") or payload.get("action") or ""
        return _safe_text(rating, 120)

    keys = [str(k) for k in payload.keys()][:5]
    return "payload keys: " + ", ".join(keys) if keys else "payloadなし"


def _normalize_improvement_body(body: str) -> str:
    clean = (body or "").strip()
    if not clean:
        return ""
    if "## 原文" in clean or "## 抽出された改善候補" in clean:
        return clean
    return "\n".join(
        [
            "## 原文",
            clean,
            "",
            "## AI整理",
            "- 課題: 未整理。Cloud Run入力イベントの原文をレビューしてください。",
            "- 改善案: 原文を確認して改善候補に分解する。",
            "- 優先度: medium",
            "- 次の行動: 改善抽出パイプラインでレビューする。",
        ]
    )


def _build_improvement_section(event: dict[str, Any], fallback_day: date) -> str | None:
    if event.get("event_type") != "improvement_note":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    title = _safe_text(payload.get("title") or "Cloud Run改善メモ", 80)
    body = _normalize_improvement_body(str(payload.get("body") or ""))
    if not body:
        return None
    event_id = _safe_text(event.get("event_id"), 80)
    surface = _safe_text(event.get("surface") or "chat_improvement", 80)
    event_time = _event_jst_datetime(event, fallback_day)
    marker = f"<!-- {IMPROVEMENT_EVENT_MARKER_PREFIX}{event_id} -->"
    lines = [
        marker,
        f"## {event_time.strftime('%H:%M')} {title}",
        "",
        "### 要点",
        body,
        "",
        "## 受付",
        "- Cloud Run入力同期から登録",
        f"- event_id: `{event_id}`",
        f"- surface: `{surface}`",
        f"- source_ts: `{_safe_text(event.get('ts'), 48)}`",
        "",
    ]
    return "\n".join(lines)


def _write_improvement_logs(vault: Path, day: date, events: list[dict[str, Any]], dry_run: bool) -> int:
    improvement_events = [event for event in events if event.get("event_type") == "improvement_note"]
    if not improvement_events:
        return 0

    rel = IMPROVEMENT_LOG_SUBDIR / f"{day.isoformat()}.md"
    path = vault / rel
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    sections: list[str] = []
    for event in improvement_events:
        event_id = str(event.get("event_id") or "")
        if event_id and f"{IMPROVEMENT_EVENT_MARKER_PREFIX}{event_id}" in current:
            continue
        section = _build_improvement_section(event, day)
        if section:
            sections.append(section)

    if not sections:
        return 0
    if dry_run:
        print(f"[dry-run] {path} improvement-events={len(sections)}")
        return len(sections)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not current.strip():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        current = f"---\ndate: {timestamp}\ntags: [チャット, 改善メモ, cloudrun]\n---\n"
    path.write_text(current.rstrip() + "\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return len(sections)


def _chat_text(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _build_chat_section(event: dict[str, Any], fallback_day: date) -> str | None:
    if event.get("event_type") != "chat_exchange":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    user_message = _chat_text(payload.get("user_message"), 1200)
    assistant_reply = _chat_text(payload.get("assistant_reply"), 1800)
    if not user_message and not assistant_reply:
        return None
    event_id = _safe_text(event.get("event_id"), 80)
    surface = _safe_text(event.get("surface"), 80)
    category = _safe_text(payload.get("category"), 80)
    response_mode = _safe_text(payload.get("response_mode"), 40)
    user_id = _safe_text(payload.get("user_id"), 80)
    event_time = _event_jst_datetime(event, fallback_day)
    marker = f"<!-- {CHAT_EVENT_MARKER_PREFIX}{event_id} -->"
    lines = [
        marker,
        f"## {event_time.strftime('%H:%M')} {surface or 'Cloud Run会話'}",
        "",
        f"- user_id: `{user_id}`",
        f"- category: `{category}`",
        f"- response_mode: `{response_mode}`",
        f"- source_ts: `{_safe_text(event.get('ts'), 48)}`",
        "",
        "### User",
        user_message or "（空）",
        "",
        "### Assistant",
        assistant_reply or "（空）",
        "",
    ]
    return "\n".join(lines)


def _write_chat_logs(vault: Path, day: date, events: list[dict[str, Any]], dry_run: bool) -> int:
    chat_events = [event for event in events if event.get("event_type") == "chat_exchange"]
    if not chat_events:
        return 0

    rel = CHAT_LOG_SUBDIR / f"{day.isoformat()}.md"
    path = vault / rel
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    sections: list[str] = []
    for event in chat_events:
        event_id = str(event.get("event_id") or "")
        if event_id and f"{CHAT_EVENT_MARKER_PREFIX}{event_id}" in current:
            continue
        section = _build_chat_section(event, day)
        if section:
            sections.append(section)

    if not sections:
        return 0
    if dry_run:
        print(f"[dry-run] {path} chat-events={len(sections)}")
        return len(sections)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not current.strip():
        current = "\n".join(
            [
                "---",
                f"date: {day.isoformat()}",
                "tags: [チャット, cloudrun, 会話ログ]",
                "source: cloudrun_input_writeback",
                "summary_only: true",
                "---",
            ]
        )
    path.write_text(current.rstrip() + "\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return len(sections)


def _build_dialogue_section(event: dict[str, Any], fallback_day: date) -> str | None:
    if event.get("event_type") != "chat_exchange":
        return None
    if str(event.get("surface") or "") != "lease_intelligence_dialogue":
        return None
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    user_message = _chat_text(payload.get("user_message"), 1200)
    assistant_reply = _chat_text(payload.get("assistant_reply"), 1800)
    if not user_message and not assistant_reply:
        return None
    event_id = _safe_text(event.get("event_id"), 80)
    event_time = _event_jst_datetime(event, fallback_day)
    marker = f"<!-- {DIALOGUE_EVENT_MARKER_PREFIX}{event_id} -->"
    return "\n".join(
        [
            marker,
            f"## {event_time.strftime('%H:%M:%S')}",
            "",
            "**ユーザー**",
            "",
            user_message or "（空）",
            "",
            "**リース知性体**",
            "",
            assistant_reply or "（空）",
            "",
            f"source_ts: `{_safe_text(event.get('ts'), 48)}`",
            "",
        ]
    )


def _write_dialogue_logs(vault: Path, day: date, events: list[dict[str, Any]], dry_run: bool) -> int:
    dialogue_events = [
        event
        for event in events
        if event.get("event_type") == "chat_exchange"
        and str(event.get("surface") or "") == "lease_intelligence_dialogue"
    ]
    if not dialogue_events:
        return 0

    rel = DIALOGUE_LOG_SUBDIR / f"{day.isoformat()}.md"
    path = vault / rel
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    sections: list[str] = []
    for event in dialogue_events:
        event_id = str(event.get("event_id") or "")
        if event_id and f"{DIALOGUE_EVENT_MARKER_PREFIX}{event_id}" in current:
            continue
        section = _build_dialogue_section(event, day)
        if section:
            sections.append(section)

    if not sections:
        return 0
    if dry_run:
        print(f"[dry-run] {path} dialogue-events={len(sections)}")
        return len(sections)

    path.parent.mkdir(parents=True, exist_ok=True)
    if not current.strip():
        current = "\n".join(
            [
                "---",
                f"date: {day.isoformat()}",
                "type: lease_intelligence_dialogue",
                "source: cloudrun_input_writeback",
                "---",
                "",
                f"# リース知性体との対話 — {day.isoformat()}",
            ]
        )
    path.write_text(current.rstrip() + "\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return len(sections)


def _build_markdown(day: str, events: list[dict[str, Any]]) -> str:
    counts = Counter(str(event.get("event_type") or "unknown") for event in events)
    surfaces = Counter(str(event.get("surface") or "unknown") for event in events)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_type[str(event.get("event_type") or "unknown")].append(event)

    lines = [
        "---",
        f"date: {day}",
        "source: cloudrun_input_writeback",
        "tags: [cloudrun, obsidian_sync, 紫苑, 入力ログ]",
        f"total_events: {len(events)}",
        "---",
        "",
        f"# {day} Cloud Run 入力同期",
        "",
        "## サマリー",
        "",
        f"- 取り込みイベント: {len(events)} 件",
        f"- 入力元surface: {', '.join(f'{k} {v}件' for k, v in surfaces.most_common()) or 'なし'}",
        "",
        "## 種別",
        "",
    ]

    for event_type, count in counts.most_common():
        label = EVENT_LABELS.get(event_type, event_type)
        lines.append(f"- {label}: {count} 件")

    lines += ["", "## 詳細", ""]
    for event_type, items in sorted(by_type.items()):
        label = EVENT_LABELS.get(event_type, event_type)
        lines += [f"### {label}", ""]
        for event in items[:30]:
            event_id = _safe_text(event.get("event_id"), 40)
            ts = _safe_text(event.get("ts"), 32)
            surface = _safe_text(event.get("surface"), 48)
            summary = _payload_summary(event)
            lines.append(f"- `{ts}` `{event_id}` {surface}: {summary}")
        if len(items) > 30:
            lines.append(f"- ほか {len(items) - 30} 件")
        lines.append("")

    lines += [
        "## 運用メモ",
        "",
        "- このノートは Cloud Run が GCS に追記した入力イベントを、ローカルMac側で同期して生成した要約です。",
        "- Obsidian Vault が正本です。Cloud Run はローカルVaultへ直接書き込みません。",
        "- 生イベントはローカル `data/cloudrun_inputs/` の JSONL を参照します。",
        "",
    ]
    return "\n".join(lines)


def _build_daily_section(day: str, events: list[dict[str, Any]], note_rel: Path) -> str:
    counts = Counter(str(event.get("event_type") or "unknown") for event in events)
    lines = [
        DAILY_SECTION_START,
        "## Cloud Run 入力同期",
        "",
        f"- 取り込みイベント: {len(events)} 件",
    ]
    for event_type, count in counts.most_common():
        label = EVENT_LABELS.get(event_type, event_type)
        lines.append(f"- {label}: {count} 件")
    lines += [
        f"- 詳細: [[{str(note_rel.with_suffix('')).replace(chr(92), '/')}]]",
        "",
        DAILY_SECTION_END,
        "",
    ]
    return "\n".join(lines)


def _replace_generated_section(original: str, section: str) -> str:
    start = original.find(DAILY_SECTION_START)
    end = original.find(DAILY_SECTION_END)
    if start >= 0 and end >= start:
        end += len(DAILY_SECTION_END)
        return original[:start].rstrip() + "\n\n" + section.rstrip() + "\n\n" + original[end:].lstrip()
    return original.rstrip() + "\n\n" + section.rstrip() + "\n"


def _write_daily_section(vault: Path, day: str, events: list[dict[str, Any]], note_rel: Path, dry_run: bool) -> None:
    daily_path = vault / DAILY_SUBDIR / f"{day}.md"
    section = _build_daily_section(day, events, note_rel)
    if dry_run:
        print(f"[dry-run] {daily_path} daily-section events={len(events)}")
        return
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    current = daily_path.read_text(encoding="utf-8") if daily_path.exists() else f"# {day}\n"
    daily_path.write_text(_replace_generated_section(current, section), encoding="utf-8")


def sync(days: int, target_date: str | None = None, dry_run: bool = False) -> dict[str, int]:
    if not OBSIDIAN_VAULT.exists():
        raise SystemExit(f"Vault が見つかりません: {OBSIDIAN_VAULT}")
    if not (OBSIDIAN_VAULT / ".obsidian").exists():
        raise SystemExit(f"Obsidian Vault ではありません: {OBSIDIAN_VAULT}")

    target_days = [date.fromisoformat(target_date)] if target_date else list(_date_range(days))
    scan_days = sorted(set(target_days + list(_date_range(days + 1))))
    events_by_jst_day = _load_events_by_jst_day(scan_days)
    out_dir = OBSIDIAN_VAULT / OUTPUT_SUBDIR
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for day in target_days:
        events = events_by_jst_day.get(day, [])
        if not events:
            skipped += 1
            print(f"[skip] {day.isoformat()} events=0")
            continue

        md = _build_markdown(day.isoformat(), events)
        rel_path = OUTPUT_SUBDIR / f"{day.isoformat()}_cloudrun_inputs.md"
        path = OBSIDIAN_VAULT / rel_path
        if dry_run:
            print(f"[dry-run] {path} events={len(events)}")
            _write_daily_section(OBSIDIAN_VAULT, day.isoformat(), events, rel_path, dry_run=True)
            _write_improvement_logs(OBSIDIAN_VAULT, day, events, dry_run=True)
            _write_chat_logs(OBSIDIAN_VAULT, day, events, dry_run=True)
            _write_dialogue_logs(OBSIDIAN_VAULT, day, events, dry_run=True)
            skipped += 1
            continue
        path.write_text(md, encoding="utf-8")
        _write_daily_section(OBSIDIAN_VAULT, day.isoformat(), events, rel_path, dry_run=False)
        improvements = _write_improvement_logs(OBSIDIAN_VAULT, day, events, dry_run=False)
        chat_logs = _write_chat_logs(OBSIDIAN_VAULT, day, events, dry_run=False)
        dialogue_logs = _write_dialogue_logs(OBSIDIAN_VAULT, day, events, dry_run=False)
        suffix = (
            (f" improvements={improvements}" if improvements else "")
            + (f" chats={chat_logs}" if chat_logs else "")
            + (f" dialogues={dialogue_logs}" if dialogue_logs else "")
        )
        print(f"[write] {path} events={len(events)}{suffix}")
        written += 1

    return {"written": written, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Run入力イベントをObsidian要約ノートへ同期する")
    parser.add_argument("--days", type=int, default=int(os.environ.get("CLOUDRUN_INPUT_OBSIDIAN_DAYS", "3")))
    parser.add_argument("--date", help="YYYY-MM-DD の1日だけ同期")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = sync(days=args.days, target_date=args.date, dry_run=args.dry_run)
    print(f"完了: written={result['written']} skipped={result['skipped']}")


if __name__ == "__main__":
    main()
