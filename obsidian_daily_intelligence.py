"""Turn daily Obsidian notes into reusable decision signals.

This is intentionally local and deterministic. It does not send private Vault
content to an external model. The output is a compact JSON/Markdown bridge from
"saved notes" back into chat, screening, memory promotion, and improvement work.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lease_news_digest import find_vault


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
LATEST_JSON = REPORTS_DIR / "obsidian_daily_intelligence_latest.json"
METRICS_JSONL = DATA_DIR / "obsidian_daily_intelligence_metrics.jsonl"


@dataclass(frozen=True)
class DailySignal:
    kind: str
    text: str
    source_path: str = ""
    priority: str = "medium"
    route: str = "chat"


def _read(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return text.strip()[:max_chars]


def _relative(vault: Path, path: Path) -> str:
    try:
        return str(path.relative_to(vault))
    except ValueError:
        return str(path)


def _extract_bullets(text: str, headings: tuple[str, ...] = ()) -> list[str]:
    if headings:
        blocks: list[str] = []
        for heading in headings:
            match = re.search(rf"^##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL | re.MULTILINE)
            if match:
                blocks.append(match.group(1))
        text = "\n".join(blocks)
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if len(item) >= 8:
                bullets.append(item[:220])
    return bullets


def _recent_files(vault: Path, date_str: str) -> list[Path]:
    candidates: list[Path] = []
    exact_paths = [
        vault / "Daily" / f"{date_str}.md",
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Dialogue" / f"{date_str}.md",
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection" / f"{date_str}.md",
    ]
    candidates.extend(path for path in exact_paths if path.exists())

    glob_dirs = [
        vault / "Projects" / "tune_lease_55" / "News",
        vault / "Projects" / "tune_lease_55" / "Feedback",
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Knowledge Corrections",
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Knowledge",
        vault / "05-クリップ_記事" / "リースニュース",
    ]
    for directory in glob_dirs:
        if not directory.exists():
            continue
        candidates.extend(directory.glob(f"{date_str}*.md"))

    unique: dict[str, Path] = {}
    for path in candidates:
        if path.is_file():
            unique[str(path)] = path
    return sorted(unique.values(), key=lambda p: str(p))


def _classify_line(line: str, source_path: str) -> DailySignal | None:
    low_source = source_path.lower()
    text = line.strip()
    if not text:
        return None
    low_value_prefixes = (
        "対象業種:",
        "対象業種：",
        "リース物件: 対象物件未特定",
        "リース物件：対象物件未特定",
        "影響方向:",
        "影響方向：",
    )
    if text.startswith(low_value_prefixes):
        return None

    improvement_keys = ("改善", "修正", "バグ", "エラー", "壊", "遅", "ラグ", "JSON", "未完了")
    screening_keys = ("審査", "承認", "否決", "金利", "与信", "物件", "リース期間", "前受金", "保証", "稼働率", "中古価値")
    memory_keys = ("覚え", "記憶", "昇格", "方針", "次回", "毎日", "繰り返", "同じ")
    stale_keys = ("税制", "法令", "制度", "会計基準", "valid_until", "更新期限", "古い")

    if any(key in text for key in improvement_keys) or "feedback" in low_source:
        return DailySignal("next_action", text, source_path, priority="high", route="improvement")
    if any(key in text for key in screening_keys) or "lease-news-actions" in low_source:
        return DailySignal("screening_signal", text, source_path, priority="high", route="screening")
    if any(key in text for key in stale_keys) or "knowledge corrections" in low_source:
        return DailySignal("stale_knowledge", text, source_path, priority="high", route="knowledge")
    if any(key in text for key in memory_keys) or "private reflection" in low_source or "dialogue" in low_source:
        return DailySignal("memory_signal", text, source_path, priority="medium", route="chat")
    return DailySignal("new_signal", text, source_path, priority="low", route="chat")


def _collect_signals(vault: Path, date_str: str) -> list[DailySignal]:
    signals: list[DailySignal] = []
    for path in _recent_files(vault, date_str):
        rel = _relative(vault, path)
        text = _read(path)
        bullets = _extract_bullets(
            text,
            headings=(
                "Promotable Items",
                "差分と再利用",
                "今日の使いどころ",
                "確認項目",
                "条件への影響",
                "次の対話や判断へ戻す内省",
                "注目論点",
                "会話サマリー",
                "AI整理",
                "次の行動",
            ),
        )
        if not bullets:
            bullets = _extract_bullets(text)[:6]
        if not bullets:
            title = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
            if title:
                bullets = [title.group(1).strip()]
        for bullet in bullets[:8]:
            signal = _classify_line(bullet, rel)
            if signal:
                signals.append(signal)
    return _dedupe_signals(signals)


def _dedupe_signals(signals: list[DailySignal]) -> list[DailySignal]:
    seen: set[tuple[str, str]] = set()
    out: list[DailySignal] = []
    for signal in signals:
        normalized = re.sub(r"\s+", "", signal.text)
        key = (signal.kind, normalized[:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(signal)
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    out.sort(key=lambda signal: (priority_rank.get(signal.priority, 9), signal.route, signal.kind))
    return out[:40]


def _signal_dict(signal: DailySignal) -> dict[str, str]:
    signal_id = hashlib.sha1(
        f"{signal.kind}|{signal.route}|{signal.source_path}|{signal.text}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": signal_id,
        "kind": signal.kind,
        "text": signal.text,
        "source_path": signal.source_path,
        "priority": signal.priority,
        "route": signal.route,
    }


def _load_metrics_events(date_str: str | None = None) -> list[dict[str, Any]]:
    if not METRICS_JSONL.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = METRICS_JSONL.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines[-5000:]:
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if date_str and event.get("date") != date_str:
            continue
        events.append(event)
    return events


def _all_signal_items(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in ("should_affect_chat", "should_affect_screening", "next_actions", "promote_to_memory", "stale_knowledge"):
        for item in bundle.get(key) or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "")
            if item_id and item_id in seen:
                continue
            if item_id:
                seen.add(item_id)
            items.append(item)
    return items


def _build_metrics_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    date_str = str(bundle.get("date") or dt.date.today().isoformat())
    events = _load_metrics_events(date_str)
    generated_ids = {str(item.get("id")) for item in _all_signal_items(bundle) if item.get("id")}
    generated_by_route = {
        "chat": {str(item.get("id")) for item in bundle.get("should_affect_chat") or [] if item.get("id")},
        "screening": {str(item.get("id")) for item in bundle.get("should_affect_screening") or [] if item.get("id")},
        "improvement": {str(item.get("id")) for item in bundle.get("next_actions") or [] if item.get("id")},
        "knowledge": {str(item.get("id")) for item in bundle.get("stale_knowledge") or [] if item.get("id")},
    }
    injected_ids: set[str] = set()
    effective_ids: set[str] = set()
    by_route: dict[str, dict[str, Any]] = {}
    for event in events:
        route = str(event.get("route") or "unknown")
        bucket = by_route.setdefault(
            route,
            {
                "injection_events": 0,
                "response_events": 0,
                "effective_events": 0,
                "injected_signal_ids": set(),
                "effective_signal_ids": set(),
            },
        )
        signal_ids = {str(v) for v in event.get("signal_ids") or [] if v}
        matched_ids = {str(v) for v in event.get("effective_signal_ids") or [] if v}
        if event.get("event") == "injected":
            bucket["injection_events"] += 1
            bucket["injected_signal_ids"].update(signal_ids)
            injected_ids.update(signal_ids)
        if event.get("event") == "response_evaluated":
            bucket["response_events"] += 1
            if matched_ids:
                bucket["effective_events"] += 1
            bucket["effective_signal_ids"].update(matched_ids)
            effective_ids.update(matched_ids)

    route_summary: dict[str, dict[str, Any]] = {}
    for route, bucket in by_route.items():
        route_injected = bucket["injected_signal_ids"]
        route_effective = bucket["effective_signal_ids"]
        route_generated = generated_by_route.get(route, set())
        route_summary[route] = {
            "generated_signals": len(route_generated),
            "injection_events": bucket["injection_events"],
            "response_events": bucket["response_events"],
            "effective_events": bucket["effective_events"],
            "unique_injected_signals": len(route_injected),
            "unique_effective_signals": len(route_effective),
            "reuse_rate": round(len(route_injected) / len(route_generated), 4) if route_generated else 0.0,
            "effect_rate": round(len(route_effective) / len(route_injected), 4) if route_injected else 0.0,
        }
    for route, route_generated in generated_by_route.items():
        if route not in route_summary and route_generated:
            route_summary[route] = {
                "generated_signals": len(route_generated),
                "injection_events": 0,
                "response_events": 0,
                "effective_events": 0,
                "unique_injected_signals": 0,
                "unique_effective_signals": 0,
                "reuse_rate": 0.0,
                "effect_rate": 0.0,
            }

    return {
        "date": date_str,
        "generated_signals": len(generated_ids),
        "unique_injected_signals": len(injected_ids),
        "unique_effective_signals": len(effective_ids),
        "reuse_rate": round(len(injected_ids) / len(generated_ids), 4) if generated_ids else 0.0,
        "effect_rate": round(len(effective_ids) / len(injected_ids), 4) if injected_ids else 0.0,
        "event_count": len(events),
        "by_route": route_summary,
    }


def build_obsidian_daily_intelligence(date_str: str | None = None, vault: Path | None = None) -> dict[str, Any]:
    date_str = date_str or dt.date.today().isoformat()
    vault = vault or find_vault()
    if not vault:
        return {"available": False, "date": date_str, "reason": "Obsidian Vault not found"}

    signals = _collect_signals(vault, date_str)
    by_route = {
        "chat": [s for s in signals if s.route == "chat"],
        "screening": [s for s in signals if s.route == "screening"],
        "improvement": [s for s in signals if s.route == "improvement"],
        "knowledge": [s for s in signals if s.route == "knowledge"],
    }
    repeated_issues = [
        s for s in signals
        if any(key in s.text for key in ("繰り返", "同じ", "再発", "停滞", "毎日", "3日"))
    ]
    promote_to_memory = [
        s for s in signals
        if s.kind in {"memory_signal", "stale_knowledge"} or s.priority == "high"
    ][:8]

    bundle = {
        "available": True,
        "date": date_str,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source_count": len(_recent_files(vault, date_str)),
        "new_signals": [_signal_dict(s) for s in signals[:12]],
        "repeated_issues": [_signal_dict(s) for s in repeated_issues[:6]],
        "promote_to_memory": [_signal_dict(s) for s in promote_to_memory],
        "stale_knowledge": [_signal_dict(s) for s in by_route["knowledge"][:6]],
        "next_actions": [_signal_dict(s) for s in by_route["improvement"][:8]],
        "should_affect_chat": [_signal_dict(s) for s in by_route["chat"][:8]],
        "should_affect_screening": [_signal_dict(s) for s in by_route["screening"][:8]],
    }
    bundle["knowledge_kpi"] = _build_metrics_summary(bundle)
    return bundle


def _write_markdown(vault: Path, bundle: dict[str, Any]) -> Path:
    date_str = str(bundle.get("date") or dt.date.today().isoformat())
    out_dir = vault / "Projects" / "tune_lease_55" / "Daily Intelligence"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date_str}_obsidian-daily-intelligence.md"
    lines = [
        "---",
        f"date: {date_str}",
        "type: obsidian_daily_intelligence",
        f"source_count: {bundle.get('source_count', 0)}",
        "---",
        f"# Obsidian Daily Intelligence — {date_str}",
        "",
    ]
    sections = [
        ("知識KPI", "knowledge_kpi"),
        ("次の対話へ戻す", "should_affect_chat"),
        ("審査へ戻す", "should_affect_screening"),
        ("改善へ回す", "next_actions"),
        ("記憶へ昇格候補", "promote_to_memory"),
        ("古くなり得る知識", "stale_knowledge"),
        ("繰り返しシグナル", "repeated_issues"),
    ]
    for heading, key in sections:
        lines.extend([f"## {heading}", ""])
        if key == "knowledge_kpi":
            kpi = bundle.get("knowledge_kpi") or {}
            lines.append(f"- 再利用率: {kpi.get('reuse_rate', 0.0)}")
            lines.append(f"- 効果率: {kpi.get('effect_rate', 0.0)}")
            lines.append(f"- 生成シグナル: {kpi.get('generated_signals', 0)}")
            lines.append(f"- 注入済みシグナル: {kpi.get('unique_injected_signals', 0)}")
            lines.append(f"- 効果推定シグナル: {kpi.get('unique_effective_signals', 0)}")
            lines.append("")
            continue
        items = bundle.get(key) or []
        if not items:
            lines.append("- なし")
        else:
            for item in items[:8]:
                source = item.get("source_path") or ""
                suffix = f"（{source}）" if source else ""
                lines.append(f"- {item.get('text', '')}{suffix}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_obsidian_daily_intelligence(date_str: str | None = None, vault: Path | None = None) -> dict[str, Any]:
    vault = vault or find_vault()
    bundle = build_obsidian_daily_intelligence(date_str=date_str, vault=vault)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if vault and bundle.get("available"):
        note_path = _write_markdown(vault, bundle)
        bundle["note_path"] = str(note_path.relative_to(vault))
    latest = LATEST_JSON
    dated = REPORTS_DIR / f"obsidian_daily_intelligence_{str(bundle.get('date', '')).replace('-', '')}.json"
    payload = json.dumps(bundle, ensure_ascii=False, indent=2)
    latest.write_text(payload, encoding="utf-8")
    dated.write_text(payload, encoding="utf-8")
    return bundle


def _items_for_route(bundle: dict[str, Any], route: str, limit: int = 5) -> list[dict[str, Any]]:
    key = "should_affect_screening" if route == "screening" else "should_affect_chat"
    items = [item for item in bundle.get(key) or [] if isinstance(item, dict)]
    return items[:limit]


def _keyword_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.findall(r"[A-Za-z0-9_]{4,}|[一-龥ぁ-んァ-ヶー]{2,}", text or ""):
        if token in {"する", "ある", "いる", "これ", "それ", "ため", "こと", "よう", "対象", "未特定"}:
            continue
        tokens.add(token[:24])
    return tokens


def _matched_signal_ids(items: list[dict[str, Any]], response_text: str) -> list[str]:
    response_tokens = _keyword_tokens(response_text)
    matched: list[str] = []
    for item in items:
        item_id = str(item.get("id") or "")
        signal_tokens = _keyword_tokens(str(item.get("text") or ""))
        if item_id and signal_tokens and len(signal_tokens & response_tokens) >= 1:
            matched.append(item_id)
    return matched


def _rewrite_latest_with_metrics(bundle: dict[str, Any]) -> None:
    if not bundle.get("available"):
        return
    bundle["knowledge_kpi"] = _build_metrics_summary(bundle)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, ensure_ascii=False, indent=2)
    LATEST_JSON.write_text(payload, encoding="utf-8")
    dated = REPORTS_DIR / f"obsidian_daily_intelligence_{str(bundle.get('date', '')).replace('-', '')}.json"
    dated.write_text(payload, encoding="utf-8")


def record_obsidian_daily_intelligence_event(
    *,
    surface: str,
    route: str = "chat",
    event: str = "injected",
    response_text: str = "",
    question: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    """Record whether daily intelligence was reused and whether it affected output.

    This is a local heuristic metric. "Effective" means the final response reused
    at least one meaningful token from an injected signal.
    """
    try:
        bundle = load_latest_obsidian_daily_intelligence()
        if not bundle.get("available"):
            return {"recorded": False, "reason": "no latest bundle"}
        items = _items_for_route(bundle, route, limit=limit)
        if not items:
            return {"recorded": False, "reason": "no route items"}
        signal_ids = [str(item.get("id")) for item in items if item.get("id")]
        if not signal_ids:
            return {"recorded": False, "reason": "no signal ids"}
        effective_ids = _matched_signal_ids(items, response_text) if event == "response_evaluated" else []
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "date": str(bundle.get("date") or dt.date.today().isoformat()),
            "surface": surface,
            "route": route,
            "event": event,
            "signal_ids": signal_ids,
            "effective_signal_ids": effective_ids,
            "question_excerpt": (question or "")[:160],
            "response_excerpt": (response_text or "")[:240],
        }
        with METRICS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _rewrite_latest_with_metrics(bundle)
        return {
            "recorded": True,
            "event": event,
            "signal_count": len(signal_ids),
            "effective_signal_count": len(effective_ids),
            "knowledge_kpi": load_latest_obsidian_daily_intelligence().get("knowledge_kpi") or {},
        }
    except Exception as exc:
        return {"recorded": False, "reason": f"record_error: {exc}"}


def load_latest_obsidian_daily_intelligence() -> dict[str, Any]:
    if not LATEST_JSON.exists():
        return {}
    try:
        data = json.loads(LATEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def obsidian_daily_intelligence_as_text(route: str = "chat", limit: int = 5) -> str:
    data = load_latest_obsidian_daily_intelligence()
    if not data.get("available"):
        return ""
    items = _items_for_route(data, route, limit=limit)
    if not items:
        return ""
    title = "【Obsidian日次知性: 審査に戻すシグナル】" if route == "screening" else "【Obsidian日次知性: 次の対話に戻すシグナル】"
    lines = [title]
    for item in items:
        lines.append(f"- {item.get('text', '')}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Obsidian Daily Intelligence")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD。省略時は今日")
    args = parser.parse_args()
    result = write_obsidian_daily_intelligence(date_str=args.date)
    print(json.dumps({
        "available": result.get("available"),
        "date": result.get("date"),
        "source_count": result.get("source_count"),
        "note_path": result.get("note_path"),
        "chat": len(result.get("should_affect_chat") or []),
        "screening": len(result.get("should_affect_screening") or []),
        "next_actions": len(result.get("next_actions") or []),
        "knowledge_kpi": result.get("knowledge_kpi") or {},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
