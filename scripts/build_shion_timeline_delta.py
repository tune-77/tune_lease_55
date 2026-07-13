"""Build a compact timeline delta from recent daily memory notes.

This is intentionally deterministic. It does not call an LLM and does not feed
the result into prompts by itself. The first use is inspection: see whether
Shion's memory is changing in a way that could improve the next response.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "shion_timeline_delta.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "shion_timeline_delta_latest.md"

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}|[一-龥ァ-ヴー]{2,}")

NOISE_TOKENS = {
    "Work",
    "Log",
    "Promotable",
    "Items",
    "Private",
    "Reflection",
    "Cloud",
    "Run",
    "User",
    "Shion",
    "紫苑",
    "追加",
    "確認",
    "実装",
    "更新",
    "生成",
    "保存",
    "修正",
    "出力",
    "対応",
    "テスト",
    "passed",
}

ENGLISH_STOPWORDS = {
    "the",
    "and",
    "not",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "when",
    "then",
    "than",
    "only",
    "after",
    "before",
    "because",
    "should",
    "could",
    "would",
    "must",
    "can",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "added",
    "updated",
    "fixed",
    "confirmed",
    "created",
    "generated",
    "deployed",
    "backfilled",
    "adjusted",
    "mirrored",
    "verification",
    "existing",
    "current",
}

ALLOWED_ASCII_TOKENS = {
    "AI",
    "B2B",
    "Cloud",
    "CloudRun",
    "LLM",
    "Obsidian",
    "Private",
    "Qrisk",
    "Q_risk",
    "RAG",
}

BEHAVIOR_TERMS = (
    "次回",
    "今後",
    "確認",
    "聞き",
    "質問",
    "反映",
    "判断",
    "評価",
    "比較",
    "検証",
    "避け",
    "優先",
    "短く",
    "高信号",
)

PRESSURE_TERMS = (
    "まだ",
    "弱い",
    "退屈",
    "ダメ",
    "不満",
    "未検証",
    "わからない",
    "できない",
    "効かない",
    "失敗",
    "ハッカソン",
    "審査員",
)


@dataclass
class DailyNote:
    day: str
    path: Path
    exists: bool
    work_items: list[str]
    promotable_items: list[str]
    behavior_items: list[str]
    pressure_items: list[str]
    tokens: Counter[str]

    @property
    def all_items(self) -> list[str]:
        return self.work_items + self.promotable_items

    @property
    def signal_items(self) -> list[str]:
        return self.promotable_items or self.all_items


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _section_text(text: str, section_name: str) -> str:
    lines = text.splitlines()
    start: int | None = None
    end = len(lines)
    for i, line in enumerate(lines):
        match = SECTION_RE.match(line)
        if not match:
            continue
        if match.group(1).strip() == section_name:
            start = i + 1
            continue
        if start is not None:
            end = i
            break
    if start is None:
        return ""
    return "\n".join(lines[start:end]).strip()


def _bullet_items(section: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for raw_line in section.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                items.append(_clean_item(" ".join(current)))
            current = [stripped[2:].strip()]
            continue
        if current and stripped and not stripped.startswith("#"):
            current.append(stripped)
    if current:
        items.append(_clean_item(" ".join(current)))
    return [item for item in items if len(item) >= 12]


def _clean_item(item: str) -> str:
    item = re.sub(r"\s+", " ", item).strip()
    return item.strip("- ")


def _tokens(items: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        for token in TOKEN_RE.findall(item):
            token = token.strip()
            lower = token.lower()
            if token in NOISE_TOKENS or lower in ENGLISH_STOPWORDS:
                continue
            if re.fullmatch(r"[A-Za-z0-9_.:-]+", token) and token not in ALLOWED_ASCII_TOKENS:
                continue
            if "/" in token or "\\" in token or token.startswith("."):
                continue
            if any(ch.isdigit() for ch in token) and not re.search(r"[A-Za-z一-龥ァ-ヴー]", token):
                continue
            if re.fullmatch(r"[0-9.:-]+", token):
                continue
            if len(token) < 2:
                continue
            counter[token] += 1
    return counter


def load_daily_note(memory_dir: Path, day: date) -> DailyNote:
    path = memory_dir / f"{day.isoformat()}.md"
    text = _read_text(path)
    work_items = _bullet_items(_section_text(text, "Work Log"))
    promotable_items = _bullet_items(_section_text(text, "Promotable Items"))
    all_items = work_items + promotable_items
    signal_items = promotable_items or all_items
    return DailyNote(
        day=day.isoformat(),
        path=path,
        exists=bool(text),
        work_items=work_items,
        promotable_items=promotable_items,
        behavior_items=[item for item in all_items if _contains_any(item, BEHAVIOR_TERMS)],
        pressure_items=[item for item in all_items if _contains_any(item, PRESSURE_TERMS)],
        tokens=_tokens(signal_items),
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return round(len(left & right) / len(union), 4)


def _top_terms(counter: Counter[str], limit: int = 12) -> list[str]:
    return [token for token, _ in counter.most_common(limit)]


def _pick_items(items: list[str], *, limit: int = 5) -> list[str]:
    return items[:limit]


def build_timeline_delta(memory_dir: Path, target_day: date, days: int = 4) -> dict[str, Any]:
    notes = [load_daily_note(memory_dir, target_day - timedelta(days=offset)) for offset in range(days)]
    today = notes[0]
    previous_notes = notes[1:]

    today_terms = set(today.tokens)
    yesterday_terms = set(previous_notes[0].tokens) if previous_notes else set()
    older_terms = set().union(*(set(note.tokens) for note in previous_notes[1:])) if len(previous_notes) > 1 else set()
    recent_terms = set().union(*(set(note.tokens) for note in previous_notes)) if previous_notes else set()

    continued_terms = sorted(today_terms & recent_terms)
    new_terms = sorted(today_terms - recent_terms)
    revived_terms = sorted((today_terms & older_terms) - yesterday_terms)
    dropped_terms = sorted(yesterday_terms - today_terms)

    day_summaries = [_daily_summary(note) for note in notes]
    layer_model = _build_layer_model(
        notes=notes,
        continued_terms=continued_terms,
        new_terms=new_terms,
        dropped_terms=dropped_terms,
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "target_date": target_day.isoformat(),
        "window_days": days,
        "sources": [
            {
                "date": note.day,
                "path": str(note.path.relative_to(REPO_ROOT)) if note.path.is_relative_to(REPO_ROOT) else str(note.path),
                "exists": note.exists,
                "work_count": len(note.work_items),
                "promotable_count": len(note.promotable_items),
            }
            for note in notes
        ],
        "delta": {
            "similarity_to_yesterday": _jaccard(today_terms, yesterday_terms),
            "continued_terms": continued_terms[:16],
            "new_terms": new_terms[:16],
            "revived_terms": revived_terms[:12],
            "dropped_terms": dropped_terms[:12],
            "top_today_terms": _top_terms(today.tokens),
        },
        "interpretation": {
            "continued_from_recent": _pick_items(_items_matching_terms(today.signal_items, continued_terms), limit=5),
            "new_or_sharpened": _pick_items(_items_matching_terms(today.signal_items, new_terms), limit=5),
            "user_pressure_points": _pick_items(today.pressure_items, limit=5),
            "next_behavior_candidates": _pick_items(today.behavior_items, limit=5),
        },
        "memory_layers": layer_model,
        "daily_summaries": day_summaries,
    }


def _build_layer_model(
    *,
    notes: list[DailyNote],
    continued_terms: list[str],
    new_terms: list[str],
    dropped_terms: list[str],
) -> dict[str, Any]:
    today = notes[0]
    token_days: dict[str, set[str]] = {}
    for note in notes:
        for token in note.tokens:
            token_days.setdefault(token, set()).add(note.day)

    repeated_terms = sorted(
        [token for token, days in token_days.items() if len(days) >= 2],
        key=lambda token: (-len(token_days[token]), token),
    )
    durable_terms = sorted(
        [token for token, days in token_days.items() if len(days) >= 3],
        key=lambda token: (-len(token_days[token]), token),
    )

    short_items = _dedupe_items(today.pressure_items + today.behavior_items + _items_matching_terms(today.signal_items, new_terms))
    mid_items = _dedupe_items(_items_matching_terms(today.signal_items, repeated_terms or continued_terms))
    long_items = _dedupe_items(
        _items_matching_terms(today.signal_items, durable_terms)
        + [
            item
            for item in today.promotable_items
            if _contains_any(item, ("should", "must", "必ず", "避け", "方針", "評価", "扱う", "すること"))
        ]
    )

    return {
        "short_term": {
            "purpose": "直近の会話運び。重複質問、露骨な記憶アピール、直前の訂正漏れを避ける。",
            "window": "minutes_to_1_day",
            "signals": {
                "new_terms": new_terms[:8],
                "dropped_terms": dropped_terms[:6],
            },
            "use": "次の返答の自然さ、聞き返しの少なさ、言い切りの調整にだけ使う。",
            "items": _pick_items(short_items, limit=5),
        },
        "mid_term": {
            "purpose": "数日単位の話題継続と圧点を見る。紫苑の次回振る舞い候補を作る。",
            "window": f"{max(2, len(notes))}_days",
            "signals": {
                "repeated_terms": repeated_terms[:12],
                "continued_terms": continued_terms[:12],
            },
            "use": "同じ不満・同じ論点が続く時だけ、応答方針を少し変える。",
            "items": _pick_items(mid_items, limit=5),
        },
        "long_term": {
            "purpose": "繰り返し残った判断基準・価値観・設計原則だけを昇格候補にする。",
            "window": "weeks_or_more",
            "signals": {
                "durable_terms": durable_terms[:12],
            },
            "use": "即プロンプト投入せず、レビューして長期記憶・判断基準へ昇格する。",
            "promotion_candidates": _pick_items(long_items, limit=5),
        },
        "anti_random_rule": "記憶を同じ棚へ入れない。短期は会話運び、中期は変化検知、長期は判断基準として扱う。",
    }


def _dedupe_items(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        clean = _clean_item(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _items_matching_terms(items: list[str], terms: list[str]) -> list[str]:
    term_set = set(terms)
    if not term_set:
        return []
    matched: list[str] = []
    for item in items:
        if set(_tokens([item])) & term_set:
            matched.append(item)
    return matched


def _daily_summary(note: DailyNote) -> dict[str, Any]:
    return {
        "date": note.day,
        "exists": note.exists,
        "top_terms": _top_terms(note.tokens, limit=8),
        "behavior_items": _pick_items(note.behavior_items, limit=3),
        "pressure_items": _pick_items(note.pressure_items, limit=3),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    delta = payload["delta"]
    interpretation = payload["interpretation"]
    layers = payload.get("memory_layers") or {}
    lines = [
        f"# Shion Timeline Delta - {payload['target_date']}",
        "",
        "## 差分サマリ",
        f"- 前日類似度: {delta['similarity_to_yesterday']}",
        f"- 継続語: {_join(delta['continued_terms'])}",
        f"- 新規/強化語: {_join(delta['new_terms'])}",
        f"- 復活語: {_join(delta['revived_terms'])}",
        f"- 前日から消えた語: {_join(delta['dropped_terms'])}",
        "",
        "## 今日の解釈",
        "### 継続している論点",
        *_markdown_items(interpretation["continued_from_recent"]),
        "",
        "### 新しく強まった論点",
        *_markdown_items(interpretation["new_or_sharpened"]),
        "",
        "### ユーザーの圧点",
        *_markdown_items(interpretation["user_pressure_points"]),
        "",
        "### 次回の振る舞い候補",
        *_markdown_items(interpretation["next_behavior_candidates"]),
        "",
        "## 記憶レイヤー",
        "### 短期記憶",
        *_layer_lines(layers.get("short_term") or {}),
        "",
        "### 中期記憶",
        *_layer_lines(layers.get("mid_term") or {}),
        "",
        "### 長期記憶",
        *_layer_lines(layers.get("long_term") or {}, candidates_key="promotion_candidates"),
        "",
        f"- ランダム化防止: {layers.get('anti_random_rule') or 'なし'}",
        "",
        "## 日別ミニサマリ",
    ]
    for summary in payload["daily_summaries"]:
        lines.append(f"### {summary['date']}")
        lines.append(f"- 主語: {_join(summary['top_terms'])}")
        if summary["behavior_items"]:
            lines.append(f"- 振る舞い候補: {summary['behavior_items'][0]}")
        if summary["pressure_items"]:
            lines.append(f"- 圧点: {summary['pressure_items'][0]}")
    lines.append("")
    return "\n".join(lines)


def _join(items: list[str]) -> str:
    return "、".join(items) if items else "なし"


def _markdown_items(items: list[str]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {item}" for item in items]


def _layer_lines(layer: dict[str, Any], *, candidates_key: str = "items") -> list[str]:
    if not layer:
        return ["- なし"]
    signals = layer.get("signals") or {}
    signal_parts = []
    for key, values in signals.items():
        signal_parts.append(f"{key}={_join(list(values or []))}")
    lines = [
        f"- 目的: {layer.get('purpose') or 'なし'}",
        f"- 窓: {layer.get('window') or 'なし'}",
        f"- 使い方: {layer.get('use') or 'なし'}",
        f"- 信号: {' / '.join(signal_parts) if signal_parts else 'なし'}",
    ]
    candidates = list(layer.get(candidates_key) or [])
    lines.append(f"- 候補: {_join(candidates)}")
    return lines


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    return date.fromisoformat(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Shion timeline delta from recent memory notes.")
    parser.add_argument("--date", default=None, help="Target date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--days", type=int, default=4, help="Number of days to compare, including target date.")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = build_timeline_delta(args.memory_dir, _parse_date(args.date), days=max(args.days, 2))
    if args.dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
