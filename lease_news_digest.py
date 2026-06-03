"""Helpers for reading the latest lease-news digest from Obsidian."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NEWS_REL_DIR = Path("リースニュース")
METRICS_PATH = Path(__file__).resolve().parent / "data" / "lease_news_metrics.json"


@dataclass(frozen=True)
class LeaseNewsFocus:
    available: bool
    note_path: str = ""
    note_date: str = ""
    profile: str = ""
    theme_summary: str = ""
    bucket_summary: str = ""
    tag_summary: str = ""
    focus_lines: tuple[str, ...] = ()
    memo_lines: tuple[str, ...] = ()
    metrics_lines: tuple[str, ...] = ()
    article_titles: tuple[str, ...] = ()

    @property
    def headline(self) -> str:
        if not self.available:
            return ""
        if self.theme_summary:
            return f"最新テーマ: {self.theme_summary}"
        return "最新ニュースの論点あり"


def _vault_candidates() -> list[Path]:
    home = Path.home()
    raw_candidates = [
        os.environ.get("OBSIDIAN_VAULT_PATH"),
        os.environ.get("OBSIDIAN_VAULT"),
        str(home / "Documents" / "Obsidian Vault"),
        str(home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"),
        str(home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Obsidian Vault"),
    ]
    out: list[Path] = []
    for raw in raw_candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists() and path.is_dir():
            out.append(path)
    return out


def find_vault() -> Path | None:
    candidates = _vault_candidates()
    return candidates[0] if candidates else None


def _latest_news_note(vault: Path) -> Path | None:
    news_dir = vault / DEFAULT_NEWS_REL_DIR
    if not news_dir.exists():
        return None
    notes = list(news_dir.glob("*_リースニュース_*.md")) or list(news_dir.glob("*_lease-news.md"))
    if not notes:
        return None
    notes.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return notes[0]


def _extract_section(text: str, heading: str) -> list[str]:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    for line in lines:
        if re.match(pattern, line):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section:
            out.append(line.rstrip())
    return out


def _extract_bullets(lines: list[str]) -> list[str]:
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _extract_prefixed_lines(text: str, prefix: str) -> list[str]:
    results: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            results.append(stripped[len(prefix):].strip())
    return results


def _extract_article_titles(text: str, limit: int = 3) -> list[str]:
    titles: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            title = stripped[4:].strip()
            if title:
                titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _infer_focus_lines(theme_summary: str, bucket_summary: str, memo_lines: list[str]) -> list[str]:
    joined = " ".join([theme_summary, bucket_summary, " ".join(memo_lines)])
    focus: list[str] = []

    def add(line: str) -> None:
        if line not in focus:
            focus.append(line)

    if any(key in joined for key in ("金利", "政策", "金融", "与信")):
        add("提示金利・競合提示金利差・基準金利の説明を厚くする。")
    if any(key in joined for key in ("設備投資", "省力化", "更新", "投資")):
        add("リース期間・中古価値・再リース余地を確認する。")
    if any(key in joined for key in ("物流", "車両", "トラック", "EV")):
        add("稼働率・車両更新・保守費用の見通しを確認する。")
    if any(key in joined for key in ("建設", "不動産", "工場")):
        add("工期・更新投資・移設可能性・現場稼働への影響を確認する。")
    if any(key in joined for key in ("製造", "DX", "自動化", "ロボット")):
        add("更新投資の回収期間と生産性改善効果を確認する。")
    if any(key in joined for key in ("法令", "制度", "税制", "補助金", "助成金")):
        add("契約条件・税制・補助金要件の再確認を優先する。")

    if not focus:
        focus.append("直近のニュースを踏まえ、提示条件と審査コメントを更新する。")

    return focus[:4]


def get_latest_lease_news_focus(vault: Path | None = None) -> LeaseNewsFocus:
    vault = vault or find_vault()
    if not vault:
        return LeaseNewsFocus(available=False)

    note = _latest_news_note(vault)
    if not note or not note.exists():
        return LeaseNewsFocus(available=False)

    try:
        text = note.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return LeaseNewsFocus(available=False)

    # 新フォーマット（PR #265）: date:, tags:, region:, importance:
    # 旧フォーマット: created:, profile:, - 主なテーマ:
    date_match = re.search(r"^date:\s*(.+)$", text, re.MULTILINE)
    created_match = re.search(r"^created:\s*(.+)$", text, re.MULTILINE)
    profile_match = re.search(r"^profile:\s*(.+)$", text, re.MULTILINE)
    region_match = re.search(r"^region:\s*(.+)$", text, re.MULTILINE)
    importance_match = re.search(r"^importance:\s*(.+)$", text, re.MULTILINE)
    # 旧フォーマット用フィールド（新フォーマットでは空になる）
    theme_match = re.search(r"^- 主なテーマ:\s*(.+)$", text, re.MULTILINE)
    bucket_match = re.search(r"^- 収集セット:\s*(.+)$", text, re.MULTILINE)
    tag_match = re.search(r"^- 重点タグ:\s*(.+)$", text, re.MULTILINE)
    memo_section = _extract_section(text, "審査メモ")
    memo_bullets = _extract_bullets(memo_section)
    metrics_section = _extract_section(text, "効果測定")
    metrics_bullets = _extract_bullets(metrics_section)
    # 新フォーマットの活用メモを審査論点として使用
    usage_memo_section = _extract_section(text, "活用メモ")
    article_review_lines = _extract_prefixed_lines(text, "- 審査論点:") or [
        line.strip() for line in usage_memo_section if line.strip()
    ]
    article_titles = _extract_article_titles(text, limit=3)

    # 新フォーマット: region/importanceをtheme_summaryとして使用
    region = (region_match.group(1).strip() if region_match else "").strip()
    importance = (importance_match.group(1).strip() if importance_match else "").strip()
    theme_summary = (theme_match.group(1).strip() if theme_match else "").strip()
    if not theme_summary and (region or importance):
        theme_summary = " / ".join(filter(None, [region, importance]))
    bucket_summary = (bucket_match.group(1).strip() if bucket_match else "").strip()
    tag_summary = (tag_match.group(1).strip() if tag_match else "").strip()
    focus_source = article_review_lines or _infer_focus_lines(theme_summary, bucket_summary, memo_bullets)
    focus_lines = tuple((focus_source[:4] if isinstance(focus_source, list) else list(focus_source)[:4]))
    note_date = ""
    if date_match:
        note_date = date_match.group(1).strip()
    elif created_match:
        note_date = created_match.group(1).strip()
    else:
        try:
            note_date = dt.date.fromtimestamp(note.stat().st_mtime).isoformat()
        except Exception:
            note_date = ""

    return LeaseNewsFocus(
        available=True,
        note_path=str(note.relative_to(vault)),
        note_date=note_date,
        profile=(profile_match.group(1).strip() if profile_match else ""),
        theme_summary=theme_summary,
        bucket_summary=bucket_summary,
        tag_summary=tag_summary,
        focus_lines=focus_lines,
        memo_lines=tuple(memo_bullets),
        metrics_lines=tuple(metrics_bullets),
        article_titles=tuple(article_titles),
    )


def lease_news_focus_as_text(vault: Path | None = None) -> str:
    focus = get_latest_lease_news_focus(vault=vault)
    if not focus.available:
        return ""
    lines = []
    if focus.theme_summary:
        lines.append(f"主なテーマ: {focus.theme_summary}")
    if focus.bucket_summary:
        lines.append(f"収集セット: {focus.bucket_summary}")
    if focus.tag_summary:
        lines.append(f"重点タグ: {focus.tag_summary}")
    lines.extend(f"- {line}" for line in focus.focus_lines)
    return "\n".join(lines).strip()


def _load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_metrics(data: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_markdown(path: Path, text: str) -> None:
    body = text.strip()
    if not body:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = "\n\n" if path.exists() and path.read_text(encoding="utf-8", errors="ignore").strip() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(prefix + body + "\n")


def _daily_note_path(vault: Path, date_str: str) -> Path:
    return vault / "Daily" / f"{date_str}.md"


def _ensure_day_bucket(data: dict, date_str: str) -> dict:
    buckets = data.setdefault("days", {})
    if date_str not in buckets:
        buckets[date_str] = {
            "collections": 0,
            "views": 0,
            "judgment_changes": 0,
            "last_note": "",
            "last_tags": "",
            "last_source_summary": "",
            "last_article_count": 0,
            "last_judgment_note": "",
            "last_judgment_source_note_date": "",
            "last_judgment_company": "",
            "last_judgment_decision": "",
            "last_judgment_reason": "",
        }
    return buckets[date_str]


def record_lease_news_collection(
    date_str: str,
    note_path: str,
    article_count: int,
    source_summary: str = "",
    tag_summary: str = "",
) -> dict:
    data = _load_metrics()
    bucket = _ensure_day_bucket(data, date_str)
    bucket["collections"] = int(bucket.get("collections", 0)) + 1
    bucket["last_note"] = note_path
    bucket["last_source_summary"] = source_summary
    bucket["last_tags"] = tag_summary
    bucket["last_article_count"] = int(article_count)
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)
    return bucket


def record_lease_news_view(date_str: str, note_path: str = "", tag_summary: str = "") -> dict:
    data = _load_metrics()
    bucket = _ensure_day_bucket(data, date_str)
    bucket["views"] = int(bucket.get("views", 0)) + 1
    if note_path:
        bucket["last_note"] = note_path
    if tag_summary:
        bucket["last_tags"] = tag_summary
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)
    return bucket


def record_lease_news_judgment_change(
    date_str: str,
    note_path: str = "",
    source_note_date: str = "",
    company_name: str = "",
    score: float | None = None,
    final_decision: str = "",
    reason: str = "",
    focus_lines: tuple[str, ...] = (),
    theme_summary: str = "",
    tag_summary: str = "",
) -> dict:
    data = _load_metrics()
    bucket = _ensure_day_bucket(data, date_str)
    bucket["judgment_changes"] = int(bucket.get("judgment_changes", 0)) + 1
    if note_path:
        bucket["last_judgment_note"] = note_path
    if source_note_date:
        bucket["last_judgment_source_note_date"] = source_note_date
    if company_name:
        bucket["last_judgment_company"] = company_name
    if final_decision:
        bucket["last_judgment_decision"] = final_decision
    if reason:
        bucket["last_judgment_reason"] = reason
    if score is not None:
        try:
            bucket["last_judgment_score"] = round(float(score), 1)
        except Exception:
            bucket["last_judgment_score"] = 0
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)

    vault = find_vault()
    if vault:
        now_label = dt.datetime.now().strftime("%H:%M")
        lines = [
            f"## {now_label} ニュースで判断変更",
            "",
            f"- 案件: {company_name or '不明'}",
            f"- スコア: {round(float(score), 1)}点" if score is not None else "- スコア: 不明",
            f"- 判定: {final_decision or '—'}",
        ]
        if theme_summary:
            lines.append(f"- 注目論点: {theme_summary}")
        if tag_summary:
            lines.append(f"- 重点タグ: {tag_summary}")
        if note_path:
            lines.append(f"- 参照ノート: {note_path}")
        if source_note_date:
            lines.append(f"- 参照ニュース日付: {source_note_date}")
        if focus_lines:
            lines.extend(["", "- 参照した論点:"])
            for line in focus_lines[:4]:
                lines.append(f"  - {line}")
        if reason:
            lines.append(f"- メモ: {reason}")
        _append_markdown(_daily_note_path(vault, date_str), "\n".join(lines))

    return bucket


def get_lease_news_metrics(date_str: str | None = None) -> dict:
    data = _load_metrics()
    days = data.get("days", {})
    if date_str:
        return dict(days.get(date_str, {}))
    return data
