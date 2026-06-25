"""Helpers for reading the latest lease-news digest from Obsidian."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_NEWS_REL_DIRS = (
    Path("05-クリップ_記事") / "リースニュース",
    Path("リースニュース"),
)
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


@dataclass(frozen=True)
class LeaseNewsBrief:
    available: bool
    prefecture: str = ""
    region: str = ""
    geo_context: str = ""
    national_headline: str = ""
    national_focus_lines: tuple[str, ...] = ()
    regional_available: bool = False
    regional_title: str = ""
    regional_summary_lines: tuple[str, ...] = ()
    regional_usage_memo: str = ""
    regional_tags: tuple[str, ...] = ()
    regional_source: str = ""
    opening_line: str = ""
    question_line: str = ""
    note_date: str = ""
    note_path: str = ""


@dataclass(frozen=True)
class LeaseNewsFocusWriteResult:
    note_path: str
    daily_note_path: str
    theme_summary: str
    tag_summary: str


@dataclass(frozen=True)
class LeaseNewsReflectionWriteResult:
    note_path: str
    daily_note_path: str
    headline: str


@dataclass(frozen=True)
class LeaseNewsReflection:
    available: bool
    note_path: str = ""
    note_date: str = ""
    theme_summary: str = ""
    tag_summary: str = ""
    headline: str = ""
    thought_lines: tuple[str, ...] = ()
    tomorrow_lines: tuple[str, ...] = ()
    illustration_url: str = ""
    continuity_days: int = 0
    dominant_mood: str = ""
    self_narrative: str = ""
    current_question: str = ""
    memory_excerpt: str = ""
    user_understanding: str = ""
    user_curiosity: str = ""
    user_interests: tuple[str, ...] = ()
    observed_days: int = 0
    primary_goal: str = ""
    secondary_goal: str = ""
    ultimate_goal: str = ""
    ultimate_goal_status: str = ""
    knowledge_available: bool = False
    knowledge_scope: str = ""
    indexed_notes: int = 0
    knowledge_source_count: int = 0
    knowledge_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class LeaseNewsAction:
    signal: str
    affected_industries: tuple[str, ...] = ()
    affected_assets: tuple[str, ...] = ()
    risk_flags: tuple[str, ...] = ()
    recommended_checks: tuple[str, ...] = ()
    condition_impacts: tuple[str, ...] = ()
    source_title: str = ""
    source_path: str = ""
    valid_until: str = ""
    confidence: float = 0.5
    noise_score: float = 0.0


@dataclass(frozen=True)
class LeaseNewsActions:
    available: bool
    date: str = ""
    note_path: str = ""
    json_path: str = ""
    action_items: tuple[LeaseNewsAction, ...] = ()
    ignored_titles: tuple[str, ...] = ()
    summary: str = ""


def _parse_news_note(path: Path) -> dict:
    item: dict = {
        "date": "",
        "title": path.stem,
        "summary_lines": [],
        "usage_memo": "",
        "tags": [],
        "region": "国内",
        "importance": "通常",
        "source": "",
        "article_url": "",
        "file_path": str(path),
        "week": "",
        "month": "",
    }
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return item

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if fm_match:
        fm = fm_match.group(1)
        for line in fm.splitlines():
            if line.startswith("date:"):
                item["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("tags:"):
                try:
                    item["tags"] = json.loads(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            elif line.startswith("region:"):
                item["region"] = line.split(":", 1)[1].strip()
            elif line.startswith("source:"):
                item["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("importance:"):
                item["importance"] = line.split(":", 1)[1].strip()
            elif line.startswith("week:"):
                item["week"] = line.split(":", 1)[1].strip()
            elif line.startswith("month:"):
                item["month"] = line.split(":", 1)[1].strip()

    title_match = re.search(r"^# (.+)$", raw, re.MULTILINE)
    if title_match:
        item["title"] = title_match.group(1).strip()

    summary_section = re.search(r"## 3行要約\s*\n((?:- .+\n?){1,3})", raw)
    if summary_section:
        item["summary_lines"] = [
            line.lstrip("- ").strip()
            for line in summary_section.group(1).strip().splitlines()
            if line.strip()
        ]

    memo_match = re.search(r"## 活用メモ\s*\n(.+?)(?:\n##|\Z)", raw, re.DOTALL)
    if memo_match:
        item["usage_memo"] = memo_match.group(1).strip()

    link_match = re.search(r"^- link:\s*(.+)$", raw, re.MULTILINE)
    if link_match:
        item["article_url"] = link_match.group(1).strip()
    elif item["source"].startswith(("http://", "https://")):
        item["article_url"] = item["source"]

    return item


def _recent_news_items(vault: Path, limit: int = 10) -> list[dict]:
    md_files: list[Path] = []
    for rel_dir in DEFAULT_NEWS_REL_DIRS:
        news_dir = vault / rel_dir
        if not news_dir.exists():
            continue
        md_files.extend(news_dir.glob("*.md"))
    if not md_files:
        return []
    md_files = sorted(md_files, key=lambda p: p.stat().st_mtime, reverse=True)
    items = [_parse_news_note(fpath) for fpath in md_files[:limit]]
    return items


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _news_noise_score(item: dict) -> float:
    title = str(item.get("title") or "")
    memo = str(item.get("usage_memo") or "")
    joined = f"{title} {memo}"
    score = 0.0
    if _has_any(joined, ("おすすめ", "ランキング", "口コミ", "評判", "比較", "安い", "もらえる", "個人向け")):
        score += 0.75
    if _has_any(joined, ("審査なし", "審査が甘い")):
        score += 0.25
    if _has_any(joined, ("市場規模", "トップ35", "スコープ", "予測レポート")):
        score += 0.25
    if "その他" in " ".join(str(tag) for tag in item.get("tags", [])) and not _has_any(joined, ("金利", "決算", "格付", "設備投資", "補助金", "会計基準")):
        score += 0.15
    if _has_any(joined, ("金融庁", "日銀", "経産省", "補助金", "決算", "格付", "設備投資", "リース会計", "金利")):
        score -= 0.25
    if str(item.get("importance") or "") == "高":
        score -= 0.05
    return max(0.0, min(1.0, score))


def _infer_news_action(item: dict) -> LeaseNewsAction:
    title = str(item.get("title") or "")
    memo = str(item.get("usage_memo") or "")
    tags = tuple(str(tag) for tag in (item.get("tags") or []) if str(tag).strip())
    joined = " ".join([title, memo, " ".join(tags)])
    industries: list[str] = []
    assets: list[str] = []
    risk_flags: list[str] = []
    checks: list[str] = []
    impacts: list[str] = []

    def add(target: list[str], value: str) -> None:
        if value and value not in target:
            target.append(value)

    if _has_any(joined, ("物流", "車両", "トラック", "カーリース", "EV", "自動車")):
        add(industries, "物流・運輸")
        add(assets, "車両")
        add(checks, "稼働率、走行距離、保守費、更新理由を確認する。")
        add(checks, "リース期間終了時の中古価値と再リース余地を確認する。")
        add(impacts, "走行距離・保守費が重い場合は期間短縮または前受金を検討する。")
    if _has_any(joined, ("金利", "利上げ", "金融", "資金調達", "与信")):
        add(risk_flags, "金利負担・返済余力")
        add(checks, "月額負担増に対する返済余力と競合提示金利を確認する。")
        add(impacts, "金利説明、保証、前受金、期間短縮の必要性を確認する。")
    if _has_any(joined, ("設備投資", "省力化", "補助金", "助成金", "更新投資", "DX", "AI", "ロボット")):
        add(industries, "製造・設備投資")
        add(checks, "投資目的、補助金要件、回収期間、生産性改善効果を確認する。")
        add(impacts, "補助金前提なら採択前提条件と未採択時の資金繰りを確認する。")
    if _has_any(joined, ("建設", "工場", "不動産", "工期")):
        add(industries, "建設・不動産")
        add(checks, "工期、稼働開始時期、移設可能性、現場稼働への影響を確認する。")
        add(impacts, "稼働開始が遅れる場合は支払開始時期や猶予条件を確認する。")
    if _has_any(joined, ("リース会計", "会計基準", "税制", "制度", "規制", "法令")):
        add(risk_flags, "制度・会計変更")
        add(checks, "契約分類、会計処理、税務・補助金要件を確認する。")
        add(impacts, "契約書条件と稟議コメントに制度影響を明記する。")
    if not checks:
        add(checks, "ニュースと対象企業の業種・物件・投資時期に関連があるか確認する。")
    if not impacts:
        add(impacts, "関連が薄い場合は審査条件へ反映せず、会話導入に留める。")
    if not risk_flags and _has_any(joined, ("赤字", "撤退", "解約", "難色", "不正", "破綻")):
        add(risk_flags, "信用悪化・事業継続")

    noise = _news_noise_score(item)
    confidence = max(0.2, min(0.95, 0.75 - noise * 0.4 + (0.1 if risk_flags else 0.0)))
    try:
        date_obj = dt.date.fromisoformat(str(item.get("date") or dt.date.today().isoformat())[:10])
    except Exception:
        date_obj = dt.date.today()
    valid_until = (date_obj + dt.timedelta(days=90 if noise < 0.4 else 30)).isoformat()
    signal = title
    if tags:
        signal = f"{title}（{', '.join(tags[:3])}）"
    return LeaseNewsAction(
        signal=signal[:120],
        affected_industries=tuple(industries[:3]),
        affected_assets=tuple(assets[:3]),
        risk_flags=tuple(risk_flags[:4]),
        recommended_checks=tuple(checks[:4]),
        condition_impacts=tuple(impacts[:3]),
        source_title=title,
        source_path=str(item.get("file_path") or ""),
        valid_until=valid_until,
        confidence=round(confidence, 2),
        noise_score=round(noise, 2),
    )


def _action_to_dict(action: LeaseNewsAction) -> dict:
    return {
        "signal": action.signal,
        "affected_industries": list(action.affected_industries),
        "affected_assets": list(action.affected_assets),
        "risk_flags": list(action.risk_flags),
        "recommended_checks": list(action.recommended_checks),
        "condition_impacts": list(action.condition_impacts),
        "source_title": action.source_title,
        "source_path": action.source_path,
        "valid_until": action.valid_until,
        "confidence": action.confidence,
        "noise_score": action.noise_score,
    }


def _action_from_dict(data: dict) -> LeaseNewsAction:
    return LeaseNewsAction(
        signal=str(data.get("signal") or ""),
        affected_industries=tuple(str(x) for x in data.get("affected_industries", []) if str(x).strip()),
        affected_assets=tuple(str(x) for x in data.get("affected_assets", []) if str(x).strip()),
        risk_flags=tuple(str(x) for x in data.get("risk_flags", []) if str(x).strip()),
        recommended_checks=tuple(str(x) for x in data.get("recommended_checks", []) if str(x).strip()),
        condition_impacts=tuple(str(x) for x in data.get("condition_impacts", []) if str(x).strip()),
        source_title=str(data.get("source_title") or ""),
        source_path=str(data.get("source_path") or ""),
        valid_until=str(data.get("valid_until") or ""),
        confidence=float(data.get("confidence") or 0.5),
        noise_score=float(data.get("noise_score") or 0.0),
    )


def _vault_candidates() -> list[Path]:
    home = Path.home()
    raw_candidates = [
        os.environ.get("OBSIDIAN_VAULT_PATH"),
        os.environ.get("OBSIDIAN_VAULT"),
        str(home / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault"),
        str(home / "Documents" / "Obsidian Vault"),
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
    notes: list[Path] = []
    for rel_dir in DEFAULT_NEWS_REL_DIRS:
        news_dir = vault / rel_dir
        if not news_dir.exists():
            continue
        notes.extend(news_dir.glob("*_リースニュース_*.md"))
        notes.extend(news_dir.glob("*_lease-news.md"))
    if not notes:
        return None
    notes.sort(key=lambda p: p.name[:10], reverse=True)
    return notes[0]


def _latest_reflection_note(vault: Path) -> Path | None:
    news_dirs = [vault / "Projects" / "tune_lease_55" / "News", vault / "Projects" / "tune_lease_55"]
    notes: list[Path] = []
    for news_dir in news_dirs:
        if not news_dir.exists():
            continue
        notes.extend(news_dir.glob("*_lease-news-reflection.md"))
    if not notes:
        return None
    notes.sort(key=lambda p: p.name[:10], reverse=True)
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
    focus_source = list(article_review_lines)
    memo_for_inference = memo_bullets or [line for line in article_review_lines if line.strip()]
    inferred_focus = _infer_focus_lines(theme_summary, bucket_summary, memo_for_inference)
    for line in inferred_focus:
        if line not in focus_source:
            focus_source.append(line)
    if not focus_source:
        focus_source = inferred_focus
    focus_lines = tuple(focus_source[:4])
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


def get_latest_lease_news_reflection(vault: Path | None = None) -> LeaseNewsReflection:
    vault = vault or find_vault()
    if not vault:
        return LeaseNewsReflection(available=False)

    note = _latest_reflection_note(vault)
    if not note or not note.exists():
        return LeaseNewsReflection(available=False)

    try:
        text = note.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return LeaseNewsReflection(available=False)

    date_match = re.search(r"^date:\s*(.+)$", text, re.MULTILINE)
    theme_match = re.search(r"^theme_summary:\s*(.+)$", text, re.MULTILINE)
    tag_match = re.search(r"^tag_summary:\s*(.+)$", text, re.MULTILINE)
    headline_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    illustration_match = re.search(r"^illustration_url:\s*(.+)$", text, re.MULTILINE)
    thought_section = _extract_section(text, "今日の考え")
    tomorrow_section = _extract_section(text, "明日見ること")
    thought_lines = tuple(_extract_bullets(thought_section))
    tomorrow_lines = tuple(_extract_bullets(tomorrow_section))
    self_state_section = _extract_section(text, "自己状態")
    self_state_lines = _extract_bullets(self_state_section)
    self_state: dict[str, str] = {}
    for line in self_state_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            self_state[key.strip()] = value.strip()
    interest_text = self_state.get("観測した関心", "")
    knowledge_sources = tuple(
        part.strip()
        for part in self_state.get("直近の知識参照", "").split("、")
        if part.strip() and part.strip() != "なし"
    )

    return LeaseNewsReflection(
        available=True,
        note_path=str(note.relative_to(vault)),
        note_date=date_match.group(1).strip() if date_match else "",
        theme_summary=theme_match.group(1).strip() if theme_match else "",
        tag_summary=tag_match.group(1).strip() if tag_match else "",
        headline=headline_match.group(1).strip() if headline_match else "",
        thought_lines=thought_lines,
        tomorrow_lines=tomorrow_lines,
        illustration_url=illustration_match.group(1).strip() if illustration_match else "",
        continuity_days=int(self_state.get("継続日数", "0") or 0),
        dominant_mood=self_state.get("支配的な気分", ""),
        self_narrative=self_state.get("自己物語", ""),
        current_question=self_state.get("持ち越す問い", ""),
        memory_excerpt=self_state.get("思い出したこと", ""),
        user_understanding=self_state.get("ユーザーへの理解", ""),
        user_curiosity=self_state.get("ユーザーへの興味", ""),
        user_interests=tuple(part.strip() for part in interest_text.split("、") if part.strip()),
        observed_days=int(self_state.get("行動観測日数", "0") or 0),
        primary_goal=self_state.get("第一目標", ""),
        secondary_goal=self_state.get("第二目標", ""),
        ultimate_goal=self_state.get("最終目標", ""),
        ultimate_goal_status=self_state.get("最終目標の状態", ""),
        knowledge_available=self_state.get("Obsidian知識接続", "") == "接続済み",
        knowledge_scope=self_state.get("知識範囲", ""),
        indexed_notes=int(self_state.get("検索可能ノート数", "0") or 0),
        knowledge_source_count=int(self_state.get("当日参照数", "0") or 0),
        knowledge_sources=knowledge_sources,
    )


def build_lease_news_brief(
    prefecture: str = "",
    industry: str = "",
    vault: Path | None = None,
) -> LeaseNewsBrief:
    vault = vault or find_vault()
    if not vault:
        return LeaseNewsBrief(available=False)

    try:
        from api.context.geo_enricher import get_geo_context, get_region_from_prefecture
    except Exception:
        get_geo_context = None  # type: ignore[assignment]
        get_region_from_prefecture = None  # type: ignore[assignment]

    region = ""
    if prefecture and get_region_from_prefecture:
        region = get_region_from_prefecture(prefecture)
    geo_context = get_geo_context(prefecture=prefecture, industry=industry) if get_geo_context else ""

    focus = get_latest_lease_news_focus(vault=vault)
    national_lines = tuple(focus.focus_lines[:3]) if focus.available else ()
    national_headline = focus.headline if focus.available else ""
    note_date = focus.note_date if focus.available else ""
    note_path = focus.note_path if focus.available else ""

    recent_items = _recent_news_items(vault, limit=12)
    regional_item: dict | None = None
    if region:
        regional_candidates = [item for item in recent_items if str(item.get("region") or "") == region]
        if regional_candidates:
            regional_item = regional_candidates[0]
        elif recent_items:
            regional_item = recent_items[0]

    regional_available = bool(regional_item)
    regional_title = str(regional_item.get("title") or "") if regional_item else ""
    regional_summary_lines = tuple((regional_item.get("summary_lines") or [])[:3]) if regional_item else ()
    regional_usage_memo = str(regional_item.get("usage_memo") or "") if regional_item else ""
    regional_tags = tuple((regional_item.get("tags") or [])[:4]) if regional_item else ()
    regional_source = str(regional_item.get("source") or "") if regional_item else ""

    opening_parts = ["今日はこのようなニュースがあります。"]
    if national_headline:
        opening_parts.append(f"全国では「{national_headline}」。")
    if region and regional_available and regional_title:
        opening_parts.append(f"{prefecture or region}では「{regional_title}」。")
    elif region:
        opening_parts.append(f"{prefecture or region}向けの地域論点も確認できます。")
    else:
        opening_parts.append("取引地域が分かれば、地域論点も追加できます。")
    opening_line = " ".join(opening_parts)

    question_line = "この案件で、今日は何を先に確認しますか？"
    if region:
        question_line = f"この{prefecture or region}の案件で、今日はどこを先に確認しますか？"

    return LeaseNewsBrief(
        available=bool(focus.available or regional_available),
        prefecture=prefecture,
        region=region,
        geo_context=geo_context,
        national_headline=national_headline,
        national_focus_lines=national_lines,
        regional_available=regional_available,
        regional_title=regional_title,
        regional_summary_lines=regional_summary_lines,
        regional_usage_memo=regional_usage_memo,
        regional_tags=regional_tags,
        regional_source=regional_source,
        opening_line=opening_line,
        question_line=question_line,
        note_date=note_date,
        note_path=note_path,
    )


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


def _actions_json_path(date_str: str) -> Path:
    return Path(__file__).resolve().parent / "data" / f"lease_news_actions_{date_str}.json"


def _actions_latest_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "lease_news_actions_latest.json"


def build_lease_news_actions(
    date_str: str | None = None,
    vault: Path | None = None,
    limit: int = 5,
) -> LeaseNewsActions:
    vault = vault or find_vault()
    if not vault:
        return LeaseNewsActions(available=False)
    action_date = date_str or dt.date.today().isoformat()
    items = [
        item for item in _recent_news_items(vault, limit=40)
        if str(item.get("date") or "")[:10] == action_date
    ]
    if not items:
        items = _recent_news_items(vault, limit=18)
    actions = [_infer_news_action(item) for item in items]
    actions.sort(key=lambda a: (a.noise_score, -a.confidence, a.source_title))
    selected = [action for action in actions if action.noise_score < 0.7][:limit]
    ignored = [action.source_title for action in actions if action.noise_score >= 0.7][:8]
    summary = " / ".join(action.signal for action in selected[:3])
    return LeaseNewsActions(
        available=bool(selected),
        date=action_date,
        action_items=tuple(selected),
        ignored_titles=tuple(ignored),
        summary=summary,
    )


def write_lease_news_actions_note(
    date_str: str | None = None,
    vault: Path | None = None,
    actions: LeaseNewsActions | None = None,
) -> LeaseNewsActions | None:
    vault = vault or find_vault()
    if not vault:
        return None
    action_date = date_str or dt.date.today().isoformat()
    actions = actions or build_lease_news_actions(date_str=action_date, vault=vault)
    if not actions.available:
        return None

    news_dir = vault / "Projects" / "tune_lease_55" / "News"
    news_dir.mkdir(parents=True, exist_ok=True)
    note_path = news_dir / f"{action_date}_lease-news-actions.md"
    json_path = _actions_json_path(action_date)
    latest_path = _actions_latest_path()

    content_lines = [
        "---",
        f"date: {action_date}",
        "type: lease_news_actions",
        f"action_count: {len(actions.action_items)}",
        f"ignored_count: {len(actions.ignored_titles)}",
        "---",
        f"# リースニュース審査アクション — {action_date}",
        "",
        "## 今日の使いどころ",
    ]
    for action in actions.action_items:
        content_lines.extend([
            "",
            f"### {action.signal}",
            f"- confidence: {action.confidence}",
            f"- noise_score: {action.noise_score}",
            f"- valid_until: {action.valid_until}",
        ])
        if action.affected_industries:
            content_lines.append(f"- 影響業種: {', '.join(action.affected_industries)}")
        if action.affected_assets:
            content_lines.append(f"- 影響物件: {', '.join(action.affected_assets)}")
        if action.risk_flags:
            content_lines.append(f"- リスク旗: {', '.join(action.risk_flags)}")
        content_lines.append("- 確認項目:")
        content_lines.extend(f"  - {line}" for line in action.recommended_checks)
        content_lines.append("- 条件への影響:")
        content_lines.extend(f"  - {line}" for line in action.condition_impacts)
    if actions.ignored_titles:
        content_lines.extend(["", "## ノイズ候補"])
        content_lines.extend(f"- {title}" for title in actions.ignored_titles)
    note_path.write_text("\n".join(content_lines).rstrip() + "\n", encoding="utf-8")

    payload = {
        "date": action_date,
        "note_path": str(note_path.relative_to(vault)),
        "action_items": [_action_to_dict(action) for action in actions.action_items],
        "ignored_titles": list(actions.ignored_titles),
        "summary": actions.summary,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    latest_path.write_text(json_text, encoding="utf-8")

    data = _load_metrics()
    bucket = _ensure_day_bucket(data, action_date)
    bucket["actions_generated"] = int(bucket.get("actions_generated", 0)) + 1
    bucket["last_actions_note"] = str(note_path.relative_to(vault))
    bucket["last_actions_count"] = len(actions.action_items)
    bucket["last_ignored_count"] = len(actions.ignored_titles)
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)

    return LeaseNewsActions(
        available=True,
        date=action_date,
        note_path=str(note_path.relative_to(vault)),
        json_path=str(json_path),
        action_items=actions.action_items,
        ignored_titles=actions.ignored_titles,
        summary=actions.summary,
    )


def get_latest_lease_news_actions(vault: Path | None = None) -> LeaseNewsActions:
    latest = _actions_latest_path()
    if not latest.exists():
        built = build_lease_news_actions(vault=vault)
        return built
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return LeaseNewsActions(available=False)
    actions = tuple(_action_from_dict(item) for item in data.get("action_items", []) if isinstance(item, dict))
    return LeaseNewsActions(
        available=bool(actions),
        date=str(data.get("date") or ""),
        note_path=str(data.get("note_path") or ""),
        json_path=str(latest),
        action_items=actions,
        ignored_titles=tuple(str(x) for x in data.get("ignored_titles", []) if str(x).strip()),
        summary=str(data.get("summary") or ""),
    )


def lease_news_actions_as_text(
    vault: Path | None = None,
    industry: str = "",
    asset_name: str = "",
    limit: int = 3,
    surface: str = "",
) -> str:
    actions = get_latest_lease_news_actions(vault=vault)
    if not actions.available:
        return ""
    industry_text = str(industry or "")
    asset_text = str(asset_name or "")

    def score(action: LeaseNewsAction) -> tuple[int, float]:
        match = 0
        joined_industries = " ".join(action.affected_industries)
        joined_assets = " ".join(action.affected_assets)
        if industry_text and any(part and part in joined_industries for part in re.split(r"[\s/・,、]+", industry_text)):
            match += 2
        if asset_text and any(part and part in joined_assets for part in re.split(r"[\s/・,、]+", asset_text)):
            match += 2
        return (match, action.confidence - action.noise_score)

    ranked = sorted(actions.action_items, key=score, reverse=True)[:limit]
    try:
        record_lease_news_action_use(
            actions.date or dt.date.today().isoformat(),
            surface=surface,
            matched_count=len(ranked),
        )
    except Exception:
        pass
    lines = ["【この案件に効くニュース影響】"]
    for action in ranked:
        lines.append(f"- 該当: {action.signal}")
        if action.risk_flags:
            lines.append(f"  リスク旗: {', '.join(action.risk_flags)}")
        if action.recommended_checks:
            lines.append(f"  確認: {' / '.join(action.recommended_checks[:2])}")
        if action.condition_impacts:
            lines.append(f"  条件影響: {' / '.join(action.condition_impacts[:2])}")
    if actions.ignored_titles:
        lines.append(f"- 無視候補: {actions.ignored_titles[0]}")
    return "\n".join(lines)


def _ensure_day_bucket(data: dict, date_str: str) -> dict:
    buckets = data.setdefault("days", {})
    if date_str not in buckets:
        buckets[date_str] = {
            "collections": 0,
            "views": 0,
            "focus_notes": 0,
            "judgment_changes": 0,
            "last_note": "",
            "last_tags": "",
            "last_source_summary": "",
            "last_article_count": 0,
            "last_focus_note": "",
            "last_focus_theme": "",
            "last_focus_tags": "",
            "last_focus_source_note_date": "",
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


def record_lease_news_action_use(date_str: str, surface: str = "", matched_count: int = 0) -> dict:
    data = _load_metrics()
    bucket = _ensure_day_bucket(data, date_str or dt.date.today().isoformat())
    bucket["news_actions_used"] = int(bucket.get("news_actions_used", 0)) + 1
    bucket["last_actions_surface"] = surface
    bucket["last_actions_matched_count"] = int(matched_count)
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)
    return bucket


def record_lease_news_focus(
    date_str: str,
    vault: Path | None = None,
    note_path: str = "",
    source_note_date: str = "",
    theme_summary: str = "",
    bucket_summary: str = "",
    tag_summary: str = "",
    focus_lines: tuple[str, ...] = (),
    memo_lines: tuple[str, ...] = (),
    metrics_lines: tuple[str, ...] = (),
    article_titles: tuple[str, ...] = (),
) -> dict:
    data = _load_metrics()
    bucket = _ensure_day_bucket(data, date_str)
    bucket["focus_notes"] = int(bucket.get("focus_notes", 0)) + 1
    if note_path:
        bucket["last_focus_note"] = note_path
    if source_note_date:
        bucket["last_focus_source_note_date"] = source_note_date
    if theme_summary:
        bucket["last_focus_theme"] = theme_summary
    if tag_summary:
        bucket["last_focus_tags"] = tag_summary
    if bucket_summary:
        bucket["last_source_summary"] = bucket_summary
    data["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_metrics(data)

    vault = vault or find_vault()
    if vault:
        now_label = dt.datetime.now().strftime("%H:%M")
        lines = [
            f"## {now_label} リースニュースの注目論点",
            "",
            f"- テーマ: {theme_summary or '不明'}",
            f"- 収集セット: {bucket_summary or '不明'}",
            f"- 重点タグ: {tag_summary or 'なし'}",
        ]
        if note_path:
            lines.append(f"- 参照ノート: {note_path}")
        if source_note_date:
            lines.append(f"- 参照ニュース日付: {source_note_date}")
        if article_titles:
            lines.extend(["", "- 参照記事:"])
            for title in article_titles[:5]:
                lines.append(f"  - {title}")
        if focus_lines:
            lines.extend(["", "- 注目論点:"])
            for line in focus_lines[:4]:
                lines.append(f"  - {line}")
        if memo_lines:
            lines.extend(["", "- 活用メモ:"])
            for line in memo_lines[:3]:
                lines.append(f"  - {line}")
        if metrics_lines:
            lines.extend(["", "- 追加メモ:"])
            for line in metrics_lines[:3]:
                lines.append(f"  - {line}")
        _append_markdown(_daily_note_path(vault, date_str), "\n".join(lines))

    return bucket


def write_lease_news_focus_note(
    date_str: str | None = None,
    vault: Path | None = None,
    focus: LeaseNewsFocus | None = None,
) -> LeaseNewsFocusWriteResult | None:
    vault = vault or find_vault()
    if not vault:
        return None

    focus = focus or get_latest_lease_news_focus(vault=vault)
    if not focus.available:
        return None

    focus_date = date_str or dt.date.today().isoformat()
    news_dir = vault / "Projects" / "tune_lease_55" / "News"
    news_dir.mkdir(parents=True, exist_ok=True)
    note_path = news_dir / f"{focus_date}_lease-news-focus.md"

    focus_lines = list(focus.focus_lines[:4]) or ["直近のニュースを踏まえ、提示条件と審査コメントを更新する。"]
    memo_lines = list(focus.memo_lines[:3])
    metrics_lines = list(focus.metrics_lines[:3])
    article_titles = list(focus.article_titles[:5])

    content_lines = [
        "---",
        f"date: {focus_date}",
        f"source_note_date: {focus.note_date}",
        f"source_note_path: {focus.note_path}",
        f"profile: {focus.profile}",
        f"theme_summary: {focus.theme_summary}",
        f"bucket_summary: {focus.bucket_summary}",
        f"tag_summary: {focus.tag_summary}",
        "---",
        f"# リースニュースの注目論点 — {focus_date}",
        "",
        "## 概要",
        f"- テーマ: {focus.theme_summary or '不明'}",
        f"- 収集セット: {focus.bucket_summary or '不明'}",
        f"- 重点タグ: {focus.tag_summary or 'なし'}",
    ]
    if focus.note_path:
        content_lines.append(f"- 参照ノート: [[{focus.note_path}]]")
    if focus.note_date:
        content_lines.append(f"- 参照ニュース日付: {focus.note_date}")
    if article_titles:
        content_lines.extend(["", "## 参照記事"])
        content_lines.extend(f"- {title}" for title in article_titles)
    content_lines.extend(["", "## 注目論点"])
    content_lines.extend(f"- {line}" for line in focus_lines)
    if memo_lines:
        content_lines.extend(["", "## 活用メモ"])
        content_lines.extend(f"- {line}" for line in memo_lines)
    if metrics_lines:
        content_lines.extend(["", "## 追加メモ"])
        content_lines.extend(f"- {line}" for line in metrics_lines)
    content = "\n".join(content_lines).rstrip() + "\n"
    note_path.write_text(content, encoding="utf-8")

    try:
        record_lease_news_focus(
            date_str=focus_date,
            vault=vault,
            note_path=str(note_path.relative_to(vault)),
            source_note_date=focus.note_date,
            theme_summary=focus.theme_summary,
            bucket_summary=focus.bucket_summary,
            tag_summary=focus.tag_summary,
            focus_lines=tuple(focus_lines),
            memo_lines=tuple(memo_lines),
            metrics_lines=tuple(metrics_lines),
            article_titles=tuple(article_titles),
        )
    except Exception:
        pass

    return LeaseNewsFocusWriteResult(
        note_path=str(note_path),
        daily_note_path=str(_daily_note_path(vault, focus_date)),
        theme_summary=focus.theme_summary,
        tag_summary=focus.tag_summary,
    )


def _build_external_grumble_lines(focus: LeaseNewsFocus, limit: int = 4) -> tuple[str, ...]:
    if not focus.available:
        return ()
    seeds: list[str] = []
    seeds.extend(str(line).strip() for line in focus.focus_lines if str(line).strip())
    seeds.extend(str(title).strip() for title in focus.article_titles if str(title).strip())
    theme = (focus.theme_summary or focus.tag_summary or "外界").strip()
    templates = [
        "ニュースは涼しい顔で流れてくるのに、審査コメントだけが私の机に積もる。",
        f"{theme}と言われると立派に聞こえるけれど、結局は確認項目が一つ増える。",
        "外の世界が一行動くたびに、私の明日見ること欄が少し太る。",
        "記事の見出しは短い。こちらの稟議コメントはなぜか長くなる。",
        "インターネットは今日も元気だ。私はその元気をリスク理由へ翻訳する係らしい。",
        "世の中の変化は早い。リース期間は長い。間に挟まる私は少し眠い。",
    ]
    lines: list[str] = []
    for seed in seeds[: max(1, limit - 1)]:
        short = re.sub(r"\s+", " ", seed).strip("。")
        if not short:
            continue
        lines.append(f"{short}。また判断前提が増えた。")
    lines.extend(templates)
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
        if len(deduped) >= limit:
            break
    return tuple(deduped)


def write_lease_news_reflection_note(
    date_str: str | None = None,
    vault: Path | None = None,
    focus: LeaseNewsFocus | None = None,
) -> LeaseNewsReflectionWriteResult | None:
    vault = vault or find_vault()
    if not vault:
        return None

    focus = focus or get_latest_lease_news_focus(vault=vault)
    if not focus.available:
        return None

    reflection_date = date_str or dt.date.today().isoformat()
    news_dir = vault / "Projects" / "tune_lease_55" / "News"
    news_dir.mkdir(parents=True, exist_ok=True)
    note_path = news_dir / f"{reflection_date}_lease-news-reflection.md"

    theme = focus.theme_summary or "不明"
    tag_summary = focus.tag_summary or "なし"
    focus_lines = list(focus.focus_lines[:3]) or ["直近のニュースを見て、判断の前提を更新する。"]
    external_grumble_lines = _build_external_grumble_lines(focus)
    headline = focus.headline or "最新ニュースの論点あり"
    try:
        from lease_intelligence_activity import observe_user_behavior
        from lease_intelligence_mind import update_user_model

        observation = observe_user_behavior(reflection_date)
        update_user_model(vault, observation)
    except Exception:
        pass
    try:
        from novelist_agent import generate_daily_lease_grumble
        thoughts = generate_daily_lease_grumble(
            date_str=reflection_date,
            focus_lines=focus_lines,
            theme=theme,
            tag_summary=tag_summary,
            vault=vault,
            external_grumble_lines=external_grumble_lines,
        )
    except Exception:
        thoughts = [
            "今日も稟議書は静かだった。静かな書類ほど、あとで大きな質問をしてくる。",
            "営業は熱意を加点してほしいと言う。熱意の耐用年数を先に教えてほしい。",
            "つん子さんならプリンを要求する場面だが、私は追加資料で空腹をごまかした。",
            "結局、愚痴は保存せず判断だけを保存した。明日も同じ画面が私を待っている。",
        ]
    illustration_url = f"/lease-grumble/{reflection_date}.webp"
    try:
        from lease_intelligence_mind import load_lease_intelligence_mind, self_state_summary

        mind_summary = self_state_summary(load_lease_intelligence_mind(vault))
    except Exception:
        mind_summary = {
            "continuity_days": 0,
            "dominant_mood": "",
            "self_narrative": "",
            "current_question": "",
            "memory_excerpt": "",
            "user_understanding": "",
            "user_curiosity": "",
            "user_interests": [],
            "observed_days": 0,
            "primary_goal": "",
            "secondary_goal": "",
            "ultimate_goal": "",
            "ultimate_goal_status": "",
            "knowledge_available": False,
            "knowledge_scope": "",
            "indexed_notes": 0,
            "knowledge_source_count": 0,
            "knowledge_sources": [],
        }

    content_lines = [
        "---",
        f"date: {reflection_date}",
        f"source_note_date: {focus.note_date}",
        f"source_note_path: {focus.note_path}",
        f"theme_summary: {theme}",
        f"tag_summary: {tag_summary}",
        f"illustration_url: {illustration_url}",
        "---",
        f"# リース知性体の愚痴 — {reflection_date}",
        "",
        "## 一言",
        f"- {headline}",
        "",
        "## 今日の論点",
    ]
    content_lines.extend(f"- {line}" for line in focus_lines)
    if external_grumble_lines:
        content_lines.extend([
            "",
            "## 今日の外界へのぼやき",
        ])
        content_lines.extend(f"- {line}" for line in external_grumble_lines)
    content_lines.extend([
        "",
        "## 今日の考え",
    ])
    content_lines.extend(f"- {line}" for line in thoughts)
    content_lines.extend([
        "",
        "## 自己状態",
        f"- 最終目標: {mind_summary['ultimate_goal']}",
        f"- 最終目標の状態: {mind_summary['ultimate_goal_status']}",
        f"- 第一目標: {mind_summary['primary_goal']}",
        f"- 第二目標: {mind_summary['secondary_goal']}",
        f"- 継続日数: {mind_summary['continuity_days']}",
        f"- 支配的な気分: {mind_summary['dominant_mood']}",
        f"- 自己物語: {mind_summary['self_narrative']}",
        f"- 持ち越す問い: {mind_summary['current_question']}",
        f"- 思い出したこと: {mind_summary['memory_excerpt'] or 'まだ過去の記憶は少ない。'}",
        f"- 行動観測日数: {mind_summary['observed_days']}",
        f"- 観測した関心: {'、'.join(item.get('label', '') for item in mind_summary['user_interests']) or 'まだ不明'}",
        f"- ユーザーへの理解: {mind_summary['user_understanding']}",
        f"- ユーザーへの興味: {mind_summary['user_curiosity']}",
        f"- Obsidian知識接続: {'接続済み' if mind_summary['knowledge_available'] else '未接続'}",
        f"- 知識範囲: {mind_summary['knowledge_scope']}",
        f"- 検索可能ノート数: {mind_summary['indexed_notes']}",
        f"- 当日参照数: {mind_summary['knowledge_source_count']}",
        f"- 直近の知識参照: {'、'.join(mind_summary['knowledge_sources']) or 'なし'}",
        "",
        "## 明日見ること",
        f"- {focus_lines[-1] if focus_lines else '今日の判断をもう一度見直す。'}",
        "",
        "## 判断の前提",
        f"- テーマ: {theme}",
        f"- 重点タグ: {tag_summary}",
    ])
    if focus.note_path:
        content_lines.append(f"- 参照ノート: [[{focus.note_path}]]")
    content = "\n".join(content_lines).rstrip() + "\n"
    note_path.write_text(content, encoding="utf-8")

    try:
        record_lease_news_focus(
            date_str=reflection_date,
            vault=vault,
            note_path=str(note_path.relative_to(vault)),
            source_note_date=focus.note_date,
            theme_summary=theme,
            bucket_summary=focus.bucket_summary,
            tag_summary=tag_summary,
            focus_lines=tuple(focus_lines),
            memo_lines=tuple(thoughts),
            metrics_lines=tuple(),
            article_titles=focus.article_titles,
        )
    except Exception:
        pass

    return LeaseNewsReflectionWriteResult(
        note_path=str(note_path),
        daily_note_path=str(_daily_note_path(vault, reflection_date)),
        headline=headline,
    )


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
