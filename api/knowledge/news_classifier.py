"""Classify recent lease-news notes into screening-oriented summary axes."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

NEWS_DIRS = (
    Path("05-クリップ_記事") / "業界リスクニュース",
    Path("業界リスクニュース"),
    Path("05-クリップ_記事") / "リースニュース",
    Path("リースニュース"),
)

LATEST_SUMMARY_PATH = Path(__file__).resolve().parents[2] / "data" / "news_classified_summary_latest.json"

INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "建設業": ("建設", "工事", "建機", "建設機械", "資材", "公共工事", "不動産", "住宅"),
    "製造業": ("製造", "工場", "工作機械", "生産", "半導体", "自動化", "ロボット", "設備投資"),
    "運輸業": ("運輸", "運送", "物流", "トラック", "配送", "倉庫", "燃料費", "2024年問題", "車両"),
    "医療・介護": ("医療", "介護", "病院", "診療", "クリニック", "介護報酬", "医療機器", "福祉"),
    "飲食・宿泊": ("飲食", "宿泊", "ホテル", "旅館", "外食", "観光", "インバウンド", "厨房"),
}

SOCIAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "物価高": ("物価", "価格", "値上げ", "コスト", "原材料", "燃料費", "賃上げ", "インフレ"),
    "人手不足": ("人手不足", "人材不足", "採用難", "省力化", "賃上げ", "労働力", "2024年問題"),
    "サプライチェーン問題": ("供給網", "サプライチェーン", "部品不足", "納期", "調達", "物流混乱", "在庫"),
    "環境規制": ("脱炭素", "環境規制", "排出", "CO2", "EV", "省エネ", "GX", "規制"),
    "地政学リスク": ("地政学", "中国", "米国", "台湾", "中東", "ウクライナ", "関税", "制裁"),
}

FINANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "金利動向": ("金利", "日銀", "利上げ", "利下げ", "政策金利", "長期金利", "基準金利"),
    "為替変動": ("為替", "円安", "円高", "ドル", "ユーロ", "輸入価格"),
    "株価": ("株価", "日経平均", "TOPIX", "市場", "上場", "投資家"),
    "信用市場": ("信用", "与信", "融資", "貸出", "資金繰り", "社債", "延滞"),
    "倒産件数": ("倒産", "破産", "廃業", "民事再生", "資金ショート", "件数"),
}

AXIS_DEFINITIONS = {
    "industry": {"label": "業種別", "categories": INDUSTRY_KEYWORDS},
    "social": {"label": "社会情勢", "categories": SOCIAL_KEYWORDS},
    "finance": {"label": "金融情報", "categories": FINANCE_KEYWORDS},
}


def _parse_jsonish(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text.strip('"')


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, re.DOTALL)
    if not match:
        return {}, raw
    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = _parse_jsonish(value)
    return meta, raw[match.end() :]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,、]", value) if item.strip()]
    return []


def _extract_section(raw: str, heading: str) -> str:
    match = re.search(rf"^## {re.escape(heading)}\s*\n(.+?)(?:\n## |\Z)", raw, re.DOTALL | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_bullets(section: str, limit: int = 5) -> list[str]:
    out: list[str] = []
    for line in section.splitlines():
        text = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if text:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def parse_news_note(path: Path, vault: Path | None = None) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    meta, body = _parse_frontmatter(raw)
    title_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    summary_lines = _extract_bullets(_extract_section(body, "3行要約"), limit=3)
    usage_memo = _extract_section(body, "活用メモ")
    screening_checks = _extract_bullets(_extract_section(body, "AI審査分類"), limit=8)
    detail = _extract_section(body, "詳細")
    link_match = re.search(r"^-\s+link:\s*(.+)$", detail, re.MULTILINE)
    rel_path = str(path)
    if vault:
        try:
            rel_path = str(path.relative_to(vault))
        except ValueError:
            pass
    return {
        "date": str(meta.get("date") or ""),
        "title": title_match.group(1).strip() if title_match else path.stem,
        "summary_lines": summary_lines,
        "usage_memo": usage_memo,
        "tags": _as_list(meta.get("tags")),
        "industries": _as_list(meta.get("industries")),
        "lease_assets": _as_list(meta.get("lease_assets")),
        "impact_direction": str(meta.get("impact_direction") or "neutral"),
        "source_reliability": str(meta.get("source_reliability") or "medium"),
        "importance": str(meta.get("importance") or "通常"),
        "source": str(meta.get("source") or ""),
        "article_url": str(meta.get("canonical_url") or (link_match.group(1).strip() if link_match else "")),
        "file_path": rel_path,
        "screening_checks": screening_checks,
        "text": " ".join([body, json.dumps(meta, ensure_ascii=False)]),
    }


def load_recent_news_notes(vault: Path, *, limit: int = 30, days: int = 14) -> list[dict[str, Any]]:
    cutoff = dt.date.today() - dt.timedelta(days=max(1, days))
    paths: list[Path] = []
    for rel_dir in NEWS_DIRS:
        directory = vault / rel_dir
        if directory.exists():
            paths.extend(directory.glob("*.md"))

    notes: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
        note = parse_news_note(path, vault=vault)
        if not note:
            continue
        try:
            note_date = dt.date.fromisoformat(str(note.get("date") or ""))
            if note_date < cutoff:
                continue
        except ValueError:
            pass
        notes.append(note)
        if len(notes) >= limit:
            break
    return notes


def _match_categories(note: dict[str, Any], categories: dict[str, tuple[str, ...]]) -> list[str]:
    haystack = str(note.get("text") or "").lower()
    matches: list[tuple[str, int]] = []
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword.lower() in haystack)
        if score:
            matches.append((category, score))
    matches.sort(key=lambda item: (-item[1], item[0]))
    return [category for category, _ in matches[:3]]


def classify_note_axes(note: dict[str, Any]) -> dict[str, list[str]]:
    return {
        axis: _match_categories(note, definition["categories"])
        for axis, definition in AXIS_DEFINITIONS.items()
    }


def _impact_profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    joined = " ".join(
        " ".join([item.get("title", ""), item.get("usage_memo", ""), *item.get("summary_lines", [])])
        for item in items
    )
    negative_terms = ("倒産", "破産", "減益", "資金繰り", "価格高騰", "人手不足", "納期", "円安", "金利")
    positive_terms = ("補助金", "助成金", "省力化", "効率化", "回復", "増益", "投資", "需要")
    negative = sum(term in joined for term in negative_terms)
    positive = sum(term in joined for term in positive_terms)
    direction = "negative" if negative > positive else "positive" if positive > negative else "mixed"
    repayment = "返済能力は中立材料として扱い、個社の受注・粗利・資金繰りで確認する。"
    residual = "残価リスクは物件種別と中古流通の変化を個別に確認する。"
    opportunity = "事業機会は投資回収時期と補助金・省力化効果を確認する。"
    if direction == "negative":
        repayment = "返済能力は悪化方向のシナリオを置き、価格転嫁・資金繰り・借入余力を厚く確認する。"
        residual = "需要鈍化やコスト高が物件稼働・処分価値へ波及しないか見る。"
    elif direction == "positive":
        repayment = "返済能力は改善余地があるが、効果発現時期と継続性を確認する。"
        opportunity = "更新投資・省力化投資・補助金活用の提案余地がある。"
    return {
        "direction": direction,
        "repayment_capacity": repayment,
        "residual_value": residual,
        "business_opportunity": opportunity,
    }


def _category_summary(axis: str, category: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    top_titles = [str(item.get("title") or "") for item in items[:5] if item.get("title")]
    key_points: list[str] = []
    for item in items:
        for line in item.get("summary_lines") or []:
            text = str(line).strip()
            if text and text not in key_points:
                key_points.append(text)
        if len(key_points) >= 4:
            break
    impact = _impact_profile(items)
    trend = f"{category}で{len(items)}件の関連ニュース。"
    if impact["direction"] == "negative":
        trend += " 審査では保守的な資金繰り確認を優先。"
    elif impact["direction"] == "positive":
        trend += " 投資機会として見られるが、効果時期の確認が必要。"
    else:
        trend += " 個社条件に応じて良悪両面を分けて見る。"
    return {
        "axis": axis,
        "axis_label": AXIS_DEFINITIONS[axis]["label"],
        "category": category,
        "article_count": len(items),
        "trend": trend,
        "key_points": key_points[:4],
        "lease_implications": impact,
        "recommended_checks": _recommended_checks(axis, category, impact["direction"]),
        "articles": [
            {
                key: item.get(key)
                for key in (
                    "date",
                    "title",
                    "summary_lines",
                    "usage_memo",
                    "source",
                    "article_url",
                    "file_path",
                    "importance",
                    "impact_direction",
                    "source_reliability",
                )
            }
            for item in items[:8]
        ],
    }


def _recommended_checks(axis: str, category: str, direction: str) -> list[str]:
    base = {
        "industry": [f"{category}の売上・受注・稼働率に同じ外部要因が出ていないか確認する。"],
        "social": [f"{category}が粗利率、納期、人員計画に与える影響を確認する。"],
        "finance": [f"{category}がリース料、競合条件、借入余力に与える影響を確認する。"],
    }[axis]
    if direction == "negative":
        base.append("返済予定表にストレスを置き、資金繰り悪化時の支援余地を確認する。")
    elif direction == "positive":
        base.append("投資効果を過大評価せず、回収期間と実績反映タイミングを確認する。")
    else:
        base.append("ニュースだけで判断を変えず、個社実績と物件保全で裏取りする。")
    return base


def build_classified_news_summary(notes: list[dict[str, Any]], *, generated_at: str | None = None) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {axis: {} for axis in AXIS_DEFINITIONS}
    classified_notes: list[dict[str, Any]] = []
    for note in notes:
        axes = classify_note_axes(note)
        note = {**note, "classifications": axes}
        classified_notes.append(note)
        for axis, categories in axes.items():
            for category in categories:
                grouped[axis].setdefault(category, []).append(note)

    axis_summaries = []
    for axis, categories in grouped.items():
        category_summaries = [
            _category_summary(axis, category, items)
            for category, items in sorted(categories.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
        axis_summaries.append(
            {
                "axis": axis,
                "label": AXIS_DEFINITIONS[axis]["label"],
                "category_count": len(category_summaries),
                "article_count": sum(item["article_count"] for item in category_summaries),
                "categories": category_summaries,
            }
        )

    top_categories = [
        category
        for axis in axis_summaries
        for category in axis["categories"]
    ]
    top_categories.sort(key=lambda item: (-int(item["article_count"]), item["axis_label"], item["category"]))
    return {
        "available": bool(notes),
        "generated_at": generated_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": "obsidian_news_notes",
        "article_count": len(notes),
        "axes": axis_summaries,
        "top_insights": [
            {
                "label": f"{item['axis_label']} / {item['category']}",
                "trend": item["trend"],
                "repayment_capacity": item["lease_implications"]["repayment_capacity"],
            }
            for item in top_categories[:5]
        ],
        "articles": [
            {
                key: note.get(key)
                for key in ("date", "title", "source", "article_url", "file_path", "classifications")
            }
            for note in classified_notes[:20]
        ],
    }


def build_classified_news_summary_from_vault(vault: Path, *, limit: int = 30, days: int = 14) -> dict[str, Any]:
    return build_classified_news_summary(load_recent_news_notes(vault, limit=limit, days=days))


def write_classified_news_summary(vault: Path, *, limit: int = 30, days: int = 14, path: Path | None = None) -> dict[str, Any]:
    summary = build_classified_news_summary_from_vault(vault, limit=limit, days=days)
    target = path or LATEST_SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_latest_classified_news_summary(path: Path | None = None) -> dict[str, Any]:
    target = path or LATEST_SUMMARY_PATH
    try:
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"available": False, "article_count": 0, "axes": [], "top_insights": [], "articles": []}
