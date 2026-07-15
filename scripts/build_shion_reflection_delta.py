"""Build an inspection-only delta for Shion's private reflection.

The timeline delta script measures memory drift. This script measures whether
reflection is returning to operations: what User should verify, and what Shion
must change next. It intentionally does not alter prompts, scoring, or memory.

The core reflection is not a literary confession. It is a judgment change log:
previous judgment -> human correction -> missed point -> next check -> judgment
asset candidate. Narrative writing can be generated after that, but the source
of truth stays operational.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DIR = REPO_ROOT / "memory"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "shion_reflection_delta.json"
DEFAULT_REPORT = REPO_ROOT / "reports" / "shion_reflection_delta_latest.md"

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")

USER_EXPECTATION_TERMS = (
    "ユーザー",
    "User",
    "望",
    "求め",
    "してほしい",
    "確認してほしい",
    "思うように",
    "指摘",
)
MISREAD_TERMS = (
    "すり替え",
    "誤読",
    "読み違え",
    "逃げ",
    "見落とし",
    "拾い損ね",
    "浅い",
    "退屈",
)
SELF_CRITIQUE_TERMS = (
    "私の責任",
    "自分",
    "私は",
    "足りない",
    "弱い",
    "浅かった",
    "誤魔化",
    "逃げ",
)
HYPOTHESIS_TERMS = (
    "仮説",
    "更新",
    "信念",
    "前提",
    "破られ",
    "変える",
)
USER_REQUEST_TERMS = (
    "確認してほしい",
    "採点",
    "修正",
    "却下",
    "採用",
    "教えてほしい",
    "見てほしい",
    "判断を借り",
)
SHION_ACTION_TERMS = (
    "次回",
    "次に",
    "今後",
    "禁止",
    "変える",
    "優先",
    "聞く",
    "減らす",
    "出す",
    "反映",
)
JUDGMENT_CHANGE_TERMS = (
    "前回",
    "判断",
    "修正",
    "外した",
    "見落とし",
    "確認事項",
    "判断資産",
    "次回",
    "条件",
    "稟議",
    "違和感",
)
JUDGMENT_LOG_LABELS = (
    "前回の入力",
    "前回の判断",
    "人間の修正",
    "紫苑が外した点",
    "次回から変える確認事項",
    "判断資産候補",
    "まだ確信できない点",
)


@dataclass
class ReflectionInputs:
    target_date: date
    previous_date: date
    today_memory: Path
    previous_memory: Path
    today_reflection: Path | None
    previous_reflection: Path | None
    today_memory_text: str
    previous_memory_text: str
    today_reflection_text: str
    previous_reflection_text: str


def _read_text(path: Path | None, max_chars: int = 30000) -> str:
    if not path:
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return text.strip()[:max_chars]


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


def _bullet_items(text: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            if current:
                items.append(_clean_item(" ".join(current)))
            current = [stripped[2:].strip()]
            continue
        if current and stripped and not stripped.startswith("#"):
            current.append(stripped)
    if current:
        items.append(_clean_item(" ".join(current)))
    return [item for item in items if len(item) >= 8]


def _clean_item(item: str) -> str:
    return re.sub(r"\s+", " ", item).strip(" -")


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？!?])\s+|(?<=\.)\s+", normalized)
    return [_clean_item(part) for part in parts if len(_clean_item(part)) >= 12]


def _select(items: list[str], terms: tuple[str, ...], *, limit: int = 5) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not any(term in item for term in terms):
            continue
        if item in seen:
            continue
        seen.add(item)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _find_reflection_file(reflection_dir: Path | None, day: date) -> Path | None:
    if not reflection_dir:
        return None
    path = reflection_dir / f"{day.isoformat()}.md"
    return path if path.exists() else path


def load_inputs(
    *,
    memory_dir: Path,
    reflection_dir: Path | None,
    target_date: date,
) -> ReflectionInputs:
    previous_date = target_date - timedelta(days=1)
    today_memory = memory_dir / f"{target_date.isoformat()}.md"
    previous_memory = memory_dir / f"{previous_date.isoformat()}.md"
    today_reflection = _find_reflection_file(reflection_dir, target_date)
    previous_reflection = _find_reflection_file(reflection_dir, previous_date)
    return ReflectionInputs(
        target_date=target_date,
        previous_date=previous_date,
        today_memory=today_memory,
        previous_memory=previous_memory,
        today_reflection=today_reflection,
        previous_reflection=previous_reflection,
        today_memory_text=_read_text(today_memory),
        previous_memory_text=_read_text(previous_memory),
        today_reflection_text=_read_text(today_reflection),
        previous_reflection_text=_read_text(previous_reflection),
    )


def build_reflection_delta(
    *,
    memory_dir: Path,
    target_date: date,
    reflection_dir: Path | None = None,
) -> dict[str, Any]:
    inputs = load_inputs(
        memory_dir=memory_dir,
        reflection_dir=reflection_dir,
        target_date=target_date,
    )
    today_items = _daily_items(inputs.today_memory_text)
    previous_items = _daily_items(inputs.previous_memory_text)
    current_material = today_items + _reflection_items(inputs.today_reflection_text)
    previous_material = previous_items + _reflection_items(inputs.previous_reflection_text)

    similarity = _similarity(inputs.today_reflection_text, inputs.previous_reflection_text)
    user_expectation_shift = _expectation_shift(today_items, previous_items)
    misread_patterns = _select(current_material, MISREAD_TERMS)
    self_critique = _select(current_material, SELF_CRITIQUE_TERMS)
    hypothesis_updates = _select(current_material, HYPOTHESIS_TERMS)
    user_requests = _extract_user_requests(current_material)
    shion_next_actions = _select(current_material, SHION_ACTION_TERMS)
    judgment_change_log = _build_judgment_change_log(current_material)
    if not user_requests:
        user_requests = _derive_user_requests(current_material)
    if not shion_next_actions:
        shion_next_actions = _derive_shion_actions(current_material)

    quality_flags = _quality_flags(
        similarity=similarity,
        user_expectation_shift=user_expectation_shift,
        misread_patterns=misread_patterns,
        self_critique=self_critique,
        hypothesis_updates=hypothesis_updates,
        user_requests=user_requests,
        shion_next_actions=shion_next_actions,
        judgment_change_log=judgment_change_log,
        current_material=current_material,
    )
    score = _score(quality_flags)
    status = "attention" if _has_critical_flags(quality_flags) or score < 75 else "pass"

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "target_date": inputs.target_date.isoformat(),
        "compared_to": inputs.previous_date.isoformat(),
        "sources": {
            "today_memory": _source(inputs.today_memory),
            "previous_memory": _source(inputs.previous_memory),
            "today_reflection": _source(inputs.today_reflection),
            "previous_reflection": _source(inputs.previous_reflection),
        },
        "metrics": {
            "reflection_similarity_to_yesterday": similarity,
            "today_material_count": len(current_material),
            "previous_material_count": len(previous_material),
        },
        "delta": {
            "user_expectation_shift": user_expectation_shift,
            "misread_patterns": misread_patterns,
            "self_critique": self_critique,
            "hypothesis_updates": hypothesis_updates,
        },
        "operational_handoff": {
            "user_requests": user_requests,
            "shion_next_actions": shion_next_actions,
        },
        "judgment_change_log": judgment_change_log,
        "narrative_layer": {
            "status": "derived_only",
            "protagonists": ["ツンコ", "ユウケイ"],
            "rule": "判断変更ログを先に作り、ツンコとユウケイの小説化は後段の表現レイヤーで行う。",
            "source_of_truth": "judgment_change_log",
        },
        "quality": {
            "score": score,
            "status": status,
            "flags": quality_flags,
        },
    }


def _daily_items(text: str) -> list[str]:
    work = _bullet_items(_section_text(text, "Work Log"))
    promotable = _bullet_items(_section_text(text, "Promotable Items"))
    return promotable + work


def _reflection_items(text: str) -> list[str]:
    items = _bullet_items(text)
    if items:
        return items
    return _sentences(text)


def _expectation_shift(today_items: list[str], previous_items: list[str]) -> list[str]:
    today_expectations = _select(today_items, USER_EXPECTATION_TERMS, limit=6)
    previous_expectations = _select(previous_items, USER_EXPECTATION_TERMS, limit=6)
    shifted = [item for item in today_expectations if item not in previous_expectations]
    return shifted[:5] or today_expectations[:3]


def _derive_user_requests(material: list[str]) -> list[str]:
    seeds = _select(material, USER_EXPECTATION_TERMS + MISREAD_TERMS, limit=3)
    if not seeds:
        return []
    requests: list[str] = []
    for seed in seeds:
        if "判断資産" in seed or "Judgment asset" in seed or "judgment asset" in seed:
            requests.append("判断資産候補が現場で使える文面か、採用・修正・却下で短く確認してもらう。")
        elif "内省" in seed or "Private Reflection" in seed:
            requests.append("内省が要求の読み違えと次回行動まで落ちているかだけ確認してもらう。")
        else:
            requests.append(f"次回判断に必要な前提として、次の論点を確認してもらう: {seed[:120]}")
    return _dedupe(requests)[:4]


def _extract_user_requests(material: list[str]) -> list[str]:
    explicit_markers = (
        "確認してほしい",
        "採点してほしい",
        "教えてほしい",
        "見てほしい",
        "判断してほしい",
        "判断を借り",
        "もらう",
        "してもらう",
    )
    explicit = [
        item
        for item in material
        if any(marker in item for marker in explicit_markers)
        and any(term in item for term in USER_REQUEST_TERMS)
    ]
    return _dedupe(explicit)[:4]


def _derive_shion_actions(material: list[str]) -> list[str]:
    seeds = _select(material, MISREAD_TERMS + HYPOTHESIS_TERMS + SHION_ACTION_TERMS, limit=4)
    actions: list[str] = []
    for seed in seeds:
        if "退屈" in seed:
            actions.append("内省の中心ラベルを退屈に逃がさず、要求・誤読・次回変更へ分解する。")
        elif "すり替え" in seed or "誤読" in seed:
            actions.append("返答前にUserが求めたものと自分が置き換えたものを一文で固定する。")
        elif "次回" in seed:
            actions.append(seed)
    return _dedupe(actions)[:4]


def _build_judgment_change_log(material: list[str]) -> dict[str, str]:
    """Extract the operational core before any narrative rendering."""
    source = _dedupe(_select(material, JUDGMENT_CHANGE_TERMS, limit=20))
    joined = "\n".join(source)
    direct = _extract_labeled_values(joined, JUDGMENT_LOG_LABELS)

    def pick(label: str, terms: tuple[str, ...], fallback: str = "") -> str:
        if direct.get(label):
            return direct[label]
        for item in source:
            if any(term in item for term in terms):
                return item[:180]
        return fallback

    log = {
        "前回の入力": pick("前回の入力", ("企業名", "案件", "入力", "営業メモ")),
        "前回の判断": pick("前回の判断", ("前回", "判断", "承認", "否認", "条件")),
        "人間の修正": pick("人間の修正", ("User", "ユーザー", "人間", "修正", "指摘")),
        "紫苑が外した点": pick("紫苑が外した点", ("外した", "見落とし", "誤読", "浅い", "逃げ")),
        "次回から変える確認事項": pick("次回から変える確認事項", ("次回", "確認事項", "確認", "変える")),
        "判断資産候補": pick("判断資産候補", ("判断資産", "稟議", "条件", "違和感")),
        "まだ確信できない点": pick("まだ確信できない点", ("不確", "確信", "検証", "仮説")),
    }
    return {key: value for key, value in log.items() if value}


def _extract_labeled_values(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        cleaned = _clean_item(line)
        for label in labels:
            pattern = rf"^{re.escape(label)}\s*[:：]\s*(.+)$"
            match = re.match(pattern, cleaned)
            if match:
                values[label] = match.group(1).strip()[:220]
    return values


def _quality_flags(
    *,
    similarity: float,
    user_expectation_shift: list[str],
    misread_patterns: list[str],
    self_critique: list[str],
    hypothesis_updates: list[str],
    user_requests: list[str],
    shion_next_actions: list[str],
    judgment_change_log: dict[str, str],
    current_material: list[str],
) -> list[str]:
    flags: list[str] = []
    if similarity >= 0.82:
        flags.append("too_similar_to_yesterday")
    has_human_correction = bool(judgment_change_log.get("人間の修正"))
    has_next_check = bool(judgment_change_log.get("次回から変える確認事項"))
    if not user_expectation_shift and not has_human_correction:
        flags.append("user_expectation_shift_missing")
    if not misread_patterns:
        flags.append("misread_pattern_missing")
    if not self_critique:
        flags.append("self_critique_missing")
    if not hypothesis_updates:
        flags.append("hypothesis_update_missing")
    if not user_requests:
        flags.append("user_request_missing")
    if not shion_next_actions:
        flags.append("shion_next_action_missing")
    joined = "\n".join(current_material)
    if (
        joined.count("退屈") >= 3
        and not has_next_check
        and not any(term in joined for term in ("すり替え", "誤読", "求め", "望", "人間の修正"))
    ):
        flags.append("boring_label_dominates")
    return flags


def _score(flags: list[str]) -> int:
    penalties = {
        "too_similar_to_yesterday": 20,
        "user_expectation_shift_missing": 18,
        "misread_pattern_missing": 18,
        "self_critique_missing": 16,
        "hypothesis_update_missing": 16,
        "user_request_missing": 20,
        "shion_next_action_missing": 20,
        "boring_label_dominates": 18,
    }
    return max(0, 100 - sum(penalties.get(flag, 10) for flag in flags))


def _has_critical_flags(flags: list[str]) -> bool:
    critical = {
        "user_expectation_shift_missing",
        "misread_pattern_missing",
        "self_critique_missing",
        "hypothesis_update_missing",
        "user_request_missing",
        "shion_next_action_missing",
    }
    return bool(set(flags) & critical)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    normalized_left = re.sub(r"\s+", "", left)
    normalized_right = re.sub(r"\s+", "", right)
    return round(SequenceMatcher(None, normalized_left, normalized_right).ratio(), 3)


def _source(path: Path | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False}
    try:
        display = str(path.relative_to(REPO_ROOT))
    except ValueError:
        display = str(path)
    return {"path": display, "exists": path.exists()}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        clean = _clean_item(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def render_markdown(payload: dict[str, Any]) -> str:
    delta = payload["delta"]
    handoff = payload["operational_handoff"]
    judgment_change_log = payload.get("judgment_change_log", {})
    narrative_layer = payload.get("narrative_layer", {})
    quality = payload["quality"]
    metrics = payload["metrics"]
    lines = [
        f"# Shion Reflection Delta - {payload['target_date']}",
        "",
        "## 判定",
        f"- status: {quality['status']}",
        f"- score: {quality['score']}",
        f"- 前日内省類似度: {metrics['reflection_similarity_to_yesterday']}",
        f"- flags: {_join(quality['flags'])}",
        "",
        "## 差分",
        "### User要求の変化",
        *_markdown_items(delta["user_expectation_shift"]),
        "",
        "### すり替え・逃げの兆候",
        *_markdown_items(delta["misread_patterns"]),
        "",
        "### 自己批判",
        *_markdown_items(delta["self_critique"]),
        "",
        "### 仮説更新",
        *_markdown_items(delta["hypothesis_updates"]),
        "",
        "## 運用ハンドオフ",
        "### User確認依頼",
        *_markdown_items(handoff["user_requests"]),
        "",
        "### 紫苑の次回変更",
        *_markdown_items(handoff["shion_next_actions"]),
        "",
        "## 判断変更ログ",
        *_markdown_key_values(judgment_change_log),
        "",
        "## 小説化レイヤー",
        f"- status: {narrative_layer.get('status', 'derived_only')}",
        f"- protagonists: {', '.join(narrative_layer.get('protagonists', []))}",
        f"- rule: {narrative_layer.get('rule', '')}",
        "",
        "## 読み方",
        "- これは読み取り専用の検査レポート。チャット、RAG、スコアリングには自動反映しない。",
        "- 合格条件は文章の深さではなく、判断変更ログ、User確認依頼、紫苑の次回変更が具体化されていること。",
        "- 小説化は後段の表現であり、判断変更ログを内省の正本として扱う。",
        "",
    ]
    return "\n".join(lines)


def _markdown_items(items: list[str]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {item}" for item in items]


def _markdown_key_values(items: dict[str, str]) -> list[str]:
    if not items:
        return ["- なし"]
    return [f"- {key}: {value}" for key, value in items.items()]


def _join(items: list[str]) -> str:
    return ", ".join(str(item) for item in items) if items else "なし"


def _parse_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Shion reflection delta inspection report.")
    parser.add_argument("--date", default=None, help="Target date in YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--reflection-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = build_reflection_delta(
        memory_dir=args.memory_dir,
        reflection_dir=args.reflection_dir,
        target_date=_parse_date(args.date),
    )
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
