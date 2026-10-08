"""検索でハブノートが当たった時に、そのハブにつながる最近のニュース永続メモを添える（REV-502・既定オフ）。

ニュース永続メモ（scripts/build_news_zettel.py）は、Jev のハブ判定を通ったものだけがハブへリンクする。
審査やチャット（/api/chat・紫苑対話室）で検索結果にハブノートが入った時、そのハブにつながる
最近の永続メモを2件まで「参考」として渡す。

- 審査の判定・スコアの根拠には使わない。「最近のニュースでは〜という見方もある」程度に触れる
  （#1297 の事実と解釈の言い分け: 永続メモは紫苑がニュースを書き直した解釈であって、記録された事実ではない）
- 未接続のメモ・Jev 判定を通っていないメモ（REV-501 より前の hub_fit 無し）は使わない
- SHION_NEWS_ZETTEL_CONTEXT=1 の時だけ有効。既定は空文字（プロンプトは変わらない）
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "news_zettel_state.json"
MAX_MEMOS = 2
MAX_AGE_DAYS = 45
HUB_FIT_MIN = float(os.environ.get("NEWS_ZETTEL_HUB_FIT_MIN", "0.7"))


def enabled() -> bool:
    return os.environ.get("SHION_NEWS_ZETTEL_CONTEXT", "0").strip() == "1"


def _hub_labels_by_file() -> dict[str, str]:
    """ハブノートのファイル名（拡張子なし）→ ハブ名。build_news_zettel の HUBS と同じ一覧を使う。"""
    from scripts.build_news_zettel import HUBS

    return {Path(hub["path"]).name: hub["label"] for hub in HUBS}


def _file_stem(ref: str) -> str:
    """検索結果の ref / file_name（例: "[[業種別傾向#節]]"・"業種別傾向.md"）からファイル名を取り出す。"""
    text = str(ref or "").strip()
    text = re.sub(r"^\[\[|\]\]$", "", text)
    text = text.split("#", 1)[0].split("|", 1)[0]
    return Path(text).stem if text.endswith(".md") else Path(text).name


def hit_hubs(refs: Iterable[str]) -> list[str]:
    """検索結果に入っていたハブ名（重複なし・検索順）。"""
    labels = _hub_labels_by_file()
    found: list[str] = []
    for ref in refs:
        label = labels.get(_file_stem(ref))
        if label and label not in found:
            found.append(label)
    return found


def _load_state(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def recent_memos(
    hubs: list[str],
    *,
    state: dict[str, dict[str, Any]],
    today: dt.date,
    limit: int = MAX_MEMOS,
) -> list[dict[str, Any]]:
    """ハブにつながった（Jev 判定を通った）最近の永続メモを新しい順に limit 件まで。"""
    cutoff = (today - dt.timedelta(days=MAX_AGE_DAYS)).isoformat()
    rows = []
    for entry in state.values():
        if not isinstance(entry, dict) or entry.get("status") != "written":
            continue
        fit = entry.get("hub_fit")
        linked = [hub for hub in entry.get("hubs") or [] if hub in hubs]
        if not linked or fit is None or float(fit) < HUB_FIT_MIN:
            continue
        memo = str(entry.get("memo") or "")
        date = Path(memo).name[:10]
        if not memo or date < cutoff:
            continue
        rows.append({"memo": memo, "date": date, "hub": linked[0]})
    rows.sort(key=lambda row: row["date"], reverse=True)
    return rows[:limit]


def _memo_text(vault: Path, rel: str) -> tuple[str, str]:
    """永続メモの見出しと本文（1〜2文）。読めなければ空。"""
    try:
        path = vault / rel
        if getattr(os.stat(path), "st_flags", 0) & 0x40000000:  # iCloud 退避中は読まない
            return "", ""
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", ""
    title = re.search(r"^# (.+)$", text, re.MULTILINE)
    body = text.split("\n# ", 1)[-1].split("\n", 1)[-1].split("\n- 元記事", 1)[0].strip()
    return (title.group(1).strip() if title else ""), " ".join(body.split())[:180]


def build_news_zettel_context(
    refs: Iterable[str],
    *,
    vault: Path | None = None,
    state_path: Path = STATE_PATH,
    today: dt.date | None = None,
) -> str:
    """プロンプトに添えるブロック。無効・ハブが当たっていない・メモが無い時は空文字。"""
    if not enabled():
        return ""
    try:
        hubs = hit_hubs(refs)
        if not hubs:
            return ""
        memos = recent_memos(hubs, state=_load_state(state_path), today=today or dt.date.today())
        if not memos:
            return ""
        if vault is None:
            from runtime_paths import resolve_obsidian_vault

            vault = resolve_obsidian_vault()
        lines = []
        for memo in memos:
            title, body = _memo_text(vault, memo["memo"])
            if body:
                lines.append(f"- {memo['date']}（{memo['hub']}）{title[:50]}: {body}")
        if not lines:
            return ""
    except Exception as exc:  # noqa: BLE001 - 参考情報なので失敗しても回答は続ける
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.news_zettel_context", "swallowed", exc, detail="ニュース永続メモを添えずに続行")
        return ""
    return "\n".join(
        [
            "【最近のニュースから（参考。審査の根拠にしない）】",
            *lines,
            "これは紫苑が最近のニュースを書き直したメモで、記録された事実ではなく解釈。"
            "触れる時は「最近のニュースでは〜という見方もある」のように参考として一言添える程度にし、"
            "審査の判定・スコア・承認条件の根拠には使わない。質問と関係が薄ければ触れなくてよい。",
        ]
    )


def context_from_hits(rag_hits: list[dict[str, Any]] | None) -> str:
    """対話室: 検索結果（hit の dict）から。main.py の行数を増やさないためここに置く。"""
    return build_news_zettel_context(str(h.get("file_name") or h.get("ref") or "") for h in rag_hits or [])


def context_from_refs(rag_refs: list[str] | None) -> str:
    """/api/chat（RAG 経路）: 検索結果の ref から。前の節と区切るため先頭に空行を付ける。"""
    block = build_news_zettel_context(rag_refs or [])
    return f"\n\n{block}" if block else ""
