"""検索でハブノートが当たった時に、そのハブにつながる最近のニュース永続メモを添える（REV-502・既定オフ）。

ニュース永続メモ（scripts/build_news_zettel.py）は、Jev のハブ判定を通ったものだけがハブへリンクする。
審査やチャット（/api/chat・紫苑対話室）で検索結果にハブノートが入った時、そのハブにつながる
最近の永続メモを2件まで「参考」として渡す。

- 審査の判定・スコアの根拠には使わない。関係があれば回答の最後に一文だけ「最近のニュースでは〜という見方もある」と触れる
  （REV-540: 「触れるなら〜程度」では回答の型が優先されて一度も触れられなかったため、置き場所と量を指定した）
  （#1297 の事実と解釈の言い分け: 永続メモは紫苑がニュースを書き直した解釈であって、記録された事実ではない）
- 未接続のメモ・Jev 判定を通っていないメモ（REV-501 より前の hub_fit 無し）は使わない
- SHION_NEWS_ZETTEL_CONTEXT=1 の時だけ有効。既定は空文字（プロンプトは変わらない）
- 質問文を渡すと、同じハブのメモの中から質問との近さで2件まで選び、近いものが無ければ添えない（REV-535）。
  近さは文字の2字組の重なりを、メモ全体でよく出る組ほど軽く数えたもの（AI 呼び出しなし・費用ゼロ）。
  質問文を渡さない呼び出しは従来どおり新しい順（recent_memos は互換のため残す）
- 質問文を渡した時は、検索で当たったハブに限らず全ハブのメモから選ぶ（REV-537）。雑談の誤ヒット
  （「今日の気分は？」→「今日の市場はどう動いた」）を防ぐため、内容語（漢字・カタカナ等の2字組で、
  メモの25%以下にしか出ないもの）が2つ以上一致することも条件にする
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

from runtime_paths import get_data_path

# DATA_DIR に従う（REV-589: 検証環境で本番の data/ を読まず、参考メモも検証用の状態ファイルから組み立てる）。
# 本番は DATA_DIR 未設定なので従来どおりリポジトリの data/
STATE_PATH = Path(get_data_path("news_zettel_state.json"))
MAX_MEMOS = 2
MAX_AGE_DAYS = 45
TITLE_CHARS = 36  # REV-533 プロンプト上限の中で確実に残すため短くする（1件 約150字）
BODY_CHARS = 100
HUB_FIT_MIN = float(os.environ.get("NEWS_ZETTEL_HUB_FIT_MIN", "0.7"))
# REV-535 本番のメモ348件で確認: 雑談は最高0.08、質問と関係の薄いメモは0.10〜0.13、関係するメモは0.16以上
RELEVANCE_MIN = float(os.environ.get("NEWS_ZETTEL_RELEVANCE_MIN", "0.13"))
# REV-537 内容語の一致数。「リース」「審査」「設備投資」のようにメモの25%超に出る組は数えない
MIN_TOPIC_HITS = 2
COMMON_GRAM_RATIO = 0.25
_CONTENT_GRAM_RE = re.compile(r"^[\u4e00-\u9fff\u30a0-\u30ffA-Za-z0-9ー々]{2}$")


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
    recheck: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """ハブにつながった（Jev 判定を通った）最近の永続メモを新しい順に limit 件まで。"""
    cutoff = (today - dt.timedelta(days=MAX_AGE_DAYS)).isoformat()
    from scripts.build_news_zettel import HUBS, hub_fit_min, hub_legacy_fit_min, hub_title_ok, load_recheck

    # REV-534/536/539 説明を絞ったハブは、既存メモも読む時に絞る（メモ・状態ファイルは書き換えない）。
    # 今の説明で判定し直した点数（data/news_zettel_hub_recheck.json）があればそれを使い、
    # 判定に使った説明が今と同じならそのハブのしきい値、古い説明の点数なら legacy のしきい値で絞る
    hub_by_label = {hub["label"]: hub for hub in HUBS}
    rechecked = load_recheck() if recheck is None else recheck
    rows = []
    for key, entry in state.items():
        if not isinstance(entry, dict) or entry.get("status") != "written":
            continue
        linked = [hub for hub in entry.get("hubs") or [] if hub in hubs]
        first = (entry.get("hubs") or [""])[0]  # hub_fit は先頭のハブについての判定
        record = rechecked.get(key) if (rechecked.get(key) or {}).get("hub") == first else None
        fit = (record or entry).get("hub_fit")
        hub = hub_by_label.get(first)
        if hub is None:
            need = HUB_FIT_MIN
        elif (record or entry).get("hub_use") == hub["use"]:
            need = hub_fit_min(hub)
        else:
            need = hub_legacy_fit_min(hub)
        if not linked or fit is None or float(fit) < need:
            continue
        memo = str(entry.get("memo") or "")
        # REV-541 見出しの条件（機械受注統計: 機械受注・工作機械）。メモのファイル名は見出しの先頭40字
        if hub is not None and not hub_title_ok(hub, Path(memo).stem[11:]):
            continue
        date = Path(memo).name[:10]
        if not memo or date < cutoff:
            continue
        rows.append({"memo": memo, "date": date, "hub": linked[0]})
    rows.sort(key=lambda row: row["date"], reverse=True)
    return rows[:limit]


def _memo_parts(vault: Path, rel: str) -> tuple[str, str]:
    """永続メモの見出しと本文（全文・空白は詰める）。読めなければ空。"""
    try:
        path = vault / rel
        if getattr(os.stat(path), "st_flags", 0) & 0x40000000:  # iCloud 退避中は読まない
            return "", ""
        text = path.read_text(encoding="utf-8")
    except OSError:
        return "", ""
    title = re.search(r"^# (.+)$", text, re.MULTILINE)
    body = text.split("\n# ", 1)[-1].split("\n", 1)[-1].split("\n- 元記事", 1)[0].strip()
    return (title.group(1).strip() if title else ""), " ".join(body.split())


def _memo_text(vault: Path, rel: str) -> tuple[str, str]:
    """永続メモの見出しと本文（1〜2文）。読めなければ空。"""
    title, body = _memo_parts(vault, rel)
    return title, body[:BODY_CHARS]


_INDEX_CACHE: dict[tuple[str, str, float], tuple[dict[str, set[str]], dict[str, int]]] = {}


def _memo_index(vault: Path, state_path: Path, state: dict[str, dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """書かれた永続メモ全部の 2字組と、組ごとの出現メモ数。状態ファイルが変わるまで使い回す（1日1回更新）。"""
    from api.chat_prompt_budget import _bigrams

    try:
        mtime = state_path.stat().st_mtime
    except OSError:
        mtime = -1.0
    key = (str(vault), str(state_path), mtime)
    if key not in _INDEX_CACHE:
        grams: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for entry in state.values():
            memo = str(entry.get("memo") or "") if isinstance(entry, dict) and entry.get("status") == "written" else ""
            if not memo or memo in grams:
                continue
            title, body = _memo_parts(vault, memo)
            if not body:
                continue
            grams[memo] = _bigrams(title + body)
            for gram in grams[memo]:
                df[gram] = df.get(gram, 0) + 1
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = (grams, df)
    return _INDEX_CACHE[key]


def relevance(question_grams: set[str], memo_grams: set[str], df: dict[str, int], total: int) -> float:
    """質問の 2字組のうちメモにもある割合。どのメモにもよく出る組（リース・審査など）ほど軽く数える。"""
    weight = {gram: math.log((total + 1) / (df.get(gram, 0) + 1)) for gram in question_grams}
    whole = sum(weight.values())
    return sum(weight[gram] for gram in question_grams & memo_grams) / whole if whole > 0 else 0.0


def topic_hits(question_grams: set[str], memo_grams: set[str], df: dict[str, int], total: int) -> int:
    """質問とメモに共通する内容語の2字組の数（ひらがな混じり・メモの25%超に出る組は数えない）。"""
    limit = COMMON_GRAM_RATIO * max(total, 1)
    return sum(1 for gram in question_grams & memo_grams if _CONTENT_GRAM_RE.match(gram) and df.get(gram, 0) <= limit)


def relevant_memos(
    hubs: list[str],
    question: str,
    *,
    state: dict[str, dict[str, Any]],
    today: dt.date,
    vault: Path,
    state_path: Path = STATE_PATH,
    limit: int = MAX_MEMOS,
    min_score: float = RELEVANCE_MIN,
    min_topic_hits: int = MIN_TOPIC_HITS,
) -> list[dict[str, Any]]:
    """ハブにつながった最近の永続メモのうち、質問に近いものを limit 件まで。

    近さが min_score 未満、または内容語の一致が min_topic_hits 未満のメモは除く。
    """
    from api.chat_prompt_budget import _bigrams

    question_grams = _bigrams(question)
    candidates = recent_memos(hubs, state=state, today=today, limit=len(state))
    if not question_grams or not candidates:
        return []
    grams, df = _memo_index(vault, state_path, state)
    rows = []
    for row in candidates:
        if row["memo"] in grams:
            score = relevance(question_grams, grams[row["memo"]], df, len(grams))
            if score >= min_score and topic_hits(question_grams, grams[row["memo"]], df, len(grams)) >= min_topic_hits:
                rows.append({**row, "score": round(score, 3)})
    rows.sort(key=lambda row: (row["score"], row["date"]), reverse=True)
    return rows[:limit]


def build_news_zettel_context(
    refs: Iterable[str],
    *,
    question: str | None = None,
    vault: Path | None = None,
    state_path: Path = STATE_PATH,
    today: dt.date | None = None,
) -> str:
    """プロンプトに添えるブロック。無効・ハブが当たっていない・メモが無い時は空文字。

    question を渡すと、全ハブのメモから質問に近いものだけを選ぶ（REV-535/537）。渡さなければ従来どおり
    検索で当たったハブのメモを新しい順。
    """
    if not enabled():
        return ""
    try:
        # REV-537 質問文がある時は検索で当たったハブに限らず全ハブから、質問に近いメモを選ぶ
        hubs = hit_hubs(refs) if question is None else list(dict.fromkeys(_hub_labels_by_file().values()))
        if not hubs:
            return ""
        if vault is None:
            from runtime_paths import resolve_obsidian_vault

            vault = resolve_obsidian_vault()
        state = _load_state(state_path)
        day = today or dt.date.today()
        if question is None:
            memos = recent_memos(hubs, state=state, today=day)
        else:
            memos = relevant_memos(hubs, question, state=state, today=day, vault=vault, state_path=state_path)
        if not memos:
            return ""
        lines = []
        for memo in memos:
            title, body = _memo_text(vault, memo["memo"])
            if body:
                lines.append(f"- {memo['date']}（{memo['hub']}）{title[:TITLE_CHARS]}: {body}")
        if not lines:
            return ""
    except Exception as exc:  # noqa: BLE001 - 参考情報なので失敗しても回答は続ける
        from silent_failure_log import record_silent_failure

        record_silent_failure("answer.news_zettel_context", "swallowed", exc, detail="ニュース永続メモを添えずに続行")
        return ""
    # 注意書きは見出し行に入れる。上限で削る時はメモ（古い方）から落ち、注意書きは残る（REV-533）
    return "\n".join(
        [
            "【最近のニュースから（参考。審査の根拠にしない）】紫苑がニュースを書き直した解釈で、記録された事実ではない。"
            "質問と関係があれば、回答の最後に一文だけ「最近のニュースでは〜という見方もある」と添える（事実として断定しない）。"
            "審査の判定・スコア・承認条件の根拠には使わない。質問と関係が薄ければ触れない。",
            *lines,
        ]
    )


def context_from_hits(rag_hits: list[dict[str, Any]] | None, question: str | None = None) -> str:
    """対話室: 検索結果（hit の dict）から。main.py の行数を増やさないためここに置く。"""
    return build_news_zettel_context((str(h.get("file_name") or h.get("ref") or "") for h in rag_hits or []), question=question)


def context_from_refs(rag_refs: list[str] | None, question: str | None = None) -> str:
    """/api/chat（RAG 経路）: 検索結果の ref から。前の節と区切るため先頭に空行を付ける。"""
    block = build_news_zettel_context(rag_refs or [], question=question)
    return f"\n\n{block}" if block else ""
