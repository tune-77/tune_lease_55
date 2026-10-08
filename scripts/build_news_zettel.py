#!/usr/bin/env python3
"""ニュースのクリップから、紫苑の言葉で書いた永続メモを作る（ツェッテルカステン式・REV-499 試行）。

1記事につき1つの永続メモ（1ノート1アイデア）を
``05-クリップ_記事/業界リスクニュース/永続メモ/`` に新規作成する。

- 中身: その記事がリース審査にとって何を意味するかを、紫苑が自分の言葉で1〜2行に書き直したもの
- リンク: 元記事と、関係するハブノート（実在するものだけ）へ永続メモ側から [[ ]] で張る。
  ハブ側は書き換えない（Obsidian のバックリンクで辿れる）。合うハブが無ければ「未接続」
- 置き場所: ニュースの読み手はフォルダ直下の *.md しか見ないので記事として数えられない。
  RAG では「05-クリップ_記事/業界リスクニュース/」の減点がそのまま効く（ニュースは低めのまま）
- 費用: flash-lite（既定モデル）で最大10件を1回の呼び出しにまとめる。feature=news_zettel は
  予算ガードの自発系（当日予算の80%で止まる）
- 対象: 既定は直近 --new-days 日の未処理クリップだけ（ニュースの件数は増やさない）。
  過去分の補完は NEWS_ZETTEL_BACKFILL=1 か --backfill の時だけ、新しい順に1日 --backfill-limit 件まで

既存のクリップ・ノートは読むだけで、消さない・上書きしない（同名の永続メモがあれば作らない）。
iCloud が退避したファイルは読まずに飛ばし、1ファイルの読み込みにも時間の上限を付ける。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_runtime_client import _MAIN_ROOT  # noqa: E402

FEATURE = "news_zettel"
NEWS_DIR = Path("05-クリップ_記事") / "業界リスクニュース"
MEMO_DIR = NEWS_DIR / "永続メモ"
# リースニュース（被リンク0が279件）にも広げる時だけ NEWS_ZETTEL_LEASE_NEWS=1（REV-502。既定オフ）。
# 永続メモは各フォルダの「永続メモ/」に置き、RAG ではそれぞれのニュースの減点がそのまま効く
LEASE_NEWS_DIR = Path("05-クリップ_記事") / "リースニュース"
MEMO_SUBDIR = "永続メモ"


def source_dirs() -> list[Path]:
    dirs = [NEWS_DIR]
    if os.environ.get("NEWS_ZETTEL_LEASE_NEWS", "0").strip() == "1":
        dirs.append(LEASE_NEWS_DIR)
    return dirs
STATE_PATH = _MAIN_ROOT / "data" / "news_zettel_state.json"
SF_DATALESS = 0x40000000
READ_TIMEOUT_SECONDS = 5
MAX_SCAN_FILES = 3000
BATCH_SIZE = 10

# ハブ候補（Vault 内の実在パス、拡張子なし）。実行時に存在するものだけを使う。
HUBS: list[dict[str, str]] = [
    {"id": "h1", "path": "03-知識_業界/業種分析/業種別傾向", "label": "業種別傾向", "use": "業種ごとの景況・需要・リース利用の傾向"},
    {"id": "h2", "path": "03-知識_業界/業種分析/倒産率とリスク", "label": "倒産率とリスク", "use": "倒産・廃業・人手不足倒産など信用悪化の兆候"},
    {"id": "h3", "path": "03-知識_業界/業種分析/業種別デフォルト率ベンチマーク", "label": "業種別デフォルト率ベンチマーク", "use": "業種別の貸倒・延滞水準"},
    {"id": "h4", "path": "03-知識_業界/リース審査実務/審査方針", "label": "審査方針", "use": "審査の基本姿勢・確認の優先順位が変わる話"},
    {"id": "h5", "path": "03-知識_業界/リース審査実務/業種別審査チェックリスト_追補", "label": "業種別審査チェックリスト", "use": "特定業種で追加で確認すべき項目"},
    {"id": "h6", "path": "Projects/tune_lease_55/Q-Risk", "label": "Q-Risk", "use": "スコアに出ない定性リスク（規制・取引先集中・地政学・人手）"},
    {"id": "h7", "path": "lease-wiki-vault/04_リスク分析/リース契約：残価リスク評価", "label": "残価リスク評価", "use": "中古価格・陳腐化・再販価値・残価"},
    {"id": "h8", "path": "lease-wiki-vault/00_Core Definitions/リース契約：金利と料率相場", "label": "金利と料率相場", "use": "金利・調達コスト・料率の動き"},
    {"id": "h9", "path": "lease-wiki-vault/00_Core Definitions/法定耐用年数×最短リース期間マスタ表", "label": "法定耐用年数×最短リース期間", "use": "耐用年数・リース期間の制度"},
    {"id": "h10", "path": "03-知識_業界/補助金・融資/補助金_制度全体像", "label": "補助金の制度全体像", "use": "補助金・助成・公的融資の制度"},
    {"id": "h11", "path": "03-知識_業界/税務・会計知識/新リース会計基準2027", "label": "新リース会計基準2027", "use": "リース会計・税制の変更"},
    {"id": "h12", "path": "Asset Knowledge/INDEX", "label": "物件知識（Asset Knowledge）", "use": "建機・製造設備・車両・医療機器など物件ごとの需要・中古相場"},
    {"id": "h15", "path": "03-知識_業界/市場分析データ/機械受注統計_2023-2026", "label": "機械受注統計", "use": "設備投資・機械受注の統計"},
]


class _ReadTimeout(Exception):
    pass


def _alarm(_signum, _frame):  # pragma: no cover - signal handler
    raise _ReadTimeout()


def available_hubs(vault: Path) -> list[dict[str, str]]:
    return [hub for hub in HUBS if (vault / f"{hub['path']}.md").exists()]


def _read_text(path: Path) -> str | None:
    """iCloud 退避（dataless）は読まない。読めても時間がかかれば諦める。"""
    try:
        # st_flags は macOS だけにある。無い環境（CI の Linux 等）では退避判定をしない
        if getattr(os.stat(path), "st_flags", 0) & SF_DATALESS:
            return None
    except OSError:
        return None
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(READ_TIMEOUT_SECONDS)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (_ReadTimeout, OSError):
        return None
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    meta: dict[str, str] = {}
    if match:
        for line in match.group(1).splitlines():
            key, sep, value = line.partition(":")
            if sep:
                meta[key.strip()] = value.strip().strip('"')
    return meta


# 市場調査会社の宣伝記事の見出しの型（中身が一般論になる。REV-500）
_PROMO_TITLE = re.compile(
    r"20\d\d年(から|〜|~|-)20\d\d年|年平均成長率|CAGR|市場規模|市場動向評価|市場予測|市場調査レポート"
    r"|アフターマーケット市場|市場における業界(分析|戦略)"
)


def is_promo_title(title: str) -> bool:
    return bool(_PROMO_TITLE.search(title))


def normalize_topic(text: str) -> str:
    """同じ出来事かを見る鍵。配信元の違い・記号・空白を落とす。"""
    text = re.sub(r"\s+-\s+[^-]+$", "", str(text or ""))
    text = re.sub(r"[\s\W_]+", "", text)
    return text[:40]


_NUMBER_TOKEN = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|％|ポイント|件|億円|億|万円|万|兆円|兆|円|倍|社|人|位|月|年度|年)?")
_STRONG_UNIT = re.compile(r"(%|％|ポイント|件|億|万|兆|円|倍|社|人)$")


def stat_key(date: str, title: str) -> str:
    """見出しの違う同じ統計記事を見分ける鍵（日付＋見出しの数値の組）。REV-501。

    例: 「8月の工作機械受注 64%増 …歴代2位」と「工作機械受注、8月64%増 …歴代2位」は同じ鍵。
    単位つきの数値（%・件・億 など）が1つも無い見出しは、月や年だけで誤って重ならないよう鍵を作らない。
    """
    body = re.sub(r"\s+-\s+[^-]+$", "", str(title or ""))
    body = re.sub(r"20\d\d", "", body) if re.search(r"\d+(?:%|％|件|億|万|兆|円|倍|社|人)", body) else body
    tokens = sorted({re.sub(r"\s+", "", t) for t in _NUMBER_TOKEN.findall(body) if t.strip()})
    if not any(_STRONG_UNIT.search(t) for t in tokens):
        return ""
    return f"{date}|{'/'.join(tokens)}"


def parse_clip(path: Path, text: str) -> dict[str, Any]:
    meta = _frontmatter(text)
    title_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    title = (title_match.group(1) if title_match else path.stem).strip()
    summary_match = re.search(r"^## 3行要約\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    summary = []
    if summary_match:
        for line in summary_match.group(1).splitlines():
            line = line.strip().lstrip("-").strip()
            # 「見出し + 配信元」をそのまま繰り返す行は要約ではないので捨てる
            if line and "詳細なし" not in line and line not in title and not line.startswith(title[:15]):
                summary.append(line)
    return {
        "path": path,
        # Vault 基準のフォルダ（例: 05-クリップ_記事/業界リスクニュース）。元記事リンクと永続メモの置き場所に使う
        "folder": Path(path.parent.parent.name) / path.parent.name,
        "date": meta.get("date") or path.name[:10],
        "title": title,
        "industries": meta.get("industries", ""),
        "lease_assets": meta.get("lease_assets", ""),
        "importance": meta.get("importance", ""),
        "topic": normalize_topic(meta.get("canonical_topic") or title),
        "stat_key": stat_key(meta.get("date") or path.name[:10], title),
        "summary": " / ".join(summary)[:300],
    }


def load_state(path: Path = STATE_PATH) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_state(state: dict[str, dict[str, Any]], path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def select_clips(
    vault: Path,
    state: dict[str, dict[str, Any]],
    *,
    today: dt.date,
    new_days: int,
    new_limit: int,
    backfill: bool,
    backfill_limit: int,
) -> list[Path]:
    """未処理のクリップを新しい順に選ぶ。直近分は new_limit 件まで、補完は backfill_limit 件まで。

    対象フォルダが複数（リースニュースを有効にした時）でも、ファイル名の日付で新しい順に混ぜて選ぶ。
    """
    entries = [
        (path.name, folder)
        for folder in source_dirs()
        for path in (vault / folder).glob("*.md")
    ]
    entries = sorted(entries, key=lambda item: item[0], reverse=True)[:MAX_SCAN_FILES]
    cutoff = (today - dt.timedelta(days=new_days)).isoformat()
    recent: list[Path] = []
    older: list[Path] = []
    for name, folder in entries:
        rel = str(folder / name)
        if rel in state:
            continue
        (recent if name[:10] >= cutoff else older).append(vault / folder / name)
    picked = recent[:new_limit]
    if backfill:
        picked += older[:backfill_limit]
    return picked


def build_prompt(items: list[dict[str, Any]], hubs: list[dict[str, str]]) -> str:
    hub_lines = "\n".join(f"- {hub['id']}: {hub['label']}（{hub['use']}）" for hub in hubs)
    # 3行要約・活用メモはクリップ作成時の定型文（「工期・更新投資…を確認する」等）が多く、
    # 渡すとそのまま書き写されるので見出し・業種・物件だけを渡す（REV-499 試行の結果）
    # body は経産省フィードの要約（記事ページは取得しない）だけ。REV-503
    article_lines = "\n".join(
        f"[{index}] {item['title']}\n    業種: {item['industries'] or '-'} / 物件: {item['lease_assets'] or '-'}"
        + (f"\n    本文の要点（{item['source_label']}の発表より）: {item['body'][:300]}" if item.get("body") else "")
        for index, item in enumerate(items)
    )
    return f"""あなたはリース審査担当の相棒「紫苑」です。業界ニュースを、審査の知識として残す「永続メモ」に書き直します。

## 記事
{article_lines}

## つなぎ先の候補（ハブノート）
{hub_lines}

## 書き方（記事ごと）
- idea: その記事がリース審査にとって何を意味するかを、紫苑のふだんの口調（です・ますを使わない短い砕けた言い方）で1〜2文（50〜120字）。
  見出しの言い換えだけにしない。何に気をつける・何を確かめる・どの業種や物件の見方が変わる、のどれかを含める。
- 見出しに書かれた事実以外は事実として書かない。解釈・推測・因果は必ず「〜かも」「〜なら確かめたい」の形にする。
  「直結する」「急増中」「証左だ」「サインだ」「〜すべき」「〜が必要」のような言い切りは使わない。
- 海外の話・行政手続き・広告など、日本のリース審査との関係が読み取れない記事は idea を空文字にする。
- hubs: 上の候補から、その記事の中心の話題がそのハブの主題そのものである時だけ、id を1個。
  周辺的・一般論的なつながりなら入れない（空配列でよい。空なら「未接続」として残す）。

JSON だけを返す: {{"items": [{{"i": 0, "idea": "...", "hubs": ["h1"]}}]}}"""


def call_model(prompt: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    from ai_runtime_client import google_genai_client
    from config import get_gemini_model
    from novelist_agent import _get_daily_gemini_api_key

    client = google_genai_client(feature=FEATURE, client_factory=genai.Client, api_key=_get_daily_gemini_api_key())
    response = client.models.generate_content(
        model=get_gemini_model(),
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2000, response_mime_type="application/json"),
    )
    return json.loads(response.text or "{}")


def validate(result: dict[str, Any], count: int, hub_ids: set[str]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in result.get("items") or []:
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < count or index in rows:
            continue
        idea = " ".join(str(row.get("idea") or "").split())
        hubs = [h for h in dict.fromkeys(str(h) for h in row.get("hubs") or []) if h in hub_ids][:1]
        rows[index] = {"idea": idea if 20 <= len(idea) <= 220 else "", "hubs": hubs}
    return rows


def _safe_stem(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|#\[\]^]", "", text)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned[:40] or "memo"


def render_memo(
    item: dict[str, Any],
    idea: str,
    hubs: list[dict[str, str]],
    *,
    model: str,
    now: str,
    hub_check: float | None = None,
    proposed: list[str] | None = None,
) -> str:
    title = re.sub(r"\s+-\s+[^-]+$", "", item["title"]).strip()  # 末尾の「 - 配信元」を外す
    if item.get("source_url"):
        # 経産省フィード由来（クリップは無い）。出典を明記し、元の発表ページへリンクする。要約は保存しない
        source_line = f'source_url: "{item["source_url"]}"'
        origin = f"[{title[:40]}]({item['source_url']})（出典: {item['source_label']}）"
    else:
        source = f"{item['folder']}/{item['path'].stem}"
        source_line = f'source_note: "[[{source}]]"'
        origin = f"[[{source}|{title[:40]}]]"
    related = " ".join(f"[[{hub['path']}|{hub['label']}]]" for hub in hubs) or "未接続"
    hub_labels = json.dumps([hub["label"] for hub in hubs], ensure_ascii=False)
    return "\n".join(
        [
            "---",
            "type: news_zettel",
            f"date: {item['date']}",
            f"connection: {'connected' if hubs else 'unconnected'}",
            f"hubs: {hub_labels}",
            *(
                [f"hub_check: {'unchecked' if hub_check is None else round(hub_check, 2)}",
                 f"proposed_hubs: {json.dumps(proposed, ensure_ascii=False)}"]
                if proposed
                else []
            ),
            source_line,
            *([f"source_label: {item['source_label']}"] if item.get("source_label") else []),
            f"generated_by: {FEATURE} ({model})",
            f"generated_at: {now}",
            "tags: [ニュース永続メモ]",
            "---",
            f"# {title[:60]}",
            "",
            idea,
            "",
            f"- 元記事: {origin}",
            f"- 関連: {related}",
            "",
        ]
    )


HUB_FIT_MIN = float(os.environ.get("NEWS_ZETTEL_HUB_FIT_MIN", "0.7"))
HUB_FIT_QUESTION = {
    "type": "noul",
    "instructions": "`items[{n}]` は、業界ニュースの見出し、それをリース審査向けに書き直したメモ、リンク先ハブノートの主題です。"
    "メモの中心の話題は、そのハブノートの主題そのものに当てはまりますか。",
    "criteria": {
        "true": "メモの中心の話題が、ハブの主題そのもの（例: 倒産件数の記事→倒産率とリスク、中古建機の再販価値→残価リスク評価）。",
        "false": "関係は言えなくもないが周辺的・一般論的なつながり、またはほぼ無関係（例: 資材価格の記事→定性リスク、保証制度の記事→補助金制度）。",
    },
}


def judge_hubs_with_jev(texts: list[str]) -> list[float | None]:
    """メモ×ハブの組ごとに「当てはまる」確率を返す。伏せ字で送れない・不通なら None（REV-501）。"""
    import typesafe_dedup_guard as transport
    from api.chat_judgment_asset_capture import mask_for_jev

    os.environ.setdefault("TYPESAFE_API_KEYCHAIN_SERVICE", "typesafe-api-key")
    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    masked = [mask_for_jev(text) for text in texts]
    sendable = [k for k, m in enumerate(masked) if m]
    results: list[float | None] = [None] * len(texts)
    if not sendable:
        return results
    questions = {
        f"item{n}": {**HUB_FIT_QUESTION, "instructions": HUB_FIT_QUESTION["instructions"].format(n=n)}
        for n in range(len(sendable))
    }
    try:
        body = transport._default_request(
            {"state": {"items": [masked[k] for k in sendable]}, "model": "jev-latest", "questions": questions}
        )
        for n, k in enumerate(sendable):
            results[k] = float(transport._noul(body["answers"], f"item{n}"))
    except Exception as exc:  # noqa: BLE001 - Jev が使えない時はリンクを残し hub_check: unchecked と記録する
        print(f"[news_zettel] ハブ判定スキップ: {type(exc).__name__}")
    return results


def check_hubs(
    batch: list[dict[str, Any]],
    rows: dict[int, dict[str, Any]],
    hub_by_id: dict[str, dict[str, str]],
    checker: Callable[[list[str]], list[float | None]] | None,
) -> dict[int, float]:
    """メモを書いた記事のうちハブ付きのものを、1回の Jev 呼び出しでまとめて判定する。"""
    targets = [(i, rows[i]) for i in rows if rows[i]["idea"] and rows[i]["hubs"]]
    if not targets or os.environ.get("NEWS_ZETTEL_HUB_CHECK", "1").strip() == "0":
        return {}
    texts = []
    for index, row in targets:
        hub = hub_by_id[row["hubs"][0]]
        texts.append(f"ハブ: {hub['label']}（{hub['use']}）\n見出し: {batch[index]['title'][:80]}\nメモ: {row['idea'][:160]}")
    scores = (checker or judge_hubs_with_jev)(texts)
    return {index: score for (index, _row), score in zip(targets, scores) if score is not None}


def processed_keys(vault: Path, state: dict[str, dict[str, Any]], field: str) -> set[str]:
    """処理済みクリップの鍵（topic / stat_key）。古い記録に無ければクリップから読んで補う（読めなければ飛ばす）。"""
    keys: set[str] = set()
    for rel, entry in state.items():
        if not isinstance(entry, dict):
            continue
        key = entry.get(field)
        if key is None:
            text = _read_text(vault / rel)
            if text is None:
                continue
            parsed = parse_clip(vault / rel, text)
            entry.setdefault("topic", parsed["topic"])
            entry.setdefault("stat_key", parsed["stat_key"])
            key = entry[field]
        if key:
            keys.add(key)
    return keys


def processed_topics(vault: Path, state: dict[str, dict[str, Any]]) -> set[str]:
    return processed_keys(vault, state, "topic")


STAT_TITLE_OVERLAP_MIN = 0.3


def _title_overlap(left: str, right: str) -> float:
    a = {left[i : i + 2] for i in range(len(left) - 1)}
    b = {right[i : i + 2] for i in range(len(right) - 1)}
    return len(a & b) / len(a | b) if a and b else 0.0


def processed_stats(vault: Path, state: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """処理済みの統計の鍵 → その見出し（topic）の一覧。"""
    processed_keys(vault, state, "stat_key")  # 古い記録の stat_key を補う
    stats: dict[str, list[str]] = {}
    for entry in state.values():
        if isinstance(entry, dict) and entry.get("stat_key"):
            stats.setdefault(entry["stat_key"], []).append(str(entry.get("topic") or ""))
    return stats


_PREFECTURE = re.compile(
    r"北海道|青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|埼玉|千葉|東京|神奈川|新潟|富山|石川|福井|山梨|長野"
    r"|岐阜|静岡|愛知|三重|滋賀|京都|大阪|兵庫|奈良|和歌山|鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知"
    r"|福岡|佐賀|長崎|熊本|大分|宮崎|鹿児島|沖縄"
)


def _same_place(left: str, right: str) -> bool:
    """両方に都道府県名があり、それが違えば別の出来事（例: 青森の倒産3件と岡山の倒産3件）。"""
    a, b = set(_PREFECTURE.findall(left)), set(_PREFECTURE.findall(right))
    return not (a and b) or bool(a & b)


def is_same_stat(item: dict[str, Any], stats: dict[str, list[str]]) -> bool:
    """日付と数値の組が同じで、見出しも重なり、場所も食い違わない。"""
    return any(
        _same_place(item["topic"], other) and _title_overlap(item["topic"], other) >= STAT_TITLE_OVERLAP_MIN
        for other in stats.get(item["stat_key"], [])
    )


def process(
    vault: Path,
    clips: list[Path],
    state: dict[str, dict[str, Any]],
    *,
    model_call: Callable[[str], dict[str, Any]] = call_model,
    hub_checker: Callable[[list[str]], list[float | None]] | None = None,
    model_name: str = "",
    dry_run: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    hubs = available_hubs(vault)
    hub_by_id = {hub["id"]: hub for hub in hubs}
    summary = {"selected": len(clips), "unreadable": 0, "calls": 0, "written": 0, "connected": 0,
               "unconnected": 0, "no_idea": 0, "exists": 0, "promo": 0, "duplicate_topic": 0,
               "duplicate_stat": 0, "hub_checked": 0, "hub_dropped": 0, "hub_unchecked": 0,
               "stopped": "", "memos": []}
    seen_topics = processed_topics(vault, state)
    seen_stats = processed_stats(vault, state)
    items: list[dict[str, Any]] = []
    for path in clips:
        text = _read_text(path)
        if text is None:
            summary["unreadable"] += 1
            continue
        item = parse_clip(path, text)
        rel = str(item["folder"] / path.name)
        # 呼び出しの前にルールで落とす（費用もかからない）。REV-500
        if is_promo_title(item["title"]):
            summary["promo"] += 1
            if not dry_run:
                state[rel] = {"status": "skipped_promo", "topic": item["topic"], "processed_at": now}
            continue
        if item["topic"] and item["topic"] in seen_topics:
            summary["duplicate_topic"] += 1
            if not dry_run:
                state[rel] = {"status": "duplicate_topic", "topic": item["topic"], "processed_at": now}
            continue
        if item["stat_key"] and is_same_stat(item, seen_stats):
            summary["duplicate_stat"] += 1
            if not dry_run:
                state[rel] = {"status": "duplicate_stat", "topic": item["topic"], "stat_key": item["stat_key"],
                              "processed_at": now}
            continue
        seen_topics.add(item["topic"])
        if item["stat_key"]:
            seen_stats.setdefault(item["stat_key"], []).append(item["topic"])
        items.append(item)
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        if dry_run:
            print(build_prompt(batch, hubs))
            break
        try:
            result = model_call(build_prompt(batch, hubs))
        except Exception as exc:  # noqa: BLE001 - 予算ガード・通信失敗はその回で止め、未処理は次回へ
            summary["stopped"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            break
        summary["calls"] += 1
        rows = validate(result, len(batch), set(hub_by_id))
        checks = check_hubs(batch, rows, hub_by_id, hub_checker)
        for index, item in enumerate(batch):
            rel = str(item["folder"] / item["path"].name)
            memo_dir = vault / item["folder"] / MEMO_SUBDIR
            row = rows.get(index) or {"idea": "", "hubs": []}
            if not row["idea"]:
                summary["no_idea"] += 1
                state[rel] = {"status": "no_idea", "topic": item["topic"], "stat_key": item["stat_key"],
                              "processed_at": now}
                continue
            memo_path = memo_dir / f"{item['date']}_{_safe_stem(item['title'])}.md"
            if memo_path.exists():
                summary["exists"] += 1
                state[rel] = {"status": "exists", "memo": str(memo_path.relative_to(vault)), "processed_at": now}
                continue
            linked = [hub_by_id[h] for h in row["hubs"]]
            check = checks.get(index)
            if linked and check is None:
                summary["hub_unchecked"] += 1
            elif linked:
                summary["hub_checked"] += 1
                if check < HUB_FIT_MIN:
                    summary["hub_dropped"] += 1
                    linked = []  # 弱いつながりはリンクせず未接続にする（REV-501）
            memo_dir.mkdir(parents=True, exist_ok=True)
            memo_path.write_text(
                render_memo(item, row["idea"], linked, model=model_name, now=now, hub_check=check,
                            proposed=[hub_by_id[h]["label"] for h in row["hubs"]]),
                encoding="utf-8",
            )
            summary["written"] += 1
            summary["connected" if linked else "unconnected"] += 1
            summary["memos"].append({"memo": memo_path.name, "idea": row["idea"], "hubs": [h["label"] for h in linked]})
            state[rel] = {"status": "written", "memo": str(memo_path.relative_to(vault)), "topic": item["topic"],
                          "stat_key": item["stat_key"], "hubs": [h["label"] for h in linked],
                          "proposed_hubs": [hub_by_id[h]["label"] for h in row["hubs"]],
                          "hub_fit": None if check is None else round(check, 3), "processed_at": now}
    return summary


# ── 経済産業省の報道発表フィード（REV-503） ─────────────────────────────────
# 記事ページと robots.txt は 403（ボット遮断）なので取りに行かない。フィードに入っている冒頭の要約だけを
# 「本文の要点」として使う（PDL1.0・出典明記）。要約は Gemini への入力にだけ使い、Vault には保存しない。
METI_FEED_URL = "https://www.meti.go.jp/ml_index_release_atom.xml"
METI_LABEL = "経済産業省"
METI_STATUS_PATH = _MAIN_ROOT / "data" / "news_zettel_meti_feed.json"
METI_MAX_AGE_DAYS = int(os.environ.get("NEWS_ZETTEL_METI_MAX_AGE_DAYS", "14"))
METI_MAX_ITEMS = 10
METI_STALE_DAYS = 30
FEED_USER_AGENT = "tunelease-news-zettel/1.0"


def fetch_meti_feed(timeout: float = 15.0) -> str:
    import urllib.request

    request = urllib.request.Request(METI_FEED_URL, headers={"User-Agent": FEED_USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(2_000_000).decode("utf-8", errors="replace")


def parse_meti_feed(xml_text: str) -> list[dict[str, str]]:
    import html as _html

    entries = []
    for block in re.findall(r"<entry>(.*?)</entry>", xml_text, re.DOTALL):
        title = re.search(r"<title[^>]*>(.*?)</title>", block, re.DOTALL)
        link = re.search(r'<link[^>]*rel="alternate"[^>]*href="([^"]+)"', block) or re.search(r'<link[^>]*href="([^"]+)"', block)
        updated = re.search(r"<(?:updated|published)>([^<]+)</", block)
        summary = re.search(r"<(summary|content)[^>]*>(.*?)</\1>", block, re.DOTALL)
        if not (title and link and updated):
            continue
        entries.append({
            "title": " ".join(_html.unescape(re.sub(r"<[^>]+>", "", title.group(1))).split()),
            "url": link.group(1).strip(),
            "date": updated.group(1).strip()[:10],
            "summary": " ".join(_html.unescape(re.sub(r"<[^>]+>", "", summary.group(2))).split()) if summary else "",
        })
    return entries


def meti_items(entries: list[dict[str, str]], state: dict[str, dict[str, Any]], *, today: dt.date) -> list[dict[str, Any]]:
    """未処理で直近 METI_MAX_AGE_DAYS 日以内の発表だけ。フィードが止まっている間は空。"""
    cutoff = (today - dt.timedelta(days=METI_MAX_AGE_DAYS)).isoformat()
    items = []
    for entry in sorted(entries, key=lambda e: e["date"], reverse=True):
        key = f"meti:{entry['url']}"
        if key in state or entry["date"] < cutoff or not entry["summary"]:
            continue
        items.append({
            "key": key,
            "title": entry["title"],
            "date": entry["date"],
            "industries": "",
            "lease_assets": "",
            "summary": "",
            "body": entry["summary"][:300],
            "source_url": entry["url"],
            "source_label": METI_LABEL,
            "topic": normalize_topic(entry["title"]),
            "stat_key": "",
        })
    return items[:METI_MAX_ITEMS]


def write_meti_status(entries: list[dict[str, str]], *, today: dt.date, error: str = "", written: int = 0,
                      path: Path | None = None) -> dict[str, Any]:
    path = path or METI_STATUS_PATH
    latest = max((e["date"] for e in entries), default="")
    days = (today - dt.date.fromisoformat(latest)).days if latest else None
    status = {"checked_at": dt.datetime.now().isoformat(timespec="seconds"), "feed": METI_FEED_URL,
              "entries": len(entries), "latest_entry_date": latest, "days_since_latest": days,
              "stale": days is None or days >= METI_STALE_DAYS, "memos_written": written, "error": error}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return status


def morning_report_lines(path: Path | None = None) -> list[str]:
    """朝報用: 経産省フィードの最新日付。長く止まっていれば分かるようにする。"""
    path = path or METI_STATUS_PATH
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if status.get("error"):
        return [f"- 📰 経産省フィード（永続メモ用）: 取得失敗 `{status['error'][:60]}`"]
    mark = "⚠️ " if status.get("stale") else ""
    return [
        f"- 📰 {mark}経産省フィード（永続メモ用）: 最新 {status.get('latest_entry_date') or '不明'}"
        f"（{status.get('days_since_latest')}日前）・前回のメモ {status.get('memos_written', 0)}件"
    ]


def process_meti(vault: Path, state: dict[str, dict[str, Any]], *, model_call=None, hub_checker=None,
                 model_name: str = "", today: dt.date | None = None, fetch=None, now: str | None = None) -> dict[str, Any]:
    today = today or dt.date.today()
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    summary = {"meti_entries": 0, "meti_new": 0, "meti_written": 0, "meti_no_idea": 0, "meti_stopped": ""}
    try:
        entries = parse_meti_feed((fetch or fetch_meti_feed)())
    except Exception as exc:  # noqa: BLE001 - フィードが取れなくても業界ニュースの処理は終わっている
        write_meti_status([], today=today, error=f"{type(exc).__name__}: {str(exc)[:80]}")
        summary["meti_stopped"] = type(exc).__name__
        return summary
    summary["meti_entries"] = len(entries)
    seen = processed_topics(vault, state)
    items = [item for item in meti_items(entries, state, today=today) if item["topic"] not in seen]
    summary["meti_new"] = len(items)
    hubs = available_hubs(vault)
    hub_by_id = {hub["id"]: hub for hub in hubs}
    memo_dir = vault / MEMO_DIR
    if items:
        try:
            result = (model_call or call_model)(build_prompt(items, hubs))
        except Exception as exc:  # noqa: BLE001 - 予算ガード等。未処理は次回へ
            summary["meti_stopped"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            write_meti_status(entries, today=today, written=0)
            return summary
        rows = validate(result, len(items), set(hub_by_id))
        checks = check_hubs(items, rows, hub_by_id, hub_checker)
        for index, item in enumerate(items):
            row = rows.get(index) or {"idea": "", "hubs": []}
            if not row["idea"]:
                summary["meti_no_idea"] += 1
                state[item["key"]] = {"status": "no_idea", "topic": item["topic"], "processed_at": now}
                continue
            memo_path = memo_dir / f"{item['date']}_{_safe_stem(item['title'])}.md"
            if memo_path.exists():
                state[item["key"]] = {"status": "exists", "topic": item["topic"], "processed_at": now}
                continue
            linked = [hub_by_id[h] for h in row["hubs"]]
            check = checks.get(index)
            if linked and check is not None and check < HUB_FIT_MIN:
                linked = []
            memo_dir.mkdir(parents=True, exist_ok=True)
            memo_path.write_text(
                render_memo(item, row["idea"], linked, model=model_name, now=now, hub_check=check,
                            proposed=[hub_by_id[h]["label"] for h in row["hubs"]]),
                encoding="utf-8",
            )
            summary["meti_written"] += 1
            state[item["key"]] = {"status": "written", "memo": str(memo_path.relative_to(vault)), "topic": item["topic"],
                                  "stat_key": "", "hubs": [h["label"] for h in linked],
                                  "proposed_hubs": [hub_by_id[h]["label"] for h in row["hubs"]],
                                  "hub_fit": None if check is None else round(check, 3),
                                  "source_label": METI_LABEL, "processed_at": now}
    write_meti_status(entries, today=today, written=summary["meti_written"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--new-days", type=int, default=2, help="この日数以内のクリップを「新規」とみなす")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("NEWS_ZETTEL_NEW_LIMIT", "20")))
    parser.add_argument("--backfill", action="store_true", help="過去の未処理クリップも新しい順に処理する")
    parser.add_argument("--backfill-limit", type=int, default=int(os.environ.get("NEWS_ZETTEL_BACKFILL_DAILY_LIMIT", "20")))
    parser.add_argument("--dry-run", action="store_true", help="プロンプトを表示するだけ（呼び出し・書き込みなし）")
    args = parser.parse_args()

    if os.environ.get("NEWS_ZETTEL_ENABLED", "1").strip() == "0":
        print("[news_zettel] NEWS_ZETTEL_ENABLED=0 のためスキップ")
        return 0
    if args.vault is None:
        from runtime_paths import resolve_obsidian_vault

        args.vault = resolve_obsidian_vault()
    backfill = args.backfill or os.environ.get("NEWS_ZETTEL_BACKFILL", "0").strip() == "1"
    state = load_state()
    clips = select_clips(
        args.vault, state, today=dt.date.today(), new_days=args.new_days, new_limit=args.limit,
        backfill=backfill, backfill_limit=args.backfill_limit,
    )
    from config import get_gemini_model

    summary = process(args.vault, clips, state, model_name=get_gemini_model(), dry_run=args.dry_run)
    if not args.dry_run and os.environ.get("NEWS_ZETTEL_METI_FEED", "1").strip() != "0":
        summary.update(process_meti(args.vault, state, model_name=get_gemini_model()))
    if not args.dry_run:
        save_state(state)
    print(json.dumps({k: v for k, v in summary.items() if k != "memos"}, ensure_ascii=False))
    for memo in summary["memos"]:
        print(f"- {memo['memo']} → {', '.join(memo['hubs']) or '未接続'} / {memo['idea']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
