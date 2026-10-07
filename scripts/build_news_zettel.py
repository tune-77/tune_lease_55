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
        if os.stat(path).st_flags & SF_DATALESS:
            return None
    except (OSError, AttributeError):
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
        "date": meta.get("date") or path.name[:10],
        "title": title,
        "industries": meta.get("industries", ""),
        "lease_assets": meta.get("lease_assets", ""),
        "importance": meta.get("importance", ""),
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
    """未処理のクリップを新しい順に選ぶ。直近分は new_limit 件まで、補完は backfill_limit 件まで。"""
    news_dir = vault / NEWS_DIR
    names = sorted((p.name for p in news_dir.glob("*.md")), reverse=True)[:MAX_SCAN_FILES]
    cutoff = (today - dt.timedelta(days=new_days)).isoformat()
    recent: list[Path] = []
    older: list[Path] = []
    for name in names:
        rel = str(NEWS_DIR / name)
        if rel in state:
            continue
        (recent if name[:10] >= cutoff else older).append(news_dir / name)
    picked = recent[:new_limit]
    if backfill:
        picked += older[:backfill_limit]
    return picked


def build_prompt(items: list[dict[str, Any]], hubs: list[dict[str, str]]) -> str:
    hub_lines = "\n".join(f"- {hub['id']}: {hub['label']}（{hub['use']}）" for hub in hubs)
    article_lines = "\n".join(
        f"[{index}] {item['title']}\n    業種: {item['industries'] or '-'} / 物件: {item['lease_assets'] or '-'}"
        + (f"\n    要約: {item['summary']}" if item["summary"] else "")
        for index, item in enumerate(items)
    )
    return f"""あなたはリース審査担当の相棒「紫苑」です。業界ニュースを、審査の知識として残す「永続メモ」に書き直します。

## 記事
{article_lines}

## つなぎ先の候補（ハブノート）
{hub_lines}

## 書き方（記事ごと）
- idea: その記事がリース審査にとって何を意味するかを、紫苑の言葉で1〜2文（60〜140字）。見出しの言い換えだけにしない。
  何に気をつける・何を確かめる・どの業種や物件の見方が変わる、のどれかを含める。
- 記事に無い数値や事実を足さない。推測は「〜かもしれない」「〜なら確かめたい」と書く。
- リース審査との関係が読み取れない記事は idea を空文字にする。
- hubs: 上の候補から、内容がはっきり関係するものの id を0〜2個。迷うなら入れない（空配列でよい）。

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
        hubs = [h for h in dict.fromkeys(str(h) for h in row.get("hubs") or []) if h in hub_ids][:2]
        rows[index] = {"idea": idea if 20 <= len(idea) <= 220 else "", "hubs": hubs}
    return rows


def _safe_stem(text: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|#\[\]^]", "", text)
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned[:40] or "memo"


def render_memo(item: dict[str, Any], idea: str, hubs: list[dict[str, str]], *, model: str, now: str) -> str:
    source = f"{NEWS_DIR}/{item['path'].stem}"
    title = re.sub(r"\s+-\s+[^-]+$", "", item["title"]).strip()  # 末尾の「 - 配信元」を外す
    related = " ".join(f"[[{hub['path']}|{hub['label']}]]" for hub in hubs) or "未接続"
    hub_labels = json.dumps([hub["label"] for hub in hubs], ensure_ascii=False)
    return "\n".join(
        [
            "---",
            "type: news_zettel",
            f"date: {item['date']}",
            f"connection: {'connected' if hubs else 'unconnected'}",
            f"hubs: {hub_labels}",
            f'source_note: "[[{source}]]"',
            f"generated_by: {FEATURE} ({model})",
            f"generated_at: {now}",
            "tags: [ニュース永続メモ]",
            "---",
            f"# {title[:60]}",
            "",
            idea,
            "",
            f"- 元記事: [[{source}|{title[:40]}]]",
            f"- 関連: {related}",
            "",
        ]
    )


def process(
    vault: Path,
    clips: list[Path],
    state: dict[str, dict[str, Any]],
    *,
    model_call: Callable[[str], dict[str, Any]] = call_model,
    model_name: str = "",
    dry_run: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    now = now or dt.datetime.now().isoformat(timespec="seconds")
    hubs = available_hubs(vault)
    hub_by_id = {hub["id"]: hub for hub in hubs}
    items: list[dict[str, Any]] = []
    unreadable = 0
    for path in clips:
        text = _read_text(path)
        if text is None:
            unreadable += 1
            continue
        items.append(parse_clip(path, text))
    summary = {"selected": len(clips), "unreadable": unreadable, "calls": 0, "written": 0, "connected": 0,
               "unconnected": 0, "no_idea": 0, "exists": 0, "stopped": "", "memos": []}
    memo_dir = vault / MEMO_DIR
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
        for index, item in enumerate(batch):
            rel = str(NEWS_DIR / item["path"].name)
            row = rows.get(index) or {"idea": "", "hubs": []}
            if not row["idea"]:
                summary["no_idea"] += 1
                state[rel] = {"status": "no_idea", "processed_at": now}
                continue
            memo_path = memo_dir / f"{item['date']}_{_safe_stem(item['title'])}.md"
            if memo_path.exists():
                summary["exists"] += 1
                state[rel] = {"status": "exists", "memo": str(memo_path.relative_to(vault)), "processed_at": now}
                continue
            linked = [hub_by_id[h] for h in row["hubs"]]
            memo_dir.mkdir(parents=True, exist_ok=True)
            memo_path.write_text(render_memo(item, row["idea"], linked, model=model_name, now=now), encoding="utf-8")
            summary["written"] += 1
            summary["connected" if linked else "unconnected"] += 1
            summary["memos"].append({"memo": memo_path.name, "idea": row["idea"], "hubs": [h["label"] for h in linked]})
            state[rel] = {"status": "written", "memo": str(memo_path.relative_to(vault)),
                          "hubs": [h["label"] for h in linked], "processed_at": now}
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
    if not args.dry_run:
        save_state(state)
    print(json.dumps({k: v for k, v in summary.items() if k != "memos"}, ensure_ascii=False))
    for memo in summary["memos"]:
        print(f"- {memo['memo']} → {', '.join(memo['hubs']) or '未接続'} / {memo['idea']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
