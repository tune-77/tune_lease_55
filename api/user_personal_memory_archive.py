"""紫苑のユーザー個人記憶を「アーカイブ」してプロンプト候補から外す。

元のファイル（data/user_personal_memory.md・MEMORY.md・USER.md・PERSISTENT_MEMORY.md）は書き換えない。
外す行を data/user_personal_memory_archive.json に原文ごと記録し、読み込み側
（api/chat_user_personal_memory.py）がその行を候補から除く。復元は記録から外すだけ（--restore）。

週次の整理（scripts/judgment_asset_dedup.py の週次ジョブから呼ぶ）の基準は 2026-10-03 の人手整理と同じ:
- 自動アーカイブ: 空のテンプレ行／記憶ではなく質問だった取り込み行／同文・ほぼ同文の重複（新しい方を残す）／
  判断資産（data/canonical_judgment_rules.json の active）と同じ内容／90日より古い日付付きの行
- アーカイブ候補（人が判断）: 関係性の語を含む古い行、日付の無い技術メモ
- 対象外: pinned（人手で残すと決めた行・復元した行）、呼び方・犬の名前などの個人事実、
  関係性の語（呼び方・好み・Mana・妹・Relationship UX など）を含む行、30日以内の行

使い方:
  .venv/bin/python -m api.user_personal_memory_archive --dry-run
  .venv/bin/python -m api.user_personal_memory_archive --list
  .venv/bin/python -m api.user_personal_memory_archive --restore <id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = Path(get_data_path("user_personal_memory_archive.json"))
CANONICAL_RULES_PATH = Path(get_data_path("canonical_judgment_rules.json"))

STALE_DAYS = 90
RECENT_DAYS = 30
NEAR_DUPLICATE = 0.8
COVERED_BY_ASSET = 0.6

_PROTECT_RE = re.compile(
    r"Dog name|Preferred name|What to call them:\*\*\s*\S|呼び方|呼んで|好み|好き|嫌い|苦手|大事|大切|"
    r"Mana|妹|家族|Relationship UX|関係性|Core Motivation|\[sensitive"
)
_CAPTURED_RE = re.compile(r"^-?\s*(\d{4}-\d{2}-\d{2})T\S*\s+\[[a-z]+/[a-z_]+\]\s+\([^)]*\)\s+(.*)$")
_INLINE_DATE_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2})\]")
_SECTION_DATE_RE = re.compile(r"^##\s.*?(\d{4}-\d{2}-\d{2})")
_REMEMBER_RE = re.compile(r"覚えておいて|覚えて|忘れないで|記憶して|メモして|保存して")
_QUESTION_END_RE = re.compile(r"(？|\?|だっけ|かな|は)$")
_TEMPLATE_RE = re.compile(r"^-?\s*\*\*[^*]+\*\*:?\s*$")
_TECH_RE = re.compile(r"`|/api/|\.py|\.md|\.json|PR #|scripts/|frontend/")


def line_key(line: str) -> str:
    """行の同一性キー（行頭の箇条記号と空白の違いを無視）。"""
    text = unicodedata.normalize("NFKC", str(line or "")).strip()
    text = re.sub(r"^[-*]\s+", "", text)
    return re.sub(r"\s+", " ", text)


def item_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"[\s、。,.()（）「」『』:：/・*#\-`\[\]]", "", unicodedata.normalize("NFKC", text))
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _jaccard(a: str, b: str) -> float:
    ga, gb = _bigrams(a), _bigrams(b)
    return len(ga & gb) / len(ga | gb) if ga and gb else 0.0


def load_archive(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads((path or ARCHIVE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    for name in ("items", "pinned", "review_candidates"):
        data.setdefault(name, [])
    return data


def archived_keys(path: Path | None = None) -> set[str]:
    """プロンプト候補から外す行のキー（読み込み側が使う。壊れていたら何も外さない）。"""
    return {str(item.get("key")) for item in load_archive(path)["items"] if item.get("key")}


def save_archive(data: dict[str, Any], path: Path | None = None, *, backup: bool = True) -> Path | None:
    """書き込む前に既存のアーカイブを data/backups/ へ退避する。退避先を返す。"""
    target = path or ARCHIVE_PATH
    backup_path = None
    if backup and target.exists():
        backup_path = target.parent / "backups" / f"user_personal_memory_archive_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup_path)
    data["updated_at"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return backup_path


def collect_candidates(repo_root: Path = PROJECT_ROOT, data_path_resolver=get_data_path) -> list[dict[str, Any]]:
    """読み込み側と同じ候補行（上限なし）を、出所と日付つきで返す。"""
    from api.chat_user_personal_memory import is_personal_memory_candidate, personal_memory_sources

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path, all_lines, _limit in personal_memory_sources(repo_root, data_path_resolver):
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        section_date = ""
        for raw in raw_lines:
            line = raw.strip()
            m = _SECTION_DATE_RE.match(line)
            if line.startswith("## "):
                section_date = m.group(1) if m else ""
            if not is_personal_memory_candidate(line, all_lines=all_lines):
                continue
            key = line_key(line)
            if key in seen:
                continue
            seen.add(key)
            captured = _CAPTURED_RE.match(line)
            inline = _INLINE_DATE_RE.search(line)
            date = captured.group(1) if captured else inline.group(1) if inline else section_date
            rows.append({
                "id": item_id(key),
                "key": key,
                "source": path.name,
                "text": line,
                "body": captured.group(2) if captured else key,
                "captured": bool(captured),
                "date": date,
            })
    return rows


def _active_statements(path: Path) -> list[str]:
    try:
        rules = json.loads(path.read_text(encoding="utf-8")).get("rules") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [str(r.get("canonical_statement") or "") for r in rules if r.get("status") == "active"]


def classify(
    rows: list[dict[str, Any]],
    *,
    today: dt.date,
    pinned: set[str],
    statements: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(自動アーカイブ, アーカイブ候補) を返す。どちらも row に reason（と merged_into）を足したもの。"""
    auto: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    archived_ids: set[str] = set()

    def age(row: dict[str, Any]) -> int | None:
        try:
            return (today - dt.date.fromisoformat(row["date"])).days
        except (TypeError, ValueError):
            return None

    def protected(row: dict[str, Any]) -> bool:
        days = age(row)
        return row["key"] in pinned or bool(_PROTECT_RE.search(row["text"])) or (days is not None and days <= RECENT_DAYS)

    def add(row: dict[str, Any], reason: str, merged_into: str = "") -> None:
        auto.append({**row, "reason": reason, **({"merged_into": merged_into} if merged_into else {})})
        archived_ids.add(row["id"])

    for row in rows:
        if protected(row):
            continue
        body = _REMEMBER_RE.sub("", row["body"]).strip(" 　。、")
        if _TEMPLATE_RE.match(row["text"]):
            add(row, "empty_template")
        elif row["captured"] and (len(body) < 4 or _QUESTION_END_RE.search(body)):
            add(row, "question_not_fact")
        elif row["captured"] and any(_jaccard(body, s) >= COVERED_BY_ASSET for s in statements):
            add(row, "covered_by_judgment_asset")

    # ほぼ同文の重複は新しい方（日付が無ければ後ろの行）を残す
    live = [r for r in rows if r["id"] not in archived_ids]
    for i, older in enumerate(live):
        if older["id"] in archived_ids or older["key"] in pinned:
            continue
        for newer in live[i + 1 :]:
            if newer["id"] in archived_ids or _jaccard(older["key"], newer["key"]) < NEAR_DUPLICATE:
                continue
            keep, drop = (older, newer) if (older["date"] or "") > (newer["date"] or "") else (newer, older)
            if not protected(drop):
                add(drop, "duplicate", keep["id"])
                break

    for row in rows:
        if row["id"] in archived_ids or row["key"] in pinned:
            continue
        days = age(row)
        if days is not None and days > STALE_DAYS:
            if protected(row):
                review.append({**row, "reason": "old_but_relational"})
            else:
                add(row, f"stale_{STALE_DAYS}d")
        elif days is None and not protected(row) and _TECH_RE.search(row["text"]):
            review.append({**row, "reason": "technical_note_undated"})
    return auto, review


# --- Jev（重複・上書きの「要確認」候補の絞り込みだけに使う。Jev 単独では自動アーカイブしない） ---
# 2026-10-03 の計測（experiments/personal_memory_dedup_jev/）: 外してよいペアの AUC Jev 0.92・埋め込み 0.71・文字一致 0.38
JEV_REVIEW_MIN = 0.4
JEV_MAX_PAIRS = 20
_NEVER_SEND_RE = re.compile(r"妹|亡くな|\[sensitive")
JEV_QUESTIONS = {
    "dup": {
        "instructions": "{a} と {b} は、ユーザーについての同じ趣旨の記憶で、片方を残せばもう片方は不要か？",
        "true": "同じ事実・好み・方針・出来事を述べている。言い回し、言語（日本語/英語）、補足の量の違いだけ。",
        "false": "対象、主張、時期、推奨する行動のどれかが違い、両方残す意味がある。似た話題でも別の論点なら false。",
    },
    "sup": {
        "instructions": "{b} は {a} より新しい情報で、{a} の内容を上書き・訂正・再定義しており、{a} はもう古くなったか？",
        "true": "{b} が {a} と同じ対象について、方針の変更・定義のやり直し・状況の変化（終了・撤回など）を述べ、{a} をそのまま使うと誤る。",
        "false": "{b} は {a} と別の話題、または {a} を補足するだけで、{a} は今も有効。",
    },
}


def jev_text(row: dict[str, Any], secrets: list[str]) -> str | None:
    """Jev へ送る本文（個人事実の値を伏せ字にする）。送ってはいけない行は None。"""
    text = re.sub(r"\s*\(`memory/[^)]*`\)\s*$", "", re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", row["body"]))
    if _NEVER_SEND_RE.search(row["text"]):
        return None
    for value in secrets:
        text = text.replace(value, "[個人名]")
    try:
        import typesafe_dedup_guard as transport

        return text if transport.is_safe_public_candidate({"title": text}) else None
    except Exception:  # noqa: BLE001 - 判定器が無ければ送らない
        return None


_SECRET_FACT_RE = re.compile(r"(?:Dog name|Preferred name|What to call them:\*\*)\s*:?\s*([^\s:*]+)")


def personal_fact_values(texts: list[str]) -> list[str]:
    """伏せ字にする個人事実の値（犬の名前・呼び方など）。アーカイブ済みの行や本体ファイルも含めて拾う。"""
    values = {m.group(1) for t in texts for m in _SECRET_FACT_RE.finditer(t)}
    return sorted((v for v in values if v not in {"User", "未記録（ユーザーから次に教えてもらったらここへ保存する）"}), key=len, reverse=True)


def jev_review_candidates(rows: list[dict[str, Any]], *, pinned: set[str], similarity_fn, pair_scorer, secrets: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """似ているペアを Jev に「重複か」「上書きか」聞き、高いものを要確認にする。(要確認, 判定ログ用) を返す。"""
    sendable = [(r, t) for r in rows for t in [jev_text(r, secrets)] if t]
    if len(sendable) < 2:
        return [], []
    sim = similarity_fn([t for _, t in sendable])
    scored = []
    for i in range(len(sendable)):
        for j in range(i + 1, len(sendable)):
            a, b = sendable[i][0], sendable[j][0]
            if a["key"] in pinned and b["key"] in pinned:
                continue
            value = float(sim[i][j]) if sim is not None else _jaccard(sendable[i][1], sendable[j][1])
            older, newer = (i, j) if (a["date"] or "") <= (b["date"] or "") else (j, i)
            scored.append((value, older, newer))
    top = sorted(scored, reverse=True)[:JEV_MAX_PAIRS]
    if not top:
        return [], []
    texts = [(sendable[o][1], sendable[n][1]) for _, o, n in top]
    scores = {name: pair_scorer(texts, question) for name, question in JEV_QUESTIONS.items()}
    review, log_items = [], []
    for k, (_, o, n) in enumerate(top):
        older, newer = sendable[o][0], sendable[n][0]
        dup, sup = scores["dup"][k], scores["sup"][k]
        for name, prob in (("dup", dup), ("sup", sup)):
            log_items.append({"subject": f"{texts[k][0]}\n{texts[k][1]}", "question": f"personal_memory_{name}", "probability": prob,
                              "choice": prob >= JEV_REVIEW_MIN, "route": "review" if prob >= JEV_REVIEW_MIN else "keep", "auto_passed": False,
                              "thresholds": {"review_min": JEV_REVIEW_MIN}})
        if max(dup, sup) >= JEV_REVIEW_MIN and older["key"] not in pinned:
            review.append({**older, "reason": "jev_duplicate" if dup >= sup else "jev_superseded", "merged_into": newer["id"], "jev": round(max(dup, sup), 2)})
    return review, log_items


def _block_stats(repo_root: Path, data_path_resolver) -> dict[str, int]:
    from api.chat_user_personal_memory import invalidate_user_personal_memory_cache, load_user_personal_memory_payload

    invalidate_user_personal_memory_cache()
    payload = load_user_personal_memory_payload(repo_root=repo_root, data_path_resolver=data_path_resolver)
    invalidate_user_personal_memory_cache()
    return {"lines": int(payload.get("line_count") or 0), "chars": len(payload.get("block") or "")}


def run(
    *,
    dry_run: bool,
    archive_path: Path | None = None,
    repo_root: Path = PROJECT_ROOT,
    data_path_resolver=get_data_path,
    canonical_path: Path | None = None,
    today: dt.date | None = None,
    similarity_fn=None,
    pair_scorer=None,
) -> dict[str, Any]:
    """週次の整理。pair_scorer（Jev）を渡すと、重複・上書きの要確認候補を Jev で絞り込む（失敗しても整理は続ける）。"""
    archive_path = archive_path or Path(data_path_resolver("user_personal_memory_archive.json"))
    today = today or dt.date.today()
    data = load_archive(archive_path)
    before = _block_stats(repo_root, data_path_resolver)
    done = {str(i.get("key")) for i in data["items"]}
    pinned = {str(p.get("key")) for p in data["pinned"]}
    all_rows = collect_candidates(repo_root, data_path_resolver)
    rows = [r for r in all_rows if r["key"] not in done]
    auto, review = classify(
        rows,
        today=today,
        pinned=pinned,
        statements=_active_statements(canonical_path or Path(data_path_resolver("canonical_judgment_rules.json"))),
    )
    jev_status = "off"
    if pair_scorer is not None:
        archived_now = {r["id"] for r in auto}
        raw_texts = [r["text"] for r in all_rows] + [str(i.get("text") or "") for i in data["items"]]
        from api.chat_user_personal_memory import personal_memory_sources

        for path, _all, _limit in personal_memory_sources(repo_root, data_path_resolver)[:2]:  # 個人記憶ファイル本体
            try:
                raw_texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
        try:
            jev_review, log_items = jev_review_candidates(
                [r for r in rows if r["id"] not in archived_now], pinned=pinned, similarity_fn=similarity_fn or (lambda _t: None), pair_scorer=pair_scorer,
                secrets=personal_fact_values(raw_texts),
            )
            listed = {r["id"] for r in review}
            review += [r for r in jev_review if r["id"] not in listed]
            jev_status = f"ok ({len(log_items) // 2}ペア)"
            if log_items and not dry_run:
                import jev_judgment_log

                jev_judgment_log.append_records(jev_judgment_log.build_records(
                    guard="user_personal_memory_dedup", run_id=jev_judgment_log.new_run_id(), mode="weekly", model="jev-latest", items=log_items))
        except Exception as exc:  # noqa: BLE001 - Jev が使えなくても決定的ルールの整理は続ける
            jev_status = f"skipped ({type(exc).__name__})"
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    report: dict[str, Any] = {"jev": jev_status, "date": today.isoformat(), "dry_run": dry_run, "before": before, "auto_archived": len(auto), "review_candidates": len(review)}
    if not dry_run:
        for row in auto:
            data["items"].append({k: row[k] for k in ("id", "key", "source", "text", "date", "reason") if k in row} | ({"merged_into": row["merged_into"]} if row.get("merged_into") else {}) | {"archived_at": now, "by": "weekly_auto"})
        data["review_candidates"] = [{k: r[k] for k in ("id", "source", "text", "date", "reason", "merged_into", "jev") if k in r} for r in review]
        report["backup"] = str(save_archive(data, archive_path) or "")
        report["after"] = _block_stats(repo_root, data_path_resolver)
        data["last_run"] = report
        save_archive(data, archive_path, backup=False)
    report["auto"] = [{"id": r["id"], "reason": r["reason"], "source": r["source"]} for r in auto]
    return report


def restore(target_id: str, *, archive_path: Path | None = None) -> dict[str, Any]:
    """アーカイブから戻す。戻した行は pinned に入れ、週次の自動整理で再びアーカイブしない。"""
    data = load_archive(archive_path)
    hit = [i for i in data["items"] if i.get("id") == target_id]
    if not hit:
        return {"restored": False, "reason": "not_found"}
    data["items"] = [i for i in data["items"] if i.get("id") != target_id]
    data["pinned"].append({"key": hit[0]["key"], "reason": "restored", "at": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
    save_archive(data, archive_path)
    return {"restored": True, "id": target_id, "text": hit[0].get("text", "")}


def morning_report_line(archive_path: Path | None = None) -> str:
    """AURION CORE 朝報向けの1行。"""
    data = load_archive(archive_path)
    last = data.get("last_run")
    if not isinstance(last, dict):
        return "- 個人記憶の整理（週次）: まだ実行されていません"
    before, after = last.get("before") or {}, last.get("after") or last.get("before") or {}
    return (
        f"- 個人記憶の整理（週次 {last.get('date')}）: {before.get('chars', 0):,}→{after.get('chars', 0):,}字、"
        f"今回アーカイブ {last.get('auto_archived', 0)}件、要確認 {last.get('review_candidates', 0)}件"
        f"（累計アーカイブ {len(data['items'])}件・`python -m api.user_personal_memory_archive --list`）"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="紫苑のユーザー個人記憶を整理する（削除せずアーカイブ）")
    parser.add_argument("--dry-run", action="store_true", help="判定だけして書き換えない")
    parser.add_argument("--restore", metavar="ID", help="アーカイブから戻す")
    parser.add_argument("--list", action="store_true", help="アーカイブ済みと要確認の一覧")
    parser.add_argument("--archive-path", type=Path, help="アーカイブJSONの保存先（Cloud Run bundle生成用）")
    args = parser.parse_args()
    if args.restore:
        print(json.dumps(restore(args.restore), ensure_ascii=False, indent=2))
    elif args.list:
        data = load_archive()
        for item in data["items"]:
            print(f"[{item.get('id')}] {item.get('reason')}  {item.get('source')}  {str(item.get('text'))[:70]}")
        for item in data["review_candidates"]:
            print(f"(要確認 {item.get('id')}) {item.get('reason')}  {item.get('source')}  {str(item.get('text'))[:70]}")
    else:
        print(json.dumps(run(dry_run=args.dry_run, archive_path=args.archive_path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
