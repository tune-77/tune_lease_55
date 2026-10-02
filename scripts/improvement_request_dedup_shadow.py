#!/usr/bin/env python3
"""改善ログの新着要望に「既に同じ要望があるか」を Jev で判定し、重複候補として記録だけする（shadow）。

対象は data/cloudrun_improvement_log.jsonl のチャット改善メモと紫苑の自己提案。
既存の重複チェック（title 完全一致・deduplicate_improvements の決定的ルール）で統合できるペアは
そのまま既存ルールに任せ、ルールが別件とした近傍ペアだけを Jev に聞く。ログ本体は書き換えない。

2026-10-02 実測（experiments/same_event_dedup_jev/）: 近傍ペア90組で AUC 0.99（既存ルール0.57・
文字Jaccard 0.86・埋め込み0.85）。確率は低めに出るため閾値は 0.45（誤統合0・正例26/34）。
Jev が使えない時は既存ルールの結果だけで終わる（候補は空のまま、exit 0）。

使い方:
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python scripts/improvement_request_dedup_shadow.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
LOG_JSONL = DATA_DIR / "cloudrun_improvement_log.jsonl"
STATE_JSON = DATA_DIR / "improvement_request_dedup_state.json"
LATEST_JSON = DATA_DIR / "improvement_request_dedup_latest.json"

REQUEST_QUESTION = {
    "instructions": "{a} と {b} は、同じ改善要望として1件に統合してよいか？",
    "true": "対象の画面・機能・応答と、求める変更が同じ。片方を実装すればもう片方も満たされる。",
    "false": "対象の画面・機能、または求める変更（追加・削除・修正の中身）が違い、別々に対応する必要がある。",
}
DUPLICATE_MIN = 0.45
NEIGHBORS = 3
FIRST_RUN_DAYS = 7

JevFn = Callable[[list[tuple[str, str]]], tuple[list[float], str]]


def _line(body: str, label: str) -> str:
    match = re.search(rf"(?:^|\n)\s*-?\s*{label}\s*[:：]\s*([^\n]+)", body)
    return match.group(1).strip() if match else ""


def _section(body: str, label: str) -> str:
    match = re.search(rf"## {label}\n(.+?)(?:\n\n|\n##|$)", body, re.S)
    return match.group(1).strip().replace("\n", " ") if match else ""


def request_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """改善要望だけを取り出す（「改善ログ: approved」等の状態変更記録は要望ではないので除く）。"""
    items = []
    for row in rows:
        title, body = str(row.get("title") or ""), str(row.get("body") or "")
        if title.startswith("改善ログ:") or row.get("source") == "domestic_mode" or not row.get("event_id"):
            continue
        if row.get("surface") == "chat_improvement":
            text = _line(body, "課題") or _section(body, "原文") or _section(body, "ユーザー要望") or title
            detail = _line(body, "改善案")
        else:
            text, detail = title, str(row.get("proposed_change") or "")
        if text:
            items.append(
                {"id": str(row["event_id"]), "surface": str(row.get("surface") or ""), "ts": str(row.get("ts") or ""), "title": text[:120], "detail": detail[:150]}
            )
    return items


def request_text(item: dict[str, Any]) -> str:
    return f"{item['title']}／{item['detail']}" if item.get("detail") else item["title"]


def rule_duplicate(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """既存の重複チェック（scheduler の title 完全一致＋deduplicate_improvements）。"""
    from scripts.extract_obsidian_improvements import deduplicate_improvements

    if a["title"] == b["title"]:
        return True
    return len(deduplicate_improvements([{"title": a["title"], "reason": ""}, {"title": b["title"], "reason": ""}])) == 1


def candidate_pairs(items: list[dict[str, Any]], new_ids: set[str], embeddings: list[list[float]] | None) -> list[tuple[int, int]]:
    """新着1件ごとに、同じ面（チャット/紫苑提案）の近傍を文字一致・埋め込みそれぞれ上位 NEIGHBORS 件拾う。"""
    from scripts.extract_obsidian_improvements import _jaccard_similarity

    pairs: dict[tuple[int, int], None] = {}
    for k, item in enumerate(items):
        if item["id"] not in new_ids:
            continue
        others = [j for j, other in enumerate(items) if j != k and other["surface"] == item["surface"]]
        near = sorted(others, key=lambda j: -_jaccard_similarity(item["title"], items[j]["title"]))[:NEIGHBORS]
        if embeddings is not None:
            near += sorted(others, key=lambda j: -embeddings[k][j])[:NEIGHBORS]
        for j in near:
            pairs[(k, j)] = None
    return list(pairs)


def _default_jev(pairs: list[tuple[str, str]]) -> tuple[list[float], str]:
    import os

    import typesafe_dedup_guard as transport

    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "60")
    return transport.judge_binary_pairs(pairs, REQUEST_QUESTION)


def _log_jev(judged: list[dict[str, Any]], model: str) -> None:
    try:
        import jev_judgment_log

        jev_judgment_log.append_records(
            jev_judgment_log.build_records(
                guard="improvement_request_dedup",
                run_id=jev_judgment_log.new_run_id(),
                mode="shadow",
                model=model,
                items=[
                    {
                        "subject": f"{c['new_text']}\n{c['existing_text']}",
                        "question": "same_request",
                        "probability": c["jev"],
                        "choice": c["jev"] >= DUPLICATE_MIN,
                        "route": "duplicate_candidate" if c["jev"] >= DUPLICATE_MIN else "distinct",
                        "auto_passed": True,
                        "thresholds": {"duplicate_min": DUPLICATE_MIN},
                    }
                    for c in judged
                ],
            )
        )
    except Exception as exc:  # noqa: BLE001 - ログ失敗で本処理を止めない
        print(f"[request-dedup] judgment log skipped: {type(exc).__name__}", file=sys.stderr)


def run(
    *,
    log_path: Path = LOG_JSONL,
    state_path: Path = STATE_JSON,
    latest_path: Path = LATEST_JSON,
    jev_fn: JevFn | None = _default_jev,
    embed_fn: Callable[[list[str]], list[list[float]] | None] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    import typesafe_dedup_guard as transport

    now = now or datetime.now(timezone.utc)
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()] if log_path.exists() else []
    items = request_items(rows)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    if state is None:
        # 初回は全履歴を新着扱いにせず、直近 FIRST_RUN_DAYS 日分だけを見る
        cutoff = (now - timedelta(days=FIRST_RUN_DAYS)).date().isoformat()
        new_ids = {item["id"] for item in items if item["ts"][:10] >= cutoff}
    else:
        seen = set(state.get("seen") or [])
        new_ids = {item["id"] for item in items if item["id"] not in seen}

    report: dict[str, Any] = {"generated_at": now.isoformat(), "new_requests": len(new_ids), "rule_duplicates": 0, "candidates": []}
    pairs = candidate_pairs(items, new_ids, (embed_fn or _embeddings)([item["title"] for item in items])) if new_ids else []
    to_judge = []
    for k, j in pairs:
        if rule_duplicate(items[k], items[j]):
            report["rule_duplicates"] += 1  # 既存ルールで統合できるものはそのまま
        elif transport.is_safe_public_candidate({"title": request_text(items[k])}) and transport.is_safe_public_candidate({"title": request_text(items[j])}):
            to_judge.append((k, j))
    report["jev_pairs"] = len(to_judge)
    if not to_judge or jev_fn is None:
        report["status"] = "skipped"
    else:
        try:
            scores, model = jev_fn([(request_text(items[k]), request_text(items[j])) for k, j in to_judge])
        except Exception as exc:  # noqa: BLE001 - Jev 不通時は既存ルールのみ
            report["status"] = "fallback"
            report["error_type"] = type(exc).__name__
        else:
            judged = [
                {
                    "new_id": items[k]["id"],
                    "existing_id": items[j]["id"],
                    "new_text": request_text(items[k]),
                    "existing_text": request_text(items[j]),
                    "jev": round(score, 3),
                }
                for (k, j), score in zip(to_judge, scores)
            ]
            _log_jev(judged, model)
            report["status"] = "applied"
            report["model"] = model
            report["candidates"] = sorted((c for c in judged if c["jev"] >= DUPLICATE_MIN), key=lambda c: -c["jev"])
    if report["status"] != "fallback":
        state_path.write_text(json.dumps({"seen": sorted(item["id"] for item in items)}, ensure_ascii=False) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _embeddings(texts: list[str]) -> list[list[float]] | None:
    from scripts.judgment_asset_dedup import embedding_similarity

    return embedding_similarity(texts)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    report = run()
    print(
        f"[request-dedup] status={report['status']} new={report['new_requests']} "
        f"rule_duplicates={report['rule_duplicates']} jev_pairs={report['jev_pairs']} candidates={len(report['candidates'])}"
    )
    for c in report["candidates"][:5]:
        print(f"  {c['jev']:.2f} {c['new_text'][:40]} ≒ {c['existing_text'][:40]}")
    # Jev 不通は既存ルールだけで続行できるが、続けば重複候補が出なくなるので失敗として見えるようにする
    return 1 if report["status"] == "fallback" else 0


if __name__ == "__main__":
    sys.exit(main())
