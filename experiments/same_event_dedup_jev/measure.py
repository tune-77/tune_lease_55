#!/usr/bin/env python3
"""「同じ出来事の別記事か」「同じ要望の重複か」で Jev が既存ルール・埋め込みより効くかを測る。

対象:
  news      同一実行（同日ノート＋関連報道、err.log の [news-guard-item] の1実行）内の見出しペア。
            既存ルール = collect_lease_news_to_obsidian._same_event（2文字Dice＋固有名/数値アンカー）
  requests  data/cloudrun_improvement_log.jsonl の改善要望（チャット改善メモ・紫苑の自己提案）の近傍ペア。
            既存ルール = deduplicate_improvements（完全一致/先頭40字/包含/テーマ群/Jaccard≥0.55）と
            scheduler の title 完全一致（ペア作成時に同題は除いているので常に0）

正解ラベルは data/same_event_dedup_labels_20261002.json（コミットしない）。
label=None は判断保留で集計から外し、ユーザー確認待ちにする。
Jev の判定は REV-424 判定ログへ mode=offline_label_eval で記録し、ラベルも append_label で残す。

使い方:
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/same_event_dedup_jev/measure.py news
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/same_event_dedup_jev/measure.py requests
  （--no-log で判定ログへ書かない。--cached で前回の Jev 結果を再利用して集計だけやり直す）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import jev_judgment_log  # noqa: E402
import typesafe_dedup_guard as transport  # noqa: E402
from judgment_asset_dedup_jev.measure import auc, bootstrap_ci, brier, loo_platt  # noqa: E402
from scripts.collect_lease_news_to_obsidian import (  # noqa: E402
    Article,
    _bigram_dice,
    _event_title,
    _normalized_title,
    _same_event,
    SAME_EVENT_JEV_QUESTION,
)
from scripts.extract_obsidian_improvements import _jaccard_similarity, deduplicate_improvements  # noqa: E402
from scripts.improvement_request_dedup_shadow import REQUEST_QUESTION, request_text  # noqa: E402
from scripts.judgment_asset_dedup import embedding_similarity  # noqa: E402

LABELS = ROOT / "data" / "same_event_dedup_labels_20261002.json"
OUT = ROOT / "data" / "same_event_dedup_jev_eval_20261002_{target}.json"
LABEL_SOURCE = "assistant:claude_rubric_20261002"



def _article(title: str) -> Article:
    return Article(title=title, link="", source="", published=None, summary="", query="")


def score_news(pairs: list[dict]) -> None:
    titles = sorted({t for p in pairs for t in (p["a"], p["b"])})
    index = {t: i for i, t in enumerate(titles)}
    emb = embedding_similarity([_event_title(_article(t)) for t in titles])
    for p in pairs:
        a, b = _article(p["a"]), _article(p["b"])
        p["rule"] = float(_same_event(a, b))
        p["dice"] = _bigram_dice(_normalized_title(_event_title(a)), _normalized_title(_event_title(b)))
        p["embedding"] = float(emb[index[p["a"]]][index[p["b"]]]) if emb is not None else 0.0


def score_requests(pairs: list[dict]) -> None:
    titles = sorted({p[k]["title"] for p in pairs for k in ("a", "b")})
    index = {t: i for i, t in enumerate(titles)}
    emb = embedding_similarity(titles)
    for p in pairs:
        ta, tb = p["a"]["title"], p["b"]["title"]
        merged = deduplicate_improvements([{"title": ta, "reason": ""}, {"title": tb, "reason": ""}])
        p["rule"] = float(len(merged) == 1)
        p["dice"] = _jaccard_similarity(ta, tb)
        p["embedding"] = float(emb[index[ta]][index[tb]]) if emb is not None else 0.0


def judge(pairs: list[dict], target: str, batch: int) -> str:
    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    texts = [(p["a"], p["b"]) if target == "news" else (request_text(p["a"]), request_text(p["b"])) for p in pairs]
    scores, model = transport.judge_binary_pairs(texts, SAME_EVENT_JEV_QUESTION if target == "news" else REQUEST_QUESTION, batch=batch)
    for p, prob in zip(pairs, scores):
        p["jev"] = prob
    return model


def summarize(pairs: list[dict]) -> dict:
    labeled = [p for p in pairs if p["label"] is not None]
    y = np.array([p["label"] for p in labeled])
    row: dict = {"n": len(labeled), "positives": int(y.sum()), "negatives": int((1 - y).sum()), "unsure": len(pairs) - len(labeled)}
    for metric in ("jev", "rule", "dice", "embedding"):
        s = np.array([p[metric] for p in labeled], dtype=float)
        row[metric] = {
            "auc": round(auc(y, s), 3),
            "auc_ci95": [round(v, 3) for v in bootstrap_ci(y, s, auc)],
            "brier_raw": round(brier(y, s), 3) if metric in ("jev", "rule") else None,
            "brier_loo_platt": round(brier(y, loo_platt(y, s)), 3),
        }
    # 既存ルールが統合できなかった正例のうち、Jev が拾えるもの（追加統合の上積み）
    for threshold in (0.5, 0.6, 0.7, 0.8):
        missed = [p for p in labeled if p["label"] == 1 and p["rule"] == 0]
        wrong = [p for p in labeled if p["label"] == 0 and p["rule"] == 0]
        row[f"jev_add_at_{threshold}"] = {
            "rule_missed_positives": len(missed),
            "jev_recovers": sum(p["jev"] >= threshold for p in missed),
            "jev_false_merges": sum(p["jev"] >= threshold for p in wrong),
        }
    return row


def log_offline(pairs: list[dict], target: str, model: str) -> int:
    guard = "news_same_event" if target == "news" else "improvement_request_dedup"
    question = "same_event" if target == "news" else "same_request"
    items = [
        {
            "subject": f"{p['a']}\n{p['b']}" if target == "news" else f"{request_text(p['a'])}\n{request_text(p['b'])}",
            "question": question,
            "probability": p["jev"],
            "choice": p["jev"] >= 0.5,
            "route": "offline_eval",
            "auto_passed": False,
            "thresholds": {},
        }
        for p in pairs
    ]
    records = jev_judgment_log.build_records(guard=guard, run_id=jev_judgment_log.new_run_id(), mode="offline_label_eval", model=model or "jev-latest", items=items)
    jev_judgment_log.append_records(records)
    for record, p in zip(records, pairs):
        if p["label"] is not None:
            jev_judgment_log.append_label(record["judgment_id"], bool(p["label"]), label_source=LABEL_SOURCE)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=("news", "requests"))
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()

    out_path = Path(str(OUT).format(target=args.target))
    pairs = json.loads(LABELS.read_text(encoding="utf-8"))[args.target]
    if args.target == "requests":
        # 社内の改善メモなので、外部送信前にローカルのプライバシー判定を通す（ニュース見出しは公開情報）
        pairs = [p for p in pairs if all(transport.is_safe_public_candidate({"title": request_text(p[k])}) for k in ("a", "b"))]
    (score_news if args.target == "news" else score_requests)(pairs)
    model = ""
    if args.cached:
        cached = {(p["a"] if isinstance(p["a"], str) else p["a"]["id"], p["b"] if isinstance(p["b"], str) else p["b"]["id"]): p["jev"] for p in json.loads(out_path.read_text(encoding="utf-8"))["pairs"]}
        for p in pairs:
            p["jev"] = cached[(p["a"] if isinstance(p["a"], str) else p["a"]["id"], p["b"] if isinstance(p["b"], str) else p["b"]["id"])]
    else:
        model = judge(pairs, args.target, args.batch)
    summary = summarize(pairs)
    summary["model"] = model
    if not args.cached and not args.no_log:
        summary["logged_records"] = log_offline(pairs, args.target, model)
    out_path.write_text(json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
