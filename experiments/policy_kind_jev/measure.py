#!/usr/bin/env python3
"""教えたノウハウの「方針/知見」判定で、現行ルール・Jev・埋め込み類似度を AUC で比べる。

入力（コミットしない）:
  data/policy_kind_items_20261004.json   collect.py の出力（文単位）
  data/policy_kind_labels_20261004.json  正解（Claude が基準に沿って付けたもの。ユーザー確認前なので甘く出る）
Jev に送る文は api.chat_judgment_asset_capture.mask_for_jev（企業名・人名の伏せ字、PII様なら送らない）を通す。
Jev の判定は REV-424 判定ログへ guard=knowledge_kind_policy, mode=offline_label_eval で残し、ラベルも append_label する。

使い方:
  DATA_DIR=<本番の data/> TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/policy_kind_jev/measure.py
  （--no-log で判定ログへ書かない。--cached で前回の Jev 結果を再利用）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import jev_judgment_log  # noqa: E402
import typesafe_dedup_guard as transport  # noqa: E402
from api.chat_judgment_asset_capture import mask_for_jev  # noqa: E402
from api.judgment_policy import POLICY, classify_knowledge_kind  # noqa: E402
from judgment_asset_dedup_jev.measure import auc, bootstrap_ci, brier, loo_platt  # noqa: E402
from runtime_paths import get_data_dir  # noqa: E402
from scripts.judgment_asset_dedup import embedding_similarity  # noqa: E402

DATA = get_data_dir()
ITEMS = DATA / "policy_kind_items_20261004.json"
LABELS = DATA / "policy_kind_labels_20261004.json"
OUT = DATA / "policy_kind_jev_eval_20261004.json"
GUARD = "knowledge_kind_policy"
QUESTION = {
    "instructions": "{x} は、リース会社の担当者が教えた一文です。これは社内の取扱い方針（どの先・物件を取り扱う/取り扱わない、慎重にする、優先する、必ず行う等の、当社の行動基準）を述べていますか。",
    "true": "当社としてどう扱うか（取り扱う・取り扱わない・慎重にする・前向きに見る・優先する・禁止・義務）を定めている。肯定形や程度の表現でもよい。",
    "false": "市場や業界の傾向、相場、事実、制度の説明、確認すべき項目や手順、感想・雑談で、当社の取扱い基準を定めていない。",
}
_SAME_PREFIX_RE = re.compile(r"^[\s）)（(]*(?:同旨[:：])?\s*")


def clean(text: str) -> str:
    return _SAME_PREFIX_RE.sub("", text).replace("**", "").strip()


def judge(items: list[dict], batch: int) -> str:
    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    model = ""
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        questions = {
            f"item{n}_policy": {
                "type": "noul",
                "instructions": QUESTION["instructions"].format(x=f"`items[{n}]`"),
                "criteria": {"true": QUESTION["true"], "false": QUESTION["false"]},
            }
            for n in range(len(chunk))
        }
        body = transport._default_request({"state": {"items": [i["masked"] for i in chunk]}, "model": "jev-latest", "questions": questions})
        model = str(body.get("model") or model)
        for n, item in enumerate(chunk):
            item["jev"] = transport._noul(body["answers"], f"item{n}_policy")
    return model


def embedding_scores(items: list[dict]) -> None:
    """ラベル済みの他の文との類似度で「方針らしさ」を出す（leave-one-out、近い3件ずつの平均の差）。"""
    sim = embedding_similarity([i["clean"] for i in items])
    for a, item in enumerate(items):
        if sim is None:
            item["embedding"] = 0.0
            continue
        pos = sorted((sim[a][b] for b, o in enumerate(items) if b != a and o["label"] == 1), reverse=True)[:3]
        neg = sorted((sim[a][b] for b, o in enumerate(items) if b != a and o["label"] == 0), reverse=True)[:3]
        item["embedding"] = float(np.mean(pos) - np.mean(neg))


def summarize(items: list[dict]) -> dict:
    y = np.array([i["label"] for i in items])
    row: dict = {"n": len(items), "policy": int(y.sum()), "insight": int((1 - y).sum())}
    for metric in ("rule", "jev", "embedding"):
        s = np.array([i[metric] for i in items], dtype=float)
        row[metric] = {
            "auc": round(auc(y, s), 3),
            "auc_ci95": [round(v, 3) for v in bootstrap_ci(y, s, auc)],
            "brier_raw": round(brier(y, s), 3) if metric in ("rule", "jev") else None,
            "brier_loo_platt": round(brier(y, loo_platt(y, s)), 3),
        }
    for threshold in (0.5, 0.6, 0.7):
        pred = np.array([i["jev"] >= threshold for i in items])
        row[f"jev_at_{threshold}"] = {
            "tp": int((pred & (y == 1)).sum()),
            "fp": int((pred & (y == 0)).sum()),
            "fn": int((~pred & (y == 1)).sum()),
            "rule_missed_policy_recovered": sum(1 for i, p in zip(items, pred) if p and i["label"] == 1 and i["rule"] == 0),
        }
    return row


def log_offline(items: list[dict], model: str, label_source: str) -> int:
    records = jev_judgment_log.build_records(
        guard=GUARD,
        run_id=jev_judgment_log.new_run_id(),
        mode="offline_label_eval",
        model=model or "jev-latest",
        items=[
            {"subject": i["masked"], "question": "is_policy", "probability": i["jev"], "choice": i["jev"] >= 0.5,
             "route": "offline_eval", "auto_passed": False, "thresholds": {}}
            for i in items
        ],
    )
    jev_judgment_log.append_records(records)
    for record, item in zip(records, items):
        jev_judgment_log.append_label(record["judgment_id"], bool(item["label"]), label_source=label_source)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    items = []
    for item in json.loads(ITEMS.read_text(encoding="utf-8")):
        label = labels["labels"].get(str(item["n"]))
        if label is None:
            continue
        text = clean(item["text"])
        masked = mask_for_jev(text)
        if not masked:
            continue  # PII様の内容が残る文は送らず、比較からも外す（同じ母集団で比べる）
        items.append({**item, "clean": text, "masked": masked, "label": label, "rule": float(classify_knowledge_kind(text) == POLICY)})
    embedding_scores(items)
    model = ""
    if args.cached:
        cached = {i["n"]: i["jev"] for i in json.loads(OUT.read_text(encoding="utf-8"))["items"]}
        for item in items:
            item["jev"] = cached[item["n"]]
    else:
        model = judge(items, args.batch)
    summary = summarize(items)
    summary["model"] = model
    summary["label_source"] = labels["label_source"]
    if not args.cached and not args.no_log:
        summary["logged_records"] = log_offline(items, model, labels["label_source"])
    OUT.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
