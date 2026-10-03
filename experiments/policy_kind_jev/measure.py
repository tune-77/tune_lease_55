#!/usr/bin/env python3
"""教えたノウハウの「方針/知見」判定で、現行ルール・Jev・埋め込み類似度を AUC で比べる。

入力（コミットしない）:
  data/policy_kind_items_20261004.json   collect.py の出力（文単位）
  data/policy_kind_labels_20261004.json  正解 v1（Claude の基準。ユーザーの基準とずれていたので無効扱い）
  data/policy_kind_labels_20261004_v2.json  正解 v2（ユーザーの説明に合わせた基準で Claude が付け直した119件）
  data/policy_kind_human_labels_20261004.json  ユーザーが付けた borderline 16件（評価専用）
--human を渡すと16件は Claude ラベルから外し、埋め込みの参照にも使わず、別に集計する。
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
GUARD = "knowledge_kind_policy"
QUESTION_V1 = {
    "instructions": "{x} は、リース会社の担当者が教えた一文です。これは社内の取扱い方針（どの先・物件を取り扱う/取り扱わない、慎重にする、優先する、必ず行う等の、当社の行動基準）を述べていますか。",
    "true": "当社としてどう扱うか（取り扱う・取り扱わない・慎重にする・前向きに見る・優先する・禁止・義務）を定めている。肯定形や程度の表現でもよい。",
    "false": "市場や業界の傾向、相場、事実、制度の説明、確認すべき項目や手順、感想・雑談で、当社の取扱い基準を定めていない。",
}
# v2: ユーザーの「方針」は審査の心構えではなく、具体的な対象の取扱い基準（やれる/やれない/こうする）
QUESTION_V2 = {
    "instructions": "{x} は、リース会社の担当者が教えた一文です。これは、特定の物件・業種・取引先（の状態）・取引形態や契約条件について、当社としての取扱い基準（取り扱える/取り扱えない/不向き、取り扱う条件や前提、審査で重点的に見ること）を決めていますか。",
    "true": "対象が具体的（ある物件や設備、業種、借手の状態、契約条件など）で、その対象をどう扱うか（可否・条件・前提・重点）を決めている。",
    "false": "対象を特定しない一般論、審査の心構え、汎用の確認手順や確認項目の列挙、市場や業界の傾向・相場・事実・制度の説明、理由づけ、感想・雑談。",
}
QUESTIONS = {"v1": QUESTION_V1, "v2": QUESTION_V2}
_SAME_PREFIX_RE = re.compile(r"^[\s）)（(]*(?:同旨[:：])?\s*")


def clean(text: str) -> str:
    return _SAME_PREFIX_RE.sub("", text).replace("**", "").strip()


def judge(items: list[dict], batch: int, question: dict) -> str:
    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    model = ""
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        questions = {
            f"item{n}_policy": {
                "type": "noul",
                "instructions": question["instructions"].format(x=f"`items[{n}]`"),
                "criteria": {"true": question["true"], "false": question["false"]},
            }
            for n in range(len(chunk))
        }
        body = transport._default_request({"state": {"items": [i["masked"] for i in chunk]}, "model": "jev-latest", "questions": questions})
        model = str(body.get("model") or model)
        for n, item in enumerate(chunk):
            item["jev"] = transport._noul(body["answers"], f"item{n}_policy")
    return model


def embedding_scores(items: list[dict]) -> None:
    """ラベル済みの他の文との類似度で「方針らしさ」を出す（leave-one-out、近い3件ずつの平均の差）。
    参照は Claude ラベルの文だけ。ユーザーの16件は参照に入れない（評価専用）。"""
    sim = embedding_similarity([i["clean"] for i in items])
    ref = [b for b, o in enumerate(items) if o["split"] == "claude"]
    for a, item in enumerate(items):
        if sim is None:
            item["embedding"] = 0.0
            continue
        pos = sorted((sim[a][b] for b in ref if b != a and items[b]["label"] == 1), reverse=True)[:3]
        neg = sorted((sim[a][b] for b in ref if b != a and items[b]["label"] == 0), reverse=True)[:3]
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


def log_offline(items: list[dict], model: str) -> int:
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
        jev_judgment_log.append_label(record["judgment_id"], bool(item["label"]), label_source=item["label_source"])
    return len(records)


def label_previous_run(items: list[dict], run_id: str) -> int:
    """前回の判定ログ（run_id）のうちユーザーが正解を付けた文へ human ラベルを追記する。"""
    path = jev_judgment_log.log_path()
    if path is None or not path.exists():
        return 0
    human = {jev_judgment_log.subject_hash(i["masked"]): i for i in items if i["split"] == "human"}
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("record_type") != "judgment" or row.get("run_id") != run_id or row.get("guard") != GUARD:
            continue
        item = human.get(row.get("subject_hash"))
        if item is not None:
            count += jev_judgment_log.append_label(row["judgment_id"], bool(item["label"]), label_source=item["label_source"])
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--cached", action="store_true")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--human", type=Path, default=None, help="ユーザーの正解（評価専用。Claude ラベルより優先）")
    parser.add_argument("--question", choices=sorted(QUESTIONS), default="v1")
    parser.add_argument("--label-previous-run", default="", help="この run_id の判定ログへ human ラベルを追記")
    args = parser.parse_args()
    out = DATA / ("policy_kind_jev_eval_20261004.json" if args.question == "v1" else f"policy_kind_jev_eval_20261004_{args.question}.json")

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    human = json.loads(args.human.read_text(encoding="utf-8")) if args.human else {"labels": {}}
    items = []
    for item in json.loads(ITEMS.read_text(encoding="utf-8")):
        key = str(item["n"])
        if key in human["labels"]:
            label, split, source = human["labels"][key], "human", human["label_source"]
        elif key in labels["labels"]:
            label, split, source = labels["labels"][key], "claude", labels["label_source"]
        else:
            continue
        text = clean(item["text"])
        masked = mask_for_jev(text)
        if not masked:
            continue  # PII様の内容が残る文は送らず、比較からも外す（同じ母集団で比べる）
        items.append({**item, "clean": text, "masked": masked, "label": label, "split": split, "label_source": source,
                      "rule": float(classify_knowledge_kind(text) == POLICY)})
    embedding_scores(items)
    model = ""
    if args.cached:
        cached = {i["n"]: i["jev"] for i in json.loads(out.read_text(encoding="utf-8"))["items"]}
        # 伏せ字の判定が変わって前回送らなかった文が増えることがある。前回と同じ母集団で比べる
        items = [{**item, "jev": cached[item["n"]]} for item in items if item["n"] in cached]
    else:
        model = judge(items, args.batch, QUESTIONS[args.question])
    summary = {"question": args.question, "model": model, "all": summarize(items)}
    for split in ("claude", "human"):
        part = [i for i in items if i["split"] == split]
        if part:
            summary[split] = {"label_source": part[0]["label_source"], **summarize(part)}
    if not args.cached and not args.no_log:
        summary["logged_records"] = log_offline(items, model)
    if args.label_previous_run and not args.no_log:
        summary["previous_run_human_labels"] = label_previous_run(items, args.label_previous_run)
    out.write_text(json.dumps({"summary": summary, "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
