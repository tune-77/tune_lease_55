#!/usr/bin/env python3
"""Jev と Cloudflare Clef / Clef-flash を、これまでの Jev 評価セットで並べて比べる（2026-10-04）。

各評価は「Jev に送った伏せ字済みの本文とラベル」を data/ の結果ファイルから読み直し、
同じ組み立て関数・同じ質問文・同じバッチ幅で3モデルへ送る（母集団と送信内容を揃える）。
  - news_repayment     ニュースが返済能力・物件価値に効くか（人手99件）
  - chat_teaching      チャット発言が審査ノウハウか（/judgment-review の昇格・却下 103件）
  - judgment_asset_dup 判断資産の同一性（人手統合由来のペア）
  - personal_memory    個人記憶の重複・上書き（Claude 判定のペア。dup/sup の max）
  - policy_kind_human  方針か知見か（ユーザーの16件、v2 質問）

指標: AUC・Brier（生／leave-one-out Platt）の95%ブートストラップCI、Jev との AUC 差の対応ありCI、
リクエスト往復のレイテンシ（中央値・p95）、入力トークン。判定は REV-424 判定ログへ
guard=<評価のguard>:<model>、mode=offline_model_compare で記録し、ラベルも付ける。

結果はモデル単位でキャッシュする（--providers で一部だけ走らせ、残りは前回結果を使う）。
  DATA_DIR=<本番 data/> JEV_JUDGMENT_LOG_PATH=<本番 data/jev_judgment_log.jsonl> \
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key \
  CLOUDFLARE_API_TOKEN="$(security find-generic-password -s cloudflare-api-token -a tune-lease-55 -w)" \
  .venv/bin/python experiments/clef_vs_jev/compare.py --providers jev,clef,clef-flash
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import jev_judgment_log  # noqa: E402
import typesafe_dedup_guard as transport  # noqa: E402
import typesafe_news_guard as news  # noqa: E402
from api.chat_judgment_asset_capture import CHAT_TEACHING_JEV_QUESTION, mask_for_jev  # noqa: E402
from judgment_asset_dedup_jev.measure import auc, bootstrap_ci, brier, loo_platt  # noqa: E402
from personal_memory_dedup_jev.measure import QUESTIONS as MEMORY_QUESTIONS  # noqa: E402
from policy_kind_jev.measure import QUESTION_V2  # noqa: E402
from runtime_paths import get_data_dir  # noqa: E402
from scripts.judgment_asset_dedup import build_jev_pair_request, parse_jev_answers  # noqa: E402

from clef_client import ClefClient, jev_request  # noqa: E402  (同じディレクトリ)

DATA = get_data_dir()
CACHE = DATA / "clef_vs_jev_20261004.json"
CHAT_ITEMS = DATA / "clef_vs_jev_chat_items_20261004.json"
PROVIDERS = ("jev", "clef", "clef-flash")


# ── 評価セット ────────────────────────────────────────────────────────


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def news_items() -> list[dict]:
    items = []
    for name in ("jev_news_label_eval_20261001.jsonl", "jev_news_label_eval_20261001_batch2.jsonl"):
        for row in _jsonl(DATA / name):
            if row.get("record_type") != "item" or row.get("label") is None:
                continue
            article = {"title": row["title"], "source": row.get("source") or ""}
            if news.is_safe_public_article(article):
                items.append({"id": f"news{row['no']}", "article": article, "label": int(bool(row["label"])),
                              "subject": row["title"], "baseline_jev": row.get("repayment")})
    return items


def chat_items() -> list[dict]:
    """PR #1206 の103件を判定ログの subject_hash から復元する（本文は data/ 内のどこかに残っている）。"""
    if CHAT_ITEMS.exists():
        return json.loads(CHAT_ITEMS.read_text(encoding="utf-8"))
    judgments, labels = {}, {}
    for row in _jsonl(Path(jev_judgment_log.log_path() or DATA / "jev_judgment_log.jsonl")):
        if row.get("record_type") == "judgment" and row.get("guard") == "chat_teaching_capture" \
                and row.get("mode") == "offline_label_eval":
            judgments[row["judgment_id"]] = row
        elif row.get("record_type") == "label":
            labels[row["judgment_id"]] = row
    wanted = {j["subject_hash"]: (bool(labels[k]["label"]), j["probability"]) for k, j in judgments.items() if k in labels}
    found: dict[str, str] = {}

    def strings(value: Any):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for v in value.values():
                yield from strings(v)
        elif isinstance(value, list):
            for v in value:
                yield from strings(v)

    for path in glob.glob(str(DATA / "*.json")) + glob.glob(str(DATA / "*.jsonl")):
        if "jev_judgment_log" in path or "clef_vs_jev" in path or os.path.getsize(path) > 60_000_000:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8")
            docs = [json.loads(line) for line in text.splitlines() if line.strip()] if path.endswith("l") else [json.loads(text)]
        except (ValueError, OSError):
            continue
        for doc in docs:
            for s in strings(doc):
                if 5 < len(s) < 3000:
                    for candidate in (s, mask_for_jev(s)):
                        digest = jev_judgment_log.subject_hash(candidate) if candidate else ""
                        if digest in wanted and digest not in found:
                            found[digest] = mask_for_jev(candidate) or candidate
    items = [{"id": f"chat{n}", "text": text, "label": int(wanted[h][0]), "subject": text, "baseline_jev": wanted[h][1]}
             for n, (h, text) in enumerate(sorted(found.items()))]
    CHAT_ITEMS.write_text(json.dumps(items, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return items


def asset_dup_items() -> list[dict]:
    pairs = json.loads((DATA / "judgment_asset_dedup_jev_eval_20261002.json").read_text(encoding="utf-8"))["pairs"]
    return [{"id": f"asset{n}", "a": p["a_text"], "b": p["b_text"], "label": int(p["label"]),
             "subject": f"{p['a_text']}\n{p['b_text']}", "baseline_jev": p.get("jev")} for n, p in enumerate(pairs)]


def memory_items() -> list[dict]:
    pairs = json.loads((DATA / "user_personal_memory_jev_eval_20261003.json").read_text(encoding="utf-8"))["pairs"]
    return [{"id": f"mem{n}", "a": p["a"], "b": p["b"], "label": int(p["kind"] != "none"),
             "subject": f"{p['a']}\n{p['b']}", "baseline_jev": p.get("jev_max")} for n, p in enumerate(pairs)]


def policy_items() -> list[dict]:
    items = json.loads((DATA / "policy_kind_jev_eval_20261004_v2.json").read_text(encoding="utf-8"))["items"]
    return [{"id": f"policy{i['n']}", "text": i["masked"], "label": int(i["label"]), "subject": i["masked"],
             "baseline_jev": i.get("jev")} for i in items if i.get("split") == "human"]


# ── 各評価の送信形（本番・既存評価と同じ組み立て） ──────────────────────


def _noul(body: dict, key: str) -> float:
    return transport._noul(body["answers"], key)


def news_batch(chunk: list[dict], request: Callable[[dict], dict]) -> list[float]:
    body = request(news.build_news_request([i["article"] for i in chunk], model="jev-latest"))
    return [j["repayment"] for j in news.parse_news_judgments(body, len(chunk))]


def chat_batch(chunk: list[dict], request: Callable[[dict], dict]) -> list[float]:
    assert len(chunk) == 1  # 本番は1件ずつ送る
    body = request({"state": {"items": [chunk[0]["text"]]}, "model": "jev-latest",
                    "questions": {"knowhow": CHAT_TEACHING_JEV_QUESTION}})
    return [_noul(body, "knowhow")]


def asset_batch(chunk: list[dict], request: Callable[[dict], dict]) -> list[float]:
    body = request(build_jev_pair_request([(i["a"], i["b"]) for i in chunk]))
    return parse_jev_answers(body, len(chunk))


def memory_batch(chunk: list[dict], request: Callable[[dict], dict]) -> list[float]:
    pairs = [(i["a"], i["b"]) for i in chunk]
    scores = []
    for question in MEMORY_QUESTIONS.values():
        scores.append(transport.parse_binary_pair_answers(request(transport.build_binary_pair_request(pairs, question)), len(chunk)))
    return [max(dup, sup) for dup, sup in zip(*scores)]


def policy_batch(chunk: list[dict], request: Callable[[dict], dict]) -> list[float]:
    questions = {
        f"item{n}_policy": {"type": "noul", "instructions": QUESTION_V2["instructions"].format(x=f"`items[{n}]`"),
                            "criteria": {"true": QUESTION_V2["true"], "false": QUESTION_V2["false"]}}
        for n in range(len(chunk))
    }
    body = request({"state": {"items": [i["text"] for i in chunk]}, "model": "jev-latest", "questions": questions})
    return [_noul(body, f"item{n}_policy") for n in range(len(chunk))]


TASKS: dict[str, dict[str, Any]] = {
    "news_repayment": {"load": news_items, "judge": news_batch, "batch": 10, "guard": "news", "question": "repayment"},
    "chat_teaching": {"load": chat_items, "judge": chat_batch, "batch": 1, "guard": "chat_teaching_capture",
                      "question": "chat_teaching_knowhow"},
    "judgment_asset_dup": {"load": asset_dup_items, "judge": asset_batch, "batch": 15, "guard": "judgment_asset_dedup",
                           "question": "same_asset"},
    "personal_memory": {"load": memory_items, "judge": memory_batch, "batch": 15, "guard": "user_personal_memory_dedup",
                        "question": "personal_memory_archive_either"},
    "policy_kind_human": {"load": policy_items, "judge": policy_batch, "batch": 15, "guard": "knowledge_kind_policy",
                          "question": "is_policy_v2"},
}


# ── 実行・集計 ───────────────────────────────────────────────────────


class Meter:
    """リクエスト単位の往復時間と usage を記録する送信関数のラッパー。"""

    def __init__(self, send: Callable[[dict], dict]):
        self.send, self.latencies, self.input_tokens, self.models = send, [], [], set()

    def __call__(self, payload: dict) -> dict:
        start = time.perf_counter()
        body = self.send(payload)
        self.latencies.append(time.perf_counter() - start)
        usage = body.get("usage") or {}
        tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        self.input_tokens.append(int(tokens) if tokens else None)
        if body.get("model"):
            self.models.add(str(body["model"]))
        return body


def run_task(name: str, items: list[dict], provider: str, send: Callable[[dict], dict]) -> dict:
    task = TASKS[name]
    meter = Meter(send)
    scores: list[float] = []
    errors = 0
    for start in range(0, len(items), task["batch"]):
        chunk = items[start : start + task["batch"]]
        try:
            scores += task["judge"](chunk, meter)
        except Exception as exc:  # noqa: BLE001 — 失敗はその塊を欠測にして続ける（件数を報告）
            errors += 1
            scores += [None] * len(chunk)
            print(f"[{name}/{provider}] batch {start} failed: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
    return {"scores": scores, "latencies": meter.latencies, "input_tokens": meter.input_tokens,
            "models": sorted(meter.models), "errors": errors, "items": len(items)}


def _ci(values: tuple[float, float]) -> list[float]:
    return [round(v, 3) for v in values]


def paired_auc_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 7) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(auc(y[idx], b[idx]) - auc(y[idx], a[idx]))
    return round(auc(y, b) - auc(y, a), 3), _ci(tuple(np.percentile(diffs, [2.5, 97.5])))


def summarize(name: str, items: list[dict], results: dict[str, dict]) -> dict:
    present = [p for p in PROVIDERS if p in results]
    keep = [k for k in range(len(items)) if all(results[p]["scores"][k] is not None for p in present)]
    y = np.array([items[k]["label"] for k in keep])
    out: dict[str, Any] = {"n": len(keep), "positives": int(y.sum()), "dropped": len(items) - len(keep)}
    for provider in present:
        r = results[provider]
        s = np.array([r["scores"][k] for k in keep], dtype=float)
        lat = np.array(r["latencies"]) if r["latencies"] else np.array([np.nan])
        tokens = [t for t in r["input_tokens"] if t]
        out[provider] = {
            "model": ",".join(r["models"]),
            "auc": round(auc(y, s), 3), "auc_ci95": _ci(bootstrap_ci(y, s, auc)),
            "brier_raw": round(brier(y, s), 3), "brier_raw_ci95": _ci(bootstrap_ci(y, s, brier)),
            "brier_loo_platt": round(brier(y, loo_platt(y, s)), 3),
            "latency_s": {"median": round(float(np.median(lat)), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                          "requests": len(r["latencies"])},
            "input_tokens_per_item": round(sum(tokens) / max(1, len(items)), 1) if tokens else None,
            "errors": r["errors"],
        }
        if provider != "jev" and "jev" in present:
            jev_scores = np.array([results["jev"]["scores"][k] for k in keep], dtype=float)
            diff, ci = paired_auc_diff(y, jev_scores, s)
            out[provider]["auc_minus_jev"] = diff
            out[provider]["auc_minus_jev_ci95"] = ci
    return out


def log_results(name: str, items: list[dict], provider: str, result: dict) -> int:
    task = TASKS[name]
    rows = [(item, score) for item, score in zip(items, result["scores"]) if score is not None]
    records = jev_judgment_log.build_records(
        guard=f"{task['guard']}:{provider}", run_id=jev_judgment_log.new_run_id(), mode="offline_model_compare",
        model=",".join(result["models"]) or provider, sample_rate=0.0,
        items=[{"subject": item["subject"], "question": task["question"], "probability": round(score, 4),
                "choice": score >= 0.5, "route": "offline_eval", "auto_passed": False, "thresholds": {}}
               for item, score in rows],
    )
    jev_judgment_log.append_records(records)
    for record, (item, _) in zip(records, rows):
        jev_judgment_log.append_label(record["judgment_id"], bool(item["label"]), label_source=f"eval_set:{name}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default="jev", help="今回呼ぶモデル（カンマ区切り）。他は前回の結果を使う")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    clef = None
    for provider in [p for p in args.providers.split(",") if p]:
        if provider == "jev":
            send = jev_request
        else:
            clef = clef or ClefClient.from_env()
            send = clef.sender(provider)
        for name in args.tasks.split(","):
            items = TASKS[name]["load"]()
            print(f"[{name}/{provider}] {len(items)} items", file=sys.stderr)
            result = run_task(name, items, provider, send)
            if not args.no_log:
                result["logged"] = log_results(name, items, provider, result)
            cache.setdefault(name, {})[provider] = result
            CACHE.write_text(json.dumps(cache, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {name: summarize(name, TASKS[name]["load"](), cache[name]) for name in args.tasks.split(",") if name in cache}
    (DATA / "clef_vs_jev_summary_20261004.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
