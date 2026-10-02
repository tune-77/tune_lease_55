#!/usr/bin/env python3
"""判断資産の同一性判定で Jev が埋め込み・文字一致より効くかを測る。

正解: 2026-10-02 の人手統合（17クラスタ・28件、data/judgment_asset_merge_report_20261002.json）。
  - 同じクラスタ内のペア = 1（統合してよい）
  - 統合前スナップショットで類似度上位だが統合しなかったペア = 0
負例は埋め込み上位と文字bigram上位の和集合から取り、片方の指標だけに有利な選び方を避ける。

統合判断の際に Jev へ相談した境界ペア（CONSULTED）はラベルが Jev の影響を受けているため、
それを除いた集計も出す。本文は外部へ送る前に typesafe_dedup_guard のプライバシー判定を通す。

使い方:
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/judgment_asset_dedup_jev/measure.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import typesafe_dedup_guard as transport  # noqa: E402
from scripts.judgment_asset_dedup import bigram_jaccard, build_jev_pair_request, parse_jev_answers  # noqa: E402

SNAPSHOT = ROOT / "data" / "backups" / "canonical_judgment_rules.before_merge_20261002_061217.json"
MERGE_REPORT = ROOT / "data" / "judgment_asset_merge_report_20261002.json"
OUT = ROOT / "data" / "judgment_asset_dedup_jev_eval_20261002.json"

# 統合判断時に Jev へ問い合わせたペア（id先頭8桁）。ラベル漏洩の感度分析用。
CONSULTED = {
    frozenset(p)
    for p in [
        ("e4af8f68", "c987146b"), ("c8c4cbd2", "c4d0f04c"), ("cbad4a88", "b0427f1b"),
        ("a2389326", "bc58c218"), ("e5474c44", "453ecc8e"), ("85ce2b1e", "2d510b6a"),
        ("93d1b22c", "b75dc949"), ("93d1b22c", "8bba0961"), ("93d1b22c", "7ed1152a"),
        ("8932df45", "10de7c81"), ("4b63961d", "7693d4ad"), ("cf61a970", "c29ed2f8"),
        ("cf61a970", "5e5c7561"), ("b76e9e0a", "3e58db70"), ("b76e9e0a", "87e11c62"),
        ("07d7c016", "150b4787"), ("07d7c016", "ffb56a11"),
    ]
}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, fn, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def loo_platt(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """1特徴のロジスティック回帰を leave-one-out で当て、較正後確率を返す。"""
    from sklearn.linear_model import LogisticRegression

    out = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        model = LogisticRegression().fit(s[mask].reshape(-1, 1), y[mask])
        out[i] = model.predict_proba([[s[i]]])[0, 1]
    return out


def build_pairs(n_negative: int) -> list[dict]:
    rules = {r["id"]: r for r in json.loads(SNAPSHOT.read_text(encoding="utf-8"))["rules"] if r["status"] == "active"}
    report = json.loads(MERGE_REPORT.read_text(encoding="utf-8"))
    clusters = [[c["representative_id"], *[m["id"] for m in c["merged"]]] for c in report["cluster_list"]]
    cluster_of = {rid: n for n, ids in enumerate(clusters) for rid in ids}
    positives = [(a, b) for ids in clusters for a, b in combinations(ids, 2)]

    from scripts.judgment_asset_dedup import embedding_similarity

    ids = list(rules)
    texts = [rules[i]["canonical_statement"] for i in ids]
    emb = embedding_similarity(texts)
    scored = []
    for i, j in combinations(range(len(ids)), 2):
        a, b = ids[i], ids[j]
        if cluster_of.get(a) is not None and cluster_of.get(a) == cluster_of.get(b):
            continue
        scored.append((a, b, float(emb[i][j]) if emb is not None else 0.0, bigram_jaccard(texts[i], texts[j])))
    by_emb = sorted(scored, key=lambda x: -x[2])[: n_negative // 2]
    by_jac = sorted(scored, key=lambda x: -x[3])[: n_negative // 2]
    negatives = list({(a, b): None for a, b, *_ in by_emb + by_jac})

    index = {rid: k for k, rid in enumerate(ids)}
    pairs = []
    for label, group in ((1, positives), (0, negatives)):
        for a, b in group:
            ta, tb = rules[a]["canonical_statement"], rules[b]["canonical_statement"]
            pairs.append(
                {
                    "a": a, "b": b, "label": label, "a_text": ta, "b_text": tb,
                    "embedding": float(emb[index[a]][index[b]]) if emb is not None else None,
                    "jaccard": bigram_jaccard(ta, tb),
                    "consulted": frozenset((a[:8], b[:8])) in CONSULTED,
                }
            )
    return pairs


def judge_with_jev(pairs: list[dict], batch: int) -> None:
    os.environ.setdefault("TYPESAFE_DEDUP_TIMEOUT_SECONDS", "90")
    for start in range(0, len(pairs), batch):
        chunk = pairs[start : start + batch]
        payload = build_jev_pair_request([(p["a_text"], p["b_text"]) for p in chunk])
        body = transport._default_request(payload)
        for p, prob in zip(chunk, parse_jev_answers(body, len(chunk))):
            p["jev"] = prob
        print(f"judged {start + len(chunk)}/{len(pairs)}", file=sys.stderr)


def summarize(pairs: list[dict]) -> dict:
    result = {}
    for name, subset in (("all", pairs), ("excluding_consulted", [p for p in pairs if not p["consulted"]])):
        y = np.array([p["label"] for p in subset])
        row = {"n": len(subset), "positives": int(y.sum()), "negatives": int((1 - y).sum())}
        for metric in ("jev", "embedding", "jaccard"):
            s = np.array([p[metric] for p in subset], dtype=float)
            calibrated = loo_platt(y, s)
            row[metric] = {
                "auc": round(auc(y, s), 3),
                "auc_ci95": [round(v, 3) for v in bootstrap_ci(y, s, auc)],
                "brier_raw": round(brier(y, s), 3) if metric == "jev" else None,
                "brier_loo_platt": round(brier(y, calibrated), 3),
            }
        result[name] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--negatives", type=int, default=60)
    parser.add_argument("--batch", type=int, default=15)
    args = parser.parse_args()

    pairs = build_pairs(args.negatives)
    safe = [p for p in pairs if transport.is_safe_public_candidate({"title": p["a_text"]}) and transport.is_safe_public_candidate({"title": p["b_text"]})]
    random.Random(11).shuffle(safe)
    print(f"pairs={len(pairs)} safe={len(safe)}", file=sys.stderr)
    judge_with_jev(safe, args.batch)
    summary = summarize(safe)
    OUT.write_text(json.dumps({"summary": summary, "pairs": safe}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
