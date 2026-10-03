#!/usr/bin/env python3
"""紫苑の個人記憶の「重複」「上書き（古くなった）」判定で、Jev が埋め込み・文字一致より効くかを測る（2026-10-03）。

正解: 2026-10-03 の個人記憶整理で Claude が内容を読んで決めた判断（POSITIVES）。
  - dup: 同じ趣旨で、片方を残せばもう片方は不要（統合したペア）
  - sup: B が新しい情報で A を上書きしており、A は古くなった
  - 負例: 全候補行のうち埋め込み上位30＋文字bigram上位20（片方の指標だけに有利な選び方を避ける）で、上のどちらでもないペア
正解が Claude 由来なので、Claude と似た判断をする指標が甘く出る可能性がある（報告に明記する）。

送信前のマスキング: 犬の名前などの個人事実の値は伏せ字に置き換え、亡くなった家族に触れる行は送らない。
さらに typesafe_dedup_guard のプライバシー判定を通ったペアだけを送る。判定は REV-424 の判定ログに記録する。

使い方:
  TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key .venv/bin/python experiments/personal_memory_dedup_jev/measure.py
"""

from __future__ import annotations

import json
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "judgment_asset_dedup_jev"))

import typesafe_dedup_guard as transport  # noqa: E402
from api.user_personal_memory_archive import collect_candidates  # noqa: E402
from measure import auc, bootstrap_ci  # noqa: E402
from scripts.judgment_asset_dedup import bigram_jaccard, embedding_similarity  # noqa: E402

OUT = ROOT / "data" / "user_personal_memory_jev_eval_20261003.json"

# (A の部分文字列, B の部分文字列, 種類)。sup は A=古い・B=新しい
POSITIVES = [
    ("**Relationship UX**", "紫苑の設計として、記憶を入れるだけでは足りない", "dup"),
    ("**Shion Conscience Layer**", "紫苑の良心レイヤーは判断を甘くするためではなく", "dup"),
    ("Shion language discipline", "Shion internal-risk doctrine", "dup"),
    ("[confirmed] Dog name:", "User's dog name is", "dup"),
    ("**Shion Memory Taxonomy**", "分類と想起経路で扱う方針", "dup"),
    ("僕の犬の名前は何だっけ", "- 2026-07-04T05:48:59", "dup"),
    ("Post-hackathon backlog: add a game-theory", "Post-hackathon Shion backlog is saved", "dup"),
    ("単なる検出結果の列挙ではなく", "Private Reflectionの定義を再整理", "sup"),
    ("紫苑が読まれていない前提で好きに考える私室", "Private Reflectionの定義を再整理", "sup"),
    ("遊びはPrivate Reflectionの継続性に必要", "Private Reflectionの定義を再整理", "sup"),
    ("Shion-HyDE RAG はハッカソン後の改善テーマ", "Shion-HyDE RAG を今すぐ進める場合でも", "sup"),
]
NEVER_SEND_RE = re.compile(r"妹|亡くな")
QUESTIONS = {
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


def body(row: dict) -> str:
    text = re.sub(r"^\[\d{4}-\d{2}-\d{2}\]\s*", "", row["body"])
    return re.sub(r"\s*\(`memory/[^)]*`\)\s*$", "", text)


def mask(text: str, secrets: list[str]) -> str:
    for value in secrets:
        text = text.replace(value, "[個人名]")
    return text


def find(rows: list[dict], sub: str) -> int:
    hits = [i for i, r in enumerate(rows) if sub in r["text"]]
    assert len(hits) == 1, (sub, len(hits))
    return hits[0]


def main() -> None:
    rows = collect_candidates()
    texts = [body(r) for r in rows]
    secrets = sorted({m.group(1).strip() for r in rows for m in [re.search(r"(?:Dog name|Preferred name):\s*(\S+)", r["text"])] if m}, key=len, reverse=True)
    emb = embedding_similarity(texts)

    labels: dict[tuple[int, int], str] = {}
    for a, b, kind in POSITIVES:
        labels[(find(rows, a), find(rows, b))] = kind
    scored = [(i, j, float(emb[i][j]), bigram_jaccard(texts[i], texts[j])) for i, j in combinations(range(len(rows)), 2)]
    positive_set = {frozenset(k) for k in labels}
    negatives = [(i, j) for i, j, *_ in sorted(scored, key=lambda x: -x[2])[:30] + sorted(scored, key=lambda x: -x[3])[:20] if frozenset((i, j)) not in positive_set]
    for i, j in dict.fromkeys(negatives):
        # 上書きの向きは日付で決める（古い方を A）
        labels[(i, j) if (rows[i]["date"] or "") <= (rows[j]["date"] or "") else (j, i)] = "none"

    pairs = []
    for (i, j), kind in labels.items():
        if NEVER_SEND_RE.search(texts[i] + texts[j]):
            continue
        a, b = mask(texts[i], secrets), mask(texts[j], secrets)
        if not (transport.is_safe_public_candidate({"title": a}) and transport.is_safe_public_candidate({"title": b})):
            continue
        pairs.append({"a_id": rows[i]["id"], "b_id": rows[j]["id"], "a": a, "b": b, "kind": kind,
                      "embedding": float(emb[i][j]), "jaccard": bigram_jaccard(texts[i], texts[j])})
    print(f"labelled={len(labels)} sendable={len(pairs)} positives={sum(p['kind'] != 'none' for p in pairs)}", file=sys.stderr)

    model = ""
    for name, question in QUESTIONS.items():
        scores, model = transport.judge_binary_pairs([(p["a"], p["b"]) for p in pairs], question, batch=15)
        for p, s in zip(pairs, scores):
            p[f"jev_{name}"] = s
    for p in pairs:
        p["jev_max"] = max(p["jev_dup"], p["jev_sup"])

    summary = {"model": model, "n": len(pairs)}
    for target, positive in (("archive_either", {"dup", "sup"}), ("dup", {"dup"}), ("sup", {"sup"})):
        subset = [p for p in pairs if p["kind"] in positive or p["kind"] == "none"]
        y = np.array([int(p["kind"] in positive) for p in subset])
        row = {"positives": int(y.sum()), "negatives": int((1 - y).sum())}
        for metric in ("jev_max", "jev_dup", "jev_sup", "embedding", "jaccard"):
            s = np.array([p[metric] for p in subset], dtype=float)
            row[metric] = {"auc": round(auc(y, s), 3), "auc_ci95": [round(v, 3) for v in bootstrap_ci(y, s, auc)]}
        summary[target] = row

    try:
        import jev_judgment_log

        items = [{"subject": f"{p['a']}\n{p['b']}", "question": f"personal_memory_{q}", "probability": p[f"jev_{q}"],
                  "choice": p[f"jev_{q}"] >= 0.5, "route": "eval", "auto_passed": False, "thresholds": {}}
                 for p in pairs for q in QUESTIONS]
        jev_judgment_log.append_records(jev_judgment_log.build_records(
            guard="user_personal_memory_dedup", run_id=jev_judgment_log.new_run_id(), mode="eval", model=model or "jev-latest", items=items))
    except Exception as exc:  # noqa: BLE001
        print(f"judgment log skipped: {type(exc).__name__}", file=sys.stderr)

    OUT.write_text(json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
