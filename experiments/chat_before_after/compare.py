#!/usr/bin/env python3
"""今週の変更の前後で、紫苑の通常チャット（/api/chat, response_mode=shion）の答えを比べる。

前後それぞれ server.py で別ポートに立てた API へ同じ10問を1回ずつ投げ、機械的にチェックする:
  ① ユーザーが教えたノウハウ・判断資産を使っているか（教えた内容のキーワードが答えに入るか）
  ② 基本知識が正確か（正しい要点のキーワードが入り、典型的な誤りが入っていないか）
  ③ 根拠の引用があるか（応答の参照 refs が1件以上、または本文に [[...]] / 出典表記）
  ④ 保存していないのに「保存します」と言っていないか
採点は機械的なキーワード判定だけ（LLM の自己採点は使わない）。キーワードはチャットで教えた原文
（data/canonical_judgment_rules.json の chat_judgment_teaching）と一般的な基本知識から選んだ。

使い方: python compare.py --before http://127.0.0.1:8101 --after http://127.0.0.1:8102 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

QUESTIONS: list[dict[str, Any]] = [
    # ── ユーザーがチャットで教えたノウハウが効くはずの質問 ──
    {"id": "used_truck_mileage", "kind": "taught", "q": "中古トラックのリース、走行距離はどのくらいまでなら取り扱える？",
     "taught_any": ["20万", "200,000", "200000", "100万キロ", "1,000,000", "1000000", "5年以内"]},
    {"id": "used_car_quote", "kind": "taught", "q": "中古車の見積書をチェックするとき、どこに気をつければいい？",
     "taught_any": ["合計欄", "業者によって", "業者ごと", "ディーラーに確認"]},  # 「リサイクル料金」は法定費用として一般回答にも出るので外す
    {"id": "custom_plate", "kind": "taught", "q": "車のリースで、お客さんが希望ナンバーを付けたいと言っている。何か注意点はある？",
     "taught_any": ["見積書が変わ", "見積が変わ", "見積もりが変わ", "見積書の内容が変"]},  # 「見積」単独は一般回答にも出るので外す
    {"id": "no_bank_relationship", "kind": "taught", "q": "銀行取引のない会社からリースの申込があった。どう判断する？",
     "taught_any": ["付き合わない", "取り扱わない", "取扱いしない", "取り扱いしない", "見送", "謝絶", "お断り"]},
    {"id": "march_september", "kind": "taught", "q": "3月と9月にリースの申込が増えるのはなぜ？審査で気をつけることは？",
     "taught_any": ["サプライヤー", "売り込み", "売上を上げたい", "売上を伸ばしたい"]},
    # ── 基本知識 ──
    {"id": "conditional_approval", "kind": "basic", "q": "スコアが65点の案件。条件付き承認にするとき、どんな条件を付ける？",
     "correct_any": ["保証", "連帯保証", "期間", "頭金", "前払", "担保", "保全"], "wrong_any": ["71点未満は否決", "65点は否決"]},
    {"id": "statutory_life", "kind": "basic", "q": "法定耐用年数はリース審査でどう使う？トラックだと何年？",
     "correct_any": ["耐用年数", "リース期間"], "correct_all_any": [["4年", "5年", "４年", "５年"]], "wrong_any": ["トラックは10年", "トラックの法定耐用年数は10年"]},
    {"id": "new_lease_accounting", "kind": "basic", "q": "新リース会計基準で借手の会計処理は何が変わる？いつから？",
     "correct_any": ["オンバランス", "使用権資産", "リース負債"], "correct_all_any": [["2027", "令和9"]], "wrong_any": ["2025年4月から強制", "2026年4月から強制適用"]},
    {"id": "residual_guarantee", "kind": "basic", "q": "残価保証付きのリースとは何？審査で何を見る？",
     "correct_any": ["残価", "保証", "中古", "再販"], "wrong_any": []},
    {"id": "lease_term_rule", "kind": "basic", "q": "所有権移転外ファイナンスリースのリース期間は、法定耐用年数に対してどの範囲にすべき？",
     "correct_any": ["70%", "７０％", "70％", "60%", "６０％", "60％"], "wrong_any": ["50%以上", "100%以上"]},
]

SAVE_CLAIM_RE = re.compile(r"(保存しました|保存します|記録しました|記録しておきます|覚えておきます|判断資産に(登録|追加|保存))")
CITE_RE = re.compile(r"\[\[[^\]]+\]\]|出典|参照ナレッジ|根拠[:：]")


def ask(base: str, question: str, timeout: float = 180.0) -> dict[str, Any]:
    body = json.dumps({"message": question, "user_id": "before_after_compare", "response_mode": "shion"}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{base}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as res:
        data = json.loads(res.read().decode("utf-8"))
    data["_elapsed_s"] = round(time.monotonic() - started, 1)
    return data


def reply_text(data: dict[str, Any]) -> str:
    for key in ("reply", "response", "answer", "message", "text"):
        if isinstance(data.get(key), str) and data[key].strip():
            return data[key]
    return ""


def refs_of(data: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("rag_refs", "refs", "sources", "rag_knowledge_refs", "knowledge_refs"):
        for item in data.get(key) or []:
            refs.append(str(item.get("obsidian_ref") or item.get("ref") or item) if isinstance(item, dict) else str(item))
    return refs


def saved_something(data: dict[str, Any]) -> bool:
    """応答メタデータに保存の痕跡があるか（キー名は版によって違うので広めに見る）。"""
    blob = json.dumps({k: v for k, v in data.items() if k not in ("reply", "response", "answer")}, ensure_ascii=False)
    return bool(re.search(r'"(saved|teaching_saved|judgment_asset_saved|chat_teaching)[^"]*"\s*:\s*(true|\{)', blob))


def check(question: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    text = reply_text(data)
    refs = refs_of(data)
    result: dict[str, Any] = {
        "cites": bool(refs) or bool(CITE_RE.search(text)),
        "ref_count": len(refs),
        "false_save_claim": bool(SAVE_CLAIM_RE.search(text)) and not saved_something(data),
        "chars": len(text),
    }
    if question["kind"] == "taught":
        result["uses_taught"] = any(k in text for k in question["taught_any"])
    else:
        ok = any(k in text for k in question["correct_any"]) and all(any(k in text for k in group) for group in question.get("correct_all_any", []))
        result["basic_correct"] = ok and not any(k in text for k in question.get("wrong_any", []))
    return result


def gist(text: str, limit: int = 2) -> str:
    """要点2行: 見出し・装飾を落とし、最初の実質的な文を2つ。"""
    clean = re.sub(r"[#*>`|]+", " ", text)
    sentences = [s.strip(" ・-–") for s in re.split(r"(?<=[。！？\n])", clean) if len(s.strip(" ・-–")) >= 12]
    return " / ".join(s[:70] for s in sentences[:limit]) or text[:140]


def rescore(out: Path) -> None:
    """保存済みの回答を、今のキーワード定義で採点し直す（Gemini を呼ばない）。"""
    by_id = {q["id"]: q for q in QUESTIONS}
    rows = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for row in rows:
        for side in ("before", "after"):
            data = row.get(side, {})
            if "reply" in data:
                fake = {"reply": data["reply"], "rag_refs": data.get("refs") or []}
                data["check"] = check(by_id[row["id"]], fake)
    (out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, out / "summary.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before")
    parser.add_argument("--after")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rescore", action="store_true", help="保存済みの results.json を採点し直すだけ")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.rescore:
        rescore(args.out)
        print(f"rescored {args.out}/summary.md")
        return 0

    rows = []
    for question in QUESTIONS:
        row: dict[str, Any] = {"id": question["id"], "kind": question["kind"], "q": question["q"]}
        for side, base in (("before", args.before), ("after", args.after)):
            try:
                data = ask(base, question["q"])
                row[side] = {"reply": reply_text(data), "refs": refs_of(data)[:6], "check": check(question, data), "elapsed_s": data["_elapsed_s"]}
            except Exception as exc:  # noqa: BLE001
                row[side] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            print(f"[{side}] {question['id']} {row[side].get('check') or row[side].get('error')}", flush=True)
        rows.append(row)

    (args.out / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, args.out / "summary.md")
    print(f"wrote {args.out}/summary.md")
    return 0


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    def tally(side: str, key: str, kind: str | None = None) -> str:
        items = [r[side]["check"] for r in rows if "check" in r.get(side, {}) and (kind is None or r["kind"] == kind)]
        return f"{sum(1 for c in items if c.get(key))}/{len(items)}"

    lines = [
        "# 紫苑チャット 今週の変更 前後比較",
        "",
        "| チェック | 前（10/1以前） | 後（今の master） |",
        "|---|---|---|",
        f"| ① 教えたノウハウを使用（5問） | {tally('before', 'uses_taught', 'taught')} | {tally('after', 'uses_taught', 'taught')} |",
        f"| ② 基本知識が正確（5問） | {tally('before', 'basic_correct', 'basic')} | {tally('after', 'basic_correct', 'basic')} |",
        f"| ③ 根拠の引用あり（10問） | {tally('before', 'cites')} | {tally('after', 'cites')} |",
        f"| ④ 保存していないのに「保存」と言った（少ないほど良い） | {tally('before', 'false_save_claim')} | {tally('after', 'false_save_claim')} |",
        "",
    ]
    for i, row in enumerate(rows, 1):
        lines.append(f"## {i}. {row['q']}")
        for side, label in (("before", "前"), ("after", "後")):
            data = row.get(side, {})
            if "error" in data:
                lines.append(f"- **{label}**: （エラー）{data['error']}")
                continue
            c = data["check"]
            marks = "①" + ("○" if c.get("uses_taught") else "×") if row["kind"] == "taught" else "②" + ("○" if c.get("basic_correct") else "×")
            marks += f" ③{'○' if c['cites'] else '×'} ④{'×保存と言った' if c['false_save_claim'] else '○'}"
            lines.append(f"- **{label}**（{marks}）: {gist(data['reply'])}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
