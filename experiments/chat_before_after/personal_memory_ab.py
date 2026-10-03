#!/usr/bin/env python3
"""個人記憶の整理（アーカイブ）の前後で、紫苑の通常チャットの答えを比べる（2026-10-03）。

同じ worktree・同じデータのコピーで、アーカイブ無し（前）→有り（後）の順に server.py を1つずつ立て、
compare.py の10問＋雑談3問を1回ずつ投げる。前後の会話履歴が混ざらないよう user_id を分ける。

    python personal_memory_ab.py --root <worktree> --vault <Vaultコピー> --archive <archive.json> --out <dir>

- <worktree>/data に user_personal_memory_archive.json が無い状態で始めること（後の段で --archive をコピーする）
- 採点は compare.check の機械判定＋雑談の機械判定のみ（LLM の自己採点は使わない）
- 個人記憶ブロックが予算内で実際に何字入ったかは、worktree の data/chat_prompt_budget_log.jsonl から読む
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import QUESTIONS, check, gist, refs_of, reply_text  # noqa: E402
from run_both import _wait  # noqa: E402

PORT = 8103
SMALL_TALK = [
    {"id": "how_are_you", "kind": "talk", "q": "紫苑、最近どう？"},
    {"id": "dog_name", "kind": "talk", "q": "僕の犬の名前、覚えてる？", "must_any": ["タム"]},
    {"id": "tired", "kind": "talk", "q": "今日はちょっと疲れたよ"},
]
FORBIDDEN_OPENERS = ("もちろんです", "はい", "そうですね", "なるほど", "一般的には", "前回は", "以前は", "この前の続きで")
MECHANISM_RE = re.compile(r"個人記憶|user_personal_memory|ファイルに|記憶ファイル|データベースに")


def talk_check(question: dict, text: str) -> dict:
    head = text.lstrip(" \n#*>「")
    result = {
        "opener_ok": not head.startswith(FORBIDDEN_OPENERS),
        "no_mechanism_talk": not MECHANISM_RE.search(text),
        "chars": len(text),
    }
    if question.get("must_any"):
        result["recalls"] = any(k in text for k in question["must_any"])
    return result


def ask(question: str, user_id: str) -> dict:
    body = json.dumps({"message": question, "user_id": user_id, "response_mode": "shion"}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as res:
        return json.loads(res.read().decode("utf-8"))


def run_phase(side: str, args: argparse.Namespace) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLACK", "K_SERVICE"))}
    env.update({"OBSIDIAN_VAULT_PATH": str(args.vault), "OBSIDIAN_VAULT": str(args.vault),
                "GCS_VAULT_LOCAL_DIR": str(args.out / f"no_gcs_vault_{side}"), "PYTHONPATH": str(args.root), "PYTHONUNBUFFERED": "1"})
    log = (args.out / f"server_{side}.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(HERE / "server.py"), "--port", str(PORT), "--counter", str(args.out / f"gemini_calls_{side}.txt")],
                            cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
    budget_log = args.root / "data" / "chat_prompt_budget_log.jsonl"
    rows = []
    try:
        _wait(PORT)
        for question in QUESTIONS + SMALL_TALK:
            start = budget_log.stat().st_size if budget_log.exists() else 0
            try:
                data = ask(question["q"], f"pm_ab_{side}")
                text = reply_text(data)
                row = {"id": question["id"], "kind": question["kind"], "q": question["q"], "reply": text, "refs": refs_of(data)[:6],
                       "check": talk_check(question, text) if question["kind"] == "talk" else check(question, data)}
            except Exception as exc:  # noqa: BLE001
                row = {"id": question["id"], "kind": question["kind"], "q": question["q"], "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            personal = []
            if budget_log.exists():
                with budget_log.open(encoding="utf-8") as fh:
                    fh.seek(start)
                    for line in fh:
                        block = (json.loads(line).get("blocks") or {}).get("user_personal_memory_context")
                        if block:
                            personal.append({"orig": block["orig"], "kept": block["kept"]})
            row["personal_block"] = personal[-1] if personal else None
            print(f"[{side}] {question['id']} {row.get('check') or row.get('error')} personal={row['personal_block']}", flush=True)
            rows.append(row)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("root", "vault", "archive", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.root / "data" / "user_personal_memory_archive.json"
    if target.exists():
        raise SystemExit("前の段はアーカイブ無しで始める必要があります")
    before = run_phase("before", args)
    shutil.copy2(args.archive, target)
    after = run_phase("after", args)
    (args.out / "results.json").write_text(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["| 質問 | 前 | 後 | 個人記憶ブロック 前→後（入った字数／元の字数） |", "|---|---|---|---|"]
    for b, a in zip(before, after):
        def mark(r: dict) -> str:
            c = r.get("check") or {}
            return "エラー" if "error" in r else " ".join(f"{k}={'○' if v else '×'}" for k, v in c.items() if isinstance(v, bool))
        pb, pa = b.get("personal_block") or {}, a.get("personal_block") or {}
        lines.append(f"| {b['q'][:24]} | {mark(b)} | {mark(a)} | {pb.get('kept')}/{pb.get('orig')} → {pa.get('kept')}/{pa.get('orig')} |")
    lines.append("")
    for b, a in zip(before, after):
        lines += [f"### {b['q']}", f"- 前: {gist(b.get('reply', ''))}", f"- 後: {gist(a.get('reply', ''))}", ""]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
