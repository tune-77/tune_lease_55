#!/usr/bin/env python3
"""紫苑本体（LLM）によるトリアージ上書き提案（P1-2 後半の実装）。

ルール分類（rule_classify_item）と LLM の判断が食い違う候補についてだけ、
classified_by=llm の「提案」記録を data/shion_improvement_triage.jsonl へ追記する。

原則:
  - User が確定済み（classified_by=user）の候補には一切提案しない
  - 提案は記録のみ。キュー除外・優先・自動承認抑制の実効判断は User 確定に限る
    （shion_triage.is_user_confirmed 参照）
  - 同じ提案の再追記はしない（冪等）
  - LLM が使えない環境では警告して正常終了する（夜間パイプラインを止めない）

使い方:
  python scripts/shion_llm_triage_proposal.py --dry-run
  python scripts/shion_llm_triage_proposal.py --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Callable

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from shion_triage import (  # noqa: E402
    TRIAGE_FILE_RELPATH,
    is_user_confirmed,
    load_triage_latest,
    rule_classify_item,
    triage_record_for_item,
)

VALID_DECISIONS = {"today", "later", "discard"}
MAX_CANDIDATES = 12


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_candidates(root: Path) -> list[dict]:
    path = root / "reports" / "latest.json"
    if not path.exists():
        return []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in report.get("needs_review") or [] if isinstance(item, dict)]


def _get_gemini_api_key(root: Path) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    secrets_path = root / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                if "GEMINI_API_KEY" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def build_prompt(rows: list[dict]) -> str:
    lines = [
        "あなたはリース審査システムの改善PM「紫苑」です。",
        "以下の改善候補を today（今日やる・小さく安全）/ later（後回し・副作用が大きい）/ discard（捨てる・重複や価値なし）に分類してください。",
        "DB・API・スコアリング・認証・デプロイ・モデルに触るものは原則 later。表示文言・導線・説明の小修正は today。",
        "出力は JSON のみ: {\"REV-XXX\": {\"decision\": \"today|later|discard\", \"reason\": \"1行\"}, ...}",
        "",
    ]
    for row in rows:
        lines.append(
            f"- {row['item_id']}: {row['title']} / 理由: {row['reason'][:80]} / ルール分類: {row['rule']}"
        )
    return "\n".join(lines)


def call_gemini(prompt: str, api_key: str) -> str:
    import google.generativeai as genai  # type: ignore[import-untyped]

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text or ""


def parse_llm_output(text: str) -> dict[str, dict]:
    """LLM 出力から {item_id: {decision, reason}} を防御的に取り出す。"""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    result: dict[str, dict] = {}
    if not isinstance(payload, dict):
        return {}
    for item_id, value in payload.items():
        if isinstance(value, str):
            value = {"decision": value}
        if not isinstance(value, dict):
            continue
        decision = str(value.get("decision") or "").strip().lower()
        if decision in VALID_DECISIONS:
            result[str(item_id)] = {
                "decision": decision,
                "reason": str(value.get("reason") or "").strip()[:120],
            }
    return result


def build_proposals(
    candidates: list[dict],
    triage_latest: dict[str, dict],
    llm_fn: Callable[[str], str],
) -> list[dict]:
    """LLM がルールと異なる判断をした候補についてのみ提案記録を作る。"""
    rows: list[dict] = []
    for item in candidates[:MAX_CANDIDATES]:
        record = triage_record_for_item(triage_latest, item)
        if is_user_confirmed(record):
            continue  # User 確定済みには提案しない
        item_id = str(item.get("id") or "").strip()
        canonical_key = str(item.get("canonical_key") or "").strip() or item_id
        if not canonical_key:
            continue
        rows.append(
            {
                "item_id": item_id,
                "canonical_key": canonical_key,
                "title": str(item.get("title") or "")[:120],
                "reason": str(item.get("reason") or ""),
                "rule": rule_classify_item(item),
                "existing": record,
            }
        )
    if not rows:
        return []

    decisions = parse_llm_output(llm_fn(build_prompt(rows)))
    now = dt.datetime.now().isoformat(timespec="seconds")
    proposals: list[dict] = []
    for row in rows:
        verdict = decisions.get(row["item_id"])
        if not verdict:
            continue
        if verdict["decision"] == row["rule"]:
            continue  # ルールと同じなら上書き提案は不要
        existing = row["existing"]
        if (
            existing
            and str(existing.get("classified_by") or "") == "llm"
            and str(existing.get("decision") or "") == verdict["decision"]
        ):
            continue  # 同じ提案の再追記はしない（冪等）
        proposals.append(
            {
                "canonical_key": row["canonical_key"],
                "item_id": row["item_id"],
                "title": row["title"],
                "decision": verdict["decision"],
                "rule_decision": row["rule"],
                "classified_by": "llm",
                "reason": verdict["reason"],
                "decided_at": now,
            }
        )
    return proposals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    candidates = load_candidates(root)
    if not candidates:
        print("[llm_triage] 改善候補がありません（スキップ）")
        return 0

    api_key = _get_gemini_api_key(root)
    if not api_key:
        print("[llm_triage] GEMINI_API_KEY 未設定のためLLM提案をスキップします")
        return 0
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        print("[llm_triage] google-generativeai 未インストールのためスキップします")
        return 0

    triage_latest = load_triage_latest(root)
    try:
        proposals = build_proposals(candidates, triage_latest, lambda p: call_gemini(p, api_key))
    except Exception as exc:
        print(f"[llm_triage] LLM呼び出しに失敗しました（提案なしで継続）: {exc}")
        return 0

    for proposal in proposals:
        print(
            f"[llm_triage] 提案: {proposal['item_id'] or proposal['canonical_key']} "
            f"ルール={proposal['rule_decision']} → 紫苑={proposal['decision']} ({proposal['reason'][:60]})"
        )
    if not proposals:
        print("[llm_triage] ルール分類との差分はありません")
        return 0
    if args.apply:
        path = root / TRIAGE_FILE_RELPATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for proposal in proposals:
                f.write(json.dumps(proposal, ensure_ascii=False) + "\n")
        print(f"[llm_triage] {len(proposals)} 件の提案を記録しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
