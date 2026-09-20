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

TypeSafe/Jev:
  TYPESAFE_TRIAGE_MODE=enforce でJevを優先し、障害時だけGeminiへ戻す。
  shadow は比較のみ、off（既定）は従来どおりGeminiを使う。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

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
TYPESAFE_TRIAGE_MODES = {"off", "shadow", "enforce"}
# This only creates non-binding proposals; user-confirmed decisions remain authoritative.
# Jev confidence measures distribution concentration, so 0.55 is intentionally more
# permissive than the threshold used for operational chat routing.
DEFAULT_TYPESAFE_CONFIDENCE = 0.55


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


def typesafe_triage_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the explicit rollout mode; unset or invalid configuration stays off."""
    env = os.environ if environ is None else environ
    mode = str(env.get("TYPESAFE_TRIAGE_MODE") or "off").strip().lower()
    return mode if mode in TYPESAFE_TRIAGE_MODES else "off"


def _typesafe_confidence_threshold(environ: Mapping[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    try:
        value = float(env.get("TYPESAFE_TRIAGE_CONFIDENCE", DEFAULT_TYPESAFE_CONFIDENCE))
    except (TypeError, ValueError):
        return DEFAULT_TYPESAFE_CONFIDENCE
    if not 0.0 <= value <= 1.0:
        return DEFAULT_TYPESAFE_CONFIDENCE
    return value


def build_typesafe_request(rows: list[dict], *, model: str | None = None) -> dict[str, Any]:
    """Build one batched Choice request; code keeps all execution policy."""
    questions: dict[str, dict[str, Any]] = {}
    public_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        public_rows.append(
            {
                "item_id": str(row["item_id"])[:80],
                "title": str(row["title"])[:120],
                "reason": str(row["reason"])[:400],
                "rule_decision": str(row["rule"]),
            }
        )
        questions[f"item_{index}"] = {
            "type": "choice",
            "instructions": (
                f"Classify improvement candidate `candidates[{index}]` by the safest next action. "
                "Choose one outcome even when it differs from `rule_decision`."
            ),
            "criteria": {
                "today": (
                    "A small, reversible, low-risk improvement suitable for today, such as wording, "
                    "guidance, or a narrow UI correction with limited side effects."
                ),
                "later": (
                    "Needs broader review or has meaningful side effects; includes database, API, "
                    "scoring, authentication, deployment, or model behavior changes."
                ),
                "discard": (
                    "Already applied, duplicate, obsolete, unsupported by the candidate evidence, "
                    "or too low-value to keep."
                ),
            },
        }
    return {
        "state": {"candidates": public_rows},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": questions,
    }


def parse_typesafe_output(
    body: Mapping[str, Any],
    rows: list[dict],
    *,
    confidence_threshold: float | None = None,
) -> dict[str, dict]:
    """Convert typed Choice answers into the existing decision contract."""
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError("TypeSafe triage response is missing answers")
    threshold = _typesafe_confidence_threshold() if confidence_threshold is None else confidence_threshold
    decisions: dict[str, dict] = {}
    reason_by_decision = {
        "today": "小さく安全で、今日対応できる改善と判定",
        "later": "影響範囲または副作用の確認が必要と判定",
        "discard": "重複・陳腐化・価値不足の可能性が高いと判定",
    }
    for index, row in enumerate(rows):
        answer = answers.get(f"item_{index}")
        if not isinstance(answer, Mapping):
            continue
        decision = str(answer.get("choice") or "").strip().lower()
        try:
            confidence = float(answer.get("confidence"))
        except (TypeError, ValueError):
            continue
        if decision not in VALID_DECISIONS or not 0.0 <= confidence <= 1.0:
            continue
        if confidence < threshold:
            continue
        decisions[str(row["item_id"])] = {
            "decision": decision,
            "reason": reason_by_decision[decision],
            "confidence": confidence,
        }
    return decisions


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


def _candidate_rows(candidates: list[dict], triage_latest: dict[str, dict]) -> list[dict]:
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
    return rows


def _proposals_from_decisions(
    rows: list[dict],
    decisions: Mapping[str, Mapping[str, Any]],
    *,
    provider: str,
) -> list[dict]:
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
        proposal = {
            "canonical_key": row["canonical_key"],
            "item_id": row["item_id"],
            "title": row["title"],
            "decision": verdict["decision"],
            "rule_decision": row["rule"],
            "classified_by": "llm",
            "reason": verdict["reason"],
            "decided_at": now,
            "model_provider": provider,
        }
        if "confidence" in verdict:
            proposal["confidence"] = round(float(verdict["confidence"]), 4)
        proposals.append(proposal)
    return proposals


def build_proposals(
    candidates: list[dict],
    triage_latest: dict[str, dict],
    llm_fn: Callable[[str], str],
) -> list[dict]:
    """LLM がルールと異なる判断をした候補についてのみ提案記録を作る。"""
    rows = _candidate_rows(candidates, triage_latest)
    if not rows:
        return []
    decisions = parse_llm_output(llm_fn(build_prompt(rows)))
    return _proposals_from_decisions(rows, decisions, provider="gemini")


def build_typesafe_proposals(
    candidates: list[dict],
    triage_latest: dict[str, dict],
    *,
    request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Create proposals from one batched Jev request."""
    import typesafe_dedup_guard as privacy

    all_rows = _candidate_rows(candidates, triage_latest)
    rows = [row for row in all_rows if privacy.is_safe_public_candidate(row)]
    excluded_count = len(all_rows) - len(rows)
    if not rows:
        return [], {
            "status": "skipped",
            "candidate_count": 0,
            "excluded_count": excluded_count,
        }
    if request_fn is None:
        from typesafe_rag_guard import request_system_one

        request_fn = request_system_one
    payload = build_typesafe_request(rows)
    body = request_fn(payload)
    decisions = parse_typesafe_output(body, rows)
    proposals = _proposals_from_decisions(rows, decisions, provider="typesafe")
    return proposals, {
        "status": "applied",
        "model": str(body.get("model") or payload["model"]),
        "candidate_count": len(rows),
        "excluded_count": excluded_count,
        "accepted_count": len(decisions),
        "usage": dict(body.get("usage") or {}),
    }


def _run_typesafe_shadow_comparison(
    candidates: list[dict],
    triage_latest: dict[str, dict],
    gemini_proposals: list[dict],
) -> None:
    """Log a best-effort comparison without invalidating Gemini's result."""
    try:
        typesafe_proposals, meta = build_typesafe_proposals(candidates, triage_latest)
    except Exception as exc:
        print(f"[llm_triage] shadow=skipped error_type={type(exc).__name__}")
        return
    typesafe_map = {p["item_id"]: p["decision"] for p in typesafe_proposals}
    gemini_map = {p["item_id"]: p["decision"] for p in gemini_proposals}
    print(
        "[llm_triage] shadow="
        + json.dumps(
            {
                "typesafe": typesafe_map,
                "gemini": gemini_map,
                "agreement": typesafe_map == gemini_map,
                "usage": meta.get("usage", {}),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _gemini_proposals(root: Path, candidates: list[dict], triage_latest: dict[str, dict]) -> list[dict]:
    api_key = _get_gemini_api_key(root)
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    try:
        import google.generativeai  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("google-generativeai is not installed") from exc
    return build_proposals(candidates, triage_latest, lambda p: call_gemini(p, api_key))


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

    triage_latest = load_triage_latest(root)
    mode = typesafe_triage_mode()
    proposals: list[dict]
    try:
        if mode == "off":
            proposals = _gemini_proposals(root, candidates, triage_latest)
            print("[llm_triage] provider=gemini mode=off")
        elif mode == "shadow":
            proposals = _gemini_proposals(root, candidates, triage_latest)
            os.environ.setdefault("TYPESAFE_API_KEYCHAIN_SERVICE", "typesafe-api-key")
            from typesafe_rag_guard import typesafe_available

            if not typesafe_available():
                print("[llm_triage] shadow=skipped reason=typesafe_unavailable")
            else:
                _run_typesafe_shadow_comparison(candidates, triage_latest, proposals)
        else:
            os.environ.setdefault("TYPESAFE_API_KEYCHAIN_SERVICE", "typesafe-api-key")
            from typesafe_rag_guard import typesafe_available

            if not typesafe_available():
                raise RuntimeError("TypeSafe credential is not configured")
            proposals, meta = build_typesafe_proposals(candidates, triage_latest)
            print(
                "[llm_triage] provider=typesafe "
                f"mode={mode} candidates={meta['candidate_count']} accepted={meta['accepted_count']} "
                f"usage={json.dumps(meta['usage'], ensure_ascii=False, separators=(',', ':'))}"
            )
    except Exception as exc:
        if mode == "enforce":
            print(f"[llm_triage] TypeSafe失敗のためGeminiへフォールバック: {type(exc).__name__}")
            try:
                proposals = _gemini_proposals(root, candidates, triage_latest)
            except Exception as fallback_exc:
                print(f"[llm_triage] LLM呼び出しに失敗しました（提案なしで継続）: {type(fallback_exc).__name__}")
                return 0
        else:
            print(f"[llm_triage] LLM呼び出しに失敗しました（提案なしで継続）: {type(exc).__name__}")
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
