"""autoresearchノートの「確認済み事実」を、調査原文と突き合わせるJevガード。

`auto_research_lease_judgment.research_topic` は接地検索の結果（raw_research）を
Geminiに再構成させてノート本文を作る。この合成ステップで、原文に無い数値や
基準が混ざりうる。ノートは判断資産候補へ流れるため、人間が読む前に
「原文で裏が取れない主張」を名指しできる必要がある。

Jevは1リクエストで主張ごとに独立したChoiceを返す。主張同士は互いの答えを
見ないので、1件の誤判定が他へ伝播しない。

環境変数:
  TYPESAFE_RESEARCH_VERIFY_MODE  off（既定） | shadow | enforce
  TYPESAFE_RESEARCH_VERIFY_MAX_CLAIMS  1リクエストの上限（既定 12）
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from typesafe_rag_guard import (  # noqa: F401  再利用: 転送層を複製しない
    TypeSafeRagError,
    request_system_one,
    typesafe_available,
)

VERIFY_MODES = {"off", "shadow", "enforce"}
DEFAULT_MODEL = "jev-latest"
DEFAULT_MAX_CLAIMS = 12
MAX_CLAIM_CHARS = 400
MAX_EVIDENCE_CHARS = 16000

VERDICTS = ("verified", "contradicted", "unsupported")

# 裏が取れない主張だけを人間に見せたい。verified を落としても損失は無いが、
# unsupported を見逃すと未検証の数値が判断資産へ流れる。
DEFAULT_CONFIDENCE_MIN = 0.60

_CLAIM_SECTION = "## 判断に使える確認済み事実"
_BULLET_RE = re.compile(r"^\s*[-*・]\s*(.+?)\s*$")


def verify_mode(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    mode = str(env.get("TYPESAFE_RESEARCH_VERIFY_MODE") or "off").strip().lower()
    return mode if mode in VERIFY_MODES else "off"


def _max_claims(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    try:
        value = int(env.get("TYPESAFE_RESEARCH_VERIFY_MAX_CLAIMS", DEFAULT_MAX_CLAIMS))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CLAIMS
    return value if 1 <= value <= 64 else DEFAULT_MAX_CLAIMS


def extract_claims(body: str, *, limit: int | None = None) -> list[str]:
    """「判断に使える確認済み事実」節の箇条書きだけを取り出す。

    他の節（推論・確認質問・反証）は原文と一致しないのが正常なので対象外。
    """
    cap = _max_claims() if limit is None else limit
    lines = (body or "").splitlines()
    claims: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            inside = stripped == _CLAIM_SECTION
            continue
        if not inside:
            continue
        match = _BULLET_RE.match(line)
        if match:
            text = match.group(1).strip()
            if len(text) >= 8:
                claims.append(text[:MAX_CLAIM_CHARS])
        if len(claims) >= cap:
            break
    return claims


def _claim_question(index: int) -> dict[str, dict[str, Any]]:
    return {
        f"c{index}_verdict": {
            "type": "choice",
            "instructions": (
                f"Judge `claims[{index}]` against `evidence` only. Do not use outside knowledge."
            ),
            "criteria": {
                "verified": "The evidence states this claim, including its figures and scope.",
                "contradicted": "The evidence states something incompatible with this claim.",
                "unsupported": (
                    "The evidence neither states nor contradicts it; the claim adds a figure, "
                    "threshold, date, or scope the evidence does not contain."
                ),
            },
        }
    }


def build_verify_request(
    claims: Sequence[str],
    evidence: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """1リクエストで全主張を独立判定する。"""
    items = list(claims)
    questions: dict[str, dict[str, Any]] = {}
    for index in range(len(items)):
        questions.update(_claim_question(index))
    return {
        "state": {
            "claims": [str(item)[:MAX_CLAIM_CHARS] for item in items],
            "evidence": str(evidence or "")[:MAX_EVIDENCE_CHARS],
        },
        "model": model or os.environ.get("TYPESAFE_MODEL", DEFAULT_MODEL),
        "questions": questions,
    }


def parse_verify_output(body: Mapping[str, Any], claims: Sequence[str]) -> list[dict[str, Any]]:
    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        raise TypeSafeRagError("TypeSafe verify response is missing answers")
    results: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        answer = answers.get(f"c{index}_verdict")
        if not isinstance(answer, Mapping):
            continue
        verdict = str(answer.get("choice") or "").strip().lower()
        try:
            confidence = float(answer.get("confidence"))
        except (TypeError, ValueError):
            continue
        if verdict not in VERDICTS or not 0.0 <= confidence <= 1.0:
            continue
        results.append({"claim": claim, "verdict": verdict, "confidence": confidence})
    return results


def verify_note_claims(
    body: str,
    evidence: str,
    *,
    request_fn: Any = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """ノート本文の確認済み事実を調査原文と突き合わせる。"""
    claims = extract_claims(body)
    if not claims:
        return {"status": "skipped", "reason": "no_claims", "results": []}
    if request_fn is None:
        request_fn = request_system_one
    payload = build_verify_request(claims, evidence)
    response = request_fn(payload)
    results = parse_verify_output(response, claims)
    flagged = [
        item
        for item in results
        if item["verdict"] != "verified" and item["confidence"] >= DEFAULT_CONFIDENCE_MIN
    ]
    return {
        "status": "applied",
        "model": str(response.get("model") or payload["model"]),
        "claim_count": len(claims),
        "judged_count": len(results),
        "flagged": flagged,
        "counts": {name: sum(1 for item in results if item["verdict"] == name) for name in VERDICTS},
        "usage": dict(response.get("usage") or {}),
    }


def render_verification_section(result: Mapping[str, Any]) -> str:
    """裏が取れなかった主張だけをノート末尾へ足す文面を作る。"""
    flagged = list(result.get("flagged") or [])
    if not flagged:
        return ""
    lines = [
        "",
        "## 出典突き合わせ（自動）",
        f"調査原文と照合し、{len(flagged)}件の主張で裏が取れませんでした。採用前に一次情報で確認してください。",
    ]
    for item in flagged:
        label = "原文と矛盾" if item["verdict"] == "contradicted" else "原文に記載なし"
        lines.append(f"- `{label}` {item['claim']}")
    return "\n".join(lines) + "\n"
