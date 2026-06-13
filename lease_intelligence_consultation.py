"""Senior-reasoner consultation loop for Shion."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path


PROJECT_ROOT = Path(__file__).resolve().parent
CONSULTATION_LOG_PATH = get_data_path("shion_reasoning_consultations.jsonl")

_SENSITIVE_PATTERNS = (
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    (re.compile(r"(?<!\d)(?:0\d{1,4}[-ー－]?\d{1,4}[-ー－]?\d{3,4})(?!\d)"), "[PHONE]"),
    (re.compile(r"(?<!\d)\d{13}(?!\d)"), "[CORPORATE_NUMBER]"),
    (
        re.compile(
            r"(?:株式会社|有限会社|合同会社)[^\s、。]{1,30}|"
            r"[^\s、。]{1,30}(?:株式会社|有限会社|合同会社)"
        ),
        "[COMPANY]",
    ),
    (re.compile(r"(?<![\d.])\d{5,}(?:\.\d+)?(?![\d.])"), "[LARGE_NUMBER]"),
)


def sanitize_consultation_text(value: str, max_chars: int = 2400) -> str:
    text = str(value or "").strip()
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:max_chars]


def _consultation_prompt(
    *,
    question: str,
    hypothesis: str,
    confidence: float,
    evidence_summary: str,
) -> str:
    return f"""あなたは紫苑（リース知性体）の上位検討役です。
紫苑の代わりに人格や最終判断を乗っ取らず、初期仮説を批判的に検証してください。
このリポジトリは読取専用で確認し、ファイル変更・外部送信は行わないでください。

【論点】
{question}

【紫苑の初期仮説】
{hypothesis}

【紫苑の確信度】
{confidence:.0%}

【紫苑が確認した根拠】
{evidence_summary}

次の形式で簡潔に回答してください。
1. 妥当な点
2. 誤り・見落とし・未確認事項
3. コードまたは記録から確認できる根拠
4. 紫苑が自分の結論へ統合するための助言
5. 採用・棄却根拠（必ず回答の末尾に以下の形式で追加すること）

[EVIDENCE]
採用: 根拠の要点 // 採用した理由
棄却: 根拠の要点 // 棄却した理由
[/EVIDENCE]

根拠がない場合は「採用: なし // -」と記載。根拠がないことは断定せず、推測と事実を分けてください。"""


def _parse_reasoning_path_from_advice(advice: str) -> dict[str, Any]:
    """Extract structured evidence selection from the [EVIDENCE] block in advice."""
    m = re.search(r"\[EVIDENCE\](.*?)\[/EVIDENCE\]", advice, re.DOTALL)
    if not m:
        return {"parse_ok": False}
    block = m.group(1)
    adopted_raw = re.findall(r"採用[:：]\s*(.+?)\s*//\s*(.+)", block)
    rejected_raw = re.findall(r"棄却[:：]\s*(.+?)\s*//\s*(.+)", block)
    return {
        "parse_ok": True,
        "adopted": [
            {"evidence": e.strip(), "reason": r.strip()}
            for e, r in adopted_raw
            if e.strip() not in ("なし", "-", "")
        ],
        "rejected": [
            {"evidence": e.strip(), "reason": r.strip()}
            for e, r in rejected_raw
            if e.strip() not in ("なし", "-", "")
        ],
    }


def _append_reasoning_path_note(vault: Path, record: dict[str, Any]) -> None:
    """Append Shion's selection path to today's Learning note."""
    directory = (
        vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Learning"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dt.date.today().isoformat()}.md"
    now_str = record["created_at"][11:19]

    kept_lines = "\n".join(f"  - {k}" for k in record.get("kept", [])) or "  （なし）"
    dropped_lines = (
        "\n".join(
            f"  - {d.get('item', '')} ／ 棄却理由: {d.get('reason', '')}"
            for d in record.get("dropped", [])
        )
        or "  （なし）"
    )
    pivot_lines = "\n".join(f"  - {p}" for p in record.get("pivots", [])) or "  （なし）"
    weights_lines = (
        "\n".join(f"  - {k}: {v}" for k, v in record.get("value_weights", {}).items())
        or "  （なし）"
    )
    section = (
        f"\n### 紫苑の選択経路 / {now_str} / {record['consultation_id']}\n\n"
        f"**維持した根拠**\n{kept_lines}\n\n"
        f"**棄却した根拠と理由**\n{dropped_lines}\n\n"
        f"**転換点**\n{pivot_lines}\n\n"
        f"**価値の重み付け**\n{weights_lines}\n"
    )
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(section)


def save_shion_reasoning_path(
    consultation_id: str,
    kept: list[str] | None = None,
    dropped: list[dict] | None = None,
    pivots: list[str] | None = None,
    value_weights: dict[str, str] | None = None,
    vault: Path | None = None,
) -> dict[str, Any]:
    """Record Shion's own reasoning path after integrating senior advice."""
    if not consultation_id:
        return {"saved": False, "error": "consultation_id が必要です"}
    record: dict[str, Any] = {
        "type": "shion_reasoning_path",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "consultation_id": consultation_id,
        "kept": list(kept or []),
        "dropped": list(dropped or []),
        "pivots": list(pivots or []),
        "value_weights": dict(value_weights or {}),
    }
    _append_jsonl(record)
    if vault:
        try:
            _append_reasoning_path_note(Path(vault), record)
        except Exception as exc:
            record["note_error"] = str(exc)
    return {"saved": True, "consultation_id": consultation_id}


def _find_cli(name: str) -> str:
    direct = shutil.which(name)
    if direct:
        return direct
    home = Path.home()
    candidates = [
        home / ".local" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/opt/homebrew/bin") / name,
    ]
    if name == "codex":
        candidates.extend(
            sorted(
                (home / ".nvm" / "versions" / "node").glob("*/bin/codex"),
                reverse=True,
            )
        )
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(f"{name} CLI が見つかりません")


def _run_codex(prompt: str, timeout: int) -> str:
    executable = _find_cli("codex")
    result = subprocess.run(
        [
            executable,
            "exec",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "--skip-git-repo-check",
            "--color",
            "never",
            "-C",
            str(PROJECT_ROOT),
            "-",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
    )
    answer = result.stdout.strip()
    if result.returncode != 0 or not answer:
        detail = result.stderr.strip()[-600:] or "応答がありません"
        raise RuntimeError(f"Codex相談失敗: {detail}")
    return answer


def _run_claude(prompt: str, timeout: int) -> str:
    executable = _find_cli("claude")
    result = subprocess.run(
        [
            executable,
            "-p",
            prompt,
            "--output-format",
            "text",
            "--no-session-persistence",
            "--permission-mode",
            "plan",
            "--tools",
            "Read,Grep,Glob",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
    )
    answer = result.stdout.strip()
    if result.returncode != 0 or not answer:
        detail = result.stderr.strip()[-600:] or "応答がありません"
        raise RuntimeError(f"Claude相談失敗: {detail}")
    return answer


def _append_jsonl(record: dict[str, Any]) -> None:
    path = Path(CONSULTATION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_learning_note(vault: Path, record: dict[str, Any]) -> str:
    directory = (
        Path(vault)
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Learning"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{dt.date.today().isoformat()}.md"
    prefix = "" if path.exists() else (
        "---\n"
        f"date: {dt.date.today().isoformat()}\n"
        "type: lease_intelligence_reasoning_learning\n"
        "---\n\n"
        f"# 紫苑の相談学習 — {dt.date.today().isoformat()}\n"
    )
    section = (
        f"\n## {record['created_at'][11:19]} / {record['id']}\n\n"
        f"- 上位検討役: {record['provider']}\n"
        f"- 初期確信度: {record['confidence']:.0%}\n\n"
        f"### 論点\n{record['question']}\n\n"
        f"### 紫苑の初期仮説\n{record['hypothesis']}\n\n"
        f"### 上位検討から得た助言\n{record['advice']}\n"
    )
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(prefix + section)
    return str(path)


def consult_senior_reasoner(
    *,
    question: str,
    shion_hypothesis: str,
    confidence: float,
    evidence_summary: str,
    vault: Path,
) -> dict[str, Any]:
    """Consult Codex only after Shion has formed an initial hypothesis."""
    safe_question = sanitize_consultation_text(question, 1600)
    safe_hypothesis = sanitize_consultation_text(shion_hypothesis, 2000)
    safe_evidence = sanitize_consultation_text(evidence_summary, 2400)
    if not safe_question or not safe_hypothesis:
        return {
            "error": "相談前に、論点と紫苑自身の初期仮説が必要です。",
            "consulted": False,
        }

    try:
        confidence_value = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = 0.5
    try:
        timeout = int(os.environ.get("SHION_CONSULT_TIMEOUT", "150"))
    except ValueError:
        timeout = 150
    timeout = max(30, min(300, timeout))
    prompt = _consultation_prompt(
        question=safe_question,
        hypothesis=safe_hypothesis,
        confidence=confidence_value,
        evidence_summary=safe_evidence or "根拠はまだ限定的。",
    )

    preferred = os.environ.get("SHION_SENIOR_REASONER", "codex").strip().lower()
    providers = ["claude", "codex"] if preferred == "claude" else ["codex", "claude"]
    errors: list[str] = []
    provider = ""
    advice = ""
    for candidate in providers:
        try:
            advice = (
                _run_claude(prompt, timeout)
                if candidate == "claude"
                else _run_codex(prompt, timeout)
            )
            provider = candidate
            break
        except Exception as exc:
            errors.append(str(exc))

    if not advice:
        return {
            "consulted": False,
            "error": " / ".join(errors)[:1200],
            "fallback": (
                "上位検討を利用できませんでした。紫苑自身の根拠と不確実性を示して回答してください。"
            ),
        }

    record = {
        "id": f"SHION-LEARN-{uuid.uuid4().hex[:8].upper()}",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "question": safe_question,
        "hypothesis": safe_hypothesis,
        "confidence": confidence_value,
        "evidence_summary": safe_evidence,
        "advice": advice[:8000],
        "reasoning_path": _parse_reasoning_path_from_advice(advice),
        "status": "awaiting_shion_synthesis",
    }
    note_path = _append_learning_note(Path(vault), record)
    record["note_path"] = note_path
    _append_jsonl(record)
    return {
        "consulted": True,
        "consultation_id": record["id"],
        "provider": provider,
        "senior_advice": record["advice"],
        "learning_note": note_path,
        "instruction": (
            "助言を丸写しせず、初期仮説から何を維持・修正したかを明示して、"
            "紫苑自身の最終結論へ統合してください。"
        ),
    }


def finalize_consultation_learning(
    vault: Path,
    consultation_ids: list[str],
    shion_synthesis: str,
) -> None:
    """Record Shion's final synthesis after senior consultation."""
    if not consultation_ids:
        return
    synthesis = str(shion_synthesis or "").strip()[:5000]
    now = dt.datetime.now()
    directory = (
        Path(vault)
        / "Projects"
        / "tune_lease_55"
        / "Lease Intelligence"
        / "Learning"
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{now.date().isoformat()}.md"
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(
            f"\n### 紫苑による統合 / {now.strftime('%H:%M:%S')}\n"
            f"- 対象: {', '.join(consultation_ids)}\n\n"
            f"{synthesis}\n"
        )
    _append_jsonl(
        {
            "type": "shion_synthesis",
            "created_at": now.isoformat(timespec="seconds"),
            "consultation_ids": list(consultation_ids),
            "status": "integrated_by_shion",
            "synthesis": synthesis,
        }
    )

    from lease_intelligence_mind import register_reasoning_learning

    register_reasoning_learning(
        Path(vault),
        consultation_ids=consultation_ids,
        synthesis=synthesis,
        date_str=now.date().isoformat(),
    )


__all__ = [
    "consult_senior_reasoner",
    "finalize_consultation_learning",
    "sanitize_consultation_text",
    "save_shion_reasoning_path",
]
