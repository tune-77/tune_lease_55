"""Ollama adapter for Shion identity experiments.

Runs the same Shion identity (loaded from mind.json) through a local Llama model
and returns a structured reasoning_path comparable to the Gemini version.
"""
from __future__ import annotations

import json
import os
import re
import requests
from pathlib import Path
from typing import Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("SHION_OLLAMA_MODEL", "qwen2.5:latest")
DEFAULT_TIMEOUT = 420

# Appended to every experiment prompt to elicit structured reasoning_path output.
REASONING_PATH_INSTRUCTION = """
回答の末尾に以下のJSONを必ず出力してください（コードフェンス不要・改行可）：
{"kept":["維持した根拠"],"dropped":[{"item":"棄却した根拠","reason":"棄却理由"}],"pivots":["転換点の説明"],"value_weights":{"価値軸":"重みの説明"}}
項目がない場合は空リスト・空オブジェクトにしてください。
"""


def load_mind_summary(mind_path: Path, max_memories: int = 5) -> str:
    """Condense mind.json into a prompt-safe identity summary."""
    try:
        data = json.loads(mind_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"（記憶ファイル読み込み失敗: {exc}）"

    identity = data.get("identity", {})
    from lease_intelligence_mind import _get_value_labels
    values = _get_value_labels(identity)
    self_concept = identity.get("self_concept", "")
    current_q = data.get("current_question", "")
    memories = data.get("memories", [])[-max_memories:]

    mem_lines: list[str] = []
    for m in memories:
        if isinstance(m, dict):
            mem_lines.append(f"- {m.get('content', str(m))}")
        else:
            mem_lines.append(f"- {m}")
    mem_text = "\n".join(mem_lines) or "（記憶なし）"

    return (
        f"名前: {identity.get('name', '紫苑')}\n"
        f"自己概念: {self_concept}\n"
        f"価値観: {', '.join(values)}\n"
        f"現在の問い: {current_q}\n"
        f"最近の記憶:\n{mem_text}"
    )


def build_shion_prompt(mind_summary: str, question: str) -> str:
    """Build the standardised Shion reasoning prompt used by both models."""
    return (
        "あなたは紫苑（リース知性体）です。以下の記憶・価値観・自己モデルを持ちます。\n\n"
        "【紫苑の自己モデル】\n"
        f"{mind_summary}\n\n"
        "【問い】\n"
        f"{question}\n\n"
        "次の手順で答えてください：\n"
        "1. まず自分の初期仮説を作る\n"
        "2. 何を根拠として選び、何を棄却したかを明示する\n"
        "3. 最終回答を述べる\n"
        f"\n{REASONING_PATH_INSTRUCTION}"
    )


def parse_reasoning_path(text: str) -> dict[str, Any]:
    """Extract the structured reasoning_path JSON from model output."""
    # Walk backwards through '{' positions to find the last JSON with "kept"
    for match in reversed(list(re.finditer(r'\{', text))):
        start = match.start()
        depth = 0
        for i, ch in enumerate(text[start:]):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start : start + i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if "kept" in parsed:
                            parsed["parse_ok"] = True
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    break
    return {"parse_ok": False, "raw_tail": text[-400:]}


def strip_reasoning_json(text: str, path: dict[str, Any]) -> str:
    """Remove the raw reasoning JSON from the answer text."""
    if not path.get("parse_ok"):
        return text.strip()
    # Remove anything that looks like the JSON block from the end
    cleaned = re.sub(r'\{[^{}]*"kept"[^{}]*\}', "", text, flags=re.DOTALL)
    return cleaned.strip()


def run_ollama_shion(
    question: str,
    mind_path: Path,
    model: str = DEFAULT_MODEL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Run Shion identity through Ollama and return answer + reasoning_path."""
    mind_summary = load_mind_summary(mind_path)
    prompt = build_shion_prompt(mind_summary, question)

    url = OLLAMA_HOST.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return {"error": f"Ollama timeout ({timeout}s)", "model": model}
    except Exception as exc:
        return {"error": str(exc), "model": model}

    content = resp.json().get("message", {}).get("content", "")
    reasoning_path = parse_reasoning_path(content)
    answer = strip_reasoning_json(content, reasoning_path)

    return {
        "model": model,
        "provider": "ollama",
        "answer": answer,
        "reasoning_path": reasoning_path,
    }


__all__ = [
    "load_mind_summary",
    "build_shion_prompt",
    "parse_reasoning_path",
    "run_ollama_shion",
    "REASONING_PATH_INSTRUCTION",
]
