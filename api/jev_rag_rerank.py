"""ローカル RAG（ChromaDB）の候補を Jev の「この質問に答えるのに役立つか」の確率で並べ直す。

- 候補ごとの独立した二値判定を1リクエストにまとめて送る（typesafe_rag_guard の資格情報安全なクライアント）。
- 送るクエリと候補の本文・題名は api.vertex_query_mask.mask_for_vertex を必ず通す。ファイルパスは送らない。
- 失敗（不通・不正応答）は例外。呼び出し側は ChromaDB 単独の順位に戻す。
- 判定は REV-424 の判定ログ（data/jev_judgment_log.jsonl、guard=rag_jev_rerank, mode=shadow）に残す。

本番適用は独立スイッチ（VERTEX_CREDIT_MODE とは無関係。Jev はクレジット対象外）:
- 毎晩の評価（scripts/eval_vertex_vs_chroma.py）が3晩連続で ChromaDB 単独より良いと確認した時だけ
  data/jev_rag_rerank_state.json の promoted が True になる
- 環境変数 JEV_RAG_RERANK=off なら promoted でも使わない
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from api.vertex_query_mask import mask_for_vertex

GUARD = "rag_jev_rerank"
QUESTION_ID = "useful"
JEV_USD_PER_1B_INPUT_TOKENS = 42.0  # typesafe.ai 掲載（2026-10-03）。出力トークンの単価は未掲載のため数えない
JPY_PER_USD = 160.0
_REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(os.environ.get("JEV_RAG_RERANK_STATE_PATH") or _REPO_ROOT / "data" / "jev_rag_rerank_state.json")
RequestFn = Callable[[dict[str, Any]], Mapping[str, Any]]


def _question(index: int) -> dict[str, Any]:
    return {
        "type": "noul",
        "instructions": f"`passages[{index}]` は、`query` の質問に答えるのに役立つか？",
        "criteria": {
            "true": "質問の主題を扱い、回答の根拠になる事実・判断基準・確認事項を含む。",
            "false": "主題が違う、言葉が重なるだけ、または回答の根拠になる内容を含まない。",
        },
    }


def build_request(query: str, hits: Sequence[Mapping[str, Any]], *, model: str = "jev-latest") -> dict[str, Any]:
    passages = [
        {
            "title": mask_for_vertex(str(hit.get("title") or hit.get("file_name") or ""))[:200],
            "text": mask_for_vertex(str(hit.get("snippet") or hit.get("text") or ""))[:1200],
        }
        for hit in hits
    ]
    return {
        "state": {"query": mask_for_vertex(query)[:1000], "passages": passages},
        "model": model,
        "questions": {f"p{i}_{QUESTION_ID}": _question(i) for i in range(len(passages))},
    }


def _default_request(payload: dict[str, Any]) -> Mapping[str, Any]:
    from typesafe_rag_guard import request_system_one

    return request_system_one(payload)


def jev_rerank(
    query: str,
    hits: Sequence[Mapping[str, Any]],
    *,
    request_fn: RequestFn | None = None,
    log: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """確率の高い順に並べ直したヒットと、{probabilities, input_tokens, calls} を返す。同点は元の順位。"""
    candidates = [dict(hit) for hit in hits]
    if not candidates:
        return [], {"probabilities": [], "input_tokens": 0, "calls": 0}
    payload = build_request(query, candidates)
    body = (request_fn or _default_request)(payload)
    answers = body.get("answers") if isinstance(body, Mapping) else None
    if not isinstance(answers, Mapping):
        raise ValueError("Jev 応答に answers が無い")
    probabilities: list[float] = []
    for index in range(len(candidates)):
        raw = answers.get(f"p{index}_{QUESTION_ID}")
        value = float(raw["noul"]) if isinstance(raw, Mapping) and raw.get("type") == "noul" else -1.0
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Jev 応答が不正: p{index}")
        probabilities.append(value)
    order = sorted(range(len(candidates)), key=lambda i: (-probabilities[i], i))
    usage = dict(body.get("usage") or {})
    meta = {
        "probabilities": probabilities,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "calls": 1,
        "model": str(body.get("model") or payload["model"]),
    }
    if log:
        _log_judgments(payload, probabilities, meta["model"])
    return [candidates[i] for i in order], meta


def _log_judgments(payload: dict[str, Any], probabilities: list[float], model: str) -> None:
    """REV-424 の判定ログへ。subject はハッシュ化にだけ使われ、本文はログに残らない。"""
    try:
        import jev_judgment_log

        query = payload["state"]["query"]
        items = [
            {
                "question": QUESTION_ID,
                "subject": f"{query}\n{passage['title']}\n{passage['text'][:200]}",
                "probability": probability,
                "route": "rank",
            }
            for passage, probability in zip(payload["state"]["passages"], probabilities)
        ]
        jev_judgment_log.append_records(
            jev_judgment_log.build_records(guard=GUARD, run_id=jev_judgment_log.new_run_id(), mode="shadow", model=model, items=items)
        )
    except Exception as exc:  # noqa: BLE001 - ログの失敗で並べ替えを止めない
        print(f"[rag_jev_rerank] 判定ログ記録スキップ: {type(exc).__name__}")


def estimate_jpy(input_tokens: int) -> float:
    return round(input_tokens * JEV_USD_PER_1B_INPUT_TOKENS / 1e9 * JPY_PER_USD, 4)


def load_state(path: Path | None = None) -> dict[str, Any]:
    try:
        data = json.loads((path or STATE_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def production_enabled(state_path: Path | None = None) -> bool:
    """本番のチャットで Jev 並べ替えを使うか。独立スイッチ JEV_RAG_RERANK=off で止まる。"""
    if (os.environ.get("JEV_RAG_RERANK") or "").strip().lower() in {"off", "0", "false", "no"}:
        return False
    return bool(load_state(state_path).get("promoted"))
