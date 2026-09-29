"""Jev判定の共通ログ（REV-424）。後からBrierスコア等で較正するための記録。

1質問1行のJSONLに、確率・行き先・閾値・モデル名を残す。記事タイトルや主張文などの
本文は書かず、subject_hash（正規化後のSHA-256先頭16桁）だけを残す。
正解ラベルは後から `append_label` で別レコードとして追記する（既存行は書き換えない。
同じ judgment_id のラベルは最後のものが有効）。

人間が見るのは review/flagged に回った判定だけなので、それだけにラベルを付けると
「自動で通った判定の誤り」が永久に観測されない。自動通過した判定から一定割合を
`sampled_for_review` で抜き取り、ラベル付けの対象にする。
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "data" / "jev_judgment_log.jsonl"
DEFAULT_SAMPLE_RATE = 0.1

_SPACE_RE = re.compile(r"\s+")


def log_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """書き込み先。`JEV_JUDGMENT_LOG_PATH=off` なら記録しない。"""
    env = os.environ if environ is None else environ
    raw = str(env.get("JEV_JUDGMENT_LOG_PATH") or "").strip()
    if raw.lower() == "off":
        return None
    return Path(raw) if raw else DEFAULT_LOG_PATH


def configured_sample_rate(environ: Mapping[str, str] | None = None) -> float:
    env = os.environ if environ is None else environ
    try:
        value = float(env.get("JEV_JUDGMENT_SAMPLE_RATE", DEFAULT_SAMPLE_RATE))
    except (TypeError, ValueError):
        return DEFAULT_SAMPLE_RATE
    return value if 0.0 <= value <= 1.0 else DEFAULT_SAMPLE_RATE


def subject_hash(text: str) -> str:
    normalized = _SPACE_RE.sub(" ", str(text or "")).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return uuid.uuid4().hex


def _now_iso(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


def build_records(
    *,
    guard: str,
    run_id: str,
    mode: str,
    model: str,
    items: Sequence[Mapping[str, Any]],
    sample_rate: float | None = None,
    rng: random.Random | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """判定の列をログレコードへ変換する。`subject` はハッシュ化にだけ使う。"""
    rate = configured_sample_rate() if sample_rate is None else sample_rate
    draw = (rng or random.Random()).random
    ts = _now_iso(now)
    # 1記事に複数の質問がある時、質問ごとに抜き取ると同じ記事の片方だけに
    # ラベルが付いて突き合わせられない。抜き取りは subject 単位で1回だけ引く。
    sampled_by_subject: dict[str, bool] = {}
    records: list[dict[str, Any]] = []
    for item in items:
        digest = subject_hash(str(item.get("subject") or ""))
        auto_passed = bool(item.get("auto_passed"))
        if digest not in sampled_by_subject:
            sampled_by_subject[digest] = draw() < rate
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "judgment",
                "ts": ts,
                "judgment_id": uuid.uuid4().hex,
                "run_id": run_id,
                "guard": guard,
                "mode": mode,
                "model": model,
                "question": str(item.get("question") or ""),
                "subject_hash": digest,
                "probability": item.get("probability"),
                "choice": item.get("choice"),
                "route": str(item.get("route") or ""),
                "thresholds": dict(item.get("thresholds") or {}),
                "auto_passed": auto_passed,
                "sampled_for_review": auto_passed and sampled_by_subject[digest],
                "label": None,
                "label_source": None,
                "labeled_at": None,
            }
        )
    return records


def _append_lines(rows: Iterable[Mapping[str, Any]], path: Path | None) -> int:
    target = log_path() if path is None else path
    if target is None:
        return 0
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    if not lines:
        return 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except OSError:
        # ログ障害でガード本体（ニュース収集・ノート保存）を止めない。
        return 0
    return len(lines)


def append_records(records: Sequence[Mapping[str, Any]], path: Path | None = None) -> int:
    return _append_lines(records, path)


def append_label(
    judgment_id: str,
    label: Any,
    *,
    label_source: str,
    path: Path | None = None,
    now: datetime | None = None,
) -> int:
    return _append_lines(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "label",
                "judgment_id": judgment_id,
                "label": label,
                "label_source": label_source,
                "labeled_at": _now_iso(now),
            }
        ],
        path,
    )
