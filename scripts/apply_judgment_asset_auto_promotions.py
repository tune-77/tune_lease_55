#!/usr/bin/env python3
"""Auto-apply pilot: promote only high-score/low-risk judgment asset candidates.

2026-07-29のアーキテクチャ監査は「正式判断資産への昇格条件をread-onlyレポートで
定義し、いきなり自動適用しない」という方針だった
(scripts/build_judgment_asset_promotion_readiness_report.py 参照)。

2026-09-06にユーザーの明示的な承認を得て、その方針を限定的に上書きする試験導入を
行う: readiness reportが `auto_apply_eligible` と判定した候補（ready_for_review
の中でも score>=AUTO_APPLY_MIN_SCORE かつ evidence>=AUTO_APPLY_MIN_EVIDENCE_COUNT
のみ）だけを、人間が「昇格」ボタンを押すのと同じ経路
（api.routers.feedback_loop._promote_judgment_asset_candidate_to_canonical）に通す。

安全装置:
- 既定オフ。環境変数 JUDGMENT_ASSET_AUTO_APPLY_ENABLED=1 を明示しない限り何もしない。
- 昇格ロジックは複製せず、人間昇格と同じ関数をそのまま呼ぶ。
- promotion_source / promoted_by を "auto_apply_pilot" にして人間昇格と区別する。
- 1件の失敗が全体を止めないよう、候補ごとに例外を捕捉して処理を続ける。
- 実行結果を data/judgment_asset_auto_apply_log.jsonl に追記し、事後監査できるようにする。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from scripts import build_judgment_asset_promotion_readiness_report as readiness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_JSONL = PROJECT_ROOT / "data" / "judgment_asset_auto_apply_log.jsonl"

ENABLE_ENV_VAR = "JUDGMENT_ASSET_AUTO_APPLY_ENABLED"
PROMOTED_BY = "auto_apply_pilot"


def is_enabled() -> bool:
    return os.environ.get(ENABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}


def _eligible_candidate_ids(payload: dict[str, Any]) -> list[str]:
    ready = payload.get("buckets", {}).get("ready_for_review") or []
    return [str(entry["id"]) for entry in ready if entry.get("auto_apply_eligible")]


def _append_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(
    *,
    candidates_jsonl: Path,
    state_json: Path,
    canonical_json: Path,
    log_jsonl: Path,
    target_date: str,
    promote_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    payload = readiness.build_report(
        target_date=target_date,
        candidates=readiness._read_jsonl(candidates_jsonl),
        state=readiness._read_json(state_json),
        canonical=readiness._read_json(canonical_json),
    )
    eligible_ids = _eligible_candidate_ids(payload)
    promoted: list[str] = []
    errors: list[dict[str, str]] = []
    now = datetime.now().isoformat(timespec="seconds")

    for candidate_id in eligible_ids:
        try:
            result = promote_fn(candidate_id, promoted_by=PROMOTED_BY)
        except Exception as exc:  # noqa: BLE001 - 1候補の異常で残りを止めない
            errors.append({"id": candidate_id, "error": str(exc)})
            continue
        promoted.append(candidate_id)
        _append_log(
            log_jsonl,
            {
                "timestamp": now,
                "candidate_id": candidate_id,
                "rule_id": (result.get("rule") or {}).get("id"),
                "status": result.get("status"),
                "promoted_by": PROMOTED_BY,
            },
        )

    return {
        "enabled": True,
        "guardrail": "opt_in_env_var_reuses_human_promotion_path_no_duplicate_logic",
        "eligible_count": len(eligible_ids),
        "promoted_count": len(promoted),
        "error_count": len(errors),
        "promoted_ids": promoted,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--candidates-jsonl", type=Path, default=readiness.DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--state-json", type=Path, default=readiness.DEFAULT_STATE_JSON)
    parser.add_argument("--canonical-json", type=Path, default=readiness.DEFAULT_CANONICAL_JSON)
    parser.add_argument("--log-jsonl", type=Path, default=DEFAULT_LOG_JSONL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not is_enabled():
        print(
            json.dumps(
                {
                    "enabled": False,
                    "reason": f"{ENABLE_ENV_VAR} is not set to 1/true/yes; pilot is opt-in and did nothing",
                },
                ensure_ascii=False,
            )
        )
        return 0

    from api.routers.feedback_loop import _promote_judgment_asset_candidate_to_canonical

    result = run(
        candidates_jsonl=args.candidates_jsonl,
        state_json=args.state_json,
        canonical_json=args.canonical_json,
        log_jsonl=args.log_jsonl,
        target_date=args.date,
        promote_fn=_promote_judgment_asset_candidate_to_canonical,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
