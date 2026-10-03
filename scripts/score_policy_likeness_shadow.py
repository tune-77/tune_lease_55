#!/usr/bin/env python3
"""active の判断資産の「方針らしさ」を Jev で採点し、3段階（方針/社内の目安/知見）のキューを更新する。

shadow: 回答には使わない。中間（社内の目安）は /judgment-review でユーザーが1タップで方針かどうか決め、
その判断を次の基準づくり・評価のラベルとして貯める。詳細は api/policy_likeness.py。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api import policy_likeness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-json", type=Path, default=policy_likeness.DEFAULT_CANONICAL_JSON)
    parser.add_argument("--queue-json", type=Path, default=policy_likeness.DEFAULT_QUEUE_JSON)
    args = parser.parse_args()
    result = policy_likeness.score_canonical_rules(canonical_path=args.canonical_json, queue_path=args.queue_json)
    review = policy_likeness.review_candidates(args.queue_json)
    print(f"方針らしさ shadow: 対象{result['targets']} 採点{result['scored']} 失敗{result['failed']} / 目安の判定待ち {review['total_count']}件")
    # 採点対象があるのに1件も採点できない＝Jev 不通。失敗を成功と記録しない
    return 1 if result["targets"] and not result["scored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
