"""紫苑記憶想起の精度評価ハーネス。

api/knowledge/shion_recall_eval_set.json の各ケースについて
`recall_memories()` を実行し、想起ルートと参照ノート（source_path）が
期待どおりかを判定する。okf_rag_eval_set.json と同じ思想の
「質問 → 期待出典」形式を、記憶想起（route / refs）向けに拡張したもの。

使い方:
    python3 scripts/eval_shion_memory_recall.py            # 索引を組み立てて評価
    python3 scripts/eval_shion_memory_recall.py --index data/shion_memory_index.json
    python3 scripts/eval_shion_memory_recall.py --min-pass-rate 0.9

チューニング時は --verbose で各ケースの想起結果を確認する。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "shion_recall_eval_set.json"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    route_ok: bool
    refs_ok: bool
    actual_route: str
    recalled_paths: list[str] = field(default_factory=list)
    detail: str = ""


def load_eval_cases(path: Path = DEFAULT_EVAL_SET) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _built_index_path(explicit: Path | None) -> Path:
    """評価に使う索引パスを返す。指定が無ければ現リポジトリから組み立てる。"""
    if explicit is not None:
        return explicit
    import scripts.build_shion_memory_index as builder

    index = builder.build_index()
    tmp = Path(tempfile.mkdtemp(prefix="shion_recall_eval_")) / "index.json"
    tmp.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return tmp


def evaluate_case(case: dict[str, Any], *, index_path: Path, limit: int = 5) -> CaseResult:
    from api.shion_memory_recall import recall_memories

    case_id = str(case.get("id") or "")
    query = str(case.get("query") or "")
    expected_route = str(case.get("expected_route") or "")
    expected_any = [str(p) for p in case.get("expected_path_any") or []]
    forbidden_any = [str(p) for p in case.get("forbidden_path_any") or []]

    recalled = recall_memories(query, limit=limit, index_path=index_path)
    actual_route = str(recalled.get("route") or "")
    paths = [str(m.get("source_path") or "") for m in recalled.get("memories") or []]

    route_ok = (not expected_route) or actual_route == expected_route
    hit_expected = (not expected_any) or any(
        exp in path for exp in expected_any for path in paths
    )
    hit_forbidden = any(bad in path for bad in forbidden_any for path in paths)
    refs_ok = hit_expected and not hit_forbidden

    details = []
    if not route_ok:
        details.append(f"route: expected={expected_route} actual={actual_route}")
    if not hit_expected:
        details.append(f"expected_path_any not recalled: {expected_any}")
    if hit_forbidden:
        details.append(f"forbidden path recalled: {forbidden_any}")

    return CaseResult(
        case_id=case_id,
        passed=route_ok and refs_ok,
        route_ok=route_ok,
        refs_ok=refs_ok,
        actual_route=actual_route,
        recalled_paths=paths,
        detail="; ".join(details),
    )


def run_eval(
    *,
    eval_set_path: Path = DEFAULT_EVAL_SET,
    index_path: Path | None = None,
    limit: int = 5,
) -> list[CaseResult]:
    cases = load_eval_cases(eval_set_path)
    resolved_index = _built_index_path(index_path)
    return [evaluate_case(case, index_path=resolved_index, limit=limit) for case in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶想起の精度評価")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--index", type=Path, default=None, help="既存の記憶索引JSON。省略時はその場で構築")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = run_eval(eval_set_path=args.eval_set, index_path=args.index, limit=args.limit)
    if not results:
        print("評価ケースがありません:", args.eval_set)
        return 1

    passed = sum(1 for r in results if r.passed)
    route_ok = sum(1 for r in results if r.route_ok)
    refs_ok = sum(1 for r in results if r.refs_ok)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.case_id} (route={r.actual_route})")
        if r.detail:
            print(f"       {r.detail}")
        if args.verbose:
            for path in r.recalled_paths:
                print(f"       - {path}")

    total = len(results)
    pass_rate = passed / total
    print()
    print(f"route accuracy: {route_ok}/{total}")
    print(f"refs accuracy:  {refs_ok}/{total}")
    print(f"overall:        {passed}/{total} ({pass_rate:.0%})")
    return 0 if pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
