#!/usr/bin/env python3
"""紫苑記憶想起の評価セット候補を実クエリから生成する（月次・レビュー前提）。

評価セット（api/knowledge/shion_recall_eval_set.json、16ケース）は骨格確認には
十分だが、これだけに合わせ込むと過学習する。実際の利用から候補を拾って
人間レビューで採用する入口を作る:

1. data/shion_memory_usage_log.jsonl の実クエリ（question フィールド）のうち、
   繰り返し聞かれている（既定: 2回以上）かつ既存評価セットに無いもの
2. data/rag_feedback_log.jsonl の低評価（bad / wrong / needs_fix）が付いたクエリ
   （想起が外れた実例 = 最も価値の高い回帰ケース）

出力はドラフト（expected_route は観測値、expected_path_any は空）。
人間が中身を確認・修正して評価セットへ手動で移す。自動では追加しない。

既定では毎月1日のみ実行（夜間パイプラインに置いても月次で動く）。--force で随時実行。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USAGE_LOG = REPO_ROOT / "data" / "shion_memory_usage_log.jsonl"
DEFAULT_RAG_FEEDBACK = REPO_ROOT / "data" / "rag_feedback_log.jsonl"
DEFAULT_EVAL_SET = REPO_ROOT / "api" / "knowledge" / "shion_recall_eval_set.json"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "reports" / "shion_eval_candidates_latest.json"
DEFAULT_OUTPUT_MD = REPO_ROOT / "reports" / "shion_eval_candidates_latest.md"

_NEGATIVE_RATINGS = {"bad", "wrong", "needs_fix", "thin", "not_shion"}
_WS_RE = re.compile(r"\s+")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _normalize_query(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip())[:120]


def _existing_eval_queries(eval_set_path: Path) -> set[str]:
    try:
        cases = json.loads(eval_set_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(cases, list):
        return set()
    return {_normalize_query(c.get("query")) for c in cases if isinstance(c, dict)}


def collect_candidates(
    usage_rows: list[dict],
    feedback_rows: list[dict],
    existing_queries: set[str],
    *,
    min_repeat: int = 2,
    limit: int = 20,
) -> list[dict]:
    candidates: dict[str, dict] = {}

    # 低評価フィードバックのクエリ（優先度: 高）
    for row in feedback_rows:
        rating = str(row.get("rating") or "").strip().lower()
        query = _normalize_query(row.get("query"))
        if not query or rating not in _NEGATIVE_RATINGS:
            continue
        if query in existing_queries or query in candidates:
            continue
        candidates[query] = {
            "id": f"candidate_feedback_{len(candidates) + 1:02d}",
            "category": "draft",
            "query": query,
            "expected_route": "",
            "expected_path_any": [],
            "forbidden_path_any": [],
            "_provenance": f"rag_feedback rating={rating}（想起が外れた実例）",
            "_priority": "high",
        }

    # 繰り返し聞かれている実クエリ
    query_routes: dict[str, Counter] = {}
    query_counts: Counter = Counter()
    for row in usage_rows:
        query = _normalize_query(row.get("question"))
        if not query or len(query) < 8:
            continue
        query_counts[query] += 1
        query_routes.setdefault(query, Counter())[str(row.get("route") or "")] += 1
    for query, count in query_counts.most_common():
        if count < min_repeat:
            break
        if query in existing_queries or query in candidates:
            continue
        route = query_routes[query].most_common(1)[0][0]
        candidates[query] = {
            "id": f"candidate_usage_{len(candidates) + 1:02d}",
            "category": "draft",
            "query": query,
            "expected_route": route,
            "expected_path_any": [],
            "forbidden_path_any": [],
            "_provenance": f"使用ログで {count} 回想起（観測ルート: {route or '不明'}）",
            "_priority": "normal",
        }

    ordered = sorted(
        candidates.values(), key=lambda c: (c["_priority"] != "high", c["id"])
    )
    return ordered[:limit]


def _render_markdown(candidates: list[dict]) -> str:
    lines = [
        "# 紫苑記憶 評価セット候補（ドラフト）",
        "",
        f"- 生成: {datetime.now().isoformat(timespec='seconds')}",
        f"- 候補: {len(candidates)} 件",
        "- 採用方法: 内容を確認し expected_route / expected_path_any を埋めて",
        "  `api/knowledge/shion_recall_eval_set.json` へ手動で追加する（自動追加はしない）",
        "",
    ]
    if not candidates:
        lines.append("候補はありません。")
        return "\n".join(lines) + "\n"
    for c in candidates:
        lines += [
            f"## {c['id']}（{c['_priority']}）",
            f"- クエリ: {c['query']}",
            f"- 観測ルート: {c['expected_route'] or '（未観測・要判断）'}",
            f"- 出所: {c['_provenance']}",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="紫苑記憶の評価セット候補を実クエリから生成")
    parser.add_argument("--usage-log", type=Path, default=DEFAULT_USAGE_LOG)
    parser.add_argument("--rag-feedback", type=Path, default=DEFAULT_RAG_FEEDBACK)
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--min-repeat", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="月初以外でも実行する")
    args = parser.parse_args()

    if not args.force and date.today().day != 1:
        print("月初ではないためスキップします（--force で随時実行可）")
        return 0

    candidates = collect_candidates(
        _load_jsonl(args.usage_log),
        _load_jsonl(args.rag_feedback),
        _existing_eval_queries(args.eval_set),
        min_repeat=args.min_repeat,
        limit=args.limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "candidates": candidates,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.output_md.write_text(_render_markdown(candidates), encoding="utf-8")
    print(f"評価セット候補: {len(candidates)} 件 → {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
