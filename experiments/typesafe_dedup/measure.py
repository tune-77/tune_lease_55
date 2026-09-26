"""Offline measurement for the TypeSafe/Jev dedup pilot (spec section 8).

Deviation from the spec, recorded deliberately
----------------------------------------------
Spec section 8 names ``~/Library/Logs/tunelease/improvement_YYYYMMDD.log`` as the
input.  Those files turned out to contain only pipeline self-noise
(``[改善] auto-improvement-pipeline 実行中...``), one or two lines per day, and no
candidates.  The real corpus is ``reports/improvement_report_YYYYMMDD.json``,
whose ``needs_review`` and ``rejected`` entries already carry ``title`` and
``reason`` -- the exact shape ``_parse_improvements()`` emits.  No log scraping and
no parser call is needed.

The script is read-only with respect to the pipeline: it imports nothing from the
daily run and writes only under ``results/``.

Usage
-----
    python3 experiments/typesafe_dedup/measure.py                 # bands only, offline
    python3 experiments/typesafe_dedup/measure.py --inspect       # print gray pairs for review
    python3 experiments/typesafe_dedup/measure.py --send          # calls TypeSafe (see section 10)
    python3 experiments/typesafe_dedup/measure.py --sweep results/judged_*.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORTS_DIR = ROOT / "reports"

sys.path.insert(0, str(ROOT))

import typesafe_dedup_guard as guard  # noqa: E402

# All reports, not a recent window.  A 30-file window holds only 6 distinct
# titles -- the same unresolved items re-emitted verbatim each day -- which makes
# the gray band empty for reasons that have nothing to do with the hypothesis.
DEFAULT_REPORT_COUNT = 1000
BAND_WIDTHS = (0.20, 0.25, 0.30)
THRESHOLD_SWEEP = (0.50, 0.60, 0.70)

# Spec section 10.  Candidates originate in Obsidian AI chat logs and may contain
# real company names or case numbers.
SEND_WARNING = (
    "灰色帯ペアの全文を目視し、実在の企業名・案件番号が含まれていないことを\n"
    "確認してから送信すること（仕様書 §10）。--inspect で全文を表示できる。"
)


def _load_extract_module():
    """Load the pipeline script as a module without importing the package.

    Same convention as ``tests/test_typesafe_dedup_guard.py``: the script is not
    importable as a package member, and loading it this way keeps
    ``_jaccard_similarity`` the single source of truth for the bigram metric.
    """
    spec = importlib.util.spec_from_file_location(
        "extract_obsidian_improvements_for_measurement",
        ROOT / "scripts" / "extract_obsidian_improvements.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# A whole-line ASCII key-value bullet, e.g. ``- status: rejected``.  Anchored on
# both ends by fullmatch(), so a real title that merely starts with a hyphen or
# merely contains a colon is not touched.
_ARTIFACT_KEY_VALUE_LINE = re.compile(r"[-*]\s*[A-Za-z_][A-Za-z0-9_]*:\s*[\x21-\x7e]*")


def _is_parse_artifact(title: str) -> bool:
    """Return True when ``title`` is a parser leftover, not an improvement.

    The 2026-09-20 inspection found four titles of the form ``- status: rejected``
    in the gray band: YAML lines from the source note that
    ``_parse_improvements()`` scraped as candidates.  They share the literal
    ``- status: `` prefix, so the bigram metric lands them in exactly the band
    this pilot sends to Jev -- paying tokens to compare two pieces of nothing.

    The error is asymmetric, the same way the routing thresholds are.  A title
    wrongly dropped here disappears before any measurement sees it and leaves no
    trace; a piece of garbage left in is visible in ``--inspect`` and costs one
    pair.  So the predicate must be narrow: it should refuse to match anything
    that could plausibly be a real (Japanese) improvement title.

    ``_ARTIFACT_KEY_VALUE_LINE`` therefore requires the *whole* title to be an
    ASCII key-value bullet.  A Japanese title cannot match, and neither can
    ``- EDINET連携の強化`` (no colon) or ``- reason: 対象ファイル未特定`` (the value
    is not ASCII).  Broader shapes of parser garbage survive on purpose: they
    stay visible in ``--inspect`` instead of silently thinning the corpus.
    """
    return bool(_ARTIFACT_KEY_VALUE_LINE.fullmatch(title))


def load_candidates(
    report_count: int = DEFAULT_REPORT_COUNT,
    *,
    distinct: bool = True,
    keep_artifacts: bool = False,
) -> list[dict[str, Any]]:
    """Collect improvement candidates from the daily reports.

    ``distinct`` keeps the first occurrence of each title.  The reports re-emit
    every unresolved candidate verbatim each day, so without it one real
    candidate contributes hundreds of identical pairs: stage 1 settles them all,
    and both the pair count and the gray-band count stop describing the corpus.
    Measure over entities, not occurrences.

    ``keep_artifacts`` disables the parse-artifact filter so the dropped titles
    can be inspected; see ``_is_parse_artifact``.
    """
    reports = sorted(REPORTS_DIR.glob("improvement_report_*.json"))[-report_count:]
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in reports:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for bucket in ("applied", "needs_review", "rejected"):
            for entry in payload.get(bucket) or []:
                title = str(entry.get("title") or "").strip()
                if not title:
                    continue
                if not keep_artifacts and _is_parse_artifact(title):
                    continue
                if distinct:
                    if title in seen:
                        continue
                    seen.add(title)
                candidates.append(
                    {
                        "title": title,
                        "reason": str(entry.get("reason") or "").strip(),
                        "id": entry.get("id"),
                        "bucket": bucket,
                        "report": path.name,
                    }
                )
    return candidates


def settled_by_cheap_stages(module, a_title: str, b_title: str) -> str | None:
    """Return the stage name that already decides this pair, or None.

    Mirrors stages 1-3 of ``deduplicate_improvements()``.  ``_in_same_theme`` is a
    closure inside that function, so the theme check is re-derived here from the
    module-level ``_THEME_GROUPS`` table rather than copied by hand.
    """
    if a_title == b_title or a_title[:40] == b_title[:40]:
        return "exact_or_prefix"
    subset_min = 8  # matches _SUBSET_MIN_LEN in deduplicate_improvements()
    if len(a_title) >= subset_min and len(b_title) >= subset_min:
        if a_title in b_title or b_title in a_title:
            return "subset"
    for group in module._THEME_GROUPS:
        if any(kw in a_title for kw in group) and any(kw in b_title for kw in group):
            return "theme_group"
    return None


def analyze(candidates: list[dict[str, Any]], module) -> dict[str, Any]:
    """Compute stage coverage and gray-band width for every configured band."""
    similarity = module._jaccard_similarity
    stage_counts: dict[str, int] = {}
    open_pairs: list[tuple[int, int, float]] = []
    total_pairs = 0

    for i in range(len(candidates)):
        title_i = candidates[i]["title"]
        for j in range(i + 1, len(candidates)):
            total_pairs += 1
            title_j = candidates[j]["title"]
            stage = settled_by_cheap_stages(module, title_i, title_j)
            if stage:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
                continue
            open_pairs.append((i, j, similarity(title_i, title_j)))

    above_threshold = sum(1 for _, _, s in open_pairs if s >= guard.GRAY_HIGH)
    bands = {}
    for width in BAND_WIDTHS:
        low = round(guard.GRAY_HIGH - width, 4)
        bands[f"{low:.2f}-{guard.GRAY_HIGH:.2f}"] = sum(
            1 for _, _, s in open_pairs if low <= s < guard.GRAY_HIGH
        )

    return {
        "candidate_count": len(candidates),
        "total_pairs": total_pairs,
        "settled_by_cheap_stages": stage_counts,
        "settled_total": sum(stage_counts.values()),
        "duplicate_by_jaccard": above_threshold,
        "gray_band_pairs": bands,
        "open_pairs": open_pairs,
    }


def gray_pairs_for(analysis: dict[str, Any], width: float) -> list[tuple[int, int]]:
    low = guard.GRAY_HIGH - width
    return [(i, j) for i, j, s in analysis["open_pairs"] if low <= s < guard.GRAY_HIGH]


# Operator policy (decided 2026-09-20).  Wider and higher than
# guard.SAME_ISSUE_MIN's plain split on purpose: 誤併合 drops a candidate that
# never reaches the ledger, so nobody notices it, while 誤分割 only creates a
# duplicate REV a human can see and close.  The cheaper mistake is the visible
# one, so automatic merging must clear a high bar and everything in between goes
# to a person instead of being forced into a binary answer.
REVIEW_LOW = 0.40
REVIEW_HIGH = 0.75


def classify_judged_pair(
    same_issue: float, *, low: float = REVIEW_LOW, high: float = REVIEW_HIGH
) -> str:
    """Apply the operator's routing policy to one raw Jev probability.

    Three-way, unlike ``guard.route_pair()``: ``[low, high)`` is the band where
    Jev is not confident enough to justify a silent merge, so the pair is handed
    to a human rather than decided.
    """
    if same_issue >= high:
        return "duplicate"
    if same_issue >= low:
        return "needs_human_review"
    return "distinct"


def _print_analysis(analysis: dict[str, Any]) -> None:
    print(f"候補数: {analysis['candidate_count']}")
    print(f"全ペア数: {analysis['total_pairs']}")
    print(f"段1〜3で確定: {analysis['settled_total']}  {analysis['settled_by_cheap_stages']}")
    print(f"Jaccard >= {guard.GRAY_HIGH} で重複確定: {analysis['duplicate_by_jaccard']}")
    print("灰色帯ペア数（= Jev 呼出し対象数）:")
    for band, count in analysis["gray_band_pairs"].items():
        print(f"  {band}: {count}")


def _print_gray_pairs(candidates, analysis, width: float) -> None:
    print(f"\n--- 灰色帯 幅{width:.2f} の全文（§10 目視確認用） ---")
    for i, j in gray_pairs_for(analysis, width):
        print(f"[{candidates[i].get('id')}] {candidates[i]['title']}")
        print(f"    理由: {candidates[i]['reason']}")
        print(f"[{candidates[j].get('id')}] {candidates[j]['title']}")
        print(f"    理由: {candidates[j]['reason']}")
        print()


def _send(candidates, analysis, width: float) -> Path:
    pairs = gray_pairs_for(analysis, width)
    if not pairs:
        raise SystemExit("灰色帯にペアがないため送信しない")
    safe_pairs, privacy_skipped = guard.filter_safe_pairs(candidates, pairs)
    if not safe_pairs:
        raise SystemExit("機密情報フィルター通過後の送信可能ペアがない")
    judged, meta = guard.judge_pairs(candidates, safe_pairs)
    meta = dict(meta)
    meta["privacy_skipped_pairs"] = privacy_skipped
    if privacy_skipped:
        print(f"機密情報の可能性があるため送信除外: {privacy_skipped} ペア")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judged_w{int(width * 100)}.json"
    out.write_text(
        json.dumps(
            {
                "band_width": width,
                "meta": meta,
                "judged": [
                    {
                        **item,
                        "a_title": candidates[item["a"]]["title"],
                        "b_title": candidates[item["b"]]["title"],
                    }
                    for item in judged
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _sweep(path: Path) -> None:
    """Re-route saved probabilities offline; no inference is repeated."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    judged = payload["judged"]
    print(f"{path.name}: {len(judged)} pairs")
    for threshold in THRESHOLD_SWEEP:
        duplicates = sum(
            1
            for item in judged
            if guard.route_pair(item["same_issue"], threshold=threshold) == "duplicate"
        )
        print(f"  SAME_ISSUE_MIN={threshold:.2f} -> duplicate {duplicates}")

    counts: dict[str, int] = {}
    for item in judged:
        route = classify_judged_pair(item["same_issue"])
        counts[route] = counts.get(route, 0) + 1
    print(f"  運用ポリシー（{REVIEW_LOW:.2f}-{REVIEW_HIGH:.2f} を人手）-> {counts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=int, default=DEFAULT_REPORT_COUNT)
    parser.add_argument(
        "--keep-repeats",
        action="store_true",
        help="同一タイトルの日次再出力も別候補として数える（母集団の確認用）",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="パース不良タイトル（`- status: xxx` 等）も候補として数える（確認用）",
    )
    parser.add_argument("--width", type=float, default=0.25, choices=BAND_WIDTHS)
    parser.add_argument("--inspect", action="store_true", help="灰色帯ペアの全文を表示")
    parser.add_argument("--send", action="store_true", help="TypeSafe に送信する")
    parser.add_argument("--sweep", type=Path, help="保存済み結果の閾値スイープ")
    args = parser.parse_args()

    if args.sweep:
        _sweep(args.sweep)
        return 0

    module = _load_extract_module()
    candidates = load_candidates(
        args.reports,
        distinct=not args.keep_repeats,
        keep_artifacts=args.keep_artifacts,
    )
    if not candidates:
        print("候補が見つからない（reports/improvement_report_*.json を確認）", file=sys.stderr)
        return 1

    analysis = analyze(candidates, module)
    _print_analysis(analysis)

    if args.inspect:
        _print_gray_pairs(candidates, analysis, args.width)

    if args.send:
        if not guard.typesafe_dedup_enabled():
            print("\nTYPESAFE_DEDUP_ENABLED と API キーが未設定のため送信しない", file=sys.stderr)
            return 1
        print(f"\n{SEND_WARNING}")
        out = _send(candidates, analysis, args.width)
        print(f"保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
