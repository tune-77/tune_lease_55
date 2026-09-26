"""Offline measurement for the TypeSafe/Jev asset-score pilot (Phase 0).

Where the corpus actually is
----------------------------
An earlier scan of ``past_cases`` read only the TOP level of each row's JSON
blob, found 9 scores, and concluded the corpus was 2 usable rows.  That was
wrong: the labels live one level down, in ``data.inputs``.  There are 2038.
``collect_db_candidates`` below documents the two populations and the measured
error floors; read that comment before trusting any number this script prints.

Most of those rows are not teaching data
----------------------------------------
The operator's form changed twice (2026-02 and 2026-06).  Before the first date
every score is one of 100/85/80 -- a three-way grade, not an assessment of the
item -- and from the second date the same case is written twice, once in each
form.  ``admit_row`` keeps only the rows that survive both facts: 250 rows, and
because a repeated (name, detail) pair is one question answered many times, just
59 distinct questions.  It owns that policy alone; extraction never filters, so
the cut can be changed and the corpus re-counted without touching anything else.

The corpus file itself stays an operator-owned JSONL fixture.  ``--seed``
appends the admitted DB rows to it; the reviewer may add or correct lines by
hand, and existing lines are never overwritten.

The script is read-only with respect to production: it imports
``typesafe_asset_guard`` (which nothing else imports) and ``asset_scorer``,
opens the DB with ``mode=ro``, and writes only under ``fixtures/`` and
``results/``.

Usage
-----
    python3 experiments/typesafe_asset/measure.py --seed      # bootstrap fixture from DB
    python3 experiments/typesafe_asset/measure.py             # corpus stats, offline
    python3 experiments/typesafe_asset/measure.py --inspect   # show the exact payloads
    python3 experiments/typesafe_asset/measure.py --send      # calls TypeSafe
    python3 experiments/typesafe_asset/measure.py --sweep results/judged_*.json
"""


from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIXTURE_PATH = HERE / "fixtures" / "asset_labels.jsonl"

sys.path.insert(0, str(ROOT))

import typesafe_asset_guard as guard  # noqa: E402
from asset_scorer import calc_asset_score  # noqa: E402

SEND_WARNING = (
    "送信される品名の全文を目視し、実在の企業名・案件番号・担当者名が\n"
    "含まれていないことを確認してから送信すること。--inspect で全文を表示できる。"
)

# Two-sided 95% Student-t critical values. For larger samples the t
# distribution is sufficiently close to normal for this diagnostic.
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _student_t_95(df: int) -> float:
    if df < 1:
        return float("inf")
    if df <= 30:
        return _T_975[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960

# ---------------------------------------------------------------------------
# Corpus extraction
# ---------------------------------------------------------------------------
# The labels live in the NESTED ``past_cases.data.inputs`` dict, not at the top
# level.  An earlier scan looked only at the top level and concluded the corpus
# was 2 rows; it is 2038.  Two disjoint populations are present -- a full scan
# found zero rows carrying both:
#
#   A: inputs.lease_asset_name + inputs.lease_asset_score   (2038 rows)
#      Real product names ("（株）ミツトヨ輪郭形状・表面粗さ測定器"), 868 distinct.
#      The score is quantised to 21 values and 100.0 is the 2025 era default:
#      ~70% share every month to 2026-01, then absent from 2026-05 onward.
#      No category column exists, so the category is inferred from the name.
#
#   B: inputs.asset_name + inputs.asset_detail + inputs.asset_score  (148 rows)
#      asset_name is a coarse type label, so the category is a table lookup.
#      17 score values with no single dominant default, but asset_detail is
#      frequently the bare word "新品" -- identical text maps to 40 and to 80.
#
# Which rows are real judgments is settled by the TIMESTAMP, not by blacklisting
# score values.  An earlier draft guessed that {100, 85, 50} were filler; the
# month-by-month composition of population A disproves it -- 85.0 holds a steady
# 11-29% share in every single month, so it is a coarse judgment, while 100.0
# vanishes completely after 2026-03.  What actually changed was the operator's
# form, on two dates:
#
#   month      n   100.0  85.0  80.0  78.0  other
#   2025-06  179     68%   15%   15%     -     2%
#   2026-01  100     77%   14%    9%     -     -
#   2026-02   70     43%   27%    6%   14%     7%
#   2026-03   62     44%   10%    3%   15%    29%
#   2026-05   18      -     -     -    50%    44%   <- ERA_START: 100.0 is gone
#   2026-06  104      -     -     5%   27%    58%   <- B_ERA_START: form B begins
#
# The era boundary is where 100.0 DISAPPEARS, not where 78.0 first appears.  An
# earlier cut at 2026-02 was wrong: 78.0 shows up that month, but the old palette
# keeps running alongside it for two more months, so 2026-02 and 2026-03 admitted
# 43-44% filler.  100.0 means "perfect collateral, zero disposal risk", which is
# never true of a used excavator -- it is a stock value, not an assessment.  It is
# absent from 2026-05 onward and 2026-04 has no rows at all, so the form change
# completed between 2026-03 and 2026-05.
#
# Note this is still an era cut by TIMESTAMP.  100.0's disappearance only locates
# the date; rows are admitted by month, never by score value.  Blacklisting a
# value would also remove 85.0, which holds a steady 10-27% share in every old
# month and is a real coarse judgment.
#
# From 2026-06 the operator writes BOTH forms for the same case, 1-2 seconds
# apart -- a full scan of that month found 104 A rows and 104 B rows, zero shared
# timestamps, and matching (name, score) pairs.  B is the strictly richer record
# because it carries ``asset_detail``:
#
#   A  21:03:15.970  車両・運搬車              50.0
#   B  21:03:17.443  車両・運搬車 / ランクル    50.0
#
# So A is kept only for [ERA_START, B_ERA_START) -- the new-era months B does not
# cover -- and B is kept entire.  Admitting both populations for 2026-06 onward
# would count the same human judgment twice.
#
# Measured on the admitted slice.  The unit that matters is the DISTINCT QUESTION
# -- the (name, detail) pair Jev is actually asked -- not the row count, because
# 56 rows reading "CNC複合加工機 / 新品" are one question answered 56 times.
# Re-cutting the era moves the bar a long way, because the discarded palette was
# dragging the corpus mean up to the 80s:
#
#   ERA_START   rows  questions  best constant  scored exactly 100
#   2026-02      250         59   13.25 (=85)                  25
#   2026-03      202         41   13.82 (=78)                  13
#   2026-05      155         22   12.32 (=62)                   0   <- in use
#   2026-06      144         17   10.06 (=50)                   0
#
# 12.32 is the bar.  Beating the 50-fixed baseline is automatic and proves
# nothing.  2026-06 would be cleaner still, but it drops population A entirely
# and buys that purity with 5 of 22 questions; 22 is already thin enough that the
# MAE estimate carries roughly +/-2.5, so only a clear margin under 12.32 counts.
ERA_START_MONTH = "2026-05"
B_ERA_START_MONTH = "2026-06"

# Names that identify no item.  The old form let the operator leave the field at
# a stock word; "一般" alone accounts for 209 rows scored 100/85/80/65.
PLACEHOLDER_NAMES = frozenset({"一般", "その他", "各種", "不明", "-", "--"})

# Population B stores a coarse type label that does not match the keys of
# ``category_config.CATEGORY_SCORE_ITEMS``.  Kept explicit so the mismatch stays
# visible rather than being papered over by a fuzzy match.
DB_CATEGORY_ALIASES = {
    "製造設備・工作機械": "産業機械",
    "車両・運搬車": "車両",
    "医療機器": "医療機器",
    "電子計算機": "IT機器",
    "IT・OA機器": "IT機器",
    "工作機械": "産業機械",
    "CNC工作機械": "産業機械",
    "CNC複合加工機": "産業機械",
    "CNCマシニングセンタ": "産業機械",
    "自動車（普通）": "車両",
    "飲食店設備": "",  # no dimension set defines it; excluded rather than guessed
}

# Population A has no category column at all.  This keyword table is a stated
# heuristic, not ground truth: it decides which dimension set the item is
# scored against, so a miss shifts the weights.  First match wins, so the
# order is deliberate -- "防犯カメラシステム" must reach IT機器 before the
# generic 機械 rule.
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("医療機器", ("医療", "診療", "治療", "手術", "歯科", "内視鏡", "レントゲン",
                "Ｘ線", "X線", "超音波", "電子カルテ", "CT")),
    ("車両", ("自動車", "車両", "トラック", "ダンプ", "フォークリフト", "エルフ",
             "キャンター", "トレーラ", "トラクタ", "バス", "運搬", "洗車", "重機",
             "ショベル", "ローダー", "ユニック", "乗用", "ハイエース", "アルファード",
             "キャラバン", "ウイング", "コンテナ")),
    ("IT機器", ("パソコン", "ＰＣ", "PC", "サーバ", "複合機", "複写機", "コピー",
               "ソフト", "カメラ", "システム", "券売機", "POS", "プリンタ",
               "電子計算機", "端末", "タブレット", "モニタ", "電話")),
    ("産業機械", ("CNC", "マシニング", "旋盤", "加工機", "工作", "測定",
                "プレス", "製造", "設備", "機械", "装置", "炉", "クレーン")),
]


def _infer_category(name: str) -> str:
    """Map a free-text product name onto a CATEGORY_SCORE_ITEMS key."""
    for category, keywords in CATEGORY_KEYWORDS:
        if any(keyword in name for keyword in keywords):
            return category
    return ""


# A default asset_score is a placeholder, not a label.
DEFAULT_ASSET_SCORE = 50.0


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def collect_db_candidates(db_path: Path) -> list[dict[str, Any]]:
    """Read every labelled asset row from both populations, unfiltered.

    Filtering is deliberately NOT done here: ``admit_row`` owns that policy so
    it can be changed and the corpus re-counted without touching extraction.
    """
    candidates: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for timestamp, blob in conn.execute("select timestamp, data from past_cases"):
            try:
                case = json.loads(blob)
            except (TypeError, ValueError):
                continue
            inputs = case.get("inputs")
            if not isinstance(inputs, dict):
                continue

            name = str(inputs.get("lease_asset_name") or "").strip()
            raw_score = inputs.get("lease_asset_score")
            population = "A"
            detail = ""
            if not name:
                label = str(inputs.get("asset_name") or "").strip()
                if not label:
                    continue
                name = label
                detail = str(inputs.get("asset_detail") or "").strip()
                raw_score = inputs.get("asset_score")
                population = "B"

            if raw_score in (None, ""):
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue

            if population == "B":
                category = DB_CATEGORY_ALIASES.get(name, "")
            else:
                category = _infer_category(name)

            candidates.append(
                {
                    "name": name,
                    "detail": detail,
                    "category": category,
                    "human_score": score,
                    "population": population,
                    "month": str(timestamp or "")[:7],
                    "source": f"lease_data.db/past_cases[{population}]",
                }
            )
    return candidates


def admit_row(row: dict[str, Any]) -> bool:
    """Decide whether one DB row is a genuine human label worth measuring against.

    ``row`` carries ``name``, ``detail``, ``category`` (may be ""),
    ``human_score``, ``population`` ("A" or "B") and ``month`` ("YYYY-MM").

    The policy and the evidence for it are in the comment above
    ``ERA_START_MONTH``.  Change this function and re-run ``--seed`` to re-cut
    the corpus; extraction never filters, so nothing else has to move.
    """
    month = str(row.get("month") or "")
    population = row.get("population")

    # calc_asset_score dispatches on the category, so a row we cannot place has
    # no dimension set to be scored against.
    if not row.get("category"):
        return False

    name = str(row.get("name") or "").strip()
    if not name or name in PLACEHOLDER_NAMES:
        return False

    # An unparseable month cannot be placed on either side of a form change.
    if len(month) != 7:
        return False

    if population == "A":
        # Pre-2026-02 is the 100/85/80 palette; 2026-06 onward is duplicated by
        # the richer B record for the same case.
        return ERA_START_MONTH <= month < B_ERA_START_MONTH
    if population == "B":
        return True
    return False


def _question_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """The unit Jev is actually asked about.

    ``detail`` belongs in the key: "車両・運搬車 / ランクル" and
    "車両・運搬車 / 大型ダンプ" are two different questions even though the name
    is identical.  Dropping it would collapse population B's 17 questions to 8.
    """
    return (
        str(row.get("name") or ""),
        str(row.get("detail") or ""),
        str(row.get("category") or ""),
    )


def seed_fixture(path: Path, db_path: Path) -> tuple[int, int]:
    """Bootstrap the fixture from the DB rows ``admit_row`` accepts.

    Returns ``(written, skipped)``.  Existing lines are never overwritten.

    One fixture line is one distinct question, not one DB row: the same item is
    leased over and over, so the admitted rows collapse hard.  The label written
    is the MEDIAN of the scores that question received rather than whichever row
    the cursor reached first -- a single outlier assessment should not become the
    thing Jev is graded against.  ``n`` and ``score_range`` keep the discarded
    spread visible, so a question whose humans disagreed can be spotted later.
    """
    existing = {_question_key(row) for row in _load_fixture(path)}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    skipped = 0

    for row in collect_db_candidates(db_path):
        if not admit_row(row):
            skipped += 1
            continue
        key = _question_key(row)
        if key in existing:
            continue
        grouped.setdefault(key, []).append(row)

    seeded: list[dict[str, Any]] = []
    for (name, detail, category), rows in grouped.items():
        scores = [float(row["human_score"]) for row in rows]
        seeded.append(
            {
                "name": name,
                "detail": detail,
                "category": category,
                "human_score": round(statistics.median(scores), 1),
                "n": len(rows),
                "score_range": [min(scores), max(scores)],
                "source": rows[0]["source"],
            }
        )

    if seeded:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in seeded:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(seeded), skipped


def _validate(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    for index, row in enumerate(rows):
        if not str(row.get("name") or "").strip():
            problems.append(f"[{index}] name が空")
        category = str(row.get("category") or "")
        try:
            guard.resolve_items(category)
        except guard.TypeSafeAssetError:
            problems.append(f"[{index}] 未知のカテゴリ: {category!r}")
        try:
            human_score = float(row["human_score"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"[{index}] human_score が数値でない")
            continue
        if not math.isfinite(human_score) or not 0.0 <= human_score <= 100.0:
            problems.append(f"[{index}] human_score が0〜100の有限値でない")
    return problems


def _print_corpus(rows: list[dict[str, Any]]) -> None:
    print(f"教師データ: {len(rows)} 問  ({FIXTURE_PATH})")
    if not rows:
        print("  --seed で DB から初期エントリを作成し、手で追記すること")
        return
    weights = [float(row.get("n") or 1) for row in rows]
    print(f"  元データ行数: {int(sum(weights))}  (同じ品名の繰り返しは1問に畳んである)")
    categories = Counter(str(row.get("category") or "?") for row in rows)
    print(f"  カテゴリ分布: {dict(categories)}")
    scores = [float(row["human_score"]) for row in rows if "human_score" in row]
    if scores:
        print(
            f"  人手スコア: min={min(scores):.0f} max={max(scores):.0f} "
            f"mean={statistics.fmean(scores):.1f}"
        )
        const, const_mae = _best_constant(scores, [1.0] * len(scores))
        print(
            f"  超えるべき基準: MAE {const_mae:.2f} (最良定数={const:.0f})  "
            f"/ 現行の50固定: "
            f"{statistics.fmean(abs(DEFAULT_ASSET_SCORE - s) for s in scores):.2f}"
        )
    split = sum(
        1
        for row in rows
        if row.get("score_range") and row["score_range"][0] != row["score_range"][1]
    )
    if split:
        print(f"  人手が割れた問: {split}  (中央値を採用。score_range に幅が残る)")
    if len(rows) < 20:
        print("  ⚠ 件数が少なく、一致率の差は偶然の範囲に収まる。追記を推奨")


def _approval_token(rows: list[dict[str, Any]]) -> str:
    """Bind an explicit send approval to the exact outbound payload set."""
    states = []
    for row in rows:
        items = guard.resolve_items(str(row["category"]))
        states.append(guard.build_asset_request(row, items)["state"])
    serialized = json.dumps(states, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _print_payloads(rows: list[dict[str, Any]]) -> str:
    print("\n--- 送信されるペイロード（これ以外は送られない） ---")
    for row in rows:
        items = guard.resolve_items(str(row["category"]))
        payload = guard.build_asset_request(row, items)
        print(json.dumps(payload["state"], ensure_ascii=False))
    token = _approval_token(rows)
    print(f"\n承認トークン: {token}")
    print("内容を確認後、別の実行で --send --approval-token <token> を指定すること")
    return token


def _compose(scores: dict[str, float], category: str) -> float:
    """Run the Jev item scores through the existing weighted average."""
    return float(calc_asset_score(category, scores)["total_score"])


def _send(rows: list[dict[str, Any]]) -> Path:
    judged: list[dict[str, Any]] = []
    for row in rows:
        category = str(row["category"])
        # Preserve every completed/billable result even if a later row fails.
        scores, meta = guard.judge_asset_if_enabled(row, category=category)
        judged.append(
            {
                "name": row["name"],
                "detail": row.get("detail", ""),
                "category": category,
                "human_score": float(row["human_score"]),
                "n": int(row.get("n") or 1),
                "jev_item_scores": scores,
                "jev_total": _compose(scores, category) if scores else None,
                "meta": meta,
            }
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judged_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(
        json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _report(judged)
    return out


def _best_constant(
    scores: list[float], weights: list[float]
) -> tuple[float, float]:
    """Return ``(constant, mae)`` for the single fixed number that fits best.

    This is the bar.  Any predictor that sees the item name is only useful if it
    beats the best predictor that sees nothing at all -- and on this corpus that
    blind predictor is far better than the 50 production currently uses, because
    the admitted scores cluster in the 70s, not around 50.  Reporting only
    "better than 50" would call a useless model a success.
    """
    total = sum(weights)
    best_c, best_mae = 0.0, float("inf")
    for candidate in sorted(set(scores)):
        mae = sum(w * abs(candidate - s) for s, w in zip(scores, weights)) / total
        if mae < best_mae:
            best_c, best_mae = candidate, mae
    return best_c, best_mae


def _report(judged: list[dict[str, Any]]) -> None:
    usable = [row for row in judged if row.get("jev_total") is not None]
    print(f"\n--- 実測 ({len(usable)}/{len(judged)} 件) ---")
    if not usable:
        print("有効な応答なし")
        return

    humans = [float(row["human_score"]) for row in usable]
    # Two weightings, because they answer different questions.  Unweighted asks
    # "does Jev understand each KIND of item"; weighted by n asks "how wrong is
    # it on the traffic actually seen", where one question can carry 56 rows.
    flat = [1.0] * len(usable)
    by_rows = [float(row.get("n") or 1) for row in usable]

    absolute = [abs(row["jev_total"] - row["human_score"]) for row in usable]
    errors = [row["jev_total"] - row["human_score"] for row in usable]
    jev_weighted = sum(w * e for w, e in zip(by_rows, absolute)) / sum(by_rows)

    const, const_mae = _best_constant(humans, flat)
    const_w, const_mae_w = _best_constant(humans, by_rows)
    default_mae = statistics.fmean(abs(DEFAULT_ASSET_SCORE - s) for s in humans)

    print(f"質問数 {len(usable)} / 元データ行 {int(sum(by_rows))}")
    print(f"MAE (Jev)            : {statistics.fmean(absolute):5.1f}   "
          f"[行数重み {jev_weighted:5.1f}]")
    print(f"MAE (最良定数={const:.0f})     : {const_mae:5.1f}   "
          f"[行数重み {const_mae_w:5.1f} / 定数={const_w:.0f}]  ← 超えるべき基準")
    print(f"MAE (現行の50固定)    : {default_mae:5.1f}   (参考。下回って当然)")
    print(f"バイアス (符号付き)   : {statistics.fmean(errors):+5.1f}")
    print(f"±10点以内            : {sum(1 for v in absolute if v <= 10)}/{len(usable)}")
    jev_mae = statistics.fmean(absolute)
    improvements = [
        abs(const - row["human_score"]) - abs(row["jev_total"] - row["human_score"])
        for row in usable
    ]
    if len(improvements) >= 2:
        critical = _student_t_95(len(improvements) - 1)
        uncertainty = critical * statistics.stdev(improvements) / math.sqrt(len(improvements))
    else:
        uncertainty = float("inf")

    if jev_mae >= const_mae:
        print("  ⚠ 最良定数に負けている。品名を読んでいる意味がない")
    elif len(usable) < 5 or statistics.fmean(improvements) <= uncertainty:
        margin = "算定不能" if not math.isfinite(uncertainty) else f"±{uncertainty:.1f}"
        print(
            f"  ⚠ 有効回答が {len(usable)} 問で、改善幅の95%誤差幅は {margin}。"
            "優位とは判定しない"
        )
    low = [item for row in usable for item in row["meta"].get("low_confidence_items", [])]
    if low:
        print(f"低確信の次元         : {dict(Counter(low))}")

    print("\n品名 / 人手 / Jev / 差")
    for row in usable:
        label = f"{row['name']} / {row.get('detail', '')}".rstrip(" /")
        print(
            f"  {label[:34]:<34} {row['human_score']:>5.0f} "
            f"{row['jev_total']:>5.0f} {row['jev_total'] - row['human_score']:>+6.1f}"
        )


def _sweep(path: Path) -> None:
    _report(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data" / "lease_data.db"
    )
    parser.add_argument("--seed", action="store_true", help="DB から初期エントリを作成")
    parser.add_argument("--inspect", action="store_true", help="送信ペイロードを全文表示")
    parser.add_argument("--send", action="store_true", help="TypeSafe に送信する")
    parser.add_argument(
        "--approval-token",
        help="直前の --inspect で表示されたペイロード固有の承認トークン",
    )
    parser.add_argument("--sweep", type=Path, help="保存済み結果を再集計")
    args = parser.parse_args()

    if args.sweep:
        _sweep(args.sweep)
        return 0

    if args.seed:
        written, skipped = seed_fixture(args.fixture, args.db)
        print(f"追記: {written} 問  (admit_row が棄却した行: {skipped} 件)")
        print(f"      {args.fixture}")

    rows = _load_fixture(args.fixture)
    if not rows:
        _print_corpus(rows)
        return 0

    problems = _validate(rows)
    if problems:
        print("\n--- フィクスチャの問題 ---")
        for problem in problems:
            print(f"  {problem}")
        return 1

    _print_corpus(rows)

    if args.inspect:
        _print_payloads(rows)

    if args.send:
        if not guard.typesafe_asset_enabled():
            print(
                "\nTYPESAFE_ASSET_ENABLED と API キーが未設定のため送信しない",
                file=sys.stderr,
            )
            return 1
        expected_token = _approval_token(rows)
        if args.approval_token != expected_token:
            print(
                "\n送信を停止: --inspect で内容を確認し、表示された承認トークンを "
                "--approval-token に指定してください",
                file=sys.stderr,
            )
            return 1
        print(f"\n{SEND_WARNING}")
        out = _send(rows)
        print(f"\n保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
