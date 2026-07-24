#!/usr/bin/env python3
"""merged PR から needs_review を自動解消する reconciler（cleanup_improvement_reviews を補完）。

cleanup_improvement_reviews.py は手動カタログ（KNOWN_CODE_APPLIED / KNOWN_TITLE_APPLIED）で
「出荷済みなのに needs_review で滞留する」項目を閉じるが、カタログ保守が要る。
本スクリプトは runtime 台帳の未決着エントリを PR 履歴から自動照合して滞留(churn)を減らす。

2 経路:
  経路1 REV番号リンク（確実・--apply で自動適用）:
    台帳エントリの rev_id が merged PR タイトルに含まれる → その実キーのまま applied 追記。
    「REV-162 が PR#485 でマージ済みなのに台帳は needs_review」のような滞留を確実に解消する。
    誤クローズが起きない（実際にその REV を出荷した PR がある場合のみ）。
  経路2 タイトル類似（提案のみ・人手ゲート）:
    rev_id を持たない生メモは merged PR タイトルとの文字bigram類似で候補を提示する。
    --apply では一切書き込まない（曖昧一致で真の未着手を誤って閉じないため）。

dry-run 既定。--apply は経路1のみ書き込む。
台帳パス: ~/Library/Logs/tunelease/ledger.jsonl（環境変数 LEDGER_PATH で上書き可）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# REV 抽出は cleanup 側の実装を単一ソースとして再利用する。
from cleanup_improvement_reviews import _extract_revs  # noqa: E402

LEDGER_PATH = Path(
    os.environ.get(
        "LEDGER_PATH",
        str(Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"),
    )
)

_OPEN_STATUSES = {"needs_review", "parked", "suppressed"}

# 経路2（提案）の表示フィルタ閾値。--apply には影響しない（経路2は書き込まない）。
DEFAULT_SIM_THRESHOLD = 0.45


# ── 類似度（言語非依存の文字bigram Jaccard）─────────────────────────────────

def _char_bigrams(text: str) -> set[str]:
    """英数字・かな漢字のみ残した文字列の隣接2文字集合を返す。"""
    s = "".join(ch for ch in str(text).lower() if ch.isalnum())
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def bigram_similarity(a: str, b: str) -> float:
    """2 文字列の文字bigram Jaccard 類似度（0.0〜1.0）。"""
    A, B = _char_bigrams(a), _char_bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def best_pr_match(title: str, pr_titles: list[str]) -> tuple[str, float]:
    """title に最も似た PR タイトルと類似スコアを返す。"""
    best_t, best_s = "", 0.0
    for t in pr_titles:
        s = bigram_similarity(title, t)
        if s > best_s:
            best_t, best_s = t, s
    return best_t, best_s


# ── コア（純粋関数・テスト対象）────────────────────────────────────────────

def reconcile(
    open_entries: list[dict],
    merged_revs: set[str],
    merged_pr_titles: list[str],
    now: str,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
) -> dict[str, list[dict]]:
    """未決着エントリを 経路1（REV番号リンク）/ 経路2（類似提案）/ 出荷痕跡なし に分類する。

    経路1 のみ applied 追記用の更新エントリを生成する（実キーを再利用）。副作用なし。
    """
    rev_linked: list[dict] = []
    suggested: list[dict] = []
    no_trace: list[dict] = []

    for e in open_entries:
        key = e.get("key") or ""
        if not key:
            continue
        rid = (e.get("rev_id") or "").upper()
        title = e.get("title") or ""
        status = e.get("status") or ""

        # 経路1: rev_id が merged PR に含まれる（確実）。
        if rid and rid in merged_revs:
            rev_linked.append({
                "key": key,
                "rev_id": rid,
                "status": "applied",
                "title": title,
                "canonical_key": key,
                "pr_url": "",
                "reason": f"merged PR に {rid} 含有 → applied（旧status={status}, rev_id一致で解決）",
                "recorded_at": now,
            })
            continue

        # 経路2: タイトル類似（提案のみ、--apply では触らない）。
        match_title, score = best_pr_match(title, merged_pr_titles)
        if score >= sim_threshold:
            suggested.append({
                "key": key,
                "rev_id": rid,
                "title": title,
                "status": status,
                "match": match_title,
                "score": round(score, 3),
            })
        else:
            no_trace.append({
                "key": key,
                "rev_id": rid,
                "title": title,
                "status": status,
                "best_score": round(score, 3),
            })

    return {"rev_linked": rev_linked, "suggested": suggested, "no_trace": no_trace}


# ── I/O ─────────────────────────────────────────────────────────────────────

def load_open_entries() -> list[dict]:
    """runtime 台帳から未決着（open）エントリを key ごと最新で返す。"""
    if not LEDGER_PATH.exists():
        return []
    latest: dict[str, dict] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = e.get("key", "")
        if k:
            latest[k] = e  # last wins
    return [
        {
            "key": k,
            "rev_id": e.get("rev_id", ""),
            "title": e.get("title", ""),
            "status": e.get("status", ""),
        }
        for k, e in latest.items()
        if e.get("status") in _OPEN_STATUSES
    ]


def fetch_merged_prs() -> tuple[set[str], list[str]]:
    """gh から merged PR を取得し (REV集合, タイトル一覧) を返す。"""
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "merged", "--limit", "400",
             "--json", "number,title"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"エラー: gh コマンド失敗: {e}", file=sys.stderr)
        sys.exit(1)
    prs = json.loads(result.stdout)
    titles = [p["title"] for p in prs]
    revs: set[str] = set()
    for p in prs:
        for num in _extract_revs(p["title"]):
            revs.add(f"REV-{num:03d}")
    return revs, titles


def _append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true",
                        help="経路1（REV番号リンク）のみ台帳へ applied 追記する")
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                        help="経路2（提案）の表示類似度しきい値")
    args = parser.parse_args()

    open_entries = load_open_entries()
    if not open_entries:
        print("未決着（needs_review/parked/suppressed）エントリなし")
        return

    print("GitHub から merged PR を取得中...")
    merged_revs, merged_titles = fetch_merged_prs()
    now = datetime.now().isoformat()
    res = reconcile(open_entries, merged_revs, merged_titles, now, args.sim_threshold)

    print(f"\n未決着エントリ: {len(open_entries)} 件")
    print(f"\n=== 経路1 REV番号リンク（{'適用' if args.apply else 'DRY-RUN'}）: {len(res['rev_linked'])} 件 ===")
    for u in res["rev_linked"]:
        print(f"  {u['rev_id']} {u['title'][:44]}  (key={u['key']})")
    print(f"\n=== 経路2 タイトル類似・要人手確認: {len(res['suggested'])} 件 ===")
    for s in sorted(res["suggested"], key=lambda x: -x["score"]):
        print(f"  score={s['score']}  {s['title'][:34]}  ← {s['match'][:42]}")
    print(f"\n=== 出荷痕跡なし（真の未着手/要判断）: {len(res['no_trace'])} 件 ===")
    for n in res["no_trace"]:
        print(f"  {n['title'][:48]}")

    if args.apply:
        for u in res["rev_linked"]:
            _append_ledger(u)
        print(f"\n経路1の {len(res['rev_linked'])} 件を applied 追記しました: {LEDGER_PATH}")
    else:
        print("\n--apply で経路1のみ台帳へ書き込みます（経路2は常に人手確認・自動適用しません）。")


if __name__ == "__main__":
    main()
