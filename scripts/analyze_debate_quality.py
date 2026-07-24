#!/usr/bin/env python3
"""マルチエージェント討論の品質を計測して弱点を可視化する（読み取り専用・分析）。

入力: data/multi_agent_debate_metrics.jsonl（api/multi_agent_screening._log_debate_metrics が追記）
      1行 = {ts, score, mode("solo"|"debate"), final, opinions{skeptic,optimist,innovator},
             same_opinion_r1, same_opinion_r2, duration_sec}

品質を5軸で集計する（審査ロジックには一切触れない）:
  ① 多様性: 討論で3ペルソナが本当に意見を割っているか（全員一致ばかりは"討論ごっこ"）
  ② 付加価値: 意見が割れて討論が働いた割合 / 全員一致を裁定が追認しただけの割合（rubber-stamp）
  ③ 少数意見の反映: 誰かが「否決」と警告したのに最終「承認」になった割合（リスク見落としの兆候）
  ④ 正しさ: 討論の判定が実際の成約/失注と合っているか（final_status 突合が必要 → 本スクリプトでは N/A）
  ⑤ 健全性: 所要時間の分布（遅延の兆候）

使い方:
  python3 scripts/analyze_debate_quality.py
  python3 scripts/analyze_debate_quality.py --path data/multi_agent_debate_metrics.jsonl --json reports/debate_quality_latest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT_PATH = "data/multi_agent_debate_metrics.jsonl"
_VALID_OPINIONS = {"承認", "否決", "条件付承認"}

# 弱点フラグの目安（ヒューリスティック初期値。運用データを見て調整する）。
_TH_LOW_DIVERSITY = 0.30     # 討論での意見不一致率がこれ未満 → 多様性が低い
_TH_RUBBER_STAMP = 0.30      # 全員一致を追認しただけの割合がこれ超 → 付加価値が低い
_TH_RISK_OVERRIDE = 0.10     # 否決警告を承認で上書きした割合がこれ超 → リスク見落とし注意
_TH_SLOW_P95_SEC = 45.0      # 討論のp95所要がこれ超 → 健全性注意


def load_metrics(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rate(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def _percentile(values: list[float], q: float) -> float:
    """簡易パーセンタイル（numpy非依存）。q は 0..1。"""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return round(float(vals[0]), 1)
    idx = q * (len(vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return round(float(vals[lo] * (1 - frac) + vals[hi] * frac), 1)


def _distinct_opinions(entry: dict) -> int:
    ops = entry.get("opinions") or {}
    return len({v for v in ops.values() if v in _VALID_OPINIONS})


def analyze(entries: list[dict]) -> dict[str, Any]:
    """メトリクスから5軸のKPIを集計して返す（純粋関数）。"""
    debate = [e for e in entries if e.get("mode") == "debate"]
    solo = [e for e in entries if e.get("mode") == "solo"]
    n_debate = len(debate)

    # ① 多様性
    r1_flags = [e.get("same_opinion_r1") for e in debate if e.get("same_opinion_r1") is not None]
    disagree_r1 = sum(1 for f in r1_flags if f is False)
    disagree_r1_rate = _rate(disagree_r1, len(r1_flags))
    # 不一致だったもののうち r2 で収束した割合
    disagreed = [e for e in debate if e.get("same_opinion_r1") is False]
    converged = sum(1 for e in disagreed if e.get("same_opinion_r2") is True)
    convergence_rate = _rate(converged, len(disagreed))
    unanimous_opinions = sum(1 for e in debate if _distinct_opinions(e) <= 1 and (e.get("opinions") or {}))

    # ② 付加価値
    split = sum(1 for e in debate if _distinct_opinions(e) >= 2)
    split_rate = _rate(split, n_debate)
    rubber_stamp = 0
    for e in debate:
        ops = [v for v in (e.get("opinions") or {}).values() if v in _VALID_OPINIONS]
        if ops and len(set(ops)) == 1 and e.get("final") == ops[0]:
            rubber_stamp += 1
    rubber_stamp_rate = _rate(rubber_stamp, n_debate)

    # ③ 少数意見の反映（リスク）
    reject_but_approved = sum(
        1 for e in debate
        if e.get("final") == "承認" and "否決" in (e.get("opinions") or {}).values()
    )
    reject_but_approved_rate = _rate(reject_but_approved, n_debate)
    skeptic_reject_overridden = sum(
        1 for e in debate
        if (e.get("opinions") or {}).get("skeptic") == "否決" and e.get("final") == "承認"
    )
    skeptic_reject_overridden_rate = _rate(skeptic_reject_overridden, n_debate)

    # ⑤ 健全性
    debate_durs = [e.get("duration_sec") for e in debate if isinstance(e.get("duration_sec"), (int, float))]
    solo_durs = [e.get("duration_sec") for e in solo if isinstance(e.get("duration_sec"), (int, float))]

    # 弱点フラグ
    flags: list[str] = []
    if r1_flags and disagree_r1_rate < _TH_LOW_DIVERSITY:
        flags.append(f"① 多様性が低い（意見不一致率 {disagree_r1_rate:.0%} < {_TH_LOW_DIVERSITY:.0%}）")
    if n_debate and rubber_stamp_rate > _TH_RUBBER_STAMP:
        flags.append(f"② 付加価値が低い（全員一致の追認 {rubber_stamp_rate:.0%} > {_TH_RUBBER_STAMP:.0%}）")
    if n_debate and reject_but_approved_rate > _TH_RISK_OVERRIDE:
        flags.append(f"③ リスク見落とし注意（否決→承認 {reject_but_approved_rate:.0%} > {_TH_RISK_OVERRIDE:.0%}）")
    if debate_durs and _percentile(debate_durs, 0.95) > _TH_SLOW_P95_SEC:
        flags.append(f"⑤ 遅延注意（討論p95 {_percentile(debate_durs, 0.95)}s > {_TH_SLOW_P95_SEC}s）")

    return {
        "totals": {"all": len(entries), "solo": len(solo), "debate": n_debate},
        "diversity": {
            "disagree_r1_rate": disagree_r1_rate,
            "convergence_rate_after_disagree": convergence_rate,
            "unanimous_opinion_debates": unanimous_opinions,
        },
        "added_value": {
            "split_rate": split_rate,
            "rubber_stamp_rate": rubber_stamp_rate,
        },
        "minority_reflection": {
            "reject_but_approved_rate": reject_but_approved_rate,
            "skeptic_reject_overridden_rate": skeptic_reject_overridden_rate,
        },
        "correctness": {
            "note": "実成約/失注（final_status）との突合が必要。本スクリプト単体では N/A。",
        },
        "health": {
            "debate_duration_p50": _percentile(debate_durs, 0.50),
            "debate_duration_p95": _percentile(debate_durs, 0.95),
            "solo_duration_p50": _percentile(solo_durs, 0.50),
        },
        "flags": flags,
    }


def format_report(kpis: dict[str, Any]) -> str:
    t = kpis["totals"]
    d = kpis["diversity"]
    a = kpis["added_value"]
    m = kpis["minority_reflection"]
    h = kpis["health"]
    lines = [
        "=== 討論品質レポート ===",
        f"件数: 全{t['all']} / solo(ファストパス){t['solo']} / debate(討論){t['debate']}",
        "",
        "① 多様性",
        f"   意見不一致率(R1): {d['disagree_r1_rate']:.0%}  / 不一致後の収束率(R2): {d['convergence_rate_after_disagree']:.0%}",
        f"   全員一致だった討論: {d['unanimous_opinion_debates']} 件",
        "② 付加価値",
        f"   意見が割れた割合: {a['split_rate']:.0%}  / 全員一致を追認しただけ(rubber-stamp): {a['rubber_stamp_rate']:.0%}",
        "③ 少数意見の反映（リスク）",
        f"   誰かが否決→最終承認: {m['reject_but_approved_rate']:.0%}  / 懐疑派否決→最終承認: {m['skeptic_reject_overridden_rate']:.0%}",
        "④ 正しさ",
        f"   {kpis['correctness']['note']}",
        "⑤ 健全性",
        f"   討論 所要 p50/p95: {h['debate_duration_p50']}s / {h['debate_duration_p95']}s  (solo p50: {h['solo_duration_p50']}s)",
        "",
    ]
    if kpis["flags"]:
        lines.append("⚠ 注目点（弱点候補）:")
        lines.extend(f"   - {f}" for f in kpis["flags"])
    else:
        lines.append("✅ 目安閾値では大きな弱点フラグなし（データ量が少ない場合は解釈注意）")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=_DEFAULT_PATH, help="メトリクスJSONLのパス")
    ap.add_argument("--json", default="", help="KPIをJSONで書き出すパス（任意）")
    args = ap.parse_args()

    entries = load_metrics(args.path)
    if not entries:
        print(f"メトリクスが空/未存在です: {args.path}")
        print("（討論を数回実行して data/multi_agent_debate_metrics.jsonl を蓄積してから再実行してください）")
        return 0

    kpis = analyze(entries)
    print(format_report(kpis))

    if args.json:
        outp = Path(args.json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(kpis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nKPIを書き出しました: {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
