"""討論品質分析（scripts/analyze_debate_quality.py）のKPI計算テスト。

合成メトリクスで、多様性・付加価値・少数意見反映・健全性の各KPIが正しく
計算されることを保証する（stdlib のみ・CIで実行可能）。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parent.parent / "scripts" / "analyze_debate_quality.py"
_spec = importlib.util.spec_from_file_location("analyze_debate_quality", _MOD_PATH)
adq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adq)


def _debate(opinions, final, r1, r2=True, dur=10.0):
    return {
        "mode": "debate", "final": final, "opinions": opinions,
        "same_opinion_r1": r1, "same_opinion_r2": r2, "duration_sec": dur,
    }


def test_totals_and_solo_debate_split():
    entries = [
        {"mode": "solo", "final": "承認", "duration_sec": 2.0},
        _debate({"skeptic": "否決", "optimist": "承認", "innovator": "条件付承認"}, "条件付承認", r1=False),
    ]
    k = adq.analyze(entries)
    assert k["totals"] == {"all": 2, "solo": 1, "debate": 1}


def test_diversity_disagreement_rate():
    entries = [
        _debate({"skeptic": "否決", "optimist": "承認"}, "条件付承認", r1=False),  # 不一致
        _debate({"skeptic": "承認", "optimist": "承認"}, "承認", r1=True),          # 一致
    ]
    k = adq.analyze(entries)
    assert k["diversity"]["disagree_r1_rate"] == 0.5


def test_rubber_stamp_detection():
    # 全員一致(条件付承認)を最終も追認 → rubber-stamp
    entries = [
        _debate({"skeptic": "条件付承認", "optimist": "条件付承認", "innovator": "条件付承認"},
                "条件付承認", r1=True),
    ]
    k = adq.analyze(entries)
    assert k["added_value"]["rubber_stamp_rate"] == 1.0
    assert k["added_value"]["split_rate"] == 0.0


def test_minority_reject_overridden_flagged():
    # 懐疑派が否決なのに最終承認 → リスク兆候
    entries = [
        _debate({"skeptic": "否決", "optimist": "承認"}, "承認", r1=False),
    ]
    k = adq.analyze(entries)
    assert k["minority_reflection"]["reject_but_approved_rate"] == 1.0
    assert k["minority_reflection"]["skeptic_reject_overridden_rate"] == 1.0
    assert any("リスク見落とし" in f for f in k["flags"])


def test_health_percentiles():
    entries = [_debate({"skeptic": "承認", "optimist": "承認"}, "承認", r1=True, dur=float(d))
               for d in (10, 20, 30, 40, 50)]
    k = adq.analyze(entries)
    assert k["health"]["debate_duration_p50"] == 30.0
    assert k["health"]["debate_duration_p95"] >= 40.0


def test_empty_is_safe():
    k = adq.analyze([])
    assert k["totals"]["debate"] == 0
    assert k["flags"] == []
    # レポート整形も落ちない
    assert "討論品質レポート" in adq.format_report(k)


def test_load_metrics_skips_bad_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps(_debate({"skeptic": "否決", "optimist": "承認"}, "条件付承認", r1=False), ensure_ascii=False)
        + "\n{ broken json\n\n",
        encoding="utf-8",
    )
    rows = adq.load_metrics(p)
    assert len(rows) == 1
