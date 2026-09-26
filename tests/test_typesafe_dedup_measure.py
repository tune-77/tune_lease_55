from __future__ import annotations

import json

import pytest

from experiments.typesafe_dedup import measure


def _candidates():
    return [
        {"title": "表示ラベル修正", "reason": "誤字"},
        {"title": "UI文言改善", "reason": "説明不足"},
        {"title": "A社の案件修正", "reason": "個別案件"},
    ]


def test_send_filters_private_pairs_before_request(monkeypatch, tmp_path, capsys):
    candidates = _candidates()
    monkeypatch.setattr(measure, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(measure, "gray_pairs_for", lambda _analysis, _width: [(0, 1), (0, 2)])
    captured = {}

    def judge(rows, pairs):
        captured["rows"] = rows
        captured["pairs"] = pairs
        return ([{"a": 0, "b": 1, "same_issue": 0.8, "route": "duplicate"}], {"status": "applied"})

    monkeypatch.setattr(measure.guard, "judge_pairs", judge)

    output = measure._send(candidates, {}, 0.25)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert captured["pairs"] == [(0, 1)]
    assert saved["meta"]["privacy_skipped_pairs"] == 1
    assert "送信除外: 1 ペア" in capsys.readouterr().out


def test_send_stops_when_every_pair_is_private(monkeypatch):
    candidates = _candidates()
    monkeypatch.setattr(measure, "gray_pairs_for", lambda _analysis, _width: [(0, 2)])

    with pytest.raises(SystemExit, match="送信可能ペアがない"):
        measure._send(candidates, {}, 0.25)
