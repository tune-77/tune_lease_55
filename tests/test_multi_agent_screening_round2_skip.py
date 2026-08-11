import importlib

screening = importlib.import_module("api.multi_agent_screening")


def _boundary_params(score: float = 55) -> dict:
    return {
        "score": score,
        "company_name": "テスト株式会社",
        "industry_major": "製造業",
        "asset_name": "工作機械",
    }


def _fake_persona_call(system, prompt, temperature, kb, kb_mode, deadline=None, max_tokens=1024):
    if "懐疑派" in system:
        return {"opinion": "承認", "reason_codes": ["finance"], "notes": []}
    if "楽観派" in system:
        return {"opinion": "承認", "reason_codes": ["finance"], "notes": []}
    return {"final": "承認", "reason_codes": ["finance"], "condition_codes": [], "notes": []}


def test_round2_skipped_when_round1_participants_fully_agree(monkeypatch):
    calls = {"n": 0}

    def counting_persona_call(system, prompt, temperature, kb, kb_mode, deadline=None, max_tokens=1024):
        calls["n"] += 1
        return _fake_persona_call(system, prompt, temperature, kb, kb_mode, deadline, max_tokens)

    monkeypatch.setattr(screening, "_persona_call", counting_persona_call)
    monkeypatch.setattr(screening, "_log_debate_metrics", lambda *a, **k: None)

    result = screening.run_debate_screening(_boundary_params())

    assert result["mode"] == "debate"
    assert result["round2_skipped"] is True
    assert result["same_opinion_r2"] is None
    assert "省略しました" in result["debate_log"]
    # 第1ラウンド（懐疑派・楽観派）＋裁定役の3回のみ。第2ラウンドのLLM呼び出しは発生しない。
    assert calls["n"] == 3


def test_round2_runs_when_round1_participants_disagree(monkeypatch):
    calls = {"n": 0}

    def disagreeing_persona_call(system, prompt, temperature, kb, kb_mode, deadline=None, max_tokens=1024):
        calls["n"] += 1
        if "懐疑派" in system:
            return {"opinion": "否決", "reason_codes": ["finance"], "notes": []}
        if "楽観派" in system:
            return {"opinion": "承認", "reason_codes": ["finance"], "notes": []}
        return {"final": "条件付承認", "reason_codes": ["finance"], "condition_codes": ["guarantee"], "notes": []}

    monkeypatch.setattr(screening, "_persona_call", disagreeing_persona_call)
    monkeypatch.setattr(screening, "_log_debate_metrics", lambda *a, **k: None)

    result = screening.run_debate_screening(_boundary_params())

    assert result["round2_skipped"] is False
    assert result["same_opinion_r2"] is not None
    assert "強制反論" in result["debate_log"]
    # 第1ラウンド2回＋第2ラウンド2回＋裁定役1回＝5回
    assert calls["n"] == 5
