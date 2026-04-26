"""
tests/test_quantum_explainer.py
================================
EX.2: QuantumExplainer クラスのテスト
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_analysis_module import QuantumGate, _extract_features
from quantum_explainer import QuantumExplainer


def _gate_predict() -> dict:
    gate = QuantumGate()
    gate.fit([])
    return gate.predict({"inputs": {
        "op_profit": 50000, "depreciation": 15000, "machines": 40000,
        "net_income": 40000, "ord_profit": 45000,
        "grade": "②B格", "industry_major": "D 建設業",
        "qualitative": {"strength_tags": [], "onehot": {}},
    }})


# ── EX.2 DoD: test_explainer_sums_to_risk ──────────────────────────────────

def test_explainer_sums_to_risk():
    """shapley_contributions の合計が explained_risk と一致する（EX.2 DoD）"""
    result = _gate_predict()
    explainer = QuantumExplainer()
    contribs = explainer.shapley_contributions(result)

    assert isinstance(contribs, dict), "shapley_contributions は dict を返すこと"
    total = sum(contribs.values())
    assert total == pytest.approx(result["explained_risk"], abs=1e-3), (
        f"sum(shapley_contributions)={total:.6f} が "
        f"explained_risk={result['explained_risk']:.6f} と一致しない"
    )


def test_explainer_returns_non_negative():
    """各変数の寄与点数は 0 以上"""
    result = _gate_predict()
    explainer = QuantumExplainer()
    contribs = explainer.shapley_contributions(result)
    for var, val in contribs.items():
        assert val >= 0.0, f"{var} の寄与点数 {val} が負"


def test_explainer_keys_are_variable_names():
    """shapley_contributions のキーはペア名でなく変数名（_x_ を含まない）"""
    result = _gate_predict()
    explainer = QuantumExplainer()
    contribs = explainer.shapley_contributions(result)
    for key in contribs:
        assert "_x_" not in key, f"キー {key!r} にペア区切り _x_ が残っている"


def test_explainer_empty_predict():
    """pair_contributions が空のとき空 dict を返す"""
    explainer = QuantumExplainer()
    result = {"pair_contributions": {}, "explained_risk": 0.0}
    assert explainer.shapley_contributions(result) == {}


def test_explainer_instantiation_without_gate():
    """QuantumExplainer はパラメータなしで生成できる"""
    explainer = QuantumExplainer()
    assert explainer is not None


# ── EX.5: counterfactual ─────────────────────────────────────────────────────

def _risky_construction_case() -> dict:
    """建設業: 高利益×極低設備 → 高 Q_risk の基準ケース"""
    return {"inputs": {
        "op_profit": 80000, "depreciation": 10, "machines": 100,
        "net_income": 64000, "ord_profit": 72000,
        "grade": "②B格", "industry_major": "D 建設業",
        "qualitative": {"strength_tags": [], "onehot": {}},
    }}


def test_counterfactual_balanced_machines_lowers_risk():
    """EX.5 DoD: 極低設備建設業で machines を増やすと Q_risk が下がる"""
    gate = QuantumGate()
    gate.fit([])
    explainer = QuantumExplainer(gate=gate)

    result = explainer.counterfactual(
        _risky_construction_case(), var="machines", target_val=10000
    )

    assert result["delta"] < 0, (
        f"設備充実で Q_risk が下がるはず: delta={result['delta']:.4f}, "
        f"original={result['original_risk']}, cf={result['counterfactual_risk']}"
    )
    assert "caution" in result, "caution キーが存在すること"
    assert "数値操作推奨ではない" in result["caution"]


def test_counterfactual_return_structure():
    """counterfactual は必要なキーをすべて含む dict を返す"""
    gate = QuantumGate()
    gate.fit([])
    explainer = QuantumExplainer(gate=gate)
    result = explainer.counterfactual(
        _risky_construction_case(), var="machines", target_val=10000
    )
    for key in ("original_risk", "counterfactual_risk", "delta", "var", "target_val", "caution"):
        assert key in result, f"キー {key!r} が存在すること"
    assert result["var"] == "machines"
    assert result["target_val"] == 10000
    assert result["delta"] == pytest.approx(
        result["counterfactual_risk"] - result["original_risk"], abs=1e-6
    )


def test_counterfactual_no_change_gives_zero_delta():
    """変数を元の値のまま差し替えると delta ≈ 0"""
    gate = QuantumGate()
    gate.fit([])
    explainer = QuantumExplainer(gate=gate)
    case = _risky_construction_case()
    original_machines = case["inputs"]["machines"]
    result = explainer.counterfactual(case, var="machines", target_val=original_machines)
    assert abs(result["delta"]) < 1e-6


def test_counterfactual_requires_gate():
    """gate 未設定で counterfactual を呼ぶと ValueError"""
    explainer = QuantumExplainer()
    with pytest.raises(ValueError, match="gate"):
        explainer.counterfactual({"inputs": {}}, var="machines", target_val=1000)


def test_counterfactual_does_not_mutate_original():
    """counterfactual は元の case dict を変更しない"""
    gate = QuantumGate()
    gate.fit([])
    explainer = QuantumExplainer(gate=gate)
    case = _risky_construction_case()
    original_machines = case["inputs"]["machines"]
    explainer.counterfactual(case, var="machines", target_val=99999)
    assert case["inputs"]["machines"] == original_machines, "元の case が書き換わっている"


# ── EX.6: industry_percentile ───────────────────────────────────────────────

def _make_cases(var: str, values: list[float]) -> list[dict]:
    return [{"inputs": {var: v, "industry_major": "D 建設業"}} for v in values]


def test_industry_percentile_none_when_n_lt_5():
    """EX.6 DoD: n<5 のとき None を返す"""
    explainer = QuantumExplainer()
    for n in range(5):
        cases = _make_cases("op_profit", list(range(n)))
        assert explainer.industry_percentile("op_profit", 50.0, cases) is None, \
            f"n={n} で None のはず"


def test_industry_percentile_returns_float_in_range():
    """EX.6 DoD: n≥5 で 0〜100 の float を返す"""
    explainer = QuantumExplainer()
    cases = _make_cases("op_profit", [10, 20, 30, 40, 50])
    result = explainer.industry_percentile("op_profit", 30.0, cases)
    assert result is not None
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_industry_percentile_median_is_50():
    """中央値は分位数 50 を返す"""
    explainer = QuantumExplainer()
    # 奇数個で中央値が明確
    cases = _make_cases("op_profit", [10, 20, 30, 40, 50])
    result = explainer.industry_percentile("op_profit", 30.0, cases)
    assert result == pytest.approx(50.0, abs=1.0), f"中央値の分位数は ~50: {result}"


def test_industry_percentile_below_min_is_zero():
    """最小値未満は 0.0"""
    explainer = QuantumExplainer()
    cases = _make_cases("op_profit", [10, 20, 30, 40, 50])
    result = explainer.industry_percentile("op_profit", 5.0, cases)
    assert result == pytest.approx(0.0, abs=0.01)


def test_industry_percentile_above_max_is_100():
    """最大値超えは 100.0"""
    explainer = QuantumExplainer()
    cases = _make_cases("op_profit", [10, 20, 30, 40, 50])
    result = explainer.industry_percentile("op_profit", 999.0, cases)
    assert result == pytest.approx(100.0, abs=0.01)


def test_industry_percentile_missing_var_skipped():
    """var が存在しない case はスキップされる（n カウントに含まない）"""
    explainer = QuantumExplainer()
    # op_profit がない case 4 件 + ある case 1 件 → n=1 < 5 → None
    cases = [{"inputs": {"industry_major": "D 建設業"}}] * 4
    cases += [{"inputs": {"op_profit": 50.0}}]
    result = explainer.industry_percentile("op_profit", 50.0, cases)
    assert result is None


def test_industry_percentile_gate_not_required():
    """industry_percentile は gate なしで動作する"""
    explainer = QuantumExplainer()
    cases = _make_cases("op_profit", [10, 20, 30, 40, 50])
    result = explainer.industry_percentile("op_profit", 25.0, cases)
    assert result is not None


# ── EX.7: build_narrative ────────────────────────────────────────────────────

def _construction_predict():
    """建設業・低減価償却ケースの predict 結果"""
    gate = QuantumGate()
    gate.fit([])
    case = {"inputs": {
        "op_profit": 80000, "depreciation": 100, "machines": 40000,
        "net_income": 64000, "ord_profit": 72000,
        "grade": "②B格", "industry_major": "D 建設業",
        "qualitative": {"strength_tags": [], "onehot": {}},
    }}
    return gate.predict(case), case


def test_build_narrative_returns_nonempty_string():
    """build_narrative は非空の文字列を返す"""
    explainer = QuantumExplainer()
    result, case = _construction_predict()
    narrative = explainer.build_narrative(result, case=case)
    assert isinstance(narrative, str) and len(narrative) > 10


def test_build_narrative_contains_contribution_label():
    """EX.7 DoD: 出力に「寄与:」が含まれる"""
    explainer = QuantumExplainer()
    result, case = _construction_predict()
    narrative = explainer.build_narrative(result, case=case)
    assert "寄与:" in narrative, f"「寄与:」が見つからない: {narrative}"


def test_build_narrative_contains_industry_risk():
    """業種情報があれば「典型リスク」が含まれる"""
    explainer = QuantumExplainer()
    result, case = _construction_predict()
    narrative = explainer.build_narrative(result, case=case)
    assert "建設業典型リスク" in narrative, f"「建設業典型リスク」が見つからない: {narrative}"


def test_build_narrative_with_industry_cases_includes_deviation():
    """industry_cases を渡すと「業種平均より」が含まれる"""
    explainer = QuantumExplainer()
    result, case = _construction_predict()
    # 同業種 cases (depreciation が全部 15000)
    industry_cases = [
        {"inputs": {
            "op_profit": 50000, "depreciation": 15000, "machines": 40000,
            "net_income": 40000, "ord_profit": 45000,
            "industry_major": "D 建設業",
            "qualitative": {"strength_tags": [], "onehot": {}},
        }}
        for _ in range(7)
    ]
    narrative = explainer.build_narrative(result, case=case, industry_cases=industry_cases)
    assert "業種平均より" in narrative, f"「業種平均より」が見つからない: {narrative}"
    assert "参考値" in narrative


def test_build_narrative_zero_risk_graceful():
    """全寄与ゼロのケースでもクラッシュせず文字列を返す"""
    explainer = QuantumExplainer()
    zero_result = {
        "pair_contributions": {}, "explained_risk": 0.0,
        "entropy_risk": 0.0, "quantum_risk": 0.0,
    }
    narrative = explainer.build_narrative(zero_result)
    assert isinstance(narrative, str) and len(narrative) > 0


# ── EX.3: OOD 検出 ──────────────────────────────────────────────────────────

def _make_training_cases() -> list[dict]:
    """学習用: op_profit 45〜55M 付近の 7 件（適度な分散あり）"""
    return [
        {"inputs": {
            "op_profit": v, "depreciation": 15000, "machines": 40000,
            "net_income": int(v * 0.8), "ord_profit": int(v * 0.9),
            "grade": "②B格", "industry_major": "D 建設業",
            "qualitative": {"strength_tags": [], "onehot": {}},
        }}
        for v in [45000, 47000, 49000, 50000, 51000, 53000, 55000]
    ]


def _fitted_gate() -> QuantumGate:
    gate = QuantumGate()
    gate.fit(_make_training_cases())
    return gate


def test_ood_check_detects_extreme_value():
    """EX.3 DoD: fit 済みモデルで業種平均の10倍の入力は OOD=True"""
    gate = _fitted_gate()
    explainer = QuantumExplainer(gate=gate)

    extreme_inputs = {
        "op_profit": 500000,   # 10x typical (50000)
        "depreciation": 15000, "machines": 40000,
        "net_income": 40000, "ord_profit": 45000,
        "grade": "②B格", "industry_major": "D 建設業",
        "qualitative": {"strength_tags": [], "onehot": {}},
    }
    rec = _extract_features(extreme_inputs)
    assert explainer.ood_check(rec) is True, "業種平均の10倍は OOD フラグが立つこと"


def test_ood_check_normal_value_is_false():
    """通常範囲（学習分布内）の入力は OOD=False"""
    gate = _fitted_gate()
    explainer = QuantumExplainer(gate=gate)

    normal_inputs = {
        "op_profit": 52000,  # training range 内
        "depreciation": 15000, "machines": 40000,
        "net_income": 40000, "ord_profit": 45000,
        "grade": "②B格", "industry_major": "D 建設業",
        "qualitative": {"strength_tags": [], "onehot": {}},
    }
    rec = _extract_features(normal_inputs)
    assert explainer.ood_check(rec) is False, "通常範囲は OOD フラグが立たないこと"


def test_ood_check_requires_gate():
    """gate 未設定で ood_check を呼ぶと ValueError"""
    explainer = QuantumExplainer()
    with pytest.raises(ValueError, match="gate"):
        explainer.ood_check({"op_profit": 50.0})
