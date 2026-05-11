"""係数定義の回帰テスト。"""
import ast
from pathlib import Path


def _load_coeffs() -> dict:
    tree = ast.parse(Path("coeff_definitions.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "COEFFS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("COEFFS が見つかりません")


def test_main_existing_coefficients_are_not_overly_optimistic():
    """既存先は取引実績を評価しつつ、旧係数のようなスコア飽和を避ける。"""
    coeffs = _load_coeffs()["全体_既存先"]

    assert 0 < coeffs["lease_credit_log"] <= 0.35
    assert 0 < coeffs["contracts"] <= 0.08
    assert 0 < coeffs["grade_4_6"] <= 0.35


def test_main_new_customer_coefficients_keep_relationship_signals():
    """新規先でも銀行与信・限定的な取引実績・標準格付をゼロ扱いしない。"""
    coeffs = _load_coeffs()["全体_新規先"]

    assert coeffs["bank_credit_log"] > 0
    assert coeffs["lease_credit_log"] > 0
    assert coeffs["contracts"] > 0
    assert coeffs["grade_4_6"] > 0
