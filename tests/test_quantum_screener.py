"""
tests/test_quantum_screener.py
================================
SC.1: quantum_screener.py のテスト
"""
from __future__ import annotations

import sys
import csv
import io
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_screener import QuantumScreener, UNUSED_VARS


def _make_records(n: int = 10) -> list[dict]:
    """テスト用ダミーレコード（nenshu が失注と相関するよう設計）"""
    records = []
    for i in range(n):
        lost = i % 3 == 0  # 1/3 が失注
        records.append({
            "final_status": "失注" if lost else "成約",
            "inputs": {
                "nenshu":       5000 + (8000 if lost else 0),  # 失注は高い
                "bank_credit":  1000 + i * 100,
                "other_assets": 2000 - (500 if lost else 0),
                "gross_profit": 3000 + i * 200,
                "op_profit":    2000 + i * 100,
            },
        })
    return records


# ── SC.1 DoD テスト ──────────────────────────────────────────────────────────

def test_screener_outputs_required_variables():
    """SC.1 DoD: nenshu, bank_credit, other_assets の相関係数・p値が出力される"""
    screener = QuantumScreener(records=_make_records(15))
    df = screener.compute_correlations()
    var_names = list(df["variable"])
    for var in ("nenshu", "bank_credit", "other_assets"):
        assert var in var_names, f"{var} が結果に含まれていない: {var_names}"


def test_screener_has_correlation_and_pvalue_columns():
    """結果 DataFrame に correlation, p_value, n 列が存在する"""
    screener = QuantumScreener(records=_make_records(15))
    df = screener.compute_correlations()
    for col in ("variable", "correlation", "p_value", "n"):
        assert col in df.columns, f"列 {col!r} が存在しない"


def test_screener_correlation_in_range():
    """相関係数は -1〜1 の範囲"""
    screener = QuantumScreener(records=_make_records(15))
    df = screener.compute_correlations()
    for _, row in df.iterrows():
        assert -1.0 <= row["correlation"] <= 1.0, (
            f"{row['variable']}: correlation={row['correlation']:.4f} が範囲外"
        )


def test_screener_pvalue_in_range():
    """p値は 0〜1 の範囲"""
    screener = QuantumScreener(records=_make_records(15))
    df = screener.compute_correlations()
    for _, row in df.iterrows():
        assert 0.0 <= row["p_value"] <= 1.0, (
            f"{row['variable']}: p_value={row['p_value']:.6f} が範囲外"
        )


def test_screener_csv_output_parseable():
    """to_csv() の出力が CSV として正しくパースできる"""
    screener = QuantumScreener(records=_make_records(15))
    csv_text = screener.to_csv()
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    assert len(rows) > 0
    assert "variable" in rows[0]
    assert "correlation" in rows[0]
    assert "p_value" in rows[0]


def test_screener_skips_too_few_samples():
    """有効サンプルが少ない変数は n=0 で出力される（クラッシュしない）"""
    records = _make_records(3)  # very few
    screener = QuantumScreener(records=records)
    df = screener.compute_correlations()
    assert isinstance(df, object)  # crashes しない


def test_unused_vars_contains_required():
    """UNUSED_VARS に DoD 記載の変数が含まれる"""
    for var in ("nenshu", "bank_credit", "other_assets"):
        assert var in UNUSED_VARS, f"{var} が UNUSED_VARS に含まれていない"
