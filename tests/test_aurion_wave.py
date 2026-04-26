"""AV.3 — aurion_wave_engine + aurion_phase_mapper 単体テスト"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aurion_wave_engine import WaveResult, compute_wave
from aurion_phase_mapper import PhaseMapper


# ── 共鳴・デコヒーレンス境界値 ──────────────────────────────────────────────────

def test_perfect_coherence():
    """v1 == v2 → amplitude_score == 1.0"""
    r = compute_wave(0.5, 0.5)
    assert r.amplitude_score == pytest.approx(1.0, abs=1e-9)


def test_perfect_decoherence_score():
    """v1=0, v2=1 → 位相差=π → amplitude_score == 0.0"""
    r = compute_wave(0.0, 1.0)
    assert r.amplitude_score == pytest.approx(0.0, abs=1e-9)


def test_perfect_decoherence_phase_diff():
    """v1=0, v2=1 → phase_diff == π"""
    r = compute_wave(0.0, 1.0)
    assert r.phase_diff == pytest.approx(math.pi, abs=1e-9)


def test_phase_diff_pi_composite_rms_zero():
    """位相差 π のとき合成波の RMS ≈ 0 (波が打ち消し合う)"""
    r = compute_wave(0.0, 1.0, n_points=2048)
    rms = float(np.sqrt(np.mean(r.composite ** 2)))
    assert rms < 1e-9


def test_amplitude_score_midpoint():
    """v1=0, v2=0.5 → 位相差=π/2 → amplitude_score==0.5"""
    r = compute_wave(0.0, 0.5)
    assert r.amplitude_score == pytest.approx(0.5, abs=1e-9)


# ── スコア範囲 ────────────────────────────────────────────────────────────────

def test_amplitude_score_in_range():
    """任意の v1, v2 で amplitude_score ∈ [0, 1]"""
    cases = [(0.0, 0.5), (0.3, 0.8), (1.0, 1.0), (0.0, 0.0), (0.1, 0.9)]
    for v1, v2 in cases:
        r = compute_wave(v1, v2)
        assert 0.0 <= r.amplitude_score <= 1.0, f"out of range for v1={v1}, v2={v2}"


def test_out_of_range_clipped():
    """[0,1] 外の入力はクリップされスコアが [0,1] 内に収まる"""
    r = compute_wave(-0.5, 1.5)
    assert 0.0 <= r.amplitude_score <= 1.0


# ── 対称性 ───────────────────────────────────────────────────────────────────

def test_symmetry():
    """compute_wave(v1, v2) と compute_wave(v2, v1) は同じ amplitude_score / phase_diff"""
    r1 = compute_wave(0.3, 0.7)
    r2 = compute_wave(0.7, 0.3)
    assert r1.amplitude_score == pytest.approx(r2.amplitude_score, abs=1e-12)
    assert r1.phase_diff == pytest.approx(r2.phase_diff, abs=1e-12)


# ── 波形 shape ────────────────────────────────────────────────────────────────

def test_wave_shape_default():
    """デフォルト (n_points=256) で各配列が shape (256,)"""
    r = compute_wave(0.2, 0.7)
    assert r.wave_a.shape == (256,)
    assert r.wave_b.shape == (256,)
    assert r.composite.shape == (256,)


def test_wave_shape_custom_t():
    """カスタム t を渡した場合も shape が一致する"""
    t = np.linspace(0, 2, 400)
    r = compute_wave(0.2, 0.7, t=t)
    assert r.wave_a.shape == (400,)
    assert r.composite.shape == (400,)


def test_composite_is_mean_of_a_and_b():
    """composite == (wave_a + wave_b) / 2 が全点で成立する"""
    r = compute_wave(0.3, 0.6)
    np.testing.assert_allclose(r.composite, (r.wave_a + r.wave_b) / 2.0, atol=1e-12)


def test_wave_a_b_amplitude_bounded():
    """wave_a, wave_b が [-1, 1] に収まる (sin の性質)"""
    r = compute_wave(0.4, 0.8, n_points=512)
    assert r.wave_a.max() <= 1.0 + 1e-9
    assert r.wave_a.min() >= -1.0 - 1e-9
    assert r.wave_b.max() <= 1.0 + 1e-9
    assert r.wave_b.min() >= -1.0 - 1e-9


# ── PhaseMapper: get_pair ────────────────────────────────────────────────────

def test_get_pair_unknown_returns_base_default():
    """未登録コード 'C'(鉱業) → BASE_PAIRS 先頭 ('op_profit','depreciation')"""
    assert PhaseMapper.get_pair("C") == ("op_profit", "depreciation")


def test_get_pair_empty_returns_base_default():
    """空文字でも BASE_PAIRS 先頭を返す"""
    assert PhaseMapper.get_pair("") == ("op_profit", "depreciation")


def test_get_pair_construction():
    """D(建設業) → industry map の最高重みペアを返す"""
    a, b = PhaseMapper.get_pair("D")
    assert isinstance(a, str) and isinstance(b, str)
    assert a != b


def test_get_pair_transport():
    """H(運輸業) → industry map ペアを返す"""
    a, b = PhaseMapper.get_pair("H")
    assert isinstance(a, str) and isinstance(b, str)


def test_get_pair_medical():
    """P(医療・福祉) → industry map ペアを返す"""
    a, b = PhaseMapper.get_pair("P")
    assert isinstance(a, str) and isinstance(b, str)


def test_get_pair_manufacturing():
    """E(製造業) → industry map ペアを返す"""
    a, b = PhaseMapper.get_pair("E")
    assert isinstance(a, str) and isinstance(b, str)


def test_get_pair_rental():
    """K(不動産・物品賃貸) → industry map ペアを返す"""
    a, b = PhaseMapper.get_pair("K")
    assert isinstance(a, str) and isinstance(b, str)


def test_get_pair_lowercase_accepted():
    """小文字コードも受け付ける"""
    assert PhaseMapper.get_pair("d") == PhaseMapper.get_pair("D")


# ── PhaseMapper: normalize ───────────────────────────────────────────────────

def test_normalize_zero_returns_zero():
    """0 → 0.0"""
    assert PhaseMapper.normalize("op_profit", 0.0) == pytest.approx(0.0, abs=1e-9)


def test_normalize_range():
    """任意の正の値が [0, 1) に収まる"""
    for var, val in [
        ("op_profit", 5000.0),
        ("depreciation", 3000.0),
        ("machines", 100000.0),
        ("net_income", 0.0),
        ("trend_val", 5.0),
    ]:
        result = PhaseMapper.normalize(var, val)
        assert 0.0 <= result < 1.0, f"out of range for {var}={val}: {result}"


def test_normalize_negative_clipped():
    """負値は 0 に相当する (tanh(0) = 0)"""
    assert PhaseMapper.normalize("op_profit", -5000.0) == pytest.approx(0.0, abs=1e-9)


def test_normalize_monotone():
    """大きい値 → 大きい正規化値"""
    v_low = PhaseMapper.normalize("op_profit", 1000.0)
    v_high = PhaseMapper.normalize("op_profit", 50000.0)
    assert v_high > v_low


def test_normalize_pair_returns_tuple():
    """normalize_pair が (float, float) を返す"""
    v1, v2 = PhaseMapper.normalize_pair("op_profit", "depreciation", 5000.0, 2000.0)
    assert 0.0 <= v1 <= 1.0
    assert 0.0 <= v2 <= 1.0


# ── PhaseMapper + WaveEngine 結合テスト ─────────────────────────────────────

def test_coherent_business_high_amplitude():
    """高利益・高償却 (整合) → amplitude_score が高い"""
    v1, v2 = PhaseMapper.normalize_pair("op_profit", "depreciation", 30000.0, 10000.0)
    r = compute_wave(v1, v2)
    # どちらも高いので振幅は中〜高
    assert r.amplitude_score > 0.3


def test_decoherent_business_low_amplitude():
    """高利益・償却ほぼゼロ (歪み) → amplitude_score が低い"""
    v1, v2 = PhaseMapper.normalize_pair("op_profit", "depreciation", 50000.0, 0.0)
    r = compute_wave(v1, v2)
    assert r.amplitude_score < 0.5
