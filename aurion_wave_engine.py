"""
AURION CORE — Wave Engine (AV.1)
2変数の正規化値 [0, 1] を位相にマップしてサイン波合成・干渉スコアを計算する。
正規化は呼び出し側 (PhaseMapper) が担う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WaveResult:
    wave_a: np.ndarray      # 変数1 のサイン波 ∈ [-1, 1]
    wave_b: np.ndarray      # 変数2 のサイン波 ∈ [-1, 1]
    composite: np.ndarray   # 合成波 = (wave_a + wave_b) / 2 ∈ [-1, 1]
    amplitude_score: float  # 共鳴スコア ∈ [0, 1]; 1=完全共鳴, 0=完全消去
    phase_diff: float       # |φ1 - φ2| ∈ [0, π] (radians)


def compute_wave(
    v1: float,
    v2: float,
    freq: float = 1.0,
    t: np.ndarray | None = None,
    n_points: int = 256,
    n_cycles: int = 2,
) -> WaveResult:
    """
    2変数のサイン波を合成し干渉スコアを返す。

    位相マッピング: φ_i = v_i * π
    - v1 == v2     → 位相差 0 → amplitude_score = 1.0 (完全共鳴)
    - |v1 - v2| == 1 → 位相差 π → amplitude_score = 0.0 (完全デコヒーレンス)

    amplitude_score = 0.5 * (1 + cos(|φ1 - φ2|))
    """
    v1 = float(np.clip(v1, 0.0, 1.0))
    v2 = float(np.clip(v2, 0.0, 1.0))

    if t is None:
        t = np.linspace(0.0, n_cycles / freq, n_points)

    phi1 = v1 * math.pi
    phi2 = v2 * math.pi

    wave_a = np.sin(2.0 * math.pi * freq * t + phi1)
    wave_b = np.sin(2.0 * math.pi * freq * t + phi2)
    composite = (wave_a + wave_b) / 2.0

    phase_diff = abs(phi1 - phi2)
    amplitude_score = 0.5 * (1.0 + math.cos(phase_diff))

    return WaveResult(
        wave_a=wave_a,
        wave_b=wave_b,
        composite=composite,
        amplitude_score=amplitude_score,
        phase_diff=phase_diff,
    )
