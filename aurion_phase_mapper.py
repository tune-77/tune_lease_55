"""
AURION CORE — Phase Mapper (AV.2)
業種コード → 最優先干渉ペア選択 + 財務値 [0,1] 正規化。
既存 quantum_analysis_module の _BASE_PAIRS / _INDUSTRY_PAIR_MAP を流用。
"""
from __future__ import annotations

import math
from typing import Sequence

from quantum_analysis_module import _BASE_PAIRS, _INDUSTRY_PAIR_MAP

# 財務変数の正規化スケール（M円 = 千円 / 1000）
_SCALE_M_YEN: dict[str, float] = {
    "op_profit":    30.0,
    "depreciation": 10.0,
    "machines":     50.0,
    "equip_total":  80.0,
    "net_income":   20.0,
    "ord_profit":   20.0,
}
# 千円変換が不要な無次元変数
_DIMENSIONLESS: frozenset[str] = frozenset({"trend_val", "qualit_score", "qualit_binary"})
_DIMENSIONLESS_SCALE: dict[str, float] = {
    "trend_val":    7.0,
    "qualit_score": 3.0,
    "qualit_binary": 1.0,
}


class PhaseMapper:
    """業種コードから干渉ペアを取得し、財務値を位相 [0,1] に正規化する。"""

    @staticmethod
    def get_pair(industry_code: str) -> tuple[str, str]:
        """
        業種大分類コード → 最優先ペア (var_a, var_b)。
        コードが不明または未登録のときは BASE_PAIRS の先頭ペアを返す。

        Examples
        --------
        >>> PhaseMapper.get_pair("D")  # 建設業
        ('op_profit', 'equip_total')
        >>> PhaseMapper.get_pair("C")  # 鉱業 (未登録) → デフォルト
        ('op_profit', 'depreciation')
        """
        pairs = _INDUSTRY_PAIR_MAP.get(industry_code.strip().upper())
        if pairs:
            # 重みが最大のペアを返す
            best = max(pairs, key=lambda t: t[2])
            return best[0], best[1]
        # デフォルト: _BASE_PAIRS の先頭
        return _BASE_PAIRS[0][0], _BASE_PAIRS[0][1]

    @staticmethod
    def get_all_pairs(industry_code: str) -> list[tuple[str, str, float]]:
        """BASE_PAIRS + 業種別追加ペアを重み付きで返す。"""
        base: list[tuple[str, str, float]] = list(_BASE_PAIRS)
        extra: list[tuple[str, str, float]] = list(
            _INDUSTRY_PAIR_MAP.get(industry_code.strip().upper(), [])
        )
        return base + extra

    @staticmethod
    def normalize(var_name: str, value_sen_yen: float) -> float:
        """
        財務値（千円単位）または無次元値 → [0, 1]。
        負値は 0 にクリップ。tanh スケーリングで大外れを抑制。

        Parameters
        ----------
        var_name : str
            変数名（'op_profit', 'depreciation', 'machines' 等）。
        value_sen_yen : float
            千円単位の財務値、または無次元変数の生値。

        Returns
        -------
        float in [0, 1]
        """
        if var_name in _DIMENSIONLESS:
            scale = _DIMENSIONLESS_SCALE.get(var_name, 1.0)
            v = max(0.0, float(value_sen_yen))
            return math.tanh(v / max(scale, 1e-9))

        scale_m = _SCALE_M_YEN.get(var_name, 30.0)
        v_m = max(0.0, float(value_sen_yen)) / 1000.0
        return math.tanh(v_m / max(scale_m, 1e-9))

    @staticmethod
    def normalize_pair(
        var_a: str,
        var_b: str,
        val_a: float,
        val_b: float,
    ) -> tuple[float, float]:
        """2変数を正規化して (v1, v2) ∈ [0,1]² を返す。"""
        return PhaseMapper.normalize(var_a, val_a), PhaseMapper.normalize(var_b, val_b)
