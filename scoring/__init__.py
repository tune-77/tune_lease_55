# -*- coding: utf-8 -*-
"""
学習モデル（業種別ハイブリッド）による単件予測。
sumaho10 の判定結果と併せて表示する用。
"""
from .predict_one import predict_one, map_industry_major_to_scoring

__all__ = ["predict_one", "map_industry_major_to_scoring"]
