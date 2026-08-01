"""
components/asset_valuer.py の残価率計算ロジックのテスト。

SHION-9BADD3対応: リース終了時の残価設定が物件の汎用性・市場価値に
見合っているかを検証する。特注・造作設備（例: セントラルキッチン等）は
AIの流動性係数が楽観的でも、ルールベースで上限をキャップすることを確認する。
"""
from __future__ import annotations

from unittest.mock import patch

import components.asset_valuer as av


def _optimistic_ai_response() -> str:
    return (
        '{"liquidity_rank":"A","liquidity_factor":0.95,'
        '"market_demand":"需要は堅調です。","risks":"特になし。"}'
    )


def _conservative_ai_response() -> str:
    return (
        '{"liquidity_rank":"D","liquidity_factor":0.4,'
        '"market_demand":"需要は限定的です。","risks":"転売困難。"}'
    )


def test_detect_low_versatility_matches_known_keywords():
    assert av._detect_low_versatility("セントラルキッチン一式", "") == "セントラルキッチン"
    assert av._detect_low_versatility("特注厨房設備", "XYZ-1") == "特注"
    assert av._detect_low_versatility("造作什器", "") == "造作"


def test_detect_low_versatility_matches_fitout_keywords():
    """内装工事・什器一式・店舗設計等、店舗フィットアウト関連語にも反応する。"""
    assert av._detect_low_versatility("店舗内装工事一式", "") == "内装工事"
    assert av._detect_low_versatility("什器一式", "") == "什器一式"
    assert av._detect_low_versatility("店舗設計・施工", "") == "店舗設計"
    assert av._detect_low_versatility("テナント工事", "") == "テナント工事"
    # 「造作工事」は既存キーワード「造作」がリスト内で先に一致するため "造作" が返る
    assert av._detect_low_versatility("造作工事", "") == "造作"


def test_detect_low_versatility_ignores_generic_equipment():
    assert av._detect_low_versatility("油圧ショベル", "コマツ PC200-10") is None
    assert av._detect_low_versatility("ハイエース", "トヨタ") is None


def test_evaluate_asset_value_caps_optimistic_liquidity_for_custom_equipment():
    """AIが特注設備に楽観的な流動性係数(0.95)を返しても、上限でキャップされる。"""
    with patch.object(av, "_ai_call", return_value=_optimistic_ai_response()):
        res = av.evaluate_asset_value(
            "セントラルキッチン設備一式", "", 20_000_000, 60
        )

    assert res["low_versatility_flag"] is True
    assert res["low_versatility_keyword"] == "セントラルキッチン"
    assert res["versatility_capped"] is True
    assert res["liquidity_factor"] <= av.LOW_VERSATILITY_LIQUIDITY_CAP


def test_evaluate_asset_value_leaves_generic_equipment_uncapped():
    """汎用設備であれば、AIの流動性係数をそのまま採用する。"""
    with patch.object(av, "_ai_call", return_value=_optimistic_ai_response()):
        res = av.evaluate_asset_value(
            "油圧ショベル", "コマツ PC200-10", 20_000_000, 60
        )

    assert res["low_versatility_flag"] is False
    assert res["versatility_capped"] is False
    assert res["liquidity_factor"] == 0.95


def test_evaluate_asset_value_does_not_raise_already_conservative_estimate():
    """AIがすでに保守的な係数を返している場合、キャップによって値を引き上げない。"""
    with patch.object(av, "_ai_call", return_value=_conservative_ai_response()):
        res = av.evaluate_asset_value(
            "特注造作什器", "", 20_000_000, 60
        )

    assert res["low_versatility_flag"] is True
    assert res["versatility_capped"] is False
    assert res["liquidity_factor"] == 0.4


def test_evaluate_asset_value_residual_pct_reflects_capped_liquidity():
    """キャップ適用後の残価率が、キャップなしの場合より低くなることを確認する。"""
    with patch.object(av, "_ai_call", return_value=_optimistic_ai_response()):
        capped = av.evaluate_asset_value(
            "セントラルキッチン設備一式", "", 20_000_000, 60
        )
    with patch.object(av, "_ai_call", return_value=_optimistic_ai_response()):
        uncapped = av.evaluate_asset_value(
            "業務用冷蔵庫", "", 20_000_000, 60
        )

    assert capped["residual_value_pct"] <= uncapped["residual_value_pct"]
