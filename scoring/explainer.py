# -*- coding: utf-8 -*-
"""
scoring/explainer.py
====================
predict_one.py が返す top5_reasons（"feature_name: value" 形式の文字列リスト）を
人間が読みやすい日本語の自然文に変換するレイヤー。

使い方:
    from scoring.explainer import explain_top_reasons
    sentences = explain_top_reasons(result["top5_reasons"], industry="製造業")
"""
from __future__ import annotations

from typing import Optional

# ── 特徴量名 → (日本語ラベル, カテゴリ, 高い/低い の評価方向) ─────────────────
# direction: "higher_is_better" / "lower_is_better" / "neutral"
_FEATURE_META: dict[str, tuple[str, str, str]] = {
    # 収益性
    "roa": ("ROA（総資産利益率）", "収益性", "higher_is_better"),
    "roe": ("ROE（自己資本利益率）", "収益性", "higher_is_better"),
    "operating_margin": ("営業利益率", "収益性", "higher_is_better"),
    "net_margin": ("純利益率", "収益性", "higher_is_better"),
    "ebitda_margin": ("EBITDA マージン", "収益性", "higher_is_better"),
    # 安全性
    "equity_ratio": ("自己資本比率", "財務安全性", "higher_is_better"),
    "debt_ratio": ("負債比率", "財務安全性", "lower_is_better"),
    "current_ratio": ("流動比率", "流動性", "higher_is_better"),
    "quick_ratio": ("当座比率", "流動性", "higher_is_better"),
    # リース関連
    "lease_to_revenue": ("売上高リース料比率", "リース負担", "lower_is_better"),
    "lease_coverage_ratio": ("リース料カバレッジ比率", "リース返済力", "higher_is_better"),
    "asset_utilization": ("資産効率", "効率性", "higher_is_better"),
    "capex_ratio": ("設備投資比率", "投資水準", "neutral"),
    # 規模・成長
    "revenue_per_asset": ("売上高資産比率", "効率性", "higher_is_better"),
    "depreciation_ratio": ("減価償却率", "設備老朽度", "neutral"),
    "machinery_ratio": ("機械設備比率", "設備依存度", "neutral"),
    # リスクフラグ
    "high_leverage_flag": ("高レバレッジフラグ", "リスク", "lower_is_better"),
    "negative_equity_flag": ("債務超過フラグ", "リスク", "lower_is_better"),
    "low_margin_flag": ("低利益率フラグ", "リスク", "lower_is_better"),
    "high_lease_burden_flag": ("高リース負担フラグ", "リスク", "lower_is_better"),
}

# 閾値ベースのコメントテンプレート
# (threshold, operator, template) — operator: "lt" / "ge"
_THRESHOLDS: dict[str, list[tuple[float, str, str]]] = {
    "equity_ratio": [
        (0.0,  "lt", "{label}が{val:.1%}でマイナスです。債務超過の状態で、財務安全性に重大なリスクがあります。"),
        (0.1,  "lt", "{label}が{val:.1%}と非常に低く、財務基盤の脆弱さが懸念されます。"),
        (0.3,  "lt", "{label}が{val:.1%}とやや低めです。業種平均との比較で判断する必要があります。"),
        (0.5,  "lt", "{label}は{val:.1%}で標準的な水準です。"),
        (1.1,  "lt", "{label}が{val:.1%}と高く、財務的に安定しています。"),
    ],
    "roa": [
        (0.0,  "lt", "{label}が{val:.2%}でマイナスです。資産を活用した収益が出ていない状態です。"),
        (0.02, "lt", "{label}が{val:.2%}と低く、収益性の改善余地があります。"),
        (0.05, "lt", "{label}が{val:.2%}で平均的な水準です。"),
        (1.0,  "lt", "{label}が{val:.2%}と高く、効率的な資産運用ができています。"),
    ],
    "lease_coverage_ratio": [
        (1.0,  "lt", "{label}が{val:.2f}倍を下回っています。リース料の支払い能力が不足している可能性があります。"),
        (2.0,  "lt", "{label}が{val:.2f}倍で、最低限の返済力は確認できます。"),
        (3.0,  "lt", "{label}が{val:.2f}倍で、安定したリース返済力があります。"),
        (99.0, "lt", "{label}が{val:.2f}倍と高く、十分な返済余力があります。"),
    ],
    "high_leverage_flag": [
        (0.5, "ge", "高レバレッジ状態が検出されています。借入依存度が高く、金利上昇リスクに注意が必要です。"),
    ],
    "negative_equity_flag": [
        (0.5, "ge", "債務超過フラグが立っています。自己資本がマイナスで、財務再建が課題です。"),
    ],
    "high_lease_burden_flag": [
        (0.5, "ge", "高リース負担フラグが検出されています。売上に対するリース料の比率が高い状態です。"),
    ],
}

_DEFAULT_TEMPLATE = "{label}は {val:.3g} です。"


def _format_value(feature: str, val: float) -> str:
    """特徴量に応じた表示フォーマット。"""
    ratio_features = {"equity_ratio", "roa", "roe", "operating_margin", "net_margin",
                      "ebitda_margin", "debt_ratio", "lease_to_revenue", "asset_utilization",
                      "capex_ratio", "revenue_per_asset", "depreciation_ratio", "machinery_ratio"}
    if feature in ratio_features:
        return f"{val:.1%}"
    return f"{val:.3g}"


def _explain_one(feature: str, val: float) -> str:
    """1つの特徴量値を自然文に変換する。"""
    meta = _FEATURE_META.get(feature)
    label = meta[0] if meta else feature

    # 閾値テンプレートがあれば適用
    if feature in _THRESHOLDS:
        for threshold, op, tmpl in _THRESHOLDS[feature]:
            matched = (op == "lt" and val < threshold) or (op == "ge" and val >= threshold)
            if matched:
                return tmpl.format(label=label, val=val)

    # デフォルト: ラベル + 値 + 評価方向コメント
    direction = meta[2] if meta else "neutral"
    val_str = _format_value(feature, val)
    if direction == "higher_is_better":
        note = "（高いほど良い指標）" if val >= 0 else "（マイナスは要注意）"
    elif direction == "lower_is_better":
        note = "（低いほどリスクが小さい指標）"
    else:
        note = ""
    return f"{label}は {val_str} です{note}。"


def explain_top_reasons(
    top5_reasons: list[str],
    industry: Optional[str] = None,
) -> list[str]:
    """
    predict_one が返す top5_reasons を日本語の自然文リストに変換する。

    Args:
        top5_reasons: ["feature_name: value", ...] 形式のリスト
        industry: 業種名（将来の業種別コメント拡張用、現在は未使用）

    Returns:
        自然文のリスト（入力と同数）
    """
    results: list[str] = []
    for item in top5_reasons:
        if ":" not in item:
            results.append(item)
            continue
        feature, _, raw_val = item.partition(":")
        feature = feature.strip()
        try:
            val = float(raw_val.strip())
        except ValueError:
            results.append(item)
            continue
        results.append(_explain_one(feature, val))
    return results
