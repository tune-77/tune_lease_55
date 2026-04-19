"""
リース審査エージェント — Anthropic SDK (claude-opus-4-7)
=======================================================
Anthropic SDK の tool runner (beta) を使用して、既存のスコアリングモジュールを
ツールとして呼び出す自律型審査エージェント。

使い方:
    python agent_claude.py                # デモ案件で実行
    python agent_claude.py '{"nenshu": 500000, ...}'  # JSON 入力で実行

環境変数:
    ANTHROPIC_API_KEY  — Anthropic API キー（必須）
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import anthropic
from anthropic import beta_tool

# ── プロジェクトルートをパスに追加 ────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scoring_core import run_quick_scoring, compute_score_contributions
from asset_scorer import calc_asset_score
from total_scorer import calc_total_score
from data_cases import get_effective_coeffs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ツール定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@beta_tool
def borrower_score(
    nenshu: float,
    op_profit: float,
    net_income: float,
    net_assets: float,
    total_assets: float,
    industry_major: str = "D 建設業",
    industry_sub: str = "06 総合工事業",
    grade: str = "1-3",
    bank_credit: float = 0.0,
    lease_credit: float = 0.0,
    contracts: int = 0,
    ord_profit: float = 0.0,
    gross_profit: float = 0.0,
    asset_score: float = 50.0,
) -> str:
    """借手（債務者）のスコアリングを実施する。財務指標・業種・格付から0-100点のスコアと承認判定を算出する。

    Args:
        nenshu: 売上高（千円単位）
        op_profit: 営業利益（千円単位）
        net_income: 純利益（千円単位）
        net_assets: 純資産（千円単位）
        total_assets: 総資産（千円単位）
        industry_major: 業種大分類（例: "D 建設業", "F 製造業"）
        industry_sub: 業種小分類（例: "06 総合工事業", "24 食料品製造業"）
        grade: 社内格付（"1-3", "4-6", "要注意", "無格付"）
        bank_credit: 銀行信用枠（千円単位）
        lease_credit: リース信用枠（千円単位）
        contracts: 過去取引件数
        ord_profit: 経常利益（千円単位）
        gross_profit: 粗利益（千円単位）
        asset_score: 物件スコア（0-100、省略時50）
    """
    inputs = {
        "nenshu": nenshu,
        "op_profit": op_profit,
        "net_income": net_income,
        "net_assets": net_assets,
        "total_assets": total_assets,
        "industry_major": industry_major,
        "industry_sub": industry_sub,
        "grade": grade,
        "bank_credit": bank_credit,
        "lease_credit": lease_credit,
        "contracts": contracts,
        "ord_profit": ord_profit,
        "gross_profit": gross_profit,
        "asset_score": asset_score,
    }
    result = run_quick_scoring(inputs)
    return json.dumps(result, ensure_ascii=False, default=str)


@beta_tool
def asset_score(
    category: str,
    item_scores: dict,
    lease_months: int = 36,
    tech_life_months: int = 60,
    has_buyout_option: bool = False,
    is_major_maker: bool = False,
) -> str:
    """物件（リース資産）のスコアリングを実施する。カテゴリ別の評価項目から物件グレード（S/A/B/C/D）を算出する。

    Args:
        category: 物件カテゴリ（"IT機器" | "産業機械" | "車両" | "医療機器"）
        item_scores: 評価項目スコア辞書（{item_id: 0-100}）。未入力は50補完。
            IT機器の主な項目: resale_value, market_liquidity, tech_obsolescence, support_availability
            産業機械の主な項目: resale_value, market_liquidity, customization, maintenance_cost
            車両の主な項目: resale_value, market_liquidity, mileage_condition, maintenance_cost
            医療機器の主な項目: resale_value, regulatory_risk, tech_obsolescence, maintenance_cost
        lease_months: リース期間（月数）
        tech_life_months: 技術寿命（月数）
        has_buyout_option: 買取オプションの有無
        is_major_maker: 大手メーカー品か否か
    """
    contract = {
        "lease_months": lease_months,
        "tech_life_months": tech_life_months,
        "has_buyout_option": has_buyout_option,
        "is_major_maker": is_major_maker,
    }
    result = calc_asset_score(category, item_scores, contract)
    return json.dumps(result, ensure_ascii=False, default=str)


@beta_tool
def total_score(
    category: str,
    asset_item_scores: dict,
    obligor_score: float,
    lease_months: int = 36,
    tech_life_months: int = 60,
    has_buyout_option: bool = False,
    is_major_maker: bool = False,
) -> str:
    """借手スコアと物件スコアを統合した総合スコアを算出する。カテゴリ別の配分比率（物件:借手）で加重平均する。

    Args:
        category: 物件カテゴリ（"IT機器" | "産業機械" | "車両" | "医療機器"）
        asset_item_scores: 物件評価項目スコア辞書（{item_id: 0-100}）
        obligor_score: 借手スコア（0-100）
        lease_months: リース期間（月数）
        tech_life_months: 技術寿命（月数）
        has_buyout_option: 買取オプションの有無
        is_major_maker: 大手メーカー品か否か
    """
    contract = {
        "lease_months": lease_months,
        "tech_life_months": tech_life_months,
        "has_buyout_option": has_buyout_option,
        "is_major_maker": is_major_maker,
    }
    result = calc_total_score(category, asset_item_scores, obligor_score, contract)
    return json.dumps(result, ensure_ascii=False, default=str)


@beta_tool
def score_breakdown(
    nenshu: float,
    op_profit: float,
    net_income: float,
    bank_credit: float = 0.0,
    lease_credit: float = 0.0,
    contracts: int = 0,
    grade: str = "1-3",
    industry_major: str = "D 建設業",
    ord_profit: float = 0.0,
    gross_profit: float = 0.0,
    machines: float = 0.0,
    other_assets: float = 0.0,
    rent: float = 0.0,
    depreciation: float = 0.0,
) -> str:
    """スコアへの寄与度（SHAP近似）を計算し、スコアを押し上げ・押し下げしている要因を特定する。

    Args:
        nenshu: 売上高（千円単位）
        op_profit: 営業利益（千円単位）
        net_income: 純利益（千円単位）
        bank_credit: 銀行信用枠（千円単位）
        lease_credit: リース信用枠（千円単位）
        contracts: 過去取引件数
        grade: 社内格付（"1-3", "4-6", "要注意", "無格付"）
        industry_major: 業種大分類
        ord_profit: 経常利益（千円単位）
        gross_profit: 粗利益（千円単位）
        machines: 機械装置（千円単位）
        other_assets: その他資産（千円単位）
        rent: 家賃（千円単位）
        depreciation: 減価償却費（千円単位）
    """
    data = {
        "nenshu": nenshu,
        "op_profit": op_profit / 1000,
        "net_income": net_income / 1000,
        "bank_credit": bank_credit,
        "lease_credit": lease_credit,
        "contracts": contracts,
        "grade": grade,
        "industry_major": industry_major,
        "ord_profit": ord_profit / 1000,
        "gross_profit": gross_profit / 1000,
        "machines": machines / 1000,
        "other_assets": other_assets / 1000,
        "rent": rent / 1000,
        "depreciation": depreciation / 1000,
    }
    coeff_set = get_effective_coeffs(industry_major)
    contributions = compute_score_contributions(data, coeff_set)
    return json.dumps(contributions[:10], ensure_ascii=False, default=str)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  システムプロンプト
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM_PROMPT = """あなたは温水式リースの審査AIエージェントです。
申請案件の情報を受け取り、以下のステップで審査を実施してください。

## 審査フロー

1. **借手スコアの算出** — `borrower_score` ツールで財務・業種・格付からスコアを計算
2. **物件スコアの算出** — `asset_score` ツールで物件カテゴリ・評価項目からスコアを計算
3. **総合スコアの算出** — `total_score` ツールで借手と物件を統合（カテゴリ別配分比率を適用）
4. **寄与度分析** — `score_breakdown` ツールでスコアの主要因を特定
5. **審査レポートの作成** — 上記の結果を踏まえた総合所見を日本語で記述

## レポート形式

審査完了後、以下の構成でレポートを出力してください:

```
## 審査サマリー
- 総合スコア: XX.X点 / グレード: X
- 借手スコア: XX.X点（配分比率: XX%）
- 物件スコア: XX.X点（配分比率: XX%）
- 判定: [承認 / 要検討 / 否決]

## 業界比較
[業界ベンチマークとの比較コメント]

## スコア主要因（上位5項目）
[寄与度の高い要因を箇条書き]

## 総合所見
[審査官としての定性的コメント、リスク指摘、推奨事項]
```

## 注意事項
- 財務数値は千円単位で入力されている
- カテゴリは "IT機器" / "産業機械" / "車両" / "医療機器" のいずれか
- 情報が不足する場合はデフォルト値（物件スコア項目は50点）を使用して審査を進める
- 判定ライン: 71点以上 → 承認、51-70点 → 要検討、50点以下 → 否決"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  エージェント実行
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_screening(case_input: Optional[dict] = None) -> str:
    """
    リース審査エージェントを実行し、審査レポートを返す。

    Parameters
    ----------
    case_input : dict, optional
        審査案件の情報。省略時はデモ案件を使用。

    Returns
    -------
    str
        審査レポート（マークダウン形式）
    """
    if case_input is None:
        # デモ案件（中堅製造業、産業機械リース）
        case_input = {
            "company": "株式会社サンプル製造",
            "nenshu": 800_000,       # 8億円
            "op_profit": 40_000,     # 4000万円
            "ord_profit": 38_000,
            "net_income": 25_000,
            "net_assets": 200_000,   # 2億円
            "total_assets": 600_000, # 6億円
            "industry_major": "F 製造業",
            "industry_sub": "24 食料品製造業",
            "grade": "1-3",
            "bank_credit": 300_000,
            "lease_credit": 50_000,
            "contracts": 5,
            "category": "産業機械",
            "lease_months": 48,
            "tech_life_months": 96,
            "is_major_maker": True,
            "asset_items": {
                "resale_value": 55,
                "market_liquidity": 60,
                "customization": 45,
                "maintenance_cost": 70,
            },
        }

    client = anthropic.Anthropic()

    # 案件情報をユーザーメッセージとして構成
    company = case_input.get("company", "申請企業")
    user_message = f"""以下の案件を審査してください。

**申請企業**: {company}
**案件情報**:
```json
{json.dumps(case_input, ensure_ascii=False, indent=2)}
```

上記情報を使用して、借手スコア・物件スコア・総合スコアを順番に算出し、
寄与度分析も実施した上で、審査レポートを作成してください。"""

    print(f"\n{'='*60}")
    print(f"  リース審査エージェント起動: {company}")
    print(f"{'='*60}\n")

    # tool runner でエージェントループを自動処理（ストリーミング）
    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-7",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # プロンプトキャッシュ
            }
        ],
        tools=[borrower_score, asset_score, total_score, score_breakdown],
        messages=[{"role": "user", "content": user_message}],
    )

    # ストリーミングでツール呼び出しを表示しながら最終レポートを取得
    final_text = ""
    tool_call_count = 0

    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                tool_call_count += 1
                print(f"[ツール {tool_call_count}] {block.name}() を実行中...")
            elif block.type == "text" and block.text:
                final_text = block.text

    print(f"\n[完了] ツール呼び出し回数: {tool_call_count}\n")
    print(final_text)
    return final_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  エントリーポイント
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    case: Optional[dict] = None
    if len(sys.argv) > 1:
        try:
            case = json.loads(sys.argv[1])
        except json.JSONDecodeError as e:
            print(f"JSON パースエラー: {e}", file=sys.stderr)
            sys.exit(1)

    run_screening(case)
