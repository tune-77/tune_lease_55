# Judgment Abstraction Demo Data Plan

## Core Idea

紫苑のデモデータは、会社データを本物らしく見せるためのものではない。

目的は、リース審査における人間の判断を抽象化し、AIが再利用・再構成・検証できる形にすること。

数学が具体物を数値や式へ抽象化するように、紫苑は個別案件の迷いを判断構文へ抽象化する。

```text
個別案件 -> 人間の違和感 -> 判断仮説 -> 承認条件/反証 -> 結果 -> 検証 -> 判断資産
```

## What To Demonstrate

普通の審査AIは、スコアリングで終わる。

紫苑は、スコアリング後に残る人間の判断を運用対象にする。

- なぜ通すのか
- 何を条件にするのか
- 何を反証として見るのか
- その判断は後から結果と合っていたのか
- 同じ判断構文は別案件でも使えるのか

## Demo Data Principle

会社名・財務数値・案件条件はダミーでよい。

ただし、判断仮説と結果の関係は筋を通す。

良いデモデータ:

- 人間判断: 信用リスクではなく競合失注リスク
- 結果: 失注、競合金利負け
- 検証: 判断仮説は有効

悪いデモデータ:

- 会社データだけそれっぽい
- 判断仮説がない
- 結果理由がない
- 何を検証したのか分からない

## Minimum Schema

```json
{
  "case_id": "demo-0001",
  "industry": "製造業",
  "asset": "CNC工作機械",
  "score_band": "70-79",
  "case_surface": {
    "customer_type": "new",
    "amount_million_yen": 38,
    "visible_score": 74,
    "q_risk": 42
  },
  "human_judgment_hypothesis": {
    "hypothesis_id": "HJ-demo-0001",
    "claim": "補助金採択前提の設備導入であり、信用リスクより未採択時の返済原資が主要論点。",
    "risk_origin": "condition_design",
    "confidence": "medium"
  },
  "judgment_asset_used": {
    "asset_id": "JA-subsidy-cashflow-001",
    "asset_claim": "補助金採択前提の設備案件では、未採択時でもリース料を払える代替返済原資を確認する。"
  },
  "recommended_condition": {
    "condition": "補助金未採択時は、自己資金・銀行借入・営業CFで支払可能な資金繰り表提出を条件とする。",
    "ringi_phrase": "補助金未採択時の代替返済原資確認を承認条件とする。"
  },
  "final_result": {
    "status": "contracted_with_condition",
    "reason": "補助金未採択だったが、自己資金と銀行借入で支払継続。",
    "post_result_status": "12か月延滞なし"
  },
  "hypothesis_outcome": {
    "result": "supported",
    "lesson": "補助金依存案件では、採択可否より未採択時の代替原資条件が有効。"
  }
}
```

## Result Categories

デモでは、成約/失注だけでなく、判断仮説を検証できる結果を持たせる。

- `contracted_no_issue`: 成約、問題なし
- `contracted_with_condition`: 条件付き成約
- `contracted_later_delinquency`: 成約後延滞
- `lost_competitor`: 競合で失注
- `lost_customer_reason`: 顧客都合で失注
- `lost_condition_mismatch`: 条件不一致で失注
- `rejected_credit_risk`: 信用リスクで否決
- `rejected_missing_material`: 資料不足で否決
- `restructured`: 条件変更で再審議

## Hypothesis Outcomes

人間を採点するのではなく、人間が残した判断仮説を検証する。

- `supported`: 結果が判断仮説を支持した
- `partially_supported`: 一部支持した
- `overcautious`: 慎重すぎた可能性
- `missed_risk`: 見落としがあった
- `wrong_origin`: リスクの由来を取り違えた
- `inconclusive`: 結果だけでは判断できない

## Statistical Analysis Targets

後から見る指標:

- 判断仮説の支持率
- 条件設定の有効率
- 業種別に効いた判断構文
- 物件別に効いた確認条件
- 過剰警戒率
- 見落とし率
- 競合リスクと信用リスクの取り違え率
- 修正された判断資産の再利用率
- 稟議文面への採用率

## Presentation Line

会社データはダミーです。
でも、判断仮説と結果の関係は、実務で検証したい構造そのものです。

紫苑は、スコアリングで終わる審査AIではありません。
スコアリングの後に残る人間の判断を抽象化し、再利用し、後から統計的に検証するための基盤です。
