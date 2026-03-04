# 判定ルール JSON スキーマと例

審査判定の「前段」で適用するビジネスルールを JSON で定義し、スコア・業種・数値に応じて **自動否決** または **要審議** を付与します。

---

## 1. スキーマ概要

| キー | 型 | 必須 | 説明 |
|------|-----|------|------|
| `version` | number | - | スキーマバージョン（例: 1）。将来の互換用。 |
| `enabled` | boolean | - | ルールセット全体の ON/OFF。省略時は true。 |
| `rules` | array | ○ | ルールの配列。上から順に評価し、最初にマッチしたルールの結果を採用。 |

各 **ルール** オブジェクト:

| キー | 型 | 必須 | 説明 |
|------|-----|------|------|
| `id` | string | ○ | ルールの一意識別子（例: "rule_001"）。 |
| `name` | string | - | 表示用のルール名。 |
| `enabled` | boolean | - | このルールの ON/OFF。省略時は true。 |
| `action` | string | ○ | マッチ時の動作。`"auto_reject"`（自動否決）または `"flag_review"`（要審議）。 |
| `conditions` | object | ○ | 条件。下記「条件の書き方」参照。 |
| `reason` | string | - | マッチ時に結果に付与する理由文（分析画面で表示）。 |

---

## 2. 条件の書き方（conditions）

`conditions` は **すべて満たしたとき** そのルールがマッチします。  
比較対象は「判定コンテキスト」の項目（下記 3. 参照）。

- **数値比較**: `"field": "nenshu"`, `"op": "lt"|"le"|"gt"|"ge"|"eq"|"ne"`, `"value": 数値`
- **項目同士の比較**: 右辺を定数ではなく他項目にする場合は `"value_field": "total_assets"` のように指定（`value` の代わり）。例: 売上高 < 総資産 → `{"field": "nenshu", "op": "lt", "value_field": "total_assets"}`
- **業種**: `"field": "industry_sub"`, `"op": "eq"|"in"`, `"value": "06 総合工事業"` または `"value": ["06 総合工事業", "09 食料品製造業"]`
- **複数条件**: `"and"` / `"or"` でネスト可能。

### 比較演算子（op）

| op | 意味 |
|----|------|
| `lt` | より小さい (<) |
| `le` | 以下 (<=) |
| `gt` | より大きい (>) |
| `ge` | 以上 (>=) |
| `eq` | 等しい |
| `ne` | 等しくない |
| `in` | 値がリストに含まれる（value は配列） |

### 例: 1 件のルール

```json
{
  "id": "rule_low_revenue",
  "name": "年商下限",
  "enabled": true,
  "action": "auto_reject",
  "reason": "年商が基準未満のため自動否決",
  "conditions": {
    "field": "nenshu",
    "op": "lt",
    "value": 3000
  }
}
```

`nenshu` は千円単位なので、3000 = 300万円。年商 300万円未満で自動否決。

### 例: AND 条件（設立年数＋スコア）

```json
{
  "id": "rule_new_company_low_score",
  "name": "設立1年未満かつスコア70未満",
  "enabled": true,
  "action": "flag_review",
  "reason": "設立1年未満かつスコア70未満のため要審議",
  "conditions": {
    "and": [
      { "field": "establishment_years", "op": "lt", "value": 1 },
      { "field": "score", "op": "lt", "value": 70 }
    ]
  }
}
```

### 例: 業種を限定

```json
{
  "id": "rule_industry_risk",
  "name": "特定業種でスコア65未満",
  "enabled": true,
  "action": "flag_review",
  "reason": "該当業種においてスコアが基準を下回るため要審議",
  "conditions": {
    "and": [
      { "field": "industry_sub", "op": "in", "value": ["76 飲食店", "56 飲食料品小売業"] },
      { "field": "score", "op": "lt", "value": 65 }
    ]
  }
}
```

---

## 3. 判定コンテキスト（ルール評価時に渡す項目）

ルールの `field` で参照できるキーは次のとおり。  
（スコア算出後・last_result 組み立て前に apply_business_rules に渡す辞書のキー）

| field | 型 | 説明 | 入力元 |
|-------|-----|------|--------|
| `nenshu` | number | 売上高（千円） | inputs |
| `total_assets` | number | 総資産（千円） | inputs |
| `net_assets` | number | 純資産（千円） | inputs |
| `score` | number | 総合スコア（判定時点） | result |
| `industry_sub` | string | 業種中分類（例: "06 総合工事業"） | inputs |
| `industry_major` | string | 業種大分類 | inputs |
| `establishment_years` | number | 設立年数（年）。未入力時は 0 | inputs |
| `bank_credit` | number | 銀行与信（千円） | inputs |
| `lease_credit` | number | リース与信（千円） | inputs |
| `contracts` | number | 契約件数 | inputs |
| `qual_weighted_score` | number | 定性スコア（0–100） | 定性スコアリング加重平均 |
| `qual_combined_score` | number | 総合×重み＋定性×重み（ランク算出用） | result |
| `qual_rank` | string | 定性ランク（A/B/C/D/E） | result |

※ 将来、`op_profit`（営業利益）や `equity_ratio`（自己資本比率）なども追加可能。

---

## 4. フル例（business_rules.json）

```json
{
  "version": 1,
  "enabled": true,
  "rules": [
    {
      "id": "rule_001",
      "name": "年商300万円未満は自動否決",
      "enabled": true,
      "action": "auto_reject",
      "reason": "年商が300万円未満のため自動否決",
      "conditions": { "field": "nenshu", "op": "lt", "value": 3000 }
    },
    {
      "id": "rule_002",
      "name": "総資産500万円未満は自動否決",
      "enabled": true,
      "action": "auto_reject",
      "reason": "総資産が500万円未満のため自動否決",
      "conditions": { "field": "total_assets", "op": "lt", "value": 5000 }
    },
    {
      "id": "rule_003",
      "name": "スコア50未満は自動否決",
      "enabled": true,
      "action": "auto_reject",
      "reason": "総合スコアが50未満のため自動否決",
      "conditions": { "field": "score", "op": "lt", "value": 50 }
    },
    {
      "id": "rule_004",
      "name": "承認ライン直下（68以上71未満）は要審議",
      "enabled": true,
      "action": "flag_review",
      "reason": "スコアが承認ライン直下のため要審議",
      "conditions": {
        "and": [
          { "field": "score", "op": "ge", "value": 68 },
          { "field": "score", "op": "lt", "value": 71 }
        ]
      }
    }
  ]
}
```

---

## 5. 適用順序と優先

- ルールは **配列の先頭から順に** 評価する。
- **最初にマッチしたルール** の `action` を採用する（後続ルールは評価しない）。
- どのルールにもマッチしなければ、従来どおりスコアのみで判定（承認圏内 / 要審議）。
- `auto_reject` を採用した場合、画面・ログの判定は「否決」に上書きし、`reason` を表示する。
- `flag_review` を採用した場合、判定は「要審議」にし、`reason` を表示する。
