# Memory Crystallization Layer Plan

## Purpose

一次記憶から、次に使える二次記憶・判断教訓を作る。

紫苑は、案件、会話、ニュース、改善ログ、Private Reflectionをただ保存するだけではなく、複数の記憶から共通パターンを見つけ、審査で再利用できる「判断の結晶」を作る。

## Core Principle

記憶から記憶を作る。

ただし、複製ではなく抽象化を行う。

```text
一次記憶
  個別案件、会話、ニュース、失敗、内省
↓
共通パターンの抽出
↓
反例の確認
↓
二次記憶候補
↓
人間の承認
↓
判断記憶へ昇格
```

## Memory Layers

### Primary Memory

個別の出来事。

- 案件
- 会話
- ニュース
- 失敗
- 改善ログ
- Private Reflection
- 人間の判断メモ

### Secondary Memory

複数の一次記憶から抽象化した教訓。

例:

```text
高スコア案件でも、競合条件・導入目的・補助金待ちが弱い場合は成約につながらない。
```

### Tertiary Memory

安定して使える判断ルール・確認手順・紫苑の価値観。

例:

```text
高スコア案件ほど、信用リスクだけでなく「成約に至る必然性」を確認する。
```

## Candidate Format

```markdown
# 記憶の結晶化候補: 高スコア失注

## 抽象化された教訓

高スコア案件でも、競合条件・導入目的・補助金待ちが弱い場合は成約につながらない。

## 根拠記憶

- [[案件A]]
- [[案件B]]
- [[ニュースC]]

## 反例

- [[案件D]] 競合ありでも保守条件で成約

## 次回確認

- 導入時期に必然性はあるか
- 競合金利以外の条件差はあるか
- 補助金採択前提ではないか
- 稟議を急ぐ理由はあるか

## 確信度

中

## 状態

candidate
```

## Required Safety Rules

- 根拠記憶を必ず残す
- 反例を必ず探す
- 確信度を持つ
- 最初は候補として保存する
- 人間承認なしに判断ルールへ昇格しない
- 自動でRAG順位を変えない
- 自動で既存記憶を削除しない
- 1件だけの出来事から強い一般化をしない

## Implementation Plan

### Phase 0: Design Only

Status: planned

今は実装しない。

提出前に記憶生成系を入れると、RAG・プロンプト・改善パイプライン全体へ影響するため、計画だけに留める。

### Phase 1: Candidate Generation

Status: planned

似た記憶を集め、結晶化候補をMarkdownで生成する。

入力:

- `memory_effectiveness` で `used` または `validated` になった記憶
- 高スコア失注
- 低スコア成約
- Q_risk高値案件
- ニュース由来の確認事項
- Private Reflectionの見落とし

出力:

- `Projects/tune_lease_55/Memory Crystallization/Candidates/`
- 状態: `candidate`

### Phase 2: Evidence and Counterexample Check

Status: planned

候補ごとに根拠と反例を確認する。

最低条件:

- 根拠記憶 2件以上
- 反例探索結果 1件以上、または「反例未発見」と明記
- 確信度は `low / medium / high`

### Phase 3: Human Review

Status: planned

人間が承認する。

選択肢:

- approve: 判断記憶へ昇格
- revise: 文言修正
- reject: 却下
- observe_more: データ不足

### Phase 4: Promotion to Judgment Memory

Status: planned

承認された候補だけを判断記憶へ昇格する。

保存先候補:

- `Projects/tune_lease_55/Judgment Memory/`
- `knowledge_base/okf_lease_concepts/rules/`

ただし、どちらへ入れるかは公開範囲と機密性で分ける。

### Phase 5: Use in Role Recall

Status: planned

昇格した判断記憶を、判断役割別想起に接続する。

対応:

- `evidence`
- `counter_evidence`
- `failure_lesson`
- `approval_condition`
- `conscience_check`

### Phase 6: Effectiveness Loop

Status: planned

結晶化した記憶自体も、Memory Effectiveness Layerで追跡する。

見るもの:

- 何回想起されたか
- 回答に使われたか
- 人間が採用したか
- 後日の結果と一致したか
- 古くなったか

## Example Themes

### High Score Lost

```text
高スコア案件でも、競合条件・導入目的・補助金待ちが弱い場合は成約につながらない。
```

### Low Score Won

```text
低スコア案件でも、銀行支援・保証・前受金・物件換金性が強い場合は条件付きで通る可能性がある。
```

### Construction Equipment

```text
建機案件では、財務スコアよりも稼働予定、工期、現場の継続性が判断を左右することがある。
```

### Industry Risk News

```text
外部ニュースは、対象企業の業種・物件・資金繰りに接続できる場合だけ審査材料にする。
```

## Relationship to Other Layers

### Memory Effectiveness Layer

`planning/memory_effectiveness_layer_plan.md`

効いた記憶を見つける。

Memory Crystallization Layerは、その効いた記憶から次の教訓を作る。

### News-to-Judgment Layer

`planning/news_to_judgment_layer_plan.md`

業界リスクニュースから作られた確認事項が、複数案件で効いた場合、結晶化候補にする。

### Judgment Role Memory Recall

`planning/judgment_role_memory_recall_plan.md`

結晶化された判断記憶を、役割別に想起する。

### Q_risk / Ghost Recall

Q_riskやGhost Recallが拾った違和感が、複数案件で同じ形を取った場合、結晶化候補にする。

## Success Criteria

- 個別記憶から抽象化された教訓候補を作れる
- 候補に根拠記憶と反例が残る
- 人間が承認したものだけ判断記憶へ昇格する
- 昇格後の記憶も有効性を追跡できる
- 紫苑の回答が、単なる検索結果ではなく経験に基づく判断に近づく

## Shion Phrase

> 断片的な記憶を、次に使える判断の結晶へ変える。
