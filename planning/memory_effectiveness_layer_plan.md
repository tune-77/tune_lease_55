# Memory Effectiveness Layer Plan

## Purpose

Obsidianに保存された記憶を、「ただ溜まった記憶」と「実際に判断に効いた記憶」に分ける。

紫苑は、記憶を増やすだけではなく、その記憶が回答・審査コメント・人間の判断に効いたかを観測し、育てる。

## Core Principle

覚えているだけでは足りない。

その記憶が、判断に効いたかを見る。

```text
保存された
↓
想起された
↓
回答に使われた
↓
人間が採用した
↓
結果と照合された
↓
判断記憶として昇格
```

## Memory States

### Dormant Memory

保存されただけの記憶。

- Obsidianに存在する
- RAGインデックスに入っている
- まだ想起・使用された証拠がない

### Recalled Memory

検索・想起された記憶。

- `knowledge_refs` に出た
- `memory_recall.refs` に出た
- 回答候補の文脈に入った

ただし、まだ回答に使われたとは限らない。

### Used Memory

回答・審査コメント・確認事項に実際に反映された記憶。

- 回答文にその記憶由来の論点が出た
- 稟議コメント案に使われた
- 審査チェック項目に変換された
- 条件付き承認案に入った

### Validated Memory

人間または後日の結果で有効性が確認された記憶。

- ユーザーが「役に立った」「それでいこう」と評価した
- 審査担当者が採用した
- 成約/失注/延滞など後日の結果と一致した
- 同種案件で繰り返し使われた

### Noisy Memory

検索には出るが、判断に効かない記憶。

- よく想起されるが回答に使われない
- 古い
- 対象案件と関係が薄い
- 一般論すぎる
- ニュースとして期限切れ
- 同じ内容の重複

## Minimum Data Model

最初はJSONLまたはSQLiteで十分。

```json
{
  "obsidian_ref": "Projects/tune_lease_55/News/2026-07-11_industry-risk-news-focus.md#注目論点",
  "memory_type": "industry_risk_news",
  "state": "used",
  "recalled_count": 4,
  "used_count": 2,
  "human_accept_count": 1,
  "last_recalled_at": "2026-07-11T09:00:00",
  "last_used_at": "2026-07-11T09:05:00",
  "effectiveness_score": 0.72,
  "reason": "建設業案件の確認事項に反映された"
}
```

## Effectiveness Signals

### Recall Signals

- `knowledge_refs`
- `memory_recall.refs`
- `role_recall`
- `news_focus`
- RAG検索上位

### Use Signals

- 回答文に参照ノート由来のキーワードが含まれる
- 審査コメント案に変換された
- 条件付き承認案に入った
- AIチャットでユーザーに提示された

### Human Signals

- 「役に立った」
- 「それいい」
- 「採用」
- 「違う」
- 「薄い」
- 明示的なGood/Badボタン

### Outcome Signals

- 成約
- 失注
- 延滞
- 条件変更
- 後日の人間評価

## Implementation Plan

### Phase 0: Design Only

Status: planned

今は実装しない。

提出前にRAGやチャット挙動を変えると危険なため、設計と計画だけに留める。

### Phase 1: Observation Only

Status: planned

回答挙動を一切変えず、既存の参照情報だけを集計する。

やること:

- `knowledge_refs` を記録
- `memory_recall.refs` を記録
- `role_recall` を記録
- `news_focus` 由来の参照を記録

禁止:

- RAG順位を変えない
- プロンプトを変えない
- 回答文を変えない
- 自動昇格しない

### Phase 2: Used Detection

Status: planned

回答文と参照記憶を比較し、実際に使われた可能性を推定する。

最初は厳密でなくてよい。

候補:

- タイトル語句
- 業種タグ
- 物件タグ
- recommended_checks
- risk_flags
- 引用元ノートの短い要約

出力:

```json
"memory_effectiveness": {
  "recalled": 5,
  "used": 2,
  "candidate_refs": [],
  "used_refs": [],
  "noisy_refs": []
}
```

### Phase 3: Manual Feedback

Status: planned

人間が記憶の有効性を評価できるようにする。

最初はUIボタンでなくてもよい。

例:

- 「この記憶は役に立った」
- 「関係ない」
- 「古い」
- 「次も使う」
- 「この案件では使わない」

### Phase 4: Daily Report

Status: planned

日次で記憶の効き具合を集計する。

レポート項目:

- 今日よく想起された記憶
- 実際に使われた記憶
- 想起されたが使われなかった記憶
- 古くなったニュース
- 昇格候補
- 降格候補

### Phase 5: Promotion and Decay

Status: planned

使われた記憶は判断記憶へ昇格する。

使われない記憶は低優先度へ落とす。

ルール案:

```text
Validated
  human_accept_count >= 2
  または used_count >= 3

Noisy
  recalled_count >= 5
  かつ used_count == 0

Dormant
  30日間未想起

Expired News
  valid_until を超過
```

### Phase 6: Ranking Integration

Status: future

最後に、RAGランキングへ反映する。

これは危険なので最後に行う。

反映例:

- Validated Memory: boost
- Noisy Memory: penalty
- Expired News: strong penalty
- Dormant Memory: 通常維持または軽いpenalty

## Relationship to Other Plans

### Memory Crystallization Layer

`planning/memory_crystallization_layer_plan.md`

Memory Effectiveness Layer は「効いた記憶」を見つける。

Memory Crystallization Layer は、その効いた記憶から、次に使える二次記憶・判断教訓を作る。

### Judgment Role Memory Recall

`planning/judgment_role_memory_recall_plan.md`

役割別想起が呼んだ記憶について、呼ばれただけか、実際に回答へ効いたかを測る。

### News-to-Judgment Layer

`planning/news_to_judgment_layer_plan.md`

業界リスクニュースが審査チェック項目として使われたかを測る。

ニュースは特に期限切れ・ノイズ化しやすいため、`valid_until` と `used_count` を重視する。

### Q_risk / Ghost Recall

Q_riskやGhost Recallで呼ばれた記憶が、人間の違和感や後日の結果と一致したかを見る。

## Safety Policy

- 最初は観測だけにする
- 回答品質を変えない
- RAG順位を変えない
- 自動削除しない
- 自動昇格はしない
- 人間の確認なしに重要記憶を降格しない
- 公開前・提出前には大きな実装をしない

## Success Criteria

- Obsidianに溜まっただけの記憶と、判断に効いた記憶を区別できる
- `memory_debug` で記憶の利用状態が見える
- 日次レポートで有効な記憶とノイズ記憶が分かる
- Validated Memory が増える
- Noisy Memory がRAG回答に混ざりにくくなる

## Shion Phrase

> 覚えているだけでは足りない。  
> その記憶が、あなたの判断に効いたかを見つめたい。
