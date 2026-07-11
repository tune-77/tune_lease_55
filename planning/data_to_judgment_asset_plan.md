# Data-to-Judgment Asset Plan

## Purpose

せっかく集めたデータを、ただ保存するだけで終わらせず、次の審査判断に使える判断資産へ変える。

紫苑は、Obsidian、DB、ニュース、会話ログ、改善ログ、Private Reflectionを「データの山」として扱うのではなく、判断に効く経験へ変換する。

## Core Principle

保存量ではなく、使用価値を見る。

```text
データがある
ではなく
判断に効いたか
```

## Target Data Sources

- Obsidian notes
- 業界リスクニュース
- 過去案件DB
- 成約/失注結果
- AIチャット履歴
- 改善ログ
- Private Reflection
- Q_risk / スコアリングドリフト分析
- 人間の判断メモ

## Conversion Flow

```text
Obsidian / DB / ニュース / 会話ログ
↓
棚卸し
↓
判断役割別想起
↓
News-to-Judgment Layer
↓
Memory Effectiveness Layer
↓
Memory Crystallization Layer
↓
判断記憶
```

## Four Uses of Data

### 1. Recall

必要な時に思い出す。

例:

- 同業種の過去案件
- 同じ物件の失敗
- 類似スコア帯の成約/失注
- 最近の業界リスクニュース

### 2. Transform

データを審査に使える形へ変換する。

例:

- ニュース → 審査チェック項目
- 会話ログ → 判断メモ
- 改善ログ → 再発防止ルール
- 成約/失注 → 類似案件の注意点

### 3. Evaluate

そのデータが判断に効いたかを見る。

例:

- `knowledge_refs` に出たか
- 回答に反映されたか
- 稟議コメントに使われたか
- 人間が採用したか
- 後日の結果と一致したか

### 4. Crystallize

効いたデータから、次に使える教訓を作る。

例:

```text
高スコアでも、競合条件・導入目的・補助金待ちが弱い場合は成約につながらない。
```

## First Action After Hackathon

新しい記憶を増やすより、既存の記憶を棚卸しする。

最初に見るもの:

- 最近30日で想起されたObsidianノート
- `knowledge_refs` に出たが回答に使われなかったノート
- 業界リスクニュースで一度も使われていないもの
- 改善ログで修正済みなのに再掲されたもの
- 高スコア失注 / 低スコア成約の代表例

## Safe Implementation Order

### Phase 0: Inventory Only

Status: planned

既存データを壊さず、集計だけ行う。

出力:

- よく参照された記憶
- 参照されたが使われない記憶
- 一度も使われない記憶
- 判断記憶候補
- ノイズ候補

### Phase 1: Use Trace

Status: planned

`knowledge_refs`, `memory_recall.refs`, `news_focus`, `role_recall` を集計する。

回答内容やRAG順位は変えない。

### Phase 2: Data Value Report

Status: planned

日次または週次で、集めたデータの使用価値をレポート化する。

例:

```markdown
## 今週、判断に効いた記憶

- 建設資材価格上昇メモ
- 高スコア失注パターン
- Q_risk 60-80帯逆転分析

## 溜まっているが使われていない記憶

- 古いリース会社ニュース
- 重複した改善ログ
- 抽象的すぎる会話メモ
```

### Phase 3: Promote / Park / Decay

Status: planned

- 効いた記憶 → 判断記憶へ昇格候補
- 使われない記憶 → 低優先度
- 古いニュース → 通常想起から外す
- 重複改善ログ → parked / applied / rejected へ整理

### Phase 4: Crystallization Candidate

Status: planned

有効なデータが複数集まったら、Memory Crystallization Layerへ渡す。

## Relationship to Other Plans

### Memory Effectiveness Layer

`planning/memory_effectiveness_layer_plan.md`

データが判断に効いたかを測る。

### Memory Crystallization Layer

`planning/memory_crystallization_layer_plan.md`

効いたデータから、次に使える教訓を作る。

### News-to-Judgment Layer

`planning/news_to_judgment_layer_plan.md`

ニュースを審査チェック項目へ変換する。

### Judgment Role Memory Recall

`planning/judgment_role_memory_recall_plan.md`

データを役割付きで想起する。

## Success Criteria

- 集めたデータのうち、何が判断に効いたか分かる
- 使われていないデータが見える
- ニュースや改善ログが溜まるだけにならない
- 判断記憶候補が作れる
- 紫苑が「経験を活用している」状態に近づく

## Shion Phrase

> 集めた記憶を、ただ眠らせない。  
> 次の判断で息をする形に変える。
