# News-to-Judgment Layer Plan

## Purpose

ニュース収集を「読むための保管」ではなく、リース審査で使える判断トリガーへ変換する。

紫苑はリース会社のニュースを集めるのではなく、借手の返済力、設備稼働、物件価値、投資回収、資金繰りに影響する外部環境を見に行く。

## Core Principle

ニュースは、そのままでは記憶ではない。

審査時に「何を疑うべきか」「何を確認すべきか」「どの条件を変えるべきか」に変換されて、はじめて判断記憶になる。

```text
業界リスクニュース
↓
業種・物件・リスク分類
↓
審査チェック項目
↓
案件入力時の想起
↓
稟議コメント・確認事項・条件案
↓
実際に使われたものだけ昇格
```

## Target News Domains

- 製造業: 工作機械受注、設備投資、半導体投資、電気代、輸出、為替
- 建設業: 公共工事、建設資材価格、建設業倒産、建機中古価格、人手不足
- 運送業: 燃料費、2024年問題、ドライバー不足、運賃改定、商用車価格
- 医療・介護: 診療報酬、設備更新、医療機器需要、人材不足
- 農業・食品: 補助金、原材料価格、農機更新、気候影響
- 共通: 金利、補助金、税制、倒産件数、中古市場、資金繰り

## Memory Layers

### Layer 1: Raw News

収集したニュース本文・出典・日付。

保存するが、通常の審査判断では低優先度。

### Layer 2: News Summary

3行要約、対象業界、対象物件、情報源、重要度。

### Layer 3: Screening Impact

リース審査への影響を短く記録する。

例:

```text
建設資材価格上昇により、工期遅延・利益率低下・資金繰り圧迫の可能性。
建設業案件では受注済工事の採算、価格転嫁、導入建機の稼働予定を確認する。
```

### Layer 4: Recall Trigger

案件入力時に照合するタグ。

- industry
- asset
- region
- risk_flags
- recommended_checks
- valid_until
- source_reliability

### Layer 5: Judgment Memory

実際にAIチャット、審査コメント、稟議コメント、条件付き承認案で使われたニュースだけ昇格する。

## Example Transformations

### Construction

```text
ニュース:
建設資材価格が上昇

審査確認:
- 受注済工事の採算は維持できているか
- 価格転嫁できているか
- 工期遅延時の支払原資はあるか
- 導入する建機の稼働予定は確定しているか
```

### Transportation

```text
ニュース:
燃料費上昇・人手不足

審査確認:
- 運賃改定できているか
- 車両稼働率は維持できるか
- ドライバー確保はできているか
- 車両更新で本当に利益改善するか
```

### Manufacturing

```text
ニュース:
工作機械受注が減少

審査確認:
- 導入設備の受注見込みはあるか
- 稼働率は十分か
- 既存設備更新か、新規投資か
- 投資回収期間は妥当か
```

## Implementation Plan

### Phase 0: Morning Intelligence Brief

Status: implemented locally / review-gated

毎朝の調査は、ニュースを大量に保存するためではなく、「今日の審査で気をつける外部環境論点」を3つ作るために使う。

朝の出力例:

```markdown
## 今日の業界リスク論点

1. 建設資材価格
   - 建設業案件では価格転嫁と工期遅延を確認

2. 運送業の燃料費
   - 車両案件では運賃改定と稼働率を確認

3. 工作機械受注
   - 製造業案件では導入設備の受注見込みを確認
```

運用ルール:

- 毎朝、外部ニュースを集める
- その中から審査に効きそうな論点を最大3つだけ昇格する
- 各論点に `industry`, `asset`, `risk_flags`, `recommended_checks`, `valid_until` を付ける
- その日の案件で使われなければ低優先度へ落とす
- 1週間使われなければ通常想起から外す
- 実際に審査コメントやAI回答に使われたら判断記憶へ昇格する

2026-07-28 implementation:

- `lease_news_digest.py` に `build_news_judgment_signals` / `write_news_judgment_signals` / `read_news_judgment_signals` を追加した。
- `scripts/build_news_judgment_signals.py` で、Obsidianに保存済みのニュースから `data/news_judgment_signals.jsonl` と `data/news_judgment_signals_latest.json` を再生成できる。
- `scripts/collect_lease_news_to_obsidian.py` は、日次ニュース収集後にニュース判断信号も更新する。
- `api/main.py` の審査向け判断資産候補選択に `source=news_judgment_signals` を合流させた。ただし `promotion_status=news_signal` のままで、判断資産本流へは自動昇格しない。
- 生成シグナルは `actionable / watch / ignore` で扱い、案件一致時だけ最大候補に混ぜる。フィードバックは既存の `judgment_asset_usage_feedback.jsonl` に戻せる。
- 同日再生成時は古い同日シグナルを置き換える。部品販売・口コミ・ランキング等のノイズは通常候補から落とす。
- LLM段階を追加した。`LEASE_NEWS_SIGNAL_LLM=1` または `scripts/build_news_judgment_signals.py --use-llm` の時だけ、Gemini が `News Judgment Signal` JSON候補を作る。LLM候補は `claim / effective_claim / use_when / do_not_use_when / recommended_checks / condition_impacts / risk_axis / valid_until / confidence` を必須にし、直接承認・否決・スコア変更・期限延長・根拠不明の断定があれば `news_signal_quarantined` に落とす。
- ループ設計は `LLM生成 -> ルール検疫 -> 案件一致時だけ提示 -> helped/noise/wrong/expiredフィードバック -> 人間レビュー -> 判断資産候補`。LLMは本流へmergeしない。
- 重複抑制を追加した。業種・物件・リスク軸・確認条件・条件影響から `signal_signature` を作り、同日内で同じ観点は1件へ折りたたむ。過去の有効期限内シグナルと同じ `signal_signature` も新規追加しない。折りたたんだ記事名は `duplicates_collapsed` に残し、`dedupe_summary` で input/output/suppressed を追える。

目的:

```text
ニュースを集める
ではなく
今日の審査で何を疑うかを作る
```

### Phase 1: Collection Axis Shift

Status: started

- `scripts/collect_lease_news_to_obsidian.py` の標準クエリを、リース会社ニュースから業界・物件・市況ニュースへ変更する。
- 新規保存先を `05-クリップ_記事/業界リスクニュース` にする。
- 旧 `リースニュース` は互換読み取りとして残す。

### Phase 2: News Classification

Status: planned

- 各ニュースに以下を付与する。
  - `industries`
  - `lease_assets`
  - `impact_direction`
  - `credit_risk_impact`
  - `screening_checks`
  - `valid_until`
  - `source_reliability`

### Phase 3: Case Matching

Status: planned

- 案件の `industry_major`, `industry_sub`, `asset_name`, `lease_asset_id`, `prefecture` とニュースタグを照合する。
- 一致したニュースだけを「今日の確認事項」として提示する。

### Phase 4: Judgment Injection

Status: planned

- AIチャット、審査アドバイス、稟議コメント案、審査討論にニュース由来の確認事項を入れる。
- ただし、ニュース本文を長く出さず、審査の問いとして出す。

### Phase 5: Effectiveness Measurement

Status: planned

- `memory_debug`, `knowledge_refs`, `memory_recall.refs` にニュース由来の参照を残す。
- 回答・稟議コメントに実際に反映されたニュースだけを `used_for_judgment=true` として昇格する。

### Phase 6: Promotion and Decay

Status: planned

- 使われたニュースは判断記憶へ昇格。
- 使われないニュース、期限切れニュース、ノイズニュースは低優先度化する。
- `valid_until` を過ぎたニュースは通常想起から外す。

## Relationship to Hierarchical Judgment Memory

このレイヤーは、判断役割別想起システムのうち `evidence`, `counter_evidence`, `conscience_check`, `approval_condition` に接続する。

Q_risk が数値側の違和感センサーなら、News-to-Judgment Layer は外部環境側の違和感センサーになる。

```text
Q_risk
  数字の歪みを見る

News-to-Judgment Layer
  外部環境の変化を見る

Ghost Recall
  数字と外部環境の違和感を、過去の失敗・条件・反証記憶へつなぐ
```

## Relationship to Memory Effectiveness Layer

`planning/memory_effectiveness_layer_plan.md` で、業界リスクニュースが実際に審査判断へ効いたかを測る。

ニュースは特に「溜まるだけ」になりやすいため、以下を必ず分ける。

- collected: 収集しただけ
- recalled: 案件で想起された
- used: 確認事項・稟議コメントに入った
- validated: 人間が採用した、または後日の結果と一致した
- noisy/expired: 想起されるが使われない、または期限切れ

## Success Criteria

- ニュースが単なる保存物ではなく、審査チェック項目として表示される。
- 審査コメントにニュース由来の確認事項が自然に混ざる。
- 使われたニュースと使われなかったニュースを区別できる。
- ニュース収集量ではなく、判断に効いた率で評価できる。

## Shion Phrase

> 外の世界で起きたことを、そのまま覚えるのではなく、次の審査で何を疑うべきかに変換する。
