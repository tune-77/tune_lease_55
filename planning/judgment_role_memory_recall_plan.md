# 判断役割別想起システム 計画

## 目的

紫苑の記憶を「保存して検索する」段階から、**判断の役割ごとに呼び分ける**段階へ進める。

今の課題は、記憶があること自体ではなく、どの記憶を何のために呼び出し、実際に審査判断へ効いたかを管理すること。

目指す状態:

> この案件では、根拠記憶を3件、反証記憶を2件、失敗記憶を1件、稟議表現記憶を1件呼び出しました。

これにより、紫苑は単なるRAG検索ではなく、判断に必要な役割の記憶を明示して使えるようになる。

## 背景

現在の紫苑システムは、Obsidian、会話ログ、改善ログ、Private Reflection、memory_debug、knowledge_refs、memory_recall.refs により、記憶を保存し、参照したかを観測できる。

次の段階では、記憶の粒度を上げる。

- これは判断を支える根拠か
- これは反対方向を見る反証か
- これは過去の失敗か
- これは条件付き承認の知恵か
- これは稟議や顧客説明の表現か
- これはUserの判断癖か
- これは良心・説明責任チェックか

やること自体は同じ。

```text
記録する -> 分ける -> 呼び出す -> 使う -> 結果を見る -> 直す
```

ただし粒度を「メモ単位の検索」から「判断役割単位の想起」へ上げる。

## 役割ラベル案

| ラベル | 役割 | 例 |
|---|---|---|
| `evidence` | 判断を支える根拠記憶 | 似た案件で成約・正常推移した記録 |
| `counter_evidence` | 反証・慎重判断用記憶 | 通りそうに見えたが危なかった案件 |
| `similar_case` | 類似案件記憶 | 業種・物件・金額帯・商流が似た案件 |
| `failure_lesson` | 見落とし・失敗の教訓 | 後で問題化した論点、薄かった回答 |
| `approval_condition` | 条件付き承認記憶 | 保全条件、追加確認、金利条件 |
| `wording` | 表現記憶 | 稟議コメント、顧客説明、審査部向け表現 |
| `user_preference` | User判断傾向記憶 | よく重視する観点、過去の修正傾向 |
| `conscience_check` | 良心・説明責任チェック | 人を雑に切っていないか、迎合していないか |

1つの記憶に複数ラベルを許可する。

例:

```json
{
  "memory_id": "case_2026_001",
  "roles": ["similar_case", "counter_evidence", "approval_condition"],
  "confidence": 0.78
}
```

## Ghost Recall: 違和感想起

記憶は分類だけではない。

明示的に「根拠」「反証」「失敗」と分類できる記憶とは別に、理由がまだ言語化されていない違和感、既視感、不自然さ、過去の失敗の気配を拾う層を置く。

呼び名:

- `ghost_recall`
- 違和感想起
- 囁き層

役割:

- 判断を直接決めない
- スコアを自動で下げない
- 「確認すべき問い」を発火させる
- 必要なら `failure_lesson`, `counter_evidence`, `approval_condition` を呼び出す

原則:

> 分類された記憶は、根拠になる。  
> 囁く記憶は、問いになる。

出力例:

```text
ゴースト想起:
過去の失敗案件と営業メモの語り口が似ています。
ただし、これは否決理由ではありません。

確認問い:
- 導入目的の具体性
- 返済原資
- 銀行支援の実効性
```

### Q_riskとの接続

Q_riskは、Ghost Recallの入口として使える。

Q_riskは直接の減点装置ではなく、**数字側の違和感センサー**として扱う。

```text
Q_riskが高い
-> 数字の歪み・不自然さを検知
-> ghost_recall を発火
-> 類似する失敗記憶・反証記憶・条件付き承認記憶を呼ぶ
-> 否決ではなく、確認質問と条件案へ変換する
```

Q_risk発火時に優先する役割:

- `failure_lesson`
- `counter_evidence`
- `approval_condition`
- `similar_case`
- `conscience_check`

### Q_risk Core / Ghost Trigger / Ghost Recall の分離

今後の実装では、Q_risk本体を無制限に肥大化させない。

役割を次の3層に分ける。

#### 1. Q_risk Core

冷たい数理センサー。

主に財務・数値の歪みを見る。

- 財務数値の異常値
- 利益・資産・借入のバランスの悪さ
- スコアは良いが財務構造に歪みがあるケース
- 過去の注意案件と似た数値パターン

Q_risk Coreは、直接の否決・減点ではなく、確認対象を発見する。

#### 2. Ghost Trigger

非数値の違和感センサー。

Q_risk Coreとは分けて管理し、営業メモ・導入目的・物件保全・競合環境などから違和感を発火させる。

例:

- 営業メモの語り口が曖昧
- 導入目的が収益に直結していない
- 銀行支援の実効性が弱い
- 財務スコアは良いのに説明が薄い
- 競合金利に負ける可能性が高い
- 代表者依存が強い
- 物件保全と事業計画が噛み合っていない
- 過去の失敗案件と似た構造がある

Ghost Triggerは、スコアを下げるためではなく、Ghost Recallを呼ぶための発火条件。

#### 3. Ghost Recall

発火した違和感に応じて、過去記憶を呼ぶ層。

- `failure_lesson`
- `counter_evidence`
- `approval_condition`
- `similar_case`
- `conscience_check`

呼び出した記憶は、否決理由ではなく「確認問い」や「条件付き承認案」に変換する。

整理:

```text
Q_risk Core:
数字の歪みを検知する

Ghost Trigger:
数字以外の違和感も発火条件にする

Ghost Recall:
発火した違和感に合う失敗記憶・反証記憶・条件記憶を呼ぶ
```

この分離により、Q_riskをブラックボックスな総合減点装置にせず、違和感を説明可能な確認プロセスへ接続する。

出力方針:

- 「危険だから否決」ではなく「確認すべき歪み」として出す
- 財務スコアが良い案件ほど、Q_riskの違和感を丁寧に扱う
- 営業メモや物件保全と矛盾する場合は、確認問いへ落とす
- 説明不能な警戒を、そのまま審査判断には使わない

例:

```text
Q_riskによる違和感:
財務スコアは一定水準ですが、利益・資産・借入のバランスに過去の注意案件と似た歪みがあります。

これは否決理由ではありません。
ただし、以下を確認してください。

- 売上増加が一過性ではないか
- 導入設備が本当に収益に直結するか
- 銀行支援が口頭ではなく継続的な枠として存在するか
- 代表者依存が強すぎないか
```

## 実装方針

今すぐ本流に入れると、既存RAG・回答品質・ハッカソン提出物を壊す可能性があるため、段階的に進める。

### Phase 0: 計画固定

- 本計画書を作成する
- READMEや動画では「今後の拡張余地」として触れる程度に留める
- 既存のObsidian検索・AIチャット・審査導線には変更を入れない

### Phase 1: ルールベース分類

既存の記憶・改善ログ・過去案件に対して、キーワードとメタデータで仮ラベルを付ける。

例:

- 「否決」「事故」「延滞」「見落とし」 -> `counter_evidence`, `failure_lesson`
- 「条件」「保全」「追加確認」「金利」 -> `approval_condition`
- 「稟議」「コメント」「説明」 -> `wording`
- 類似案件ID・業種・物件・金額帯 -> `similar_case`

出力先候補:

- `data/shion_memory_role_index.json`
- `reports/shion_memory_role_index_latest.md`

### Phase 2: AI一次分別

AIに記憶本文を読ませ、役割ラベルを提案させる。

重要:

- AIのラベルは確定ではなく `suggested_roles` とする
- `confidence` と `reason` を必ず持たせる
- `user_preference` や `conscience_check` は過剰検出しない

出力例:

```json
{
  "memory_id": "obsidian:Projects/tune_lease_55/AI Chat/2026-07-11.md#001",
  "suggested_roles": ["failure_lesson", "approval_condition"],
  "confidence": 0.72,
  "reason": "条件付き承認時の見落としと追加確認に関する記述があるため"
}
```

### Phase 3: 人間確認

改善ログ画面または専用画面で、AIの分類を人間が確認する。

必要な操作:

- 採用
- 修正
- 却下
- 複数ラベル追加

確定ラベルは `confirmed_roles` として保存する。

### Phase 4: 役割別想起

審査回答前に、案件文脈から必要な役割を選ぶ。

例:

- 通りそうな案件 -> `counter_evidence`, `failure_lesson` を優先
- 否決寄り案件 -> `approval_condition`, `similar_case` を優先
- 稟議作成 -> `wording`, `approval_condition` を優先
- ユーザーが違和感を言った -> `failure_lesson`, `user_preference` を優先

回答時には、内部で次のように保持する。

```json
{
  "recall_plan": {
    "evidence": 3,
    "counter_evidence": 2,
    "failure_lesson": 1,
    "wording": 1,
    "ghost_recall": 1
  }
}
```

Q_riskが高い、または財務スコアと営業メモ・物件保全・導入目的の間に違和感がある場合は、`ghost_recall` を読み取り専用で発火する。

この段階では回答を強制変更せず、`memory_debug.role_recall` に「発火した理由」「呼んだ記憶」「確認問い」を出すだけにする。

### Phase 5: 効果測定

役割別想起が回答品質に効いたかを観測する。

見る指標:

- `memory_debug.role_recall.used`
- `memory_debug.role_recall.ghost_recall_triggered`
- 役割別参照件数
- 回答に反映された役割数
- Human Response Feedback
- 「薄い」「一般論」「紫苑らしくない」の減少
- 稟議コメント採用率
- 条件付き承認案の修正率

### Phase 6: 学習分類器

確定ラベルが溜まったら、RandomForestやLogisticRegressionで分類補助を作る。

目的:

- AI分類の補助
- ルールとAI分類のズレ検知
- 構造化特徴を使った説明可能な分類

候補特徴量:

- 業種
- 物件種別
- 金額帯
- スコア帯
- final_status
- キーワード
- 過去に参照された回数
- 参照後の人間評価
- 審査結果との関係

ただし、教師データが少ない段階では導入しない。

## 安全方針

- 既存RAG導線をすぐ置き換えない
- 最初は別ファイル・別API・別画面で観測する
- 役割ラベルは提案扱いから始める
- 本番回答へ強制注入しない
- `user_preference` や個人情報に関わる記憶は慎重に扱う
- Cloud Runデプロイは明示依頼があるまで行わない

## News-to-Judgment Layer との接続

外部ニュースはそのまま記憶に入れず、`planning/news_to_judgment_layer_plan.md` の方針に従って、審査チェック項目へ変換してから役割別想起に渡す。

接続先:

- `evidence`: 業界・市況が案件判断を補強する場合
- `counter_evidence`: スコアは良いが外部環境が悪化している場合
- `approval_condition`: 価格転嫁、稼働率、補助金、工期などを条件にする場合
- `conscience_check`: ニュースを過信していないか、対象企業に本当に関係するかを確認する場合

Q_risk が数値側の違和感を見るのに対し、News-to-Judgment Layer は外部環境側の違和感を見る。

## Memory Effectiveness Layer との接続

役割別想起で呼んだ記憶は、`planning/memory_effectiveness_layer_plan.md` の方針に従って、呼ばれただけか、回答や審査コメントに実際に効いたかを観測する。

最初は `memory_debug` への観測だけに留め、RAG順位や回答内容は変えない。

## 最初に作るべきもの

1. `data/shion_memory_role_index.json` のスキーマ
2. ルールベース分類スクリプト
3. `reports/shion_memory_role_index_latest.md`
4. memory_debug に出すだけの読み取り専用 `role_recall` 情報
5. 小さなテストデータでの分類精度確認

## 一言

記憶を「倉庫」から「道具箱」に変える。

紫苑は、ただ覚えているだけではなく、今この判断に必要な記憶を、役割付きで思い出すようになる。
