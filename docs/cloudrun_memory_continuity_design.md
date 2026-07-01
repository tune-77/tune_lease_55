# Cloud Run 記憶・内省システムの継続性設計

## 結論

Cloud Run 上の紫苑の動的な記憶・内省・学習記録は、Git へ都度 push しない。
Git はコードと静的な知識パックの配布に使い、リクエスト中に生まれる記憶は GCS / DB 系の永続ストレージへ直接保存する。

## 役割分担

### 静的知識

- Obsidian 公開知識
- ChromaDB / RAG インデックス
- デモ用の初期DB
- プロンプトテンプレート

これらはビルド時またはデプロイ前に GCS / bundle へ同期する。
大きな ChromaDB は Git 管理しない。GCS に配置し、必要なら Cloud Run 再デプロイまたは起動時ロードで最新版へ寄せる。

### 動的記憶

- 紫苑レビュー
- 人間フィードバック
- 審査ループの違和感・稟議方針
- 会話品質評価
- 内省ログ
- デモ中の入力イベント

これらは Git を介さず、Cloud Run から永続ストレージへ直接書く。
現在のハッカソン構成では GCS の `cloudrun-inputs/YYYY-MM-DD/events.jsonl` を一次保存先とし、ローカル同期後に `data/cloudrun_experience_return.db` へ隔離する。

## なぜ Git push しないか

### 粒度の問題

内省や学習記録は細かく発生する。
それを都度コミットすると、コミット履歴がノイズ化し、コード変更と記憶更新が混ざる。

### リアルタイム性の問題

Git 同期をまとめると、Cloud Run 上の「現在の思考」と利用可能な記憶に遅延が出る。
逆に毎回 push すると、レスポンス性能とリポジトリ管理が悪化する。

### Cloud Run ライフサイクルの問題

Cloud Run インスタンスは停止・再起動する。
リクエスト終了前に git push が完了しない設計は、記憶欠損のリスクを持つ。
リクエスト中に永続ストレージへ append し、成功/失敗をイベント単位で扱う方が安全。

### リポジトリ肥大化の問題

ChromaDB、内省ログ、学習履歴を Git に入れると clone / build / push が重くなる。
コード配布と記憶保存を分けることで、起動性能と運用の見通しを守る。

## ハッカソン版の安全経路

1. Cloud Run は `demo.db` で動く。
2. 入力・評価・紫苑レビューは GCS event log へ append する。
3. ローカルで `scripts/sync_cloudrun_inputs_from_gcs.py` を実行する。
4. データはまず `data/cloudrun_experience_return.db` へ入る。
5. `/cloudrun-return-review` で人間が `承認 / 保留 / 破棄` を付ける。
6. `scripts/promote_cloudrun_return_data.py --apply` で承認済みだけを `data/demo.db` へ昇格する。

この経路では Cloud Run のデモ入力が `data/lease_data.db` へ直接入らない。

## 昇格方針

### 紫苑レビュー

承認済みなら、既定では `data/demo.db` の `shion_screening_reviews` へ昇格する。
これは次回の審査レビューや紫苑の判断補助に使える。

### 審査入力 / OCR

現時点では本体案件テーブルへ直接混ぜない。
Cloud Run デモ由来の入力は匿名化・欠損・文脈不足の可能性があるため、まず `cloudrun_return_promotions` へ昇格ログとして残す。
案件登録へ反映する場合は、別途、人間確認付きのマッピング画面またはインポートスクリプトを通す。

## 将来の本番構成

### 最小構成

- GCS event log: append-only の一次保存
- SQLite quarantine DB: ローカル確認
- 手動昇格: `data/demo.db` へ反映

### 強化構成

- Firestore: 会話フィードバック、内省、短い記憶イベント
- Cloud SQL: 審査レビュー、案件、統計に使う構造化データ
- GCS: RAGインデックス、ChromaDB、イベントアーカイブ、バックアップ

## 原則

- Git は記憶のリアルタイム保存先にしない。
- Cloud Run の一時ディスクを正としない。
- 動的記憶はイベント単位で append する。
- 本体 `data/lease_data.db` への昇格は通常行わず、デモ帰還データは `data/demo.db` に戻す。
- ChromaDB は大きくなるため Git ではなく GCS に置く。
- 「現在の紫苑」と「保存済みの紫苑」の差が広がらないよう、動的記憶は同期より先に永続化する。
