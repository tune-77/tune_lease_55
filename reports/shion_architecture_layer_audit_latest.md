# 紫苑 アーキテクチャ層監査

- Date: 2026-08-21
- Mode: read_only_architecture_audit
- Guardrail: read_only_no_prompt_no_scoring_no_rag_no_memory_write_no_auto_promotion
- Overall status: attention

## これは何か

HOIKUAGENT 型の `harness / agents / improver` という見方を、紫苑に壊さず当てはめた read-only 監査です。
既存の回答生成、RAG、スコアリング、日次改善パイプラインには接続しません。

## Summary

- Active judgment assets: 9
- Lineage assets: 1
- Field feedback sleeping: 9
- Field validation score: 12.0
- Growth: 育っている途中
- Loop engineering: warn

## 3層の見立て

### harness: 型と安全ゲート

- Status: usable
- Role: 必須確認、様式、承認、記憶汚染防止、評価ゲートを決定的に管理する層。
- 強み:
  - 評価ヘルス画面で成長・記憶・改善ログを読める
  - 判断資産昇格スクリプトがあり、正式資産の入口を分けられる
  - preflight guard により実装前後の最低限の破綻を検出できる
  - 判断資産系統樹で親子関係を読み取り専用に可視化している
- リスク:
  - 正式判断資産への反映可否を人間が確認する専用画面はまだ薄い
- 安全な次の一手: 正式判断資産への昇格条件を read-only レポートで定義し、いきなり自動適用しない。

### agents: 判断を作るAI

- Status: attention
- Role: 審査コメント、確認質問、調査、レビューを担うAIの層。
- 強み:
  - 審査画面・リース知性体画面があり、判断生成と対話の入口が分かれている
  - 評価GUIにより回答品質を実行後に点検できる
  - 記憶・RAG参照の効果測定レポートがあり、参照しただけで終わらない
- リスク:
  - 作成AI・レビューAI・調査AIの役割境界は概念としてはあるが、harnessが強制する構造にはまだ寄せ切っていない
  - どの判断資産を使ったかを案件ごとの manifest として保存する仕組みは今後の課題
- 安全な次の一手: 各回答で使った判断資産・根拠・質問を manifest として観測する案を作る。まだ回答生成には接続しない。

### improver: 判断資産を育てる層

- Status: attention
- Role: 実案件フィードバック、改善ログ、系統樹から再利用できる勘所を提案する層。
- 強み:
  - 改善ログと loop engineering により改善ループの状態を読める
  - 判断資産フィードバック記録スクリプトがあり、実案件反応を後から残せる
  - 判断資産に親子系統を持たせ始めており、派生の履歴を追える
- リスク:
  - active 判断資産 9 件のうち、実案件フィードバック未記録が 9 件ある
  - 改善提案を正式資産にする承認ゲートは、まず read-only 設計から固めるべき
- 安全な次の一手: 未使用判断資産を削除せず、次の実案件で1件ずつ helped/neutral/challenged を記録する。

## 壊さないためのガードレール

- 既存の回答生成、RAG検索、スコアリング、日次改善パイプラインには接続しない。
- AI提案は正式判断資産に自動昇格しない。
- 未承認のAI出力を長期記憶へ同期しない。
- まず観測レポートとして不足を見える化し、人間承認後に小さく実装する。

## Safe Sequence

- 1. read-only 監査レポートで層の不足を見る。
- 2. 実案件フィードバックを1件ずつ記録する。
- 3. 承認状態ラベル案をレポートで固める。
- 4. 人間承認後にだけ判断資産へ反映する。
