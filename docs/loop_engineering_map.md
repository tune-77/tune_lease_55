# Loop Engineering Map

この文書は、tune_lease_55 の改善ループを増やすためではなく、既存ループの正本・派生物・観測点を固定するための地図である。

## 方針

- 正本を増やさない
- 既存パイプラインを止めない
- 読み取り専用の観測から始める
- 自動反映より、誤学習の抑制を優先する
- 判断基準は「作ったか」ではなく「再利用され、改善したか」に置く

## 主要ループ

| ループ | 入力 | 正本 | 派生物 | 観測点 |
| --- | --- | --- | --- | --- |
| 改善候補ループ | Obsidian改善タグ、診断出力、改善レポート | `reports/latest.json` | `reports/improvement_report_*.json` | applied / needs_review / failed |
| 再帰的自己改善ループ | 改善レポート、prompt feedback、Obsidian候補 | `reports/latest.json`, `ledger.jsonl` | `reports/recursive_self_improvement_*.json` | repeat_issue_rate / reuse_rate / noise_rate |
| AI応答改善ループ | AI応答ログ、PDCAルール、前回差分 | `data/prompt_feedback_log.jsonl` | prompt feedback summary | pdca_rate / response_changed_rate |
| 知識昇格ループ | 日次メモ、Obsidianメモ、wiki候補 | `MEMORY.md`, Obsidian Vault | wiki promotion queue | 知識化率 / 再利用率 |
| モデル監視ループ | scoring harness、ドリフト分析、成約/失注 | DB、モデル成果物 | scoring reports | AUC差分 / segment gap / drift |

## 最小実装の範囲

今回の最小版は、以下に限定する。

1. この地図を追加する
2. 既存成果物を読み取り専用で集計する
3. `reports/loop_engineering_latest.json` と Markdown を生成できるようにする
4. 日次パイプラインやUIにはまだ接続しない

## 現在測れる指標

| 指標 | 読み取り元 | 意味 |
| --- | --- | --- |
| applied_count | `reports/latest.json` | 自動または手動で反映済みの改善数 |
| needs_review_count | `reports/latest.json` | 人間レビューが必要な改善数 |
| failed_count | `reports/latest.json` | 反映失敗数 |
| pdca_rate | prompt feedback summary | PDCAルールが応答に入った割合 |
| response_changed_rate | recursive report / prompt feedback | 前回応答との差分が出た割合 |
| repeat_issue_rate | recursive report | 重複・再発候補の割合 |
| reuse_rate | recursive report | 台帳上すでに扱った候補の再利用・再検出割合 |
| noise_rate | recursive report | 抑制候補の割合 |

## まだ測らない指標

以下は有用だが、最小版では実装しない。

- knowledge_reuse_quality: 参照知識が回答に実際に使われた率
- judgment_change_rate: 知識・ニュース・過去ログで判断が変わった率
- stale_rule_rate: 古いPDCAルールや記憶が残っている率
- loop_latency: 観測から反映までの日数

## 自動反映してよい領域

- UI文言
- 説明補強
- ドキュメント補足
- Obsidianリンク整理
- 重複候補の抑制
- 低リスクのプロンプト補助

## 人間レビュー必須領域

- スコアリング本体
- DBスキーマ
- API契約
- モデル再学習
- 外部連携
- インフラ
- 紫苑の長期記憶・人格・判断方針に影響する変更

## 次の候補

- P6-003: loop metrics のUI表示
- P6-004: Obsidian判断信号の抽出
- P6-005: auto fix policy のリスク判定強化

