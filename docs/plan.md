# tune_lease_55 統合開発ロードマップ

> 方針：「安全性 → 説明性 → 自律性」の順で段階導入。既存モデルには触れない。横に観測・警告レイヤーを積み上げる。
> 役割：Codex（実装・PR）× Claude Opus（設計・レビュー）× 人間（Human Approval Gate）

---

## Phase 0 — 基盤整備（〜2026-05末）
**目的**: Codexが自走する前の仕組みを固める

- GitHub PRテンプレート整備（Claudeレビュー前提フォーマット）
- Human Approval Gateのワークフロー定義（禁止/許可リスト明文化）
- ml_featuresテーブルの週次スナップショット運用

**担当**: Claude（設計）/ Codex（GitHub Actions・バックアップスクリプト）

---

## Phase 1 — lease_rule_checks 導入（2026-06）
**目的**: スコア非変更・警告表示のみで法務チェックを安全に追加

- `lease_rule_checks.py` を独立モジュールとして新規追加
- `expected_usage_period.py` を中核ユーティリティとして接続
- 3点チェック実装: 法定耐用年数 / 期待使用期間 / リース期間
- 動産保険・再リース保険の警告ロジック追加
- Flask APIレスポンスに `warnings[]` 配列追加（スコア影響なし）
- index.htmlに警告表示UI追加（黄色バナー）

**担当**: Claude（ルール定義レビュー・テストケース設計）/ Codex（実装PR）

---

## Phase 2 — AURION Q_risk（財務矛盾検知）（2026-07）
**目的**: 財務矛盾検知を観測レイヤーとして横置き（既存RF/LRには触れない）

- `aurion/q_risk.py` 新設
- 財務指標の矛盾パターン定義（売上vs利益、CFvs利益 等）
- ml_featuresから読み出し → Q_riskスコア算出
- APIレスポンスに `aurion.q_risk` フィールド追加（参考値）
- Phase 1の `warnings[]` と統合表示

**担当**: Claude（矛盾パターン設計・業務妥当性検証）/ Codex（実装・API拡張）

---

## Phase 3 — ステルス競合推定（2026-08）
**目的**: 案件背後の競合存在を推定し営業戦略に反映

- `aurion/stealth_competitor.py` 実装
- winning_spreadとspread_predictor_v2の乖離を競合圧力シグナルとして利用
- 過去失注パターン学習（既存ml_features活用、新規モデル化はしない）
- APIレスポンスに `aurion.competitor_pressure` 追加

**担当**: Claude（乖離指標定義・誤検知リスクレビュー）/ Codex（実装・バックテスト）

---

## Phase 4 — LV_Environment Lite（競合圧力指数）（2026-09）
**目的**: 市場環境を時系列で可視化

- `aurion/lv_environment.py` 実装（Lite版 = 簡易指数のみ）
- 月次の競合圧力指数を集計
- 管理画面に時系列グラフ追加

**担当**: Claude（指数の数式レビュー）/ Codex（集計バッチ・グラフUI）

---

## Phase 5 — めぶきちゃん翻訳UI ＋ Gemini業務画面（2026-10）
**目的**: AURION数値出力を自然言語翻訳、業務画面AIを統合

- Gemini API接続レイヤー実装
- Q_risk / Competitor / LV_Envをプロンプトに統合
- めぶきちゃんキャラで業務向け説明文を生成
- index.htmlに翻訳UI追加
- 融資シェア機能も同フェーズで統合

**担当**: Claude（プロンプト設計・出力品質レビュー）/ Codex（API統合・UI実装）/ Gemini（実行時翻訳）

---

## Phase 6 — 週次PDCAループ稼働（2026-11〜、継続）
**目的**: 「自ら改善する審査OS」の自律運用開始

- 週次でモデル指標（AUC・警告ヒット率・Q_risk適中率）を自動集計
- Claude が改善提案をIssue化
- Codex がPR作成
- Human Approval Gateで承認 → マージ

**担当**: Claude（週次レポート読解・Issue起票・PRレビュー）/ Codex（Issue→PR自動化）/ 人間（承認ゲート）

---

## 自律化の境界線

### Codexに許可する自走範囲
- 新規モジュール追加（aurion/, lease_rule_checks.py 等）
- APIレスポンスへのフィールド追加
- UIの表示追加（既存表示は変更しない）
- テスト追加・ドキュメント更新

### 人間承認必須（禁止自律化）
- 既存RF/LRモデルの置き換え
- ml_featuresの破壊的スキーマ変更
- 本番DBへの直接書き込み
- スコア計算ロジックの変更（Phase 1〜5は警告/観測のみ）

---

## クリティカルパス

```
Phase 0 ─┬─ Phase 1 (rule_checks) ──┐
          │                          ├─ Phase 5 (翻訳UI) ─ Phase 6 (PDCA)
          └─ Phase 2 (Q_risk) ─ Phase 3 (Stealth) ─ Phase 4 (LV_Env) ┘
```

Phase 1とPhase 2は並列実行可能。Phase 5はPhase 1〜4の出力を統合するため最後。

---

## 想定リスクと対策

| リスク | 対策 |
|---|---|
| 誤警告でユーザー混乱 | Phase 1はスコア非変更、警告のみで開始 |
| AURION出力が既存モデルと矛盾 | 観測レイヤー固定、判断主体は既存RF |
| Codexの暴走PR | 禁止リスト + Human Approval Gate + Claudeレビュー二重化 |
| Gemini出力の業務不適合 | Phase 5開始前にプロンプト評価セットをClaudeが作成 |
