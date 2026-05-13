# tune_lease_55 開発計画（Spec-Driven Development）

> 方針：「安全性 → 説明性 → 自律性」の順で段階導入。既存モデルには触れない。
> 体制：Claude Sonnet（SPEC執筆・レビュー）× Codex（SPEC準拠実装）× 人間（2段承認ゲート）

---

## SDDワークフロー（全Phase共通）

```
Step 1: SPEC Drafting    → Claude Sonnet が specs/phaseN/ に draft 作成
Step 2: SPEC Review Gate → Human 承認 → status: approved に変更（★承認ゲート①）
Step 3: Implementation   → Codex が SPEC のみを読んで実装・テスト・PR作成
Step 4: Implementation Review → Claude が SPEC準拠性・AC網羅性をレビュー
Step 5: Merge Gate       → Human 最終承認 → main マージ（★承認ゲート②）
```

**Codexへの引き渡し定型文：**
```
specs/phaseN/PX-YYY-xxx.md を読み、その内容のみを実装してください。
- status: approved のSPECのみ実装可
- SPEC外の機能追加は禁止（疑問は質問で返す）
- 全AC-xxxに対応するテストを作成
- PRタイトル: "[PX-YYY] タイトル"
```

---

## Phase 0 — 基盤整備（〜2026-05末）
SPEC作成：Phase着手前にClaudeが一括作成
- GitHub PRテンプレート（SPEC準拠チェックリスト付き）
- Human Approval Gateのワークフロー定義（禁止/許可リスト）
- ml_featuresテーブルの週次スナップショット
- `scripts/gen_tests_from_spec.py`（AC-xxxからテストスケルトン生成）

## Phase 1 — lease_rule_checks（2026-06）
SPEC作成：Phase 0完了後
- `lease_rule_checks.py` 独立モジュール
- 法定耐用年数 / 期待使用期間 / リース期間 3点チェック
- 動産保険・再リース保険の警告ロジック
- APIレスポンスに `warnings[]` 追加（スコア影響なし）
- index.htmlに警告UI（黄色バナー）

## Phase 2 — AURION Q_risk（2026-07）
SPEC作成：Phase 1主要SPEC承認後（Phase 1と並行可）
- `aurion/q_risk.py`（財務矛盾検知、既存RF/LRには触れない）
- 財務矛盾パターン定義
- APIに `aurion.q_risk` フィールド追加（参考値）

## Phase 3 — ステルス競合推定（2026-08）
SPEC作成：Phase 2完了後
- `aurion/stealth_competitor.py`
- winning_spread乖離を競合圧力シグナルとして活用
- APIに `aurion.competitor_pressure` 追加

## Phase 4 — LV_Environment Lite（2026-09）
SPEC作成：Phase 3完了後
- `aurion/lv_environment.py`（月次競合圧力指数）
- 管理画面に時系列グラフ

## Phase 5 — めぶきちゃん翻訳UI + Gemini（2026-10）
SPEC作成：Phase 4完了後
- Gemini API接続レイヤー
- Q_risk / Competitor / LV_Env をプロンプト統合
- めぶきちゃん自然言語解説UI
- 融資シェア機能統合

## Phase 6 — 週次PDCAループ（2026-11〜）
SPEC作成：Phase 5完了後（継続的に追補）
- 週次指標自動集計
- Claude → Issue起票 → Codex PR → Human承認 → マージ

---

## 自律化の境界線

### Codexに許可する自走範囲
- `approved` 済みSPECの実装
- 新規モジュール追加（aurion/, lease_rule_checks.py 等）
- APIフィールド追加、UI表示追加、テスト追加

### 人間承認必須（禁止自律化）
- 既存RF/LRモデルの置き換え
- ml_featuresの破壊的スキーマ変更
- 本番DBへの直接書き込み
- スコア計算ロジックの変更

### SPEC変更ルール
承認済みSPECの変更は `specs/adr/ADR-xxx.md` で記録・再承認。直接書き換え禁止。

---

## ディレクトリ構成

```
tune_lease_55/
├── specs/
│   ├── README.md          # SPEC運用ガイド
│   ├── _template/
│   │   └── SPEC_TEMPLATE.md
│   ├── phase0/
│   ├── phase1/
│   │   └── INDEX.md       # Phase内SPEC一覧と依存関係
│   ├── phase2/ 〜 phase6/
│   ├── adr/               # 承認済みSPECの変更記録
│   └── glossary.md        # ドメイン用語集
├── docs/
│   └── plan.md（本ファイル）
└── tests/
    └── spec_traceability.md
```

---

## クリティカルパス

```
Phase 0 ─┬─ Phase 1 ──────────────────────┐
          │                                ├─ Phase 5 ─ Phase 6
          └─ Phase 2 ─ Phase 3 ─ Phase 4 ──┘
```
