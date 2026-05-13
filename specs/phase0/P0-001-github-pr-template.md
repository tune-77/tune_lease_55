---
spec_id: P0-001
phase: 0
title: GitHub PRテンプレート整備
status: draft
author: Claude Opus
reviewer: human
version: 1.0.0
depends_on: []
---

## 1. Goal
tune_lease_55リポジトリにおけるPull Requestの品質とSPEC準拠を担保するため、`.github/pull_request_template.md`を整備する。Codexが実装した変更を人間レビュアが二段承認ゲートで判定できるよう、PRに必要なチェック項目とコミット規約を明文化する。

## 2. Scope
### In Scope
- `.github/pull_request_template.md`の新規作成
- SPEC準拠チェックリスト（spec_id記載、AC対応テスト確認、SPEC外実装の不在、Claude Opus承認済み）
- コミットメッセージ規約 `[PX-YYY] <概要>` の明記
- レビュア向け確認観点（モデル置換禁止、DB直接書込み禁止等）への参照リンク（`docs/approval_gate.md`）

### Out of Scope
- CODEOWNERSの設定
- GitHub Actionsによる自動ラベリング（P0-002で対応）
- Issueテンプレートの整備

## 3. Inputs / Outputs
| 区分 | 内容 |
|------|------|
| Input | なし |
| Output | `.github/pull_request_template.md`（UTF-8、LF改行、日本語） |

## 4. Data Model
該当なし。

## 5. API / Interface
該当なし。PR作成時にGitHubが自動的にテンプレートを展開する。

## 6. Business Rules
- **BR-001**: PRタイトルは `[PX-YYY] <概要>` 形式とし、spec_idと一致させる
- **BR-002**: PR本文には対応SPECのリンクを必ず含める
- **BR-003**: チェックリスト全項目に明示的にチェックが入っていないPRはマージ禁止
- **BR-004**: コミットメッセージ先頭は `[PX-YYY]` プレフィックスを必須とする
- **BR-005**: 「Claude Opus によるSPEC承認済み」のチェックは、当該SPECのfrontmatter `status: approved` を確認した上でのみ入れる
- **BR-006**: SPEC外の実装が含まれる場合はPRを分割するか別SPEC起票

## 7. UI / UX
GitHub PR作成画面で以下のセクションが本文プレースホルダとして自動表示される：概要（What/Why）、対応SPEC、変更点サマリ、SPEC準拠チェックリスト（チェックボックス）、テスト結果、スクリーンショット/ログ（任意）、レビュア向けメモ

## 8. Error Handling
該当なし（静的ファイル）。

## 9. Acceptance Criteria
- **AC-001**: Given リポジトリに `.github/pull_request_template.md` が存在する状態で、When 新規PRを作成する、Then テンプレート本文がPR本文欄に自動展開される
- **AC-002**: Given PRテンプレートを表示した状態で、When 本文を確認する、Then 「対応SPEC ID」「SPEC準拠チェックリスト4項目」「コミットメッセージ規約」「approval_gate.mdへのリンク」が全て含まれている
- **AC-003**: Given チェックリスト4項目を含むPRを作成した状態で、When 人間レビュアがレビューを開始する、Then 各項目を独立にチェック可能なGitHub Markdownチェックボックスとして表示される
- **AC-004**: Given PRテンプレートを開いた状態で、When 「コミットメッセージ規約」セクションを参照する、Then `[PX-YYY] <概要>` 形式の具体例が最低2件記載されている
- **AC-005**: Given PRテンプレートを開いた状態で、When 「レビュア向け確認観点」を参照する、Then `docs/approval_gate.md` への相対リンクが含まれている

## 10. Non-Functional
- 文字コード: UTF-8 / LF
- 言語: 日本語
- 200行以内

## 11. Implementation Notes（Codex向け）
`.github/pull_request_template.md` を以下の章構成で作成：
```
## 概要
<!-- 何を/なぜ -->

## 対応 SPEC
- spec_id: PX-YYY
- SPEC: `specs/phaseN/PX-YYY-<slug>.md`

## 変更点
-

## SPEC 準拠チェックリスト
- [ ] PRタイトル/コミットに `spec_id` (PX-YYY) を記載した
- [ ] SPECのAC-xxxに対応するテストを `tests/spec_phaseN/test_PX-YYY.py` に追加しローカルでpassした
- [ ] SPEC外の実装（仕様追加・リファクタ・バグ修正）は含めていない
- [ ] Claude Opus によるSPEC承認（`status: approved`）を確認した

## コミットメッセージ規約
- 形式: `[PX-YYY] <概要>`
- 例: `[P0-001] add PR template`
- 例: `[P1-003] fix snapshot path bug`

## テスト結果
$ pytest tests/spec_phase0/test_P0-001.py

## スクリーンショット/ログ
<!-- 任意 -->

## レビュア向けメモ
- 承認ゲート手順は [`docs/approval_gate.md`](../docs/approval_gate.md) を参照
```
ブランチ: `feature/p0-001-pr-template`、PRタイトル: `[P0-001] add pull request template`

## 12. Test Plan
- 手動T-001: GitHub UIで新規PRドラフトを開き、テンプレートが展開されることを確認（AC-001）
- 手動T-002: 章構成/チェックリスト4項目/規約例/リンクの存在を目視確認（AC-002〜005）
