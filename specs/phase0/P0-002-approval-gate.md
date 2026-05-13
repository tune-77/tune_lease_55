---
spec_id: P0-002
phase: 0
title: Human Approval Gate ワークフロー定義
status: approved
author: Claude Opus
reviewer: human
version: 1.0.0
depends_on: [P0-001]
---

## 1. Goal
tune_lease_55の開発体制（Claude Opus × Codex × 人間）における二段承認ゲートの手順・禁止事項・GitHubラベル運用を `docs/approval_gate.md` として明文化する。

## 2. Scope
### In Scope
- `docs/approval_gate.md`の新規作成
- 承認ゲート①（SPEC承認）の手順
- 承認ゲート②（実装承認）の手順
- 禁止リスト/許可リスト（表形式）
- GitHub Labelsの名称・色・用途定義

### Out of Scope
- GitHub Labels自動同期スクリプト
- Slack/メール通知連携

## 3. Inputs / Outputs
| 区分 | 内容 |
|------|------|
| Input | なし |
| Output | `docs/approval_gate.md`（UTF-8/LF/日本語） |

## 4. Data Model
GitHub Labels定義：
| label | color(hex) | description |
|---|---|---|
| spec-draft | #cccccc | SPEC起票直後 |
| spec-approved | #0e8a16 | SPEC承認済み |
| impl-in-progress | #fbca04 | 実装中 |
| needs-human-review | #d93f0b | 人間レビュー待ち |
| impl-approved | #1d76db | 実装承認済み・マージ可 |
| blocked | #b60205 | ブロック中 |

## 5. API / Interface
該当なし（ドキュメント）。

## 6. Business Rules
- **BR-001**: SPECは `status: draft` で作成され、ゲート①通過時に `status: approved` へ更新。`draft`のまま実装着手は禁止
- **BR-002**: ゲート①はClaude Opus SPEC完成→人間承認→frontmatter更新の3ステップで完了
- **BR-003**: ゲート②はCodex PR提出→CI pass→人間がSPEC準拠確認→`impl-approved`ラベルで完了
- **BR-004**: `impl-approved`ラベル無しのPRはmasterへマージ禁止
- **BR-005**: 禁止リスト記載の操作はSPECで明示的に許可されていない限りCodexは実行禁止
- **BR-006**: 緊急対応でもゲート②スキップ禁止。`hotfix-*`ブランチ+事後SPEC起票で対応

## 7. UI / UX
docs/approval_gate.md の章構成：
1. 概要図（mermaid フロー）
2. ゲート①：SPEC承認（手順3ステップ）
3. ゲート②：実装承認（手順6ステップ）
4. 禁止リスト/許可リスト（表）
5. GitHub Labels一覧（表）
6. 例外運用
7. FAQ

## 8. Error Handling
該当なし（ドキュメント）。

## 9. Acceptance Criteria
- **AC-001**: Given `docs/approval_gate.md`が存在する状態で、When ファイルを開く、Then ゲート①・②それぞれの手順が番号付きリストで3ステップ以上で記載されている
- **AC-002**: Given ファイルを開いた状態で、When 「禁止リスト」セクションを参照する、Then 「既存学習済みモデルの置換」「`lease_data.db`への直接書込み」「masterへの直接push」を含む最低5行の表が存在する
- **AC-003**: Given ファイルを開いた状態で、When 「許可リスト」セクションを参照する、Then SPEC記載のカラム追加・新規スクリプト追加・テスト追加を含む最低5行の表が存在する
- **AC-004**: Given ファイルを開いた状態で、When 「GitHub Labels」セクションを参照する、Then 6ラベルがname/color/descriptionの3列で表形式で記載されている
- **AC-005**: Given ファイルを開いた状態で、When 冒頭の概要図を参照する、Then spec-draft→approved→impl→needs-human-review→impl-approved→mergeの遷移がmermaidまたはASCII artで図示されている
- **AC-006**: Given P0-001のPRテンプレートが存在する状態で、When テンプレート内のリンクを辿る、Then `docs/approval_gate.md`へ到達できる

## 10. Non-Functional
- 文字コード: UTF-8/LF
- 言語: 日本語
- 分量: 300〜500行

## 11. Implementation Notes（Codex向け）
`docs/approval_gate.md` に以下を記載：
- mermaid flowchart（spec-draft→approved→impl→needs-human-review→impl-approved→merge）
- ゲート①手順: 1.Claude Opus起票(draft) 2.人間レビュー 3.status: approvedに変更+spec-approvedラベル
- ゲート②手順: 1.Codex実装 2.テスト作成・pass 3.PR起票 4.needs-human-reviewラベル 5.人間がチェックリスト確認 6.impl-approvedラベル→squash merge
- 禁止リスト（最低6件）: 既存RF/LGBMモデル置換、lease_data.dbへの直接INSERT/UPDATE/DELETE、masterへの直接push、SPEC未記載の依存パッケージ追加、APIキー・DB接続情報のコミット、port 5001以外のデフォルト起動変更
- 許可リスト（最低5件）: SPEC記載のカラム追加/新規テーブル作成、scripts/配下の新規スクリプト追加、tests/spec_phaseN/配下のテスト追加・更新、mobile_app/配下のHTML/JS/CSS更新、.github/workflows/追加
- GitHub Labels6件の表（name/color/description）
- 付録: `gh label create` コマンド6件
ブランチ: `feature/p0-002-approval-gate`、PRタイトル: `[P0-002] add approval gate doc`

## 12. Test Plan
- 手動T-001: GitHub UIでmermaid図が正しくレンダリングされることを確認（AC-005）
- 手動T-002: 表崩れなしを目視確認（AC-002〜004）
- 手動T-003: P0-001のPRテンプレートからリンクが切れていないか確認（AC-006）
