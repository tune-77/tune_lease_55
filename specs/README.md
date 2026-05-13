# SPEC運用ガイド

このディレクトリはtune_lease_55のSpec-Driven Development（SDD）における仕様書群を管理する。
実装はSPECを唯一の真実（Single Source of Truth）として行う。

---

## SPECの書き方

テンプレートは `_template/SPEC_TEMPLATE.md` を使う。

### ファイル命名規則
```
specs/phaseN/PN-NNN-short-name.md

例:
  specs/phase1/P1-001-lease-term-check.md
  specs/phase2/P2-001-q-risk-module.md
```

### フロントマター必須項目
```yaml
spec_id: P1-001
phase: 1
title: リース期間3点チェック
status: draft          # draft | review | approved | implemented
author: Claude Opus
reviewer: ""           # 承認者名を記入
version: "1.0"
depends_on: []         # 依存するspec_id のリスト
```

---

## ステータス管理

| ステータス | 意味 | 次のアクション |
|-----------|------|--------------|
| `draft` | Claude が作成中・未レビュー | Human がレビューして `review` へ |
| `review` | Human レビュー待ち | 承認なら `approved`、修正要求なら `draft` 差し戻し |
| `approved` | Human 承認済み・実装可 | Codex に渡して実装開始 |
| `implemented` | PRマージ済み | 変更は ADR 経由のみ |

**ルール：`approved` 以外のSPECをCodexに渡してはならない。**

---

## Codexへの渡し方

以下の定型文をそのままCodexに貼る。

```
specs/phaseN/PN-NNN-xxx.md を読み、その内容のみを実装してください。
- status: approved のSPECのみ実装可
- SPEC外の機能追加は禁止（疑問は質問で返す）
- 全 AC-xxx に対応するテストを作成
- PRタイトル: "[PN-NNN] タイトル"
```

Codexが「SPEC外の要件」を実装した場合はPRを差し戻す。実装→SPECの順は禁止。

---

## ADR（Architecture Decision Record）の使い方

承認済みSPECを変更する必要が生じた場合、SPECを直接書き換えてはならない。
必ず以下の手順を踏む。

```
1. specs/adr/ADR-NNN-short-title.md を新規作成
2. 変更理由・変更内容・影響範囲・代替案を記述
3. Human 承認後に ADR ステータスを accepted へ
4. 元SPECに `superseded_by: ADR-NNN` を記載し status を更新
5. 新しいSPECファイルを作成（バージョン番号を上げる）
```

ADRテンプレートは `_template/ADR_TEMPLATE.md` を参照（Phase 0で作成予定）。

---

## ディレクトリ構成

```
specs/
├── README.md              # 本ファイル
├── _template/
│   ├── SPEC_TEMPLATE.md   # SPEC雛形
│   └── ADR_TEMPLATE.md    # ADR雛形（Phase 0で追加）
├── phase0/                # 基盤整備SPEC群
├── phase1/                # lease_rule_checks SPEC群
│   └── INDEX.md           # Phase内SPEC一覧と依存関係
├── phase2/                # AURION Q_risk SPEC群
├── phase3/                # ステルス競合推定 SPEC群
├── phase4/                # LV_Environment SPEC群
├── phase5/                # めぶきちゃん翻訳UI SPEC群
├── phase6/                # 週次PDCA SPEC群
├── adr/                   # 承認済みSPEC変更記録
└── glossary.md            # ドメイン用語集（Phase 0で作成）
```

---

## チェックリスト（SPEC作成時）

- [ ] フロントマター全項目埋め済み
- [ ] Goal が1文で明確に書かれている
- [ ] Scope の In/Out が明示されている
- [ ] AC-xxx が Given-When-Then 形式で全て書かれている
- [ ] BR-xxx（ビジネスルール）に番号が付いている
- [ ] Implementation Notes に Codex向け注意事項がある
- [ ] depends_on が正確に記載されている
