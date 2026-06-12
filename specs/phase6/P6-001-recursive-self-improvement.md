---
spec_id: P6-001
phase: 6
title: 再帰的自己改善ループ
status: draft
author: Claude Sonnet
reviewer: ""
version: "1.0"
created: 2026-06-12
updated: 2026-06-12
depends_on:
  []
superseded_by: ""
---

# P6-001 — 再帰的自己改善ループ

---

## 1. Goal

改善候補の抽出・検証・反映・観測・再抽出を一連の閉ループとして扱い、改善結果が次の改善候補を生む再帰構造を定義する。
既存の改善パイプラインを壊さず、低リスク変更だけを自動反映し、それ以外は review に送る。

---

## 2. Scope

### In scope
- 改善パイプラインの入力源を定義する
- 改善候補の正規化・重複排除・優先度付けの方針を定義する
- 反映後の観測結果を次回入力へ戻す状態遷移を定義する
- 台帳に `candidate -> validated -> applied -> measured -> promoted/suppressed` の流れを記録する
- `prompt_feedback` と改善レポートを同じ観測層で扱う

### Out of scope
- スコアリング本体のロジック変更
- DB スキーマ変更
- 認証・外部API・インフラの自動修正拡張
- high risk 領域の自動適用
- パイプライン全体の再設計や新規収集源の大幅追加

---

## 3. Inputs / Outputs

### Inputs
| 項目名 | 型 | 必須/任意 | 説明 |
|-------|-----|----------|------|
| chat_log | list[str] | 必須 | ユーザーとの会話ログ。改善トリガーの一次入力 |
| improvement_report | dict | 必須 | 日次改善レポート。抽出済み候補と検証結果を含む |
| prompt_feedback_log | list[dict] | 任意 | `record_prompt_feedback()` で蓄積された応答差分ログ |
| obsidian_notes | list[dict] | 任意 | 実装済みメモ、改善ノート、日次ログ |
| execution_status | dict | 任意 | 自動実行の状態、quota、blocked 状態 |

### Outputs
| 項目名 | 型 | 説明 |
|-------|-----|------|
| canonical_candidates | list[dict] | 重複排除済みの改善候補 |
| ranked_queue | list[dict] | 実行順に並べた候補キュー |
| ledger_events | list[dict] | 台帳へ記録するイベント |
| measurement_summary | dict | 改善後の観測結果の要約 |
| suppressions | list[dict] | 再発抑制または保留にする候補 |

---

## 4. Data Model

```python
from typing import TypedDict, Literal, NotRequired


class ImprovementCandidate(TypedDict):
    id: str
    canonical_key: str
    title: str
    description: str
    source: str
    category: Literal[
        "quick_ui",
        "obsidian_chat",
        "logic_light",
        "db_api",
        "external",
        "infra",
    ]
    risk: Literal["low", "medium", "high"]
    effort: int
    impact: int
    duplicate_ids: NotRequired[list[str]]
    group_id: NotRequired[str]


class ImprovementState(TypedDict):
    key: str
    status: Literal[
        "candidate",
        "validated",
        "applied",
        "measured",
        "promoted",
        "suppressed",
        "needs_review",
        "rejected",
    ]
    reason: str
    source_report: str
    recorded_at: str


class MeasurementSummary(TypedDict):
    pdca_rate: float
    response_changed_rate: float
    repeat_issue_rate: float
    reuse_rate: float
    noise_rate: float
```

---

## 5. API / Interface

### パイプライン入出力
```python
def run_recursive_self_improvement(
    chat_log: list[str],
    improvement_report: dict,
    prompt_feedback_log: list[dict] | None = None,
    obsidian_notes: list[dict] | None = None,
    execution_status: dict | None = None,
) -> dict:
    """
    Returns:
        {
            "canonical_candidates": [...],
            "ranked_queue": [...],
            "ledger_events": [...],
            "measurement_summary": {...},
            "suppressions": [...],
        }
    """
```

### 状態遷移
```text
candidate -> validated -> applied -> measured -> promoted
candidate -> validated -> needs_review
candidate -> validated -> rejected
applied -> measured -> suppressed
measured -> promoted / suppressed
```

---

## 6. Business Rules

**BR-001**: 再帰の定義
- 条件: 反映済みの改善が新しい観測結果を生む場合
- 処理: その観測結果を次回の改善候補抽出へ再投入する
- 根拠: 改善結果を閉ループ化しないと学習が蓄積しないため

**BR-002**: 低リスク自動適用
- 条件: 変更が単一ファイル・低リスク・UI文言や軽微設定に限られる場合
- 処理: 自動適用候補に入れる
- 根拠: 安全性を保ちながら改善ループの回転数を上げるため

**BR-003**: 高リスク遮断
- 条件: スコアリング、DB、認証、外部連携、インフラ、複数ファイル変更を含む場合
- 処理: `needs_review` に落とし、自動適用しない
- 根拠: 既存の安定運用を守るため

**BR-004**: 重複抑制
- 条件: 同一テーマの改善候補が複数回抽出される場合
- 処理: `canonical_key` を基準に統合し、`duplicate_ids` を付与する
- 根拠: 再帰ループが同じ提案を増殖させないため

**BR-005**: 観測の再入力
- 条件: 反映済み改善の効果が計測できる場合
- 処理: `prompt_feedback`、改善レポート、Obsidianメモを次回候補抽出に戻す
- 根拠: 再帰的自己改善の核は観測を次の入力に戻すことだから

**BR-006**: 抑制判定
- 条件: 改善が再発しない、またはノイズ化している場合
- 処理: `suppressed` として再優先化を止める
- 根拠: 無限再提案を防ぐため

---

## 7. UI / UX（フロントエンド変更がある場合のみ）

- 本SPECではUI追加を必須にしない
- ただし改善ログ画面では、`pdca_rate`、`response_changed_rate`、`repeat_issue_rate` を確認できることが望ましい
- 画面表示は「改善したか」より「再利用されたか」を優先する

---

## 8. Error Handling

| エラー条件 | 処理 | ユーザー向けメッセージ |
|-----------|------|---------------------|
| 改善ログが空 | パイプラインをスキップ | 「改善候補がありません」 |
| `prompt_feedback_log` が壊れている | その入力だけ無視する | 表示しない |
| `canonical_key` 生成失敗 | 候補を `needs_review` に落とす | 「候補の正規化に失敗しました」 |
| high risk 判定 | 自動適用しない | 「手動レビューが必要です」 |
| 観測値が不足 | `measured` までで止める | 「効果測定待ちです」 |

---

## 9. Acceptance Criteria

**AC-001**: 入力統合
- Given: 改善ログ、prompt feedback、Obsidianメモが入力される
- When: 再帰的自己改善ループを実行する
- Then: 3系統の入力が同一の改善候補集合に統合される

**AC-002**: 重複排除
- Given: 同一タイトルまたは同一内容の改善候補が複数ある
- When: 正規化を行う
- Then: `canonical_key` が一致し、代表候補に `duplicate_ids` が付く

**AC-003**: 低リスク自動適用
- Given: 単一ファイルで低リスクの UI 文言改善がある
- When: キューに流す
- Then: `ranked_queue` に入り、自動適用対象になる

**AC-004**: 高リスク遮断
- Given: スコアリング変更や DB/API 変更を含む候補がある
- When: 判定を行う
- Then: `needs_review` になり、自動適用されない

**AC-005**: 再入力
- Given: 改善反映後の `prompt_feedback` が得られる
- When: 次回の抽出を行う
- Then: その観測結果が次回候補の入力として使われる

**AC-006**: 抑制
- Given: 同種の改善が繰り返しノイズとして現れる
- When: 再発判定を行う
- Then: `suppressed` に落ち、再優先化されない

**AC-007**: 台帳記録
- Given: 候補が `validated` から `applied` に進む
- When: 台帳へ記録する
- Then: `status` と `reason` が保存され、次回の重複抽出に使われる

---

## 10. Non-Functional Requirements

- **安全性**: high risk 変更は自動適用しない
- **可観測性**: 実行結果は台帳とレポートの両方に残す
- **冪等性**: 同じ `canonical_key` は繰り返し増殖しない
- **継続性**: パイプラインは失敗しても主業務を止めない
- **最小変更**: 既存の `run_daily_improvement_core.sh` と `pipeline_ledger.py` の責務を壊さない

---

## 11. Implementation Notes（Codex向け）

> このセクションはCodexへの実装指示。設計判断の「なぜ」を書く。

- **Source of Truth**: 正本の切り分けは `docs/improvement_source_of_truth.md` を参照し、このSPECは再帰的自己改善の振る舞いだけを定義する
- **既存の実行器**: `scripts/run_daily_improvement_core.sh` を外側ループとして維持する
- **台帳**: `pipeline_ledger.py` に状態遷移を記録する
- **入力源**: `prompt_feedback.py` / `prompt_feedback_metrics.py` / `reports/improvement_report_*.json` を使う
- **自動キュー**: `scripts/build_codex_auto_queue.py` の `safe/maybe/blocked` 方針を踏襲する
- **禁止事項**: スコアリング・DB・認証・外部API・インフラの自動拡張はしない
- **実装方針**: まず設計レベルでループを固定し、その後に必要最小限の状態追加だけ行う

---

## 12. Test Plan

### 単体テスト
| テストID | 対応AC | テスト内容 |
|---------|--------|-----------|
| test_001 | AC-001 | 複数入力源が統合されること |
| test_002 | AC-002 | `canonical_key` による重複排除 |
| test_003 | AC-003 | 低リスク候補が自動適用対象になること |
| test_004 | AC-004 | high risk 候補が `needs_review` になること |
| test_005 | AC-005 | 反映後の観測が次回入力に戻ること |
| test_006 | AC-006 | ノイズ候補が抑制されること |
| test_007 | AC-007 | 台帳に状態遷移が記録されること |

### 回帰テスト
- `run_daily_improvement_core.sh` の既存出力が壊れないこと
- `build_codex_auto_queue.py` の safe/maybe/blocked 判定が変わらないこと
- `prompt_feedback` の既存ログ形式が維持されること

### 手動確認
- [ ] 改善レポートに再発候補と抑制候補が分かれて見える
- [ ] 自動適用候補は low risk に限定される
- [ ] 反映後の観測が次の改善候補に反映される
