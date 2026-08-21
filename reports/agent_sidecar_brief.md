# Agent Sidecar Brief

> Generated: 2026-08-22 04:12 | source: `.claude/reports` | mode: read-only advisory

## Operating Boundary
- This brief is advisory context only.
- Do not let sidecar reports update scores, models, production DBs, or final approvals directly.
- Use findings as review prompts, RAG hints, or weekly PDCA inputs.
- Reports older than 30 days are demoted to the 再確認TODO section and must be re-verified against current code before use.

## Reports

### scoring-audit (failure)
- Source: `.claude/reports/scoring-audit/latest.md`
- Task: 自動監査
- Timestamp: 2026-08-22 04:12

**Summary:**
監査を完了できなかった: 想定外の例外: '<' not supported between instances of 'NoneType' and 'str'

**Risks:**
- 想定外の例外: '<' not supported between instances of 'NoneType' and 'str'

**Handoff:**
なし

### data-quality-checker (partial)
- Source: `.claude/reports/data-quality/latest.md`
- Task: SQLite 審査データの件数・異常値チェック
- Timestamp: 2026-08-22 04:12

**Summary:**
27 テーブルを確認し、要確認事項を 1 件検出した。

**Risks:**
- 空のテーブル: emotion_feedback, judgment_lifecycle_events, payment_history, phrase_weights, sync_log

**Handoff:**
なし

### rule-validator (partial)
- Source: `.claude/reports/rule-validation/latest.md`
- Task: ウェイト合計・グレード閾値の整合性チェック
- Timestamp: 2026-08-22 04:12

**Summary:**
整合性の逸脱を 1 件検出した。

**Risks:**
- IT機器: ウェイト合計 95（期待 100）

**Handoff:**
逸脱したカテゴリの定義を確認すること。スコア結果に直結する。

### build-runner (success)
- Source: `.claude/reports/build/latest.md`
- Task: コアモジュールのインポート確認
- Timestamp: 2026-08-22 04:12

**Summary:**
コアモジュール 6 件すべて import できる。

**Risks:**
- なし

**Handoff:**
なし

### api-health-checker (partial)
- Source: `.claude/reports/api-health/latest.md`
- Task: 依存サービスの設定・到達性確認
- Timestamp: 2026-08-22 04:12

**Summary:**
依存サービスに要確認事項を 2 件検出した。

**Risks:**
- GEMINI_API_KEY が未設定（該当機能は動作しない）
- SLACK_BOT_TOKEN が未設定（該当機能は動作しない）

**Handoff:**
なし

<!-- generated-by: build_agent_self_reports.py -->
<!-- 決定論的監査・LLM不使用。手で編集しても次回実行で上書きされる。
     『過去のエージェント指摘』は解消したら手で消すこと。 -->

### log-file-analyzer (partial)
- Source: `.claude/reports/log-analysis/latest.md`
- Task: ログのエラー・警告抽出
- Timestamp: 2026-08-22 04:12

**Summary:**
12 ファイルから ERROR 165 行、WARNING 57 行を検出した。

**Risks:**
- なし

**Handoff:**
頻出パターンは analyze_error_logs.py が改善台帳へ起票する。

<!-- generated-by: build_agent_self_reports.py -->
<!-- 決定論的監査・LLM不使用。手で編集しても次回実行で上書きされる。
     『過去のエージェント指摘』は解消したら手で消すこと。 -->

## 再確認TODO

古い指摘を現在の真実として扱わないこと。最新コードで再検証したものだけを有効扱いにする。

- [ ] `change-impact-analyzer` — 2026-03-28 10:30 時点 / `.claude/reports/impact-analysis/latest.md`
  - 当時の指摘: ### [高] Parent_Guarantor の FC・HC 二重寄与
- [ ] `security-checker` — 2026-03-28 11:15 時点 / `.claude/reports/security/latest.md`
  - 当時の指摘: | 重大度 | 件数 |
- [ ] `code-reviewer` — 2026-05-23 12:30 時点 / `.claude/reports/code-review/latest.md`
  - 当時の指摘: **重大**
- [ ] `agent-team（田辺・ダッシュ・鈴木・プランナー）` — 2026-03-21 17:30 時点 / `.claude/reports/agent-team/asset_value_discussion.md`
  - 当時の指摘: 4エージェントが物件資産価値スコアリングの現状を精査し、合計22件の改善提案を提出した。
- [ ] `agent-team (プランナー / ダッシュ / 田中さん / 鈴木さん)` — 2026-03-20 00:00 時点 / `.claude/reports/agent-team/subsidy_plan.md`
  - 当時の指摘: 4エージェントが「リース補助金の活用」について討論し、補助金マスタ管理・スコア加点・
- [ ] `agent-team (プランナー / ダッシュ / 田中さん / 鈴木さん)` — 2026-03-21 09:45 時点 / `.claude/reports/agent-team/report_ux_plan.md`
  - 当時の指摘: 業界ベンチマーク表示・単位統一・補助金シミュレーションが整った現在、次のフェーズとして「営業現場で即使えるレポート」への昇格を目指す。4エージェントの討論により、優先度付き実施項目13件とロードマップをまとめた。
- [ ] `file-searcher` — 2026-03-28 00:00 時点 / `.claude/reports/file-searcher/latest.md`
  - 当時の指摘: **BNモデルの CPT 再検証が必要**: 親ノードが1つ増えたことで `TabularCPD` の組み合わせ数が倍増した。`pt_fc` の計算が `_prob_financial_creditworthiness(*c)` のアンパッ
- [ ] `report-stylist` — 2026-03-20 12:30 時点 / `.claude/reports/report-stylist/latest.md`
  - 当時の指摘: `stats.rejection_rate` が 0.0% 固定のため統計カードには表示していない。運用改善後に追加を検討すること
