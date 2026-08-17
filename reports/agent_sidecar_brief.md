# Agent Sidecar Brief

> Generated: 2026-08-17 11:16 | source: `.claude/reports` | mode: read-only advisory

## Operating Boundary
- This brief is advisory context only.
- Do not let sidecar reports update scores, models, production DBs, or final approvals directly.
- Use findings as review prompts, RAG hints, or weekly PDCA inputs.
- Reports older than 30 days are demoted to the 再確認TODO section and must be re-verified against current code before use.

## Reports

### scoring-audit (failure)
- Source: `.claude/reports/scoring-audit/latest.md`
- Task: 自動監査
- Timestamp: 2026-08-17 11:16

**Summary:**
監査を完了できなかった: 想定外の例外: '<' not supported between instances of 'NoneType' and 'str'

**Risks:**
- 想定外の例外: '<' not supported between instances of 'NoneType' and 'str'

**Handoff:**
なし

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

### data-quality-checker (failure)
- Source: `.claude/reports/data-quality/latest.md`
- Task: SQLite 審査データの件数・異常値チェック
- Timestamp: 2026-08-17 11:16

**Summary:**
監査を完了できなかった: DB が見つからない: data/lease_data.db

**Risks:**
- DB が見つからない: data/lease_data.db

**Handoff:**
なし

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

### rule-validator (partial)
- Source: `.claude/reports/rule-validation/latest.md`
- Task: ウェイト合計・グレード閾値の整合性チェック
- Timestamp: 2026-08-17 11:16

**Summary:**
整合性の逸脱を 1 件検出した。

**Risks:**
- IT機器: ウェイト合計 95（期待 100）

**Handoff:**
逸脱したカテゴリの定義を確認すること。スコア結果に直結する。

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

### build-runner (partial)
- Source: `.claude/reports/build/latest.md`
- Task: コアモジュールのインポート確認
- Timestamp: 2026-08-17 11:16

**Summary:**
1/6 のモジュールが import できない。

**Risks:**
- scoring_core: ModuleNotFoundError: No module named 'numpy'

**Handoff:**
依存パッケージの不足かモジュール側の構文エラー。先に解消しないと他の監査も動かない。

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

### api-health-checker (partial)
- Source: `.claude/reports/api-health/latest.md`
- Task: 依存サービスの設定・到達性確認
- Timestamp: 2026-08-17 11:16

**Summary:**
依存サービスに要確認事項を 4 件検出した。

**Risks:**
- data/lease_data.db が存在しない
- GEMINI_API_KEY が未設定（該当機能は動作しない）
- SLACK_BOT_TOKEN が未設定（該当機能は動作しない）
- Ollama に到達できない（http://localhost:11434）: URLError

**Handoff:**
なし

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

### log-file-analyzer (success)
- Source: `.claude/reports/log-analysis/latest.md`
- Task: ログのエラー・警告抽出
- Timestamp: 2026-08-17 11:16

**Summary:**
logs/ が存在しないため解析対象なし。

**Risks:**
- なし

**Handoff:**
なし

<!-- scripts/build_agent_self_reports.py が自動生成（決定論的監査・LLM不使用）。
     手で編集しても次回実行で上書きされる。 -->

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
