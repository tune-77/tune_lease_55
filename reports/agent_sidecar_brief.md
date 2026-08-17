# Agent Sidecar Brief

> Generated: 2026-08-17 07:05 | source: `.claude/reports` | mode: read-only advisory

## Operating Boundary
- This brief is advisory context only.
- Do not let sidecar reports update scores, models, production DBs, or final approvals directly.
- Use findings as review prompts, RAG hints, or weekly PDCA inputs.
- Reports older than 30 days are demoted to the 再確認TODO section and must be re-verified against current code before use.

## Reports
_No fresh reports. All 12 report(s) are older than 30 days — see 再確認TODO._

## 再確認TODO

古い指摘を現在の真実として扱わないこと。最新コードで再検証したものだけを有効扱いにする。

- [ ] `scoring-auditor` — 2026-04-04 11:30 時点 / `.claude/reports/scoring-audit/latest.md`
  - 当時の指摘: 1. **[高] lease_credit_log 係数の飽和問題**: scoring_core.py 内の全体_既存先係数。リース信用枠さえ入力すれば財務内容に関わらず90点超に張り付く。財務悪化企業が正当に否決されない可能性がある。係
- [ ] `data-quality-checker` — 2026-03-28 14:55 時点 / `.claude/reports/data-quality/latest.md`
  - 当時の指摘: 1. **即時リスク**: 現在のアプリが誤ったダミーデータを参照してスコア比較・傾向分析を行っている
- [ ] `rule-validator` — 2026-03-28 11:00 時点 / `.claude/reports/rule-validation/latest.md`
  - 当時の指摘: | 深刻度 | 内容 |
- [ ] `change-impact-analyzer` — 2026-03-28 10:30 時点 / `.claude/reports/impact-analysis/latest.md`
  - 当時の指摘: ### [高] Parent_Guarantor の FC・HC 二重寄与
- [ ] `security-checker` — 2026-03-28 11:15 時点 / `.claude/reports/security/latest.md`
  - 当時の指摘: | 重大度 | 件数 |
- [ ] `code-reviewer` — 2026-05-23 12:30 時点 / `.claude/reports/code-review/latest.md`
  - 当時の指摘: **重大**
- [ ] `build` — 時刻不明 時点 / `.claude/reports/build/latest.md`
  - 当時の指摘: # ビルドチェック結果
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
