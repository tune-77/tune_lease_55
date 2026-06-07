# Agent Sidecar Brief

> Generated: 2026-06-08 07:35 | source: `.claude/reports` | mode: read-only advisory

## Operating Boundary
- This brief is advisory context only.
- Do not let sidecar reports update scores, models, production DBs, or final approvals directly.
- Use findings as review prompts, RAG hints, or weekly PDCA inputs.

## Reports

### scoring-auditor (success / stale)
- Source: `.claude/reports/scoring-audit/latest.md`
- Task: スコアリングロジック全面監査（DB異常値・ロジック整合性・MLモデル・重複データ）
- Timestamp: 2026-04-04 11:30

**Summary:**
全160件の審査レコードを調査した結果、**5種類の深刻な問題**が確認された。最も重大なのは「lease_credit_log 係数が異常に大きく（1.14）、リース信用枠の入力だけで財務内容に関わらず承認圏内へ引き上げる」構造的バグと、「全財務値が10百万円という同一のテストデータが51件も蓄積されており、承認圏内（97.3点）で記録されている」点である。scoring_core.py と predict_one.py が別パスで動作し、意味論（承認確率 vs デフォルト確率）が混在している問題も深刻。

---

**Risks:**
1. **[高] lease_credit_log 係数の飽和問題**: scoring_core.py 内の全体_既存先係数。リース信用枠さえ入力すれば財務内容に関わらず90点超に張り付く。財務悪化企業が正当に否決されない可能性がある。係数の再調整か、上限スケーリングが必要。

2. **[高] テストデータの本番DB混在**: 全値10百万円の51件が蓄積されており、MLモデルの再学習に使われた場合にモデルが歪む。削除またはフラグ管理が必要。

3. **[高] predict_one.py の判定論理逆転**: industry_hybrid_model の legacy_prob は「承認確率」だが hybrid_prob < 0.5 → 承認 と「デフォルト確率」として扱われており逆の判定になる。現在はデッドコードだが将来有効化で重大な誤判定を招く。

4. **[高] 学習データ不足による過学習**: 実質独立サンプル約53件、建設業偏重74%でLightGBM（n_estimators=100）を学習。ai_prob の信頼性は極めて低く、70%の重みで使用するのは危険。

5. **[中] 承認ライン71点 vs 70点の不統一**: screening_report.py と charts.py が70点を使用、境界線案件で表示上の判定が乖離する。

6. **[中] 債務超過のペナルティ欠如**: scoring_core.py の z 値計算に純資産マイナス

**Handoff:**
- **code-reviewer**: scoring_core.py の lease_credit_log 係数（1.140132）の再調整が最優先。scoring/models/industry_specific/industry_coefficients.pkl の係数もサービス業の rent_to_revenue = -0.22 など符号・大きさの妥当性を確認すること。scoring/predict_one.py:193 の hybrid_prob < 0.5 → 承認 という判定の意味論が逆転していないか要確認。customer_db.py の equity_ratio カラムが全件 NULL になる原因（_round_ratio() 関数）の調査も必要。

- **test-runner**: 追加すべきエッジケーステスト:
  1. lease_credit = 0 かつ全財務健全 → スコアが正常範囲（50〜80点）に収まるか
  2. 債務超過（net_assets < 0）案件 → 要審議または否決になるか
  3. scoring/predict_one.py で健全財務 → decision が "承認" になるか（現在は "否決" になる可能性あり）
  4. スコア70点ちょうど → screening_report.py と scoring_core.py で判定が一致するか

- **data-quality-checker**: screeni

### data-quality-checker (success / stale)
- Source: `.claude/reports/data-quality/latest.md`
- Task: lease_data.db 全テーブル監査・データ減少原因特定
- Timestamp: 2026-03-28 14:55

**Summary:**
`lease_data.db` の `past_cases` テーブルが **本物の審査履歴 33件から手製のダミーデータ 8件に差し替えられている**。
`save_all_cases()` の `DELETE FROM past_cases` → 再INSERT ロジックが、ダミーデータを引数に呼び出されたことが原因と特定。
バックアップ（`data/backups/lease_data.db.20260328_0608`）には本物データ 33件が保存されている。

---

**Risks:**
1. **即時リスク**: 現在のアプリが誤ったダミーデータを参照してスコア比較・傾向分析を行っている
2. **再発リスク**: `save_all_cases()` の全削除設計が残る限り、同様の事故が再発し得る
3. **データ二重管理リスク**: `subsidies` vs `subsidy_master` の不一致が補助金スコアリングに影響する可能性
4. **ワークツリー間 DB 同期問題**: 各ワークツリーが独立した DB を参照しており、本番データが特定パスにしか存在しない

---

**Handoff:**
- **scoring-auditor**: `past_cases` がダミーデータ 8件しかない状態で類似案件スコアリング・比較分析が行われていた可能性を確認してください。`screening_db.sqlite`（153件）との使い分けも確認を
- **migration-validator**: `save_all_cases()` の `DELETE FROM past_cases` → 再INSERT 設計をUPSERT（`INSERT OR REPLACE`）に変更する移行が必要です。データ消失リスクの根本対策
- **復旧手順**（推奨）:
  ```bash
  cp data/backups/lease_data.db.20260327_0128 data/lease_data.db
  ```
  その後、消えた34件目（ID: `20260326211518452246`）は `20260327_0128` バックアップに存在するため完全復旧可能

### rule-validator (success / stale)
- Source: `.claude/reports/rule-validation/latest.md`
- Task: bayesian_engine.py ビジネスルール整合性検証
- Timestamp: 2026-03-28 11:00

**Summary:**
構造的バグ・整合性エラーはゼロ。Parent_Guarantor の FC・HC 二重寄与は設計意図として数値的に整合。
nr と pg の相互作用は修正後に正しく動作していることを確認。
低深刻度の文書化不足 2 件、クリップ発動による情報損失 1 件。

---

**Risks:**
| 深刻度 | 内容 |
|--------|------|
| 低 | BN閾値 THRESHOLD_APPROVAL=0.70 vs constants.py APPROVAL_LINE=71 の1点ズレ |
| 低 | fc=1,hc=1最優良ケースで av/st/ot 加算が無効化（クリップ） |

---

**Handoff:**
- **test-runner**: _prob_final_decision(1,1,1,1,1) クリップパターンのテスト追加推奨
- **test-runner**: _prob_financial_creditworthiness(1,0,0,0,1,1)=0.484 の回帰テスト追加推奨

### change-impact-analyzer (success / stale)
- Source: `.claude/reports/impact-analysis/latest.md`
- Task: 変更影響分析（bayesian_engine.py Parent_Guarantor エッジ追加 + Streamlit API 一括置換）
- Timestamp: 2026-03-28 10:30

**Summary:**
影響の大きさ: **高**

`bayesian_engine.py` へ `Parent_Guarantor → Financial_Creditworthiness` エッジを追加したことで、審査スコアリングの中核ロジックが変化した。`Parent_Guarantor` は既存の `Hedge_Condition` ノードにも親ノードとして存在するため、同一フラグが FC・HC の二重経路で最終判断に寄与し、承認確率を約 +0.53 押し上げる。`High_Network_Risk` と `Parent_Guarantor` の相互作用については修正済み（pg底上げ後にnr割引を適用する順序に変更）。UI 修正（18 ファイルの Streamlit API 置換）はロジック変更を含まない。

---

**Risks:**
### [高] Parent_Guarantor の FC・HC 二重寄与
pg=1 フラグが FC と HC の両方の親ノードになっており、単体で承認確率を約 +0.53 押し上げる。設計意図の明文化が必要。

### [修正済み] High_Network_Risk 抑制効果の実質無効化
`bayesian_engine.py:311-324` の適用順序を修正済み（pg底上げ→nr割引の順）。

### [中] Streamlit バージョン互換
`width='stretch'` / `width='content'` が動作する最低バージョンが requirements.txt で固定されているか未確認。

---

**Handoff:**
- **security-checker**: Slack 経由での pg フラグ不正セットによる審査結果操作リスクを確認
- **rule-validator**: Parent_Guarantor の FC・HC 二重寄与がビジネスルール上意図的かを確認

### security-checker (success / stale)
- Source: `.claude/reports/security/latest.md`
- Task: セキュリティレビュー（bayesian_engine.py CPT拡張 + Streamlit UI 18ファイル一括置換）
- Timestamp: 2026-03-28 11:15

**Summary:**
| 重大度 | 件数 |
|--------|------|
| Critical | 1 |
| High | 2 |
| Medium | 2 |
| Low | 2 |

今回の変更（a4e4820, 3cc1bb5）に直接起因する新規脆弱性は [C-1] のみ（既存コードの発見）。
UI 置換（width=）はセキュリティリスクゼロ。

---

**Handoff:**
- C-1 は今回のコミット以前から存在する既存リスク（優先対処推奨）
- H-1, H-2 は今回の UI 変更とは無関係の既存問題

### code-reviewer (partial / stale)
- Source: `.claude/reports/code-review/latest.md`
- Task: PR #160 — PD除去 & 業種別倒産率ベンチマーク追加 & LGBM成約モデル統合
- Timestamp: 2026-05-23 12:30

**Summary:**
PD除去は主要パスで概ね完了しているが、アクティブコード内に複数の残存参照（report_generator.py、batch_scoring.py、floating_bot.py）がある。LGBM統合のロジック自体は堅牢だが単体で呼ばれておらず dead code に近い状態。業種ベンチマークの実装は適切。

**Risks:**
**重大**
1. `scoring_result` が常に `None` のため、LGBM 統合は `predict_one.py` に実装されたものの `score_calculation.py` から一切呼ばれていない。LGBM の「否決」ペナルティも発動しない。`_build_learning_pd_result()` の呼び出しを復活させるか、直接 `predict_one()` を呼んで `scoring_result` に代入する必要がある。
2. `report_generator.py:332` の `pd_percent` 参照が残存しており、審査レポートのデフォルト確率セクションが常に0%表示になる。レポートの品質に直接影響する。

**軽微**
3. `shinsa_gunshi_logic.py:846/856` の `s['pd_pct']` で KeyError リスク。過去事例DBに `pd_pct` カラムが存在する限りは動作するが、DBリセット後や新規インストール環境で問題になる可能性がある。
4. `scoring_output_bridge.json` への絶対パスハードコード（`/Users/kobayashiisaoryou/clawd/...`）が2箇所に存在。本番・CI 環境移植時に問題になる。
5. `api/main.py:516-517` で GEMINI_API_KEY チェックを削除し `gunshi_gemini.py`

**Handoff:**
- security-checker: `scoring_output_bridge.json` への絶対パスハードコードが `components/score_calculation.py` に2箇所。パス漏洩ではないが環境依存コードとして要確認。`data/industry_bankruptcy_bench.py` は外部データ（TDB）のハードコード参照のため出典・ライセンス確認を推奨。
- test-runner: `scoring/predict_one.py` の `_load_lgbm_contract_bundle()` と `_build_lgbm_contract_row()` は単体テストが未整備。特に特徴量名の一致チェック（`fmap.get(f, nan)` の網羅性）と NaN 混入時の LGBM 推論動作を確認すること。`data/industry_bankruptcy_bench.py` の `get_bankruptcy_bench()` に対し、未知業種・空文字・プレフィックスのみのケースをテストすること。

### build (unknown / stale)
- Source: `.claude/reports/build/latest.md`
- Task: latest
- Timestamp: -

**Summary:**
# ビルドチェック結果

日時: 2026-05-02

| 項目 | 結果 |
|---|---|
| コアモジュール | ✅ OK |
| スコアリング | ✅ OK |
| Slack/AI | ✅ OK |
| コンポーネント | ✅ OK |
| 依存パッケージ | ✅ OK (lightgbm/scipy 含む) |
| secrets.toml | ✅ 存在 |

### agent-team（田辺・ダッシュ・鈴木・プランナー） (success / stale)
- Source: `.claude/reports/agent-team/asset_value_discussion.md`
- Task: 物件資産価値スコアリングの改善点議論
- Timestamp: 2026-03-21 17:30

**Summary:**
4エージェントが物件資産価値スコアリングの現状を精査し、合計22件の改善提案を提出した。
最重要課題は「情報欠如がリスクとして扱われない」「Slack・バッチ・個別で3系統のロジックが分裂している」「asset_scorer.pyの豊富な出力（grade・warnings・recommendation）がUIで完全に死んでいる」の3点で全員一致。

---

**Handoff:**
- **実装担当**: 🔴の5件（Slack不整合・情報欠如ペナルティ・deafault50・warningsUI・ID統一）を最初のスプリントで実装。合計4〜5日
- **code-reviewer**: `category_config.py` への評価軸追加はウェイト合計100維持の確認が必要
- **test-runner**: `calc_asset_score()` への引数追加（subsidy_info等）は既存テストに影響する可能性あり

### agent-team (プランナー / ダッシュ / 田中さん / 鈴木さん) (success / stale)
- Source: `.claude/reports/agent-team/subsidy_plan.md`
- Task: リース補助金活用 討論 & PLAN策定
- Timestamp: 2026-03-20 00:00

**Summary:**
4エージェントが「リース補助金の活用」について討論し、補助金マスタ管理・スコア加点・
営業ツール化・実装ロードマップの合意を形成した。

### agent-team (プランナー / ダッシュ / 田中さん / 鈴木さん) (success / stale)
- Source: `.claude/reports/agent-team/report_ux_plan.md`
- Task: 審査レポートUI/UX改善 & 営業ツール強化 討論 & PLAN策定
- Timestamp: 2026-03-21 09:45

**Summary:**
業界ベンチマーク表示・単位統一・補助金シミュレーションが整った現在、次のフェーズとして「営業現場で即使えるレポート」への昇格を目指す。4エージェントの討論により、優先度付き実施項目13件とロードマップをまとめた。

---

### agent-discussion (リースくん / Tune / 八奈見杏奈 / 審査軍師 / Dr.Algo / タム / プランナー / ダッシュ) (success / stale)
- Source: `.claude/reports/agent-discussion/ux_debate_latest.md`
- Task: 「ユーザーがもっと直感的に使えるようにするには？」UX改善討論
- Timestamp: 2026-03-28 11:30

**Summary:**
8エージェントが「直感的な使いやすさ」をテーマに白熱討論。リースくんが入口UXを、ダッシュが視覚設計を、八奈見が「余計なものを削れ」と一刀両断、Dr.Algoがオンボーディングの数学的最適化を提示、タムが「においがしないUI」問題を指摘、軍師が戦略的優先順位を整理し、Tuneが6項目を承認した。

---

**Risks:**
- ロール別UIは既存のセッション管理（`SK.*`）の変更を伴う可能性があり、鈴木さんに事前アセスメントを依頼すること
- ストリーミングレスポンスはOllama/Gemini両エンジンへの対応が必要。Ollamaは `/api/generate` のストリームモードで対応可能だが、Gemini側の実装確認が必要
- デザインシステム統一は既存コンポーネント全てへの影響あり。段階的移行計画を立てること

**Handoff:**
- **code-reviewer**: Phase 1 実装完了後にレビュー依頼
- **build-runner**: 各Phase完了後に起動確認
- **report-stylist**: Phase 3のデザインシステム統一完了後にビジュアルレポート生成
- **鈴木さん**: ロール別UIの工数見積もりとセッション管理影響調査を先行で依頼することを推奨

### file-searcher (success / stale)
- Source: `.claude/reports/file-searcher/latest.md`
- Task: 変更ファイル調査（直近3コミット）
- Timestamp: 2026-03-28 00:00

**Summary:**
直近3コミット（a4e4820, 3cc1bb5, 4afc5aa）で計20ファイルが変更された。
変更の性質は2種類に明確に分類される。

1. **Streamlit API 廃止警告対応**（3cc1bb5）: `use_container_width=True/False` を新 API `width='stretch'` / `width='content'` に一括置換。18ファイルに渡る純粋なUI修正。
2. **ベイジアンBN モデルの機能追加**（a4e4820）: `Parent_Guarantor → Financial_Creditworthiness` エッジを追加し、親会社連帯保証を信用力評価に反映。
3. **セッションデータ更新**（4afc5aa）: `data/last_case.json`, `data/recent_modes.json` の内容更新（コード変更なし）。

---

**Risks:**
- **BNモデルの CPT 再検証が必要**: 親ノードが1つ増えたことで `TabularCPD` の組み合わせ数が倍増した。`pt_fc` の計算が `_prob_financial_creditworthiness(*c)` のアンパック展開に依存しており、`pg` 引数の順序が `evidence` リストの順序と一致しているか要確認。
- **Streamlit バージョン依存**: `width='stretch'` / `width='content'` は Streamlit の新 API。旧バージョンを使う環境ではエラーになる可能性がある。requirements.txt / pyproject.toml のバージョン固定状況を確認すること。
- **`form_apply.py` の `width='content'`**: 他ファイルは `'stretch'` に統一されているが、このボタンのみ `'content'` に変換されており、UIの意図的な差異である可能性がある。意図通りかレビューで確認が望ましい。

---

**Handoff:**
- **scoring-auditor**: `bayesian_engine.py` の CPT 変更（`Parent_Guarantor` エッジ追加）が審査スコアに与える影響を検証すること。特に `Financial_Creditworthiness` の事後確率分布が既存案件で大きく変化していないか確認を推奨。
- **rule-validator**: `bayesian_engine.py` の `BN_EDGES` および CPT 整合性チェックを実施すること。`_prob_financial_creditworthiness(*c)` のアンパック順序が `evidence` リストの順序と一致しているか要確認。
- **code-reviewer**: `bayesian_engine.py` の論理変更部分のレビューを推奨。UIファイル群（18件）は機械的置換のみのため優先度低。
- **build-runner**: Streamlit の新 API `width='stretch'` が requirements に定義されたバージョンで動作するか確認を推奨。

### general-purpose (success / stale)
- Source: `.claude/reports/general-purpose/latest.md`
- Task: SQLite DBからシステム改善レポートデータを生成し data/improvement_report_data.py として保存
- Timestamp: 2026-03-20 00:00

**Summary:**
`data/lease_data.db`（past_cases テーブル、26件）を直接集計し、統計データ・業種別ランキング・スコア分布・直近10件・改善提案8件を `data/improvement_report_data.py` に `REPORT_DATA` dict として保存した。

---

**Risks:**
- 「要審議」が最終判定として残っているケースが多く、実質の否決率が不明。`rejection_rate` は現状 0.0% だが実態を反映していない可能性がある
- 26件というサンプル数は統計的に小さく、業種別傾向の信頼性は限定的
- `data/improvement_report_data.py` は `.gitignore` 対象外のため、機密データ（個社スコア等）を含まないよう集計値のみを記載した

---

**Handoff:**
- UI側（`components/` 配下）で `REPORT_DATA` を import して「システム改善レポート」タブに表示する実装が次ステップ
- 週次自動再生成のための Cron 設定（CronCreate ツール）を検討
- `rejection_rate` の定義を「否決」判定のみから「要審議+否決」に変更するかどうか、運用側で合意が必要

### novelist-weekly-scheduler (partial / stale)
- Source: `.claude/reports/novelist/latest.md`
- Task: 波乱丸 第9話（第2026年03月24日号）生成
- Timestamp: 2026-03-24 09:04

**Summary:**
毎週火曜日の自動タスクとして波乱丸 第9話を生成した。AIサービス（Streamlit）がスケジューラ環境では利用不可のため、フォールバック小説を採用した。novelist_agent.db のジャーナルファイルロックにより本番DBへの書き戻しは未完了。

**Risks:**
- novelist_agent.db に未解決のジャーナルファイル（.db-journal）が存在し、SQLite の disk I/O error が発生している
- スケジューラ環境では Streamlit がインポートできないため、AI生成は常にフォールバックになる
- novelist_agent_work.db（一時コピー）には正常に第9話が書き込まれているが、本番DBへの反映が未完了

**Handoff:**
- data/novelist_agent.db-journal の削除が必要（手動またはStreamlitアプリ起動後にSQLiteが自動処理する可能性あり）
- StreamlitアプリをAIなし環境で起動できる軽量モードの検討が必要

### report-stylist (success / stale)
- Source: `.claude/reports/report-stylist/latest.md`
- Task: 審査レポート画面への改善レポートセクション追加
- Timestamp: 2026-03-20 12:30

**Summary:**
`data/improvement_report_data.py` の `REPORT_DATA` を読み込み、`components/report.py` の `render_report()` 末尾から呼び出される `render_improvement_report()` 関数を新規実装した。
審査統計サマリー・スコア分布バー（Plotly）・業種別ランキング・改善提案カード・直近10件テーブルの5セクションを既存CSSと統一したスタイルで追加。`_REPORT_CSS` にも改善レポート専用スタイル群を追記した。

---

**Risks:**
- `stats.rejection_rate` が 0.0% 固定のため統計カードには表示していない。運用改善後に追加を検討すること
- `data/improvement_report_data.py` の自動再生成（週次 Cron）が未実装のため、データが古くなる可能性がある

---

**Handoff:**
- `data/improvement_report_data.py` の週次自動再生成を CronCreate ツールで設定することを推奨（general-purpose レポートに記載あり）
- `rejection_rate` の定義を「否決＋要審議」で再定義した場合、stat カードへの再追加を検討すること
- 現状 `render_improvement_report()` は `render_report()` 内からしか呼ばれないため、スタンドアロンタブとして独立させる場合はセッション審査データ依存を切り離す必要あり
