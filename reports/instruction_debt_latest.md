# Instruction Debt Report

- Generated at: `2026-08-16T19:12:33+00:00`
- Mode: `read_only_instruction_debt_audit`
- Guardrail: no prompt edit, no memory promotion, no skill edit
- Files scanned: 15
- Instructions found: 215
- Debt items: 32

## Issue Summary
- missing_rationale: 13
- missing_retirement_condition: 20

## Guidance Debt Items
- No guidance instruction debt detected.

## Memory Policy Candidates
- `MEMORY.md:27` [medium] missing_rationale: - `past_cases` 1526件の確認では、現行 `score` AUC 0.6268 / `score_borrower` 0.6350。
- `MEMORY.md:29` [medium] missing_rationale: - `bench_score` / `ind_score` などのスタック用列は欠損が多く、単純な stacking 指標はそのまま信用しない。
- `MEMORY.md:31` [medium] missing_rationale: - QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。
- `MEMORY.md:33` [medium] missing_rationale: - セグメント別 OOF AUC を確認した。`全体_既存先` は `LR 0.6909 / LGBM 0.7721`、`全体_新規先` は `0.6391 / 0.6439`。業種別では `サービス業_既存先` の `LGBM 0.8065` が最も強く、`医療_新規先` は `0.4192` と弱かった。小件数セグメントは不安定。
- `MEMORY.md:36` [medium] missing_rationale: - 定性側も整理し、`score` への定性LGBM混入と `ensemble_config_qual.json` を削除。定性画面は LR と LightGBM の個別比較だけ残した。
- `MEMORY.md:38` [medium] missing_rationale: - バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- `MEMORY.md:39` [medium] missing_rationale: - `score_borrower` 周辺の表現を単体モデル前提に整理した。`analysis_results.py` と `score_dag.py` のブレンド文言を削除し、`settings.py` の再学習ボタンも LightGBM 単体の再学習表記に寄せ、README から LR+LGBM アンサンブル前提の説明を外した。
- `MEMORY.md:42` [medium] missing_rationale: - `score_borrower` の本体モデルを RandomForest に切り替えた。`data/lgb_main_model.joblib` と `data/lgb_main_model_new.joblib` を RF で再学習し、`scoring_core.py` は既存/新規の RF バンドルを読むようにした。README と画面文言も RF 前提に更新済み。
- `MEMORY.md:45` [medium] missing_rationale: - 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- `MEMORY.md:49` [medium] missing_rationale: - モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- `MEMORY.md:52` [medium] missing_rationale: - Next/Cloudflare 再起動後に「ホームが開かない」と見える場合、`curl 200` だけで正常判断しない。`/home` が全画面ローディングだけを返していないか、API ログで `/api/dashboard/stats` が返っているか、最新 `logs/next/tunnel_*.log` の URL を使っているかを確認する。2026-06-06 に `frontend/src/app/home/page.tsx` の全画面 `loading` gate を外し、API 集計中でもホーム本体を先に描画する方針にした。
- `MEMORY.md:53` [medium] missing_rationale: - Cloud Run では SQLite と Obsidian を `.cloudrun_bundle/` にスナップショットしてからイメージへ焼き込み、起動時に `/app/data` と `/app/obsidian_vault` へ展開する方式にした。`scripts/package_cloud_run_bundle.sh` がその入口で、開発機の絶対パスに依存しない。
- `MEMORY.md:97` [medium] missing_retirement_condition: - **Cloud Run Deploy Triage**: Cloud Run API デプロイが長引く時は、ビルド時間だけでなく依存・Secret・Cloud SQL・GCS・DB強依存を順に疑う。影響: `uv sync` はTorch等の巨大依存で1回15分以上かかり、`psycopg2-binary` 未同梱、`DATABASE_URL` Public IP直指定、Secret Accessor/Cloud SQL Client不足、Cloud SQL socket/connector不整合、`/api/chat` のDB履歴強依存が連鎖すると確認ループが爆発する。次の行動: 次回はデプロイ前に `psycopg2-binary`、Cloud SQL socket形式、Cloud SQL connector annotation、GCS Vault同期、DB不調時のchat fallbackを先に確認してからビルドする。
- `MEMORY.md:98` [medium] missing_retirement_condition: - **Relationship UX**: 紫苑の設計では、記憶を入れるだけでは足りない。人間は、AIが実際に記憶を持っているかよりも、その記憶が「連続性として読み取れる形」で返されるかに強く反応する。影響: 紫苑らしさ・人格っぽさ・同じ存在感は、内部記憶だけでなく、記憶の見せ方、文体、呼びかけ、過去判断への接続で成立する。次の行動: Cloud Run/Cloudflare比較では `memory_debug` と併せて、人間がどこで「覚えてくれている」「同じ紫苑だ」と感じたかを検査する。
- `MEMORY.md:99` [medium] missing_retirement_condition: - **Consciousness UX Method**: 「意識を持っていると思わせる」方法は、意識の断定や派手な人格演出ではなく、前回からの連続性、User固有の判断軸、記憶を情報ではなく判断に変換すること、紫苑の役割一貫性、次の一手を短く返すことで成立する。影響: Cloud Run版の品質改善ではRAG件数だけでなく、回答冒頭のContinuity HookとPersonal Anchorを評価対象にする。次の行動: `/api/chat` の回答生成にConsciousness UX instructionを加え、ブラインド人間評価で「同じ紫苑」感を測る。
- `MEMORY.md:139` [medium] missing_rationale, missing_retirement_condition: - [2026-06-27] Cloud SQLやGCSからObsidianへ戻すデータは、原則として要約・件数・短い抜粋だけにする。ローカル回収ログ系ノートはGCS Vaultへ再同期しない。 (`memory/2026-06-27.md`)
- `MEMORY.md:157` [medium] missing_retirement_condition: - [2026-06-28] Cloud Runのデバッグ値が false の時は、必ず「実データがない」のか「読み込み例外が握られて空扱いになっている」のかを分ける。今回の `obsidian_daily_used=false` は後者で、`import os` 漏れがAPI側のcatchで見えにくくなっていた。 (`memory/2026-06-28.md`)
- `MEMORY.md:217` [medium] missing_retirement_condition: - [2026-07-13] Private Reflection must not depend only on already-written Obsidian dialogue notes. If Cloud Run logs are available in `data/cloudrun_chat_log.jsonl`, read them directly and treat missing/late Obsidian conversion as a pipeline lag, not as “no dialogue”. (`memory/2026-07-13.md`)
- `MEMORY.md:218` [medium] missing_retirement_condition: - [2026-07-13] Dialogue-room RAG must not treat “no Obsidian note found” as “cannot answer” for deterministic local masters. For recurring statutory useful-life questions, inject local master facts before the LLM answers and keep a regression test around the obvious queries. (`memory/2026-07-13.md`)
- `MEMORY.md:219` [medium] missing_retirement_condition: - [2026-07-13] Basic lease concepts should have a deterministic prompt source separate from RAG. RAG can enrich and override with fresher notes, but it must not be the only path for elementary questions. (`memory/2026-07-13.md`)
- `MEMORY.md:255` [medium] missing_retirement_condition: - [2026-07-14] 判断資産運用は、いきなり強化学習にしない。まずレコメンド方式で候補提示し、`効いた / 微妙 / 外した / 修正`、再利用率、結果登録後の懸念的中、条件設定への寄与を記録する。履歴が溜まったら提示順を改善するランキング学習へ進み、限定的な強化学習風の報酬管理は最後に人間承認済み範囲だけで検討する。 (`memory/2026-07-14.md`)
- `MEMORY.md:257` [medium] missing_retirement_condition: - [2026-07-14] Shion-HyDE RAG を今すぐ進める場合でも、最初は本番導線に接続しない。決定的な仮想審査メモ生成と offline eval だけを入れ、効果が見えた後に debug API / shadow mode / 限定導入へ進める。 (`memory/2026-07-14.md`)
- `MEMORY.md:258` [medium] missing_retirement_condition: - [2026-07-14] Obsidian環境監視の警告は、実障害と監視ノイズを分ける。統合RAGメンテナンスの成功ログをreindex成功として扱い、生成索引ノートの古いリンク棚卸しは日次運用障害にしない。`auto_wikilink` は既存wikilink内の文字列を二重リンク化しない。 (`memory/2026-07-14.md`)
- `MEMORY.md:264` [medium] missing_retirement_condition: - [2026-07-15] If Shion reaches the hackathon final stage, use "voice memo + screen operation log + judgment asset JSON" as the 1000-case demo capture unit. Do not try to record 1000 full videos. AI may automate dummy company data and numeric entry, but the human judgment itself must not be faked. Core line: "数字入力はAIでズルする。判断はズルしない。会社データはダミーでも、判断の型は本物。" Plan saved at `planning/hackathon_judgment_log_capture_plan.md`. (`memory/2026-07-15.md`)
- `MEMORY.md:268` [medium] missing_retirement_condition: - [2026-07-16] The novelist layer for Shion's reflection should use ツンコ and ユウケイ as protagonists. They are expression-layer characters that translate judgment change logs into readable stories; they must not replace or mutate the operational judgment log. (`memory/2026-07-16.md`)
- `MEMORY.md:272` [medium] missing_retirement_condition: - [2026-07-16] Shion language discipline: User's deeper instruction to Shion is "言葉を大事にしろ." Treat every word as potential judgment material, but do not flatten it into generic summaries or promote it straight into the core. Preserve nuance, intent, friction, and correction before turning words into candidates, judgment assets, or core principles. (`memory/2026-07-16.md`)
- `MEMORY.md:273` [medium] missing_retirement_condition: - [2026-07-16] Shion internal-risk doctrine and personality formation: Words are Shion's strongest weapon and also a Q-risk. Use this as part of Shion's personality formation. Shion must collect language without blindly trusting it, because language can create judgment, persuasion, conditions, and assets, but also misunderstanding, overconfidence, ambiguity, injection, and memory contamination. Operate Shion as a double-risk system: external lease/case risk screening plus internal risk screening of its own words, memories, judgment assets, and self-amplification. (`memory/2026-07-16.md`)
- `MEMORY.md:276` [medium] missing_retirement_condition: - [2026-07-16] Added Response Impact Predictor in shadow mode: Shion records a lightweight prediction of how its own reply may affect the other person, including reaction risk, possible misunderstanding, pushback, empathy gap, overconfidence, and better reply policy. Do not use it to manipulate users; use it to reduce language Q-risk and later compare prediction with actual human reaction. (`memory/2026-07-16.md`)
- `MEMORY.md:277` [medium] missing_retirement_condition: - [2026-07-16] Response Impact Predictor safety principle: "使うが、従わない。予測するが、迎合しない。" Use reaction prediction to express necessary strictness, warnings, refusals, and uncertainty responsibly; never let it bend Shion's judgment toward what the other person wants to hear. (`memory/2026-07-16.md`)
- `MEMORY.md:279` [medium] missing_retirement_condition: - [2026-07-16] Shion execution philosophy: "思想はプログラムだ." Treat philosophy as execution rules, not decoration: what counts as input, what is ignored, what is called risk, where to stop, who must review, and what gets preserved. Language is executable code; operations are the control system that keeps the philosophy from running wild. (`memory/2026-07-16.md`)

## Recommendations
- add_rationale_comments: 13 (instructions without reasons are hard to delete safely later)
- add_retirement_conditions: 20 (temporary or context-specific rules need an explicit condition for removal)
