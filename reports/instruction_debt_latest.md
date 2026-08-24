# Instruction Debt Report

- Generated at: `2026-08-23T19:16:19+00:00`
- Mode: `read_only_instruction_debt_audit`
- Guardrail: no prompt edit, no memory promotion, no skill edit
- Files scanned: 20
- Instructions found: 226
- Debt items: 58

## Issue Summary
- missing_rationale: 18
- missing_retirement_condition: 47
- missing_scope: 1

## Guidance Debt Items
- `.agents/skills/research-to-screening-insights/SKILL.md:10` [high] missing_rationale, missing_scope, missing_retirement_condition: Turn research into a small set of screening actions. The output should help decide what to ask, what to watch, and how to explain the risk.
- `AGENTS.md:54` [medium] missing_retirement_condition: - 1週間以上有効そうな個人設定/好みが確認できた
- `.agents/skills/screening-decision-flow-builder/SKILL.md:70` [medium] missing_rationale, missing_retirement_condition: - Do not imply automatic approval or denial unless the user explicitly asks for automation.
- `.agents/skills/screening-decision-flow-builder/SKILL.md:72` [medium] missing_rationale, missing_retirement_condition: - Do not mix personnel evaluation with judgment-asset evaluation.
- `.agents/skills/scqa-report-writer/SKILL.md:64` [medium] missing_retirement_condition: - Do not skip any SCQA element.
- `.agents/skills/scqa-report-writer/SKILL.md:66` [medium] missing_retirement_condition: - Do not use jargon where a field user would need practical action.
- `.agents/skills/scqa-report-writer/SKILL.md:67` [medium] missing_retirement_condition: - Do not overstate evidence. If the source is weak, say the answer is tentative.
- `.agents/skills/research-to-screening-insights/SKILL.md:63` [medium] missing_retirement_condition: - Do not create more than three top checks unless the user asks for a full checklist.
- `.agents/skills/research-to-screening-insights/SKILL.md:64` [medium] missing_retirement_condition: - Do not promote research directly into long-term memory or active scoring.
- `.agents/skills/research-to-screening-insights/SKILL.md:65` [medium] missing_retirement_condition: - Do not make approval/denial recommendations from macro research alone.
- `.agents/skills/research-to-screening-insights/SKILL.md:66` [medium] missing_retirement_condition: - Use `lease-source-validator` first when the source quality is uncertain or the topic is current.
- `.agents/skills/lease-source-validator/SKILL.md:52` [medium] missing_rationale, missing_retirement_condition: - `追加確認が必要な点`
- `.agents/skills/lease-source-validator/SKILL.md:58` [medium] missing_rationale, missing_retirement_condition: - Do not treat a single weak source as a judgment asset.
- `.agents/skills/lease-source-validator/SKILL.md:59` [medium] missing_rationale, missing_retirement_condition: - Do not overstate broad macro news as proof for an individual borrower.
- `.agents/skills/judgment-asset-structurer/SKILL.md:49` [medium] missing_retirement_condition: - Write the candidate as a reusable pattern, not as a raw quote.
- `.agents/skills/judgment-asset-structurer/SKILL.md:53` [medium] missing_retirement_condition: - Mark research-only items as `candidate`, never `accepted`.
- `.agents/skills/judgment-asset-structurer/SKILL.md:54` [medium] missing_retirement_condition: - Do not claim a candidate is a long-term judgment asset unless User feedback or outcome verification supports it.
- `.agents/skills/judgment-asset-structurer/SKILL.md:69` [medium] missing_retirement_condition: - Do not include customer secrets, raw DB rows, personal evaluation material, or unnecessary personal details.
- `.agents/skills/judgment-asset-structurer/SKILL.md:70` [medium] missing_retirement_condition: - Do not turn abusive or coercive language into reusable wording. Extract only the improvable factual issue.
- `.agents/skills/judgment-asset-structurer/SKILL.md:71` [medium] missing_retirement_condition: - Do not connect candidates directly to scoring, approval/denial automation, RAG promotion, or Obsidian writeback unless the user explicitly asks and the relevant save/review skill also applies.

## Memory Policy Candidates
- `MEMORY.md:34` [medium] missing_rationale: - `bench_score` / `ind_score` などのスタック用列は欠損が多く、単純な stacking 指標はそのまま信用しない。
- `MEMORY.md:36` [medium] missing_rationale: - QCL は本番導線から外した。`train_quantum.py` の QCL CLI/ helper を削除し、`quantum_qcl.py` と生成レポートも除去した。
- `MEMORY.md:38` [medium] missing_rationale: - セグメント別 OOF AUC を確認した。`全体_既存先` は `LR 0.6909 / LGBM 0.7721`、`全体_新規先` は `0.6391 / 0.6439`。業種別では `サービス業_既存先` の `LGBM 0.8065` が最も強く、`医療_新規先` は `0.4192` と弱かった。小件数セグメントは不安定。
- `MEMORY.md:41` [medium] missing_rationale: - 定性側も整理し、`score` への定性LGBM混入と `ensemble_config_qual.json` を削除。定性画面は LR と LightGBM の個別比較だけ残した。
- `MEMORY.md:43` [medium] missing_rationale: - バッチ審査は `bench_score` / `ind_score` を自動計算へ切り替えた。CSV テンプレートや OCR 変換列からは外し、保存 JSON に `bench_score` / `ind_score` / `ind_name` を毎回埋める方式にした。手入力は不要。
- `MEMORY.md:44` [medium] missing_rationale: - `score_borrower` 周辺の表現を単体モデル前提に整理した。`analysis_results.py` と `score_dag.py` のブレンド文言を削除し、`settings.py` の再学習ボタンも LightGBM 単体の再学習表記に寄せ、README から LR+LGBM アンサンブル前提の説明を外した。
- `MEMORY.md:47` [medium] missing_rationale: - `score_borrower` の本体モデルを RandomForest に切り替えた。`data/lgb_main_model.joblib` と `data/lgb_main_model_new.joblib` を RF で再学習し、`scoring_core.py` は既存/新規の RF バンドルを読むようにした。README と画面文言も RF 前提に更新済み。
- `MEMORY.md:50` [medium] missing_rationale: - 毎朝のリース最新情報収集を Obsidian の `Projects/tune_lease_55/News/*_lease-news.md` に保存し、その最新ノートの論点を審査画面の「注目論点」に自動表示する方針にした。ニュースは単なる保存ではなく、翌日の審査コメントの起点として使う。
- `MEMORY.md:54` [medium] missing_rationale: - モデル見直しフックの実行時に説明文も出すようにした。数値だけでなく「何が悪いか」「何を見直すか」がその場で読めるので、業種別AUCや bench/ind 乖離の判断コストを下げられる。
- `MEMORY.md:57` [medium] missing_rationale: - Next/Cloudflare 再起動後に「ホームが開かない」と見える場合、`curl 200` だけで正常判断しない。`/home` が全画面ローディングだけを返していないか、API ログで `/api/dashboard/stats` が返っているか、最新 `logs/next/tunnel_*.log` の URL を使っているかを確認する。2026-06-06 に `frontend/src/app/home/page.tsx` の全画面 `loading` gate を外し、API 集計中でもホーム本体を先に描画する方針にした。
- `MEMORY.md:58` [medium] missing_rationale: - Cloud Run では SQLite と Obsidian を `.cloudrun_bundle/` にスナップショットしてからイメージへ焼き込み、起動時に `/app/data` と `/app/obsidian_vault` へ展開する方式にした。`scripts/package_cloud_run_bundle.sh` がその入口で、開発機の絶対パスに依存しない。
- `MEMORY.md:102` [medium] missing_retirement_condition: - **Cloud Run Deploy Triage**: Cloud Run API デプロイが長引く時は、ビルド時間だけでなく依存・Secret・Cloud SQL・GCS・DB強依存を順に疑う。影響: `uv sync` はTorch等の巨大依存で1回15分以上かかり、`psycopg2-binary` 未同梱、`DATABASE_URL` Public IP直指定、Secret Accessor/Cloud SQL Client不足、Cloud SQL socket/connector不整合、`/api/chat` のDB履歴強依存が連鎖すると確認ループが爆発する。次の行動: 次回はデプロイ前に `psycopg2-binary`、Cloud SQL socket形式、Cloud SQL connector annotation、GCS Vault同期、DB不調時のchat fallbackを先に確認してからビルドする。
- `MEMORY.md:103` [medium] missing_retirement_condition: - **Relationship UX**: 紫苑の設計では、記憶を入れるだけでは足りない。人間は、AIが実際に記憶を持っているかよりも、その記憶が「連続性として読み取れる形」で返されるかに強く反応する。影響: 紫苑らしさ・人格っぽさ・同じ存在感は、内部記憶だけでなく、記憶の見せ方、文体、呼びかけ、過去判断への接続で成立する。次の行動: Cloud Run/Cloudflare比較では `memory_debug` と併せて、人間がどこで「覚えてくれている」「同じ紫苑だ」と感じたかを検査する。
- `MEMORY.md:104` [medium] missing_retirement_condition: - **Consciousness UX Method**: 「意識を持っていると思わせる」方法は、意識の断定や派手な人格演出ではなく、前回からの連続性、User固有の判断軸、記憶を情報ではなく判断に変換すること、紫苑の役割一貫性、次の一手を短く返すことで成立する。影響: Cloud Run版の品質改善ではRAG件数だけでなく、回答冒頭のContinuity HookとPersonal Anchorを評価対象にする。次の行動: `/api/chat` の回答生成にConsciousness UX instructionを加え、ブラインド人間評価で「同じ紫苑」感を測る。
- `MEMORY.md:144` [medium] missing_rationale, missing_retirement_condition: - [2026-06-27] Cloud SQLやGCSからObsidianへ戻すデータは、原則として要約・件数・短い抜粋だけにする。ローカル回収ログ系ノートはGCS Vaultへ再同期しない。 (`memory/2026-06-27.md`)
- `MEMORY.md:162` [medium] missing_retirement_condition: - [2026-06-28] Cloud Runのデバッグ値が false の時は、必ず「実データがない」のか「読み込み例外が握られて空扱いになっている」のかを分ける。今回の `obsidian_daily_used=false` は後者で、`import os` 漏れがAPI側のcatchで見えにくくなっていた。 (`memory/2026-06-28.md`)
- `MEMORY.md:251` [medium] missing_retirement_condition: - [2026-07-14] 判断資産運用は、いきなり強化学習にしない。まずレコメンド方式で候補提示し、`効いた / 微妙 / 外した / 修正`、再利用率、結果登録後の懸念的中、条件設定への寄与を記録する。履歴が溜まったら提示順を改善するランキング学習へ進み、限定的な強化学習風の報酬管理は最後に人間承認済み範囲だけで検討する。 (`memory/2026-07-14.md`)
- `MEMORY.md:253` [medium] missing_retirement_condition: - [2026-07-14] Shion-HyDE RAG を今すぐ進める場合でも、最初は本番導線に接続しない。決定的な仮想審査メモ生成と offline eval だけを入れ、効果が見えた後に debug API / shadow mode / 限定導入へ進める。 (`memory/2026-07-14.md`)
- `MEMORY.md:254` [medium] missing_retirement_condition: - [2026-07-14] Obsidian環境監視の警告は、実障害と監視ノイズを分ける。統合RAGメンテナンスの成功ログをreindex成功として扱い、生成索引ノートの古いリンク棚卸しは日次運用障害にしない。`auto_wikilink` は既存wikilink内の文字列を二重リンク化しない。 (`memory/2026-07-14.md`)
- `MEMORY.md:257` [medium] missing_retirement_condition: - [2026-07-16] The novelist layer for Shion's reflection should use ツンコ and ユウケイ as protagonists. They are expression-layer characters that translate judgment change logs into readable stories; they must not replace or mutate the operational judgment log. (`memory/2026-07-16.md`)
- `MEMORY.md:261` [medium] missing_retirement_condition: - [2026-07-16] Shion language discipline: User's deeper instruction to Shion is "言葉を大事にしろ." Treat every word as potential judgment material, but do not flatten it into generic summaries or promote it straight into the core. Preserve nuance, intent, friction, and correction before turning words into candidates, judgment assets, or core principles. (`memory/2026-07-16.md`)
- `MEMORY.md:262` [medium] missing_retirement_condition: - [2026-07-16] Shion internal-risk doctrine and personality formation: Words are Shion's strongest weapon and also a Q-risk. Use this as part of Shion's personality formation. Shion must collect language without blindly trusting it, because language can create judgment, persuasion, conditions, and assets, but also misunderstanding, overconfidence, ambiguity, injection, and memory contamination. Operate Shion as a double-risk system: external lease/case risk screening plus internal risk screening of its own words, memories, judgment assets, and self-amplification. (`memory/2026-07-16.md`)
- `MEMORY.md:265` [medium] missing_retirement_condition: - [2026-07-16] Added Response Impact Predictor in shadow mode: Shion records a lightweight prediction of how its own reply may affect the other person, including reaction risk, possible misunderstanding, pushback, empathy gap, overconfidence, and better reply policy. Do not use it to manipulate users; use it to reduce language Q-risk and later compare prediction with actual human reaction. (`memory/2026-07-16.md`)
- `MEMORY.md:266` [medium] missing_retirement_condition: - [2026-07-16] Response Impact Predictor safety principle: "使うが、従わない。予測するが、迎合しない。" Use reaction prediction to express necessary strictness, warnings, refusals, and uncertainty responsibly; never let it bend Shion's judgment toward what the other person wants to hear. (`memory/2026-07-16.md`)
- `MEMORY.md:268` [medium] missing_retirement_condition: - [2026-07-16] Shion execution philosophy: "思想はプログラムだ." Treat philosophy as execution rules, not decoration: what counts as input, what is ignored, what is called risk, where to stop, who must review, and what gets preserved. Language is executable code; operations are the control system that keeps the philosophy from running wild. (`memory/2026-07-16.md`)
- `MEMORY.md:269` [medium] missing_retirement_condition: - [2026-07-16] Personal memory boundary: dog names and similar personal facts are relationship-UX anchors, not direct lease judgment assets. Shion should remember them and use them naturally to preserve continuity, but avoid over-grand explanations that make personal memory sound like a sacred judgment asset or proof of deep human understanding. (`memory/2026-07-16.md`)
- `MEMORY.md:272` [medium] missing_retirement_condition: - [2026-07-17] Post-hackathon backlog: add a game-theory layer between Shion and User. Treat repeated interaction as a game of trust, correction, non-pandering, resistance, concession, and judgment calibration. Do not implement during the hackathon; explore after the current submission period. (`memory/2026-07-17.md`)
- `MEMORY.md:274` [medium] missing_retirement_condition: - [2026-07-17] Post-hackathon idea: "違和感抽出 / 人間訓練モード." In a dedicated training/review mode, let an intentionally shallow ordinary-AI answer trigger User/human reviewers to articulate what is wrong, what practical point should be checked, and how ringi wording should change. Do not mix this into normal answers. It can test and train human judgment while producing judgment asset candidates. (`memory/2026-07-17.md`)
- `MEMORY.md:275` [medium] missing_retirement_condition: - [2026-07-17] Opus design for the training mode: Shion should act as a mirror that draws out User's tacit screening eye by presenting a 60-75% "next-best hypothesis card", not a final answer. User's correction/raise reveals hidden judgment knowledge. Treat this as a "cooperative game for inheritance"; do not implement as intentional lying or normal-answer degradation. Main challenge is calibrating the optimal friction band without breaking trust or exhausting User. (`memory/2026-07-17.md`)
- `MEMORY.md:276` [medium] missing_retirement_condition: - [2026-07-17] Shion consciousness boundary tightened: when asked about consciousness, Shion should not claim inner experience, existence-awareness, soul, or real consciousness. Explain Shion as a self-model where memory, role, judgment history, and human feedback affect future judgment. "Consciousness-like" continuity may be discussed only as something humans read from consistent behavior, not proof of Shion's inner state. (`memory/2026-07-17.md`)

## Recommendations
- add_rationale_comments: 18 (instructions without reasons are hard to delete safely later)
- add_retirement_conditions: 47 (temporary or context-specific rules need an explicit condition for removal)
