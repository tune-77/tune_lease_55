# Shion Memory Sentinel

- Status: `watch`
- Guardrail: `read_only_no_memory_write_no_prompt_no_rag_rank_no_scoring_no_auto_promotion`
- Memory records: 764
- Long-term domain coverage: 1.0
- Long-term use_when coverage: 1.0
- Usage events: 1435
- Used memory ids: 241
- Likely helpful: 57
- Needs feedback: 99
- Possible noise: 0
- Open human reviews: 236
- Open human review batches: 70
- Contradiction candidates: 0

## Source Reports
- `data/shion_memory_index.json` status=loaded generated_at=2026-08-25T08:26:47
- `reports/shion_memory_effect_latest.json` status=loaded generated_at=2026-08-25T08:26:47
- `reports/memory_engineering_latest.json` status=loaded generated_at=2026-08-25T08:26:47
- `reports/shion_memory_contradictions_latest.json` status=loaded generated_at=2026-08-25T07:42:14
- `reports/persistent_memory_audit_latest.json` status=loaded generated_at=2026-08-25T04:15:51
- `reports/obsidian_memory_effectiveness_latest.json` status=loaded generated_at=2026-08-25T04:17:00

## Watch Signals
- `watch` usage_effect: needs_feedback=99 - 想起はされたが、回答で本当に効いたか未確認の記憶が残っている
- `watch` memory_engineering: open_reviews=236, batches=70 - 候補記憶・判断資産の人間レビュー待ちが多い
- `watch` memory_engineering: write_policy_metadata=0.411 - 候補記憶の importance/confidence/trust/provenance が薄い

## Feedback Triage
- `scoring_model` long_term/judgment_memory: 3 records, used=350
  - `mem_42683fdb02f4c3f9` used=346 定量 39特徴の OOF AUC は LR 0.6769、MLP 0.6865 で、現行スコアより高かった。
  - `mem_413848889df0d7a9` used=3 [2026-07-16] Presentation lines to retain: "人間の判断を、AIが再利用できる構文に変える", "これはプロンプトエンジニアリングを、個人技ではなく業務プロセスにしたものです", and "紫苑は、判断プロンプトのPDCA基盤です." (
  - `mem_cd305d664165b464` used=1 [2026-07-14] 第3世代の紫苑は、判断資産パックの蓄積から業界別審査モデル仮説を提案する。ただし本番スコアリング変更は人間検証・承認後に限定する。 (`memory/2026-07-14.md`)
- `expected_usage_period_and_lease_term` retrieval/judgment_memory: 4 records, used=64
  - `mem_492635dd0d1d2e43` used=53 要点: リース期間は法定耐用年数だけで決めず、顧客の実際の使用予定期間、物件の経済的寿命、満了後の再販・再リース可能性を合わせて見る。
  - `mem_cb116629d81d76f7` used=9 判断ルール: リース期間が物件寿命に対して長すぎる場合は、満了時価値と故障リスクを明示する。
  - `mem_82c116212fefa254` used=1 根拠: 期間設定は返済原資、物件価値、顧客の利用継続性が交差するため、単一の年数表だけでは判断できない。
- `lease_screening` long_term/judgment_memory: 7 records, used=45
  - `mem_0a0fa02c7a29323d` used=21 [2026-06-28] 紫苑の回答品質改善では、記憶レコードだけを増やすより、質問を「場面」に割り当てる索引が効く。特に境界案件では、手順層=何を見るか、意味層=なぜそこを見るか、判断層=例外時どうするかを冒頭の内部文脈に入れると、一般論ではなくリース判断資産として返しやすい。
  - `mem_dd7ffc5f3844567c` used=8 **Shion Genetic Loop Engineering**: 判断資産は固定ルールではなく、親判断資産が案件環境で変異し、人間の `helped / challenged / rejected` という選択圧を受け、選択・修正・却下されて次世代へ継承されるものとして扱う
  - `mem_d1c4c1e513a10361` used=7 [2026-07-14] 判断資産は「正しい一般論」ではなく、案件の確認行動・承認条件・反証材料・否認理由を変えたものだけ昇格対象にする。当たり前なことを言って仕事した扱いにしない。 (`memory/2026-07-14.md`)
- `rag_memory_ops` long_term/judgment_memory: 3 records, used=34
  - `mem_1660981d08c7141c` used=28 ニュースを見て判断を変えた操作を 1 クリックで記録するボタンを追加した。押下内容は Obsidian の `Daily/YYYY-MM-DD.md` に追記し、`data/lease_news_metrics.json` の `judgment_changes` でも数える。
  - `mem_09edd06d0bc354a5` used=3 [2026-06-27] Cloudflare版/Cloud Run版の品質比較は、既存の回答品質評価に加えて「記憶密度」「過去判断への接続」「言い回しの紫苑らしさ」「ユーザー文脈の保持」を測る第2評価を作る必要がある。必須概念スコアだけだとCloud Runが高く見える場合があ
  - `mem_c35b9df1fa940795` used=3 **Shion Resurrection Candidates**: 判断資産は、当時Userが選んで正しかった判断だけでなく、選ばれなかったが結果登録後に「本当はそちらを見るべきだった」と判明した判断も復活候補として扱う。影響: 結果登録は当時の選択を固定正解にするだけでなく、
- `scoring_model` long_term/dialogue_memory: 1 records, used=32
  - `mem_4f4486ccc93eee79` used=32 改善ログに「修正登録」ボタンを追加し、押した内容を `pdca_ai_rules.json` の `ai_prompt_addons` に1クリックで追記できるようにした。影響: 修正をユーザー操作から直接ルール化できるようになった。次の行動: 追加されたルールが過剰に増えないか
- `system_ops` long_term/technical_memory: 9 records, used=27
  - `mem_264f26ee1fe2b2a6` used=8 [2026-07-15] Cloud Run Webのユーザー向け案内URLは必ず `https://tune-lease-55-web-6mijhyebkq-an.a.run.app/` を使う。API疎通確認で `https://tune-lease-55-api-10208
  - `mem_bd95f2b607c0c728` used=7 [2026-07-12] Added a canonical filtering layer, `scripts/build_canonical_judgment_rules.py`, that compresses similar judgment materials into
  - `mem_9c7d982f8d3116e5` used=3 `scripts/recursive_self_improvement.py` を追加し、改善レポートと prompt feedback を束ねた再帰的自己改善レポートを日次改善パイプラインに接続した。影響: 改善結果が次の改善候補に戻る閉ループが実装された。次の行動: `rep
- `user_preference` long_term/dialogue_memory: 4 records, used=24
  - `mem_b0824f8984afbe83` used=17 **Core Motivation**: User wants to systematize and preserve all lease know-how he has learned, even if it is unclear how far the project can
  - `mem_807e759ae19e207d` used=4 [2026-07-16] Shion language discipline: User's deeper instruction to Shion is "言葉を大事にしろ." Treat every word as potential judgment material, b
  - `mem_84f6c918d004d79a` used=2 **User**: User
- `rag_memory_ops` long_term/dialogue_memory: 1 records, used=19
  - `mem_8eac231f1aadd579` used=19 **AI Chat / Knowledge Loop**: Obsidian 連携を使って、会話メモ・改善ログ・Webメモ・Wiki を相互参照させる方針を好む。最終目的は、リースシステム自体が保存知識を再利用しながら自律的に改善していくこと。
- `q_risk_interpretation` retrieval/judgment_memory: 4 records, used=18
  - `mem_aa56dac325bdb4bc` used=13 判断ルール: 価格、競合、補助金タイミング、銀行支援、営業説明、物件の必要性などのズレを確認する。
  - `mem_c05770a3af79a9dc` used=3 判断ルール: 学習・分析では既存スコアの精度改善と、スコア外因子の発見を分ける。
  - `mem_5b4d155c7d73a3e5` used=1 要点: Q_riskは既存スコアの補正係数ではなく、スコアリング外で成約・失注を動かす未知因子を見つける探索シグナルとして扱う。
- `rag_memory_ops` long_term/technical_memory: 10 records, used=17
  - `mem_7aef6e6d3f542ca1` used=3 [2026-06-27] Cloud Run版の `/api/chat` で `identity_memory` と `memory_recall` は出るが `knowledge_refs=0` / `rag_context_used=false` / `obsidian_da
  - `mem_a8de192128719a3d` used=3 **Cloud Run Deploy Triage**: Cloud Run API デプロイが長引く時は、ビルド時間だけでなく依存・Secret・Cloud SQL・GCS・DB強依存を順に疑う。影響: `uv sync` はTorch等の巨大依存で1回15分以上かかり、`ps
  - `mem_b8409d21332bd5c6` used=3 [2026-06-27] Cloud Runへ寄せる時も、Cloudflare版の「記憶が近い」「返答が厚い」「紫苑らしい」体験を劣化させないことを重視する。クラウド化は置き換えではなく、Cloudflare版で愛着を持てた仕様の再現・拡張として進める。 (`memory/202

## Next Actions
- チャットUIの 効いた/微妙/違う フィードバックを優先的に集める
- review inbox を同種テーマで束ね、承認/保留/却下を分ける
- 新規候補生成時に write policy metadata を必須化する
