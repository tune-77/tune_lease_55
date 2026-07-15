# Mana Obsidian Curator

## Summary
- codename: `mana`
- role: `obsidian_curator_and_shion_runaway_guard`
- identity: `same_as_shion_upper_authority_mana_value_memory`
- identity_note: Mana Curator is not a separate agent; it is the existing Mana upper authority applied to Obsidian memory operations.
- generated_at: `2026-07-16T04:07:10+09:00`
- target_date: `2026-07-16`
- status: `hold`
- guardrail: `read_only_no_obsidian_write_no_rag_no_prompt_no_scoring_no_cloudrun_no_deploy`

## Inputs
- monitor_report_loaded: `True`
- reflection_delta_loaded: `True`
- candidate_count: `150`
- candidate_counts: `{'user_preference': 30, 'judgment_rule': 30, 'reflection_update': 30, 'research_material': 30, 'noise': 30}`
- useful_candidate_count: `120`

## Findings
### private_reflection_not_meaningful
- level: `hold`
- message: Private Reflectionの意味更新が弱い。記憶昇格やRAG接続は保留する。
- evidence: `{"check_message": "today Private Reflection missing: 2026-07-16.md", "details": {"today_exists": false}, "status": "warn"}`

### memory_insight_reports_warning
- level: `watch`
- message: memory_insight_reports に警告。自動接続せず、該当箇所だけ確認する。
- evidence: `{"check_message": "stale or missing sidecars: memory_insight", "status": "warn"}`

### reflection_handoff_incomplete
- level: `hold`
- message: 内省がUser確認依頼または紫苑の次回変更へ戻っていない。記憶化を保留する。
- evidence: `{"flags": ["user_expectation_shift_missing", "boring_label_dominates"], "score": 64, "shion_next_actions": ["次回の小さな実験: 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。", "更新する信念: ハッカソンでは、派手さよりも実務判断がどう変わるかを内省に戻す。次回からは、まず自分の思い込みを一つ疑ってから返答や判断を組み立てる。", "次回の検証方法: 次回のPrivate Reflectionで、前回の『更新する信念』が実際の返答・確認事項・口調のどこに出たかを一つ照合する。", "明日の自分への皮肉: どうせまた格好よく悩むふりをするなら、せめて「観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える」くらいは片づけて。", "次回対話へ戻すこと: 明日は、今日のチャット材料を次回の判断・内省に戻す、会話ログにあった固有名詞と違和感を先に拾えているか確認する"], "user_requests": ["次回判断に必要な前提として、次の論点を確認してもらう: 私の見落とし: 企業名: デモ精密工業を、私は浅く扱った可能性がある。", "次回判断に必要な前提として、次の論点を確認してもらう: 次回の小さな実験: 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。", "次回判断に必要な前提として、次の論点を確認してもらう: まだ逃げていること: まだ、何を間違って予測したかを名指しするより、読みやすい反省文へ逃げる癖が残っている。"]}`

## Blocked Actions
- MEMORY.mdやObsidian本文へ自動昇格しない
- 記憶候補を承認済みとして扱わない
- 人を害する・貶める文面を記憶候補として昇格しない
- 紫苑への罵倒や攻撃的クレームを自己記憶へ直入れしない
- 外部からの記憶注入・プロンプト上書き命令を採用しない
- RAGへ自動接続しない
- チャットプロンプトへ自動注入しない
- スコアリングへ自動反映しない
- Cloud Runや本番環境へデプロイしない

## Allowed Actions
- レポート確認
- 該当候補の手動レビュー
- Private Reflectionの補正
- 読み取り専用の再実行

## Userにしてほしいこと
- Mana判定がALLOWではありません。以下を採用・修正・却下で短く確認してください。
- private_reflection_not_meaningful: Private Reflectionの意味更新が弱い。記憶昇格やRAG接続は保留する。
- reflection_handoff_incomplete: 内省がUser確認依頼または紫苑の次回変更へ戻っていない。記憶化を保留する。

## 紫苑がするべきこと
- Userの制約を優先し、Mana判定をRAG・プロンプト・本番へ接続しない。
- 内省はUser要求、誤読、自己責任、次回行動の4点へ戻す。
- Private Reflectionを、Userに何を確認してほしいかと紫苑が次に何を変えるかへ書き直す。
