# Obsidian Environment Monitor

## Summary
- generated_at: `2026-07-20T04:08:06+09:00`
- target_date: `2026-07-20`
- status: `warn`
- guardrail: `monitor_only_no_obsidian_write_no_rag_no_prompt_no_cloudrun`

## Viewpoints
- 鮮度: 今日/昨日のDaily・対話・Private Reflectionが揃っているか
- 内省品質: Private Reflectionが昨日と違い、User要求・誤読・自己責任・次回行動を含むか
- 同期: Cloud Run会話ログがObsidianへ戻っているか
- 検索性: reindex/ChromaDBが古くないか
- 記憶形成: 内省差分・記憶候補・Obsidian insightが生成されているか
- ワーム化防止: 自分のレポート・内省・Daily作業ログを材料に候補が増殖していないか
- ノイズ: 技術ログや一時出力が知識ノートを汚していないか
- リンク: 直近ノートのwikilinkが解決できるか
- 安全性: 監視は読み取り専用で、本番・Cloud Run・RAGに接続しない

## Checks
### vault
- status: `ok`
- message: Vault reachable, markdown files=1384
- details: `{"md_count": 1384}`

### key_paths
- status: `ok`
- message: all key paths exist
- details: `{"missing": []}`

### daily_notes
- status: `warn`
- message: missing daily notes: 2026-07-20.md
- details: `{"today": false, "yesterday": true}`

### surface_freshness
- status: `ok`
- message: dialogue/reflection surfaces fresh
- details: `{"cloudrun_conversation": {"age_hours": 0.1, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-19.md"}, "dialogue": {"age_hours": 0.1, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-19.md"}, "private_reflection": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-20.md"}}`

### private_reflection_meaning
- status: `warn`
- message: Private Reflection exists but meaningful update is weak: too_similar_to_yesterday:0.931
- details: `{"matched_labels": ["今日の観察:", "私の見落とし:", "仮説の更新:", "次回の小さな実験:", "前回の入力:", "前回の判断:", "人間の修正:", "紫苑が外した点:", "次回から変える確認事項:", "判断資産候補:", "まだ確信できない点:", "私の責任:", "更新する信念:", "次回の検証方法:"], "missing_categories": [], "required_categories": ["misread", "next_behavior", "self_responsibility", "user_expectation"], "similarity_to_yesterday": 0.931, "today_length": 2958, "today_path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-20.md", "yesterday_path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-19.md"}`

### rag_index
- status: `ok`
- message: RAG index fresh
- details: `{"chroma_age_hours": 0.1, "chroma_db": "/Users/kobayashiisaoryou/clawd/tune_lease_55/api/chroma_db/chroma.sqlite3", "chroma_size": 84754432, "completion_source": "rag_daily_maintenance", "last_reindex_age_hours": 1.1, "reindex_log": "/Users/kobayashiisaoryou/Library/Logs/tune_lease_55_obsidian_reindex.out.log", "total_in_db": 1180}`

### memory_insight_reports
- status: `ok`
- message: memory insight sidecars fresh
- details: `{"memory_insight": {"age_hours": 18.9, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/obsidian_memory_insight_latest.md"}, "promotion_queue": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_memory_promotion_queue_latest.md"}, "reflection_delta": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_reflection_delta_latest.md"}}`

### self_reference_loop
- status: `ok`
- message: no obvious self-reference loop in memory candidates
- details: `{"candidate_count": 150, "candidate_path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/obsidian_memory_insight_candidates.jsonl", "candidate_type_counts": {"judgment_rule": 30, "noise": 30, "reflection_update": 30, "research_material": 30, "user_preference": 30}, "meta_hit_sample": [{"claim": "ハッカソン環境を壊さないため、監視は読み取り専用・未連携に限定した。", "source": "Daily/2026-07-14.md"}, {"claim": "ユーザーが改善PMレポートにstatusログ、修正済み候補、通常リース相談が出る問題を指摘。", "source": "Daily/2026-07-18.md"}, {"claim": "明日は、観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。", "source": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-19.md"}, {"claim": "構造分析：所有権移転リースと非所有権移転リースの本質的差異。", "source": "Projects/tune_lease_55/Research/所有権移転リース　総合レポート_(executive_report)_report (1).md"}, {"claim": "[ ] 現在、登録申請中（登録完了予定日：＿＿年＿＿月頃）。", "source": "Projects/tune_lease_55/Research/インボイス　総合レポート_(executive_report)_report (1).md"}, {"claim": "期間安定...`

### recent_note_noise
- status: `ok`
- message: recent note technical-noise ratio=0.0054, noisy_files=3
- details: `{"noisy_files": ["Daily/2026-07-18.md", "Daily/2026-07-17.md", "Daily/2026-07-16.md"], "ratio": 0.0054}`

### wikilinks
- status: `ok`
- message: recent wikilinks=105, unresolved_sample=0
- details: `{"link_count": 105, "unresolved_sample": []}`

## Next Safe Action
- `warn` が出た項目だけ手動で確認する。
- 監視結果をRAGやチャットへ自動注入しない。まず3日分を比較して、警告が実際に役立つか見る。
