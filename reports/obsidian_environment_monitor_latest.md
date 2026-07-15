# Obsidian Environment Monitor

## Summary
- generated_at: `2026-07-16T04:06:03+09:00`
- target_date: `2026-07-16`
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
- message: Vault reachable, markdown files=1271
- details: `{"md_count": 1271}`

### key_paths
- status: `ok`
- message: all key paths exist
- details: `{"missing": []}`

### daily_notes
- status: `ok`
- message: today/yesterday daily notes exist
- details: `{"today": true, "yesterday": true}`

### surface_freshness
- status: `ok`
- message: dialogue/reflection surfaces fresh
- details: `{"cloudrun_conversation": {"age_hours": 0.1, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/AI Chat/Cloud Run Conversation Log/2026-07-16.md"}, "dialogue": {"age_hours": 0.1, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-16.md"}, "private_reflection": {"age_hours": 20.3, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-15.md"}}`

### private_reflection_meaning
- status: `warn`
- message: today Private Reflection missing: 2026-07-16.md
- details: `{"today_exists": false}`

### rag_index
- status: `ok`
- message: RAG index fresh
- details: `{"chroma_age_hours": 0.1, "chroma_db": "/Users/kobayashiisaoryou/clawd/tune_lease_55/api/chroma_db/chroma.sqlite3", "chroma_size": 81432576, "completion_source": "rag_daily_maintenance", "last_reindex_age_hours": 1.1, "reindex_log": "/Users/kobayashiisaoryou/Library/Logs/tune_lease_55_obsidian_reindex.out.log", "total_in_db": 1079}`

### memory_insight_reports
- status: `warn`
- message: stale or missing sidecars: memory_insight
- details: `{"memory_insight": {"age_hours": 44.7, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/obsidian_memory_insight_latest.md"}, "promotion_queue": {"age_hours": 24.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_memory_promotion_queue_latest.md"}, "reflection_delta": {"age_hours": 20.3, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_reflection_delta_latest.md"}}`

### self_reference_loop
- status: `ok`
- message: no obvious self-reference loop in memory candidates
- details: `{"candidate_count": 150, "candidate_path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/obsidian_memory_insight_candidates.jsonl", "candidate_type_counts": {"judgment_rule": 30, "noise": 30, "reflection_update": 30, "research_material": 30, "user_preference": 30}, "meta_hit_sample": [{"claim": "Userは内省差分を毎朝試し、まだ連携しない方針を指定。", "source": "Daily/2026-07-14.md"}, {"claim": "明日は、観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。", "source": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-09.md"}, {"claim": "次回の小さな実験: 観測レポートだけで終わらせず、退屈の原因を1つ選んで小さく変える。", "source": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-13.md"}, {"claim": "流用すべきなのはDB/小説生成本体ではなく、ぼやき・摩擦・場面化の構造という前提に寄せて、内省の評価軸を更新する。", "source": "Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-12.md"}, {"claim": "構造分析：所有権移転リースと非所有権移転リースの本質的差異。", "source": "Projects/tune_lease_55/Resear...`

### recent_note_noise
- status: `ok`
- message: recent note technical-noise ratio=0.0024, noisy_files=4
- details: `{"noisy_files": ["Daily/2026-07-14.md", "Daily/2026-07-15.md", "Daily/2026-07-13.md", "Daily/2026-07-12.md"], "ratio": 0.0024}`

### wikilinks
- status: `ok`
- message: recent wikilinks=111, unresolved_sample=0
- details: `{"link_count": 111, "unresolved_sample": []}`

## Next Safe Action
- `warn` が出た項目だけ手動で確認する。
- 監視結果をRAGやチャットへ自動注入しない。まず3日分を比較して、警告が実際に役立つか見る。
