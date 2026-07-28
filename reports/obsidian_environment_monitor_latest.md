# Obsidian Environment Monitor

## Summary
- generated_at: `2026-07-29T02:30:53+09:00`
- target_date: `2026-07-29`
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
- message: Vault reachable, markdown files=1612
- details: `{"md_count": 1612}`

### key_paths
- status: `ok`
- message: all key paths exist
- details: `{"missing": []}`

### daily_notes
- status: `ok`
- message: today/yesterday daily notes exist
- details: `{"today": true, "yesterday": true}`

### surface_freshness
- status: `warn`
- message: stale or missing surfaces: cloudrun_conversation
- details: `{"cloudrun_conversation": {"age_hours": null, "exists": false}, "dialogue": {"age_hours": 23.6, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Dialogue/2026-07-28.md"}, "private_reflection": {"age_hours": 22.4, "exists": true, "path": "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/Projects/tune_lease_55/Lease Intelligence/Private Reflection/2026-07-28.md"}}`

### private_reflection_meaning
- status: `warn`
- message: today Private Reflection missing: 2026-07-29.md
- details: `{"today_exists": false}`

### rag_index
- status: `ok`
- message: RAG index fresh
- details: `{"chroma_age_hours": 20.3, "chroma_db": "/Users/kobayashiisaoryou/clawd/tune_lease_55/api/chroma_db/chroma.sqlite3", "chroma_size": 91029504, "completion_source": "rag_daily_maintenance", "last_reindex_age_hours": 23.5, "reindex_log": "/Users/kobayashiisaoryou/Library/Logs/tune_lease_55_obsidian_reindex.out.log", "total_in_db": 1365}`

### memory_insight_reports
- status: `ok`
- message: memory insight sidecars fresh
- details: `{"memory_insight": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/obsidian_memory_insight_latest.md"}, "promotion_queue": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_memory_promotion_queue_latest.md"}, "reflection_delta": {"age_hours": 22.4, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/shion_reflection_delta_latest.md"}}`

### self_reference_loop
- status: `ok`
- message: no obvious self-reference loop in memory candidates
- details: `{"candidate_count": 123, "candidate_path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/obsidian_memory_insight_candidates.jsonl", "candidate_type_counts": {"judgment_rule": 30, "noise": 30, "reflection_update": 30, "research_material": 30, "user_preference": 3}, "meta_hit_sample": [{"claim": "構造分析：所有権移転リースと非所有権移転リースの本質的差異。", "source": "Projects/tune_lease_55/Research/所有権移転リース　総合レポート_(executive_report)_report (1).md"}, {"claim": "[ ] 現在、登録申請中（登録完了予定日：＿＿年＿＿月頃）。", "source": "Projects/tune_lease_55/Research/インボイス　総合レポート_(executive_report)_report (1).md"}, {"claim": "期間安定性: 5年以上の長期間、仕様変更なく利用する環境か。", "source": "Projects/tune_lease_55/Research/ソフトリース　総合レポート_(executive_report)_report (1).md"}, {"claim": "補助金ルールにおいて、「交付決定前の契約」は補助対象外です。", "source": "Projects/tune_lease_55/Research/補助金　総合レポート.md"}, {"claim": "以下の3点を必ず遵守してください。", "source": "Projects/tune_lease_55/Research/補助金　総合レポート.md"}],...`

### recent_note_noise
- status: `ok`
- message: recent note technical-noise ratio=0.005, noisy_files=4
- details: `{"noisy_files": ["Daily/2026-07-28.md", "Daily/2026-07-26.md", "Daily/2026-07-27.md", "Daily/2026-07-25.md"], "ratio": 0.005}`

### wikilinks
- status: `ok`
- message: recent wikilinks=100, unresolved_sample=0
- details: `{"link_count": 100, "unresolved_sample": []}`

## Next Safe Action
- `warn` が出た項目だけ手動で確認する。
- 監視結果をRAGやチャットへ自動注入しない。まず3日分を比較して、警告が実際に役立つか見る。
