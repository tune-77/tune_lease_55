# Ops Friction Doctor

## Gitship dirty noise [high]
- score: 89
- scope: Gitship前
- reason: recent logs mention this pattern 5 times; current dirty tree has 84 generated-like files
- next: python scripts/classify_git_ship_candidates.py
- log: reports/agent_worklog_digest_latest.md (5 hits): - Decisions: 通常経路とGemini agent経路が単一制約JSONを共有し、検証はdetached git worktreeで行う。運用レベルはL2-assistedを維持する。 / - Decisions: 自動改善はL2 Assistedを維持し、全実装経路を単一JSON制約とdetached-worktree verifierへ集約。L3はtoken budget/pause-all実装まで見送る。 / - Decisions: 大量の既存dirty変更を混入させず、origin/master起点の専用worktreeで今回の7ファイルだけをPR化した。

## Memory pipeline review backlog [high]
- score: 25
- scope: 記憶/判断資産パイプライン確認時
- reason: recent logs mention this pattern 25 times
- next: python scripts/build_shion_memory_sentinel_report.py
- auto-safe: python scripts/build_shion_memory_sentinel_report.py
- log: reports/shion_memory_sentinel_latest.md (1 hits): - `watch` usage_effect: needs_feedback=137 - 想起はされたが、回答で本当に効いたか未確認の記憶が残っている
- log: reports/obsidian_environment_monitor_latest.md (1 hits): - details: `{"memory_insight": {"age_hours": 0.0, "exists": true, "path": "/Users/kobayashiisaoryou/clawd/tune_lease_55/reports/obsidian_memory_insight_latest.md"}, "promotion_queu
- log: reports/orphaned_scripts_latest.md (1 hits): - `scripts/reconcile_needs_review_from_prs.py` (test exists)

## Cloud Run return sync gap [low]
- score: 2
- scope: Cloud Runで入力/feedbackを使った後
- reason: recent logs mention this pattern 2 times
- next: CLOUDRUN_INPUT_SYNC_DAYS=2 python scripts/sync_cloudrun_inputs_from_gcs.py
- log: reports/shion_memory_effect_latest.md (1 hits): - mem_bcdd44c169f26e87 [stale] state=needs_review reason=stale 状態だが想起されている [2026-07-07] Fixed Cloud Run input improvement reflection by adding `improvement_note` handling to `scrip
- log: reports/screening_terms_audit_latest.md (1 hits): - `review` `actual_pd` `scripts/sync_cloudrun_inputs_from_gcs.py:545` — PD表記だが、算出済みか補助指標かの区別が読み取りにくい

## Local deploy restart ambiguity [low]
- score: 2
- scope: ローカル起動/再起動前
- reason: recent logs mention this pattern 2 times
- next: python scripts/local_deploy_doctor.py --public-tunnel
- log: reports/shion_memory_effect_latest.md (1 hits): - mem_e679b29aa5943233 [stale] cloudflared インストール: `brew install cloudflare/cloudflare/cloudflared`
- log: reports/agent_worklog_digest_latest.md (1 hits): - Changes: PR #966: APIキー認証、Trusted Hosts、公開Web境界、sessionStorage、CSP、デプロイスクリプト、セキュリティ報告。PR #968: LaunchAgent認証情報、judgment-drill内部認証、/healthz診断。


## Auto-Safe Results
- memory_pipeline_review: applied exit=0 `python scripts/build_shion_memory_sentinel_report.py`
