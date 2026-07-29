# Obsidian Memory Effectiveness

- Date: 2026-07-30
- Mode: read_only_observation
- Guardrail: no_rag_rank_change_no_prompt_change_no_obsidian_write_no_auto_promotion
- Knowledge dir: `/Users/kobayashiisaoryou/clawd/tune_lease_55/knowledge_base/okf_lease_concepts`

## Summary

- Total: 12
- Dormant: 0
- Recalled: 2
- Used: 9
- Validated: 1
- Noisy: 0

## Records

### OKF-style lease knowledge pack
- Ref: `knowledge_base/okf_lease_concepts/README.md`
- Type: `index` / Domain: `rag`
- State: `recalled` / Score: 20.0
- Signals: recalled=5, used=0, helped=0, challenged=0, rejected=0
- Next: 回答・確認事項・稟議文面に実際に使われたかを確認する。

### 紫苑の記憶参照と検索効率
- Ref: `knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md`
- Type: `agent_policy` / Domain: `rag`
- State: `recalled` / Score: 18.0
- Signals: recalled=3, used=0, helped=0, challenged=0, rejected=0
- Next: 回答・確認事項・稟議文面に実際に使われたかを確認する。

### Q_riskの解釈
- Ref: `knowledge_base/okf_lease_concepts/rules/q_risk_interpretation.md`
- Type: `risk_signal` / Domain: `q_risk`
- State: `used` / Score: 40.0
- Signals: recalled=6, used=4, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### スコア60-80帯の成約率逆転
- Ref: `knowledge_base/okf_lease_concepts/rules/score_60_80_inversion.md`
- Type: `risk_signal` / Domain: `credit`
- State: `used` / Score: 40.0
- Signals: recalled=4, used=5, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 期待使用期間とリース期間
- Ref: `knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md`
- Type: `lease_rule` / Domain: `contract`
- State: `used` / Score: 40.0
- Signals: recalled=8, used=3, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 条件付き承認の判断ルール
- Ref: `knowledge_base/okf_lease_concepts/rules/conditional_approval_playbook.md`
- Type: `lease_rule` / Domain: `credit`
- State: `used` / Score: 36.0
- Signals: recalled=4, used=2, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 法定耐用年数データの扱い
- Ref: `knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md`
- Type: `lease_rule` / Domain: `asset_life`
- State: `used` / Score: 36.0
- Signals: recalled=4, used=2, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### フォークリフトの残価・再販リスク
- Ref: `knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md`
- Type: `asset_profile` / Domain: `asset_life`
- State: `used` / Score: 28.0
- Signals: recalled=6, used=1, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 医療機器の保守期限・撤去費・再販リスク
- Ref: `knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md`
- Type: `asset_profile` / Domain: `asset_life`
- State: `used` / Score: 28.0
- Signals: recalled=6, used=1, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 油圧ショベルのリース期間
- Ref: `knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md`
- Type: `asset_profile` / Domain: `asset_life`
- State: `used` / Score: 28.0
- Signals: recalled=10, used=1, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### AI回答の現在日時コンテキスト
- Ref: `knowledge_base/okf_lease_concepts/rules/current_datetime_prompt_context.md`
- Type: `agent_policy` / Domain: `agent`
- State: `used` / Score: 14.0
- Signals: recalled=1, used=1, helped=0, challenged=0, rejected=0
- Next: User評価を取り、helped / neutral / challenged を記録する。

### 工作機械の残価・再販リスク
- Ref: `knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md`
- Type: `asset_profile` / Domain: `asset_life`
- State: `validated` / Score: 44.0
- Signals: recalled=3, used=2, helped=1, challenged=0, rejected=0
- Next: 同種案件で再利用し、結果照合できる材料を待つ。

## Notes

- このレポートは観測のみ。RAG順位、プロンプト、Obsidian本文、判断資産active storeは変更しない。
- RAG評価で引けたものは `recalled` として扱うが、実業務で使われた証拠とは分ける。
- `validated` は人間の helped 等の明示フィードバックがある場合だけ強く扱う。
