# LlamaIndex RAG Comparison

- Generated: 2026-08-24T02:16:05
- Status: completed
- Method: llama_index.core.VectorStoreIndex + local char-ngram embedding
- Top K: 5
- Guardrail: sidecar_only_no_rag_rank_change_no_prompt_change_no_scoring_no_obsidian_write

## Summary

- hit@1: 12/12 (100.0%)
- hit@5: 12/12 (100.0%)
- mrr: 1.000
- forbidden_cases: 0/12
- passed: True

## Cases

### okf_pack_index

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/README.md
2. knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md
3. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md
4. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md
5. knowledge_base/okf_lease_concepts/rules/current_datetime_prompt_context.md

### okf_hydraulic_excavator_period

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md
2. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
3. knowledge_base/okf_lease_concepts/README.md
4. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
5. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md

### okf_forklift_resale

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
2. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md
3. knowledge_base/okf_lease_concepts/rules/conditional_approval_playbook.md
4. knowledge_base/okf_lease_concepts/rules/current_datetime_prompt_context.md
5. knowledge_base/okf_lease_concepts/rules/score_60_80_inversion.md

### okf_machine_tool_resale

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md
2. knowledge_base/okf_lease_concepts/README.md
3. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md
4. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
5. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md

### okf_medical_equipment_resale

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
2. knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md
3. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md
4. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
5. knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md

### okf_conditional_approval

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/conditional_approval_playbook.md
2. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md
3. knowledge_base/okf_lease_concepts/rules/score_60_80_inversion.md
4. knowledge_base/okf_lease_concepts/README.md
5. knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md

### okf_current_datetime

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/current_datetime_prompt_context.md
2. knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md
3. knowledge_base/okf_lease_concepts/README.md
4. knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md
5. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md

### okf_expected_usage_period

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md
2. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
3. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md
4. knowledge_base/okf_lease_concepts/assets/hydraulic_excavator_lease_period.md
5. knowledge_base/okf_lease_concepts/README.md

### okf_q_risk

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/q_risk_interpretation.md
2. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
3. knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md
4. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
5. knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md

### okf_score_60_80_inversion

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/score_60_80_inversion.md
2. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md
3. knowledge_base/okf_lease_concepts/rules/q_risk_interpretation.md
4. knowledge_base/okf_lease_concepts/rules/conditional_approval_playbook.md
5. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md

### okf_shion_memory_retrieval

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md
2. knowledge_base/okf_lease_concepts/README.md
3. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
4. knowledge_base/okf_lease_concepts/rules/score_60_80_inversion.md
5. knowledge_base/okf_lease_concepts/assets/forklift_resale_risk.md

### okf_statutory_useful_life

- Status: PASS
- Rank: 1

1. knowledge_base/okf_lease_concepts/rules/statutory_useful_life.md
2. knowledge_base/okf_lease_concepts/assets/medical_equipment_resale_risk.md
3. knowledge_base/okf_lease_concepts/rules/expected_usage_period_and_lease_term.md
4. knowledge_base/okf_lease_concepts/rules/shion_memory_retrieval.md
5. knowledge_base/okf_lease_concepts/assets/machine_tool_resale_risk.md
