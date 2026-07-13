# Shion Timeline Delta - 2026-07-13

## 差分サマリ
- 前日類似度: 0.0667
- 継続語: Obsidian、RAG
- 新規/強化語: B2B、LLM、Q_risk、Qrisk、事前判断、人間、仮説、判断基準、前回、前回経験、実際、差分、成約リスク、日前、次回応答、競合
- 復活語: なし
- 前日から消えた語: 場面化、安定、小説エンジン、小説生成本体、摩擦、構造、波乱丸、流用、直接依存

## 今日の解釈
### 継続している論点
- Private Reflection must not depend only on already-written Obsidian dialogue notes. If Cloud Run logs are available in `data/cloudrun_chat_log.jsonl`, read them directly and treat missing/late Obsidian conversion as a pipeline lag, not as “no dialogue”.
- Dialogue-room RAG must not treat “no Obsidian note found” as “cannot answer” for deterministic local masters. For recurring statutory useful-life questions, inject local master facts before the LLM answers and keep a regression test around the obvious queries.
- Basic lease concepts should have a deterministic prompt source separate from RAG. RAG can enrich and override with fresher notes, but it must not be the only path for elementary questions.
- Shion's memory growth should be evaluated by time-series deltas, not only by whether recall exists. Comparing 1日前/2日前/3日前 can show whether user concerns, failed assumptions, and next behavior candidates are changing; this is closer to measuring "前回経験が次回応答に効いたか" than normal RAG retrieval.

### 新しく強まった論点
- Q_risk demo data backfill should treat Q_risk as a discovery/search signal, not an automatic score deduction. Store both numeric score and reason/search tags so similar-case retrieval can explain why a case is relevant.
- Qrisk human feedback should separate at least two axes: credit concern vs. sales/competition concern. A low score with good performance may still be a “競合・成約リスク” case rather than a credit-negative case.
- Judgment asset learning model: `事前判断 = 人間の仮説`, `結果登録 = 実際の結果`, `差分 = 判断基準の精度検証`. This lets the system learn not only outcomes, but whether the human concern was well-calibrated.
- Keep old Qrisk and new Qrisk conceptually separate: `financial_consistency_risk` checks whether financial/input numbers make sense; `quantum_risk` is the broader exploration trigger. Do not merge them into one score.
- Dialogue-room RAG must not treat “no Obsidian note found” as “cannot answer” for deterministic local masters. For recurring statutory useful-life questions, inject local master facts before the LLM answers and keep a regression test around the obvious queries.

### ユーザーの圧点
- なし

### 次回の振る舞い候補
- Added README section “ストロングポイント：判断資産を育てるAI” to explain Qrisk, human pre-judgment, result registration, and risk-origin separation as the project's core differentiation.
- Judgment asset learning model: `事前判断 = 人間の仮説`, `結果登録 = 実際の結果`, `差分 = 判断基準の精度検証`. This lets the system learn not only outcomes, but whether the human concern was well-calibrated.
- Shion's memory growth should be evaluated by time-series deltas, not only by whether recall exists. Comparing 1日前/2日前/3日前 can show whether user concerns, failed assumptions, and next behavior candidates are changing; this is closer to measuring "前回経験が次回応答に効いたか" than normal RAG retrieval.

## 記憶レイヤー
### 短期記憶
- 目的: 直近の会話運び。重複質問、露骨な記憶アピール、直前の訂正漏れを避ける。
- 窓: minutes_to_1_day
- 使い方: 次の返答の自然さ、聞き返しの少なさ、言い切りの調整にだけ使う。
- 信号: new_terms=B2B、LLM、Q_risk、Qrisk、事前判断、人間、仮説、判断基準 / dropped_terms=場面化、安定、小説エンジン、小説生成本体、摩擦、構造
- 候補: Added README section “ストロングポイント：判断資産を育てるAI” to explain Qrisk, human pre-judgment, result registration, and risk-origin separation as the project's core differentiation.、Judgment asset learning model: `事前判断 = 人間の仮説`, `結果登録 = 実際の結果`, `差分 = 判断基準の精度検証`. This lets the system learn not only outcomes, but whether the human concern was well-calibrated.、Shion's memory growth should be evaluated by time-series deltas, not only by whether recall exists. Comparing 1日前/2日前/3日前 can show whether user concerns, failed assumptions, and next behavior candidates are changing; this is closer to measuring "前回経験が次回応答に効いたか" than normal RAG retrieval.、Q_risk demo data backfill should treat Q_risk as a discovery/search signal, not an automatic score deduction. Store both numeric score and reason/search tags so similar-case retrieval can explain why a case is relevant.、Qrisk human feedback should separate at least two axes: credit concern vs. sales/competition concern. A low score with good performance may still be a “競合・成約リスク” case rather than a credit-negative case.

### 中期記憶
- 目的: 数日単位の話題継続と圧点を見る。紫苑の次回振る舞い候補を作る。
- 窓: 4_days
- 使い方: 同じ不満・同じ論点が続く時だけ、応答方針を少し変える。
- 信号: repeated_terms=Obsidian、RAG / continued_terms=Obsidian、RAG
- 候補: Private Reflection must not depend only on already-written Obsidian dialogue notes. If Cloud Run logs are available in `data/cloudrun_chat_log.jsonl`, read them directly and treat missing/late Obsidian conversion as a pipeline lag, not as “no dialogue”.、Dialogue-room RAG must not treat “no Obsidian note found” as “cannot answer” for deterministic local masters. For recurring statutory useful-life questions, inject local master facts before the LLM answers and keep a regression test around the obvious queries.、Basic lease concepts should have a deterministic prompt source separate from RAG. RAG can enrich and override with fresher notes, but it must not be the only path for elementary questions.、Shion's memory growth should be evaluated by time-series deltas, not only by whether recall exists. Comparing 1日前/2日前/3日前 can show whether user concerns, failed assumptions, and next behavior candidates are changing; this is closer to measuring "前回経験が次回応答に効いたか" than normal RAG retrieval.

### 長期記憶
- 目的: 繰り返し残った判断基準・価値観・設計原則だけを昇格候補にする。
- 窓: weeks_or_more
- 使い方: 即プロンプト投入せず、レビューして長期記憶・判断基準へ昇格する。
- 信号: durable_terms=なし
- 候補: Q_risk demo data backfill should treat Q_risk as a discovery/search signal, not an automatic score deduction. Store both numeric score and reason/search tags so similar-case retrieval can explain why a case is relevant.、Demo reactions should be treated as judgment-asset candidates through quarantine and human approval, not as immediate model training data. Explicit reactions become reviewable evidence first, then approved items can be promoted into `demo.db` for reuse.、Qrisk human feedback should separate at least two axes: credit concern vs. sales/competition concern. A low score with good performance may still be a “競合・成約リスク” case rather than a credit-negative case.、Private Reflection must not depend only on already-written Obsidian dialogue notes. If Cloud Run logs are available in `data/cloudrun_chat_log.jsonl`, read them directly and treat missing/late Obsidian conversion as a pipeline lag, not as “no dialogue”.、Dialogue-room RAG must not treat “no Obsidian note found” as “cannot answer” for deterministic local masters. For recurring statutory useful-life questions, inject local master facts before the LLM answers and keep a regression test around the obvious queries.

- ランダム化防止: 記憶を同じ棚へ入れない。短期は会話運び、中期は変化検知、長期は判断基準として扱う。

## 日別ミニサマリ
### 2026-07-13
- 主語: Qrisk、Obsidian、RAG、日前、Q_risk、競合、成約リスク、事前判断
- 振る舞い候補: Added README section “ストロングポイント：判断資産を育てるAI” to explain Qrisk, human pre-judgment, result registration, and risk-origin separation as the project's core differentiation.
### 2026-07-12
- 主語: Obsidian、波乱丸、小説エンジン、直接依存、安定、流用、小説生成本体、摩擦
- 振る舞い候補: Added `## 本格内省プロトコル` to `lease_intelligence_reflection.py`, requiring 事前の思い込み / 破られた前提 / 私の責任 / まだ逃げていること / 更新する信念 / 次回の検証方法.
- 圧点: Added `## 本格内省プロトコル` to `lease_intelligence_reflection.py`, requiring 事前の思い込み / 破られた前提 / 私の責任 / まだ逃げていること / 更新する信念 / 次回の検証方法.
### 2026-07-11
- 主語: なし
### 2026-07-10
- 主語: なし
