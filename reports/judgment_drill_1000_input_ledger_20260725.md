# 1000本判断ドリル 入力台帳

- Date: 2026-07-25
- CSV: `data/judgment_drills/judgment_drill_1000_20260725.csv`
- Input UI: `/judgment-drill`
- Rows: 1000
- Source: synthetic demo cases
- Status: pending_user_judgment
- Guardrail: User採点前は実戦検証・成果指標・RAG強化に使わない。採点後は human_judgment_asset として扱い、実案件結果検証とは分ける。

## 使い方

通常は `/judgment-drill` の入力画面から1件ずつ保存する。CSVを直接編集する場合は、次の空欄を埋める。

- `credit_score_20`: 信用力 0-20
- `repayment_source_score_20`: 返済原資 0-20
- `asset_exit_score_20`: 物件性・出口 0-20
- `plan_specificity_score_20`: 計画具体性 0-20
- `uncertainty_control_score_20`: 未確定リスク管理 0-20
- `total_score_100`: 合計 0-100
- `user_decision`: 承認 / 条件付き承認 / 追加確認 / 否認
- `heaviest_issue`: 一番重い論点
- `additional_checks`: 追加確認事項。複数ある場合は `;` 区切り
- `ringi_sentence`: 稟議に残す一文
- `score_decision_gap_note`: 点数と結論がズレた理由
- `ai_feedback_outcome`: helped / neutral / challenged / rejected
- `ai_feedback_note`: AI案へのダメ出し、または採用理由

銀行・既存取引の見方:

- `bank_is_main`: 銀行がメイン / 非メイン
- `bank_credit_balance_million_yen`: その銀行の与信残高 百万円
- `lease_customer_status`: リース既存先 / 新規先
- `existing_lease_credit_balance_million_yen`: 既存先の場合のリース与信残高 百万円。新規先は0
- `equity_ratio_percent`: 自己資本比率 %

## 判定目安

- 80点以上: 承認候補
- 65-79点: 条件付き承認候補
- 50-64点: 追加確認・保留候補
- 49点以下: 否認候補

点数は適当でよい。重要なのは、点数よりも「なぜその点にしたか」「どの条件なら通すか」「稟議にどう残すか」。

## 先頭ケース例

- case_id: `CASE-JD-20260725-0001`
- 業種: 製造業
- 銀行: メイン / 与信残高 0百万円
- リース取引: 既存先 / 既存リース与信残高 5百万円
- 自己資本比率: 8%
- 物件: CNC工作機械
- 申込: 12百万円 / 4年
- 目的: 老朽設備更新と増産対応
- 財務: 直近期売上90百万円、営業利益-800万円、借入金2500万円、自己資本比率8%、ステージ=安定黒字。
- 良い材料: 既存主力先からの受注残があり、設備必要性は明確。
- 懸念: 補助金採択前にリース開始予定で、未採択時の資金繰り説明が弱い。

## プレゼンで出す時の言い方

この台帳は実案件結果を当てたデータではなく、開発者本人が1000本の模擬案件に対して採点・条件・稟議文面・AI案へのダメ出しを入れた human_judgment_asset である。したがって、アウトカム検証ではなく、現場判断の形式知化・AI共同作業による判断資産化の証跡として提示する。
