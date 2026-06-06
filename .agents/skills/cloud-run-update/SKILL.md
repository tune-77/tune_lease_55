---
name: cloud-run-update
description: tune_lease_55 の Cloud Run を更新・再デプロイする定型スキル。「Cloud Run更新」「再デプロイ」「Cloud Run deploy」「bundle作成して更新」などで使う。bundle→deploy→疎通確認の3行手順を固定する。
---

# Cloud Run Update

`tune_lease_55` を Cloud Run に反映するときは、毎回この順で進める。

## 標準手順

```bash
ALLOW_UNAUTHENTICATED=1 ./scripts/deploy_cloud_run.sh
URL="$(gcloud run services describe tune-lease-55 --region asia-northeast1 --format='value(status.url)')"
curl --max-time 20 -sS "$URL/api/score/full" -X POST -H 'Content-Type: application/json' -d '{"company_name":"テスト株式会社","industry":"サービス業","sales":1000,"operating_profit":100,"ordinary_profit":100,"total_assets":2000,"equity":500,"current_assets":800,"current_liabilities":400,"long_term_debt":300,"cash":200,"trade_receivables":100,"inventories":50,"fixed_assets":1200,"short_term_borrowings":100,"long_term_borrowings":200,"interest_expense":10,"depreciation":20,"num_employees":50,"established_year":2010,"lease_term_months":36,"lease_amount":100,"existing_relationship":"既存","payment_delay_history":"なし","bankruptcy_history":"なし","deal_source_bank":"本店","risk_notes":"","customer_new":false}'
```

## 使い方

- 1行目で bundle 作成と Cloud Run 反映をまとめて行う
- 2行目で現在のサービス URL を取る
- 3行目で FastAPI の `/api/score/full` が `200` で返るか確認する

## 判定

- `Ready=True` かつ 3 行目が `200` なら完了
- `unable to open database file` や `no such table` が出たら、データ同梱か参照パスを疑う
- `GEMINI_API_KEY` エラーが出たら Secret Manager 側を確認する
