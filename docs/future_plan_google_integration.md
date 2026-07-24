# 今後の展開 — Google エコシステム統合計画

紫苑の次フェーズ：入力自動化・完全クラウド化・スケールアウト

---

## 現状の課題と解決の方向

| 課題 | 解決策 |
|------|--------|
| 審査データの手動入力が多い | 会計ソフト・CRM連携で自動取得 |
| Obsidianがローカル依存 | Google Drive / GCSへ移行 |
| フィードバックがゼロコスト化できない | CRM連携で契約成否を自動取得 |
| 本番データが使えない（コンプライアンス） | オンプレ or VPC構成で対応 |

---

## Phase 2 — 財務データ自動取得

**freee / MFクラウド会計 → CSV/API → 紫苑**

- freee・MFクラウドのエクスポートCSVをパースして審査入力を自動生成
- API連携に移行すればリアルタイム取得も可能
- 取得項目：売上高・粗利・営業利益・経常利益・純利益・純資産・総資産
- **効果：財務入力項目の手動作業がほぼゼロになる**

---

## Phase 3 — CRM連携・フィードバック自動化

**kintone / Salesforce → 紫苑 → judgment_asset_usage_feedback.jsonl**

- kintoneに案件データ・顧客情報が入っている会社なら即連携可能
- 審査後の契約成否をCRMから自動取得 → フィードバックを自動記録
- **効果：ユーザーがボタンを押さなくても紫苑が学習し続ける**

---

## Phase 4 — Google Drive化（Obsidian依存の解消）

**現状：** ローカルの Obsidian Vault → Cloud Run の ChromaDB に同期  
**課題：** ローカルMac依存。スマホだけでは知識を追加できない

**移行先：Google Drive フォルダ**

```
スマホ / PCブラウザ
      ↓ Googleドキュメント / Drive に書く
Google Drive フォルダ（Markdownファイル）
      ↓ Google Drive API（Cloud Run から定期ポーリング）
ChromaDB（Cloud Run）
      ↓
紫苑の知識ベース
```

- 紫苑が生成するReflection・改善ログもDriveに直接書き込み
- **効果：どのデバイスからでも知識を育てられる。Obsidianアプリ不要**

---

## セキュリティ設計（エンタープライズ向け）

Google Driveは個人情報管理には不十分なため、本番環境では以下に切り替える：

```
Google Cloud Storage（GCS）
  + IAM による厳密なアクセス制御
  + Cloud Audit Logs で操作履歴を記録
  + VPC Service Controls でデータ境界を設定
  + CMEK（顧客管理暗号化キー）でデータ暗号化
```

- Cloud Run はすでに GCS にアクセスできる構成が整っている
- ChromaDB → Cloud SQL（PostgreSQL + pgvector）への移行で可用性も向上

---

## マルチテナント化

- 会社ごとに `tenant_id` カラムを追加
- GCSバケットをテナントごとに分離
- 1つの紫苑インスタンスで複数社の審査データを安全に管理

---

## まとめ（プレゼン用1段落）

> 現在はデモデータと手動入力で動作検証を行っている。  
> 次のステップは freee/MFクラウドとの連携で財務データ入力をゼロにし、  
> Google Drive / GCS への移行でObsidianのローカル依存を解消する。  
> これにより、どこからでも紫苑の知識を育て、  
> 実データで自己改善し続けるフルクラウドAIエージェントになる。  
> セキュリティはGCS＋IAM＋VPCで担保し、マルチテナント対応も視野に入れている。

---

_作成日: 2026-07-25_
