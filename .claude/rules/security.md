# セキュリティ規約

## コミット禁止リスト

| パス | 理由 |
|------|------|
| `.streamlit/secrets.toml` | Slack トークン・API キー |
| `data/*.db`, `data/*.sqlite` | 案件 DB（個人情報含む） |
| `data/*.jsonl` | セッションログ |
| `models/*.pkl`, `models/*.bak.*` | 学習済みモデル |
| `data/` 配下すべて | 機密情報含む |

## コーディング上の注意

- コマンドインジェクション・XSS・SQL インジェクション（OWASP Top 10）に注意
- AI プロンプトに個人情報・機密財務データを混入させない（マスキング必須）
- Slack トークンは環境変数か `secrets.toml` から取得（ハードコード禁止）
- 外部 API キーは `os.environ` 経由で取得
