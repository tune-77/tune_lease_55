# コーディング規約

- Python 3.10+、型アノテーション推奨
- 数値単位: UI/DB=千円、scoringモジュール=円 (変換注意)
- `st.session_state` 操作は副作用注意
- コメント: WHYが非自明な場合のみ1行
- docstring: 多行ブロック禁止

## 禁止コミット
- `data/*.db`, `data/*.sqlite`, `data/*.jsonl`
- `.streamlit/secrets.toml`
- `scoring_output_bridge.json` (実行時生成)

## 命名
- スネークケース (Python標準)
- session_keys.py でセッションキー定数管理

## セキュリティ
- Slackトークン: 環境変数 or secrets.toml
- SQLインジェクション対策必須 (パラメタライズドクエリ)
