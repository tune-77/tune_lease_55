# Langfuse observability

紫苑の Google ADK エージェントは、任意で Langfuse へ実行トレースを送信できます。
既定は無効です。Langfuse が停止している場合や設定に失敗した場合も、審査処理は継続します。

## 記録するもの

- エージェント・モデル・ツールの span
- 実行順序、所要時間、成否
- モデル名、トークン数
- Langfuse/ADK が生成する匿名のトレース識別子
- `shion-screening` という安定したトレース名
- 会話をまとめるセッションID、機能・経路・警告有無などの低カーディナリティメタデータ

## 記録しないもの

プライバシーポリシーは `api/langfuse_observability.py` で固定しています。次の内容は
OpenInference の `TraceConfig` により常に秘匿されます。

- ユーザー入力とモデル回答
- 企業名、財務数値、案件情報
- ツール引数・ツール結果の本文
- 検索文書、埋め込みテキスト・ベクトル
- LLM invocation parameters

## 有効化

1. Langfuse Cloud またはセルフホスト環境でプロジェクトを作成する。
2. `.env.example` を参照し、実行環境へ次を設定する（秘密鍵はGitへ保存しない）。

```dotenv
SHION_LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://jp.cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=development
```

3. APIを再起動する。初回の紫苑ADK実行後、Langfuseの `Traces` を確認する。

認証情報が欠けている場合、初期化は警告を残して停止します。起動時に外部への
認証確認リクエストは行いません。

## 無効化

`SHION_LANGFUSE_ENABLED=false` に戻してAPIを再起動します。依存パッケージが存在しても、
このフラグが有効でなければLangfuse SDKは読み込まれません。
