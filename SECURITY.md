# セキュリティ運用ガイド

リース審査AIシステムの本番・開発環境でのセキュリティベストプラクティス。

---

## 📋 目次

1. [APIキー管理](#apiキー管理)
2. [secrets.toml 標準化](#secretstoml-標準化)
3. [環境別設定](#環境別設定)
4. [ログ出力時の注意](#ログ出力時の注意)
5. [デプロイメント](#デプロイメント)
6. [トラブルシューティング](#トラブルシューティング)

---

## APIキー管理

### 対象キー

| キー | 用途 | 優先度 | 参照先 |
|------|------|--------|--------|
| `GEMINI_API_KEY` | Google Gemini LLM | **必須** | AI チャット、軍師コメント生成 |
| `SLACK_BOT_TOKEN` | Slack ボット連携 | オプション | Slack ワークフロー、通知 |
| `ANYTHING_LLM_API_KEY` | AnythingLLM（ローカル RAG） | オプション | 知識ベース検索 |

### 優先順位

**環境変数 > secrets.toml > ローカルデフォルト**

```python
# ✅ 正しい優先順位の実装
api_key = (
    os.environ.get("GEMINI_API_KEY", "").strip()      # 1. 環境変数（最優先）
    or st.secrets.get("GEMINI_API_KEY", "").strip()   # 2. secrets.toml
    or _get_gemini_key_from_secrets()                  # 3. 他の設定
)
```

---

## secrets.toml 標準化

### ファイル位置

```
.streamlit/secrets.toml          # Streamlit 用（開発環境）
/etc/lease-system/secrets.env    # 本番環境（Linux）
/var/lib/lease-system/.env       # Docker・k8s 環境
```

### テンプレート

```toml
# .streamlit/secrets.toml
# ⚠️ 注意: このファイルは絶対にコミットしないこと（.gitignore に登録済み）

[gemini]
api_key = "YOUR_GEMINI_API_KEY_HERE"
model = "gemini-1.5-pro"

[slack]
bot_token = "xoxb-YOUR-BOT-TOKEN"
signing_secret = "YOUR-SIGNING-SECRET"

[anything_llm]
api_key = "YOUR_ANYTHING_LLM_KEY"
base_url = "http://127.0.0.1:3001/api/v1"
workspace = "lease"

[database]
sqlite_path = "./data/lease_data.db"
backup_path = "./data/backups/"

[environment]
is_production = false              # ← 必ず本番では true に
debug_mode = false                 # ← ログレベルを制御
```

### 各環境での設定方法

#### 開発環境（ローカル）

```bash
# 1. .streamlit/secrets.toml を作成
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
[gemini]
api_key = "your-dev-key"
EOF

# 2. 権限を制限（-rw-------)
chmod 600 .streamlit/secrets.toml

# 3. 起動
streamlit run tune_lease_55.py
```

#### 本番環境（環境変数）

```bash
# 1. 環境変数を設定（secrets 管理ツール推奨）
export GEMINI_API_KEY="prod-key-from-vault"
export SLACK_BOT_TOKEN="xoxb-prod-token"

# 2. secrets.toml は配置しない（環境変数が優先）
# 3. 起動
streamlit run tune_lease_55.py
```

#### Docker / Kubernetes

```dockerfile
# Dockerfile
FROM python:3.10

# 環境変数を ARG で受け取る（ビルド時に渡す）
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# secrets.toml は COPY しない
CMD ["streamlit", "run", "tune_lease_55.py"]
```

```yaml
# kubernetes/deployment.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lease-system-secrets
type: Opaque
data:
  GEMINI_API_KEY: <base64-encoded-key>
  SLACK_BOT_TOKEN: <base64-encoded-token>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lease-system
spec:
  containers:
  - name: streamlit
    image: lease-system:latest
    env:
    - name: GEMINI_API_KEY
      valueFrom:
        secretKeyRef:
          name: lease-system-secrets
          key: GEMINI_API_KEY
```

---

## 環境別設定

### 開発環境チェックリスト

- [ ] `.streamlit/secrets.toml` 作成済み
- [ ] `.gitignore` に `.streamlit/secrets.toml` が登録済み
- [ ] ローカルテスト時は環境変数で override 可能か確認
- [ ] ログに APIキー混入なし（以下のコマンドで検証）

```bash
grep -r "GEMINI_API_KEY\|Bearer" logs/
# 出力なし = OK
```

### 本番環境チェックリスト

- [ ] 環境変数が secrets 管理ツール（Vault / AWS Secrets Manager）から供給される
- [ ] `.streamlit/secrets.toml` が本番サーバーに存在しない
- [ ] CI/CD ログに APIキーが出力されていない（以下のコマンドで検証）

```bash
# GitHub Actions ログから機密情報をマスク
git log --all -S "GEMINI_API_KEY" --oneline
# 出力なし = 安全
```

---

## ログ出力時の注意

### ❌ 危険なパターン

```python
# API キーがログに出力される可能性
logger.info(f"Using API key: {api_key}")
logger.debug(f"Request headers: {headers}")  # Authorization が含まれる
logger.error(f"Failed with response: {resp.text}")  # 応答に秘密情報が？
```

### ✅ 安全なパターン

```python
# キーをマスク
logger.info(f"Using API key: {api_key[:20]}...")

# ヘッダーをフィルタ
safe_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
logger.debug(f"Request headers: {safe_headers}")

# 応答を適度に切る（PII 除去）
logger.error(f"HTTP {resp.status_code}: {resp.text[:100]}")
```

### 実装例（Logging Filter）

```python
import logging

class SensitiveDataFilter(logging.Filter):
    """ログから機密情報（APIキー等）をマスク"""
    SENSITIVE_KEYS = ["GEMINI_API_KEY", "SLACK_BOT_TOKEN", "Authorization", "Bearer"]

    def filter(self, record):
        msg = record.getMessage()
        for key in self.SENSITIVE_KEYS:
            msg = msg.replace(key, f"{key}=***")
        record.msg = msg
        return True

# ハンドラに フィルタを追加
logger = logging.getLogger("lease_system")
for handler in logger.handlers:
    handler.addFilter(SensitiveDataFilter())
```

---

## デプロイメント

### Pre-Deploy チェック

```bash
# 1. secrets.toml が git に追加されていないか確認
git status | grep secrets.toml
# 出力なし = OK

# 2. APIキーが コミットメッセージ・コード内に混入していないか確認
git log --all -p | grep -i "GEMINI_API_KEY\|xoxb-"
# 出力なし = OK

# 3. ログに APIキーが残っていないか確認
grep -r "api_key=\|Bearer " . --include="*.log" --include="*.txt"
# 出力なし = OK
```

### デプロイ後チェック

```bash
# 1. 本番環境で環境変数が設定されているか確認
echo $GEMINI_API_KEY | head -c 20
# 出力: 最初の20文字が見える（キー全体は見えない）

# 2. secrets.toml が本番に配置されていないか確認
ls -la /app/.streamlit/secrets.toml
# エラー: No such file = OK

# 3. ログファイルにキーが混入していないか確認
tail -f /var/log/lease-system/app.log | grep -i "GEMINI_API_KEY"
# 出力なし = OK
```

---

## トラブルシューティング

### APIキーが認識されない

```
❌ エラー: GEMINI_API_KEY not found
```

**原因と対処**

| 原因 | 対処法 |
|------|--------|
| secrets.toml が作成されていない | `.streamlit/secrets.toml` を作成（上記のテンプレート参照） |
| 環境変数が設定されていない | `export GEMINI_API_KEY="..."` で設定 |
| secrets.toml のフォーマットが不正 | `toml-cli validate .streamlit/secrets.toml` で検証 |
| キーの前後に空白がある | `.strip()` で削除（コードは既に対応） |

### ログに APIキーが混入

```
❌ ログ: Using API key: sk-xxx...xxx
```

**対処法**

1. 上記の「ログ出力時の注意」セクション参照
2. `SensitiveDataFilter` を logger に追加
3. ログレベルを `INFO` 以上に変更（`DEBUG` で詳細出力）

### Streamlit secrets が反映されない

```bash
# 1. Streamlit キャッシュをクリア
streamlit cache clear

# 2. .streamlit/secrets.toml をリロード
# → Streamlit を再起動
pkill -f streamlit
streamlit run tune_lease_55.py

# 3. 権限確認
ls -la .streamlit/secrets.toml
# -rw------- (600) であることを確認
```

---

## 参考資料

- [Streamlit Secrets Management](https://docs.streamlit.io/deploy/streamlit-cloud/manage-apps/secrets-management)
- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [Google Cloud: API キー管理](https://cloud.google.com/docs/authentication/api-keys)

---

**最終更新**: 2026-05-10
**維持者**: システム管理チーム
