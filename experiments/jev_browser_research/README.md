# Jev Ultrafast browser-research pilot

`jev-ultrafast` を紫苑の本番経路へ接続する前に、官公庁などの公開サイトで
操作成功率・時間・費用・誤操作を測るための隔離PoCです。

## 現在の境界

- メインの `pyproject.toml` とCloud Runには追加しない
- 開始URLと遷移先を許可ホストに限定する
- 上流がリクエスト遮断を公開するまでは、`--auto`でもCLICK/SELECT/TYPE_TEXTを実行しない
- ログイン、登録、申請、送信、購入、削除、アップロードをラベルで停止する
- 既定はdry-run。`--auto` を明示した時だけTypeSafeを呼び、操作する
- 操作は最大8回（設定可能範囲1〜12回）
- 個別審査情報・顧客情報は入力しない
- `DONE` は調査完了の証明に使わず、取得URLと内容を別工程で検証する

ラベル拒否は補助ガードであり、サイトの意味を完全に理解するセキュリティ境界では
ありません。初期評価は公開情報の閲覧・検索だけに限定します。

## セットアップ

Python 3.12以上が必要です。依存関係は上流コミット
`1231850a0bf1a0c0341fe408ef1668dbbfdfac46` に固定しています。

```bash
cd experiments/jev_browser_research
UV_CACHE_DIR=/private/tmp/tune-jev-uv-cache \
UV_PROJECT_ENVIRONMENT=/private/tmp/tune-jev-venv \
uv sync --frozen
```

普段のChromeプロファイルと分離した検証用Chromeを起動します。

```bash
open -na "Google Chrome" --args \
  --remote-debugging-port=9222 \
  --user-data-dir=/private/tmp/tune-jev-chrome \
  --no-first-run --no-default-browser-check about:blank
```

## dry-run

APIを呼ばず、ページの観測と操作候補だけを表示します。

```bash
BH_HOME=/private/tmp/tune-jev-bh-home \
UV_PROJECT_ENVIRONMENT=/private/tmp/tune-jev-venv \
uv run python safe_run.py \
  --url https://www.e-stat.go.jp/ \
  --goal "統計データを検索できる画面を確認する"
```

## 自動操作

`TYPESAFE_API_KEY` と `TEXT_MODEL_API_KEY` は環境変数またはmacOS Keychainから
読みます。秘密値をログ、シェル履歴、`.env` に保存しないでください。

OpenRouterを使う場合は、取得したAPIキーを次のコマンドでKeychainへ保存します。
末尾の `-w` により値は対話入力になり、コマンド履歴へ残りません。

```bash
security add-generic-password -U \
  -a "$USER" -s jev-text-model-api-key -w
```

実行時はKeychainのサービス名とOpenAI互換エンドポイントを指定します。

```bash
TYPESAFE_API_KEYCHAIN_SERVICE=typesafe-api-key \
TEXT_MODEL_API_KEYCHAIN_SERVICE=jev-text-model-api-key \
TEXT_MODEL_BASE_URL=https://openrouter.ai/api/v1 \
TEXT_MODEL=inception/mercury-2.5 \
TEXT_MODEL_REASONING=none \
BH_HOME=/private/tmp/tune-jev-bh-home \
UV_PROJECT_ENVIRONMENT=/private/tmp/tune-jev-venv \
uv run python safe_run.py --auto \
  --url https://www.e-stat.go.jp/ \
  --goal "機械受注統計の公開ページへ移動し、資料が見えるところで止まる"
```

## 初回実測（2026-09-20）

- 上流オフラインテスト: `31 passed`
- Ruff: 合格
- 実ChromeのDOM安全ガード: `21 passed`
- e-Statトップページ: 読み込み成功、操作可能要素22件を認識
- TypeSafe実操作: 成功
  - Jevが対象21番「統計データの自動取得（API）」を選択
  - `CLICK` confidence `0.99`、target confidence `1.0`
  - `https://www.e-stat.go.jp/developer` への遷移を確認
  - 遷移先で `DONE` を選択し正常終了
- 文字入力操作: 成功
  - Mercury 2.5が検索文字列「機械受注統計」を生成
  - Jevが検索欄への `TYPE_TEXT` と検索ボタンの `CLICK` を選択
  - `https://www.e-stat.go.jp/stat-search?query=機械受注統計` への遷移を確認
  - 公開HTMLに「機械受注統計調査」が含まれることを独立確認
  - 読み込み待機後、Jevがconfidence `0.91`で `DONE` を選択して正常終了
  - 検索結果画面でデータベース1件・ファイル1件を認識

この初回実測後、JavaScriptイベントによる遷移先の差し替えをhref検査だけでは
遮断できないことが判明したため、現在はCLICK/SELECT/TYPE_TEXTを安全停止します。上記は当時の
実測記録であり、現行ポリシーで同じ自動遷移を許可するものではありません。
