# lease_logic_sumaho8.py 改善メモ

約6,000行のStreamlitアプリ。動作は十分だが、保守性・安全性・運用面で以下の改善余地あり。

---

## 1. 構成・保守性

### ファイル分割（優先: 高）
- **現状**: 1ファイルにUI・スコアリング・AI呼び出し・データ読込・グラフ・設定が混在。
- **提案**: モジュール分割で責務を分ける。
  - `lease_ui.py` … ページ設定・タブ・サイドバー・レイアウト
  - `lease_scoring.py` … 係数・スコア計算・回帰分析
  - `lease_ai.py` … Ollama/Gemini 呼び出し・チャット・相談
  - `lease_data.py` … JSON/JSONL 読込・保存・キャッシュ
  - `lease_plots.py` … ゲージ・ウォーターフォール・レーダー等
  - `config.py` … 定数・パス・CHART_STYLE
- **効果**: 変更の影響範囲が分かりやすく、テスト・リファクタがしやすい。

### 型ヒントの統一
- 主要関数にはすでに型ヒントあり。引数・戻り値をファイル全体で統一すると、IDE補完とリファクタが楽になる。

---

## 2. セキュリティ・設定

### APIキーの扱い
- **現状**: Gemini APIキーを `st.session_state["gemini_api_key"]` に平文で保持。サイドバー入力もそのまま反映。
- **提案**:
  - 本番・共有環境では `st.secrets["GEMINI_API_KEY"]` を優先し、サイドバー入力は「上書き」用途に限定。
  - ログやエラーメッセージにキーが含まれないよう注意（現状はメッセージには出していそうだが、デバッグ出力で漏れないように）。

### ハードコードされたパス
- **現状**: `_dashboard_image_base_dirs()` 内に絶対パスが直書き（415–416行付近）。
  ```python
  "/Users/kobayashiisaoryou/.cursor/projects/Users-kobayashiisaoryou-clawd/assets"
  ```
- **提案**: 環境変数（例: `DASHBOARD_IMAGES_ASSETS`）か、`BASE_DIR` からの相対パスのみにし、他環境でも動くようにする。

---

## 3. 例外・エラー処理

### 裸の `except:`
- **場所**: 908行付近（`load_all_cases` 内の JSONL 1行パース）。
  ```python
  except:
      continue
  ```
- **問題**: `KeyboardInterrupt` や `SystemExit` まで握りつぶす可能性。
- **提案**: `except json.JSONDecodeError:` に限定する。

### 広い `except Exception`
- 多数の `except Exception` で握りつぶしている箇所あり。重要な箇所では:
  - キャッチする例外を「予想されるもの」に限定する。
  - `logging.exception()` や `st.error()` で原因を残す。
- 例: ファイル読込失敗時は `FileNotFoundError` / `PermissionError` を個別に扱うと原因切り分けがしやすい。

---

## 4. 状態管理・スレッド

### 相談モードの結果受け渡し
- **現状**: グローバル `_chat_result_holder = {"result": None, "done": False}` でスレッド→メインに結果を渡している（48行付近）。
- **リスク**: 複数ユーザー（複数セッション）では本来 `session_state` で分離されるが、同一プロセス内でグローバル共有になる。
- **提案**: 可能なら `session_state` にキーを付けて「このセッション用」の結果ホルダーを置く。または「相談は1セッション1リクエスト」と割り切り、ドキュメントに明記する。

---

## 5. UI・UX

### `red_label` とCSS
- **現状**: `red_label()` 関数の直後に、関数の「中」のように見えるコメントが1行ある（76行付近）。実際の大きな `st.markdown`（CSS）はモジュールレベルで実行されている。
- **提案**: コメントを「このCSSはページ全体用」と分かるようにするか、`red_label` の直後からは「ページ共通CSS」とコメントで区切る。関数の責務（赤ラベル表示のみ）が読み取りやすくなる。

### キャッシュクリア
- 設定タブの「キャッシュをクリア」で `st.cache_data.clear()` のあと `st.rerun()` している。連打防止のため、ボタンに `disabled` を短時間付与するか、`st.fragment` で局所更新するのもあり。

---

## 6. パフォーマンス・データ

### キャッシュ方針
- `load_json_data` は `@st.cache_data(ttl=3600)` で適切。
- `load_all_cases()` はキャッシュなしで毎回 JSONL を読んでいる。案件数が非常に多い場合は、読み込み頻度や `@st.cache_data(ttl=60)` などの短いTTLを検討してもよい（「登録したらすぐ一覧に反映」を優先するなら現状のままが無難）。

### 外部API・スクレイピング
- 業界トレンド・ベンチマーク取得で Web アクセスしている箇所あり。タイムアウト・リトライ・403対策がすでに入っていればそのまま。未設定なら `requests` に `timeout=` とリトライを入れると安定する。

---

## 7. ngrok 運用

- 前回は `--server.address 127.0.0.1` で起動。ngrok がローカルの 8501 に転送するため、このままでアクセスは可能。
- 同一LAN内の別端末から「PCのIP:8501」で直接開く運用にする場合は、`--server.address 0.0.0.0` に変更する必要あり。
- 本番公開時は認証（Streamlit の `server.enableCORS` / リバースプロキシ側の認証など）を検討する。

---

## 8. すぐできる小さな修正例

| 項目 | 場所 | 修正例 |
|------|------|--------|
| 裸の `except` | 908行付近 | `except json.JSONDecodeError:` に変更 |
| 画像パス | 414–416行 | 環境変数または `BASE_DIR` 相対に変更 |
| コメント | 76行付近 | 「以下はページ共通CSS」などと明記 |

上から順に手を入れていくと、リスクを抑えつつ保守性が上がる。
