# 再現手順・DEBUGログ取得手順（審査入力後スコアリングで Streamlit が反応しない）

注意: 以下は実行手順の定義と収集方法です。実行はユーザーの指示に従ってください。

## 目的
- フォーム送信後に Streamlit が応答しなくなる現象を再現・計測し、ボトルネック（モデルロード / 外部API / DB 読込 / 可視化等）を特定する。

## 準備
1. 開発端末で `LOG_LEVEL=DEBUG` を設定して起動する。

```bash
export LOG_LEVEL=DEBUG
streamlit run tune_lease_55.py
```

2. ターミナルでシステムリソースを監視（再現中）

```bash
# 別ターミナル
top -o %CPU
# or
htop
# メモリ監視
vm_stat 1
```

3. アプリのログ出力先を確認（デフォルト stdout）。/var/log に出す場合は事前に設定。

## 再現手順
1. ストリームリット上で通常通り審査フォームを入力。
2. 「スコアリング実行」ボタンを押す。
3. 押した瞬間から以下を記録する（タイムスタンプ付）:
   - フォーム送信時刻
   - Streamlit によるレスポンス受信（ブラウザで未応答→タイムアウト）
   - ターミナルに出力される Traceback / INFO / DEBUG
   - `top` の CPU/MEM のスナップショット（1 分毎）

## 収集するログ項目（必須）
- Streamlit stdout/stderr
- `app.log` などがあれば併せて
- ブラウザのデベロッパーツールでの Network タブ（リクエスト・待ち時間）
- 可能なら `ps aux --forest` 実行時のプロセス一覧

## 計測ポイント（ログ埋め込み推奨）
- モデルロード開始 / 終了
- モデル予測開始 / 終了
- 外部API 呼び出し開始 / 終了（Gemini, AnythingLLM 等）
- DB 読込開始 / 終了（`load_all_cases()` など）
- 大きな DataFrame 作成後の行数・メモリサイズ

### サンプル計測デコレータ（挿入する場合）
```python
from utils.monitoring import timeit

@timeit
def heavy_predict(...):
    ...
```

## 短期回避（実行はしない）
- モデルの逐次ロードをやめ、起動時に一度だけロードしてキャッシュする（`scoring/predict_one.py` は既にキャッシュあり）
- 外部API のタイムアウトを短めに設定（例: 15 秒）し、リトライは最大 3 回に制限
- 長い処理はチャンクに分割して逐次実行し、UI には進捗を返す

## 重要なコマンド（収集時）
```bash
# Streamlit 起動（DEBUG）
export LOG_LEVEL=DEBUG
streamlit run tune_lease_55.py 2>&1 | tee streamlit_debug.log

# CPU/MEM を1秒ごとに記録
while true; do date +"%Y-%m-%dT%H:%M:%S"; top -b -n1 | head -n20; sleep 1; done > top_snapshots.log &

# ブラウザ側での Network キャプチャは devtools で HAR を保存
```

## 成果物
- `streamlit_debug.log`
- `top_snapshots.log`
- ブラウザの HAR（Network）
- 問題発生時刻のタイムスタンプ

---
作業者はこのファイルを基に実行して報告してください。記録が取れたら、次は収集ログの解析フェーズに進みます。
