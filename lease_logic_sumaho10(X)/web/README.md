# リース審査 Web アプリ（Flask）

Streamlit 非依存のモダンな Web UI で、簡易スコア・判定を表示します。

## 必要な環境

- Python 3.10+
- リポジトリルートに `industry_benchmarks.json`、`coeff_definitions.py` があること
- `lease_logic_sumaho10/data_cases.py` が利用可能（係数・重みの読み込み）

## インストール

```bash
# リポジトリルートで
pip install -r lease_logic_sumaho10/web/requirements.txt
# 共通で必要なパッケージ（data_cases 等が使う）
pip install -r requirements.txt
```

## 起動方法

**必ずリポジトリルート（clawd）で実行してください。**

```bash
cd /path/to/clawd
python lease_logic_sumaho10/web/app.py
```

ブラウザで **http://localhost:5050** を開きます。（`PORT=5001` などで変更可能。）

- **403 が出る場合**: **http://127.0.0.1:5050** と **http://localhost:5050** の両方を試してください。

または:

```bash
cd /path/to/clawd
export FLASK_APP=lease_logic_sumaho10.web.app
flask run --host=0.0.0.0 --port=5050
```

### スマホ／他PCから開く（同じ Wi‑Fi）

起動時に表示される **「スマホ／他PCから: http://192.168.x.x:5050」** の URL をブラウザで開いてください。  
まず **http://192.168.x.x:5050/health** で「接続できるか」だけ確認することを推奨します。

### 「このページは動作していません」「ERR_EMPTY_RESPONSE」「データが送信されませんでした」が出る場合

1. **Flask が起動しているか**  
   ターミナルで `python lease_logic_sumaho10/web/app.py` を実行したままにし、終了（Ctrl+C）していないか確認。
2. **接続確認**  
   - PC のブラウザで **http://127.0.0.1:5050/health** を開く。`{"status":"ok"}` が表示されればサーバーは動いています。
   - スマホから開く場合は **http://(PCのIP):5050/health**（例: http://192.168.0.106:5050/health）を試す。
3. **ファイアウォール**  
   Windows／Mac でポート 5050 が許可されているか確認。必要なら「Python」または「ポート 5050」を許可。
4. **同じネットワークか**  
   スマホと PC が同じ Wi‑Fi（同じ 192.168.x.x のネットワーク）に接続されているか確認。

## 機能

- **新規審査**: 売上高・営業利益・総資産・純資産・業種などを入力して判定
- **結果表示**: 総合スコア・承認圏内/要審議・業界比較
- **スコア可視化** (`/visualization`): 過去案件＋ダミーを可視化用JSONで表示。**データビュー**（テーブル）と **Tune Space**（Canvas ビジュアル＋Tone.js）のタブ切り替え。

詳細な相談・討論・学習モデル・定性スコアは Streamlit 版（`streamlit run lease_logic_sumaho10/lease_logic_sumaho10.py`）をご利用ください。
