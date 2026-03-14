# ローカル環境セットアップ手順

## 前提条件

| 項目 | バージョン |
|------|-----------|
| Python | 3.11 以上推奨 |
| pip | 最新版（`pip install --upgrade pip`） |
| Git | 任意のバージョン |

---

## 1. リポジトリを取得

```bash
git clone <リポジトリURL>
cd tune_lease_55

# 今回の実装ブランチに切り替える
git checkout claude/mobile-input-sales-agent-TJw04
```

---

## 2. 仮想環境を作成して有効化

```bash
# 仮想環境を作成
python3 -m venv .venv

# 有効化 (Mac/Linux)
source .venv/bin/activate

# 有効化 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

---

## 3. パッケージをインストール

**Streamlit 本審査アプリ用:**
```bash
pip install -r requirements.txt
```

**Flask 簡易審査アプリ用:**
```bash
pip install -r lease_logic_sumaho11/web/requirements.txt
```

まとめて1行でやる場合:
```bash
pip install -r requirements.txt -r lease_logic_sumaho11/web/requirements.txt
```

---

## 4. アプリを起動する

### 方法A: スクリプトで両方同時起動（推奨）

```bash
# スクリプトに実行権限を付与（初回のみ）
chmod +x run_lease_app.sh

# 起動
./run_lease_app.sh
```

| アプリ | URL |
|--------|-----|
| 本審査（Streamlit） | http://localhost:8505 |
| 簡易審査（Flask） | http://localhost:5050 |

`Ctrl+C` で両方停止します。

---

### 方法B: 個別に起動する

**Flask 簡易審査アプリ（今回の実装が含まれる）:**
```bash
cd lease_logic_sumaho11/web
python app.py
# → http://localhost:5050 で起動
```

**Streamlit 本審査アプリ:**
```bash
streamlit run lease_logic_sumaho11/lease_logic_sumaho11.py --server.port 8501
# → http://localhost:8501 で起動
```

---

## 5. 今回実装した機能の確認ポイント

Flask 簡易審査アプリ（`http://localhost:5050`）で以下を確認してください。

| # | 機能 | 確認方法 |
|---|------|---------|
| ① | **万円スイッチ** | 右上の「千円 / 万円」トグルを切り替え → 入力値が自動変換されることを確認 |
| ② | **ドラフト自動保存** | 数値を入力 → ページをリロード → 「✅ 前回の入力内容を復元しました」バナーが出ることを確認 |
| ③ | **ステッパーUI** | 各フィールドの「－」「＋」ボタンをタップ・長押し → 値が増減・加速することを確認 |
| ④ | **アコーディオン** | 「業種・与信」セクションのヘッダをクリック → 展開/折り畳みの切り替えを確認 |
| ⑦ | **スマホ1カラム化** | ブラウザの幅を640px未満に縮小 → 1カラムレイアウト＋画面下固定の送信ボタンを確認 |

---

## 6. スマホ実機で確認する場合

PCとスマホを同じWi-Fiに接続した状態で、PCのローカルIPアドレスを使います。

```bash
# PCのIPアドレスを確認
# Mac/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```

スマホのブラウザで `http://<PCのIPアドレス>:5050` にアクセスしてください。

> **注意:** Flask は `0.0.0.0` でバインドしているので、ファイアウォールを許可すれば同一ネットワークから接続できます。

---

## 7. よくあるエラーと対処

| エラー | 原因 | 対処 |
|--------|------|------|
| `ModuleNotFoundError: flask` | パッケージ未インストール | `pip install -r lease_logic_sumaho11/web/requirements.txt` |
| `ModuleNotFoundError: streamlit` | パッケージ未インストール | `pip install -r requirements.txt` |
| `Address already in use` ポート5050 | 前回のプロセスが残っている | `lsof -ti:5050 \| xargs kill -9` |
| `Address already in use` ポート8501 | 前回のStreamlitが残っている | `lsof -ti:8501 \| xargs kill -9` |
| 403 Forbidden | HostヘッダがFlaskの許可リスト外 | `http://localhost:5050` または `http://127.0.0.1:5050` でアクセス |

---

## 8. ファイル構成（主要ファイル）

```
tune_lease_55/
├── requirements.txt                     # Streamlit 用パッケージ
├── run_lease_app.sh                     # 一括起動スクリプト
├── lease_logic_sumaho11/
│   ├── lease_logic_sumaho11.py          # Streamlit 本審査アプリ（メイン）
│   ├── scoring_core.py                  # スコア計算ロジック（共通）
│   └── web/
│       ├── app.py                       # Flask サーバー
│       ├── requirements.txt             # Flask 用パッケージ
│       └── templates/
│           └── index.html              # ★ 今回実装した入力フォーム
```
