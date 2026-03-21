# 文明操作パネル（Streamlit）の起動方法

## 1. ターミナルでこのフォルダに移動

```bash
cd lease_logic_sumaho10(X)/civilization_alkaia
```

## 2. 起動

```bash
streamlit run app_civilization_panel.py
```

## 3. ブラウザで開く

表示された **Local URL** をブラウザで開いてください。

- 例: `http://localhost:8501` または `http://127.0.0.1:8501`

ブラウザが自動で開かない場合は、上記URLを手動でコピーしてアドレスバーに貼り付けてください。

## うまく開かない場合

- **「接続できません」**: ターミナルで `streamlit run app_civilization_panel.py` が動いているか確認する
- **ポートが使われている**: `streamlit run app_civilization_panel.py --server.port 8502` のように別ポートを指定する
- **白い画面・エラー**: ブラウザの開発者ツール（F12）のコンソールに表示されるエラーを確認する
