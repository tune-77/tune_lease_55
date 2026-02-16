# lease_logic_sumaho9

温水式リース審査AIを一つのフォルダにまとめた版です。  
データファイル（past_cases.jsonl 等）は **リポジトリルート** にあり、lease_logic_sumaho8 と共通です。

## 起動方法

リポジトリのルート（clawd）で実行してください。

```bash
cd /path/to/clawd
streamlit run lease_logic_sumaho9/lease_logic_sumaho9.py
```

## フォルダ構成

- **lease_logic_sumaho9.py** … メインアプリ（起動用）
- **config.py** … 設定・定数（将来の分割用）
- **data_holder.py** … データ保持用（将来の分割用）
- **README.md** … この説明

## 注意

- `coeff_definitions.py` はリポジトリルートにあります（sumaho8 と同じ）。
- JSON データ（industry_trends_jsic.json 等）もリポジトリルートを参照します。
