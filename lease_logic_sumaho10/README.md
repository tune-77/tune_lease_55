# lease_logic_sumaho10

温水式リース審査AI **sumaho9 とリース与信スコアリング（学習モデル）を融合した版**です。  
データファイル（past_cases.jsonl 等）は **リポジトリルート** にあり、sumaho8/9 と共通です。

## 起動方法

リポジトリのルート（clawd）で実行してください。

```bash
cd /path/to/clawd
streamlit run lease_logic_sumaho10/lease_logic_sumaho10.py
```

## 学習モデル（業種別ハイブリッド）について

- 判定開始時に、**総資産・純資産**が入力されていれば、学習モデルによる「既存確率・AI確率・ハイブリッド確率」と判定理由 Top5 を分析結果タブに表示します。
- モデルファイルは次のいずれかへ配置してください。
  - `lease_logic_sumaho10/scoring/models/industry_specific/` に `industry_coefficients.pkl`, `industry_intercepts.pkl`, `unified_ai_model.pkl`, `scaler.pkl`, `label_encoder.pkl` を置く
  - または環境変数 `LEASE_SCORING_MODELS_DIR` で上記 pkl が入ったディレクトリを指定
- モデルが無い場合は学習モデル欄は表示されず、本システムの係数スコアのみで動作します。

## フォルダ構成

- **lease_logic_sumaho10.py** … メインアプリ（起動用）
- **config.py** … 設定・定数（SCORING_MODELS_DIR 含む）
- **data_holder.py** … データ保持用
- **scoring/** … 学習モデル用（feature_engineering, industry_hybrid_model, model, predict_one）
- **README.md** … この説明

## 判定まわり

- **承認ライン**: 総合スコアが **71 以上**で「承認圏内」。根拠・変更履歴は `REVIEW_EVALUATION.md` 参照。
- **重み**: 借手/物件はデフォルト 85%/15%、総合/定性（ランク用）は 60%/40%。**定量要因分析**タブの「回帰で重みを最適化」で推奨値を算出し、保存すると今後の審査スコアに反映される。
- 学習モデルが「否決」のときは全スコアを 0.5 倍（定数 `SCORE_PENALTY_IF_LEARNING_REJECT`）。

## 注意

- `coeff_definitions.py` はリポジトリルートにあります（sumaho8/9 と同じ）。
- JSON データ（industry_trends_jsic.json 等）もリポジトリルートを参照します。
