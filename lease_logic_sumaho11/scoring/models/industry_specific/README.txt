学習モデル（業種別ハイブリッド）の pkl ファイルをここに配置してください。

必要なファイル:
- industry_coefficients.pkl
- industry_intercepts.pkl
- unified_ai_model.pkl
- scaler.pkl
- label_encoder.pkl

これらは Downloads/lease_scoring_system で
  python main_industry_hybrid.py
を実行すると models/industry_specific/ に出力されます。
その中身をこのフォルダにコピーするか、
環境変数 LEASE_SCORING_MODELS_DIR でそのパスを指定してください。

モデルが無い場合も sumaho10 は動作し、学習モデル欄だけ非表示になります。
