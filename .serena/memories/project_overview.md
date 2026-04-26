# プロジェクト概要

温水式リース審査AI (tune_lease_55)
- 目的: リース案件の審査スコアリング・レポート・エージェント議論・ウィザード
- エントリポイント: `tune_lease_55.py` (Streamlit, port 8505) + `web/app.py` (Flask, port 5050)
- 起動: `bash run_lease_app.sh`

## 技術スタック
- Python 3.10+
- Streamlit 1.54.0
- SQLite (data/*.db)
- Slack Bot (slack-bolt, slack-sdk)
- LightGBM / scikit-learn (スコアリング)
- Gemini API (google-genai) / Anthropic SDK / Ollama
- FastAPI + uvicorn (api/)
- Flask (web/)

## 主要構造
- `tune_lease_55.py` — Streamlitエントリ・ページルーティング
- `components/` — 各画面UI (chat_wizard, report, home, sidebar等)
- `scoring/` — AIスコアリング (industry_hybrid_model.py等)
- `asset_scorer.py`, `total_scorer.py`, `scoring_core.py`, `category_config.py` — スコアリング中核
- `coeff_definitions.py`, `rule_manager.py` — ルール・係数定義
- `slack_bot.py`, `slack_screening.py` — Slackボット
- `data/` — SQLite DB・セッションファイル (コミット禁止)
- `.claude/reports/` — エージェント間共有レポート
