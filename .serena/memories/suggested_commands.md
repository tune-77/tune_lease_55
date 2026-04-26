# よく使うコマンド

## 起動
```bash
bash run_lease_app.sh          # Streamlit(8505) + Flask(5050) 同時起動
streamlit run tune_lease_55.py --server.port 8505  # Streamlitのみ
```

## テスト
```bash
make test        # pytest tests/ -q
make test-v      # pytest tests/ -v --tb=short
python3 -m pytest tests/ -q
```

## Lint
```bash
make lint        # pyflakes scoring_core.py rule_manager.py indicators.py credit_limit.py
```

## 分析
```bash
make iv          # IV分析
make shap        # SHAP可視化
```

## DB
```bash
python3 migrate_to_sqlite.py   # マイグレーション
python3 check_integrity.py     # 整合性チェック
```

## Git
```bash
git log --oneline -10
git diff HEAD
```
