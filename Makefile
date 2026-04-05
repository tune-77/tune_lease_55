# ─────────────────────────────────────────────────────────────
# Makefile — よく使うコマンドをショートカット化
# 使い方: make <コマンド名>
# ─────────────────────────────────────────────────────────────

.PHONY: test test-v lint iv shap app backup help

## ──────────────────────────────────
## テスト関連
## ──────────────────────────────────

# テスト実行（シンプル）
test:
	python3 -m pytest tests/ -q

# テスト実行（詳細表示）
test-v:
	python3 -m pytest tests/ -v --tb=short --color=yes

# 構文チェック
lint:
	python3 -m pyflakes scoring_core.py rule_manager.py indicators.py credit_limit.py

## ──────────────────────────────────
## 分析スクリプト
## ──────────────────────────────────

# IV（情報価値）分析
iv:
	python3 scripts/iv_analysis.py

# SHAP 判定根拠の可視化（画像出力）
shap:
	python3 scripts/shap_analysis.py

## ──────────────────────────────────
## アプリ起動
## ──────────────────────────────────

# Streamlit アプリを起動
app:
	streamlit run lease_logic_sumaho12.py

## ──────────────────────────────────
## バックアップ
## ──────────────────────────────────

# データを今すぐバックアップ
backup:
	python3 -c "from backup_manager import run_backup; import json; print(json.dumps(run_backup(force=True), ensure_ascii=False, indent=2))"

## ──────────────────────────────────
## ヘルプ
## ──────────────────────────────────

help:
	@echo ""
	@echo "使い方: make <コマンド>"
	@echo ""
	@echo "  make test     テストを実行（シンプル）"
	@echo "  make test-v   テストを実行（詳細）"
	@echo "  make lint     構文チェック"
	@echo "  make iv       IV分析を実行"
	@echo "  make shap     SHAP分析を実行（画像出力）"
	@echo "  make app      Streamlitアプリを起動"
	@echo "  make backup   データを今すぐバックアップ"
	@echo ""
