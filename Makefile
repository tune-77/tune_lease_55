# ─────────────────────────────────────────────────────────────
# Makefile — よく使うコマンドをショートカット化
# 使い方: make <コマンド名>
# ─────────────────────────────────────────────────────────────

.PHONY: test test-v lint iv shap app help migrate-fluid migrate-grade9 feed feed-dry launchd-install launchd-uninstall pipeline-status retrain

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
	python3 -m py_compile tune_lease_55.py lease_logic_sumaho12.py scoring_core.py rule_manager.py indicators.py credit_limit.py ai_chat.py
	@if python3 -c "import pyflakes" >/dev/null 2>&1; then \
		python3 -m pyflakes tune_lease_55.py lease_logic_sumaho12.py scoring_core.py rule_manager.py indicators.py credit_limit.py ai_chat.py; \
	else \
		echo "[lint] pyflakes warnings skipped (legacy unused imports exist)"; \
	fi

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
	streamlit run tune_lease_55.py

## ──────────────────────────────────
## 流体化パイプライン（Fluid Architecture）
## ──────────────────────────────────

# Phase 0: DBマイグレーション（初回のみ）
migrate-fluid:
	python3 migrate_outcomes.py

# 格付9ケースを delinquent=1 として screening_outcomes へ移行（初回のみ）
migrate-grade9:
	python3 migrate_grade9_to_outcomes.py

# メタモデル手動再学習
retrain:
	python3 -c "from retraining_pipeline import run_retraining; import json; print(json.dumps(run_retraining('manual'), ensure_ascii=False, indent=2, default=str))"

# 知識フィード: 今すぐ全タスク実行
feed:
	python3 daily_knowledge_feed.py

# 知識フィード: 確認のみ（DRY RUN）
feed-dry:
	python3 daily_knowledge_feed.py --dry-run

# FluidPipeline の状態確認
pipeline-status:
	python3 -c "from fluid_pipeline import FluidPipeline; import json; print(json.dumps(FluidPipeline().status(), ensure_ascii=False, indent=2, default=str))"

# launchd 登録（毎月1日6時の自動実行を有効化）
launchd-install:
	cp launchd/com.tunelease.daily-knowledge-feed.plist ~/Library/LaunchAgents/
	launchctl load ~/Library/LaunchAgents/com.tunelease.daily-knowledge-feed.plist
	@echo "✅ 毎月1日 6時に daily_knowledge_feed.py が自動実行されます"

# launchd 登録解除
launchd-uninstall:
	launchctl unload ~/Library/LaunchAgents/com.tunelease.daily-knowledge-feed.plist 2>/dev/null || true
	rm -f ~/Library/LaunchAgents/com.tunelease.daily-knowledge-feed.plist
	@echo "✅ launchd 登録解除完了"

## ──────────────────────────────────
## ヘルプ
## ──────────────────────────────────

help:
	@echo ""
	@echo "使い方: make <コマンド>"
	@echo ""
	@echo "  make test           テストを実行（シンプル）"
	@echo "  make test-v         テストを実行（詳細）"
	@echo "  make lint           構文チェック"
	@echo "  make iv             IV分析を実行"
	@echo "  make shap           SHAP分析を実行（画像出力）"
	@echo "  make app            Streamlitアプリを起動"
	@echo ""
	@echo "  [流体化パイプライン]"
	@echo "  make migrate-fluid  DBマイグレーション実行（初回のみ）"
	@echo "  make migrate-grade9 格付9ケースを delinquent=1 として移行（初回のみ）"
	@echo "  make retrain        メタモデルを手動で再学習"
	@echo "  make feed           知識フィードを今すぐ全実行"
	@echo "  make feed-dry       知識フィードの確認のみ（DRY RUN）"
	@echo "  make pipeline-status FluidPipelineの現在状態を表示"
	@echo "  make launchd-install  毎月1日6時の自動実行を有効化"
	@echo "  make launchd-uninstall 自動実行を無効化"
	@echo ""
