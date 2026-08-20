"""紫苑 ADK エージェントに登録する読み取り専用ツール群。

ADK（google.adk）に依存させず import できるよう、ツール関数の選定だけをここに集約する。
これにより CI（google.adk 未導入）でもエージェントのツール構成を検証できる。

ここに載せるのは **ローカル DB / ローカルファイル読み取りのみ** のツール。
外部 API 課金（埋め込み等）が発生するツールは含めない。
"""

from __future__ import annotations

from lease_intelligence_tools import (
    audit_ledger_consistency,
    build_judgment_preview,
    get_pipeline_item_details,
    get_portfolio_stats,
    get_recent_errors,
    get_score_detail,
    get_system_overview,
    get_weekly_trend,
    recall_judgment_memory,
    score_full_case,
    search_cases,
    search_obsidian_context,
)

# ローカル SQLite / JSON / ログファイル読み取りのみ。外部 API を叩かない（追加課金ゼロ）。
READ_ONLY_DB_TOOLS = [
    search_cases,              # 類似・過去案件の検索
    get_score_detail,          # 企業名からスコア内訳を取得（過去の採点結果）
    score_full_case,           # 目の前の案件を新規に採点（DB保存なしの試算）
    get_portfolio_stats,       # 審査DB全体の統計（成約率・分布・業種構成）
    get_weekly_trend,          # 週次トレンド
    get_system_overview,       # モデル・閾値・データ規模のスナップショット
    get_recent_errors,         # logs/api.log・app.log の頻出エラー調査
    get_pipeline_item_details, # 改善パイプライン台帳（ledger_rules.json）の個別項目詳細
    recall_judgment_memory,    # 正準ルール＋紫苑の記憶索引から判断根拠を想起
    build_judgment_preview,    # 判断材料プレビュー（レビュー前の下書き）を取得
    search_obsidian_context,   # Obsidian Vaultの知識ノート検索
    audit_ledger_consistency,  # REV改善台帳のREV番号・canonical_key・status整合性監査
]
