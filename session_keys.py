"""
session_keys.py
==============
st.session_state で使用する全キー文字列を一元管理する定数モジュール。

使い方:
    from session_keys import SK
    st.session_state[SK.LAST_RESULT] = result
    val = st.session_state.get(SK.LAST_RESULT)
"""

from __future__ import annotations


class SK:
    """Session-state key 定数クラス（属性アクセスで取得）。"""

    # ─────────────────────────────────────────────
    # ナビゲーション / UI
    # ─────────────────────────────────────────────
    NAV_MODE_WIDGET       = "nav_mode_widget"      # サイドバー ラジオウィジェットキー
    NAV_INDEX             = "nav_index"            # タブインデックス保持
    NAV_PENDING           = "_nav_pending"         # ページ遷移保留フラグ
    JUMP_TO_ANALYSIS      = "_jump_to_analysis"    # 分析結果へジャンプフラグ
    SHOW_BATTLE           = "show_battle"          # バトル表示フラグ
    SELECTED_ASSET_INDEX  = "selected_asset_index" # 選択資産インデックス
    CURRENT_IMAGE         = "current_image"        # 表示中画像

    # ─────────────────────────────────────────────
    # フォーム入力値（財務）
    # ─────────────────────────────────────────────
    NENSHUU               = "nenshuu"              # 年収
    NUM_NENSHUU           = "num_nenshuu"          # 年収人数
    NET_ASSETS            = "net_assets"           # 純資産
    TOTAL_ASSETS          = "total_assets"         # 総資産
    RIEKI                 = "rieki"                # 利益
    GRADE                 = "grade"                # 格付け
    BANK_CREDIT           = "bank_credit"          # 銀行与信
    LEASE_CREDIT          = "lease_credit"         # リース与信
    LEASE_TERM            = "lease_term"           # リース期間
    ACQUISITION_COST      = "acquisition_cost"     # 取得費用
    CONTRACT_TYPE         = "contract_type"        # 契約タイプ
    COMPETITOR_RATE       = "competitor_rate"      # 競合レート
    ACCEPTANCE_YEAR       = "acceptance_year"      # 承認年
    CUSTOMER_TYPE         = "customer_type"        # 顧客タイプ
    DEAL_SOURCE           = "deal_source"          # 案件ソース

    # ─────────────────────────────────────────────
    # フォーム入力値（財務指標 item4〜12）
    # ─────────────────────────────────────────────
    ITEM4_ORD_PROFIT      = "item4_ord_profit"     # 経常利益
    ITEM5_NET_INCOME      = "item5_net_income"     # 当期純利益
    ITEM6_MACHINE         = "item6_machine"        # 機械設備
    ITEM7_OTHER           = "item7_other"          # その他
    ITEM8_RENT            = "item8_rent"           # 賃借料
    ITEM9_GROSS           = "item9_gross"          # 売上総利益
    ITEM10_DEP            = "item10_dep"           # 減価償却
    ITEM11_DEP_EXP        = "item11_dep_exp"       # 減価償却費
    ITEM12_RENT_EXP       = "item12_rent_exp"      # 賃借費用

    # ─────────────────────────────────────────────
    # 審査結果
    # ─────────────────────────────────────────────
    LAST_RESULT           = "last_result"          # 直前の審査結果
    LAST_SUBMITTED_INPUTS = "last_submitted_inputs"# 直前の送信入力値
    CURRENT_CASE_ID       = "current_case_id"      # 現在の案件ID
    CONTRACTS             = "contracts"            # 契約一覧
    FORM_RESTORED         = "form_restored_from_submit" # フォーム復元済みフラグ

    # ─────────────────────────────────────────────
    # 定量分析
    # ─────────────────────────────────────────────
    QUANT_BY_INDICATOR        = "quant_by_indicator"
    QUANT_BY_INDUSTRY         = "quant_by_industry"
    QUANTITATIVE_RESULT       = "quantitative_analysis_result"
    QUALITATIVE_RESULT        = "qualitative_analysis_result"
    INDICATOR_AI_ANALYSIS     = "indicator_ai_analysis"
    INDICATOR_AI_ANALYSIS_CID = "indicator_ai_analysis_case_id"
    REGRESSION_COEFFS         = "regression_coeffs"
    REGRESSION_ACCURACY       = "regression_accuracy"
    WEIGHT_OPTIMIZE_RESULT    = "weight_optimize_result"

    # ─────────────────────────────────────────────
    # ベイジアンネットワーク (BN)
    # ─────────────────────────────────────────────
    BN_S_RESULT           = "_bn_s_result"         # BN推論結果
    BN_S_EVIDENCE         = "_bn_s_evidence"       # BN入力エビデンス
    AUTO_JUDGE            = "_auto_judge"           # 自動判定フラグ

    # ─────────────────────────────────────────────
    # モンテカルロ
    # ─────────────────────────────────────────────
    MC_PORTFOLIO_RESULT   = "mc_portfolio_result"  # ポートフォリオ分析結果
    MC_COMPANIES          = "mc_companies"         # 対象企業リスト

    # ─────────────────────────────────────────────
    # AI / チャット
    # ─────────────────────────────────────────────
    AI_ENGINE             = "ai_engine"            # 使用AIエンジン
    GEMINI_API_KEY        = "gemini_api_key"       # Gemini APIキー
    GEMINI_API_KEY_INPUT  = "gemini_api_key_input" # 入力ウィジェット用
    GEMINI_MODEL          = "gemini_model"         # Geminiモデル名
    OLLAMA_MODEL          = "ollama_model"         # Ollamaモデル名
    OLLAMA_TEST_RESULT    = "ollama_test_result"   # Ollamaテスト結果
    LAST_GEMINI_DEBUG     = "last_gemini_debug"    # デバッグ情報
    MESSAGES              = "messages"             # チャット履歴
    CHAT_RESULT           = "chat_result"          # チャット結果
    CHAT_LOADING          = "chat_loading"         # チャット読み込み中フラグ
    CHAT_LOADING_STARTED  = "chat_loading_started_at" # 読み込み開始時刻

    # ─────────────────────────────────────────────
    # AI コメント ID（重複実行防止用）
    # ─────────────────────────────────────────────
    AI_QUICK_COMMENT_ID   = "ai_quick_comment_id"
    AI_3D_COMMENT_ID      = "ai_3d_comment_id"
    AI_BYOKI_CASE_ID      = "ai_byoki_case_id"
    AI_BYOKI_TEXT         = "ai_byoki_text"
    AI_HONNE_TEXT         = "ai_honne_text"

    # ─────────────────────────────────────────────
    # ナレッジベース設定
    # ─────────────────────────────────────────────
    KB_USE_CASES          = "kb_use_cases"
    KB_USE_FAQ            = "kb_use_faq"
    KB_USE_IMPROVEMENT    = "kb_use_improvement"
    KB_USE_INDUSTRY       = "kb_use_industry"
    KB_USE_MANUAL         = "kb_use_manual"

    # ─────────────────────────────────────────────
    # 業種フラグメント（JSIC分類）
    # ─────────────────────────────────────────────
    FRAG_JSIC_DATA        = "_frag_jsic_data"
    FRAG_MAJOR            = "_frag_major"
    FRAG_MAPPED_COEFF     = "_frag_mapped_coeff"
    FRAG_SUB              = "_frag_sub"
    FRAG_SUB_DATA         = "_frag_sub_data"

    # ─────────────────────────────────────────────
    # 資産ファイナンス（AF）
    # ─────────────────────────────────────────────
    AF_LAST_RESULT        = "af_last_result"       # AFタブの直前計算結果
    AF_LAST_DATA          = "af_last_data"         # AFタブの直前入力データ

    # ─────────────────────────────────────────────
    # 感度分析
    # ─────────────────────────────────────────────
    YOUDEN_RESULT         = "_youden_result"       # ユーデン指数分析結果

    # ─────────────────────────────────────────────
    # 審査ルール設定
    # ─────────────────────────────────────────────
    CUSTOM_RULES_UI_DATA  = "custom_rules_ui_data"       # カスタムルールのUI用データ
    RULES_SAVED_SNAPSHOT  = "_rules_saved_snapshot"      # 保存済みルールのスナップショット
    RULES_FORCE_RELOAD    = "_rules_force_reload"        # ルール強制リロードフラグ
    RULE_PAGE_PREV_MODE   = "_rule_page_prev_mode"       # ルールページの直前モード

    # ─────────────────────────────────────────────
    # エージェントチーム (AT)
    # ─────────────────────────────────────────────
    AT_THEME              = "at_theme"             # 議論テーマ入力値
    AT_HISTORY            = "at_history"           # ラウンド履歴リスト
    AT_CURRENT_ROUND      = "at_current_round"     # 最新ラウンド結果（None可）
    AT_RUNNING            = "at_running"           # 実行中フラグ（二重送信防止）
    AT_CODE_RESULTS       = "at_code_results"      # 生成済みコード（round_num → code str）
    AT_PENDING            = "at_pending"           # つね決裁待ち（4人の意見を保持）
    AT_APPLIED_FILES      = "at_applied_files"     # 適用済みファイル履歴（round_num → list）

    # ─────────────────────────────────────────────
    # その他
    # ─────────────────────────────────────────────
    BATTLE_DATA           = "battle_data"          # バトルデータ
    DEBATE_HISTORY        = "debate_history"       # ディベート履歴
    NEWS_RESULTS          = "news_results"         # ニュース検索結果
    SELECTED_NEWS_CONTENT = "selected_news_content"# 選択ニュース内容
    CONSULTATION_INPUT    = "consultation_input"   # 相談入力
    CONSULTATION_PENDING  = "consultation_pending_q" # 保留中の相談
    SLIDE_NENSHUU         = "slide_nenshuu"        # スライダー年収


# 後方互換: 文字列辞書としても参照できるようにする
SESSION_KEYS: dict[str, str] = {
    attr: getattr(SK, attr)
    for attr in dir(SK)
    if not attr.startswith("__")
}

__all__ = ["SK", "SESSION_KEYS"]
