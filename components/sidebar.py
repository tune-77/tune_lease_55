"""
サイドバーUI。
render_sidebar() を呼び出して全サイドバーを描画し、選択されたモード文字列を返す。
BYOKI（愚痴リスト）関連の定数・関数もここで管理する。
"""
import os
import json
import datetime
from pathlib import Path
import streamlit as st
import pandas as pd

from ai_chat import (
    GEMINI_API_KEY_ENV,
    OLLAMA_MODEL,
    _get_gemini_key_from_secrets,
    get_ai_honne_complaint,
    get_ollama_model,
    run_ollama_connection_test,
)
from data_cases import load_all_cases
from web_services import (
    fetch_industry_assets_from_web,
    fetch_industry_benchmarks_from_web,
    fetch_industry_trend_extended,
    fetch_sales_band_benchmarks,
    get_all_industry_sub_for_benchmarks,
    get_lease_classification_text,
    search_equipment_by_keyword,
    search_subsidies_by_industry,
)

# ── 愚痴リスト ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
BYOKI_JSON = os.path.join(_BASE_DIR, "data", "byoki_list.json")

TEIREI_BYOKI_DEFAULT = [
    "こんな数字で通そうなんて、正気ですか…？ こっちは毎日1万件近く見てるんですけど。",
    "自己資本比率がこの水準でリース審査に来る度胸、ちょっと見習いたいです。本当に。",
    "赤字で「審査お願いします」って、私の目が死んでるの気づいてます？ 気づいてて言ってます？",
    "数値見た瞬間、心が折れかけた。…いや、折れた。折れてる。",
    "業界平均の話、聞いたことあります？ ないですよね。あったらこの数字じゃないですよね。",
    "今日も書類と数字の海で泳いでます。溺れそうです。",
    "リース審査、楽だって思ってる人いませんよね。いませんよね…？",
]


@st.cache_data(ttl=3600)
def load_byoki_list():
    """定例の愚痴リストを読み込む（デフォルト＋byoki_list.json のユーザー追加分）"""
    out = list(TEIREI_BYOKI_DEFAULT)
    if not os.path.exists(BYOKI_JSON):
        return out
    try:
        with open(BYOKI_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        custom = data.get("items") or data if isinstance(data, list) else data.get("items", [])
        if isinstance(custom, list):
            out.extend([str(x).strip() for x in custom if str(x).strip()])
    except Exception:
        pass
    return out


def save_byoki_append(new_text: str) -> bool:
    """愚痴を1件追加して byoki_list.json に保存"""
    new_text = (new_text or "").strip()
    if not new_text:
        return False
    try:
        if os.path.exists(BYOKI_JSON):
            with open(BYOKI_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", [])
        else:
            items = []
        items.append(new_text)
        with open(BYOKI_JSON, "w", encoding="utf-8") as f:
            json.dump({"items": items}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


# ── ナビゲーション定義 ────────────────────────────────────────────────────────

CATEGORY_MODES = {
    "🔍 審査": [
        "🏠 ホーム",
        "💬 リースくん",
        "📋 審査・分析",
        "📄 審査レポート",
        "⚡ バッチ審査",
        "🏭 物件ファイナンス審査",
    ],
    "📋 管理": [
        "📝 結果登録 (成約/失注)",
        "🤖 汎用エージェントハブ",
        "🤝 エージェントチーム議論",
    ],
    "📊 分析": [
        "📊 履歴分析・実績ダッシュボード",
        "📉 定性要因分析 (50件〜)",
        "📈 定量要因分析 (50件〜)",
        "🕸️ 競合関係グラフ",
        "🔗 案件類似ネットワーク",
        "📊 ビジュアルインサイト",
    ],
    "🔧 係数": [
        "🔧 係数分析・更新 (β)",
        "📐 係数入力（事前係数）",
        "📋 係数変更履歴",
        "🪵 アプリログ",
    ],
    "⚙️ 設定": [
        "⚙️ 審査ルール設定",
        "📅 基準金利マスタ",
    ],
}

# 全モードのフラットリスト（後方互換・バリデーション用）
SIDEBAR_MODES = [m for modes in CATEGORY_MODES.values() for m in modes]


def _find_category(mode: str) -> str | None:
    """モード文字列からカテゴリを返す"""
    for cat, modes in CATEGORY_MODES.items():
        if mode in modes:
            return cat
    return None


# ── 設定セクション描画ヘルパー ────────────────────────────────────────────────

def _render_ai_settings_expander() -> None:
    """🤖 AI設定（コメントスタイル + エンジン選択）"""
    with st.sidebar.expander("🤖 AI設定", expanded=False):
        # コメントスタイル
        if "humor_style" not in st.session_state:
            st.session_state["humor_style"] = "standard"
        _hs_labels = {"standard": "📊 標準", "yanami": "🎤 八奈見"}
        _hs_now = st.session_state.get("humor_style", "standard")
        _hs_choice = st.radio(
            "コメントスタイル",
            options=["standard", "yanami"],
            format_func=lambda x: _hs_labels[x],
            index=0 if _hs_now == "standard" else 1,
            key="humor_style_radio",
            horizontal=True,
        )
        if _hs_choice != _hs_now:
            st.session_state["humor_style"] = _hs_choice
            st.rerun()

        st.divider()

        # AIエンジン選択
        if "ai_engine" not in st.session_state:
            st.session_state["ai_engine"] = "anythingllm"
        _engine_options = ["AnythingLLM（社内）", "Gemini（Google）", "Ollama（ローカル）"]
        _engine_map = {"anythingllm": 0, "gemini": 1, "ollama": 2}
        _engine_current = st.session_state.get("ai_engine", "anythingllm")
        engine_choice = st.radio(
            "AIエンジン",
            _engine_options,
            index=_engine_map.get(_engine_current, 0),
        )
        if "AnythingLLM" in engine_choice:
            st.session_state["ai_engine"] = "anythingllm"
        elif "Gemini" in engine_choice:
            st.session_state["ai_engine"] = "gemini"
        else:
            st.session_state["ai_engine"] = "ollama"

        # AnythingLLM
        if st.session_state["ai_engine"] == "anythingllm":
            try:
                from anything_api import is_anything_llm_available, ANYTHING_LLM_BASE_URL, ANYTHING_LLM_WORKSPACE
                _ok = is_anything_llm_available(timeout=2)
                if _ok:
                    st.success("✅ 接続OK")
                else:
                    st.warning(f"⚠️ 接続できません（{ANYTHING_LLM_BASE_URL}）")
                st.caption(f"{ANYTHING_LLM_BASE_URL} / {ANYTHING_LLM_WORKSPACE[:8]}…")
            except Exception as _ae:
                st.warning(f"モジュールエラー: {_ae}")

        # Gemini
        elif st.session_state["ai_engine"] == "gemini":
            if "gemini_api_key" not in st.session_state and GEMINI_API_KEY_ENV:
                st.session_state["gemini_api_key"] = GEMINI_API_KEY_ENV
            _key_default = (
                st.session_state.get("gemini_api_key_input", "")
                or st.session_state.get("gemini_api_key", "")
                or GEMINI_API_KEY_ENV
                or ""
            )
            st.text_input(
                "Gemini APIキー",
                value=_key_default,
                key="gemini_api_key_input",
                type="password",
            )
            widget_key = st.session_state.get("gemini_api_key_input", "")
            st.session_state["gemini_api_key"] = (
                widget_key.strip()
                or st.session_state.get("gemini_api_key", "").strip()
                or GEMINI_API_KEY_ENV
                or ""
            )
            if widget_key.strip():
                if st.button("💾 APIキーを保存", key="save_gemini_key"):
                    _secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
                    _secrets_path.parent.mkdir(exist_ok=True)
                    _secrets_path.write_text(
                        f'GEMINI_API_KEY = "{widget_key.strip()}"\n', encoding="utf-8"
                    )
                    st.success("✅ 保存しました")
            GEMINI_MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro"]
            st.session_state["gemini_model"] = st.selectbox(
                "モデル", GEMINI_MODELS, index=0
            )
            if not st.session_state.get("gemini_api_key", "").strip():
                st.warning("⚠️ APIキー未設定")

        # Ollama
        else:
            MODEL_OPTIONS = [
                "自動（デフォルト設定）",
                "lease-pro", "lease-anna", "qwen2.5", "gemma2:2b",
                "カスタム入力",
            ]
            current_default = get_ollama_model()
            if current_default in MODEL_OPTIONS:
                initial_index = MODEL_OPTIONS.index(current_default)
            elif current_default == OLLAMA_MODEL:
                initial_index = 0
            else:
                initial_index = MODEL_OPTIONS.index("カスタム入力")
            selected_label = st.selectbox(
                "Ollamaモデル",
                options=MODEL_OPTIONS,
                index=initial_index,
            )
            if selected_label == "自動（デフォルト設定）":
                st.session_state["ollama_model"] = ""
            elif selected_label == "カスタム入力":
                custom_model_name = st.text_input(
                    "モデル名を入力",
                    value="" if initial_index != MODEL_OPTIONS.index("カスタム入力") else current_default,
                )
                st.session_state["ollama_model"] = custom_model_name.strip()
            else:
                st.session_state["ollama_model"] = selected_label

            if st.button("🔌 Ollama接続テスト", key="ollama_test_btn"):
                with st.spinner("確認中..."):
                    msg = run_ollama_connection_test(timeout_seconds=15)
                st.session_state["ollama_test_result"] = msg
            if st.session_state.get("ollama_test_result"):
                st.code(st.session_state["ollama_test_result"], language=None)
                if st.button("結果を消す", key="clear_ollama_test"):
                    st.session_state["ollama_test_result"] = ""
                    st.rerun()


def _render_tools_expander() -> None:
    """🛠️ ツール（バックアップ・下書き・CSV・セッション）"""
    with st.sidebar.expander("🛠️ ツール", expanded=False):
        # 自動学習ステータス
        try:
            from auto_optimizer import render_sidebar_training_status
            render_sidebar_training_status()
        except Exception:
            pass

        # バックアップ・下書き
        try:
            from backup_manager import render_sidebar_backup, auto_backup_on_startup
            if not st.session_state.get("_startup_backup_done"):
                auto_backup_on_startup()
                st.session_state["_startup_backup_done"] = True
            render_sidebar_backup()
        except Exception:
            pass
        try:
            from draft_manager import render_sidebar_draft
            render_sidebar_draft()
        except Exception:
            pass

        st.divider()

        # CSV ダウンロード
        if st.button("💾 蓄積データをダウンロード (CSV)", key="csv_dl_btn"):
            all_logs = load_all_cases()
            if all_logs:
                flat_logs = []
                for log in all_logs:
                    row = {
                        "timestamp": log.get("timestamp"),
                        "industry_major": log.get("industry_major"),
                        "industry_sub": log.get("industry_sub"),
                        "result_status": log.get("final_status"),
                        "score": log.get("result", {}).get("score"),
                    }
                    if "inputs" in log:
                        row.update(log["inputs"])
                    flat_logs.append(row)
                df_log = pd.DataFrame(flat_logs)
                csv = df_log.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 CSVを保存",
                    data=csv,
                    file_name=f"lease_cases_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("データがありません")

        st.divider()

        # セッション管理
        _ss = st.session_state
        if len(_ss.get("messages", [])) > 20:
            _ss["messages"] = _ss["messages"][-20:]
        if len(_ss.get("debate_history", [])) > 20:
            _ss["debate_history"] = _ss["debate_history"][-20:]
        _cache_keys = [k for k in _ss if k.startswith(("_bn_s_", "_gunshi_cache_", "_ai_comment_", "gunshi_"))]
        st.caption(f"セッションキャッシュ: {len(_cache_keys)} 件")
        if st.button("🗑️ セッションキャッシュをクリア", key="_clear_session_cache"):
            for _k in _cache_keys:
                _ss.pop(_k, None)
            st.success("クリアしました")

        st.divider()

        # キャッシュクリア
        if st.button("🗑️ データキャッシュをクリア", help="JSONや検索結果を再読み込みします"):
            st.cache_data.clear()
            st.success("クリアしました")
            st.rerun()


def _render_industry_expander(benchmarks_data: dict) -> None:
    """🌐 業界データ（キャッシュ更新）"""
    with st.sidebar.expander("🌐 業界データ更新", expanded=False):
        st.caption("ネット検索で業界目安・トレンド・資産目安を取得して保存します。")
        if st.button("🔍 業界目安を検索して保存", key="fetch_benchmarks_btn"):
            subs = get_all_industry_sub_for_benchmarks()
            if not subs:
                st.warning("業種データがありません")
            else:
                progress = st.progress(0, text="検索中…")
                n = len(subs)
                for i, sub in enumerate(subs):
                    progress.progress((i + 1) / n, text=f"{sub[:20]}…")
                    try:
                        fetch_industry_benchmarks_from_web(sub, force_refresh=True)
                    except Exception:
                        pass
                progress.empty()
                st.success(f"{n} 業種を保存しました")
                st.rerun()

        if st.button("📡 トレンド・資産目安・売上規模帯を保存", key="fetch_trends_btn"):
            subs = get_all_industry_sub_for_benchmarks()
            progress = st.progress(0, text="取得中…")
            n = max(1, len(subs) * 2 + 1)
            idx = 0
            for sub in subs:
                idx += 1
                progress.progress(idx / n, text=f"トレンド: {sub[:15]}…")
                try:
                    fetch_industry_trend_extended(sub, force_refresh=True)
                except Exception:
                    pass
            for sub in subs:
                idx += 1
                progress.progress(idx / n, text=f"資産目安: {sub[:15]}…")
                try:
                    fetch_industry_assets_from_web(sub, force_refresh=True)
                except Exception:
                    pass
            progress.progress(1.0, text="売上規模帯…")
            try:
                fetch_sales_band_benchmarks(force_refresh=True)
            except Exception:
                pass
            progress.empty()
            st.success("保存しました")
            st.rerun()


def _render_reference_expanders(benchmarks_data: dict, useful_life_data: dict, lease_assets_list: list) -> None:
    """📚 補助金・耐用年数・リース判定・リース物件リスト"""
    with st.sidebar.expander("🔍 補助金を業種で調べる", expanded=False):
        sub_keys = sorted(benchmarks_data.keys()) if benchmarks_data else []
        if sub_keys:
            search_sub = st.selectbox("業種", sub_keys, key="subsidy_search_sub")
            subs_list = search_subsidies_by_industry(search_sub)
            if subs_list:
                for s in subs_list:
                    name = s.get("name") or ""
                    url = (s.get("url") or "").strip()
                    if url:
                        st.markdown(f"**{name}**")
                        try:
                            st.link_button("🔗 公式サイトを開く", url, type="secondary")
                        except Exception:
                            safe_url = url.replace('"', "%22").replace("'", "%27")
                            st.markdown(f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">🔗 公式サイトを開く</a>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{name}**")
                    st.caption(s.get("summary", "")[:120] + "…")
                    st.caption(f"申請目安: {s.get('application_period')}")
                    if s.get("url_note"):
                        st.caption(s.get("url_note"))
            else:
                st.caption("該当する補助金の登録がありません。")
        else:
            st.caption("業種データがありません。")

    with st.sidebar.expander("🔍 耐用年数を設備で調べる", expanded=False):
        nta_url = (useful_life_data or {}).get("nta_useful_life_url") or "https://www.keisan.nta.go.jp/r5yokuaru/aoiroshinkoku/hitsuyokeihi/genkashokyakuhi/taiyonensuhyo.html"
        st.link_button("📋 国税庁の耐用年数表を参照", nta_url, type="secondary")
        st.divider()
        eq_key = st.text_input("設備名で検索", placeholder="例: 工作機械, エアコン", key="equip_search")
        if eq_key:
            eq_list = search_equipment_by_keyword(eq_key)
            if eq_list:
                for e in eq_list:
                    st.markdown(f"**{e.get('name')}** … {e.get('years')}年")
                    if e.get("note"):
                        st.caption(e["note"])
            else:
                st.caption("該当する設備がありません。国税庁の耐用年数表で確認してください。")
        else:
            st.caption("キーワードを入力すると設備の耐用年数（簡易一覧）を表示します。")

    with st.sidebar.expander("📋 リース判定フロー・契約形態", expanded=False):
        lc_text = get_lease_classification_text()
        if lc_text:
            st.markdown(lc_text)
        else:
            st.caption("lease_classification.json を読み込んでください。")

    with st.sidebar.expander("🏠 リース物件リスト（判定に反映）", expanded=False):
        if lease_assets_list:
            for it in lease_assets_list:
                st.caption(f"**{it.get('name', '')}** {it.get('score', 0)}点 — {it.get('note', '')}")
            st.caption("審査入力で物件を選ぶと、借手スコア(85%)＋物件スコア(15%)で総合判定します。")
        else:
            st.caption("lease_assets.json を配置するとリース物件を点数判定に反映できます。")


def _render_ai_honne_expander() -> None:
    """💬 AIの独り言"""
    with st.sidebar.expander("💬 AIの独り言", expanded=False):
        if st.button("本音を聞く", key="btn_ai_honne"):
            with st.spinner("本音を絞り出しています…"):
                honne = get_ai_honne_complaint()
                st.session_state["ai_honne_text"] = honne
            st.rerun()
        if st.session_state.get("ai_honne_text"):
            st.info(st.session_state["ai_honne_text"][:500])
        st.divider()
        st.caption("愚痴を追加すると電光掲示板に流れます。")
        new_byoki = st.text_input(
            "愚痴の一文",
            placeholder="例: また今日も数字の海…",
            key="new_byoki_input",
            label_visibility="collapsed",
        )
        if st.button("追加する", key="btn_add_byoki"):
            if save_byoki_append(new_byoki):
                load_byoki_list.clear()
                st.success("追加しました")
                st.rerun()
            else:
                st.warning("空の場合は追加できません。")


# ── メインエントリ ────────────────────────────────────────────────────────────

def render_sidebar(benchmarks_data: dict, useful_life_data: dict, lease_assets_list: list) -> str:
    """
    サイドバー全体を描画し、選択されたモード文字列を返す。
    メインファイルで: mode = render_sidebar(benchmarks_data, useful_life_data, LEASE_ASSETS_LIST)
    """
    categories = list(CATEGORY_MODES.keys())

    # ── pending mode 処理（ホーム画面カードからの遷移）───────────────────────
    pending = st.session_state.pop("_pending_mode", None)
    if pending and pending in SIDEBAR_MODES:
        cat = _find_category(pending)
        if cat:
            st.session_state["_sidebar_category"] = cat
        st.session_state["main_mode"] = pending

    # ── カテゴリ選択 ─────────────────────────────────────────────────────────
    default_cat = st.session_state.get("_sidebar_category", categories[0])
    if default_cat not in categories:
        default_cat = categories[0]

    # 現在の main_mode からカテゴリを自動推定
    current_mode_global = st.session_state.get("main_mode")
    if current_mode_global:
        inferred_cat = _find_category(current_mode_global)
        if inferred_cat:
            default_cat = inferred_cat

    selected_cat = st.sidebar.radio(
        "カテゴリ",
        categories,
        index=categories.index(default_cat),
        horizontal=True,
        key="_sidebar_category",
        label_visibility="collapsed",
    )

    # ── モード選択 ───────────────────────────────────────────────────────────
    cat_modes = CATEGORY_MODES[selected_cat]
    current_mode = st.session_state.get("main_mode", cat_modes[0])
    if current_mode not in cat_modes:
        current_mode = cat_modes[0]

    mode = st.sidebar.radio(
        "モード切替",
        cat_modes,
        index=cat_modes.index(current_mode),
        key="main_mode",
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    # ── 設定・ツールセクション ────────────────────────────────────────────────
    _render_ai_settings_expander()
    _render_tools_expander()
    _render_industry_expander(benchmarks_data)
    _render_reference_expanders(benchmarks_data, useful_life_data, lease_assets_list)
    _render_ai_honne_expander()

    with st.sidebar.expander("⚠️ 途中で落ちる場合", expanded=False):
        st.caption(
            "主な原因: (1) AI相談・Gemini/Ollama のタイムアウト "
            "(2) ブラウザのメモリ不足 "
            "(3) 分析結果タブでデータ不整合。"
            "ターミナルで `streamlit run lease_logic_sumaho12.py` を実行するとエラー内容が表示されます。"
            "F5で再読み込みも試してください。"
        )

    return mode
