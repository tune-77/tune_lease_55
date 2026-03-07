"""
サイドバーUI。
render_sidebar() を呼び出して全サイドバーを描画し、選択されたモード文字列を返す。
BYOKI（愚痴リスト）関連の定数・関数もここで管理する。
"""
import os
import json
import datetime
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
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)  # lease_logic_sumaho12/ ディレクトリ
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


# ── セクション描画ヘルパー ────────────────────────────────────────────────────

def _render_humor_style() -> None:
    """🎭 コメントスタイル切り替え"""
    st.sidebar.markdown("### 🎭 コメントスタイル")
    if "humor_style" not in st.session_state:
        st.session_state["humor_style"] = "standard"
    _hs_labels = {"standard": "📊 標準モード", "yanami": "🎤 八奈見モード"}
    _hs_now = st.session_state.get("humor_style", "standard")
    _hs_choice = st.sidebar.radio(
        "AIコメントの口調",
        options=["standard", "yanami"],
        format_func=lambda x: _hs_labels[x],
        index=0 if _hs_now == "standard" else 1,
        key="humor_style_radio",
        help="八奈見モードにすると、AI分析コメントが八奈見口調になります。",
    )
    if _hs_choice != _hs_now:
        st.session_state["humor_style"] = _hs_choice
        st.rerun()


def _render_ai_model_settings() -> None:
    """🤖 AIモデル設定（Gemini / Ollama 選択・APIキー・接続テスト）"""
    if "ai_engine" not in st.session_state:
        st.session_state["ai_engine"] = "gemini"  # デフォルトはGemini
    st.sidebar.markdown("### 🤖 AIモデル設定")
    engine_choice = st.sidebar.radio(
        "AIエンジン",
        ["Gemini API（Google）", "Ollama（ローカル）"],
        index=0 if st.session_state.get("ai_engine") == "gemini" else 1,
        help="Gemini 2.0 Flash は無料枠で月50件なら実質0円。APIキーはGoogle AI Studioで取得できます。",
    )
    st.session_state["ai_engine"] = "gemini" if "Gemini" in engine_choice else "ollama"

    if st.session_state["ai_engine"] == "gemini":
        if "gemini_api_key" not in st.session_state and GEMINI_API_KEY_ENV:
            st.session_state["gemini_api_key"] = GEMINI_API_KEY_ENV
        _key_default = (
            st.session_state.get("gemini_api_key_input", "")
            or st.session_state.get("gemini_api_key", "")
            or GEMINI_API_KEY_ENV
            or ""
        )
        st.sidebar.text_input(
            "Gemini APIキー",
            value=_key_default,
            key="gemini_api_key_input",
            type="password",
            help="環境変数 GEMINI_API_KEY が設定されていればここに表示されます。入力すると上書きされます。",
        )
        widget_key = st.session_state.get("gemini_api_key_input", "")
        st.session_state["gemini_api_key"] = (
            widget_key.strip()
            or st.session_state.get("gemini_api_key", "").strip()
            or GEMINI_API_KEY_ENV
            or ""
        )
        GEMINI_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
        st.session_state["gemini_model"] = st.sidebar.selectbox(
            "Gemini モデル",
            GEMINI_MODELS,
            index=0,
            help="gemini-2.0-flash がおすすめです。",
        )
        if not st.session_state.get("gemini_api_key", "").strip():
            st.sidebar.warning("⚠️ APIキー未設定。[Google AI Studio](https://aistudio.google.com/app/apikey) で無料取得 → 上欄に貼り付けてください。")
        else:
            st.sidebar.caption("✅ Gemini 2.0 Flash：月50件審査なら無料枠で収まります。")
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
        selected_label = st.sidebar.selectbox(
            "使用するOllamaモデル",
            options=MODEL_OPTIONS,
            index=initial_index,
            help="一覧からモデルを選ぶか、「カスタム入力」で任意のモデル名を指定できます。",
        )
        if selected_label == "自動（デフォルト設定）":
            st.session_state["ollama_model"] = ""
        elif selected_label == "カスタム入力":
            custom_model_name = st.sidebar.text_input(
                "モデル名を直接入力",
                value="" if initial_index != MODEL_OPTIONS.index("カスタム入力") else current_default,
                help="例: llama3, phi3 など。",
            )
            st.session_state["ollama_model"] = custom_model_name.strip()
        else:
            st.session_state["ollama_model"] = selected_label

        if st.sidebar.button("🔌 Ollama接続テスト", use_container_width=True, help="Ollama が起動しているか・選択中のモデルが応答するかを確認します"):
            with st.sidebar:
                with st.spinner("接続確認中..."):
                    msg = run_ollama_connection_test(timeout_seconds=15)
                st.session_state["ollama_test_result"] = msg
        if st.session_state.get("ollama_test_result"):
            st.sidebar.code(st.session_state["ollama_test_result"], language=None)
            if st.sidebar.button("テスト結果を消す", key="clear_ollama_test"):
                st.session_state["ollama_test_result"] = ""
                st.rerun()


def _render_auto_optimizer_status() -> None:
    """係数自動学習ステータス"""
    try:
        from auto_optimizer import render_sidebar_training_status
        render_sidebar_training_status()
    except Exception:
        pass


def _render_backup_and_draft() -> None:
    """バックアップ + フォーム下書き保存"""
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


def _render_session_cleanup() -> None:
    """🧹 セッション管理"""
    with st.sidebar.expander("🧹 セッション管理", expanded=False):
        _ss = st.session_state
        if len(_ss.get("messages", [])) > 20:
            _ss["messages"] = _ss["messages"][-20:]
        if len(_ss.get("debate_history", [])) > 20:
            _ss["debate_history"] = _ss["debate_history"][-20:]
        _cache_keys = [k for k in _ss if k.startswith(("_bn_s_", "_gunshi_cache_", "_ai_comment_", "gunshi_"))]
        st.caption(f"キャッシュキー数: {len(_cache_keys)}")
        if st.button("🗑️ キャッシュをクリア", use_container_width=True, key="_clear_session_cache"):
            for _k in _cache_keys:
                _ss.pop(_k, None)
            st.success("クリアしました")


def _render_csv_download() -> None:
    """💾 蓄積データをダウンロード（CSV）"""
    if st.sidebar.button("💾 蓄積データをダウンロード (CSV)", use_container_width=True):
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
            st.sidebar.download_button(
                "📥 CSVを保存",
                data=csv,
                file_name=f"lease_cases_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        else:
            st.sidebar.warning("データがありません")


def _render_industry_cache(benchmarks_data: dict) -> None:
    """🌐 業界目安キャッシュ（検索・保存ボタン）"""
    st.sidebar.markdown("### 🌐 業界目安キャッシュ")
    st.sidebar.caption("下のボタンでネット検索し、営業利益率・自己資本比率に加え、売上高総利益率・ROA・流動比率など指標の業界目安を web_industry_benchmarks.json に保存します。")
    if st.sidebar.button("🔍 今のデータを検索して保存（次回は4月1日更新）", use_container_width=True):
        subs = get_all_industry_sub_for_benchmarks()
        if not subs:
            st.sidebar.warning("業種データがありません（industry_benchmarks.json または過去案件を登録してください）")
        else:
            progress = st.sidebar.progress(0, text="検索中…")
            n = len(subs)
            for i, sub in enumerate(subs):
                progress.progress((i + 1) / n, text=f"{sub[:20]}…")
                try:
                    fetch_industry_benchmarks_from_web(sub, force_refresh=True)
                except Exception:
                    pass
            progress.empty()
            st.sidebar.success(f"{n} 業種を検索して保存しました。次回の自動更新は4月1日です。")
            st.rerun()

    if st.sidebar.button("📡 業界トレンド拡充・資産目安・売上規模帯を検索して保存", use_container_width=True):
        subs = get_all_industry_sub_for_benchmarks()
        progress = st.sidebar.progress(0, text="トレンド・資産目安…")
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
        st.sidebar.success("業界トレンド拡充・資産目安・売上規模帯を保存しました。")
        st.rerun()


def _render_reference_expanders(benchmarks_data: dict, useful_life_data: dict, lease_assets_list: list) -> None:
    """📚 補助金・耐用年数・リース判定・リース物件リスト"""
    st.sidebar.markdown("### 📚 補助金・耐用年数・リース判定")

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
        st.caption("上記リンクで国税庁の公式耐用年数表（減価償却資産）が開きます。")
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
                st.caption("該当する設備がありません。上記「国税庁の耐用年数表」で正式な年数を確認してください。")
        else:
            st.caption("キーワードを入力すると設備の耐用年数（簡易一覧）を表示します。正式な年数は国税庁の耐用年数表で確認してください。")

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
            st.caption("lease_assets.json を配置すると、ネット・社内のリース物件をリスト化して点数で判定に反映できます。")


def _render_cache_and_ai_honne() -> None:
    """⚙️ キャッシュクリア + 🤖 AIの独り言"""
    st.sidebar.markdown("### ⚙️ キャッシュ")
    if st.sidebar.button("🗑️ キャッシュをクリア", use_container_width=True, help="JSONや検索結果のキャッシュを消して再読み込みします。補助金・業界データを更新した後に押してください。"):
        st.cache_data.clear()
        st.sidebar.success("キャッシュをクリアしました。再読み込みしています…")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("### 🤖 AIの独り言")
    if st.sidebar.button("本音を聞く", key="btn_ai_honne", use_container_width=True):
        with st.spinner("本音を絞り出しています…"):
            honne = get_ai_honne_complaint()
            st.session_state["ai_honne_text"] = honne
        st.rerun()
    if st.session_state.get("ai_honne_text"):
        st.sidebar.caption("**AIの本音**")
        st.sidebar.info(st.session_state["ai_honne_text"][:500])
    with st.sidebar.expander("愚痴を追加", expanded=False):
        st.sidebar.caption("追加した愚痴は、メニュー下の電光掲示板に流れます。")
        new_byoki = st.sidebar.text_input(
            "愚痴の一文",
            placeholder="例: また今日も数字の海…",
            key="new_byoki_input",
            label_visibility="collapsed",
        )
        if st.sidebar.button("追加する", key="btn_add_byoki"):
            if save_byoki_append(new_byoki):
                load_byoki_list.clear()
                st.sidebar.success("追加しました。掲示板に反映されます。")
                st.rerun()
            else:
                st.sidebar.warning("空の場合は追加できません。")


# ── メインエントリ ────────────────────────────────────────────────────────────

SIDEBAR_MODES = [
    "📋 審査・分析",
    "⚡ バッチ審査",
    "🏭 物件ファイナンス審査",
    "📝 結果登録 (成約/失注)",
    "🔧 係数分析・更新 (β)",
    "📐 係数入力（事前係数）",
    "📊 履歴分析・実績ダッシュボード",
    "📉 定性要因分析 (50件〜)",
    "📈 定量要因分析 (50件〜)",
    "⚙️ 審査ルール設定",
]


def render_sidebar(benchmarks_data: dict, useful_life_data: dict, lease_assets_list: list) -> str:
    """
    サイドバー全体を描画し、選択されたモード文字列を返す。
    メインファイルで: mode = render_sidebar(benchmarks_data, useful_life_data, LEASE_ASSETS_LIST)
    """
    mode = st.sidebar.radio("モード切替", SIDEBAR_MODES, key="main_mode")

    with st.sidebar.expander("⚠️ 途中で落ちる場合", expanded=False):
        st.caption("主な原因: (1) AI相談・Gemini/Ollama のタイムアウト (2) ブラウザのメモリ不足 (3) 分析結果タブでデータ不整合。ターミナルで `streamlit run lease_logic_sumaho8.py` を実行するとエラー内容が表示されます。F5で再読み込みも試してください。")

    _render_humor_style()
    _render_ai_model_settings()
    _render_auto_optimizer_status()
    _render_backup_and_draft()
    _render_session_cleanup()
    _render_csv_download()
    _render_industry_cache(benchmarks_data)
    _render_reference_expanders(benchmarks_data, useful_life_data, lease_assets_list)
    _render_cache_and_ai_honne()

    return mode
