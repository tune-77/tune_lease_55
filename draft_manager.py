"""
フォーム入力の途中保存モジュール。
session_state の審査入力キーを data/draft_form.json に保存・復元する。
アプリリロード後でも入力を引き継げる。
"""
import json
import os
from datetime import datetime

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DRAFT_FILE  = os.path.join(_SCRIPT_DIR, "data", "draft_form.json")

# 保存対象の session_state キー（フォーム入力項目）
_DRAFT_KEYS = [
    # 業種選択
    "select_major", "select_sub",
    # 取引状況
    "customer_type", "contract_type", "deal_source", "main_bank", "competitor",
    # P/L 財務数値
    "item9_gross", "rieki", "item4_ord_profit", "item5_net_income",
    "item10_dep", "item11_dep_exp", "item8_rent", "item12_rent_exp",
    "item6_machine", "item7_other",
    # B/S
    "net_assets", "total_assets",
    # 与信・契約
    "bank_credit", "lease_credit", "contracts",
    # リース条件
    "lease_term", "acquisition_cost", "acceptance_year",
    # 格付・物件
    "grade", "selected_asset_id",
    # スライダー系（_slider_and_number で使う prefix キー）
    "nenshuu_num", "rieki_num", "item4_num", "item5_num",
    "item10_num", "net_assets_num", "total_assets_num",
]


# ─────────────────────────────────────────────────────────────────────────────
# 保存・読み込み
# ─────────────────────────────────────────────────────────────────────────────

def save_draft() -> bool:
    """現在の session_state から審査入力値を JSON ファイルに保存する。"""
    try:
        import streamlit as st
        data: dict = {"_saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for key in _DRAFT_KEYS:
            val = st.session_state.get(key)
            if val is not None:
                try:
                    # シリアライズ可能かチェック
                    json.dumps(val)
                    data[key] = val
                except (TypeError, ValueError):
                    data[key] = str(val)
        os.makedirs(os.path.dirname(_DRAFT_FILE), exist_ok=True)
        with open(_DRAFT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_draft() -> dict:
    """保存済み下書きを読み込んで返す（存在しなければ空 dict）。"""
    if not os.path.exists(_DRAFT_FILE):
        return {}
    try:
        with open(_DRAFT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def restore_draft() -> bool:
    """下書きを session_state に復元する。復元したキー数 > 0 なら True。"""
    try:
        import streamlit as st
        data = load_draft()
        restored = 0
        for key in _DRAFT_KEYS:
            if key in data and data[key] is not None:
                st.session_state[key] = data[key]
                restored += 1
        return restored > 0
    except Exception:
        return False


def delete_draft() -> None:
    """下書きファイルを削除する。"""
    try:
        if os.path.exists(_DRAFT_FILE):
            os.remove(_DRAFT_FILE)
    except OSError:
        pass


def has_draft() -> bool:
    """保存済み下書きが存在するかを返す。"""
    return os.path.exists(_DRAFT_FILE)


def get_draft_saved_at() -> str | None:
    """下書きの保存日時を返す（なければ None）。"""
    data = load_draft()
    return data.get("_saved_at")


# ─────────────────────────────────────────────────────────────────────────────
# サイドバーウィジェット（Streamlit）
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_draft() -> None:
    """サイドバーに途中保存・復元ボタンを表示する。"""
    import streamlit as st

    with st.sidebar.expander("📝 フォーム下書き保存", expanded=False):
        if has_draft():
            saved_at = get_draft_saved_at()
            st.caption(f"保存済み: {saved_at or '（日時不明）'}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📂 復元", width='stretch', key="_draft_restore"):
                    if restore_draft():
                        st.success("復元しました")
                        st.rerun()
                    else:
                        st.warning("復元失敗")
            with col2:
                if st.button("🗑️ 削除", width='stretch', key="_draft_delete"):
                    delete_draft()
                    st.rerun()
        else:
            st.caption("下書きなし")

        if st.button("💾 今すぐ保存", width='stretch', key="_draft_save"):
            if save_draft():
                st.success("下書きを保存しました")
            else:
                st.error("保存失敗")
