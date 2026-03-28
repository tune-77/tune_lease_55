"""
フォーム入力の途中保存モジュール（多人数対応版）。
session_state の審査入力キーを data/drafts/{username}_form.json に保存・復元する。
ユーザー別にファイルを分離することで同時利用時の上書き競合を防ぐ。
"""
import json
import os
from datetime import datetime, timedelta

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DRAFTS_DIR  = os.path.join(_SCRIPT_DIR, "data", "drafts")

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


def _get_draft_file() -> str:
    """現在のログインユーザー名に対応する下書きファイルパスを返す。"""
    import streamlit as st
    username = st.session_state.get("username") or "default"
    # ファイル名に使えない文字を除去
    safe_name = "".join(c for c in username if c.isalnum() or c in ("-", "_"))
    safe_name = safe_name or "default"
    os.makedirs(_DRAFTS_DIR, exist_ok=True)
    return os.path.join(_DRAFTS_DIR, f"{safe_name}_form.json")


def _cleanup_old_drafts(days: int = 7) -> None:
    """7日以上更新のない下書きファイルを削除する。"""
    if not os.path.isdir(_DRAFTS_DIR):
        return
    threshold = datetime.now() - timedelta(days=days)
    for fname in os.listdir(_DRAFTS_DIR):
        if not fname.endswith("_form.json"):
            continue
        fpath = os.path.join(_DRAFTS_DIR, fname)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime < threshold:
                os.remove(fpath)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 保存・読み込み
# ─────────────────────────────────────────────────────────────────────────────

def save_draft() -> bool:
    """現在の session_state から審査入力値を JSON ファイルに保存する。"""
    try:
        import streamlit as st
        draft_file = _get_draft_file()
        data: dict = {"_saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for key in _DRAFT_KEYS:
            val = st.session_state.get(key)
            if val is not None:
                try:
                    json.dumps(val)
                    data[key] = val
                except (TypeError, ValueError):
                    data[key] = str(val)
        with open(draft_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_draft() -> dict:
    """保存済み下書きを読み込んで返す（存在しなければ空 dict）。"""
    try:
        draft_file = _get_draft_file()
        if not os.path.exists(draft_file):
            return {}
        with open(draft_file, "r", encoding="utf-8") as f:
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
        draft_file = _get_draft_file()
        if os.path.exists(draft_file):
            os.remove(draft_file)
    except OSError:
        pass


def has_draft() -> bool:
    """保存済み下書きが存在するかを返す。"""
    try:
        return os.path.exists(_get_draft_file())
    except Exception:
        return False


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
