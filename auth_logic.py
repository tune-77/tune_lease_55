# -*- coding: utf-8 -*-
from __future__ import annotations
"""
auth_logic.py
=============
パスワード認証モジュール。
パスワードはソースコードに書かず、SHA-256ハッシュを .app_password ファイルに保存。
"""
import streamlit as st
import hashlib
import os

_PKG_DIR   = os.path.dirname(os.path.abspath(__file__))
_PW_FILE   = os.path.join(_PKG_DIR, ".app_password")   # ハッシュ保存先
_TIMEOUT_S = 60 * 60 * 8  # 8時間でセッション期限切れ


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _load_hash() -> str | None:
    """保存済みパスワードハッシュを読む。ファイルがなければ None。"""
    if not os.path.exists(_PW_FILE):
        return None
    with open(_PW_FILE, "r") as f:
        h = f.read().strip()
    return h if h else None


def _save_hash(pw: str):
    """パスワードをハッシュ化して保存し、権限を 600 に設定。"""
    with open(_PW_FILE, "w") as f:
        f.write(_hash(pw))
    os.chmod(_PW_FILE, 0o600)


def _setup_ui():
    """初回パスワード設定画面。"""
    st.title("🔒 初回セットアップ — パスワード設定")
    st.info("初めて起動しました。アプリのパスワードを設定してください。")

    def _on_set():
        pw1 = st.session_state.get("setup_pw1", "")
        pw2 = st.session_state.get("setup_pw2", "")
        if len(pw1) < 6:
            st.session_state["_setup_err"] = "6文字以上で設定してください。"
        elif pw1 != pw2:
            st.session_state["_setup_err"] = "パスワードが一致しません。"
        else:
            _save_hash(pw1)
            st.session_state["_setup_done"] = True
            st.session_state["_setup_err"]  = ""

    st.text_input("新しいパスワード（6文字以上）",
                  type="password", key="setup_pw1", label_visibility="visible")
    st.text_input("もう一度入力",
                  type="password", key="setup_pw2", label_visibility="visible",
                  on_change=_on_set)

    if st.button("✅ 設定する", type="primary"):
        _on_set()

    if st.session_state.get("_setup_err"):
        st.error(st.session_state["_setup_err"])
    if st.session_state.get("_setup_done"):
        st.success("✅ パスワードを設定しました。再読み込みします。")
        st.rerun()


def _login_ui(stored_hash: str):
    """ログイン画面。"""
    st.title("🔒 リース審査AI — ログイン")

    def _on_pw():
        pw = st.session_state.get("login_password", "")
        if pw and _hash(pw) == stored_hash:
            import time
            st.session_state["authenticated"]    = True
            st.session_state["auth_time"]        = time.time()
            st.session_state["_pw_error"]        = False
        elif pw:
            st.session_state["_pw_error"] = True

    st.text_input(
        "パスワードを入力して Enter",
        type="password",
        key="login_password",
        label_visibility="collapsed",
        placeholder="パスワードを入力して Enter",
        on_change=_on_pw,
    )
    if st.session_state.get("_pw_error"):
        st.error("❌ パスワードが違います。")
    if st.session_state.get("authenticated"):
        st.success("✅ 認証成功！")
        st.rerun()


def _change_password_sidebar():
    """サイドバーにパスワード変更UIを表示（認証済み時のみ）。"""
    with st.sidebar.expander("🔑 パスワード変更", expanded=False):
        pw_old = st.text_input("現在のパスワード", type="password", key="chg_pw_old")
        pw_new = st.text_input("新しいパスワード（6文字以上）", type="password", key="chg_pw_new")
        pw_new2 = st.text_input("新しいパスワード（確認）", type="password", key="chg_pw_new2")
        if st.button("変更する", key="chg_pw_btn"):
            stored = _load_hash()
            if not pw_old or _hash(pw_old) != stored:
                st.error("現在のパスワードが違います。")
            elif len(pw_new) < 6:
                st.error("6文字以上で設定してください。")
            elif pw_new != pw_new2:
                st.error("新しいパスワードが一致しません。")
            else:
                _save_hash(pw_new)
                st.success("✅ パスワードを変更しました。")


def authenticate_user() -> bool:
    """
    認証チェック。
    - .app_password がなければ初回設定画面を表示
    - 未認証 or セッション期限切れならログイン画面を表示
    - 認証済みなら True を返す
    """
    import time

    stored_hash = _load_hash()

    # 初回セットアップ
    if stored_hash is None:
        _setup_ui()
        return False

    # セッション期限チェック（8時間）
    auth_time = st.session_state.get("auth_time", 0)
    if st.session_state.get("authenticated") and (time.time() - auth_time) > _TIMEOUT_S:
        st.session_state["authenticated"] = False
        st.warning("⏱ セッションが期限切れになりました。再ログインしてください。")

    # 未認証ならログイン画面
    if not st.session_state.get("authenticated"):
        _login_ui(stored_hash)
        return False

    # 認証済み: サイドバーにパスワード変更UIを出す
    _change_password_sidebar()
    return True
