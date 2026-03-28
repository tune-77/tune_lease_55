# -*- coding: utf-8 -*-
from __future__ import annotations
"""
auth_logic.py
=============
ユーザー別パスワード認証モジュール（多人数対応版）。
ユーザー情報は data/users.db（SQLite）に保存。

移行: .app_password が存在する場合、初回起動時に admin ユーザーとして自動取り込む。
"""
import streamlit as st
import hashlib
import os
import sqlite3
import time
from contextlib import closing

_PKG_DIR   = os.path.dirname(os.path.abspath(__file__))
_USERS_DB  = os.path.join(_PKG_DIR, "data", "users.db")
_PW_FILE   = os.path.join(_PKG_DIR, ".app_password")   # 旧形式（移行用）
_TIMEOUT_S = 60 * 60 * 8  # 8時間でセッション期限切れ


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# SQLite ユーザーDB
# ─────────────────────────────────────────────────────────────────────────────

def _init_db() -> None:
    """usersテーブルを作成し、.app_password からの移行を行う。"""
    os.makedirs(os.path.dirname(_USERS_DB), exist_ok=True)
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username     TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role         TEXT NOT NULL DEFAULT 'user',
                created_at   TEXT NOT NULL,
                last_login   TEXT
            )
        """)
        conn.commit()

        # .app_password からの移行
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        if row[0] == 0 and os.path.exists(_PW_FILE):
            with open(_PW_FILE, "r") as f:
                old_hash = f.read().strip()
            if old_hash:
                conn.execute(
                    "INSERT INTO users(username, password_hash, role, created_at) VALUES (?,?,?,datetime('now'))",
                    ("admin", old_hash, "admin"),
                )
                conn.commit()


def _get_user(username: str) -> dict | None:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        row = conn.execute(
            "SELECT username, password_hash, role FROM users WHERE username=?",
            (username,),
        ).fetchone()
    if row:
        return {"username": row[0], "password_hash": row[1], "role": row[2]}
    return None


def _add_user(username: str, password: str, role: str = "user") -> bool:
    try:
        with closing(sqlite3.connect(_USERS_DB)) as conn:
            conn.execute(
                "INSERT INTO users(username, password_hash, role, created_at) VALUES (?,?,?,datetime('now'))",
                (username, _hash(password), role),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def _delete_user(username: str) -> None:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        conn.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()


def _change_password(username: str, new_password: str) -> None:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (_hash(new_password), username),
        )
        conn.commit()


def _update_last_login(username: str) -> None:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        conn.execute(
            "UPDATE users SET last_login=datetime('now') WHERE username=?",
            (username,),
        )
        conn.commit()


def _list_users() -> list[dict]:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        rows = conn.execute(
            "SELECT username, role, created_at, last_login FROM users ORDER BY username"
        ).fetchall()
    return [{"username": r[0], "role": r[1], "created_at": r[2], "last_login": r[3]} for r in rows]


def _has_any_user() -> bool:
    with closing(sqlite3.connect(_USERS_DB)) as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def _setup_ui() -> None:
    """初回ユーザー登録画面（DBにユーザーがいない場合）。"""
    st.title("🔒 初回セットアップ — 管理者アカウント作成")
    st.info("初めて起動しました。管理者アカウントを作成してください。")

    def _on_set():
        uname = st.session_state.get("setup_uname", "").strip()
        pw1   = st.session_state.get("setup_pw1", "")
        pw2   = st.session_state.get("setup_pw2", "")
        if not uname:
            st.session_state["_setup_err"] = "ユーザー名を入力してください。"
        elif len(pw1) < 6:
            st.session_state["_setup_err"] = "パスワードは6文字以上で設定してください。"
        elif pw1 != pw2:
            st.session_state["_setup_err"] = "パスワードが一致しません。"
        else:
            _add_user(uname, pw1, role="admin")
            st.session_state["_setup_done"] = True
            st.session_state["_setup_err"]  = ""

    st.text_input("管理者ユーザー名", key="setup_uname")
    st.text_input("パスワード（6文字以上）", type="password", key="setup_pw1")
    st.text_input("パスワード（確認）",      type="password", key="setup_pw2", on_change=_on_set)

    if st.button("✅ 作成する", type="primary"):
        _on_set()

    if st.session_state.get("_setup_err"):
        st.error(st.session_state["_setup_err"])
    if st.session_state.get("_setup_done"):
        st.success("✅ 管理者アカウントを作成しました。再読み込みします。")
        st.rerun()


def _login_ui() -> None:
    """ログイン画面。"""
    st.title("🔒 リース審査AI — ログイン")

    def _on_login():
        uname = st.session_state.get("login_username", "").strip()
        pw    = st.session_state.get("login_password", "")
        if not uname or not pw:
            return
        user = _get_user(uname)
        if user and user["password_hash"] == _hash(pw):
            st.session_state["authenticated"] = True
            st.session_state["auth_time"]     = time.time()
            st.session_state["username"]      = uname
            st.session_state["user_role"]     = user["role"]
            st.session_state["_login_error"]  = False
            _update_last_login(uname)
        else:
            st.session_state["_login_error"] = True

    st.text_input("ユーザー名", key="login_username")
    st.text_input(
        "パスワードを入力して Enter",
        type="password",
        key="login_password",
        label_visibility="collapsed",
        placeholder="パスワードを入力して Enter",
        on_change=_on_login,
    )
    if st.button("ログイン", type="primary"):
        _on_login()

    if st.session_state.get("_login_error"):
        st.error("❌ ユーザー名またはパスワードが違います。")
    if st.session_state.get("authenticated"):
        st.success("✅ 認証成功！")
        st.rerun()


def _user_management_sidebar() -> None:
    """サイドバーのユーザー管理UI（管理者のみ）。"""
    if st.session_state.get("user_role") != "admin":
        return

    with st.sidebar.expander("👥 ユーザー管理", expanded=False):
        users = _list_users()
        for u in users:
            st.caption(f"{'👑' if u['role']=='admin' else '👤'} {u['username']}  最終ログイン: {u['last_login'] or '未'}")

        st.divider()
        st.markdown("**新規ユーザー追加**")
        new_uname = st.text_input("ユーザー名", key="_mgmt_uname")
        new_pw    = st.text_input("パスワード", type="password", key="_mgmt_pw")
        new_role  = st.selectbox("権限", ["user", "admin"], key="_mgmt_role")
        if st.button("追加", key="_mgmt_add"):
            if new_uname and new_pw:
                if _add_user(new_uname, new_pw, new_role):
                    st.success(f"✅ {new_uname} を追加しました。")
                    st.rerun()
                else:
                    st.error("既に存在するユーザー名です。")
            else:
                st.warning("ユーザー名とパスワードを入力してください。")

        st.divider()
        st.markdown("**ユーザー削除**")
        usernames = [u["username"] for u in users if u["username"] != st.session_state.get("username")]
        if usernames:
            del_target = st.selectbox("削除対象", usernames, key="_mgmt_del_target")
            if st.button("削除", key="_mgmt_del", type="secondary"):
                _delete_user(del_target)
                st.success(f"✅ {del_target} を削除しました。")
                st.rerun()
        else:
            st.caption("削除できるユーザーがいません。")


def _change_password_sidebar() -> None:
    """サイドバーにパスワード変更UIを表示（認証済み時のみ）。"""
    with st.sidebar.expander("🔑 パスワード変更", expanded=False):
        pw_old  = st.text_input("現在のパスワード",         type="password", key="chg_pw_old")
        pw_new  = st.text_input("新しいパスワード（6文字以上）", type="password", key="chg_pw_new")
        pw_new2 = st.text_input("新しいパスワード（確認）",  type="password", key="chg_pw_new2")
        if st.button("変更する", key="chg_pw_btn"):
            uname = st.session_state.get("username", "")
            user  = _get_user(uname)
            if not user or user["password_hash"] != _hash(pw_old):
                st.error("現在のパスワードが違います。")
            elif len(pw_new) < 6:
                st.error("6文字以上で設定してください。")
            elif pw_new != pw_new2:
                st.error("新しいパスワードが一致しません。")
            else:
                _change_password(uname, pw_new)
                st.success("✅ パスワードを変更しました。")


# ─────────────────────────────────────────────────────────────────────────────
# 公開 API
# ─────────────────────────────────────────────────────────────────────────────

def authenticate_user() -> bool:
    """
    認証チェック。
    - DBにユーザーがいなければ初回設定画面を表示
    - 未認証 or セッション期限切れならログイン画面を表示
    - 認証済みなら True を返す
    """
    _init_db()

    # 初回セットアップ
    if not _has_any_user():
        _setup_ui()
        return False

    # セッション期限チェック（8時間）
    auth_time = st.session_state.get("auth_time", 0)
    if st.session_state.get("authenticated") and (time.time() - auth_time) > _TIMEOUT_S:
        st.session_state["authenticated"] = False
        st.warning("⏱ セッションが期限切れになりました。再ログインしてください。")

    # 未認証ならログイン画面
    if not st.session_state.get("authenticated"):
        _login_ui()
        return False

    # 認証済み: サイドバーUI
    _change_password_sidebar()
    _user_management_sidebar()
    return True


def get_current_username() -> str:
    """現在ログイン中のユーザー名を返す。未認証時は 'default'。"""
    return st.session_state.get("username") or "default"
