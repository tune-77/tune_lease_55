import streamlit as st
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

# プロジェクトのルートディレクトリを取得
PROJECT_ROOT = Path(__file__).parent.resolve()

# .envファイルの読み込み
load_dotenv()
from utils import session_keys, ai_chat

# ページのタイトルとアイコンの設定
st.set_page_config(page_title="リース審査AI", page_icon=":money_with_wings:")

# セッション状態の初期化
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False

# ログイン処理
def login():
    """
    ログイン処理を行う関数
    """
    username = st.session_state.username
    password = st.session_state.password
    # ユーザー情報を読み込む
    try:
        with open(PROJECT_ROOT / "users.json", "r") as f:
            users: Dict[str, str] = json.load(f)
    except FileNotFoundError:
        st.error("ユーザー情報ファイルが見つかりません。")
        return False
    except json.JSONDecodeError:
        st.error("ユーザー情報ファイルの読み込みに失敗しました。")
        return False

    if username in users and users[username] == password:
        st.success(f"{username}さん、ようこそ！")
        st.session_state["is_logged_in"] = True
        return True
    else:
        st.error("ユーザー名またはパスワードが間違っています。")
        return False

# ログインフォーム
if not st.session_state["is_logged_in"]:
    st.title("リース審査AIへようこそ")
    st.markdown("ログインしてサービスをご利用ください。")
    with st.form("login_form"):
        st.session_state.username = st.text_input("ユーザー名")
        st.session_state.password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")
        if submitted:
            login()

# ログイン後の表示
import streamlit as st
import os
import json
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any

# プロジェクトのルートディレクトリを取得
PROJECT_ROOT = Path(__file__).parent.resolve()

# .envファイルの読み込み
load_dotenv()
from utils import session_keys, ai_chat
from components.dashboard import display_score_dashboard

# ページのタイトルとアイコンの設定
st.set_page_config(page_title="リース審査AI", page_icon=":money_with_wings:")

# セッション状態の初期化
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False

# ログイン処理
def login():
    """
    ログイン処理を行う関数
    """
    username = st.session_state.username
    password = st.session_state.password
    # ユーザー情報を読み込む
    try:
        with open(PROJECT_ROOT / "users.json", "r") as f:
            users: Dict[str, str] = json.load(f)
    except FileNotFoundError:
        st.error("ユーザー情報ファイルが見つかりません。")
        return False
    except json.JSONDecodeError:
        st.error("ユーザー情報ファイルの読み込みに失敗しました。")
        return False

    if username in users and users[username] == password:
        st.success(f"{username}さん、ようこそ！")
        st.session_state["is_logged_in"] = True
        return True
    else:
        st.error("ユーザー名またはパスワードが間違っています。")
        return False

# ログインフォーム
if not st.session_state["is_logged_in"]:
    st.title("リース審査AIへようこそ")
    st.markdown("ログインしてサービスをご利用ください。")
    with st.form("login_form"):
        st.session_state.username = st.text_input("ユーザー名")
        st.session_state.password = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("ログイン")
        if submitted:
            login()

# ログイン後の表示
if st.session_state.get("is_logged_in"):
    st.sidebar.title("メニュー")
    selected_page = st.sidebar.selectbox("ページを選択", ["ホーム", "審査ダッシュボード", "チャット"])

    if selected_page == "審査ダッシュボード":
        # ダミーデータ
        score = 97
        details = {
            '収益性': {'値': 95, 'リスク': '低'},
            '自己資本比率': {'値': 100, 'リスク': '低'},
            'DSCR': {'値': 80, 'リスク': '低'},
            '契約期待度': {'値': 90, 'リスク': '中'},
            '業種': {'値': 70, 'リスク': '高'}
        }
        display_score_dashboard(score, details)

    elif selected_page == "チャット":
        ai_chat.main()

    else:
        st.write("ホーム画面")