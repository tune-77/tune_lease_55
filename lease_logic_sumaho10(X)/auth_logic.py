import streamlit as st
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_FACE_PATH = os.path.join(_SCRIPT_DIR, "admin_face.jpg")
ADMIN_PASSWORD = "123456"

# 顔認証ライブラリの読み込み試行
try:
    import face_recognition
    FACE_AUTH_AVAILABLE = True
except ImportError:
    FACE_AUTH_AVAILABLE = False

def authenticate_user():
    """
    顔認証またはパスワード認証を行い、認証済みであればTrueを返す。
    未認証の場合は認証UIを表示し、認証試行を行う。
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 セキュリティ認証")
    st.markdown("このアプリケーションにアクセスするには、**パスワード** または **顔認証** が必要です。")

    # パスワード認証 - st.button() はフォームコンテキスト問題を起こすため
    # on_change コールバック方式に変更（Enter キーでログイン）
    st.markdown("#### 🔑 パスワード")

    def _on_password_change():
        pw = st.session_state.get("login_password", "")
        if pw == ADMIN_PASSWORD:
            st.session_state.authenticated = True
            st.session_state["_pw_error"] = False
        elif pw:
            st.session_state["_pw_error"] = True

    st.text_input(
        "パスワードを入力（Enterで認証）",
        type="password",
        key="login_password",
        label_visibility="collapsed",
        placeholder="パスワードを入力して Enter",
        on_change=_on_password_change,
    )
    if st.session_state.get("_pw_error"):
        st.error("パスワードが違います。")
    if st.session_state.get("authenticated"):
        st.success("パスワード認証成功！")
        st.rerun()

    # 顔認証はエクスパンダーで（フォームコンテキスト外）
    with st.expander("📷 顔認証でログイン", expanded=False):
        if not FACE_AUTH_AVAILABLE:
            st.warning("現在、顔認証システムは準備中です（ライブラリインストール中）。パスワードをご利用ください。")
        else:
            if 'admin_encoding' not in st.session_state:
                if not os.path.exists(ADMIN_FACE_PATH):
                    st.error("管理者画像が見つかりません。")
                else:
                    try:
                        if 'admin_image_loaded' not in st.session_state:
                            admin_image = face_recognition.load_image_file(ADMIN_FACE_PATH)
                            encodings = face_recognition.face_encodings(admin_image)
                            if len(encodings) > 0:
                                st.session_state.admin_encoding = encodings[0]
                                st.session_state.admin_image_loaded = True
                            else:
                                st.error("管理者画像から顔を検出できませんでした。")
                    except Exception as e:
                        st.error(f"顔認証システムエラー: {e}")

            if 'admin_encoding' in st.session_state:
                img_file_buffer = st.camera_input("カメラで顔を撮影してください", key="login_camera")
                if img_file_buffer is not None:
                    try:
                        user_image = face_recognition.load_image_file(img_file_buffer)
                        user_encodings = face_recognition.face_encodings(user_image)
                        if len(user_encodings) == 0:
                            st.warning("顔が検出されませんでした。")
                        else:
                            match = face_recognition.compare_faces([st.session_state.admin_encoding], user_encodings[0], tolerance=0.5)
                            if match[0]:
                                st.success("顔認証成功！")
                                st.session_state.authenticated = True
                                st.rerun()
                            else:
                                st.error("顔が一致しません。")
                    except Exception as e:
                        st.error(f"認証エラー: {e}")

    return False
