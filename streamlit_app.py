

# ── エージェントチーム承認コード ─────────────────────
import streamlit as st
import random
import sqlite3

# セッションキーの定義 (既存の定義に追加)
class session_keys:
    CHAT_MESSAGES = "chat_messages"
    MODEL_NAME = "model_name"
    TEMPERATURE = "temperature"
    PROMPT = "prompt"
    SYSTEM_PROMPT = "system_prompt"
    OLLAMA_MODEL = "ollama_model"
    GEMINI_MODEL = "gemini_model"
    THREAD_ID = "thread_id"
    LIKED_ARTICLES = "liked_articles" # 追加

# データベース接続関数
def get_db_connection():
    conn = sqlite3.connect('articles.db')
    conn.row_factory = sqlite3.Row  # 行を辞書型で取得
    return conn

# テーブル作成関数 (初回のみ実行)
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS liked_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# いいね！された文章を保存する関数
def save_liked_article(content):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO liked_articles (content) VALUES (?)", (content,))
    conn.commit()
    conn.close()

# いいね！された文章を取得する関数
def get_liked_articles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM liked_articles")
    articles = [row['content'] for row in cursor.fetchall()]
    conn.close()
    return articles

# お題をランダムに選ぶ関数
def get_random_topic():
    topics = [
        "猫が主人公の冒険物語",
        "AIが支配する未来",
        "タイムトラベルで過去の自分に会いに行く",
        "宇宙人が地球にやってきた",
        "夢の中で起きた不思議な出来事"
    ]
    return random.choice(topics)

# 初期化処理
if session_keys.LIKED_ARTICLES not in st.session_state:
    st.session_state[session_keys.LIKED_ARTICLES] = get_liked_articles()

create_table() # テーブルがなければ作成

# UIの構築
st.title("文豪AI (仮)")

# お題ガチャボタン
if st.button("お題ガチャ"):
    st.session_state.topic = get_random_topic()
    st.session_state.generated_text = None # 新しいお題なので、生成されたテキストをクリア

if "topic" in st.session_state:
    st.write(f"お題: {st.session_state.topic}")

    # 文章生成ボタン (お題が選択されたら表示)
    if st.button("文章を生成"):
        with st.spinner("文豪AIが執筆中…"):
            try:
                from novelist_agent import generate_novel
                result = generate_novel(custom_theme=st.session_state.topic)
                st.session_state.generated_text = f"**{result['title']}**\n\n{result['body']}"
            except Exception as e:
                st.error(f"生成エラー: {e}")
                st.session_state.generated_text = None

    # 生成された文章を表示
    if "generated_text" in st.session_state and st.session_state.generated_text:
        st.write("生成された文章:")
        st.write(st.session_state.generated_text)

        # いいね！ボタン
        if st.button("いいね！"):
            save_liked_article(st.session_state.generated_text)
            st.session_state[session_keys.LIKED_ARTICLES] = get_liked_articles() # 状態を更新
            st.success("いいね！しました")

# いいね！した文章の表示
st.subheader("いいね！した文章")
if st.session_state[session_keys.LIKED_ARTICLES]:
    for article in st.session_state[session_keys.LIKED_ARTICLES]:
        st.write(article)
else:
    st.write("まだいいね！した文章はありません。")