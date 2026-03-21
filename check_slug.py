import requests
import streamlit as st

# secrets.toml からキーを取得
api_key = st.secrets.get("ANYTHING_LLM_API_KEY")
BASE_URL = "http://127.0.0.1:3001/api/v1"

if not api_key:
    print("APIキーが設定されていません。")
else:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        # ワークスペース一覧を取得する命令
        response = requests.get(f"{BASE_URL}/workspaces", headers=headers)
        if response.status_code == 200:
            workspaces = response.json().get("workspaces", [])
            print("\n--- あなたのワークスペース一覧 ---")
            for ws in workspaces:
                # ここに表示される 'slug' が本名です
                print(f"表示名: {ws['name']} -> 本名(Slug): {ws['slug']}")
            print("----------------------------------\n")
        else:
            print(f"エラーが発生しました: {response.status_code}")
    except Exception as e:
        print(f"接続に失敗しました: {e}")