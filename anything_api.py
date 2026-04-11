"""
AnythingLLM ローカル知識ベースとの連携モジュール。

ワークスペースに登録された審査マニュアル・業種ガイド・事例集を
RAG（検索拡張生成）で参照し、AIチャットのコンテキストに追加する。

使い方:
    from anything_api import get_anything_llm_context, is_anything_llm_available
"""
import pathlib
import time
import requests
import streamlit as st

ANYTHING_LLM_BASE_URL = "http://127.0.0.1:3001/api/v1"
ANYTHING_LLM_WORKSPACE = "lease"


def _get_anything_llm_key() -> str:
    """
    APIキーを取得する。優先順:
      1. st.secrets（Streamlit が正しく読めた場合）
      2. このファイルの親ディレクトリを遡って .streamlit/secrets.toml を検索
         （worktree など secrets.toml が手元にない場合のフォールバック）
    """
    # 1. st.secrets から試みる
    try:
        if hasattr(st, "secrets"):
            key = st.secrets.get("ANYTHING_LLM_API_KEY", "") or ""
            if key:
                return key
    except Exception:
        pass

    # 2. ファイルシステムを遡って secrets.toml を探す（最大 6 階層）
    search = pathlib.Path(__file__).resolve().parent
    for _ in range(6):
        candidate = search / ".streamlit" / "secrets.toml"
        if candidate.exists():
            try:
                import toml
                data = toml.load(str(candidate))
                key = data.get("ANYTHING_LLM_API_KEY", "") or ""
                if key:
                    return key
            except Exception:
                pass
        search = search.parent

    return ""


@st.cache_data(ttl=300, show_spinner=False)
def is_anything_llm_available(timeout: int = 3) -> bool:
    """AnythingLLM サーバーが起動していて認証が通るかチェック。"""
    api_key = _get_anything_llm_key()
    if not api_key:
        return False
    try:
        resp = requests.get(
            f"{ANYTHING_LLM_BASE_URL}/auth",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


def query_anything_llm(message: str, workspace_slug: str = ANYTHING_LLM_WORKSPACE) -> str:
    """
    AnythingLLM のワークスペースに問い合わせ、回答テキストを返す。
    mode="query" で知識ベース検索（RAG）に特化させる。
    失敗時は空文字を返す（例外は握りつぶす）。
    """
    api_key = _get_anything_llm_key()
    if not api_key:
        return ""
    try:
        url = f"{ANYTHING_LLM_BASE_URL}/workspace/{workspace_slug}/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"message": message, "mode": "query"}
        resp = None
        for attempt in range(4):
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(2 ** attempt)
        if resp.status_code == 200:
            if not resp.text or not resp.text.strip():
                return ""
            try:
                return resp.json().get("textResponse", "") or ""
            except ValueError:
                return ""
        return ""
    except Exception:
        return ""


def chat_anything_llm(messages: list, workspace_slug: str = ANYTHING_LLM_WORKSPACE, timeout: int = 120) -> dict:
    """
    messages ([{"role": "...", "content": "..."}]) を受け取り AnythingLLM でチャットして
    {"message": {"content": "..."}} 形式で返す。
    Ollama / Gemini と同じ戻り値形式にすることで ai_chat.py から透過的に呼べる。
    """
    api_key = _get_anything_llm_key()
    if not api_key:
        return {"message": {"content": "AnythingLLM の APIキーが設定されていません。サイドバーで入力してください。"}}

    # messages を1つのテキストに結合してAnythingLLMへ送る
    # system → 先頭に、user/assistant はロールを明示して連結
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    conv_parts = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        label = "ユーザー" if role == "user" else "アシスタント"
        conv_parts.append(f"[{label}] {m.get('content', '')}")

    combined = ""
    if system_parts:
        combined += "\n".join(system_parts) + "\n\n"
    combined += "\n".join(conv_parts)

    try:
        url = f"{ANYTHING_LLM_BASE_URL}/workspace/{workspace_slug}/chat"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"message": combined, "mode": "chat"}
        resp = None
        for attempt in range(4):
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code != 429:
                break
            wait = 2 ** attempt  # 1s → 2s → 4s → 8s
            time.sleep(wait)
        if resp.status_code == 200:
            if not resp.text or not resp.text.strip():
                return {"message": {"content": "（AnythingLLM から空の応答が返りました。サーバーの状態を確認してください）"}}
            try:
                text = resp.json().get("textResponse", "") or ""
            except ValueError:
                return {"message": {"content": f"AnythingLLM の応答がJSONではありません。（受信内容: {resp.text[:100]}）"}}
            return {"message": {"content": text or "（AnythingLLM から空の応答でした）"}}
        return {"message": {"content": f"AnythingLLM エラー: HTTP {resp.status_code} — {resp.text[:200]}"}}
    except requests.exceptions.ConnectionError:
        return {"message": {"content": "AnythingLLM に接続できません。http://127.0.0.1:3001 が起動しているか確認してください。"}}
    except requests.exceptions.Timeout:
        return {"message": {"content": f"AnythingLLM がタイムアウトしました（{timeout}秒）。"}}
    except Exception as e:
        return {"message": {"content": f"AnythingLLM 呼び出しエラー: {e}"}}


def get_anything_llm_context(q: str, res: dict | None = None, industry: str = "") -> str:
    """
    審査案件情報を元に AnythingLLM で知識ベースを検索し、
    AIチャットのコンテキストブロック文字列として返す。

    Args:
        q:        ユーザーの質問文
        res:      審査結果 dict（score, hantei 等）
        industry: 業種名（sub分類）

    Returns:
        空文字（利用不可 or 結果なし）または
        "【社内知識ベース（AnythingLLM）】\n..." 形式の文字列
    """
    if not is_anything_llm_available():
        return ""

    res = res or {}
    score = res.get("score", 0) or 0
    hantei = res.get("hantei", "") or ""

    # 案件情報を含む検索クエリを構築
    parts = [q]
    if industry:
        parts.append(f"業種:{industry}")
    if hantei:
        parts.append(f"判定:{hantei}")
    if score:
        parts.append(f"スコア:{score:.0f}%")

    search_query = " ".join(parts)

    result = query_anything_llm(search_query)
    if not result or len(result.strip()) < 10:
        return ""

    return f"【社内知識ベース（AnythingLLM）】\n{result[:1500]}"
