"""
AI チャット・ディベート関連のバックエンド関数。
Ollama / Gemini API の呼び出し・リトライ・接続テストをまとめたモジュール。
UI（Streamlit 側）は lease_logic_sumaho10.py に残す。
"""
import os
import json
import time
import concurrent.futures
import datetime
import streamlit as st

from config import (
    OLLAMA_MODEL,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
    DEBATE_FILE,
)

# スレッド → メインで結果を渡す用（session_state はスレッドから更新不可）
_chat_result_holder: dict = {"result": None, "done": False}


def _get_gemini_key_from_secrets() -> str:
    """secrets.toml が無くても例外にしない。キーがあれば返す。"""
    try:
        if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
            return st.secrets.get("GEMINI_API_KEY", "") or ""
    except Exception:
        pass
    return ""


def get_ollama_model() -> str:
    """
    実際に使用するモデル名を取得するヘルパー。
    st.session_state['ollama_model'] があればそれを優先、なければ環境変数ベースの OLLAMA_MODEL を返す。
    """
    model = st.session_state.get("ollama_model", "").strip() if "ollama_model" in st.session_state else ""
    return model or OLLAMA_MODEL


def _ollama_chat_http(model: str, messages: list, timeout_seconds: int):
    """Ollama の HTTP API を直接叩く。requests の timeout で確実に切る。"""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests が必要です: pip install requests")

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = base + "/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            f"Ollama が {timeout_seconds} 秒以内に応答しませんでした。\n"
            "・ターミナルで `ollama serve` が動いているか確認してください。\n"
            "・モデルが重い場合は初回の応答に時間がかかります。軽いモデル（例: lease-anna）を試すか、Gemini API に切り替えてください。"
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Ollama に接続できませんでした。\n"
            "・ターミナルで **ollama serve** を実行してから再度お試しください。\n"
            f"・接続先: {base}\n"
            f"・詳細: {e}"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama がタイムアウトしました（{timeout_seconds}秒）。\n"
            "・軽いモデル（lease-anna 等）を試すか、サイドバーで Gemini API に切り替えてください。"
        )

    if resp.status_code == 404:
        try:
            err_body = resp.json()
            err_msg = err_body.get("error", resp.text)
        except Exception:
            err_msg = resp.text
        raise RuntimeError(
            f"モデル「{model}」が見つかりません。\n"
            f"・ターミナルで **ollama pull {model}** を実行してモデルを取得してください。\n"
            f"・またはサイドバー「AIモデル設定」で別のモデル（例: lease-anna）を選択してください。\n"
            f"・Ollamaの詳細: {err_msg[:200]}"
        )
    resp.raise_for_status()
    data = resp.json()
    if "message" in data and "content" in data["message"]:
        return {"message": {"content": data["message"]["content"]}}
    raise RuntimeError("Ollama の応答形式が不正です。")


def _gemini_chat(api_key: str, model: str, messages: list, timeout_seconds: int):
    """
    Gemini API でチャット。messages は [{"role":"user","content":"..."}] 形式。
    最後の user メッセージをプロンプトとして送り、返答テキストを返す。
    """
    if not api_key or not api_key.strip():
        return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
    prompt = ""
    for m in messages:
        if m.get("role") == "user" and m.get("content"):
            prompt = m["content"]
    if not prompt:
        return {"message": {"content": "送信する内容がありません。"}}
    try:
        import google.generativeai as genai
    except ImportError:
        return {"message": {"content": "Gemini を使うには pip install google-generativeai を実行してください。"}}

    try:
        genai.configure(api_key=api_key.strip())
        gemini_model = genai.GenerativeModel(model)
        try:
            config = genai.types.GenerationConfig(max_output_tokens=2048, temperature=0.7)
            response = gemini_model.generate_content(prompt, generation_config=config)
        except (AttributeError, TypeError):
            response = gemini_model.generate_content(prompt)

        if not response:
            return {"message": {"content": "Gemini から応答が返りませんでした。"}}

        text = None
        try:
            if response.text:
                text = response.text
        except (ValueError, AttributeError):
            pass
        if not text and getattr(response, "candidates", None):
            for c in response.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            text = (text or "") + p.text
                    if text:
                        break
        if text and text.strip():
            return {"message": {"content": text.strip()}}
        return {"message": {"content": "Gemini から空の応答か、安全フィルターでブロックされた可能性があります。プロンプトを変えて再試行してください。"}}
    except Exception as e:
        err = str(e).strip().lower()
        if "429" in err or "quota" in err or "resource_exhausted" in err or "rate limit" in err:
            return {"message": {"content": (
                "**Gemini の利用枠（無料枠の1日制限）に達している可能性があります。**\n\n"
                "・無料枠は1日あたりのリクエスト数に上限があります。\n"
                "・明日になるまでお待ちいただくか、[Google AI Studio](https://aistudio.google.com/) で利用状況を確認してください。\n"
                "・有料プランにすると制限が緩和されます。\n\n"
                f"【APIの詳細】{str(e)[:300]}"
            )}}
        return {"message": {"content": f"Gemini API エラー: {str(e)}\n\nAPIキーとモデル名（{model}）を確認し、ネット接続を確認してください。"}}


def _chat_for_thread(engine: str, model: str, messages: list, timeout_seconds: int, api_key: str = "", gemini_model: str = ""):
    """
    バックグラウンドスレッドから呼ぶ用。st.session_state を参照しない。
    engine が "gemini" のときは api_key と gemini_model を使用。
    """
    if engine == "gemini":
        api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_gemini_chat, api_key, gemini_model or "gemini-2.0-flash", messages, timeout_seconds)
                return future.result(timeout=min(timeout_seconds + 30, 90))
        except Exception as e:
            return {"message": {"content": f"Gemini が応答しませんでした。\n\n【詳細】{str(e)}"}}
    try:
        return _ollama_chat_http(model, messages, timeout_seconds)
    except Exception as e:
        return {"message": {"content": f"AIサーバーが応答しませんでした: {e}"}}


def chat_with_retry(model, messages, retries=2, timeout_seconds=120):
    """AI へのチャット呼び出し。エンジンが Gemini の場合は Gemini API、否则 Ollama。"""
    engine = st.session_state.get("ai_engine", "ollama")
    if engine == "gemini":
        api_key = (st.session_state.get("gemini_api_key") or "").strip() or GEMINI_API_KEY_ENV
        api_key = api_key or _get_gemini_key_from_secrets()
        gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
        if "last_gemini_debug" not in st.session_state:
            st.session_state["last_gemini_debug"] = ""
        for i in range(retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_gemini_chat, api_key, gemini_model, messages, timeout_seconds)
                    try:
                        out = future.result(timeout=min(timeout_seconds + 30, 90))
                    except concurrent.futures.TimeoutError:
                        st.session_state["last_gemini_debug"] = "タイムアウト（応答が返るまで待ちましたが応答がありませんでした）"
                        st.error("Gemini がタイムアウトしました。ネット接続を確認するか、しばらくして再試行してください。")
                        return {"message": {"content": "Gemini がタイムアウトしました。ネット接続を確認するか、しばらくして再試行してください。"}}
                content = (out.get("message") or {}).get("content", "")
                st.session_state["last_gemini_debug"] = "OK" if content and "APIキーが" not in content and "Gemini API エラー:" not in content else (content[:200] + "..." if len(content or "") > 200 else (content or "（空）"))
                if content and (
                    "APIキーが設定されていません" in content
                    or "Gemini API エラー:" in content
                    or "pip install" in content
                    or "応答が返りませんでした" in content
                    or "安全フィルターでブロック" in content
                    or "利用枠" in content
                    or "無料枠" in content
                ):
                    st.error(content)
                return out
            except Exception as e:
                err = str(e)
                st.session_state["last_gemini_debug"] = f"例外: {err}"
                if "429" in err or "quota" in err.lower() or "resource_exhausted" in err.lower() or "rate limit" in err.lower():
                    time.sleep(2 * (i + 1))
                    continue
                st.error(f"Gemini API エラー: {err}")
                return {"message": {"content": f"Gemini が応答しませんでした。\n\n【詳細】{err}"}}
        st.session_state["last_gemini_debug"] = "リトライ上限（または利用枠の可能性）"
        return {"message": {"content": (
            "Gemini が応答しませんでした。\n\n"
            "**無料枠の1日あたりの制限に達している可能性があります。**\n"
            "・明日までお待ちいただくか、[Google AI Studio](https://aistudio.google.com/) で利用状況を確認してください。\n"
            "・APIキー・モデル名・ネット接続もあわせて確認してください。"
        )}}

    last_error = None
    for i in range(retries):
        try:
            return _ollama_chat_http(model, messages, timeout_seconds)
        except Exception as e:
            last_error = str(e)
            if "429" in last_error:
                time.sleep(2 * (i + 1))
                continue
            break

    if last_error:
        st.error(f"AIサーバーが応答しませんでした: {last_error}")
        detail = f"\n\n【技術的な詳細】{last_error}"
        if "timed out" in last_error or "Timeout" in last_error:
            detail += "\n\n💡 左サイドバー「AIモデル設定」で **Gemini API** に切り替えるか、**lease-anna** 等の軽いモデルを試してください。"
    else:
        st.error("AIサーバーが応答しませんでした。")
        detail = ""
    return {
        "message": {
            "content": "AIが応答しませんでした。時間を置くか、Gemini API に切り替えて再試行してください。" + detail
        }
    }


def generate_battle_special_move(strength_tags: list, passion_text: str) -> tuple:
    """
    定性データから「必殺技名」と「特殊効果」を1つ生成する。
    戻り値: (name: str, effect: str)。失敗時はフォールバックを返す。
    """
    fallback = ("逆転の意気", "スコア+5%")
    if not strength_tags and not (passion_text or "").strip():
        return fallback
    model = get_ollama_model() if st.session_state.get("ai_engine") == "ollama" else GEMINI_MODEL_DEFAULT
    tags_str = "、".join(strength_tags) if strength_tags else "なし"
    text_snippet = (passion_text or "")[:300]
    prompt = f"""以下から、審査ゲーム用の「必殺技」を1つだけ考えてください。
強みタグ: {tags_str}
熱意・裏事情（抜粋）: {text_snippet or "なし"}

必殺技は「名前」と「効果」の2つだけ。1行で答えてください。形式は必ず:
必殺技名 / 効果の短い説明
例: 老舗の暖簾 / ダメージ無効
例: 業界人脈の盾 / 流動性+10%
日本語で、必殺技名は10文字以内、効果は15文字以内。他は出力しない。"""
    try:
        out = chat_with_retry(model, [{"role": "user", "content": prompt}], retries=1, timeout_seconds=15)
        content = ((out.get("message") or {}).get("content") or "").strip()
        if " / " in content:
            parts = content.split(" / ", 1)
            return (parts[0].strip()[:20] or fallback[0], (parts[1].strip()[:25] or fallback[1]))
    except Exception:
        pass
    return fallback


def is_ollama_available(timeout_seconds: int = 3) -> bool:
    """
    Ollamaサーバーが起動しているかを簡易チェックする。
    起動していない状態で chat_with_retry を呼ぶと永遠待ちになりやすいので、
    事前にここで検知してユーザーに案内を出す。
    """
    try:
        import requests
    except ImportError:
        return False

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = base + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout_seconds)
        return resp.status_code == 200
    except Exception:
        return False


def is_ai_available(timeout_seconds: int = 3) -> bool:
    """
    現在選択中のAIエンジンが利用可能かどうか。
    Gemini の場合は API キーが設定されていれば True。
    Ollama の場合はサーバーが起動していれば True。
    """
    engine = st.session_state.get("ai_engine", "ollama")
    if engine == "gemini":
        key = st.session_state.get("gemini_api_key", "").strip() or GEMINI_API_KEY_ENV
        key = key or _get_gemini_key_from_secrets()
        return bool(key)
    return is_ollama_available(timeout_seconds)


def run_ollama_connection_test(timeout_seconds: int = 10) -> str:
    """
    Ollama の接続とモデル応答をテストし、結果メッセージを返す。
    サイドバーの「Ollama接続テスト」ボタン用。
    """
    try:
        import requests
    except ImportError:
        return "❌ requests がインストールされていません: pip install requests"

    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = get_ollama_model() or OLLAMA_MODEL

    try:
        r = requests.get(base + "/api/tags", timeout=5)
        if r.status_code != 200:
            return f"❌ Ollama サーバー応答異常: {base} (HTTP {r.status_code})"
    except requests.exceptions.ConnectionError:
        return (
            f"❌ Ollama に接続できません。\n"
            f"接続先: {base}\n\n"
            "**対処:** ターミナルで以下を実行してください。\n"
            "```\nollama serve\n```"
        )
    except requests.exceptions.Timeout:
        return f"❌ Ollama サーバーが応答しませんでした（5秒でタイムアウト）。\n接続先: {base}"

    try:
        r = requests.post(
            base + "/api/chat",
            json={"model": model, "messages": [{"role": "user", "content": "こん"}], "stream": False},
            timeout=timeout_seconds,
        )
        if r.status_code == 404:
            return (
                f"⚠️ サーバーは動いていますが、モデル「{model}」が見つかりません。\n\n"
                f"**対処:** ターミナルで以下を実行してください。\n"
                f"```\nollama pull {model}\n```\n\n"
                "またはサイドバーで別のモデル（例: lease-anna）を選択してください。"
            )
        r.raise_for_status()
        data = r.json()
        content = (data.get("message") or {}).get("content", "")
        if content:
            return f"✅ 接続OK（モデル: {model}）\n応答: {content[:80]}{'…' if len(content) > 80 else ''}"
        return f"✅ 接続OK（モデル: {model}）\n（応答本文は空でした）"
    except requests.exceptions.Timeout:
        return (
            f"⚠️ モデル「{model}」が {timeout_seconds} 秒以内に応答しませんでした。\n\n"
            "・初回はモデルの読み込みで時間がかかることがあります。\n"
            "・軽いモデル（lease-anna 等）を試すか、Gemini API に切り替えてください。"
        )
    except Exception as e:
        return f"❌ チャットテスト失敗: {e}"


def save_debate_log(data: dict):
    """ディベート結果を保存"""
    data["timestamp"] = datetime.datetime.now().isoformat()
    try:
        with open(DEBATE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        st.error(f"ディベート保存エラー: {e}")
