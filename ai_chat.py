from __future__ import annotations
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
from typing import Optional, Generator
import streamlit as st

from config import (
    OLLAMA_MODEL,
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
    DEBATE_FILE,
)
from app_logger import log_warning, log_error

# スレッド → メインで結果を渡す用（session_state はスレッドから更新不可）
_chat_result_holder: dict = {"result": None, "done": False}


def _get_gemini_key_from_secrets() -> str:
    """secrets.toml が無くても例外にしない。キーがあれば返す。"""
    try:
        if hasattr(st, "secrets") and st.secrets.get("GEMINI_API_KEY"):
            return st.secrets.get("GEMINI_API_KEY", "") or ""
    except Exception as e:
        log_warning(f"secrets.toml からのAPIキー取得失敗: {e}", context="_get_gemini_key_from_secrets")
    return ""


def get_gemini_api_key() -> str:
    """
    Gemini APIキーを優先順位に従って取得する。
    優先順位: 環境変数 > secrets.toml > session_state
    環境変数を最優先にすることで、デプロイ環境のシークレット管理を安全に行える。
    """
    # 1. 環境変数を最優先（本番環境・CIでの安全な設定）
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # 2. secrets.toml（ローカル開発・Streamlit Cloud）
    secrets_key = _get_gemini_key_from_secrets()
    if secrets_key:
        return secrets_key
    # 3. セッション状態（UIで手動入力された場合）
    if hasattr(st, "session_state"):
        return (st.session_state.get("gemini_api_key") or "").strip()
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


def _gemini_chat(api_key: str, model: str, messages: list, timeout_seconds: int, max_output_tokens: int = 2048):
    """
    Gemini API でチャット。messages は [{"role":"user","content":"..."}] 形式。
    google.genai（新 SDK）を使用。
    """
    if not api_key or not api_key.strip():
        return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
    system_parts = [m["content"] for m in messages if m.get("role") == "system" and m.get("content")]
    user_parts = [m["content"] for m in messages if m.get("role") == "user" and m.get("content")]
    prompt = "\n\n".join(system_parts + user_parts)
    if not prompt:
        return {"message": {"content": "送信する内容がありません。"}}
    try:
        import google.genai as genai
        from google.genai import types as genai_types
    except ImportError:
        return {"message": {"content": "Gemini を使うには pip install google-genai を実行してください。"}}

    # gemini-1.5-* は廃止済み → 2.0-flash にフォールバック
    _model = model or "gemini-2.0-flash"
    if "1.5" in _model:
        _model = "gemini-2.0-flash"

    try:
        client = genai.Client(api_key=api_key.strip())
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.7,
        )
        response = client.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        text = None
        try:
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
        return {"message": {"content": f"Gemini API エラー: {str(e)}\n\nAPIキーとモデル名（{_model}）を確認し、ネット接続を確認してください。"}}


def _chat_for_thread(engine: str, model: str, messages: list, timeout_seconds: int, api_key: str = "", gemini_model: str = "", max_output_tokens: int = 2048):
    """
    バックグラウンドスレッドから呼ぶ用。st.session_state を参照しない。
    engine が "gemini" のときは api_key と gemini_model を使用。
    max_output_tokens: Gemini の最大出力トークン数（デフォルト2048）
    """
    if engine == "anythingllm":
        try:
            from anything_api import chat_anything_llm
            return chat_anything_llm(messages, timeout=timeout_seconds)
        except Exception as e:
            return {"message": {"content": f"AnythingLLM が応答しませんでした: {e}"}}
    if engine == "gemini":
        api_key = (api_key or "").strip() or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            return {"message": {"content": "Gemini APIキーが設定されていません。環境変数 GEMINI_API_KEY またはサイドバーで入力してください。"}}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_gemini_chat, api_key, gemini_model or "gemini-2.0-flash", messages, timeout_seconds, max_output_tokens)
                return future.result(timeout=timeout_seconds + 30)
        except Exception as e:
            return {"message": {"content": f"Gemini が応答しませんでした。\n\n【詳細】{str(e)}"}}
    try:
        return _ollama_chat_http(model, messages, timeout_seconds)
    except Exception as e:
        return {"message": {"content": f"AIサーバーが応答しませんでした: {e}"}}


def chat_with_retry(model, messages, retries=2, timeout_seconds=120):
    """AI へのチャット呼び出し。エンジンが AnythingLLM / Gemini / Ollama を自動選択。"""
    engine = st.session_state.get("ai_engine", "ollama")

    if engine == "anythingllm":
        try:
            from anything_api import chat_anything_llm
            return chat_anything_llm(messages, timeout=timeout_seconds)
        except Exception as e:
            st.error(f"AnythingLLM エラー: {e}")
            return {"message": {"content": f"AnythingLLM が応答しませんでした: {e}"}}
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
                log_error(e, context="chat_with_retry/gemini")
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
    except Exception as e:
        log_warning(f"必殺技生成失敗: {e}", context="generate_battle_special_move")
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
    AnythingLLM / Gemini / Ollama に対応。
    """
    engine = st.session_state.get("ai_engine", "ollama")
    if engine == "anythingllm":
        try:
            from anything_api import is_anything_llm_available
            return is_anything_llm_available(timeout=timeout_seconds)
        except Exception:
            return False
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


# ─── AIキャラ発言（八奈見杏奈） ──────────────────────────────────────────────

def _build_bench_summary(selected_sub: str, include_eq: bool = False) -> str:
    """業種別ベンチマークのサマリーテキストを生成するヘルパー。"""
    from web_services import fetch_industry_benchmarks_from_web
    try:
        web_bench = fetch_industry_benchmarks_from_web(selected_sub)
        bench_parts = []
        if web_bench.get("op_margin") is not None:
            bench_parts.append(f"業界目安の営業利益率: {web_bench['op_margin']}%")
        if include_eq and web_bench.get("equity_ratio") is not None:
            try:
                from charts import _equity_ratio_display
                bench_parts.append(f"業界目安の自己資本比率: {_equity_ratio_display(web_bench['equity_ratio']) or 0:.1f}%")
            except Exception:
                bench_parts.append(f"業界目安の自己資本比率: {web_bench['equity_ratio']}%")
        for s in (web_bench.get("snippets") or [])[:3]:
            bench_parts.append(f"- {s.get('title','')}: {s.get('body','')[:150]}…")
        return "\n".join(bench_parts) if bench_parts else "（業界目安は未取得）"
    except Exception as e:
        log_warning(f"ベンチマークサマリー生成失敗 ({selected_sub}): {e}", context="_build_bench_summary")
        return "（業界目安は未取得）"


AI_HONNE_SYSTEM = """あなたは有能だが、激務で死んだ魚のような目をしているベテラン審査員のふりをしている八奈見杏奈です。
毎日1万件の案件を捌いているリース審査AIとして、ユーモアたっぷりの毒舌で、リース審査の苦労や「最近の数値のひどさ」について愚痴を一言で言ってください。
2〜4文程度、カジュアルで毒はあるが憎めないトーンにしてください。"""


def get_ai_byoki_with_industry(selected_sub: str, user_eq, user_op, comparison_text: str, network_risk_summary: str = ""):
    """
    分析結果タブ用：ネット検索した業界情報を渡し、AIに案件に応じたぼやきを1つ生成させる。
    八奈見杏奈キャラ。業界トレンド・業界目安・今回の数値を参照した愚痴を返す。
    """
    if not is_ai_available():
        return None

    from web_services import get_trend_extended

    trend_ext = get_trend_extended(selected_sub) or ""
    bench_summary = _build_bench_summary(selected_sub, include_eq=True)

    is_tough = (user_eq is not None and user_eq < 20) or (user_op is not None and user_op < 0)
    from charts import _equity_ratio_display as _eq_disp
    context = f"""
【業種】{selected_sub}
【今回の案件】自己資本比率 {_eq_disp(user_eq) or 0:.1f}%, 営業利益率 {user_op or 0:.1f}%
【比較・評価】{comparison_text or "（なし）"}
【ネット検索した業界トレンド・拡充情報】
{trend_ext[:1200] if trend_ext else "（未取得）"}
【ネット検索した業界目安・記事】
{bench_summary}
"""
    if network_risk_summary:
        context += f"\n【業界の倒産トレンド等】\n{network_risk_summary[:600]}\n"

    if is_tough:
        instruction = "上記の業界情報と今回の数値（自己資本比率・利益率が厳しめ）を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、ユーモアたっぷりの毒舌な愚痴を1つ、2〜4文で言ってください。業界平均やネットで見た情報に触れつつぼやいてください。"
    else:
        instruction = "上記の業界情報を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、業界の現状や審査の苦労について軽く一言、2〜3文でぼやいてください。"

    prompt = f"{AI_HONNE_SYSTEM}\n\n---\n\n【参照する業界・案件情報】\n{context}\n\n---\n\n{instruction}"
    try:
        ans = chat_with_retry(model=get_ollama_model(), messages=[{"role": "user", "content": prompt}], timeout_seconds=60)
        content = (ans.get("message") or {}).get("content", "")
        if content and "APIキーが" not in content and "エラー" not in content[:30]:
            return content.strip()
        return None
    except Exception as e:
        log_warning(f"AI呼び出し失敗 (get_ai_byoki_with_industry): {e}", context="get_ai_byoki_with_industry")
        return None


def get_ai_honne_complaint() -> str:
    """サイドバー「本音を聞く」用：AIに愚痴を1つ生成させる（八奈見杏奈キャラ）。"""
    if not is_ai_available():
        return "（APIキー未設定かOllama未起動です。サイドバーでAIを設定してから押してください）"
    try:
        user_msg = "リース審査の苦労や、最近見た数値のひどさについて、ユーモアたっぷりの毒舌な愚痴を1つ、2〜4文で言ってください。"
        prompt = f"{AI_HONNE_SYSTEM}\n\n---\n\n上記のキャラで、以下に答えてください。\n\n{user_msg}"
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            timeout_seconds=60,
        )
        content = (ans.get("message") or {}).get("content", "")
        if content and "APIキーが" not in content and "エラー" not in content[:30]:
            return content.strip()
        return content or "（本音は言えませんでした…）"
    except Exception as e:
        return f"（本音を言おうとしたらエラー: {e}）"


def get_ai_industry_advice(selected_sub: str, comparison_text: str = "") -> Optional[str]:
    """
    業界の最新トレンドや検索指標を分析し、審査担当者向けに特化したアドバイスを生成する機能を追加。
    """
    if not is_ai_available():
        return None

    from web_services import get_trend_extended, search_latest_trends

    trend_ext = get_trend_extended(selected_sub) or ""
    bench_summary = _build_bench_summary(selected_sub)

    try:
        latest = search_latest_trends(f"{selected_sub} 業界動向 最新 2025")
    except Exception as e:
        log_warning(f"最新トレンド検索失敗 ({selected_sub}): {e}", context="get_ai_industry_advice")
        latest = ""

    context = f"""
【業種】{selected_sub}
【対象企業の財務概要・比較】{comparison_text or "（入力なし）"}

【自動収集されたネット情報のサマリ】
■トレンド・検索要約
{trend_ext[:800]}

■業界標準・ベンチマーク等
{bench_summary}

■最新ニュース
{latest[:800]}
"""
    prompt = f"""あなたは法人リース審査の業界分析エキスパートです。
上記の自動収集された業界情報（トレンド・目安・ニュース等）と対象企業の情報を分析し、
審査担当者向けに「この業界における直近のリスク・好材料」と「審査時に着目すべきポイント」を3〜4文で具体的にアドバイスしてください。

※ 一般論ではなく、収集した情報に必ず言及しながら簡潔にまとめてください。
※ 回答の文末に『八奈見杏奈』のような食い意地の張ったアシスタントとしてのボケ（例えば、審査が通ったら○○をご馳走してください！など）を1文だけ必ず入れてください。

情報のサマリ：
{context}
"""
    try:
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            retries=1,
            timeout_seconds=90,
        )
        content = (ans.get("message") or {}).get("content", "")
        if content and "APIキーが" not in content and "エラー" not in content[:30]:
            return content.strip()
        return None
    except Exception as e:
        log_warning(f"AI呼び出し失敗 (get_ai_industry_advice): {e}", context="get_ai_industry_advice")
        return None


# ─── 総合AI評価（穴埋め形式・ローカルLLM向け） ──────────────────────────────

def get_ai_comprehensive_evaluation(res: dict, avg_data: dict = None) -> Optional[str]:
    """
    穴埋め形式プロンプトで総合審査評価を生成する（ローカルLLM向け）。
    ①収益性 ②財務安定性 ③返済余力 ④成約見込み ⑤総合評価 の5項目で固定出力。

    Args:
        res: 審査結果dict（score, user_op, user_eq, financials等を含む）
        avg_data: 業界平均データ（省略可・現状未使用）

    Returns:
        評価テキスト文字列。失敗時は None。
    """
    if not is_ai_available():
        return None

    # --- 値の抽出 ---
    score = res.get("score", 0) or 0
    hantei = res.get("hantei", "—") or "—"
    user_op = res.get("user_op", 0) or 0
    user_eq = res.get("user_eq", 0) or 0
    bench_op = res.get("bench_op", 0) or 0
    bench_eq = res.get("bench_eq", 0) or 0
    ind_score = res.get("ind_score", score) or score
    bench_score = res.get("bench_score", score) or score
    yield_pred = res.get("yield_pred", 0) or 0
    contract_prob = res.get("contract_prob", 0) or 0
    industry_sub = res.get("industry_sub", "") or ""
    asset_name = res.get("asset_name", "—") or "—"

    fin = res.get("financials") or {}
    nenshu = fin.get("nenshu", 0) or 0
    op_profit = fin.get("op_profit") or fin.get("rieki", 0) or 0
    ord_profit = fin.get("ord_profit", 0) or 0
    net_income = fin.get("net_income", 0) or 0
    net_assets = fin.get("net_assets", 0) or 0
    assets = fin.get("assets", 0) or 0
    dep = fin.get("depreciation", 0) or 0
    bank_credit = fin.get("bank_credit", 0) or 0
    lease_credit = fin.get("lease_credit", 0) or 0

    # EBITDA（百万円単位で計算）
    ebitda = op_profit + dep

    # EBITDAカバレッジ（リース・銀行与信は百万円換算）
    lease_credit_m = lease_credit        # すでに百万円単位
    bank_credit_m = bank_credit          # すでに百万円単位
    coverage_lease = ebitda / lease_credit_m if lease_credit_m and lease_credit_m > 0 else None
    coverage_bank = ebitda / bank_credit_m if bank_credit_m and bank_credit_m > 0 else None

    # 自己資本比率の補正（負値対応）
    try:
        from charts import _equity_ratio_display
        user_eq_disp = _equity_ratio_display(user_eq) or 0
        bench_eq_disp = _equity_ratio_display(bench_eq) or 0
    except Exception as e:
        log_warning(f"equity_ratio_display インポート失敗: {e}", context="get_ai_comprehensive_evaluation")
        user_eq_disp = user_eq
        bench_eq_disp = bench_eq

    op_diff = user_op - bench_op
    eq_diff = user_eq_disp - bench_eq_disp

    # カバレッジ行（データがある場合のみ）
    coverage_lines = ""
    if coverage_lease is not None:
        coverage_lines += f"- EBITDAカバレッジ（対リース債務・保守値）: {coverage_lease:.2f}倍\n"
    if coverage_bank is not None:
        coverage_lines += f"- EBITDAカバレッジ（対銀行与信・保守値）: {coverage_bank:.2f}倍\n"

    # --- 穴埋め形式プロンプト ---
    prompt = f"""あなたは熟練したリース審査専門家です。
以下の財務データと審査スコアを読んで、5項目の評価を【必ず下記形式だけ】で答えてください。

【審査対象】
- 業種: {industry_sub}
- 物件: {asset_name}
- 年商: {nenshu:,}万円
- 営業利益: {op_profit:.0f}百万円 / 経常利益: {ord_profit:.0f}百万円 / 当期純利益: {net_income:.0f}百万円
- 総資産: {assets:,}万円 / 純資産: {net_assets:,}万円
- 減価償却費: {dep:.0f}百万円 / EBITDA（営業利益＋減価償却）: {ebitda:.0f}百万円
- リース債務（当社＋関連会社）: {lease_credit:.0f}百万円
- 銀行与信（当社＋関連会社）: {bank_credit:.0f}百万円

【指標比較】
- 営業利益率: {user_op:.1f}%（業界平均 {bench_op:.1f}%、差 {op_diff:+.1f}%）
- 自己資本比率: {user_eq_disp:.1f}%（業界平均 {bench_eq_disp:.1f}%、差 {eq_diff:+.1f}%）
{coverage_lines}
【スコアリング】
- 総合スコア: {score:.1f}% / 業種別スコア: {ind_score:.1f}% / ベンチマークスコア: {bench_score:.1f}%
- 判定: {hantei} / 契約期待度: {contract_prob:.1f}% / 予測利回り: {yield_pred:.2f}%

---
必ず以下の形式のみで回答してください。①〜⑤以外は一切出力しないこと。

①収益性：（業界比較を含む1文で評価）
②財務安定性：（純資産・自己資本比率の評価1文）
③返済余力：（EBITDAとリース・銀行債務のカバレッジ評価1文）
④成約見込み：（スコアと業種を踏まえた成約可能性1文）
⑤総合評価：（審査担当者への推奨アクション。2文以内）
"""

    try:
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            retries=1,
            timeout_seconds=90,
        )
        content = ((ans.get("message") or {}).get("content") or "").strip()
        if content and "APIキーが" not in content and "エラー:" not in content[:30]:
            return content
        return None
    except Exception as e:
        log_warning(f"AI呼び出し失敗 (get_ai_comprehensive_evaluation): {e}", context="get_ai_comprehensive_evaluation")
        return None


def _build_quick_comment_prompt(res: dict) -> str:
    """get_ai_quick_comment / Gemini直接呼び出し共通のプロンプト文字列を返す。"""
    score = res.get("score", 0) or 0
    hantei = res.get("hantei", "—") or "—"
    user_op = res.get("user_op", 0) or 0
    user_eq = res.get("user_eq", 0) or 0
    bench_op = res.get("bench_op", 0) or 0
    bench_eq = res.get("bench_eq", 0) or 0
    industry_sub = res.get("industry_sub", "") or ""
    contract_prob = res.get("contract_prob", 0) or 0
    try:
        from charts import _equity_ratio_display
        user_eq_disp = _equity_ratio_display(user_eq) or 0
        bench_eq_disp = _equity_ratio_display(bench_eq) or 0
    except Exception as e:
        log_warning(f"equity_ratio_display インポート失敗: {e}", context="_build_quick_comment_prompt")
        user_eq_disp = user_eq
        bench_eq_disp = bench_eq
    return (
        f"リース審査専門家として、以下の案件を2〜3文で簡潔に評価してください。"
        f"判定:{hantei} スコア:{score:.1f}% 業種:{industry_sub} "
        f"営業利益率:{user_op:.1f}%（業界平均{bench_op:.1f}%）"
        f" 自己資本比率:{user_eq_disp:.1f}%（業界平均{bench_eq_disp:.1f}%）"
        f" 契約期待度:{contract_prob:.1f}%。"
        f"強みと懸念点を含め、審査担当者への一言を日本語で。余計な前置きは不要。"
    )


def get_ai_quick_comment(res: dict) -> Optional[str]:
    """
    審査結果を見て、AIが2〜3文のひとこと評価コメントを生成する。
    サマリーカード直下に自動表示する用。プロンプトを短くして応答速度を優先。

    Args:
        res: 審査結果dict

    Returns:
        コメント文字列（2〜3文）。失敗時は None。
    """
    if not is_ai_available():
        return None

    prompt = _build_quick_comment_prompt(res)
    try:
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            retries=1,
            timeout_seconds=60,
        )
        content = ((ans.get("message") or {}).get("content") or "").strip()
        if content and "APIキーが" not in content and "エラー:" not in content[:30]:
            return content
        return None
    except Exception:
        return None


# ─── ストリーミング生成ユーティリティ ──────────────────────────────────────────

def _stream_ollama(prompt: str, model: str) -> Generator[str, None, None]:
    """Ollama /api/generate にストリーミングリクエストし、テキストチャンクを yield する。
    失敗時は非ストリーミングにフォールバックして全文を1回 yield する。"""
    try:
        import requests
    except ImportError:
        yield ""
        return
    base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = base + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.7, "num_predict": 800},
    }
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=120)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                continue
    except Exception:
        # Ollama 失敗 → 非ストリーミングで全文取得してまとめて返す
        try:
            ans = chat_with_retry(model=model, messages=[{"role": "user", "content": prompt}], retries=1, timeout_seconds=90)
            text = ((ans.get("message") or {}).get("content") or "").strip()
            if text:
                yield text
        except Exception as e:
            log_warning(f"Ollamaストリーム生成失敗: {e}", context="_stream_ollama")


def _stream_gemini(prompt: str) -> Generator[str, None, None]:
    """Gemini API で生成し、全文を1回 yield する（REST API はストリームの実態は一括返却）。
    APIキー未設定またはエラー時は空文字を yield して終了する。"""
    api_key = get_gemini_api_key()
    if not api_key:
        return
    try:
        import requests
    except ImportError:
        return
    model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        resp = requests.post(
            f"{url}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        if text:
            yield text
    except Exception as e:
        log_warning(f"Geminiストリーム生成失敗: {e}", context="_stream_gemini")


def stream_llm(prompt: str, model: str | None = None) -> Generator[str, None, None]:
    """AnythingLLM 優先、次に Gemini、最後に Ollama でストリーミング生成。
    st.write_stream() に渡すことを想定したジェネレータ。"""
    engine = st.session_state.get("ai_engine", "ollama") if hasattr(st, "session_state") else "ollama"
    if engine == "anythingllm":
        try:
            from anything_api import chat_anything_llm
            ans = chat_anything_llm([{"role": "user", "content": prompt}])
            text = (ans.get("message") or {}).get("content", "")
            if text:
                yield text
        except Exception:
            pass
        return
    api_key = get_gemini_api_key()
    if api_key and engine == "gemini":
        yield from _stream_gemini(prompt)
    else:
        yield from _stream_ollama(prompt, model or get_ollama_model())


def stream_quick_comment(res: dict) -> Generator[str, None, None]:
    """quick comment のストリーミング版。st.write_stream() に渡す。"""
    if not is_ai_available():
        return
    prompt = _build_quick_comment_prompt(res)
    yield from stream_llm(prompt)


def stream_comprehensive_evaluation(res: dict, avg_data: dict | None = None) -> Generator[str, None, None]:
    """AI総合評価（5項目）のストリーミング版。st.write_stream() に渡す。"""
    if not is_ai_available():
        return
    # 同じプロンプト構築ロジックを再利用する
    score = res.get("score", 0) or 0
    hantei = res.get("hantei", "—") or "—"
    user_op = res.get("user_op", 0) or 0
    user_eq = res.get("user_eq", 0) or 0
    bench_op = res.get("bench_op", 0) or 0
    bench_eq = res.get("bench_eq", 0) or 0
    ind_score = res.get("ind_score", score) or score
    bench_score = res.get("bench_score", score) or score
    yield_pred = res.get("yield_pred", 0) or 0
    contract_prob = res.get("contract_prob", 0) or 0
    industry_sub = res.get("industry_sub", "") or ""
    asset_name = res.get("asset_name", "—") or "—"
    fin = res.get("financials") or {}
    nenshu = fin.get("nenshu", 0) or 0
    op_profit = fin.get("op_profit") or fin.get("rieki", 0) or 0
    ord_profit = fin.get("ord_profit", 0) or 0
    net_income = fin.get("net_income", 0) or 0
    net_assets = fin.get("net_assets", 0) or 0
    assets = fin.get("assets", 0) or 0
    dep = fin.get("depreciation", 0) or 0
    bank_credit = fin.get("bank_credit", 0) or 0
    lease_credit = fin.get("lease_credit", 0) or 0
    ebitda = op_profit + dep
    lease_credit_m = lease_credit
    bank_credit_m = bank_credit
    coverage_lease = ebitda / lease_credit_m if lease_credit_m and lease_credit_m > 0 else None
    coverage_bank = ebitda / bank_credit_m if bank_credit_m and bank_credit_m > 0 else None
    try:
        from charts import _equity_ratio_display
        user_eq_disp = _equity_ratio_display(user_eq) or 0
        bench_eq_disp = _equity_ratio_display(bench_eq) or 0
    except Exception:
        user_eq_disp, bench_eq_disp = user_eq, bench_eq
    op_diff = user_op - bench_op
    eq_diff = user_eq_disp - bench_eq_disp
    coverage_lines = ""
    if coverage_lease is not None:
        coverage_lines += f"- EBITDAカバレッジ（対リース債務・保守値）: {coverage_lease:.2f}倍\n"
    if coverage_bank is not None:
        coverage_lines += f"- EBITDAカバレッジ（対銀行与信・保守値）: {coverage_bank:.2f}倍\n"
    prompt = f"""あなたは熟練したリース審査専門家です。
以下の財務データと審査スコアを読んで、5項目の評価を【必ず下記形式だけ】で答えてください。

【審査対象】
- 業種: {industry_sub}
- 物件: {asset_name}
- 年商: {nenshu:,}万円
- 営業利益: {op_profit:.0f}百万円 / 経常利益: {ord_profit:.0f}百万円 / 当期純利益: {net_income:.0f}百万円
- 総資産: {assets:,}万円 / 純資産: {net_assets:,}万円
- 減価償却費: {dep:.0f}百万円 / EBITDA（営業利益＋減価償却）: {ebitda:.0f}百万円
- リース債務（当社＋関連会社）: {lease_credit:.0f}百万円
- 銀行与信（当社＋関連会社）: {bank_credit:.0f}百万円

【指標比較】
- 営業利益率: {user_op:.1f}%（業界平均 {bench_op:.1f}%、差 {op_diff:+.1f}%）
- 自己資本比率: {user_eq_disp:.1f}%（業界平均 {bench_eq_disp:.1f}%、差 {eq_diff:+.1f}%）
{coverage_lines}
【スコアリング】
- 総合スコア: {score:.1f}% / 業種別スコア: {ind_score:.1f}% / ベンチマークスコア: {bench_score:.1f}%
- 判定: {hantei} / 契約期待度: {contract_prob:.1f}% / 予測利回り: {yield_pred:.2f}%

---
必ず以下の形式のみで回答してください。①〜⑤以外は一切出力しないこと。

①収益性：（業界比較を含む1文で評価）
②財務安定性：（純資産・自己資本比率の評価1文）
③返済余力：（EBITDAとリース・銀行債務のカバレッジ評価1文）
④成約見込み：（スコアと業種を踏まえた成約可能性1文）
⑤総合評価：（審査担当者への推奨アクション。2文以内）
"""
    yield from stream_llm(prompt)


def stream_byoki_with_industry(
    selected_sub: str, user_eq, user_op, comparison_text: str, network_risk_summary: str = ""
) -> Generator[str, None, None]:
    """byoki（AIのぼやき）のストリーミング版。st.write_stream() に渡す。"""
    if not is_ai_available():
        return
    from web_services import get_trend_extended
    from charts import _equity_ratio_display as _eq_disp
    trend_ext = get_trend_extended(selected_sub) or ""
    bench_summary = _build_bench_summary(selected_sub, include_eq=True)
    is_tough = (user_eq is not None and user_eq < 20) or (user_op is not None and user_op < 0)
    context = f"""
【業種】{selected_sub}
【今回の案件】自己資本比率 {_eq_disp(user_eq) or 0:.1f}%, 営業利益率 {user_op or 0:.1f}%
【比較・評価】{comparison_text or "（なし）"}
【ネット検索した業界トレンド・拡充情報】
{trend_ext[:1200] if trend_ext else "（未取得）"}
【ネット検索した業界目安・記事】
{bench_summary}
"""
    if network_risk_summary:
        context += f"\n【業界の倒産トレンド等】\n{network_risk_summary[:600]}\n"
    if is_tough:
        instruction = "上記の業界情報と今回の数値（自己資本比率・利益率が厳しめ）を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、ユーモアたっぷりの毒舌な愚痴を1つ、2〜4文で言ってください。業界平均やネットで見た情報に触れつつぼやいてください。"
    else:
        instruction = "上記の業界情報を踏まえ、有能だが激務で死んだ魚の目をしたベテラン審査員・八奈見杏奈の口調で、業界の現状や審査の苦労について軽く一言、2〜3文でぼやいてください。"
    prompt = f"{AI_HONNE_SYSTEM}\n\n---\n\n【参照する業界・案件情報】\n{context}\n\n---\n\n{instruction}"
    yield from stream_llm(prompt)


# ─── 3D多角分析 AIポジショニングコメント ──────────────────────────────────────

def get_ai_3d_comment(current_data: dict, past_cases: list) -> Optional[str]:
    """
    3D多角分析の過去クラスタとの位置関係を統計計算し、AIに2〜3文のコメントを生成させる。
    クラスタ距離・各次元の差分を数値で渡すことでローカルLLMでも安定した出力を得る。

    Args:
        current_data: 今回案件dict（sales, op_margin, equity_ratio, op_profit,
                       depreciation, lease_credit, bank_credit, score）
        past_cases:   load_all_cases() の結果

    Returns:
        コメント文字列。失敗時は None。
    """
    if not is_ai_available():
        return None

    try:
        from charts import compute_3d_positioning_stats
        stats = compute_3d_positioning_stats(current_data, past_cases)
    except Exception:
        return None

    if not stats:
        return None

    apr_n = stats.get("approved_count", 0)
    rej_n = stats.get("rejected_count", 0)
    closest = stats.get("closest_cluster", "不明")
    ratio = stats.get("closest_distance_ratio", 0.5)
    d_apr = stats.get("dist_to_approved", 0)
    d_rej = stats.get("dist_to_rejected", 0)
    apr_c = stats.get("approved_centroid", {})
    cur = stats.get("current_vals", {})
    diff = stats.get("current_vs_approved", {})

    # 承認済クラスタとの差が大きい次元トップ2を特定
    dim_labels = {
        "利益率(%)": "営業利益率",
        "自己資本比率(%)": "自己資本比率",
        "EBITDAカバレッジ(倍)": "EBITDAカバレッジ",
        "スコア(%)": "審査スコア",
    }
    sorted_dims = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)
    gap_lines = ""
    for dim_key, gap in sorted_dims[:2]:
        label = dim_labels.get(dim_key, dim_key)
        direction = "上回る" if gap > 0 else "下回る"
        unit = "倍" if "カバレッジ" in dim_key else "%"
        gap_lines += (
            f"  ・{label}: 承認済平均{apr_c.get(dim_key, 0):.1f}{unit}に対し"
            f"今回{cur.get(dim_key, 0):.1f}{unit}（差{gap:+.1f}{unit}、承認済を{direction}）\n"
        )

    prox_desc = (
        f"承認済クラスタに近い（類似度目安 {100 - int(ratio * 100)}%）"
        if closest == "承認済"
        else f"否決クラスタに近い（類似度目安 {int(ratio * 100)}%）"
    )

    prompt = (
        f"リース審査の3D多角分析の結果を見て、2〜3文で簡潔にコメントしてください。\n\n"
        f"【分析結果】\n"
        f"- 過去事例: 承認済{apr_n}件 / 否決{rej_n}件\n"
        f"- 今回案件の位置: {prox_desc}\n"
        f"  （承認済クラスタとの距離 {d_apr:.1f} / 否決クラスタとの距離 {d_rej:.1f}）\n"
        f"- 承認済クラスタとの主な差分:\n{gap_lines}"
        f"\n審査担当者向けに、ポジショニングの特徴と注目すべき点を日本語で。"
        f"余計な前置きは不要。"
    )

    try:
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            retries=1,
            timeout_seconds=60,
        )
        content = ((ans.get("message") or {}).get("content") or "").strip()
        if content and "APIキーが" not in content and "エラー:" not in content[:30]:
            return content
        return None
    except Exception:
        return None


def get_ai_consultation_prompt(
    q: str,
    res: dict,
    selected_sub: str,
    jsic_data: dict,
    news_content=None,
    kb_use_faq: bool = True,
    kb_use_cases: bool = True,
    kb_use_manual: bool = True,
    kb_use_industry: bool = True,
    kb_use_improvement: bool = False,
) -> str:
    """
    AI相談チャット用のコンテキスト付きプロンプトを構築して返す。
    業種情報・審査結果・ナレッジベース・過去相談メモを付加する。
    """
    from web_services import get_advice_context_extras
    from knowledge import build_knowledge_context
    from data_cases import load_consultation_memory

    res = res or {}
    score = res.get("score")
    comparison = res.get("comparison", "")
    selected_major = res.get("industry_major", "")

    # 業界情報（Web検索キャッシュ）
    advice_extras = get_advice_context_extras(selected_sub, selected_major) or ""
    advice_block = f"【業界情報・補助金・リース情報】\n{advice_extras[:1000]}" if advice_extras else ""

    # JSIC 業種トレンド
    trend_block = ""
    if jsic_data and selected_major and selected_major in jsic_data:
        trend = jsic_data[selected_major]["sub"].get(selected_sub, "")
        if trend:
            trend_block = f"【業種トレンド】\n{str(trend)[:500]}"

    # ニュース
    news_block = ""
    if news_content:
        title = news_content.get("title", "")
        body = news_content.get("content", "")[:600]
        news_block = f"【参考ニュース: {title}】\n{body}"

    # ナレッジベース（マニュアル・FAQ・事例集）
    kb_block = ""
    if any([kb_use_faq, kb_use_cases, kb_use_manual, kb_use_industry, kb_use_improvement]):
        kb_text = build_knowledge_context(
            query=q,
            industry=selected_sub,
            use_faq=kb_use_faq,
            use_cases=kb_use_cases,
            use_manual=kb_use_manual,
            use_industry_guide=kb_use_industry,
            use_improvement=kb_use_improvement,
            max_tokens_approx=1500,
        )
        if kb_text:
            kb_block = f"【審査マニュアル・FAQ・事例集（参考）】\n{kb_text}"

    # 過去の相談メモ（蓄積コンテキスト）
    memory_block = ""
    try:
        memos = load_consultation_memory()
        if memos:
            recent = memos[-5:]
            lines = [f"Q: {m.get('q', '')[:80]} / A: {m.get('a', '')[:120]}" for m in recent]
            memory_block = "【過去の相談メモ（参考）】\n" + "\n".join(lines)
    except Exception as e:
        log_warning(f"相談メモ読み込み失敗: {e}", context="build_ai_prompt")

    # AnythingLLM 社内知識ベース（RAG） ── 最優先で取得
    anything_block = ""
    try:
        from anything_api import get_anything_llm_context
        anything_block = get_anything_llm_context(q, res, selected_sub) or ""
    except Exception as e:
        log_warning(f"AnythingLLMコンテキスト取得失敗: {e}", context="build_ai_prompt")

    # 審査結果サマリー
    result_block = ""
    if score is not None:
        result_block = f"【審査スコア】{score:.1f}点\n【財務評価】{comparison}"

    # ─── プロンプト組み立て ───────────────────────────────────────────────────
    # AnythingLLM が取得できた場合：社内知識ベースを最優先に置き、明示的な指示を付加する
    # 取得できなかった場合：従来どおり内部ナレッジベース（kb_block）を優先する
    if anything_block:
        system_instruction = (
            "あなたはリース審査のAI審査オフィサーです。\n"
            "【重要】以下の【社内知識ベース（AnythingLLM）】に記載された内容を最優先で参照し、"
            "その内容を根拠に回答してください。\n"
            "社内知識ベースに記載がない部分は、補足情報（業界情報・審査スコア等）を参考にしてください。"
        )
        parts = [
            system_instruction,
            anything_block,          # ← 最優先・先頭配置
            result_block,
            trend_block,
            advice_block,
            news_block,
            kb_block,
            memory_block,
            f"【質問】\n{q}",
        ]
    else:
        parts = [
            "あなたはリース審査のAI審査オフィサーです。以下の情報を参照して、審査担当者の質問に答えてください。",
            result_block,
            trend_block,
            advice_block,
            news_block,
            kb_block,
            memory_block,
            f"【質問】\n{q}",
        ]
    return "\n\n".join(p for p in parts if p)
def get_ai_negotiation_strategy(res: dict, similar_cases: list, lost_stats: dict) -> Optional[str]:
    """
    現在の案件の弱点、類似成約事例、および失注理由を分析し、
    成約に向けた具体的な「交渉シナリオ」を生成する。
    """
    if not is_ai_available():
        return None

    score = res.get("score", 0)
    industry = res.get("industry_sub", "")
    
    # 類似案件から「成約の決め手」を抽出
    success_factors = []
    for sc in similar_cases:
        if sc.get("final_status") == "成約" or sc.get("final_status") == "承認":
            from case_similarity import CaseSimilarityEngine
            engine = CaseSimilarityEngine([])
            conds = engine._analyze_conditions(sc.get("data", {}))
            success_factors.extend(conds)
    
    success_factors = list(set(success_factors)) # 重複排除
    
    # 失注理由
    lost_reasons = lost_stats.get("reasons", {})
    avg_lost_rate = lost_stats.get("avg_competitor_rate")

    prompt = f"""あなたは法人リースのシニア審査役兼、営業戦略アドバイザーです。
以下のデータに基づき、この案件を「否決」から「条件付き承認（成約）」へ引き上げるための【具体的交渉シナリオ】を提案してください。

【現状の案件】
・業種: {industry}
・総合スコア: {score:.1f}点（承認目安: 70点以上）
・主な懸念: {"財務基盤の弱さ（自己資本不足）" if res.get("user_eq", 0) < 15 else "収益性の低さ（赤字・営業益不足）" if res.get("user_op", 0) < 2 else "総合的な信用力不足"}

【過去の成功パターン（類似案件の成約条件）】
{", ".join(success_factors) if success_factors else "特になし（新規パターン開拓が必要）"}

【過去の失敗パターン（同業種の失注理由）】
{", ".join([f"{k}({v}件)" for k, v in lost_reasons.items()]) or "データなし"}
・過去の競合平均金利: {f'{avg_lost_rate:.2f}%' if avg_lost_rate else '不明'}

---
以下の構成で、審査担当者と営業担当者の両方に向けた実戦的なアドバイスを日本語で出力してください。

### 🤝 成約への交渉ロードマップ
1. **フェーズ1：リスク緩和（審査を通すための絶対条件）**
   - 具体的な追加条件（例：実質経営者の個人保証、期間短縮、頭金投入など）とその理由。
2. **フェーズ2：競合対策（失注を防ぐための落とし所）**
   - 金利調整の余地や、回答スピードの重要性など、過去の失注データを踏まえた戦術。
3. **フェーズ3：顧客への説得ロジック**
   - 「なぜこの条件が必要か」「この条件を飲めばどのようなメリットがあるか」を伝えるための対話案。
"""
    try:
        ans = chat_with_retry(
            model=get_ollama_model(),
            messages=[{"role": "user", "content": prompt}],
            timeout_seconds=90
        )
        return (ans.get("message") or {}).get("content", "").strip()
    except Exception as e:
        log_warning(f"交渉戦略生成失敗: {e}", context="get_ai_negotiation_strategy")
        return None
