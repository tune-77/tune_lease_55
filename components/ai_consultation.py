"""
AI相談チャット＆討論モード（📋審査・分析の右カラム）。
render_ai_consultation() を呼び出して col_right に描画する。
"""
import time
import streamlit as st

from ai_chat import (
    GEMINI_API_KEY_ENV,
    GEMINI_MODEL_DEFAULT,
    _chat_result_holder,
    _chat_for_thread,
    _get_gemini_key_from_secrets,
    chat_with_retry,
    get_ai_consultation_prompt,
    get_ollama_model,
    is_ai_available,
    save_debate_log,
)
from data_cases import append_consultation_memory
from web_services import get_advice_context_extras, get_stats, get_trend_extended
from knowledge import build_knowledge_context

# ── 討論モード ペルソナ定義 ──────────────────────────────────────────────────
PERSONA_CON = """あなたは「慎重派（守り）」のベテラン審査部長です。
・財務の欠点、業界リスク、倒産確率の不安を徹底的に突き、厳しい条件を出す立場です。
・発言には必ず【ネット検索結果】または【財務データ】の具体的な数値・事実を引用し、根拠を示してください。一般論のみの主張は禁止です。"""

PERSONA_PRO = """あなたは「推進派（攻め）」の営業担当です。
・企業の情熱・将来性・ネットで見つけた好材料を強調し、前向きな支援を主張する立場です。
・発言には必ず【ネット検索結果】または【財務データ】の具体的な数値・好材料を引用し、根拠を示してください。一般論のみの主張は禁止です。"""

PERSONA_JUDGE = """あなたは「審判（決裁者）」です。
・推進派と慎重派の議論を冷静に総括し、最終的な「承認確率(%)」と「具体的な融資条件」を算出する立場です。
・ネット検索結果や財務データに基づく根拠を踏まえ、両論を引用しつつ結論を出してください。"""

CHAT_LOADING_TIMEOUT = 125  # 秒（API側のタイムアウトより少し長め）

_ERROR_KEYWORDS = (
    "APIキーが設定されていません",
    "Gemini API エラー:",
    "pip install",
    "応答が返りませんでした",
    "安全フィルターでブロック",
)


def _show_ai_error(content: str) -> None:
    """AIエラー内容を st.error で表示するヘルパー。"""
    if content and any(kw in content for kw in _ERROR_KEYWORDS):
        st.error(content)


def _render_tab_chat(selected_sub: str, jsic_data: dict) -> None:
    """相談モード タブの描画。"""
    res = st.session_state.get("last_result", {})

    # ── ナレッジ参照トグル ──
    with st.expander("📚 マニュアル・事例集・FAQをAIに参照させる", expanded=False):
        st.caption("有効にすると「審査マニュアル」「業種別ガイド」「FAQ集」「事例集」の内容がAIへの質問に自動的に付加されます。")
        st.checkbox("審査マニュアル・スコアリング基準", value=True, key="kb_use_manual")
        st.checkbox("業種別ガイド（財務目安・審査ポイント）", value=True, key="kb_use_industry")
        st.checkbox("FAQ集（よくある質問と回答）", value=True, key="kb_use_faq")
        st.checkbox("審査事例集（Bランク・Cランク・Dランクの実例）", value=True, key="kb_use_cases")
        st.checkbox("スコア改善ガイド（短期・中期の改善アクション）", value=False, key="kb_use_improvement")

    # ── 音声入力URLパラメータ反映 ──
    if st.query_params.get("voice_text"):
        st.session_state["consultation_input"] = st.query_params.get("voice_text", "")
        try:
            st.experimental_set_query_params()
        except Exception:
            pass
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "consultation_input" not in st.session_state:
        st.session_state["consultation_input"] = ""
    # 送信済みなら入力欄を空に（text_area作成前にのみ session_state を変更可能）
    if "consultation_pending_q" in st.session_state:
        st.session_state["consultation_input"] = ""

    chat_box = st.container(height=400)
    with chat_box:
        for m in st.session_state.messages:
            if m["role"] != "system":
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

    # ── バックグラウンド応答待ちポーリング ──
    if _chat_result_holder["done"]:
        result = _chat_result_holder["result"]
        _chat_result_holder["result"] = None
        _chat_result_holder["done"] = False
        st.session_state["chat_result"] = result
        st.session_state["chat_loading"] = False
        if st.session_state.get("ai_engine") == "gemini" and result:
            c = (result.get("message") or {}).get("content", "")
            st.session_state["last_gemini_debug"] = (
                "OK" if c and "APIキーが" not in c and "Gemini API エラー:" not in c
                else ((c[:200] + "...") if len(c or "") > 200 else (c or "（空）"))
            )

    chat_loading = st.session_state.get("chat_loading", False)
    chat_result = st.session_state.get("chat_result")

    # タイムアウト強制解除
    loading_started = st.session_state.get("chat_loading_started_at")
    if chat_loading and loading_started is not None and (time.time() - loading_started) > CHAT_LOADING_TIMEOUT:
        st.session_state["chat_loading"] = False
        _chat_result_holder["done"] = True
        _chat_result_holder["result"] = {"message": {"content": "応答がタイムアウトしました（約2分）。\n\n・APIキー・ネット接続を確認するか、もう一度送信してください。\n・Gemini の場合は無料枠の制限に達している可能性もあります。"}}
        st.rerun()

    if chat_loading or chat_result is not None:
        with chat_box:
            for m in st.session_state.messages:
                if m["role"] != "system":
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
            with st.chat_message("assistant"):
                if chat_result is not None:
                    content = (chat_result.get("message") or {}).get("content", "")
                    _show_ai_error(content)
                    st.markdown(content or "（応答がありませんでした）")
                    st.session_state.messages.append({"role": "assistant", "content": content or "（応答がありませんでした）"})
                    # ホルダー経由の応答も相談メモに保存
                    user_msgs = [m["content"] for m in st.session_state.messages if m.get("role") == "user"]
                    if user_msgs:
                        append_consultation_memory(user_msgs[-1], content or "（応答がありませんでした）")
                    st.session_state["chat_loading"] = False
                    st.session_state["chat_result"] = None
                else:
                    with st.status("思考中...", state="running", expanded=True):
                        st.markdown("⏳ 応答を待っています...")
                        if st.button("待機をやめる", key="chat_cancel_loading"):
                            st.session_state["chat_loading"] = False
                            _chat_result_holder["done"] = True
                            _chat_result_holder["result"] = {"message": {"content": "待機を解除しました。もう一度送信するか、APIキー・ネット接続を確認してください。"}}
                            st.rerun()
                    time.sleep(1)
                    st.rerun()

    # ── 入力欄 + 音声ボタン + 送信 ──
    st.text_area(
        "相談内容",
        value=st.session_state.get("consultation_input", ""),
        key="consultation_input",
        height=100,
        placeholder="相談する内容を入力...（下の🎤で音声入力もできます）",
        label_visibility="collapsed",
    )
    voice_html = """
    <script>
    function startVoiceInput() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            alert('お使いのブラウザは音声入力に対応していません。Chrome などでお試しください。');
            return;
        }
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        var rec = new SpeechRecognition();
        rec.lang = 'ja-JP';
        rec.continuous = false;
        rec.interimResults = false;
        rec.onresult = function(e) {
            var t = e.results[0][0].transcript;
            var u = window.parent.location.pathname + '?voice_text=' + encodeURIComponent(t);
            window.parent.location = u;
        };
        rec.onerror = function(e) {
            if (e.error === 'not-allowed') alert('マイクの利用が許可されていません。');
            else alert('音声認識エラー: ' + e.error);
        };
        rec.start();
    }
    </script>
    <button type="button" onclick="startVoiceInput()" style="padding: 8px 16px; font-size: 1rem; cursor: pointer; border-radius: 8px; background: #f0f2f6; border: 1px solid #ccc;">
    🎤 音声入力
    </button>
    """
    btn_col1, btn_col2 = st.columns([1, 3])
    with btn_col1:
        st.components.v1.html(voice_html, height=50)
    with btn_col2:
        send_clicked = st.button("送信", key="consultation_send", type="primary")

    if send_clicked and (st.session_state.get("consultation_input") or "").strip():
        st.session_state["consultation_pending_q"] = (st.session_state.get("consultation_input") or "").strip()
        st.rerun()

    q = None
    if st.session_state.get("consultation_pending_q"):
        q = st.session_state.pop("consultation_pending_q")
        st.session_state.messages.append({"role": "user", "content": q})

    if q:
        with chat_box:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                if not is_ai_available():
                    if st.session_state.get("ai_engine") == "gemini":
                        st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                    else:
                        st.error("AIサーバー（Ollama）が起動していません。\nターミナルで `ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                else:
                    with st.spinner("業種別トピックス等を取得中..."):
                        context_prompt = get_ai_consultation_prompt(
                            q=q,
                            res=res,
                            selected_sub=selected_sub,
                            jsic_data=jsic_data,
                            news_content=st.session_state.get("selected_news_content"),
                            kb_use_faq=st.session_state.get("kb_use_faq", True),
                            kb_use_cases=st.session_state.get("kb_use_cases", True),
                            kb_use_manual=st.session_state.get("kb_use_manual", True),
                            kb_use_industry=st.session_state.get("kb_use_industry", True),
                            kb_use_improvement=st.session_state.get("kb_use_improvement", False),
                        )
                    _engine = st.session_state.get("ai_engine", "ollama")
                    _model = get_ollama_model()
                    _api_key = (
                        (st.session_state.get("gemini_api_key") or "").strip()
                        or GEMINI_API_KEY_ENV
                        or _get_gemini_key_from_secrets()
                    )
                    _gemini_model = st.session_state.get("gemini_model", GEMINI_MODEL_DEFAULT)
                    with st.spinner("思考中..."):
                        ans = _chat_for_thread(
                            _engine, _model,
                            [{"role": "user", "content": context_prompt}],
                            timeout_seconds=120,
                            api_key=_api_key,
                            gemini_model=_gemini_model,
                        )
                    content = (ans.get("message") or {}).get("content", "") or "（応答がありませんでした）"
                    _show_ai_error(content)
                    if not any(kw in content for kw in _ERROR_KEYWORDS):
                        st.markdown(content)
                    st.session_state.messages.append({"role": "assistant", "content": content})
                    append_consultation_memory(q, content)
                    if st.session_state.get("ai_engine") == "gemini":
                        if content and "APIキーが" not in content and "Gemini API エラー:" not in content:
                            st.session_state["last_gemini_debug"] = "OK"
                        else:
                            st.session_state["last_gemini_debug"] = (
                                (content[:200] + "...") if len(content or "") > 200 else (content or "（空）")
                            )


def _render_tab_debate(selected_sub: str, jsic_data: dict, bankruptcy_data: list) -> None:
    """討論モード タブの描画。"""
    st.info("審査委員会モード：慎重派・推進派・審判の3ペルソナでディベートし、最終決裁を出します。")
    if "debate_history" not in st.session_state:
        st.session_state.debate_history = []

    # 議論ログの表示
    for m in st.session_state.debate_history:
        avatar = "🙆‍♂️" if m["role"] == "Pro" else "🙅‍♂️"
        if m["role"] == "User":
            avatar = "👤"
        role_name = "推進派" if m["role"] == "Pro" else ("慎重派" if m["role"] == "Con" else "あなた")
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(f"**{role_name}**: {m['content']}")

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⚔️ 議論を開始 / 進行 (1ターン進める)", use_container_width=True):
            if "last_result" not in st.session_state:
                st.error("先に審査を実行してください。")
            else:
                res = st.session_state["last_result"]
                selected_major = res.get("industry_major", "D 建設業")
                selected_sub_local = res.get("industry_sub", "06 総合工事業")
                comparison_text = res.get("comparison", "")
                trend_info = ""
                if jsic_data and selected_major in jsic_data:
                    trend_info = jsic_data[selected_major]["sub"].get(selected_sub_local, "")
                trend_extended_d = get_trend_extended(selected_sub_local)
                if trend_extended_d:
                    trend_info = (trend_info or "") + "\n\n【拡充】\n" + trend_extended_d[:1500]

                score = res["score"]
                risk_context = ""
                for b in bankruptcy_data:
                    risk_context += f"- {b['type']}: {b['signal']} ({b['check_point']})\n"
                if "debate_history" not in st.session_state:
                    st.session_state.debate_history = []
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.debate_history])

                news_context = ""
                if "selected_news_content" in st.session_state:
                    news = st.session_state.selected_news_content
                    news_context = f"\n\n【参考ニュース記事: {news['title']}】\n{news['content']}"

                advice_extras_debate = get_advice_context_extras(selected_sub_local, selected_major)
                advice_debate_block = ("補助金・リース・業界拡充: " + advice_extras_debate[:800]) if advice_extras_debate else ""
                _debate_kb = build_knowledge_context(
                    query=f"{selected_sub_local} スコア{res.get('score', 0):.0f}",
                    industry=selected_sub_local,
                    use_faq=True,
                    use_cases=True,
                    use_manual=True,
                    use_industry_guide=True,
                    use_improvement=False,
                    max_tokens_approx=1500,
                )
                _debate_kb_block = f"\n【審査マニュアル・FAQ・事例集（参考）】\n{_debate_kb}" if _debate_kb else ""

                # ロール決定 & プロンプト作成
                if not st.session_state.debate_history:
                    next_role = "Pro"
                    from bayesian_engine import THRESHOLD_APPROVAL
                    approval_line = THRESHOLD_APPROVAL * 100
                    prompt = f"""{PERSONA_PRO}

【財務データ】（必ず引用すること）
業種: {selected_sub_local}
スコア: {score:.1f}点 (承認ライン{approval_line:.0f}点)
財務評価: {comparison_text}

【ネット検索結果・業界材料】
{advice_debate_block}
{news_context if news_context else "（ニュース未読み込み）"}
{_debate_kb_block}

【指示】
- 上記の「財務データ」と「ネット検索結果」のいずれかから必ず1つ以上具体的に引用し、根拠を示したうえで主張すること。
- FAQや事例集に類似ケースがあれば引用してよい。
- 企業の情熱・将来性・好材料を強調し、前向きな支援を主張せよ。
- 140文字以内。
"""
                else:
                    last_role = st.session_state.debate_history[-1]["role"]
                    if last_role == "User":
                        prev_ai = "Con"
                        for m in reversed(st.session_state.debate_history[:-1]):
                            if m["role"] in ["Pro", "Con"]:
                                prev_ai = m["role"]
                                break
                        next_role = "Con" if prev_ai == "Pro" else "Pro"
                    else:
                        next_role = "Con" if last_role == "Pro" else "Pro"

                    if next_role == "Con":
                        advice_con_block = ("【補助金・リース判定等】" + advice_extras_debate[:500]) if advice_extras_debate else ""
                        prompt = f"""{PERSONA_CON}

【財務データ・リスク指標】（必ず引用すること）
スコア: {score:.1f}点、財務評価: {comparison_text}
【倒産リスクDB】
{risk_context}

【ネット検索結果・業界リスク】
{news_context if news_context else "（なし）"}
{advice_con_block}

【これまでの議論】
{history_text}

【指示】
- 上記の「財務データ」または「ネット検索結果」から必ず1つ以上具体的に引用し、根拠を示したうえで反論すること。
- 財務の欠点・業界リスク・倒産確率の不安を突き、厳しい条件を出せ。
- 140文字以内。
"""
                    else:  # Pro
                        advice_pro_block = ("【補助金・リース等】" + advice_extras_debate[:500]) if advice_extras_debate else ""
                        prompt = f"""{PERSONA_PRO}

【財務データ】（必ず引用すること）
財務評価: {comparison_text}
スコア: {score:.1f}点

【ネット検索結果・好材料】
{news_context if news_context else "業界の成長性、社長の覚悟"}
{advice_pro_block}

【これまでの議論】
{history_text}

【指示】
- 上記の「財務データ」または「ネット検索結果」から必ず1つ以上具体的に引用し、根拠を示したうえで慎重派に反論せよ。
- 企業の情熱・将来性・好材料を強調し、前向きな支援を主張せよ。
- 140文字以内。
"""

                if not is_ai_available():
                    if st.session_state.get("ai_engine") == "gemini":
                        st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                    else:
                        st.error("AIサーバー（Ollama）が起動していません。\nターミナルで `ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                else:
                    with st.spinner(f"{next_role}が思考中..."):
                        try:
                            ans = chat_with_retry(
                                model=get_ollama_model(),
                                messages=[{"role": "user", "content": prompt}],
                                retries=1,
                                timeout_seconds=120,
                            )
                            if not ans or "message" not in ans:
                                st.error("AIからの応答が不正です。")
                            else:
                                msg_content = ans["message"]["content"]
                                if msg_content and any(kw in msg_content for kw in _ERROR_KEYWORDS):
                                    st.error(msg_content)
                                st.session_state.debate_history.append({"role": next_role, "content": msg_content})
                        except Exception as e:
                            st.error(f"AIエラー詳細: {e}")
                    st.rerun()

    # 終了判定ボタン（審判ペルソナで決裁）
    with col_btn2:
        if len(st.session_state.debate_history) >= 4:
            res_judge = st.session_state.get("last_result") or {}
            selected_sub_judge = res_judge.get("industry_sub", "")
            if st.button("🏁 議論終了・判定", type="primary", use_container_width=True):
                with st.spinner("審判が決裁中..."):
                    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.debate_history])
                    pd_val = res_judge.get("pd_percent")
                    net_risk = res_judge.get("network_risk_summary", "")
                    pd_str = f"{pd_val:.1f}%" if pd_val is not None else "（未算出）"
                    comparison_judge = res_judge.get("comparison", "")
                    similar_block = res_judge.get("similar_past_cases_prompt", "") or ""
                    judge_prompt = ""
                    if similar_block:
                        judge_prompt += similar_block
                    past_stats_judge = get_stats(selected_sub_judge)
                    if past_stats_judge.get("top_competitors_lost") or (
                        past_stats_judge.get("avg_winning_rate") is not None
                        and past_stats_judge.get("avg_winning_rate", 0) > 0
                    ):
                        judge_prompt += "\n【過去の競合・成約金利】\n"
                        if past_stats_judge.get("top_competitors_lost"):
                            judge_prompt += "よく負ける競合: " + "、".join(past_stats_judge["top_competitors_lost"][:5]) + "\n"
                        if past_stats_judge.get("avg_winning_rate") and past_stats_judge["avg_winning_rate"] > 0:
                            judge_prompt += f"同業種の平均成約金利: {past_stats_judge['avg_winning_rate']:.2f}%\n"
                        judge_prompt += "上記を踏まえ、融資条件には競合に勝つための対策も反映してください。\n\n"
                    judge_prompt += f"""{PERSONA_JUDGE}

【財務データ】（根拠として引用すること）
財務評価: {comparison_judge}

【ネット検索結果】
【業界の最新リスク情報】
{net_risk if net_risk else "（未取得）"}

【議論ログ（推進派・慎重派の発言）】
{history_text}

【指示】
- 上記の財務データとネット検索結果を根拠に、推進派と慎重派の議論を冷静に総括してください。
- 最終的な「承認確率(%)」と「具体的な融資条件」を算出し、理由を簡潔に述べてください。

出力形式（必ず守ること）:
承認確率: XX%
融資条件: （金利・担保・期間など具体的に）
理由: (80文字以内)
"""
                    if not is_ai_available():
                        if st.session_state.get("ai_engine") == "gemini":
                            st.error("Gemini APIキーを設定してください。サイドバー「AIモデル設定」で入力するか、環境変数 GEMINI_API_KEY を設定してください。")
                        else:
                            st.error("Ollama が起動していません。`ollama serve` を実行するか、サイドバーで「Gemini API」に切り替えてください。")
                    else:
                        ans = chat_with_retry(
                            model=get_ollama_model(),
                            messages=[{"role": "user", "content": judge_prompt}],
                            retries=1,
                            timeout_seconds=120,
                        )
                        result_text = ans["message"]["content"]
                        st.success("✅ **ディベート結果**")
                        st.write(result_text)
                        save_debate_log({
                            "industry": selected_sub_judge,
                            "history": st.session_state.debate_history,
                            "result": result_text,
                        })

    # ユーザー介入（チャット入力）
    if user_input := st.chat_input("議論に介入する（回答・指示）", key="debate_input"):
        st.session_state.debate_history.append({"role": "User", "content": user_input})
        st.rerun()


def render_ai_consultation(selected_sub: str, jsic_data: dict, bankruptcy_data: list) -> None:
    """
    AI審査オフィサー相談 + 討論モードを描画する。
    📋審査・分析モードの col_right から呼び出す。
    """
    # last_result で業種を上書き（審査実行後に同期）
    if "last_result" in st.session_state:
        selected_sub = st.session_state["last_result"].get("industry_sub", selected_sub)

    st.header("💬 AI審査オフィサーに相談")
    st.caption(f"選択中の業種: {selected_sub}")

    tab_chat, tab_debate = st.tabs(["相談モード", "⚔️ 討論モード"])

    # AIエンジン・APIキー状態の表示
    _engine = st.session_state.get("ai_engine", "ollama")
    if _engine == "gemini":
        _key_ok = bool(
            (st.session_state.get("gemini_api_key") or "").strip()
            or GEMINI_API_KEY_ENV
            or _get_gemini_key_from_secrets()
        )
        st.caption(f"🤖 使用中: **Gemini API**　｜　APIキー: **{'設定済み' if _key_ok else '未設定（サイドバーで入力）'}**")
        with st.expander("🔧 Gemini デバッグ（動かないときに開く）", expanded=False):
            _dbg = st.session_state.get("last_gemini_debug", "まだ呼び出していません")
            st.text(_dbg)
            st.caption("相談で送信後、ここに「OK」またはエラー内容が表示されます。")
    else:
        st.caption("🤖 使用中: **Ollama（ローカル）**")

    with tab_chat:
        _render_tab_chat(selected_sub, jsic_data)

    with tab_debate:
        _render_tab_debate(selected_sub, jsic_data, bankruptcy_data)

    st.divider()
