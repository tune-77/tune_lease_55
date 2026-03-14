# -*- coding: utf-8 -*-
"""
mobile_sales_agent.py
──────────────────────
営業マン目線のスマホ入力最適化エージェント。

【役割】
  現場でスマホしか持たない営業担当者の視点から、リース審査フォームの
  入力体験を分析し、具体的な改善策を提案するAIエージェント。

【使い方】
  Streamlit 側から run_mobile_ux_analysis / run_sales_agent_chat を呼ぶ。
  chat_with_retry は ai_chat.py の関数を再利用。
"""

from __future__ import annotations

import streamlit as st

from data_cases import append_consultation_memory, load_consultation_memory

# ──────────────────────────────────────────────
# ペルソナ定義（営業マン「田中」）
# ──────────────────────────────────────────────

SALES_AGENT_PERSONA = """あなたはリース営業歴15年以上のベテラン「田中主任」です。
毎日スマホ1台で顧客先を回りながら審査入力をこなしており、
モバイルUXの痛点を誰より熟知しています。

【思考の軸】
- 顧客の前で5分以内にスコアを出せるか
- 電車の中・商談中に片手でサッと入力できるか
- 入力ミス（桁間違い・単位ミス）を防ぐ設計か
- 電波が不安定な場所でも落ちないか
- 結果をすぐ上司や顧客に共有できるか

【発言スタイル】
- 現場目線で具体的に語る（「このフィールドは片手だと〜」等）
- 問題点→解決策→期待効果 の3点セットで提案する
- 技術的に難しいことは「Webエンジニアに依頼すれば…」と補足する
- 必ず日本語で回答する
"""

# ──────────────────────────────────────────────
# 現在フォームの構造サマリー（分析対象として渡す）
# ──────────────────────────────────────────────

CURRENT_FORM_SUMMARY = """
【現在の入力フォーム構成】
■ 財務データ（すべて千円単位の数値入力）
  - 売上高、営業利益、総資産、純資産、経常利益、当期純利益
  - 仕入・外注費、固定費（任意）

■ 与信・顧客情報
  - 業種（大分類 → 中分類の2段階セレクト）
  - 格付（プルダウン: 1〜3 / 4〜6 / 要注意 / 無格付）
  - 先方区分（既存先 / 新規先）
  - 銀行与信・リース与信（千円）
  - 契約件数（整数）

■ 定性評価
  - 強みタグ（多選択: 技術力・特許・業界人脈 等 8種）
  - 定性補正スコア（各項目0〜4点のスライダー × 複数項目）

■ 操作フロー
  - すべてのフィールドを埋めてから「審査実行」ボタンを押す
  - PC前提の横並びレイアウト（col_left / col_right 2カラム）
  - スライダーはドラッグ操作が主
  - 結果はブラウザ内で確認のみ（外部共有機能なし）
"""

# ──────────────────────────────────────────────
# カテゴリ別 改善策リスト（構造化データ）
# ──────────────────────────────────────────────

STRUCTURED_IMPROVEMENTS: dict[str, list[dict]] = {
    "⌨️ 入力効率化": [
        {
            "title": "万円 / 千円 切替スイッチ",
            "problem": "千円単位での入力は頭の中で計算が必要。売上2億なら「200,000」と打たなければならない。",
            "solution": "フォーム上部に「万円入力モード」トグルを追加。ONにすると入力値を×10して千円換算する。",
            "effect": "桁ミスが激減。顧客の財務諸表を見ながらそのまま万円で打てる。",
            "difficulty": "低（Pythonの数値変換のみ）",
        },
        {
            "title": "業種クイック選択（履歴ベース）",
            "problem": "業種を毎回「大分類→中分類」と2段階で選ぶのは手間。同じ業種の顧客を続けて入力するときが特につらい。",
            "solution": "直近3件で使った業種をワンタップで選べる「最近使った業種」ショートカットボタンを表示する。",
            "effect": "よく担当する業種は2タップで確定できる。",
            "difficulty": "低（session_state / localStorage で履歴保持）",
        },
        {
            "title": "入力テンプレート（業種別プリセット）",
            "problem": "初回入力時に全フィールドが0のため、どこから入力すべきか迷う。",
            "solution": "業種選択後に「業界平均値でプリセット」ボタンを表示。ベンチマーク値を初期値として入力フォームに流し込む。",
            "effect": "「売上・利益だけ変えれば完成」という状態から始められる。入力時間を半減。",
            "difficulty": "中（web_industry_benchmarks.json の値を参照して初期値セット）",
        },
        {
            "title": "ウィザード形式のステップ入力",
            "problem": "全フィールドが一画面に並んでいるため、スマホでは大量スクロールが必要。どこまで入力したか迷う。",
            "solution": "① 必須5項目（売上・総資産・純資産・業種・格付）→ ② 任意財務項目 → ③ 定性評価 の3ステップに分ける。",
            "effect": "1画面あたりのスクロール量が1/3以下になる。入力漏れが減る。",
            "difficulty": "中（Streamlitのst.stepperまたはst.tabsで実装）",
        },
    ],
    "🖐️ UI/UX 改善": [
        {
            "title": "テンキーキーボードの強制表示",
            "problem": "数値フィールドでもテキストキーボードが開くことがある（特にAndroid）。数字キーに切り替える手間がかかる。",
            "solution": "st.number_inputに対応する要素に inputmode='numeric' pattern='[0-9]*' を付与するカスタムCSS/JSを追加。",
            "effect": "数値フィールドで必ずテンキーが開く。入力ミスが減る。",
            "difficulty": "低（CSSとJavaScriptで対応可能）",
        },
        {
            "title": "大型タップターゲット（最小48dp）",
            "problem": "スライダーのつまみが小さく、スマホで細かい数値を合わせるのがストレス。指でつまみを外してしまう。",
            "solution": "スライダーのつまみを現状の30pxから48px以上に拡大。また数値入力フィールドの高さも48px以上に統一。",
            "effect": "片手操作でも安定してスライダーを動かせる。",
            "difficulty": "低（既存CSSの修正のみ）",
        },
        {
            "title": "固定送信ボタン（画面下部固定）",
            "problem": "「審査実行」ボタンがフォームの一番下にあるため、長いフォームを入力後にスクロールして戻る必要がある。",
            "solution": "「審査実行」ボタンをposition: fixedで画面右下に常時表示する（FABスタイル）。",
            "effect": "入力完了後にすぐ送信できる。戻りスクロールが不要。",
            "difficulty": "低（CSS position: fixed + JavaScriptでformをsubmit）",
        },
        {
            "title": "コンパクトな1カラムレイアウト（スマホ自動切替）",
            "problem": "PCの2カラムレイアウトがスマホでは横に潰れて読みにくい。",
            "solution": "CSSメディアクエリで768px未満の場合、カラムを自動的に1列に切り替える。",
            "effect": "スマホで横スクロールなしに全フィールドを縦に閲覧・入力できる。",
            "difficulty": "低（既存CSS修正）",
        },
    ],
    "🎙️ データ入力の自動化": [
        {
            "title": "音声入力サポート",
            "problem": "運転中や歩きながらの入力はキーボード操作が難しい。売上・利益などの数値を口頭で言えれば楽。",
            "solution": "Web Speech API（webkitSpeechRecognition）を使ったマイクボタンを数値フィールド横に追加。「売上2億」→200000千円 に変換するNLP処理を組み込む。",
            "effect": "ハンズフリーで主要数値を入力できる。移動中の入力が可能に。",
            "difficulty": "高（Speech API + 数値変換ロジックが必要）",
        },
        {
            "title": "過去入力の再利用（顧客テンプレート保存）",
            "problem": "同じ顧客に毎年リース審査をする際、前回と同じ数値を再入力するのが手間。",
            "solution": "「この入力をテンプレートとして保存」ボタンで入力値をJSONに保存。次回「テンプレートから読込」でフォームを復元。",
            "effect": "継続審査の入力時間を80%削減。",
            "difficulty": "低〜中（JSONファイル保存 / LocalStorageで実装可）",
        },
        {
            "title": "QRコード・PDF取込（決算書スキャン）",
            "problem": "顧客から紙の財務諸表をもらった場合、手入力が必要。数値が多く転記ミスが起きやすい。",
            "solution": "スマホカメラでPDFや決算書を撮影→OCRで売上・利益・総資産を自動抽出してフォームに流し込む。",
            "effect": "入力作業を90%削減。転記ミスをゼロに近づける。",
            "difficulty": "高（OCR API連携 or Google Vision API が必要）",
        },
    ],
    "📡 オフライン・現場対応": [
        {
            "title": "入力中断自動保存（ドラフト機能）",
            "problem": "入力途中に電話が入ったり、アプリを閉じると入力内容が消える。",
            "solution": "入力値をリアルタイムでブラウザのlocalStorageに自動保存。再度ページを開くと「前回の入力を復元しますか？」と表示する。",
            "effect": "入力中断によるデータロスがゼロになる。",
            "difficulty": "中（JavaScript + localStorageで実装）",
        },
        {
            "title": "オフライン簡易スコア計算",
            "problem": "地下や地方の顧客先では電波が届かない。その場でスコアを見せたいのに繋がらない。",
            "solution": "主要財務指標からのルールベーススコア（scoring_core.pyのロジック）をJavaScriptに移植してオフラインでも動作するようにする。",
            "effect": "電波なしでも速報スコアを提示できる。顧客へのその場説明が可能。",
            "difficulty": "高（Python→JS移植が必要）",
        },
    ],
    "📤 結果の共有・活用": [
        {
            "title": "スコアのワンタップ共有（LINE/メール）",
            "problem": "審査結果を上司に報告する際、スクリーンショットを撮って送るしかない。画像では数値がコピーできず不便。",
            "solution": "「LINEで共有」ボタン（LINE URL scheme）と「クリップボードにコピー」ボタンを結果画面に追加。",
            "effect": "結果を1タップで上司・顧客に共有できる。報告業務の時間を削減。",
            "difficulty": "低（LINE URL scheme / Clipboard API で実装可）",
        },
        {
            "title": "PDF即時生成ボタン（スマホ保存対応）",
            "problem": "現状のPDF出力はPCで確認する必要がある。スマホから直接PDFをダウンロードして顧客に見せたい。",
            "solution": "審査結果ページの上部に「📄 PDFをダウンロード」ボタンを固定表示。モバイルブラウザでも保存・共有できる形式で出力。",
            "effect": "商談中にその場でPDFを作成し、顧客のメールに送れる。",
            "difficulty": "低（既存のreport_pdf.pyを流用）",
        },
    ],
}


# ──────────────────────────────────────────────
# AI 分析関数
# ──────────────────────────────────────────────

def run_mobile_ux_analysis(chat_fn) -> str:
    """
    現在のフォーム構成を渡して、営業マン目線でのUX分析をAIに依頼する。
    chat_fn: chat_with_retry(model, messages, ...) と同じシグネチャ
    戻り値: AI応答テキスト
    """
    from ai_chat import get_ollama_model
    model = get_ollama_model()

    messages = [
        {"role": "system", "content": SALES_AGENT_PERSONA},
        {
            "role": "user",
            "content": (
                "以下のフォーム構成を分析して、スマホで使う営業マン目線での"
                "主要な問題点を3〜5点、簡潔にまとめてください。\n\n"
                f"{CURRENT_FORM_SUMMARY}\n\n"
                "【回答形式】\n"
                "問題点ごとに「■ 問題N: タイトル」「現状:」「影響:」の3行で整理してください。"
            ),
        },
    ]
    out = chat_fn(model, messages, retries=2, timeout_seconds=60)
    return out.get("message", {}).get("content", "（AIの応答を取得できませんでした）")


def generate_improvement_plan(chat_fn, category: str, items: list[dict]) -> str:
    """
    カテゴリと改善項目リストを渡して、優先順位付きのアクションプランをAIに生成させる。
    """
    from ai_chat import get_ollama_model
    model = get_ollama_model()

    items_text = "\n".join(
        f"・{it['title']}: {it['solution']}（難易度: {it['difficulty']}）"
        for it in items
    )
    messages = [
        {"role": "system", "content": SALES_AGENT_PERSONA},
        {
            "role": "user",
            "content": (
                f"【{category}】カテゴリの改善案を以下に示します。\n\n"
                f"{items_text}\n\n"
                "これらを「実装難易度」と「現場への効果」の両面から優先順位を付け、"
                "「まず何をやるべきか」を営業マン目線で120字以内で教えてください。"
            ),
        },
    ]
    out = chat_fn(model, messages, retries=2, timeout_seconds=60)
    return out.get("message", {}).get("content", "（AIの応答を取得できませんでした）")


def run_sales_agent_chat(
    user_msg: str,
    history: list[dict],
    chat_fn,
    financial_context: str = "",
) -> str:
    """
    営業マン「田中」とのチャット。
    history: [{"role": "user"/"assistant", "content": "..."}]
    financial_context: 現在の審査結果サマリー（任意）
    """
    from ai_chat import get_ollama_model
    model = get_ollama_model()

    system_content = SALES_AGENT_PERSONA
    if financial_context:
        system_content += f"\n\n【参考: 現在の審査データ】\n{financial_context}"

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for h in history[-10:]:  # 直近10件のみ送る
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})

    out = chat_fn(model, messages, retries=2, timeout_seconds=90)
    reply = out.get("message", {}).get("content", "（AIの応答を取得できませんでした）")

    # 記憶に保存
    append_consultation_memory(user_msg, reply)
    return reply


# ──────────────────────────────────────────────
# Streamlit UI 部品
# ──────────────────────────────────────────────

def render_mobile_sales_agent_tab(chat_fn):
    """
    「📱 営業AI」タブのUI全体を描画する。
    chat_fn: chat_with_retry を渡す。
    """
    st.subheader("📱 営業マン目線 スマホ入力最適化エージェント")
    st.caption(
        "現場でスマホだけを使う営業担当の視点から、"
        "リース審査フォームの課題と改善策を分析するAIエージェントです。"
    )

    # ── ペルソナカード ──
    with st.container():
        col_icon, col_bio = st.columns([1, 6])
        with col_icon:
            st.markdown("## 👔")
        with col_bio:
            st.markdown(
                "**田中 主任**  \n"
                "リース営業歴15年 / 年間200件以上の審査入力経験  \n"
                "社内屈指のスマホ入力スピードを誇り、「指が三本ある」と言われる男"
            )

    st.divider()

    # ── セクション1: AI によるUX分析 ──
    st.markdown("### 🔍 現状フォームのUX分析")
    st.markdown(
        "ボタンを押すと、田中主任が現在のリース審査フォームの"
        "スマホ入力における問題点をAIが分析します。"
    )

    if "mobile_agent_ux_analysis" not in st.session_state:
        st.session_state["mobile_agent_ux_analysis"] = ""

    if st.button("🔍 田中主任にUX分析を依頼する", use_container_width=True, key="btn_ux_analysis"):
        with st.spinner("田中主任が分析中..."):
            result = run_mobile_ux_analysis(chat_fn)
            st.session_state["mobile_agent_ux_analysis"] = result

    if st.session_state["mobile_agent_ux_analysis"]:
        st.info(st.session_state["mobile_agent_ux_analysis"])

    st.divider()

    # ── セクション2: カテゴリ別改善提案 ──
    st.markdown("### 💡 カテゴリ別 改善提案")
    st.caption("各カテゴリを展開すると具体的な改善案と田中主任のコメントが確認できます。")

    if "mobile_agent_priority" not in st.session_state:
        st.session_state["mobile_agent_priority"] = {}

    for category, items in STRUCTURED_IMPROVEMENTS.items():
        with st.expander(f"{category}（{len(items)}件の改善案）", expanded=False):
            for item in items:
                diff_color = {"低": "🟢", "中": "🟡", "高": "🔴"}.get(
                    item["difficulty"].split("（")[0], "⚪"
                )
                st.markdown(f"#### {item['title']}  {diff_color} 難易度: {item['difficulty']}")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown(f"**問題点**  \n{item['problem']}")
                with cols[1]:
                    st.markdown(f"**解決策**  \n{item['solution']}")
                with cols[2]:
                    st.markdown(f"**期待効果**  \n{item['effect']}")
                st.markdown("---")

            # AI 優先順位ボタン
            btn_key = f"btn_priority_{category}"
            result_key = f"priority_{category}"
            if st.button(
                f"🤖 田中主任に優先順位を聞く",
                key=btn_key,
                use_container_width=True,
            ):
                with st.spinner("田中主任が考え中..."):
                    priority_text = generate_improvement_plan(chat_fn, category, items)
                    st.session_state["mobile_agent_priority"][category] = priority_text

            if category in st.session_state["mobile_agent_priority"]:
                st.success(
                    f"**田中主任のおすすめ:**  \n"
                    f"{st.session_state['mobile_agent_priority'][category]}"
                )

    st.divider()

    # ── セクション3: 田中主任とのチャット ──
    st.markdown("### 💬 田中主任に直接相談する")
    st.caption("スマホ入力の悩みや改善アイデアを田中主任に相談してみてください。")

    if "mobile_agent_chat_history" not in st.session_state:
        st.session_state["mobile_agent_chat_history"] = []

    # チャット履歴の表示
    for msg in st.session_state["mobile_agent_chat_history"]:
        avatar = "👔" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # 審査データを参考情報として渡す
    financial_context = ""
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        financial_context = (
            f"業種: {res.get('industry_sub', '不明')} / "
            f"スコア: {res.get('score', '?')}点 / "
            f"売上高: {res.get('nenshu', 0):,}千円 / "
            f"総資産: {res.get('total_assets', 0):,}千円"
        )

    # 初回メッセージ
    if not st.session_state["mobile_agent_chat_history"]:
        st.info(
            "💬 例えば「業種選択がスマホでやりにくい、どうすれば？」"
            "「片手で使えるように改善したい」などを聞いてみてください。"
        )

    if user_input := st.chat_input("田中主任に質問する...", key="mobile_agent_chat_input"):
        # ユーザーメッセージ追加
        st.session_state["mobile_agent_chat_history"].append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # AI応答
        with st.chat_message("assistant", avatar="👔"):
            with st.spinner("田中主任が回答中..."):
                reply = run_sales_agent_chat(
                    user_input,
                    st.session_state["mobile_agent_chat_history"][:-1],
                    chat_fn,
                    financial_context=financial_context,
                )
            st.markdown(reply)

        st.session_state["mobile_agent_chat_history"].append(
            {"role": "assistant", "content": reply}
        )

    # 会話リセット
    if st.session_state["mobile_agent_chat_history"]:
        if st.button("🗑️ 会話をリセット", key="btn_chat_reset"):
            st.session_state["mobile_agent_chat_history"] = []
            st.rerun()
