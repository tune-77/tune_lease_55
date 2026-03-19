# -*- coding: utf-8 -*-
"""
components/chat_wizard.py
=========================
「リースくん」— チャットウィザード形式の審査データ入力。

仕様:
  - 10ステップで全審査項目を順に収集（損益を1ステップに統合）
  - 全ステップ中2回だけランダムでユーモアコメントを挿入
  - 入力途中は data/wizard_draft.json に自動保存・再開可能
  - 完了後は st.session_state["wizard_form_result"] に格納し審査を実行

カラー: ネイビー #1A1A2E / ゴールド #E8A838 / 淡黄 #FFF8E8 / ペーパー #F4F1EC
フォント: Zen Kaku Gothic New（Google Fonts）
"""
from __future__ import annotations
import json
import os
import random
import datetime
import streamlit as st

# ── ディレクトリ ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR   = os.path.dirname(_SCRIPT_DIR)
_DATA_DIR   = os.path.join(_BASE_DIR, "data")
_DRAFT_PATH = os.path.join(_DATA_DIR, "wizard_draft.json")
_STATIC_DIR = os.path.join(_BASE_DIR, "static_data")

# ── CSS ────────────────────────────────────────────────────────────────────
_WIZ_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;700;900&family=IBM+Plex+Mono&display=swap');
.wiz-wrap { font-family:'Zen Kaku Gothic New',sans-serif; max-width:760px; margin:0 auto; }
.wiz-header {
    background:linear-gradient(135deg,#1A1A2E 0%,#2d2d4e 100%);
    color:#E8A838; border-radius:14px; padding:1.2rem 1.6rem;
    margin-bottom:1.4rem; display:flex; align-items:center; gap:1rem;
}
.wiz-header-title { font-size:1.5rem; font-weight:900; margin:0; }
.wiz-header-sub   { font-size:.82rem; opacity:.75; margin-top:.2rem; }
.wiz-progress-bar {
    height:6px; background:#e2e8f0; border-radius:3px; margin-bottom:1.2rem; overflow:hidden;
}
.wiz-progress-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,#1A1A2E,#E8A838); transition:width .4s; }
.wiz-bubble-bot {
    background:#fff; border:2px solid #1A1A2E; border-radius:14px 14px 14px 0;
    padding:.85rem 1.1rem; margin:.5rem 0 .5rem 0; font-size:.92rem; line-height:1.65;
    color:#1A1A2E; max-width:85%;
}
.wiz-bubble-user {
    background:#1A1A2E; color:#fff; border-radius:14px 14px 0 14px;
    padding:.7rem 1rem; margin:.5rem 0 .5rem auto; font-size:.88rem;
    line-height:1.55; max-width:80%; text-align:right;
}
.wiz-bubble-humor {
    background:#FFF8E8; border:2px solid #E8A838; border-radius:14px;
    padding:.85rem 1.1rem; margin:.7rem 0; font-size:.88rem; line-height:1.65; color:#92400e;
}
.wiz-bubble-humor-lbl { font-size:.72rem; font-weight:700; color:#E8A838; margin-bottom:.3rem; letter-spacing:.04em; }
.wiz-step-label {
    font-size:.72rem; font-weight:700; color:#94a3b8; letter-spacing:.06em;
    margin:.8rem 0 .2rem; text-transform:uppercase;
}
.wiz-summary-box {
    background:#f0f4ff; border:1px solid #c7d2fe; border-radius:10px;
    padding:1rem 1.2rem; margin:.8rem 0; font-size:.88rem; line-height:1.8;
}
.wiz-footer { text-align:center; color:#94a3b8; font-size:.72rem; margin-top:1.5rem; }
</style>
"""

# ── ユーモアコメント一覧 ──────────────────────────────────────────────────
_HUMOR_COMMENTS = [
    "ちなみに弊社の審査AIは徹夜で学習しました。目の下のクマは業界平均より深いです。",
    "この項目で高スコアの企業ほど、担当者の週末出勤率が低い傾向にあります（当社調べ）。",
    "リース審査の神様がいるとすれば、今ちょうどあなたの入力内容を覗き見しています。",
    "営業利益率がよければ、次の飲み会の経費は通りやすくなります。たぶん。",
    "ここを入力し終えると、あなたは今日の審査AI最大の理解者になります。",
    "この情報は厳重に管理されます。ただし弊社サーバーは少し暑がりです。",
    "上司が「直感で行け」と言った瞬間、このシステムが産声を上げました。",
    "DSCRが1.2倍を下回ると、私（リースくん）が少し不安な顔をします。見えませんが。",
]

# ── ステップ定義（損益を1ステップに統合 → 10ステップ）─────────────────────
_STEPS = [
    {"id": "industry",    "label": "業種選択",          "emoji": "🏭"},
    {"id": "deal",        "label": "取引状況・競合",    "emoji": "🤝"},
    {"id": "asset",       "label": "リース物件",        "emoji": "🚜"},
    {"id": "pl",          "label": "損益計算書",        "emoji": "📊"},
    {"id": "assets_main", "label": "資産情報",          "emoji": "🏦"},
    {"id": "expenses",    "label": "経費・減価償却",    "emoji": "💸"},
    {"id": "credit",      "label": "信用情報",          "emoji": "💳"},
    {"id": "contract",    "label": "契約条件",          "emoji": "📝"},
    {"id": "qualitative", "label": "定性評価",          "emoji": "🎯"},
    {"id": "intuition",   "label": "直感スコア・確認",  "emoji": "💡"},
]
_N_STEPS = len(_STEPS)


# ── データロード ────────────────────────────────────────────────────────────
def _load_json(filename: str) -> dict:
    for base in [_STATIC_DIR, _BASE_DIR]:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


def _load_jsic() -> dict:
    return _load_json("industry_trends_jsic.json")


def _load_assets() -> list:
    raw = _load_json("lease_assets.json")
    return raw.get("items", [])


# ── 下書き保存・読み込み ────────────────────────────────────────────────────
def _save_draft(data: dict, step: int = 0, history: list | None = None) -> None:
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        # 内部フラグ（_で始まるキー）は保存対象から除外
        clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
        with open(_DRAFT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "ts": datetime.datetime.now().isoformat(),
                "step": step,
                "history": history or [],
                "data": clean_data,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_draft() -> dict:
    try:
        if os.path.exists(_DRAFT_PATH):
            with open(_DRAFT_PATH, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _clear_draft() -> None:
    try:
        if os.path.exists(_DRAFT_PATH):
            os.remove(_DRAFT_PATH)
    except Exception:
        pass


# ── セッション初期化 ────────────────────────────────────────────────────────
def _init_session(jsic_data: dict) -> None:
    if "wiz_data" not in st.session_state:
        draft = _load_draft()
        # 24時間以上経過した下書きは自動削除
        if draft:
            try:
                ts = datetime.datetime.fromisoformat(draft.get("ts", "2000-01-01"))
                if (datetime.datetime.now() - ts).total_seconds() > 86400:
                    _clear_draft()
                    draft = {}
            except Exception:
                pass
        st.session_state["wiz_data"]    = draft.get("data", {})
        st.session_state["wiz_step"]    = draft.get("step", 0)
        st.session_state["wiz_history"] = draft.get("history", [])
    if "wiz_step" not in st.session_state:
        st.session_state["wiz_step"] = 0
    if "wiz_history" not in st.session_state:
        st.session_state["wiz_history"] = []
    if "wiz_humor_steps" not in st.session_state:
        st.session_state["wiz_humor_steps"] = random.sample(range(1, _N_STEPS), 2)
    if "wiz_major_keys" not in st.session_state:
        st.session_state["wiz_major_keys"] = list(jsic_data.keys()) if jsic_data else ["D 建設業"]


# ── チャット表示ヘルパー ────────────────────────────────────────────────────
def _bot(msg: str) -> None:
    st.markdown(f'<div class="wiz-bubble-bot">🤖 {msg}</div>', unsafe_allow_html=True)


def _user(msg: str) -> None:
    st.markdown(f'<div class="wiz-bubble-user">{msg}</div>', unsafe_allow_html=True)


def _humor(msg: str) -> None:
    st.markdown(
        f'<div class="wiz-bubble-humor">'
        f'<div class="wiz-bubble-humor-lbl">💬 リースくんの審査コメント</div>'
        f'{msg}</div>',
        unsafe_allow_html=True,
    )


# ── 過去ステップの履歴表示 ──────────────────────────────────────────────────
def _render_history() -> None:
    for entry in st.session_state["wiz_history"]:
        _bot(entry["question"])
        _user(entry["answer"])
        if entry.get("humor"):
            _humor(entry["humor"])


# ── 業種別ユーモアコメント ──────────────────────────────────────────────────
_INDUSTRY_HUMOR: dict[str, list[str]] = {
    "D 建設業": [
        "あちゃー建設業かあ…最近成約できてないんですよね。応援してます。",
        "建設業！現場の土の匂いが好きです。私、AIですけど。",
        "建設業かあ。天気に左右されるのが大変ですよね。私は晴れても雨でも同じですが。",
    ],
    "E 製造業": [
        "製造業！工場見学って何度行っても楽しいですよね。私は行けませんが。",
        "製造業かあ。機械の音が響く職場、ロマンですよね。",
        "製造業！原価率が気になりますね。私も気になります。とても。",
    ],
    "F 電気・ガス・熱供給・水道業": [
        "ライフライン系ですね。停電は困りますよね。私もサーバーが落ちると困ります。",
        "インフラ企業！安定感が羨ましいです。私の応答は時々遅いですが。",
    ],
    "G 情報通信業": [
        "情報通信業！同業者に近い気がして少し親近感があります。",
        "ITですね。SES案件の審査は件数多くてちょっと大変なんですよね…。",
        "情報通信業！デジタル系は成長著しいですね。私の学習データも更新してほしいです。",
    ],
    "H 運輸業，郵便業": [
        "運輸業！車両リースが多いですよね。私の得意分野です（たぶん）。",
        "物流かあ。2024年問題、大変でしたね。残業上限の話、私も他人事ではありません。",
    ],
    "I 卸売業，小売業": [
        "卸売・小売！在庫管理が鍵ですよね。私は記憶が揮発性なので在庫ゼロです。",
        "小売業かあ。消費者の財布の紐が緩む日を祈っています。",
    ],
    "J 金融業，保険業": [
        "金融・保険業！同業に近いような気が…。お互いリスク管理、頑張りましょう。",
        "金融系！自己資本比率の感覚が鋭そうですね。審査しやすいです（たぶん）。",
    ],
    "K 不動産業，物品賃貸業": [
        "不動産！金利動向が気になりますよね。私も毎日気になっています。",
        "不動産業かあ。物件ファイナンスとの相性抜群ですね。",
    ],
    "L 学術研究，専門・技術サービス業": [
        "専門・技術系！知的な雰囲気、好きです。私も知的でありたいです。",
        "コンサルやシンクタンク系ですね。頭脳で勝負、格好いいです。",
    ],
    "M 宿泊業，飲食サービス業": [
        "飲食業！昔コックになりたかったなあ…私、AIですけど。",
        "宿泊・飲食か。コロナ以降、審査が本当に難しくなりましたよね。一緒に頑張りましょう。",
        "飲食業！美味しいご飯を提供する会社の審査、気合い入ります。",
    ],
    "N 生活関連サービス業，娯楽業": [
        "娯楽・生活サービス！人々を笑顔にする仕事、素敵ですよね。私も笑顔にしたいです。",
        "美容・理容・レジャーか。生活を豊かにする業界ですね。",
    ],
    "O 教育，学習支援業": [
        "教育業！未来への投資ですよね。私も毎日学習中です（させられています）。",
        "学習支援か。先生は大変ですよね。私も毎日質問攻めです。",
    ],
    "P 医療，福祉": [
        "医療・福祉！社会に絶対必要な業界ですね。審査も責任重大と感じます。",
        "医療系か。お医者さんってリース審査に来ると数字が独特で面白いんですよね。",
    ],
    "Q 複合サービス事業": [
        "複合サービスか。農協・郵便局系ですね。地域密着、いいですよね。",
    ],
    "R サービス業（他に分類されないもの）": [
        "その他サービスか。これが一番幅広くて審査が奥深いんですよね。",
        "サービス業！多様で面白い業界ですね。私も「その他AI」に分類されそうです。",
    ],
    "S 公務": [
        "公務！安定感ナンバーワンですよね。羨ましい限りです。",
    ],
    "A 農業，林業": [
        "農業・林業！自然相手のお仕事ですね。天気に一喜一憂する気持ち、わかります（わかりません）。",
    ],
    "B 漁業": [
        "漁業！海、広いですよね。私の知識の海とどちらが広いか勝負です（負けます）。",
    ],
    "C 鉱業，採石業，砂利採取業": [
        "鉱業！なかなかレアな審査案件ですね。ちょっと緊張します。",
    ],
}


def _get_industry_humor(major: str) -> str:
    comments = _INDUSTRY_HUMOR.get(major)
    if not comments:
        for key, vals in _INDUSTRY_HUMOR.items():
            if any(c in major for c in key.split()) or any(c in key for c in major.split()):
                comments = vals
                break
    if comments:
        return random.choice(comments)
    return f"「{major}」ですね。全力でサポートします！"


# ── ナビゲーションボタン ─────────────────────────────────────────────────────
def _nav_buttons(step: int, question: str, answer: str, updates: dict,
                 can_proceed: bool = True, warn_msg: str = "") -> None:
    """「前へ」「次へ」ボタンを描画する。can_proceed=False の場合は次へを非活性表示。"""
    if not can_proceed and warn_msg:
        st.warning(warn_msg)

    col_back, col_next = st.columns([1, 3])
    with col_back:
        if step > 0 and st.button("← 前へ", key=f"wiz_back_{step}"):
            hist = st.session_state["wiz_history"]
            if hist:
                st.session_state["wiz_history"] = hist[:-1]
            st.session_state["wiz_step"] = step - 1
            st.rerun()
    with col_next:
        label = "内容を確認して審査実行 🚀" if step == _N_STEPS - 1 else "次へ →"
        btn = st.button(label, key=f"wiz_next_{step}", type="primary",
                        disabled=not can_proceed)
        if btn and can_proceed:
            _advance(step, question, answer, updates)


def _advance(step: int, question: str, answer: str, updates: dict) -> None:
    d = st.session_state["wiz_data"]
    d.update(updates)
    next_step = step + 1 if step < _N_STEPS - 1 else step
    _save_draft(d, step=next_step, history=st.session_state.get("wiz_history", []))

    humor_comment = ""
    if step in st.session_state.get("wiz_humor_steps", []):
        humor_comment = random.choice(_HUMOR_COMMENTS)

    st.session_state["wiz_history"].append({
        "question": question,
        "answer":   answer,
        "humor":    humor_comment,
    })

    if step < _N_STEPS - 1:
        st.session_state["wiz_step"] = step + 1
    else:
        _submit_wizard(d)

    st.rerun()


# ── ステップレンダラー ──────────────────────────────────────────────────────
def _render_step(step: int, jsic_data: dict, assets: list) -> None:
    sid = _STEPS[step]["id"]
    d   = st.session_state["wiz_data"]

    # ── STEP: industry ─────────────────────────────────────────────────────
    if sid == "industry":
        # 前回案件の複写ボタン
        _last_case_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "last_case.json"
        )
        if os.path.exists(_last_case_path):
            if st.button("📋 前回案件を複写", key="wiz_copy_last", help="前回入力した案件データを引き継ぎます"):
                try:
                    import json as _json
                    with open(_last_case_path, encoding="utf-8") as _f:
                        _last = _json.load(_f)
                    # 内部管理キーは引き継がない
                    _skip = {"_last_humor_major", "_industry_humor_msg", "_frag_major", "_frag_sub"}
                    for _k, _v in _last.items():
                        if _k not in _skip:
                            st.session_state["wiz_data"][_k] = _v
                    st.toast("✅ 前回案件を複写しました", icon="📋")
                    st.rerun()
                except Exception:
                    st.warning("前回案件の読み込みに失敗しました")

        _bot("はじめまして！リースくんです 🎩<br>"
             "まず、審査対象の<b>業種</b>を選んでください。<br>"
             "大分類→中分類の順に絞り込んでいきます。")
        major_keys = st.session_state["wiz_major_keys"]
        cur_major  = d.get("selected_major", major_keys[0])
        if cur_major not in major_keys:
            cur_major = major_keys[0]
        major = st.selectbox("大分類（日本標準産業分類）", major_keys,
                             index=major_keys.index(cur_major), key="wiz_sel_major")
        if major != d.get("_last_humor_major"):
            d["_last_humor_major"] = major
            d["_industry_humor_msg"] = _get_industry_humor(major)
        if d.get("_industry_humor_msg"):
            _humor(d["_industry_humor_msg"])

        sub_data = jsic_data.get(major, {}).get("sub", {}) if jsic_data else {}
        sub_keys = list(sub_data.keys()) if sub_data else ["06 総合工事業"]
        cur_sub  = d.get("selected_sub", sub_keys[0])
        if cur_sub not in sub_keys:
            cur_sub = sub_keys[0]
        sub = st.selectbox("中分類", sub_keys,
                           index=sub_keys.index(cur_sub), key="wiz_sel_sub")
        if st.button("次へ →", key="wiz_next_industry", type="primary"):
            _advance(step, question="業種を選択してください",
                     answer=f"{major} ／ {sub}",
                     updates={"selected_major": major, "selected_sub": sub,
                              "_frag_major": major, "_frag_sub": sub})

    # ── STEP: deal ─────────────────────────────────────────────────────────
    elif sid == "deal":
        _bot("この案件、<b>うちがメイン取引先</b>ですか？<br>"
             "それと、競合他社は入っていますか？正直に教えてください 😅")
        main_bank  = st.radio("取引区分", ["メイン先", "非メイン先"],
                              index=["メイン先","非メイン先"].index(d.get("main_bank","メイン先")),
                              horizontal=True, key="wiz_main_bank")
        competitor = st.radio("競合状況", ["競合なし", "競合あり"],
                              index=["競合なし","競合あり"].index(d.get("competitor","競合なし")),
                              horizontal=True, key="wiz_competitor")
        comp_rate = 0.0
        if competitor == "競合あり":
            comp_rate = st.number_input("競合の提示金利（%）", 0.0, 30.0,
                                        float(d.get("competitor_rate_input", 0.0)), 0.1,
                                        key="wiz_comp_rate")
        num_comp = st.selectbox("競合社数", ["未入力","0社（指名）","1社","2社","3社以上"],
                                index=["未入力","0社（指名）","1社","2社","3社以上"]
                                .index(d.get("num_competitors","未入力")), key="wiz_num_comp")
        occ = st.selectbox("発生経緯", ["不明","指名","相見積もり"],
                           index=["不明","指名","相見積もり"].index(d.get("deal_occurrence","不明")),
                           key="wiz_deal_occ")
        ans = f"{main_bank} / {competitor}" + (f" / 競合金利{comp_rate}%" if competitor=="競合あり" else "")
        _nav_buttons(step, question="取引区分・競合状況を教えてください", answer=ans,
                     updates={"main_bank": main_bank, "competitor": competitor,
                              "competitor_rate_input": comp_rate,
                              "num_competitors": num_comp, "deal_occurrence": occ})

    # ── STEP: asset ─────────────────────────────────────────────────────────
    elif sid == "asset":
        _bot("何をリースしますか？<br>リース<b>物件</b>を一覧から選んでください。")
        asset_names = [a.get("name", f"物件{i}") for i, a in enumerate(assets)]
        if not asset_names:
            asset_names = ["その他"]
        cur_idx = int(d.get("asset_index", 0))
        if cur_idx >= len(asset_names):
            cur_idx = 0
        idx = st.selectbox("物件", asset_names, index=cur_idx, key="wiz_asset")
        sel_idx   = asset_names.index(idx) if idx in asset_names else 0
        sel_asset = assets[sel_idx] if assets and sel_idx < len(assets) else {}
        asset_id    = sel_asset.get("id", "other")
        asset_score = sel_asset.get("score", 0)
        asset_name  = sel_asset.get("name", "その他")

        vtype = ""
        if asset_id == "vehicle":
            vtype_opts = ["", "ハイエース バン（商用バン）", "キャラバン（商用バン）",
                          "軽商用バン（エブリイ / ハイゼット等）", "営業用トラック",
                          "自家用トラック", "一般乗用車（ヤリス / カローラ等）",
                          "レクサス・外車等（役員車）"]
            vtype = st.selectbox("車種・車両タイプ（任意）", vtype_opts, key="wiz_vtype")
            if vtype:
                asset_name = f"{asset_name}（{vtype}）"

        _nav_buttons(step, question="リース物件を選んでください",
                     answer=asset_name,
                     updates={"asset_index": sel_idx, "selected_asset_id": asset_id,
                              "asset_score": asset_score, "asset_name": asset_name,
                              "asset_category": sel_asset.get("category"),
                              "asset_vtype_select": vtype})

    # ── STEP: pl（損益計算書 — 旧pl_main+pl_other を統合）──────────────────
    elif sid == "pl":
        _bot("損益計算書の数字を教えてください 📄<br>"
             "<b>売上高は必須です</b>（スコア算出の基準になります）。<br>"
             "わからない項目は空欄のまま「次へ」で 0 として扱います。")
        nenshu = st.number_input("売上高（千円）📌必須", 0, 90_000_000,
                                 value=int(d["nenshu"]) if "nenshu" in d else None,
                                 step=100, key="wiz_nenshu", placeholder="例: 50000")
        rieki  = st.number_input("営業利益（千円）💡推奨", -100_000, 90_000_000,
                                 value=int(d["rieki"]) if "rieki" in d else None,
                                 step=100, key="wiz_rieki", placeholder="例: 3000（赤字は -3000）")
        gross  = st.number_input("売上総利益（千円）", -500_000, 90_000_000,
                                 value=int(d["item9_gross"]) if "item9_gross" in d else None,
                                 step=100, key="wiz_gross", placeholder="空欄→0")
        ord_p  = st.number_input("経常利益（千円）", -100_000, 90_000_000,
                                 value=int(d["item4_ord_profit"]) if "item4_ord_profit" in d else None,
                                 step=100, key="wiz_ordp", placeholder="空欄→0")
        net_i  = st.number_input("当期純利益（千円）", -100_000, 90_000_000,
                                 value=int(d["item5_net_income"]) if "item5_net_income" in d else None,
                                 step=100, key="wiz_neti", placeholder="空欄→0")
        nenshu_v = nenshu or 0
        rieki_v  = rieki  or 0
        gross_v  = gross  or 0
        ord_p_v  = ord_p  or 0
        net_i_v  = net_i  or 0
        ok = nenshu_v > 0
        _nav_buttons(step,
                     question="損益計算書の主要項目を教えてください",
                     answer=f"売上高: {nenshu_v:,}千円 / 営業利益: {rieki_v:,}千円",
                     updates={"nenshu": nenshu_v,
                              "rieki": rieki_v, "num_rieki": rieki_v,
                              "item9_gross": gross_v, "num_sourieki": gross_v,
                              "item4_ord_profit": ord_p_v, "num_item4_ord_profit": ord_p_v,
                              "item5_net_income": net_i_v, "num_item5_net_income": net_i_v},
                     can_proceed=ok,
                     warn_msg="⚠️ 売上高は1以上の値を入力してください。スコア算出の基準になります。")

    # ── STEP: assets_main ───────────────────────────────────────────────────
    elif sid == "assets_main":
        _bot("次は貸借対照表です 🏦<br>"
             "<b>総資産は必須です</b>（自己資本比率の計算に使います）。<br>"
             "決算書の「資産の部 合計」をそのまま入力してください。")
        total  = st.number_input("総資産（千円）📌必須", 0, 90_000_000,
                                 value=int(d["total_assets"]) if "total_assets" in d else None,
                                 step=100, key="wiz_total", placeholder="例: 80000")
        net_a  = st.number_input("純資産（千円）💡推奨", -30_000, 90_000_000,
                                 value=int(d["net_assets"]) if "net_assets" in d else None,
                                 step=100, key="wiz_neta", placeholder="例: 25000")
        mach   = st.number_input("機械装置（千円）", 0, 90_000_000,
                                 value=int(d["item6_machine"]) if "item6_machine" in d else None,
                                 step=100, key="wiz_mach", placeholder="空欄→0")
        other  = st.number_input("その他資産（千円）", 0, 90_000_000,
                                 value=int(d["item7_other"]) if "item7_other" in d else None,
                                 step=100, key="wiz_otha", placeholder="空欄→0")
        total_v = total or 0
        net_a_v = net_a or 0
        mach_v  = mach  or 0
        other_v = other or 0
        ok = total_v > 0
        _nav_buttons(step,
                     question="資産情報を入力してください",
                     answer=f"総資産: {total_v:,} / 純資産: {net_a_v:,}（千円）",
                     updates={"total_assets": total_v, "num_total_assets": total_v,
                              "net_assets": net_a_v, "num_net_assets": net_a_v,
                              "item6_machine": mach_v, "num_item6_machine": mach_v,
                              "item7_other": other_v, "num_item7_other": other_v},
                     can_proceed=ok,
                     warn_msg="⚠️ 総資産は1以上の値を入力してください。自己資本比率の計算に必要です。")

    # ── STEP: expenses ──────────────────────────────────────────────────────
    elif sid == "expenses":
        _bot("経費・減価償却費を入力してください 💸<br>"
             "決算書に記載がない場合や分からない場合は、<b>スキップして 0 のまま</b>で構いません。")
        dep_a  = st.number_input("減価償却費（資産）千円", 0, 90_000_000,
                                  value=int(d["item10_dep"]) if "item10_dep" in d else None,
                                  step=100, key="wiz_depa", placeholder="空欄→0")
        dep_e  = st.number_input("減価償却費（経費）千円", 0, 90_000_000,
                                  value=int(d["item11_dep_exp"]) if "item11_dep_exp" in d else None,
                                  step=100, key="wiz_depe", placeholder="空欄→0")
        rent_a = st.number_input("賃借料（資産）千円", 0, 90_000_000,
                                  value=int(d["item8_rent"]) if "item8_rent" in d else None,
                                  step=100, key="wiz_renta", placeholder="空欄→0")
        rent_e = st.number_input("賃借料（経費）千円", 0, 90_000_000,
                                  value=int(d["item12_rent_exp"]) if "item12_rent_exp" in d else None,
                                  step=100, key="wiz_rente", placeholder="空欄→0")
        dep_a_v  = dep_a  or 0
        dep_e_v  = dep_e  or 0
        rent_a_v = rent_a or 0
        rent_e_v = rent_e or 0

        col_skip, _ = st.columns([1, 3])
        with col_skip:
            if st.button("⏭️ この項目をスキップ（全て0）", key="wiz_skip_expenses"):
                _advance(step, question="経費・減価償却費",
                         answer="スキップ（全て0）",
                         updates={"item10_dep": 0, "num_item10_dep": 0,
                                  "item11_dep_exp": 0, "num_item11_dep_exp": 0,
                                  "item8_rent": 0, "num_item8_rent": 0,
                                  "item12_rent_exp": 0, "num_item12_rent_exp": 0})

        _nav_buttons(step, question="経費・減価償却費を入力してください",
                     answer=f"減価償却費（資産）{dep_a_v:,} / 賃借料（資産）{rent_a_v:,}千円",
                     updates={"item10_dep": dep_a_v, "num_item10_dep": dep_a_v,
                              "item11_dep_exp": dep_e_v, "num_item11_dep_exp": dep_e_v,
                              "item8_rent": rent_a_v, "num_item8_rent": rent_a_v,
                              "item12_rent_exp": rent_e_v, "num_item12_rent_exp": rent_e_v})

    # ── STEP: credit ────────────────────────────────────────────────────────
    elif sid == "credit":
        _bot("信用情報を確認させてください 💳<br>"
             "格付・与信残高・既存の契約件数を入力してください。")
        grade_opts = ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"]
        grade  = st.radio("格付", grade_opts,
                          index=grade_opts.index(d.get("grade","②4-6 (標準)")),
                          horizontal=True, key="wiz_grade")
        bank_c = st.number_input("うちの銀行与信（千円）", 0, 90_000_000,
                                  value=int(d["bank_credit"]) if "bank_credit" in d else None,
                                  step=100, key="wiz_bankc", placeholder="なければ空欄→0")
        lease_c= st.number_input("うちのリース与信（千円）", 0, 90_000_000,
                                  value=int(d["lease_credit"]) if "lease_credit" in d else None,
                                  step=100, key="wiz_leasec", placeholder="なければ空欄→0")
        cont   = st.number_input("既存の契約数（件）", 0, 30,
                                  value=int(d["contracts"]) if "contracts" in d else None,
                                  step=1, key="wiz_cont", placeholder="なければ空欄→0")
        bank_c_v  = bank_c  or 0
        lease_c_v = lease_c or 0
        cont_v    = cont    or 0
        _nav_buttons(step, question="信用情報を入力してください",
                     answer=f"格付: {grade} / 銀行与信: {bank_c_v:,}千円 / リース与信: {lease_c_v:,}千円",
                     updates={"grade": grade, "bank_credit": bank_c_v, "num_bank_credit": bank_c_v,
                              "lease_credit": lease_c_v, "num_lease_credit": lease_c_v,
                              "contracts": cont_v, "num_contracts": cont_v})

    # ── STEP: contract ──────────────────────────────────────────────────────
    elif sid == "contract":
        _bot("契約の条件を教えてください 📝<br>"
             "期間・金額・顧客区分など、今回の案件の基本情報です。")
        cust_type = st.radio("顧客区分", ["既存先","新規先"],
                             index=["既存先","新規先"].index(d.get("customer_type","既存先")),
                             horizontal=True, key="wiz_cust")
        cont_type = st.radio("契約種類", ["一般","自動車"],
                             index=["一般","自動車"].index(d.get("contract_type","一般")),
                             horizontal=True, key="wiz_ctype")
        deal_src  = st.radio("商談ソース", ["銀行紹介","その他"],
                             index=["銀行紹介","その他"].index(d.get("deal_source","その他")),
                             horizontal=True, key="wiz_dsrc")
        term      = st.select_slider("契約期間（月）", options=list(range(0,121)),
                                     value=int(d.get("lease_term", 60)), key="wiz_term")
        year      = st.number_input("検収年（西暦）", 1900, 9999,
                                    int(d.get("acceptance_year", datetime.date.today().year)),
                                    1, key="wiz_year")
        acq       = st.number_input("取得価格（千円）", 0, 90_000_000,
                                    value=int(d["acquisition_cost"]) if "acquisition_cost" in d else None,
                                    step=100, key="wiz_acq", placeholder="例: 3000（＝300万円）")
        acq_v = acq or 0
        _nav_buttons(step, question="契約条件を入力してください",
                     answer=f"{cust_type} / {cont_type} / {term}ヶ月 / {acq_v:,}千円",
                     updates={"customer_type": cust_type, "contract_type": cont_type,
                              "deal_source": deal_src, "lease_term": term,
                              "acceptance_year": year,
                              "acquisition_cost": acq_v, "num_acquisition_cost": acq_v})

    # ── STEP: qualitative ───────────────────────────────────────────────────
    elif sid == "qualitative":
        _bot("定性評価を教えてください 🎯<br>"
             "数字には表れない、担当者としての目線です。<br>"
             "「わからない」「未選択」のままでも審査は実行できます。")
        _q_opts = {
            "qual_corr_company_history":      ("設立・経営年数（重み10%）", ["未選択","20年以上","10年〜20年","5年〜10年","3年〜5年","3年未満"]),
            "qual_corr_customer_stability":   ("顧客安定性（重み20%）",     ["未選択","非常に安定（大口・長期）","安定（分散良好）","普通","やや不安定（集中あり）","不安定・依存大"]),
            "qual_corr_repayment_history":    ("返済履歴（重み25%）",        ["未選択","5年以上問題なし","3年以上問題なし","遅延少ない","遅延・リスケあり","問題あり・要確認"]),
            "qual_corr_business_future":      ("事業将来性（重み15%）",      ["未選択","有望（成長・ニーズ確実）","やや有望","普通","やや懸念","懸念（縮小・競争激化）"]),
            "qual_corr_equipment_purpose":    ("設備目的（重み15%）",        ["未選択","収益直結・受注必須","生産性向上・省力化","更新・維持・法定対応","やや不明確","不明確・要説明"]),
            "qual_corr_main_bank":            ("メイン取引銀行（重み15%）",  ["未選択","メイン先で取引良好・支援表明","メイン先","サブ扱い・取引あり","取引浅い・他社メイン","取引なし・不安"]),
        }
        qual_updates = {}
        for key, (label, opts) in _q_opts.items():
            cur = d.get(key, "未選択")
            if cur not in opts:
                cur = "未選択"
            val = st.selectbox(label, opts, index=opts.index(cur), key=f"wiz_{key}")
            qual_updates[key] = val
        _nav_buttons(step, question="定性スコアリングを入力してください",
                     answer="定性評価 入力完了",
                     updates=qual_updates)

    # ── STEP: intuition（最終ステップ — 直感スコア + 確認サマリー）──────────
    elif sid == "intuition":
        _bot("最後に、担当者としての<b>直感スコア</b>を教えてください 💡<br>"
             "1＝懸念あり　→　5＝確信あり<br>"
             "その後、入力内容を確認してから審査を実行します。")
        score = st.slider("直感スコア", 1, 5, int(d.get("intuition_score", 3)), 1,
                          key="wiz_intuition", format="%d点")
        labels = {1:"😟 懸念", 2:"🤔 やや懸念", 3:"😐 中立", 4:"🙂 やや良好", 5:"😄 確信"}
        st.caption(labels.get(score, ""))

        # ── 確認サマリー ────────────────────────────────────────────────
        st.markdown("---")
        _bot("以下の内容で審査を実行します。間違いがあれば「← 前へ」で戻って修正してください。")
        nenshu_v = int(d.get("nenshu", 0))
        rieki_v  = int(d.get("rieki", 0))
        total_v  = int(d.get("total_assets", 0))
        net_v    = int(d.get("net_assets", 0))
        op_rate  = round(rieki_v / nenshu_v * 100, 1) if nenshu_v > 0 else "—"
        eq_rate  = round(net_v / total_v * 100, 1) if total_v > 0 else "—"
        st.markdown(f"""
<div class="wiz-summary-box">
<b>業種</b>　{d.get("selected_major","—")} ／ {d.get("selected_sub","—")}<br>
<b>物件</b>　{d.get("asset_name","—")}<br>
<b>売上高</b>　{nenshu_v:,}千円　／　<b>営業利益</b>　{rieki_v:,}千円　（営業利益率 {op_rate}%）<br>
<b>総資産</b>　{total_v:,}千円　／　<b>純資産</b>　{net_v:,}千円　（自己資本比率 {eq_rate}%）<br>
<b>格付</b>　{d.get("grade","—")}　／　<b>取引区分</b>　{d.get("main_bank","—")}　／　<b>競合</b>　{d.get("competitor","—")}<br>
<b>取得価格</b>　{int(d.get("acquisition_cost",0)):,}千円　／　<b>期間</b>　{d.get("lease_term","—")}ヶ月<br>
<b>直感スコア</b>　{score}点　{labels.get(score,"")}
</div>
""", unsafe_allow_html=True)

        _nav_buttons(step, question="担当者の直感スコアを教えてください",
                     answer=f"直感スコア: {score}点 {labels.get(score,'')}",
                     updates={"intuition_score": score, "intuition": score})


# ── 審査実行 ────────────────────────────────────────────────────────────────
def _submit_wizard(d: dict) -> None:
    form_result = {
        "submitted_apply": False,
        "submitted_judge": True,
        "selected_major":  d.get("selected_major", "D 建設業"),
        "selected_sub":    d.get("selected_sub", "06 総合工事業"),
        "main_bank":       d.get("main_bank", "メイン先"),
        "competitor":      d.get("competitor", "競合なし"),
        "competitor_rate_input": float(d.get("competitor_rate_input", 0.0)),
        "num_competitors": d.get("num_competitors", "未入力"),
        "deal_occurrence": d.get("deal_occurrence", "不明"),
        "nenshu":          int(d.get("nenshu", 0)),
        "item9_gross":     int(d.get("item9_gross", 0)),
        "rieki":           int(d.get("rieki", 0)),
        "item4_ord_profit":int(d.get("item4_ord_profit", 0)),
        "item5_net_income":int(d.get("item5_net_income", 0)),
        "item10_dep":      int(d.get("item10_dep", 0)),
        "item11_dep_exp":  int(d.get("item11_dep_exp", 0)),
        "item8_rent":      int(d.get("item8_rent", 0)),
        "item12_rent_exp": int(d.get("item12_rent_exp", 0)),
        "item6_machine":   int(d.get("item6_machine", 0)),
        "item7_other":     int(d.get("item7_other", 0)),
        "net_assets":      int(d.get("net_assets", 0)),
        "total_assets":    int(d.get("total_assets", 0)),
        "grade":           d.get("grade", "②4-6 (標準)"),
        "bank_credit":     int(d.get("bank_credit", 0)),
        "lease_credit":    int(d.get("lease_credit", 0)),
        "contracts":       int(d.get("contracts", 1)),
        "customer_type":   d.get("customer_type", "既存先"),
        "contract_type":   d.get("contract_type", "一般"),
        "deal_source":     d.get("deal_source", "その他"),
        "lease_term":      int(d.get("lease_term", 60)),
        "acceptance_year": int(d.get("acceptance_year", datetime.date.today().year)),
        "acquisition_cost":int(d.get("acquisition_cost", 1000)),
        "selected_asset_id": d.get("selected_asset_id", "other"),
        "asset_score":     int(d.get("asset_score", 0)),
        "asset_name":      d.get("asset_name", "その他"),
        "asset_category":  d.get("asset_category"),
        "intuition":       int(d.get("intuition", 3)),
        "qual_corr_company_history":    d.get("qual_corr_company_history", "未選択"),
        "qual_corr_customer_stability": d.get("qual_corr_customer_stability", "未選択"),
        "qual_corr_repayment_history":  d.get("qual_corr_repayment_history", "未選択"),
        "qual_corr_business_future":    d.get("qual_corr_business_future", "未選択"),
        "qual_corr_equipment_purpose":  d.get("qual_corr_equipment_purpose", "未選択"),
        "qual_corr_main_bank":          d.get("qual_corr_main_bank", "未選択"),
    }

    st.session_state["wizard_form_result"] = form_result

    # 前回案件として保存（複写ボタン用）
    try:
        import json as _json
        _last_case_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "last_case.json"
        )
        with open(_last_case_path, "w", encoding="utf-8") as _f:
            _json.dump(d, _f, ensure_ascii=False, default=str)
    except Exception:
        pass

    # score_calculation.py は st.session_state から数値を読み直すため全キーを書き込む
    for k in ["nenshu", "item9_gross", "rieki", "item4_ord_profit", "item5_net_income",
              "item10_dep", "item11_dep_exp", "item8_rent", "item12_rent_exp",
              "item6_machine", "item7_other", "net_assets", "total_assets",
              "bank_credit", "lease_credit", "contracts", "lease_term",
              "acquisition_cost", "acceptance_year"]:
        if k in form_result:
            st.session_state[k] = form_result[k]

    st.session_state["_wizard_submitted"]  = True
    _clear_draft()

    st.session_state["_pending_mode"]     = "📋 審査・分析"
    st.session_state["nav_mode_widget"]   = "📊 分析結果"
    st.session_state["_jump_to_analysis"] = True
    for k in ["wiz_step", "wiz_data", "wiz_history", "wiz_humor_steps", "wiz_major_keys"]:
        st.session_state.pop(k, None)


# ── メイン描画 ──────────────────────────────────────────────────────────────
def render_chat_wizard() -> None:
    st.markdown(_WIZ_CSS, unsafe_allow_html=True)

    jsic_data = _load_jsic()
    assets    = _load_assets()
    _init_session(jsic_data)

    step = st.session_state["wiz_step"]

    st.markdown('<div class="wiz-wrap">', unsafe_allow_html=True)

    st.markdown(f"""
<div class="wiz-header">
  <div style="font-size:2.2rem;">🎩</div>
  <div>
    <div class="wiz-header-title">リースくん</div>
    <div class="wiz-header-sub">チャット形式でリース審査情報を収集します</div>
  </div>
</div>""", unsafe_allow_html=True)

    pct = int(step / _N_STEPS * 100)
    st.markdown(
        f'<div class="wiz-progress-bar"><div class="wiz-progress-fill" style="width:{pct}%;"></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"ステップ {step + 1} / {_N_STEPS}  ──  {_STEPS[step]['emoji']} {_STEPS[step]['label']}")

    _render_history()

    st.markdown(f'<div class="wiz-step-label">── {_STEPS[step]["emoji"]} {_STEPS[step]["label"]}</div>',
                unsafe_allow_html=True)
    _render_step(step, jsic_data, assets)

    # 下書き情報・操作ボタン
    st.divider()
    draft_ts = ""
    try:
        if os.path.exists(_DRAFT_PATH):
            with open(_DRAFT_PATH, encoding="utf-8") as f:
                draft_ts = json.load(f).get("ts", "")[:16].replace("T", " ")
    except Exception:
        pass

    col_r, col_d, col_info = st.columns([1, 1, 3])
    with col_r:
        if st.button("🔄 最初からやり直す", key="wiz_reset"):
            for k in ["wiz_step", "wiz_data", "wiz_history", "wiz_humor_steps", "wiz_major_keys"]:
                st.session_state.pop(k, None)
            _clear_draft()
            st.rerun()
    with col_d:
        if draft_ts:
            if st.button("🗑️ 下書きを削除", key="wiz_delete_draft",
                         help="入力途中の財務情報を端末から消去します。審査済みDBは保持されます。"):
                for k in ["wiz_step", "wiz_data", "wiz_history", "wiz_humor_steps", "wiz_major_keys"]:
                    st.session_state.pop(k, None)
                _clear_draft()
                st.success("下書きを削除しました。審査済みデータ（DB）は保持されています。")
                st.rerun()
    with col_info:
        if draft_ts:
            st.caption(f"💾 下書き保存済み（{draft_ts}）— 前回の続きから再開できます  |  ⚠️ 財務情報が一時保存されています")

    st.markdown('<div class="wiz-footer">リースくん v1.1 ── 温水式 リース審査AI</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
