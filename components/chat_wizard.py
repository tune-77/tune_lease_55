# -*- coding: utf-8 -*-
"""
components/chat_wizard.py
=========================
「リースくん」— チャットウィザード形式の審査データ入力。

仕様:
  - 10ステップで全審査項目を順に収集
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
/* チャットバブル */
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
.wiz-footer { text-align:center; color:#94a3b8; font-size:.72rem; margin-top:1.5rem; }
</style>
"""

# ── ユーモアコメント一覧（全ステップ中2回だけランダムに選んで表示）────────
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

# ── ステップ定義 ────────────────────────────────────────────────────────────
_STEPS = [
    {"id": "industry",    "label": "業種選択",        "emoji": "🏭"},
    {"id": "deal",        "label": "取引状況・競合",  "emoji": "🤝"},
    {"id": "asset",       "label": "リース物件",      "emoji": "🚜"},
    {"id": "pl_main",     "label": "売上高・営業利益", "emoji": "📊"},
    {"id": "pl_other",    "label": "その他損益",      "emoji": "📈"},
    {"id": "assets_main", "label": "資産情報",        "emoji": "🏦"},
    {"id": "expenses",    "label": "経費・減価償却",  "emoji": "💸"},
    {"id": "credit",      "label": "信用情報",        "emoji": "💳"},
    {"id": "contract",    "label": "契約条件",        "emoji": "📝"},
    {"id": "qualitative", "label": "定性評価",        "emoji": "🎯"},
    {"id": "intuition",   "label": "直感スコア",      "emoji": "💡"},
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
        with open(_DRAFT_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "ts": datetime.datetime.now().isoformat(),
                "step": step,
                "history": history or [],
                "data": data,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_draft() -> dict:
    """ドラフト全体を返す。キー: data, step, history"""
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
        st.session_state["wiz_data"]    = draft.get("data", {})
        st.session_state["wiz_step"]    = draft.get("step", 0)
        st.session_state["wiz_history"] = draft.get("history", [])
    if "wiz_step" not in st.session_state:
        st.session_state["wiz_step"] = 0
    if "wiz_history" not in st.session_state:
        st.session_state["wiz_history"] = []
    if "wiz_humor_steps" not in st.session_state:
        # 全ステップから2か所をランダムで選ぶ
        st.session_state["wiz_humor_steps"] = random.sample(range(1, _N_STEPS), 2)
    # 業種リストの初期化
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


# ── ステップレンダラー ──────────────────────────────────────────────────────
def _render_step(step: int, jsic_data: dict, assets: list) -> None:
    sid = _STEPS[step]["id"]
    d   = st.session_state["wiz_data"]
    humor_steps = st.session_state["wiz_humor_steps"]

    # ── STEP: industry ─────────────────────────────────────────────────────
    if sid == "industry":
        _bot("はじめまして！リースくんです 🎩<br>まず、審査対象の<b>業種</b>を選んでください。")
        major_keys = st.session_state["wiz_major_keys"]
        cur_major  = d.get("selected_major", major_keys[0])
        if cur_major not in major_keys:
            cur_major = major_keys[0]
        major = st.selectbox("大分類（日本標準産業分類）", major_keys,
                             index=major_keys.index(cur_major), key="wiz_sel_major")
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
        _bot("取引区分と<b>競合状況</b>を教えてください。")
        main_bank  = st.radio("取引区分", ["メイン先", "非メイン先"],
                              index=["メイン先","非メイン先"].index(d.get("main_bank","メイン先")),
                              horizontal=True, key="wiz_main_bank")
        competitor = st.radio("競合状況", ["競合なし", "競合あり"],
                              index=["競合なし","競合あり"].index(d.get("competitor","競合なし")),
                              horizontal=True, key="wiz_competitor")
        comp_rate  = 0.0
        if competitor == "競合あり":
            comp_rate = st.number_input("競合提示金利（%）", 0.0, 30.0,
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
        _bot("リース<b>物件</b>を選んでください。")
        asset_names = [a.get("name", f"物件{i}") for i, a in enumerate(assets)]
        if not asset_names:
            asset_names = ["その他"]
        cur_idx = int(d.get("asset_index", 0))
        if cur_idx >= len(asset_names):
            cur_idx = 0
        idx = st.selectbox("物件", asset_names, index=cur_idx, key="wiz_asset")
        sel_idx    = asset_names.index(idx) if idx in asset_names else 0
        sel_asset  = assets[sel_idx] if assets else {}
        asset_id   = sel_asset.get("id", "other")
        asset_score= sel_asset.get("score", 0)
        asset_name = sel_asset.get("name", "その他")

        # 車両タイプ
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
                     updates={"asset_index": idx, "selected_asset_id": asset_id,
                              "asset_score": asset_score, "asset_name": asset_name,
                              "asset_category": sel_asset.get("category"),
                              "asset_vtype_select": vtype})

    # ── STEP: pl_main ───────────────────────────────────────────────────────
    elif sid == "pl_main":
        _bot("損益計算書の主要項目を入力してください。<br>"
             "<b>売上高・総資産は必須</b>です（1以上）。")
        nenshu = st.number_input("売上高（千円）📌必須", 0, 90_000_000,
                                 int(d.get("nenshuu", 10_000)), 100, key="wiz_nenshu")
        rieki  = st.number_input("営業利益（千円）💡推奨", -100_000, 90_000_000,
                                 int(d.get("rieki", 10_000)), 100, key="wiz_rieki")
        _nav_buttons(step, question="売上高・営業利益を入力してください",
                     answer=f"売上高: {nenshu:,}千円 / 営業利益: {rieki:,}千円",
                     updates={"nenshuu": nenshu, "num_nenshuu": nenshu,
                              "rieki": rieki, "num_rieki": rieki})

    # ── STEP: pl_other ──────────────────────────────────────────────────────
    elif sid == "pl_other":
        _bot("その他の損益項目です。<br>分からない場合は 0 のままでも構いません。")
        gross  = st.number_input("売上総利益（千円）", -500_000, 90_000_000,
                                  int(d.get("item9_gross", 10_000)), 100, key="wiz_gross")
        ord_p  = st.number_input("経常利益（千円）", -100_000, 90_000_000,
                                  int(d.get("item4_ord_profit", 10_000)), 100, key="wiz_ordp")
        net_i  = st.number_input("当期純利益（千円）", -100_000, 90_000_000,
                                  int(d.get("item5_net_income", 10_000)), 100, key="wiz_neti")
        _nav_buttons(step, question="その他損益を入力してください",
                     answer=f"売上総利益: {gross:,} / 経常: {ord_p:,} / 当期純利益: {net_i:,}（千円）",
                     updates={"item9_gross": gross, "num_sourieki": gross,
                              "item4_ord_profit": ord_p, "num_item4_ord_profit": ord_p,
                              "item5_net_income": net_i, "num_item5_net_income": net_i})

    # ── STEP: assets_main ───────────────────────────────────────────────────
    elif sid == "assets_main":
        _bot("貸借対照表の主要項目を入力してください。<br>"
             "<b>総資産は必須</b>です（1以上）。")
        total  = st.number_input("総資産（千円）📌必須", 0, 90_000_000,
                                  int(d.get("total_assets", 10_000)), 100, key="wiz_total")
        net_a  = st.number_input("純資産（千円）💡推奨", -30_000, 90_000_000,
                                  int(d.get("net_assets", 10_000)), 100, key="wiz_neta")
        mach   = st.number_input("機械装置（千円）", 0, 90_000_000,
                                  int(d.get("item6_machine", 10_000)), 100, key="wiz_mach")
        other  = st.number_input("その他資産（千円）", 0, 90_000_000,
                                  int(d.get("item7_other", 10_000)), 100, key="wiz_otha")
        _nav_buttons(step, question="資産情報を入力してください",
                     answer=f"総資産: {total:,} / 純資産: {net_a:,}（千円）",
                     updates={"total_assets": total, "num_total_assets": total,
                              "net_assets": net_a, "num_net_assets": net_a,
                              "item6_machine": mach, "num_item6_machine": mach,
                              "item7_other": other, "num_item7_other": other})

    # ── STEP: expenses ──────────────────────────────────────────────────────
    elif sid == "expenses":
        _bot("経費・減価償却費を入力してください。<br>不明な場合は 0 で構いません。")
        dep_a  = st.number_input("減価償却費（資産）千円", 0, 90_000_000,
                                  int(d.get("item10_dep", 0)), 100, key="wiz_depa")
        dep_e  = st.number_input("減価償却費（経費）千円", 0, 90_000_000,
                                  int(d.get("item11_dep_exp", 0)), 100, key="wiz_depe")
        rent_a = st.number_input("賃借料（資産）千円", 0, 90_000_000,
                                  int(d.get("item8_rent", 0)), 100, key="wiz_renta")
        rent_e = st.number_input("賃借料（経費）千円", 0, 90_000_000,
                                  int(d.get("item12_rent_exp", 0)), 100, key="wiz_rente")
        _nav_buttons(step, question="経費・減価償却費を入力してください",
                     answer=f"減価償却費（資産）{dep_a:,} / 賃借料（資産）{rent_a:,}千円",
                     updates={"item10_dep": dep_a, "num_item10_dep": dep_a,
                              "item11_dep_exp": dep_e, "num_item11_dep_exp": dep_e,
                              "item8_rent": rent_a, "num_item8_rent": rent_a,
                              "item12_rent_exp": rent_e, "num_item12_rent_exp": rent_e})

    # ── STEP: credit ────────────────────────────────────────────────────────
    elif sid == "credit":
        _bot("信用情報を入力してください。")
        grade_opts = ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"]
        grade  = st.radio("格付", grade_opts,
                          index=grade_opts.index(d.get("grade","②4-6 (標準)")),
                          horizontal=True, key="wiz_grade")
        bank_c = st.number_input("うちの銀行与信（千円）", 0, 90_000_000,
                                  int(d.get("bank_credit", 0)), 100, key="wiz_bankc")
        lease_c= st.number_input("うちのリース与信（千円）", 0, 90_000_000,
                                  int(d.get("lease_credit", 0)), 100, key="wiz_leasec")
        cont   = st.number_input("契約数（件）", 0, 30, int(d.get("contracts", 1)), 1,
                                  key="wiz_cont")
        _nav_buttons(step, question="信用情報を入力してください",
                     answer=f"格付: {grade} / 銀行与信: {bank_c:,}千円 / リース与信: {lease_c:,}千円",
                     updates={"grade": grade, "bank_credit": bank_c, "num_bank_credit": bank_c,
                              "lease_credit": lease_c, "num_lease_credit": lease_c,
                              "contracts": cont, "num_contracts": cont})

    # ── STEP: contract ──────────────────────────────────────────────────────
    elif sid == "contract":
        _bot("契約条件を入力してください。")
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
                                    int(d.get("acquisition_cost", 1_000)), 100, key="wiz_acq")
        _nav_buttons(step, question="契約条件を入力してください",
                     answer=f"{cust_type} / {cont_type} / {term}ヶ月 / {acq:,}千円",
                     updates={"customer_type": cust_type, "contract_type": cont_type,
                              "deal_source": deal_src, "lease_term": term,
                              "acceptance_year": year,
                              "acquisition_cost": acq, "num_acquisition_cost": acq})

    # ── STEP: qualitative ───────────────────────────────────────────────────
    elif sid == "qualitative":
        _bot("定性スコアリングを入力してください。<br>"
             "全て任意です。分からない場合は「未選択」のままで構いません。")
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

    # ── STEP: intuition ─────────────────────────────────────────────────────
    elif sid == "intuition":
        _bot("最後に、担当者としての<b>直感スコア</b>を教えてください。<br>"
             "1＝懸念あり　→　5＝確信あり")
        score = st.slider("直感スコア", 1, 5, int(d.get("intuition_score", 3)), 1,
                          key="wiz_intuition",
                          format="%d点")
        labels = {1:"😟 懸念", 2:"🤔 やや懸念", 3:"😐 中立", 4:"🙂 やや良好", 5:"😄 確信"}
        st.caption(labels.get(score, ""))
        _nav_buttons(step, question="担当者の直感スコアを教えてください",
                     answer=f"直感スコア: {score}点 {labels.get(score,'')}",
                     updates={"intuition_score": score, "intuition": score})


# ── ナビゲーションボタン ─────────────────────────────────────────────────────
def _nav_buttons(step: int, question: str, answer: str, updates: dict) -> None:
    """「前へ」「次へ」ボタンを描画し、押されたらステップを更新する。"""
    col_back, col_next = st.columns([1, 3])
    with col_back:
        if step > 0 and st.button("← 前へ", key=f"wiz_back_{step}"):
            # 現在ステップの履歴を削除して戻る
            hist = st.session_state["wiz_history"]
            if hist:
                st.session_state["wiz_history"] = hist[:-1]
            st.session_state["wiz_step"] = step - 1
            st.rerun()
    with col_next:
        label = "審査を実行する 🚀" if step == _N_STEPS - 1 else "次へ →"
        if st.button(label, key=f"wiz_next_{step}", type="primary"):
            _advance(step, question, answer, updates)


def _advance(step: int, question: str, answer: str, updates: dict) -> None:
    """データ更新・履歴追記・ステップ進行。"""
    d = st.session_state["wiz_data"]
    d.update(updates)
    next_step = step + 1 if step < _N_STEPS - 1 else step
    _save_draft(d, step=next_step, history=st.session_state.get("wiz_history", []))

    # ユーモアコメントを付与するか
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
        # 最終ステップ → 審査実行
        _submit_wizard(d)

    st.rerun()


# ── 審査実行 ────────────────────────────────────────────────────────────────
def _submit_wizard(d: dict) -> None:
    """収集したデータを form_result 形式に変換して session_state にセット。"""
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
        "nenshu":          int(d.get("nenshuu", 0)),   # 売上高
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
        "num_competitors": d.get("num_competitors", "未入力"),
        "deal_occurrence": d.get("deal_occurrence", "不明"),
        "intuition":       int(d.get("intuition", 3)),
        # 定性スコア
        "qual_corr_company_history":    d.get("qual_corr_company_history", "未選択"),
        "qual_corr_customer_stability": d.get("qual_corr_customer_stability", "未選択"),
        "qual_corr_repayment_history":  d.get("qual_corr_repayment_history", "未選択"),
        "qual_corr_business_future":    d.get("qual_corr_business_future", "未選択"),
        "qual_corr_equipment_purpose":  d.get("qual_corr_equipment_purpose", "未選択"),
        "qual_corr_main_bank":          d.get("qual_corr_main_bank", "未選択"),
    }

    st.session_state["wizard_form_result"] = form_result

    # score_calculation.py は st.session_state から数値を読み直すため、
    # ここで全フィールドを session_state にも書き込む
    _num_keys = [
        "nenshu", "item9_gross", "rieki", "item4_ord_profit", "item5_net_income",
        "item10_dep", "item11_dep_exp", "item8_rent", "item12_rent_exp",
        "item6_machine", "item7_other", "net_assets", "total_assets",
        "bank_credit", "lease_credit", "contracts", "lease_term",
        "acquisition_cost", "acceptance_year",
    ]
    for k in _num_keys:
        if k in form_result:
            st.session_state[k] = form_result[k]

    st.session_state["_wizard_submitted"]  = True
    _clear_draft()

    # 審査・分析モードへ遷移（メインファイルが _wizard_submitted を検知して run_scoring を呼ぶ）
    st.session_state["_pending_mode"]    = "📋 審査・分析"
    st.session_state["nav_mode_widget"]  = "📊 分析結果"
    st.session_state["_jump_to_analysis"] = True
    # ウィザードのステートをリセット
    for k in ["wiz_step", "wiz_data", "wiz_history", "wiz_humor_steps",
              "wiz_major_keys"]:
        st.session_state.pop(k, None)


# ── メイン描画 ──────────────────────────────────────────────────────────────
def render_chat_wizard() -> None:
    st.markdown(_WIZ_CSS, unsafe_allow_html=True)

    jsic_data = _load_jsic()
    assets    = _load_assets()
    _init_session(jsic_data)

    step = st.session_state["wiz_step"]

    st.markdown('<div class="wiz-wrap">', unsafe_allow_html=True)

    # ヘッダー
    st.markdown(f"""
<div class="wiz-header">
  <div style="font-size:2.2rem;">🎩</div>
  <div>
    <div class="wiz-header-title">リースくん</div>
    <div class="wiz-header-sub">チャット形式でリース審査情報を収集します</div>
  </div>
</div>""", unsafe_allow_html=True)

    # プログレスバー
    pct = int(step / _N_STEPS * 100)
    st.markdown(
        f'<div class="wiz-progress-bar"><div class="wiz-progress-fill" style="width:{pct}%;"></div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"ステップ {step + 1} / {_N_STEPS}  ──  {_STEPS[step]['emoji']} {_STEPS[step]['label']}")

    # 過去ステップの履歴
    _render_history()

    # 現在のステップ
    st.markdown(f'<div class="wiz-step-label">── {_STEPS[step]["emoji"]} {_STEPS[step]["label"]}</div>',
                unsafe_allow_html=True)
    _render_step(step, jsic_data, assets)

    # リセットボタン
    st.divider()
    col_r, _ = st.columns([1, 4])
    with col_r:
        if st.button("🔄 最初からやり直す", key="wiz_reset"):
            for k in ["wiz_step","wiz_data","wiz_history","wiz_humor_steps","wiz_major_keys"]:
                st.session_state.pop(k, None)
            _clear_draft()
            st.rerun()

    # 下書き再開通知
    draft_ts = ""
    try:
        if os.path.exists(_DRAFT_PATH):
            with open(_DRAFT_PATH, encoding="utf-8") as f:
                draft_ts = json.load(f).get("ts","")[:16].replace("T"," ")
    except Exception:
        pass

    if draft_ts:
        st.caption(f"💾 下書き保存済み（{draft_ts}）— 前回の続きから再開できます")

    st.markdown('<div class="wiz-footer">リースくん v1.0 ── 温水式 リース審査AI</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
