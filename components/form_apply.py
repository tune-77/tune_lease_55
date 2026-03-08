import streamlit as st
import datetime
from constants import (
    QUALITATIVE_SCORING_CORRECTION_ITEMS,
    QUALITATIVE_SCORING_LEVELS,
)
from utils import _slider_and_number, _reset_shinsa_inputs

@st.fragment
def _fragment_nenshu():
    # 本来は lease_logic_sumaho12.py の _fragment_nenshu() で定義されているものと同じですが、
    # 依存関係を減らすため、呼び出し元から関数を渡すか、ここで再定義します。
    pass # 呼び出し側から描画関数を受け取る形で実装します。

def render_apply_form(
    jsic_data, 
    get_image,
    get_stats, 
    scrape_article_text, 
    is_japanese_text,
    append_case_news,
    fragment_nenshu_func,
    lease_assets_list
):
    """📝 審査入力画面のUIとフォームデータの収集を行います。"""
    st.header("📝 1. 審査データの入力")
    image_placeholder = st.empty()
    if 'current_image' not in st.session_state: st.session_state['current_image'] = "guide"
    img_path = get_image(st.session_state['current_image'])
    if img_path: image_placeholder.image(img_path, width=280)
    st.divider()

    selected_major = 'D 建設業'
    selected_sub = '06 総合工事業'

    # 業界・取引を expander で折りたたみ
    with st.expander("📌 業界選択・取引状況", expanded=True):
        if not jsic_data:
            st.error("業界データ(industry_trends_jsic.json)が見つかりません。")
            major_keys = ["D 建設業"]
        else:
            major_keys = list(jsic_data.keys())
        last_inp = st.session_state.get("last_submitted_inputs") or {}
        # session_state 未設定時だけ初期値を入れる（Session State API との競合回避）
        if "select_major" not in st.session_state:
            _default_major = last_inp.get("selected_major")
            st.session_state["select_major"] = _default_major if _default_major in major_keys else major_keys[0]
        # session_state の値が major_keys に含まれているか確認（リスト変動時の安全策）
        if st.session_state.get("select_major") not in major_keys:
            st.session_state["select_major"] = major_keys[0]
        selected_major = st.selectbox("大分類 (日本標準産業分類)", major_keys, key="select_major")
        if jsic_data:
            sub_data = jsic_data[selected_major]["sub"]
            sub_keys = list(sub_data.keys())
            mapped_coeff_category = jsic_data[selected_major]["mapping"]
        else:
            sub_data = {}
            sub_keys = ["06 総合工事業"]
            mapped_coeff_category = "④建設業"
        if "select_sub" not in st.session_state:
            _default_sub = last_inp.get("selected_sub")
            st.session_state["select_sub"] = _default_sub if _default_sub in sub_keys else sub_keys[0]
        if st.session_state.get("select_sub") not in sub_keys:
            st.session_state["select_sub"] = sub_keys[0]
        selected_sub = st.selectbox("中分類", sub_keys, key="select_sub")
        st.session_state["_frag_major"] = selected_major
        st.session_state["_frag_sub"] = selected_sub
        st.session_state["_frag_mapped_coeff"] = mapped_coeff_category
        st.session_state["_frag_sub_data"] = sub_data
        st.session_state["_frag_jsic_data"] = jsic_data
        trend_info = sub_data.get(selected_sub, "情報なし")
        past_stats = get_stats(selected_sub)
        past_info_text = "過去データなし"
        alert_msg = ""
        if past_stats["count"] > 0:
            past_info_text = f"過去{past_stats['count']}件 (平均: {past_stats['avg_score']:.1f}点)"
            if past_stats["close_rate"] > 0:
                past_info_text += f"\n成約率: {past_stats['close_rate']:.0%}"
            if past_stats.get("avg_winning_rate") is not None and past_stats["avg_winning_rate"] > 0:
                past_info_text += f"\n平均成約金利: {past_stats['avg_winning_rate']:.2f}%"
            if past_stats.get("top_competitors_lost"):
                past_info_text += f"\nよく負ける競合: {', '.join(past_stats['top_competitors_lost'][:3])}"
            if past_stats["avg_score"] < 40 and past_stats["count"] >= 3:
                alert_msg = "⚠️ 過去のスコアが低迷している業種です"
        st.info(f"**{selected_sub}** の動向:\n{trend_info}\n\n**当社の実績**: {past_info_text}")
        if alert_msg:
            st.warning(alert_msg)

        # 関連ニュース（表示とAI読み込みボタン）
        if 'news_results' in st.session_state and st.session_state.news_results:
            st.markdown("##### 📰 関連ニュース（AIに読み込ませる）")
            # 検索ボタンを横に配置
            col_search_btn1, col_search_btn2 = st.columns([1, 1])
            with col_search_btn1:
                if st.button("🔄 関連ニュースを再検索", key="btn_re_search_news"):
                    with st.spinner(f"「{selected_sub}」の最新ニュースを検索中..."):
                        try:
                            from web_services import search_industry_news
                            news = search_industry_news(selected_sub)
                            st.session_state.news_results = news
                            st.rerun()
                        except Exception as e:
                            st.error(f"検索エラー: {e}")
            if 'news_results' in st.session_state and st.session_state.news_results:
                for i, res in enumerate(st.session_state.news_results):
                    st.markdown(f"**[{res['title']}]({res['href']})**")
                    st.caption(res['body'])
                    if st.button(f"この記事をAIに読み込ませる", key=f"read_news_{i}"):
                        with st.spinner(f"「{res['title']}」を読み込んでいます..."):
                            content = scrape_article_text(res['href'])
                            # 日本語記事のみAIに読み込ませる
                            if content and isinstance(content, str) and not content.startswith("記事の読み込みに失敗しました"):
                                if is_japanese_text(content):
                                    news_obj = {
                                        "title": res['title'],
                                        "url": res['href'],
                                        "content": content,
                                    }
                                    st.session_state.selected_news_content = news_obj
                                    case_id = st.session_state.get("current_case_id")
                                    if case_id:
                                        append_case_news({"case_id": case_id, **news_obj})
                                    st.success("日本語記事の読み込み完了！AIへの相談・ディベート時に内容が反映されます。")
                                else:
                                    st.warning("この記事は日本語ではない可能性が高いため、AIへの読み込みをスキップしました。")
                            elif isinstance(content, str) and content.startswith("記事の読み込みに失敗しました"):
                                st.error(content)
                            else:
                                st.error("記事の本文を取得できませんでした。")
                    st.divider()
        if 'selected_news_content' in st.session_state:
            with st.container(border=True):
                st.write("📖 **現在読み込み中の記事:**")
                st.write(st.session_state.selected_news_content['title'])
                if st.button("読み込みをクリア"):
                    del st.session_state.selected_news_content
                    st.rerun()
        st.markdown("##### 🤝 取引・競合状況")
        col_q1, col_q2 = st.columns(2)
        with col_q1: main_bank = st.selectbox("取引区分", ["メイン先", "非メイン先"], key="main_bank", index=0 if (last_inp.get("main_bank") or "メイン先") == "メイン先" else 1)
        with col_q2: competitor = st.selectbox("競合状況", ["競合なし", "競合あり"], key="competitor", index=0 if (last_inp.get("competitor") or "競合なし") == "競合なし" else 1)
        # 競合ありの場合のみ「競合提示金利」を入力（金利差で成約率補正に利用）
        if competitor == "競合あり":
            comp_rate = st.number_input(
                "競合提示金利 (%)",
                min_value=0.0,
                max_value=30.0,
                value=float(st.session_state.get("competitor_rate") or 0.0),
                step=0.1,
                format="%.1f",
                key="competitor_rate_input",
                help="競合他社の提示金利を入力すると、自社が有利な場合に成約率をプラス補正します。"
            )
            st.session_state["competitor_rate"] = comp_rate if comp_rate > 0 else None
        else:
            st.session_state["competitor_rate"] = None

        # ── 契約期待度モデル用データ収集（任意入力） ────────────────────────────
        st.caption("📊 以下は「契約期待度」モデルのデータ収集用です。入力するとデータが蓄積され、将来の精度向上に役立ちます（任意）。")
        _nc_options = ["未入力", "0社（指名）", "1社", "2社", "3社以上"]
        _nc_default = last_inp.get("num_competitors", "未入力")
        _nc_idx = _nc_options.index(_nc_default) if _nc_default in _nc_options else 0
        num_competitors = st.selectbox(
            "競合社数（任意）",
            _nc_options,
            index=_nc_idx,
            key="num_competitors",
            help="他社が何社参加しているか。0社＝指名案件。将来の契約期待度モデルに使用します。"
        )
        _do_options = ["不明", "指名", "相見積もり"]
        _do_default = last_inp.get("deal_occurrence", "不明")
        _do_idx = _do_options.index(_do_default) if _do_default in _do_options else 0
        deal_occurrence = st.selectbox(
            "発生経緯（任意）",
            _do_options,
            index=_do_idx,
            key="deal_occurrence",
            help="案件の発生経緯。指名＝当社指定、相見積もり＝複数社比較。将来の契約期待度モデルに使用します。"
        )
    st.caption("💡 数字入力で画面がガタつく場合：スライダーで大まかに合わせてから直接入力で微調整してください。")
    st.caption("📌 数値とスライダーは連動します。Enter は「入力確定」にだけ効き、判定には行きません。")
    if st.button("🆕 新しく入力する", help="全フィールドを初期値にリセットします", use_container_width=False):
        _reset_shinsa_inputs()
        st.rerun()

    # ── リース物件選択（フォーム外に配置：選択した瞬間に即時反映） ──────────────
    selected_asset_id = "other"
    asset_score = 50
    asset_name = "未選択"
    asset_detail = ""
    with st.expander("📦 リース物件（選択は即時反映）", expanded=True):
        if not lease_assets_list:
            st.caption("lease_assets.json を配置すると物件リストから選択できます。")
        else:
            _a_options = [f"{it.get('name', '')}（{it.get('score', 0)}点）" for it in lease_assets_list]
            _a_default = min(st.session_state.get("selected_asset_index", 0), len(_a_options) - 1) if "selected_asset_index" in st.session_state else 0
            _a_sel = st.selectbox(
                "物件を選択（点数が判定に反映）",
                range(len(_a_options)),
                format_func=lambda i: _a_options[i],
                index=_a_default,
                key="lease_asset_select",
                help="選択した物件の点数を借手スコアに反映します。",
            )
            st.session_state["selected_asset_index"] = _a_sel
            _a_item = lease_assets_list[_a_sel]
            selected_asset_id = _a_item.get("id", "other")
            asset_score = int(_a_item.get("score", 50))
            asset_name = _a_item.get("name", "その他")
            if _a_item.get("note"):
                st.caption(f"💡 {_a_item['note']}")
            # 車両・運搬車選択時: 車種タイプ選択欄（フォーム外なので即時反映）
            if selected_asset_id == "vehicle":
                _VT_OPTS = [
                    "", "ハイエース バン（商用バン）", "キャラバン（商用バン）",
                    "軽商用バン（エブリイ / ハイゼット等）", "営業用トラック", "自家用トラック",
                    "一般乗用車（ヤリス / カローラ等）", "レクサス・外車等（役員車）",
                ]
                if "asset_vtype_select" not in st.session_state:
                    st.session_state["asset_vtype_select"] = ""
                if st.session_state.get("asset_vtype_select") not in _VT_OPTS:
                    st.session_state["asset_vtype_select"] = ""
                asset_detail = st.selectbox(
                    "🚗 車種・車両タイプ（任意）",
                    _VT_OPTS,
                    key="asset_vtype_select",
                    format_func=lambda x: "（選択なし）" if x == "" else x,
                    help="車両タイプを選択すると残価率・承認確率の推定精度が向上します。"
                         "ハイエース・キャラバンは+0.20ブースト / 役員車は要注意フラグが立ちます。",
                )
                if "役員車" in asset_detail or "レクサス" in asset_detail or "外車" in asset_detail:
                    st.warning(
                        "⚠️ 役員車・高級輸入車は事業収益を直接生み出しません。\n"
                        "審査部から使用目的を厳しく問われる可能性があります。"
                    )
                elif asset_detail:
                    st.caption(f"✅ 車両タイプ「{asset_detail}」を検出 → 専用フレーズ・残価率を適用")
            if asset_detail.strip():
                asset_name = f"{asset_name} {asset_detail.strip()}"
    # ────────────────────────────────────────────────────────────────────────

    with st.form("shinsa_form"):
        st.warning(
            "📌 **必須** 売上高・総資産は **1以上** を入力してください（未入力だと判定がブロックされます）。\n\n"
            "💡 **推奨** 営業利益・純資産も入力すると精度が向上します（未入力でも判定は続行しますが警告を表示します）。"
        )
        submitted_apply = st.form_submit_button("入力確定（Enterで反映）", type="secondary", help="数字入力でEnterを押したときはここが押された扱いになり、判定には行きません。")
        with st.expander("📊 1. 損益計算書 (P/L) ― 📌必須・💡推奨あり", expanded=True):
            # ①売上高（フラグメント化で入力時のガタつき軽減）
            # fragment_nenshu_func の内部で "### 売上高 📌 必須（1以上）" と表示するため、呼び出し前に注記なし
            fragment_nenshu_func()

            #  ②売上高総利益（スライダーは従来どおり、手入力のみ900億千円まで）
            st.markdown("### 売上高総利益")
            item9_gross = _slider_and_number("item9_gross", "sourieki", 10000, -500000, 1000000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            # #③営業利益
            st.markdown("### 営業利益 💡 推奨（未入力だと営業利益率が 0% で計算されます）")
            rieki = _slider_and_number("rieki", "rieki", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 経常利益")
            item4_ord_profit = _slider_and_number("item4_ord_profit", "item4_ord_profit", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 当期利益")
            item5_net_income = _slider_and_number("item5_net_income", "item5_net_income", 10000, -100000, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切

        with st.expander("🏢 2. 資産・経費・その他", expanded=False):
        
            st.markdown("### 減価償却費")
            item10_dep = _slider_and_number("item10_dep", "item10_dep", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 減価償却費(経費)")
            item11_dep_exp = _slider_and_number("item11_dep_exp", "item11_dep_exp", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            # #⑧賃借料
            st.markdown("### 賃借料")
            item8_rent = _slider_and_number("item8_rent", "item8_rent", 10000, 0, 100000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 賃借料（経費）")
            item12_rent_exp = _slider_and_number("item12_rent_exp", "item12_rent_exp", 10000, 0, 100000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            #⑩機械装置
            st.markdown("### 機械装置")
            item6_machine = _slider_and_number("item6_machine", "item6_machine", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### その他資産")
            item7_other = _slider_and_number("item7_other", "item7_other", 10000, 0, 200000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 純資産 💡 推奨（未入力だと自己資本比率・学習モデル精度が低下します）")
            net_assets = _slider_and_number("net_assets", "net_assets", 10000, -30000, 500000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### 総資産 📌 必須（1以上）")
            total_assets = _slider_and_number("total_assets", "total_assets", 10000, 0, 1000000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
        with st.expander("💳 3. 信用情報", expanded=False):

            # default値をリスト内の文字列と完全に一致させる必要があります
            grade = st.segmented_control("格付", ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"], default=st.session_state.get("grade", "②4-6 (標準)"), key="grade")
            st.markdown("### うちの銀行与信")
            st.caption("当社の与信です（総銀行与信ではありません）")
            bank_credit = _slider_and_number("bank_credit", "bank_credit", 10000, 0, 3000000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            st.markdown("### うちのリース与信")
            st.caption("当社の与信です（総リース与信ではありません）")
            lease_credit = _slider_and_number("lease_credit", "lease_credit", 10000, 0, 300000, 100, 1, max_val_number=90_000_000)
            st.divider() # 次の項目との区切
            # #16契約数
            st.markdown("### 契約数")
            contracts = _slider_and_number("contracts", "contracts", 1, 0, 30, 1, 1, unit="件")
            st.divider() # 次の項目との区切

        with st.expander("📋 4. 契約条件・取得価格", expanded=False):
            customer_type = st.radio("顧客区分", ["既存先", "新規先"], horizontal=True, index=0 if st.session_state.get("customer_type", "既存先") == "既存先" else 1, key="customer_type")
            st.divider()
            st.markdown("##### 📈 契約条件・属性 (利回り予測用)")
            with st.container():
                c_y1, c_y2, c_y3 = st.columns(3)
                contract_type = c_y1.radio("契約種類", ["一般", "自動車"], horizontal=True, index=0 if st.session_state.get("contract_type", "一般") == "一般" else 1, key="contract_type")
                deal_source = c_y2.radio("商談ソース", ["銀行紹介", "その他"], horizontal=True, index=0 if st.session_state.get("deal_source", "その他") == "銀行紹介" else 1, key="deal_source")
                lease_term = c_y3.select_slider("契約期間（月）", options=range(0, 121, 1), value=60)
                st.divider()
                c_l, c_r = st.columns([0.7, 0.3])
                with c_l:
                    acceptance_year = st.number_input("検収年 (西暦)", value=2026, step=1)
                st.session_state.lease_term = lease_term
                st.session_state.acceptance_year = acceptance_year
            st.markdown("### 取得価格")
            acquisition_cost = _slider_and_number("acquisition_cost", "acquisition_cost", 1000, 0, 500000, 100, 100, label_slider="取得価格調整", max_val_number=90_000_000)
            # ---------- 5. 定性スコアリング（総合×重み＋定性×重みでランクA〜E。定性未選択時は総合スコアのみ） ----------
            with st.expander("📋 定性スコアリング", expanded=False):
                st.caption("審査員が定性面を項目別に評価します。ランク（A〜E）は **総合スコア×重み＋定性×重み**（デフォルト60%/40%）で算出。定性を1件も選んでいない場合はランクは出さず、総合スコアのみで判定します。（未選択の項目は定性スコアに含めません）")
                for item in QUALITATIVE_SCORING_CORRECTION_ITEMS:
                    opts = item.get("options") or QUALITATIVE_SCORING_LEVELS
                    opts_display = ["未選択"] + [o[1] for o in opts]
                    st.selectbox(
                        f"{item['label']}（重み{item['weight']}%）",
                        range(len(opts_display)),
                        format_func=lambda i, d=opts_display: d[i],
                        key=f"qual_corr_{item['id']}",
                    )
                # 入力値は判定開始ブロックで session_state から取得
        submitted_judge = st.form_submit_button("判定開始", type="primary", use_container_width=True)

    return {
        "submitted_apply": submitted_apply,
        "submitted_judge": submitted_judge,
        "selected_major": selected_major,
        "selected_sub": selected_sub,
        "main_bank": main_bank,
        "competitor": competitor,
        "item9_gross": item9_gross,
        "rieki": rieki,
        "item4_ord_profit": item4_ord_profit,
        "item5_net_income": item5_net_income,
        "item10_dep": item10_dep,
        "item11_dep_exp": item11_dep_exp,
        "item8_rent": item8_rent,
        "item12_rent_exp": item12_rent_exp,
        "item6_machine": item6_machine,
        "item7_other": item7_other,
        "net_assets": net_assets,
        "total_assets": total_assets,
        "grade": grade,
        "bank_credit": bank_credit,
        "lease_credit": lease_credit,
        "contracts": contracts,
        "customer_type": customer_type,
        "contract_type": contract_type,
        "deal_source": deal_source,
        "lease_term": lease_term,
        "acceptance_year": acceptance_year,
        "acquisition_cost": acquisition_cost,
        "selected_asset_id": selected_asset_id if lease_assets_list else "other",
        "asset_score": asset_score if lease_assets_list else 50,
        "asset_name": asset_name if lease_assets_list else "未選択",
        "num_competitors": num_competitors,
        "deal_occurrence": deal_occurrence,
    }

def render_quick_edit_panel(jsic_data, lease_assets_list):
    """✏️ クイック再入力パネルのUIとフォームデータの収集を行います。"""
    st.caption("すべての入力項目をここから変更できます。「🔄 再判定」で即座に再計算します。")

    # ─── 業種 ───────────────────────────────────────────────
    st.markdown("#### 🏭 業種")
    _q_major_keys = list(jsic_data.keys()) if jsic_data else ["D 建設業"]
    _q_cur_major = st.session_state.get("select_major") or st.session_state.get("last_submitted_inputs", {}).get("selected_major", _q_major_keys[0])
    _q_major_idx = _q_major_keys.index(_q_cur_major) if _q_cur_major in _q_major_keys else 0
    _q_major = st.selectbox("大分類", _q_major_keys, index=_q_major_idx, key="_quick_major")
    _q_sub_keys = list(jsic_data[_q_major]["sub"].keys()) if jsic_data and _q_major in jsic_data else ["06 総合工事業"]
    _q_cur_sub = st.session_state.get("select_sub") or st.session_state.get("last_submitted_inputs", {}).get("selected_sub", _q_sub_keys[0])
    _q_sub_idx = _q_sub_keys.index(_q_cur_sub) if _q_cur_sub in _q_sub_keys else 0
    _q_sub = st.selectbox("中分類", _q_sub_keys, index=_q_sub_idx, key="_quick_sub")

    st.divider()

    # ─── 損益計算書 ─────────────────────────────────────────
    st.markdown("#### 📊 損益計算書 P/L（千円）")
    _q1, _q2, _q3 = st.columns(3)
    with _q1:
        _q_nenshu = st.number_input("売上高", min_value=0, max_value=90_000_000, value=int(st.session_state.get("nenshu", 0)), step=1000, key="_quick_nenshu")
    with _q2:
        _q_gross = st.number_input("売上総利益（粗利）", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item9_gross", 0)), step=1000, key="_quick_gross")
    with _q3:
        _q_rieki = st.number_input("営業利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("rieki", 0)), step=1000, key="_quick_rieki")
    _q4, _q5 = st.columns(2)
    with _q4:
        _q_ord = st.number_input("経常利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item4_ord_profit", 0)), step=1000, key="_quick_ord")
    with _q5:
        _q_net_income = st.number_input("当期利益", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("item5_net_income", 0)), step=1000, key="_quick_net_income")

    st.divider()

    # ─── 資産・経費 ──────────────────────────────────────────
    st.markdown("#### 🏢 資産・経費（千円）")
    _qA1, _qA2, _qA3 = st.columns(3)
    with _qA1:
        _q_dep = st.number_input("減価償却費（資産）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item10_dep", 0)), step=1000, key="_quick_dep")
        _q_dep_exp = st.number_input("減価償却費（経費）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item11_dep_exp", 0)), step=1000, key="_quick_dep_exp")
    with _qA2:
        _q_rent = st.number_input("賃借料（資産）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item8_rent", 0)), step=1000, key="_quick_rent")
        _q_rent_exp = st.number_input("賃借料（経費）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item12_rent_exp", 0)), step=1000, key="_quick_rent_exp")
    with _qA3:
        _q_machine = st.number_input("機械装置", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item6_machine", 0)), step=1000, key="_quick_machine")
        _q_other = st.number_input("その他資産", min_value=0, max_value=90_000_000, value=int(st.session_state.get("item7_other", 0)), step=1000, key="_quick_other")
    _qB1, _qB2 = st.columns(2)
    with _qB1:
        _q_net = st.number_input("純資産", min_value=-90_000_000, max_value=90_000_000, value=int(st.session_state.get("net_assets", 0)), step=1000, key="_quick_net")
    with _qB2:
        _q_total = st.number_input("総資産", min_value=0, max_value=90_000_000, value=int(st.session_state.get("total_assets", 0)), step=1000, key="_quick_total")

    st.divider()

    # ─── 信用情報 ────────────────────────────────────────────
    st.markdown("#### 💳 信用情報")
    _qC1, _qC2 = st.columns(2)
    with _qC1:
        _grade_opts = ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"]
        _q_cur_grade = st.session_state.get("grade", "②4-6 (標準)")
        _q_grade_idx = _grade_opts.index(_q_cur_grade) if _q_cur_grade in _grade_opts else 1
        _q_grade = st.selectbox("格付", _grade_opts, index=_q_grade_idx, key="_quick_grade")
        _q_bank = st.number_input("銀行与信（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("bank_credit", 0)), step=1000, key="_quick_bank")
    with _qC2:
        _q_lease = st.number_input("リース与信（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("lease_credit", 0)), step=1000, key="_quick_lease")
        _q_contracts = st.number_input("契約数（件）", min_value=0, max_value=200, value=int(st.session_state.get("contracts", 0)), step=1, key="_quick_contracts")

    st.divider()

    # ─── 契約条件 ────────────────────────────────────────────
    st.markdown("#### 📋 契約条件・物件")
    _qD1, _qD2, _qD3 = st.columns(3)
    with _qD1:
        _q_ctype = st.radio("顧客区分", ["既存先", "新規先"], index=0 if st.session_state.get("customer_type", "既存先") == "既存先" else 1, horizontal=True, key="_quick_ctype")
        _q_contract_type = st.radio("契約種類", ["一般", "自動車"], index=0 if st.session_state.get("contract_type", "一般") == "一般" else 1, horizontal=True, key="_quick_contract_type")
    with _qD2:
        _q_deal_source = st.radio("商談ソース", ["銀行紹介", "その他"], index=0 if st.session_state.get("deal_source", "その他") == "銀行紹介" else 1, horizontal=True, key="_quick_deal_source")
        _q_lease_term = st.number_input("契約期間（月）", min_value=0, max_value=120, value=int(st.session_state.get("lease_term", 0)), step=1, key="_quick_lease_term")
    with _qD3:
        _q_acceptance_year = st.number_input("検収年（西暦）", min_value=2000, max_value=2100, value=int(st.session_state.get("acceptance_year", 2026)), step=1, key="_quick_acceptance_year")
        _q_acq = st.number_input("取得価格（千円）", min_value=0, max_value=90_000_000, value=int(st.session_state.get("acquisition_cost", 0)), step=100, key="_quick_acq")
    _q_asset_detail = ""
    if lease_assets_list:
        _q_asset_opts = [f"{it.get('name', '')}（{it.get('score', 0)}点）" for it in lease_assets_list]
        _q_asset_idx = min(st.session_state.get("selected_asset_index", 0), len(_q_asset_opts) - 1)
        _q_asset_sel = st.selectbox("リース物件", range(len(_q_asset_opts)), format_func=lambda i: _q_asset_opts[i], index=_q_asset_idx, key="_quick_asset")
        # 車両・運搬車選択時: 車種タイプ選択欄
        if lease_assets_list[_q_asset_sel].get("id") == "vehicle":
            _VEHICLE_TYPE_OPTIONS_Q = [
                "",
                "ハイエース バン（商用バン）",
                "キャラバン（商用バン）",
                "軽商用バン（エブリイ / ハイゼット等）",
                "営業用トラック",
                "自家用トラック",
                "一般乗用車（ヤリス / カローラ等）",
                "レクサス・外車等（役員車）",
            ]
            if st.session_state.get("_quick_asset_detail") not in _VEHICLE_TYPE_OPTIONS_Q:
                st.session_state["_quick_asset_detail"] = ""
            _q_asset_detail = st.selectbox(
                "🚗 車種・車両タイプ（任意）",
                _VEHICLE_TYPE_OPTIONS_Q,
                key="_quick_asset_detail",
                format_func=lambda x: "（選択なし）" if x == "" else x,
                help="車両タイプを選択すると残価率・承認確率の推定精度が向上します。",
            )
            if "役員車" in _q_asset_detail or "レクサス" in _q_asset_detail or "外車" in _q_asset_detail:
                st.warning("⚠️ 役員車・高級輸入車は事業収益を直接生み出しません。使用目的を稟議書に明記してください。")
            elif _q_asset_detail:
                st.caption(f"✅ 車両タイプ「{_q_asset_detail}」を検出 → 専用フレーズ・残価率を適用")
    else:
        _q_asset_sel = None

    st.divider()

    # ─── 定性スコアリング ────────────────────────────────────
    st.markdown("#### 📝 定性スコアリング")
    _q_qual = {}
    for _qi, _qitem in enumerate(QUALITATIVE_SCORING_CORRECTION_ITEMS):
        _qopts = _qitem.get("options") or QUALITATIVE_SCORING_LEVELS
        _qopts_display = ["未選択"] + [o[1] for o in _qopts]
        _qcur = st.session_state.get(f"qual_corr_{_qitem['id']}", 0)
        _q_qual[_qitem["id"]] = st.selectbox(
            f"{_qitem['label']}（重み{_qitem['weight']}%）",
            range(len(_qopts_display)),
            format_func=lambda i, d=_qopts_display: d[i],
            index=_qcur,
            key=f"_quick_qual_{_qitem['id']}",
        )

    st.divider()

    # 再判定ボタンと更新状態を返す
    rejudge_clicked = st.button("🔄 再判定", type="primary", use_container_width=True)
    
    return {
        "rejudge_clicked": rejudge_clicked,
        "q_major": _q_major, "q_sub": _q_sub,
        "q_nenshu": _q_nenshu, "q_gross": _q_gross, "q_rieki": _q_rieki,
        "q_ord": _q_ord, "q_net_income": _q_net_income,
        "q_dep": _q_dep, "q_dep_exp": _q_dep_exp,
        "q_rent": _q_rent, "q_rent_exp": _q_rent_exp,
        "q_machine": _q_machine, "q_other": _q_other,
        "q_net": _q_net, "q_total": _q_total,
        "q_grade": _q_grade, "q_bank": _q_bank, "q_lease": _q_lease, "q_contracts": _q_contracts,
        "q_ctype": _q_ctype, "q_contract_type": _q_contract_type, "q_deal_source": _q_deal_source,
        "q_lease_term": _q_lease_term, "q_acceptance_year": _q_acceptance_year, "q_acq": _q_acq,
        "q_asset_sel": _q_asset_sel,
        "q_asset_detail": _q_asset_detail,
        "q_qual": _q_qual
    }
