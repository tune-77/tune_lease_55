"""
⚔️ 軍師モード — UI層（render_gunshi / render_gunshi_in_results 等）

shinsa_gunshi_logic.py（ロジック層）からインポートして使用する。
"""
from __future__ import annotations

import os
import time
from typing import Generator

import streamlit as st

from shinsa_gunshi_logic import (
    _get_gemini_key,
    _get_asset_market_ctx,
    _learn_evidence_weights_from_db,
    refresh_evidence_weights,
    compute_prior,
    compute_posterior,
    select_top_phrases,
    _gemini_generate,
    _gemini_first_stream,
    _ollama_stream,
    build_gunshi_prompt,
    generate_counter_offers,
    GUNSHI_DB_PATH,
    OLLAMA_BASE,
    OLLAMA_CHAT_URL,
    OLLAMA_STREAM_URL,
    DEFAULT_MODEL,
)
from components.shinsa_gunshi_db import (
    init_db,
    save_case,
    update_result,
    load_history,
    get_success_patterns,
)

def render_gunshi() -> None:
    """軍師モード メイン UI"""
    # DB 初期化
    init_db()

    # ─── カラーテーマ CSS ────────────────────────────────────────────────
    st.markdown("""
    <style>
    :root {
        --navy:  #1e3a5f;
        --green: #2d8a4e;
        --cream: #f7f7f2;
    }
    .gunshi-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a8e 100%);
        color: #f7f7f2;
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }
    .gunshi-header h2 { color: #f7f7f2 !important; margin: 0; }
    .phrase-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border-left: 5px solid #2d8a4e;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        box-shadow: 0 2px 6px rgba(45,138,78,0.10);
    }
    .counter-card {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border-left: 5px solid #f97316;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .prob-bar-wrap {
        background: #e2e8f0;
        border-radius: 999px;
        height: 22px;
        overflow: hidden;
        margin: 0.4rem 0 1rem 0;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.8s ease;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 0.5rem;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="gunshi-header">
        <h2>⚔️ 軍師モード — 審査承認奪取システム</h2>
        <p style="margin:0.3rem 0 0 0; opacity:0.85; font-size:0.9rem;">
            逐次ベイズ学習 × 100選爆速表示 × ローカルLLMストリーミング
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─── 2カラム構成 ────────────────────────────────────────────────────
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📋 案件データ入力")

        industry_cat = st.selectbox(
            "業種カテゴリ",
            ["運送業", "製造業", "医療法人"],
            key="gu_industry",
        )
        industry_detail = st.text_input(
            "詳細業種・物件名（任意）",
            placeholder="例: 冷凍食品輸送 / 40tトラック",
            key="gu_industry_detail",
        )

        score = st.slider(
            "スコア（0〜100）",
            min_value=0, max_value=100, value=55, step=1,
            key="gu_score",
        )
        pd_pct = st.slider(
            "PD（デフォルト確率 %）",
            min_value=0.0, max_value=30.0, value=8.0, step=0.5,
            key="gu_pd",
        )

        resale = st.radio(
            "中古市場リセール評価",
            ["高", "中", "低"],
            horizontal=True,
            key="gu_resale",
        )
        repeat_cnt = st.slider(
            "過去リピート回数（遅延なし）",
            min_value=0, max_value=10, value=0, step=1,
            key="gu_repeat",
        )
        subsidy = st.checkbox(
            "✅ 補助金採択・公的支援あり",
            key="gu_subsidy",
        )
        bank = st.checkbox(
            "✅ メイン銀行の支援あり",
            key="gu_bank",
        )
        intuition = st.slider(
            "担当者の直感（1=懸念 〜 5=確信）",
            min_value=1, max_value=5, value=3, step=1,
            key="gu_intuition",
        )

        # モデル選択
        with st.expander("🤖 LLM設定", expanded=False):
            model_name = st.text_input(
                "Ollama モデル名",
                value=st.session_state.get("ollama_model", DEFAULT_MODEL),
                key="gu_model",
                placeholder="llama3 / qwen2.5 / mistral など",
            )

        run_btn = st.button(
            "⚔️ 軍師に分析を依頼する",
            type="primary",
            width='stretch',
            key="gu_run",
        )

    # ─── 結果エリア ──────────────────────────────────────────────────────
    with col_result:
        st.subheader("🎯 分析結果")

        if run_btn:
            # ── ベイズ計算 ──
            prior = compute_prior(score, pd_pct)
            patterns = get_success_patterns(industry_cat)
            posterior = compute_posterior(
                prior=prior,
                resale=resale,
                repeat_cnt=repeat_cnt,
                subsidy=subsidy,
                bank=bank,
                intuition=intuition,
                success_ratio=patterns["success_ratio"],
                similar_case_count=patterns["wins"],
            )

            # フレーズ選択
            top_phrases = select_top_phrases(
                industry_cat=industry_cat,
                score=score,
                pd_pct=pd_pct,
                resale=resale,
                repeat_cnt=repeat_cnt,
                subsidy=subsidy,
                bank=bank,
                posterior=posterior,
                n=3,
            )

            # ── Step 1: 即時フレーズ表示 ──
            st.markdown("#### ⚡ Step 1: 爆速・最強フレーズ（即時）")
            for i, phrase in enumerate(top_phrases, 1):
                boost_pct = int(phrase["prob_boost"] * 100)
                st.markdown(
                    f'<div class="phrase-card">'
                    f'<strong>#{i} [{phrase["category"]}]</strong> '
                    f'<span style="color:#2d8a4e;font-size:0.8rem;">承認確率 +{boost_pct}%</span><br>'
                    f'{phrase["text"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # 承認確率バー（フレーズ分の boost を反映）
            phrase_boost_total = sum(p["prob_boost"] for p in top_phrases)
            display_prob = min(0.99, posterior + phrase_boost_total * 0.3)
            pct = int(display_prob * 100)
            bar_color = (
                "#2d8a4e" if pct >= 70
                else "#f97316" if pct >= 50
                else "#ef4444"
            )
            bar_label = (
                "✅ 承認圏内" if pct >= 70
                else "⚠️ 要審議" if pct >= 50
                else "❌ 再考必要"
            )
            st.markdown(f"""
            <div style="margin:0.8rem 0 0.3rem 0">
                <strong>ベイズ推定 承認確率</strong>&nbsp;
                <span style="font-size:1.5rem;font-weight:700;color:{bar_color}">{pct}%</span>
                &nbsp;<span style="color:{bar_color}">{bar_label}</span>
            </div>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill"
                    style="width:{pct}%;background:{bar_color}">
                    {pct}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f"事前確率 **{prior*100:.1f}%** → "
                f"証拠統合後 **{posterior*100:.1f}%** → "
                f"フレーズ演出込み **{pct}%**"
            )

            # カウンターオファー（85%未満の場合）
            if posterior < 0.85:
                st.markdown("---")
                st.markdown("#### 🔄 逆転の条件（カウンターオファー）")
                st.caption("承認確率が70%未満のため、以下の条件を追加提示することで一気に承認圏内へ！")
                offers = generate_counter_offers(
                    posterior=posterior,
                    resale=resale,
                    repeat_cnt=repeat_cnt,
                    subsidy=subsidy,
                    bank=bank,
                    pd_pct=pd_pct,
                    score=score,
                )
                for o in offers:
                    new_p = min(0.99, posterior + o["prob_gain"])
                    st.markdown(
                        f'<div class="counter-card">'
                        f'<strong>{o["title"]}</strong>'
                        f'<span style="float:right;color:#f97316;font-weight:700;">'
                        f'+{int(o["prob_gain"]*100)}% → {int(new_p*100)}%</span><br>'
                        f'<span style="font-size:0.85rem">{o["detail"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # ── Step 2: 推論演出ログ ──
            st.markdown("---")
            st.markdown("#### 🔍 Step 2: 深層スキャン")
            with st.status("データベースと業界データをスキャン中...", expanded=True) as status:
                st.write(f"📂 過去の成約データをスキャン中... ({patterns['total']} 件を確認)")
                time.sleep(0.4)
                st.write(f"📊 {industry_cat}の同業種成約率: {patterns['success_ratio']*100:.1f}%")
                time.sleep(0.4)
                st.write(f"🏷️ 中古市場の流動性を計算中... リセール評価: {resale}")
                time.sleep(0.4)
                st.write(f"🔢 ベイズ後験確率を更新中: {prior*100:.1f}% → {posterior*100:.1f}%")
                time.sleep(0.3)
                if patterns["success_samples"]:
                    st.write(f"✅ 類似成約事例 {len(patterns['success_samples'])} 件を参照済み")
                    time.sleep(0.3)
                if patterns["fail_samples"]:
                    st.write(f"⚠️ 過去の非成約事例 {len(patterns['fail_samples'])} 件の教訓を適用")
                    time.sleep(0.2)
                st.write("📝 軍師プロンプトを構築中...")
                time.sleep(0.3)
                st.write("🤖 ローカルLLMを起動中...")
                status.update(label="スキャン完了。LLM推論を開始します。", state="complete")

            # ── Step 3: LLM ストリーミング出力 ──
            st.markdown("---")
            st.markdown("#### 🧠 Step 3: 軍師からの推薦文（LLM生成）")

            model_to_use = st.session_state.get("gu_model", DEFAULT_MODEL) or DEFAULT_MODEL
            _res_for_prompt = st.session_state.get("last_result") or {}
            prompt = build_gunshi_prompt(
                industry=f"{industry_cat}（{industry_detail}）" if industry_detail else industry_cat,
                score=score,
                pd_pct=pd_pct,
                resale=resale,
                repeat_cnt=repeat_cnt,
                subsidy=subsidy,
                bank=bank,
                intuition=intuition,
                posterior=posterior,
                success_patterns=patterns,
                top_phrases=top_phrases,
                trend_info=st.session_state.get("_gunshi_trend_300", ""),
                comparison_text=_res_for_prompt.get("comparison", ""),
                humor_style=st.session_state.get("humor_style", "standard"),
                asset_market_context=_get_asset_market_ctx(),
            )

            llm_placeholder = st.empty()

            # ストリーミング試行（Ollama が落ちていてもエラーにしない）
            full_text = ""
            try:
                def _stream_gen():
                    yield from _ollama_stream(prompt, model_to_use)

                full_text = st.write_stream(_stream_gen())
            except Exception as e:
                # LLM が動いていない場合は定型文フォールバック
                fallback = (
                    f"【軍師の推薦文（定型文フォールバック）】\n\n"
                    f"本案件（{industry_cat}）は承認を強く推奨します。\n"
                    f"スコア{score:.0f}点・PD{pd_pct:.1f}%という数値は、\n"
                    + top_phrases[0]["text"] + "\n\n"
                    + top_phrases[1]["text"] + "\n\n"
                    "以上の観点から、本案件のリスクは定性的強みで完全に相殺されており、"
                    "速やかな承認決定を推薦いたします。\n\n"
                    f"⚠️ LLM接続エラー: {e}\n"
                    "Ollama が起動していることを確認してください（`ollama serve`）"
                )
                llm_placeholder.markdown(fallback)
                full_text = fallback

            # ── 案件をDBに保存 ──
            case_data = {
                "industry": industry_cat,
                "score": score,
                "pd_pct": pd_pct,
                "resale": resale,
                "repeat_cnt": repeat_cnt,
                "subsidy": subsidy,
                "bank": bank,
                "intuition": intuition,
                "prior_prob": prior,
                "posterior": posterior,
                "result": "未登録",
            }
            new_case_id = save_case(case_data)
            st.session_state["gunshi_last_case_id"] = new_case_id
            st.session_state["gunshi_last_posterior"] = posterior

            st.info(
                f"案件 ID: **{new_case_id}** として保存しました。"
                "下のセクションで成約/非成約を登録してください。"
            )

        # ── 結果登録エリア ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📝 結果登録（逐次学習）")
        st.caption(
            "成約・非成約を登録するたびに、AIの推薦精度が自動向上します。"
        )

        last_id = st.session_state.get("gunshi_last_case_id")
        # ── 週次戦略の表示 ──
        _render_weekly_strategy_panel()

        if last_id:
            col_r1, col_r2 = st.columns(2)
            notes_input = st.text_input(
                "備考（否決理由や成約のポイントなど）",
                key="gu_notes",
                placeholder="例: 期間短縮条件を提示して承認 / 財務指標が基準未達",
            )
            with col_r1:
                if st.button(
                    "🎉 成約で登録",
                    type="primary",
                    width='stretch',
                    key="gu_reg_win",
                ):
                    update_result(last_id, "成約", notes_input)
                    st.success(f"案件 {last_id}: 成約を登録しました。学習データに反映されます。")
                    st.session_state.pop("gunshi_last_case_id", None)
                    st.rerun()
            with col_r2:
                if st.button(
                    "💔 非成約で登録",
                    width='stretch',
                    key="gu_reg_lose",
                ):
                    update_result(last_id, "非成約", notes_input)
                    st.warning(f"案件 {last_id}: 非成約を登録しました。次回の推薦に改善が反映されます。")
                    st.session_state.pop("gunshi_last_case_id", None)
                    st.rerun()
        else:
            st.caption("上で「軍師に分析を依頼する」を実行すると、結果登録が有効になります。")

    # ─── 履歴・学習データダッシュボード ─────────────────────────────────
    st.markdown("---")
    with st.expander("📊 学習履歴ダッシュボード（全業種）", expanded=False):
        hist = load_history(100)
        if not hist:
            st.info("まだ案件データがありません。分析を実行して学習を開始してください。")
        else:
            import pandas as pd

            df = pd.DataFrame(hist)
            df["created_at"] = pd.to_datetime(df["created_at"])

            # KPI
            total = len(df)
            wins = (df["result"] == "成約").sum()
            loses = (df["result"] == "非成約").sum()
            pending = (df["result"] == "未登録").sum()
            win_rate = wins / (wins + loses) * 100 if (wins + loses) else 0

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("総案件数", total)
            k2.metric("成約", wins)
            k3.metric("非成約", loses)
            k4.metric("成約率", f"{win_rate:.1f}%")

            # 業種別集計
            st.markdown("**業種別成約率**")
            by_ind = (
                df[df["result"].isin(["成約", "非成約"])]
                .groupby(["industry", "result"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            if not by_ind.empty:
                st.dataframe(by_ind, width='stretch', hide_index=True)

            # 最新履歴テーブル
            st.markdown("**最新20件**")
            display_cols = ["id", "created_at", "industry", "score", "pd_pct",
                            "resale", "repeat_cnt", "posterior", "result"]
            show_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(
                df[show_cols].head(20).rename(columns={
                    "id": "ID", "created_at": "日時", "industry": "業種",
                    "score": "スコア", "pd_pct": "PD%", "resale": "リセール",
                    "repeat_cnt": "リピート", "posterior": "承認確率", "result": "結果",
                }),
                width='stretch',
                hide_index=True,  # noqa: keep trailing comma
            )


# ==============================================================================
# 自動抽出ヘルパー（審査結果 res から軍師データを組み立てる）
# ==============================================================================

# 業種大分類 → 軍師カテゴリ マッピング
# ※ JSICデータの実際のキーは「・」（中点）を使用
_INDUSTRY_MAJOR_MAP: dict[str, str] = {
    # 「・」区切り（実際のJSIC industry_trends_jsic.json キー）
    "H 運輸業・郵便業":            "運送業",
    "E 製造業":                    "製造業",
    "P 医療・福祉":                "医療法人",
    "D 建設業":                    "汎用",
    "I 卸売業・小売業":            "汎用",
    "K 不動産業・物品賃貸業":      "汎用",
    "M 宿泊業・飲食サービス業":    "汎用",
    "R サービス業(他に分類されないもの)": "汎用",
    # 「、」区切りフォールバック（旧データ互換）
    "H 運輸業、郵便業":            "運送業",
    "P 医療、福祉":                "医療法人",
}

# 物件名キーワード → リセール評価
# ※ 上位エントリを優先マッチするため、具体的な車種名を先に配置すること
_ASSET_RESALE_MAP: list[tuple[list[str], str]] = [
    # 普通商用バン（ハイエース・キャラバンクラス）― 5年落ちでも60-70%残存、実質担保最高位
    (["ハイエース", "キャラバン", "ハイエースバン"], "高"),
    # 軽商用バン（エブリイ・ハイゼット等）― 3年落ち65-75%、"動けば買い手がつく"資産
    (["エブリイ", "エブリィ", "ハイゼット", "NV100", "バネット", "アトレー"], "高"),
    # 営業用トラック ― 稼働率高・業務直結で流動性良好
    (["営業用トラック"], "高"),
    # 自家用トラック ― 稼働確認要・流動性は営業用より低め
    (["自家用トラック"], "中"),
    # 役員車・高級輸入車（レクサス・外車等）― 事業収益を生まず担保価値も不安定
    (["役員車", "レクサス", "外車", "輸入車", "LEXUS", "ベンツ", "BMW", "アウディ", "メルセデス"], "中"),
    # 一般営業用コンパクトカー（ヤリス・ノート等）― 3年落ち45-55%、流動性高
    (["ヤリス", "ノート", "フィット", "アクア", "カローラ", "ヴィッツ", "マーチ", "スイフト"], "中"),
    # 汎用車両・重機
    (["車両", "トラック", "バン", "ダンプ", "乗用", "フォーク", "リフト"], "高"),
    (["工作機械", "マシニング", "旋盤", "プレス", "溶接", "ロボット"],     "高"),
    (["医療", "MRI", "CT", "超音波", "内視鏡", "X線", "レントゲン"],      "高"),
    (["建機", "ショベル", "クレーン", "ブルドーザー", "アスファルト"],      "中"),
    (["ドローン", "無人機"],                                               "中"),
    (["PC", "サーバー", "タブレット", "スマホ", "IT", "コンピュータ"],      "低"),
]

# 車種別ベイズ追加ブースト（_ASSET_RESALE_MAP のリセール評価に加えて上乗せ）
# (キーワードリスト, 追加ブースト値, 説明ラベル)
_VEHICLE_EXTRA_BOOST: list[tuple[list[str], float, str]] = [
    (["ハイエース", "キャラバン"],
     0.20, "普通商用バン（ハイエースクラス）― 5年落ち60%超/LGDほぼゼロ"),
    (["エブリイ", "エブリィ", "ハイゼット", "NV100", "バネット", "アトレー"],
     0.15, "軽商用バン ― 3年落ち65-75%/動けば即買い手"),
    (["ヤリス", "ノート", "フィット", "アクア", "カローラ", "ヴィッツ", "マーチ", "スイフト"],
     0.05, "一般営業用コンパクトカー ― 流動性高・早期回収容易"),
    (["営業用トラック"],
     0.10, "営業用トラック ― 業務直結・稼働率高・物件スコア安定"),
    (["自家用トラック"],
     0.04, "自家用トラック ― 業務使用目的を確認・流動性は営業用より低"),
    # 役員車はブースト0（警告フレーズで対処）
    (["役員車", "レクサス", "外車", "LEXUS"],
     0.00, "役員車・高級輸入車 ― 事業収益への直接貢献なし（要注意）"),
]

# 役員車・高級外車の検出キーワード
_EXEC_CAR_KEYWORDS: list[str] = [
    "役員車", "レクサス", "LEXUS", "外車", "輸入車",
    "ベンツ", "BMW", "アウディ", "AUDI", "メルセデス", "MERCEDES", "BENZ",
    "ポルシェ", "PORSCHE", "フェラーリ",
]

# 役員車・高級外車案件への警告フレーズ（最終コメントに強制挿入）
_EXEC_CAR_WARNING_PHRASES: list[dict] = [
    {
        "id": "EX01",
        "text": (
            "⚠️【要注意】役員車（レクサス・外車等）は事業収益を直接生み出しません。"
            "「なぜ会社にとって必要な資産か」を稟議書に明記し、審査部の疑念を事前に封じてください。"
            "お金を産まない投資である点を十分に考慮した上で、使用目的の明確化が必須です。"
        ),
        "tags": ["役員車", "警告", "稟議", "使用目的"],
        "prob_boost": 0.0,
        "category": "役員車警告",
    },
    {
        "id": "EX02",
        "text": (
            "高級輸入車・役員車リースは審査部から『お金を生まない投資』と見られます。"
            "期間短縮（法定耐用年数の60%以下）＋前受金3ヶ月分のセット提案で印象を改善し、"
            "リスクを最小化してください。"
        ),
        "tags": ["役員車", "期間短縮", "前受"],
        "prob_boost": 0.0,
        "category": "役員車警告",
    },
    {
        "id": "EX03",
        "text": (
            "役員車は中古市場での流動性が商用車より低く担保価値も不安定です。"
            "万一の場合に備え、保証金の積み増しまたは前受金を設定した上で"
            "承認稟議書に使用目的を詳細に記載することを強く推奨します。"
        ),
        "tags": ["役員車", "担保", "前受"],
        "prob_boost": 0.0,
        "category": "役員車警告",
    },
]


def _resale_from_asset_name(asset_name: str) -> str:
    """物件名からリセール評価（高/中/低）を推定する。"""
    name = (asset_name or "").upper()
    for keywords, level in _ASSET_RESALE_MAP:
        if any(k.upper() in name for k in keywords):
            return level
    return "中"  # デフォルト


def _vehicle_boost_from_asset_name(asset_name: str) -> tuple[float, str]:
    """物件名から車種別ベイズ追加ブースト値と説明ラベルを返す。"""
    name = (asset_name or "").upper()
    for keywords, boost, description in _VEHICLE_EXTRA_BOOST:
        if any(k.upper() in name for k in keywords):
            return boost, description
    return 0.0, ""


def _repeat_from_qualitative(res: dict) -> int:
    """qualitative_scoring_correction の repayment_history スコアをリピート回数に変換。"""
    qsc = res.get("qualitative_scoring_correction") or {}
    item = qsc.get("items", {}).get("repayment_history", {}) if "items" in qsc else qsc.get("repayment_history", {})
    score_val = item.get("score", None) if isinstance(item, dict) else None
    mapping = {4: 5, 3: 3, 2: 1, 1: 0, 0: 0}
    return mapping.get(int(score_val), 0) if score_val is not None else 0


def _subsidy_from_res(res: dict, submitted_inputs: dict | None) -> bool:
    """補助金採択フラグを res / submitted_inputs から取得。"""
    tags = res.get("strength_tags") or []
    if any("補助金" in t for t in tags):
        return True
    if submitted_inputs:
        for v in submitted_inputs.values():
            if isinstance(v, str) and "補助金" in v:
                return True
    return False


def _bank_from_res(res: dict, submitted_inputs: dict | None) -> bool:
    """メイン銀行支援フラグを取得。"""
    main_bank = res.get("main_bank", "") or ""
    if "メイン" in main_bank:
        return True
    if submitted_inputs:
        mb = submitted_inputs.get("main_bank", "")
        if mb and "メイン" in str(mb):
            return True
    return False


def compute_gunshi_from_res(
    res: dict,
    submitted_inputs: dict | None = None,
    bn_evidence: dict | None = None,
    bn_approval_prob: float | None = None,
) -> dict:
    """
    審査結果 res（st.session_state["last_result"]）から軍師モードの入力を自動抽出し、
    ベイズ推論・フレーズ選択・カウンターオファーを実行して結果 dict を返す。

    Returns:
        {
            "score": float,
            "pd_pct": float,
            "industry_cat": str,
            "resale": str,
            "repeat_cnt": int,
            "subsidy": bool,
            "bank": bool,
            "intuition": int,        # 直感スコア（submitted_inputs["intuition"] 優先、なければ3）
            "prior": float,
            "posterior": float,
            "display_prob": float,
            "top_phrases": list[dict],
            "offers": list[dict],
            "patterns": dict,
        }
    """
    init_db()

    score   = float(res.get("score", 0))
    pd_pct  = float(res.get("pd_percent", 0))

    # 業種カテゴリ
    industry_major = res.get("industry_major", "")
    industry_cat   = _INDUSTRY_MAJOR_MAP.get(industry_major, "汎用")

    # 物件名からリセール評価
    asset_name = res.get("asset_name", "") or ""
    resale = _resale_from_asset_name(asset_name)

    # 車種別ベイズ追加ブースト
    vehicle_boost, vehicle_type = _vehicle_boost_from_asset_name(asset_name)

    # リピート回数：フォーム入力の実契約件数を優先、なければ返済履歴スコアから推定
    _contracts_val = (
        res.get("contracts")                                    # scoring_core が res に格納した値
        if res.get("contracts") is not None
        else (submitted_inputs or {}).get("contracts")          # フォーム submitted_inputs から
    )
    if _contracts_val is not None:
        try:
            repeat_cnt = int(_contracts_val)
        except (TypeError, ValueError):
            repeat_cnt = _repeat_from_qualitative(res)
    else:
        repeat_cnt = _repeat_from_qualitative(res)             # どちらもなければ定性スコアから推定

    # 補助金・銀行支援
    subsidy = _subsidy_from_res(res, submitted_inputs)
    bank    = _bank_from_res(res, submitted_inputs)

    # 直感スコア：submitted_inputs に入力値があれば使用、なければ3（ニュートラル）
    _raw_intuition = (submitted_inputs or {}).get("intuition")
    if _raw_intuition is None:
        _raw_intuition = (submitted_inputs or {}).get("intuition_score")  # セッション保存キーにも対応
    try:
        intuition = max(1, min(5, int(_raw_intuition))) if _raw_intuition is not None else 3
    except (TypeError, ValueError):
        intuition = 3

    # ベイズ計算
    prior    = compute_prior(score, pd_pct)
    patterns = get_success_patterns(industry_cat)
    posterior = compute_posterior(
        prior=prior,
        resale=resale,
        repeat_cnt=repeat_cnt,
        subsidy=subsidy,
        bank=bank,
        intuition=intuition,
        success_ratio=patterns["success_ratio"],
        similar_case_count=patterns["wins"],
    )

    # 車種別追加ブーストを事後確率に上乗せ
    posterior = min(0.99, posterior + vehicle_boost)

    # フレーズ選択（asset_name を渡して商用車フレーズを優先、BN条件があれば連携）
    top_phrases = select_top_phrases(
        industry_cat=industry_cat,
        score=score,
        pd_pct=pd_pct,
        resale=resale,
        repeat_cnt=repeat_cnt,
        subsidy=subsidy,
        bank=bank,
        posterior=posterior,
        asset_name=asset_name,
        n=3,
        bn_evidence=bn_evidence,
    )

    phrase_boost_total = sum(p["prob_boost"] for p in top_phrases)
    # BN逆転シミュレータでチェックが入っている場合は BN の承認確率を優先
    _bn_active = bn_evidence and any(v == 1 for v in bn_evidence.values())
    if _bn_active and bn_approval_prob is not None:
        display_prob = min(0.99, bn_approval_prob + phrase_boost_total * 0.3)
    else:
        display_prob = min(0.99, posterior + phrase_boost_total * 0.3)

    # カウンターオファー（BNチェック済み条件は除外）
    offers = generate_counter_offers(
        posterior=posterior,
        resale=resale,
        repeat_cnt=repeat_cnt,
        subsidy=subsidy,
        bank=bank,
        pd_pct=pd_pct,
        score=score,
        bn_evidence=bn_evidence,
    ) if posterior < 0.85 else []

    # 役員車フラグ（UIでの警告表示に使用）
    exec_car = any(k.upper() in (asset_name or "").upper() for k in _EXEC_CAR_KEYWORDS)

    return {
        "score":         score,
        "pd_pct":        pd_pct,
        "industry_cat":  industry_cat,
        "asset_name":    asset_name,
        "resale":        resale,
        "repeat_cnt":    repeat_cnt,
        "subsidy":       subsidy,
        "bank":          bank,
        "intuition":     intuition,
        "prior":         prior,
        "posterior":     posterior,
        "display_prob":  display_prob,
        "top_phrases":   top_phrases,
        "offers":        offers,
        "patterns":      patterns,
        "vehicle_type":  vehicle_type,
        "vehicle_boost": vehicle_boost,
        "exec_car":      exec_car,
    }


# ==============================================================================
# 審査結果画面に埋め込むUI（入力不要・resから全自動）
# ==============================================================================

_GUNSHI_CSS = """
<style>
.gunshi-phrase-card {
    background: linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
    border-left: 5px solid #2d8a4e;
    padding:.8rem 1.1rem;
    border-radius:8px;
    margin-bottom:.5rem;
    font-size:.93rem;
    box-shadow:0 2px 6px rgba(45,138,78,.10);
}
.gunshi-counter-card {
    background: linear-gradient(135deg,#fff7ed 0%,#ffedd5 100%);
    border-left: 5px solid #f97316;
    padding:.8rem 1.1rem;
    border-radius:8px;
    margin-bottom:.5rem;
    font-size:.88rem;
}
.gunshi-prob-bar-wrap {
    background:#e2e8f0;border-radius:999px;height:20px;overflow:hidden;margin:.3rem 0 .8rem 0;
}
.gunshi-prob-bar-fill {
    height:100%;border-radius:999px;display:flex;align-items:center;
    justify-content:flex-end;padding-right:.5rem;font-size:.78rem;
    font-weight:700;color:white;
}
</style>
"""


def render_gunshi_in_results(
    res: dict,
    submitted_inputs: dict | None = None,
    model_name: str = DEFAULT_MODEL,
    bn_evidence: dict | None = None,
    bn_approval_prob: float | None = None,
) -> dict | None:
    """
    分析結果画面内に軍師セクションを表示する（サイドバー入力不要版）。
    st.session_state["gunshi_auto_result"] にキャッシュして返す。
    PDF 用の dict も返す。
    """
    st.markdown(_GUNSHI_CSS, unsafe_allow_html=True)

    # ── 計算（キャッシュ利用） ──────────────────────────────────────
    cache_key = "gunshi_auto_result"
    last_score   = st.session_state.get("_gunshi_cache_score")
    last_bn_hash = st.session_state.get("_gunshi_cache_bn_hash")
    cur_score    = res.get("score", 0)
    cur_bn_hash  = hash((frozenset((bn_evidence or {}).items()), bn_approval_prob))
    if cache_key not in st.session_state or last_score != cur_score or last_bn_hash != cur_bn_hash:
        with st.spinner("軍師データを計算中..."):
            st.session_state[cache_key] = compute_gunshi_from_res(
                res, submitted_inputs,
                bn_evidence=bn_evidence,
                bn_approval_prob=bn_approval_prob,
            )
            st.session_state["_gunshi_cache_score"]   = cur_score
            st.session_state["_gunshi_cache_bn_hash"] = cur_bn_hash

    g = st.session_state[cache_key]
    pct       = int(g["display_prob"] * 100)
    posterior = g["posterior"]
    bar_color = "#2d8a4e" if pct >= 70 else ("#f97316" if pct >= 50 else "#ef4444")
    bar_label = "✅ 承認圏内" if pct >= 70 else ("⚠️ 要審議" if pct >= 50 else "❌ 再考必要")

    # ── 自動抽出パラメータ表示 ─────────────────────────────────────
    with st.expander("📌 自動抽出パラメータ（クリックで確認）", expanded=False):
        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        pc1.metric("スコア",         f"{g['score']:.0f}")
        pc2.metric("PD",             f"{g['pd_pct']:.1f}%")
        pc3.metric("業種カテゴリ",   g["industry_cat"])
        pc4.metric("リセール評価",   g["resale"])
        pc5.metric("リピート回数",   g["repeat_cnt"])
        pc6.metric("補助金",         "あり" if g["subsidy"] else "なし")

    # ── Step 1: 最強フレーズ（即時） ──────────────────────────────
    st.markdown("#### ⚡ 最強フレーズ 3選")
    for i, phrase in enumerate(g["top_phrases"], 1):
        boost_pct = int(phrase["prob_boost"] * 100)
        st.markdown(
            f'<div class="gunshi-phrase-card">'
            f'<strong>#{i} [{phrase["category"]}]</strong> '
            f'<span style="color:#2d8a4e;font-size:.8rem;">承認確率 +{boost_pct}%</span><br>'
            f'{phrase["text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 承認確率バー ───────────────────────────────────────────────
    st.markdown(
        f'<div style="margin:.6rem 0 .2rem 0"><strong>⚔️ ベイズ推定 承認確率</strong>&nbsp;'
        f'<span style="font-size:1.5rem;font-weight:700;color:{bar_color}">{pct}%</span>'
        f'&nbsp;<span style="color:{bar_color}">{bar_label}</span></div>'
        f'<div class="gunshi-prob-bar-wrap">'
        f'<div class="gunshi-prob-bar-fill" style="width:{pct}%;background:{bar_color}">{pct}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    vehicle_label = f" / 車種ブースト: {g['vehicle_type']} +{int(g.get('vehicle_boost', 0)*100)}%" if g.get("vehicle_boost", 0) > 0 else ""
    st.caption(
        f"事前確率 {g['prior']*100:.1f}% → 証拠統合後 {posterior*100:.1f}% "
        f"→ フレーズ込み {pct}%　"
        f"（物件: {g['asset_name'] or '不明'} / リセール: {g['resale']} / "
        f"リピート: {g['repeat_cnt']}回{vehicle_label}）"
    )

    # ── カウンターオファー ────────────────────────────────────────
    if g["offers"]:
        st.markdown("#### 🔄 逆転の条件（カウンターオファー）")
        for o in g["offers"]:
            new_p = min(0.99, posterior + o["prob_gain"])
            st.markdown(
                f'<div class="gunshi-counter-card">'
                f'<strong>{o["title"]}</strong>'
                f'<span style="float:right;color:#f97316;font-weight:700;">'
                f'+{int(o["prob_gain"]*100)}% → {int(new_p*100)}%</span><br>'
                f'<span style="font-size:.85rem">{o["detail"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Step 2 演出ログ ────────────────────────────────────────────
    st.markdown("#### 🔍 深層スキャンログ")
    with st.status("データベースをスキャン中...", expanded=False) as status:
        st.write(f"📂 過去の成約データ: {g['patterns']['total']} 件確認")
        st.write(f"📊 {g['industry_cat']} 成約率: {g['patterns']['success_ratio']*100:.1f}%")
        st.write(f"🏷️ リセール評価: {g['resale']}  / リピート回数: {g['repeat_cnt']} 回")
        st.write(f"🔢 承認確率: {g['prior']*100:.1f}% → {posterior*100:.1f}%")
        status.update(label="スキャン完了", state="complete")

    # ── Step 3: LLM 推薦文 ────────────────────────────────────────
    st.markdown("#### 🧠 軍師の推薦文（LLM生成）")
    run_llm = st.button(
        "🤖 軍師に推薦文を生成させる（Gemini優先）",
        key="gunshi_run_llm_in_results",
        width='stretch',
    )
    if run_llm:
        _trend = st.session_state.get("_gunshi_trend_300", "")
        _comp  = (res or {}).get("comparison", "")
        prompt = build_gunshi_prompt(
            industry=g["industry_cat"],
            score=g["score"],
            pd_pct=g["pd_pct"],
            resale=g["resale"],
            repeat_cnt=g["repeat_cnt"],
            subsidy=g["subsidy"],
            bank=g["bank"],
            intuition=g["intuition"],
            posterior=posterior,
            success_patterns=g["patterns"],
            top_phrases=g["top_phrases"],
            asset_name=g.get("asset_name", ""),
            vehicle_type=g.get("vehicle_type", ""),
            trend_info=_trend,
            comparison_text=_comp,
            humor_style=st.session_state.get("humor_style", "standard"),
            asset_market_context=_get_asset_market_ctx(),
        )
        full_text = ""
        try:
            full_text = st.write_stream(_gemini_first_stream(prompt, model_name))
        except Exception as e:
            full_text = (
                f"【定型文フォールバック】\n\n"
                f"本案件（{g['industry_cat']}）は承認を強く推奨します。\n"
                + g["top_phrases"][0]["text"] + "\n\n"
                + g["top_phrases"][1]["text"] + "\n\n"
                f"⚠️ LLM接続エラー: {e}"
            )
            st.markdown(full_text)
        st.session_state["gunshi_llm_text"] = full_text

    # ── 結果登録 ───────────────────────────────────────────────────
    st.markdown("---")
    st.caption("✅ 結果を登録すると、次回以降の推薦精度が向上します（逐次ベイズ学習）")
    last_id = st.session_state.get("gunshi_ar_case_id")
    if not last_id:
        if st.button("📝 この案件を学習DBに保存", key="gunshi_ar_save", width='stretch'):
            new_id = save_case({
                "industry":   g["industry_cat"],
                "score":      g["score"],
                "pd_pct":     g["pd_pct"],
                "resale":     g["resale"],
                "repeat_cnt": g["repeat_cnt"],
                "subsidy":    g["subsidy"],
                "bank":       g["bank"],
                "intuition":  g["intuition"],
                "prior_prob": g["prior"],
                "posterior":  g["posterior"],
                "result":     "未登録",
            })
            st.session_state["gunshi_ar_case_id"] = new_id
            st.success(f"案件 ID {new_id} を保存しました。")
            st.rerun()
    else:
        notes_in = st.text_input(
            "備考（任意）",
            key="gunshi_ar_notes",
            placeholder="期間短縮で承認 / 財務指標基準未達 等",
        )
        col_w, col_l = st.columns(2)
        with col_w:
            if st.button("🎉 成約", type="primary", key="gunshi_ar_win", width='stretch'):
                update_result(last_id, "成約", notes_in)
                st.success("成約を登録しました。")
                st.session_state.pop("gunshi_ar_case_id", None)
                st.rerun()
        with col_l:
            if st.button("💔 非成約", key="gunshi_ar_lose", width='stretch'):
                update_result(last_id, "非成約", notes_in)
                st.warning("非成約を登録しました。")
                st.session_state.pop("gunshi_ar_case_id", None)
                st.rerun()

    return g


def build_gunshi_pdf_data(g: dict) -> dict:
    """
    軍師分析結果 dict を PDF 埋め込み用フラット dict に変換する。
    screening_report.py の build_screening_report_pdf に extra["gunshi"] として渡す。
    """
    return {
        "display_prob":   g["display_prob"],
        "posterior":      g["posterior"],
        "prior":          g["prior"],
        "industry_cat":   g["industry_cat"],
        "resale":         g["resale"],
        "repeat_cnt":     g["repeat_cnt"],
        "subsidy":        g["subsidy"],
        "bank":           g["bank"],
        "asset_name":     g.get("asset_name", ""),
        "vehicle_type":   g.get("vehicle_type", ""),
        "vehicle_boost":  g.get("vehicle_boost", 0.0),
        "top_phrases":    [{"text": p["text"], "category": p["category"],
                            "prob_boost": p["prob_boost"]} for p in g["top_phrases"]],
        "offers":         [{"title": o["title"], "detail": o["detail"],
                            "prob_gain": o["prob_gain"]} for o in g["offers"]],
        "success_ratio":  g["patterns"]["success_ratio"],
        "similar_wins":   g["patterns"]["wins"],
        "llm_text":       st.session_state.get("gunshi_llm_text", ""),
    }


# ==============================================================================
# AIコメント軽量表示（審査結果画面に直接埋め込む用）
# ==============================================================================

def render_gunshi_ai_comment(
    res: dict,
    submitted_inputs: dict | None = None,
    model_name: str = DEFAULT_MODEL,
    trend_info: str = "",   # ← 業界動向テキスト（analysis_results から渡す）
    bn_evidence: dict | None = None,
    bn_approval_prob: float | None = None,
) -> None:
    """
    審査結果画面の上部に直接表示するAIコメントセクション。
    軍師Exパンダを開かなくても承認確率・推薦フレーズ・LLM推薦文が見える。
    """
    st.markdown(_GUNSHI_CSS, unsafe_allow_html=True)

    # 業界動向テキストをセッションに保存（LLMボタン押下時に参照）
    if trend_info and trend_info.strip():
        st.session_state["_gunshi_trend_info"] = trend_info

    # ── 計算（render_gunshi_in_results と共有キャッシュ） ─────────────────
    cache_key    = "gunshi_auto_result"
    last_score   = st.session_state.get("_gunshi_cache_score")
    last_bn_hash = st.session_state.get("_gunshi_cache_bn_hash")
    cur_score    = res.get("score", 0)
    cur_bn_hash  = hash((frozenset((bn_evidence or {}).items()), bn_approval_prob))
    if cache_key not in st.session_state or last_score != cur_score or last_bn_hash != cur_bn_hash:
        with st.spinner("軍師AIを起動中..."):
            st.session_state[cache_key] = compute_gunshi_from_res(
                res, submitted_inputs,
                bn_evidence=bn_evidence,
                bn_approval_prob=bn_approval_prob,
            )
            st.session_state["_gunshi_cache_score"]   = cur_score
            st.session_state["_gunshi_cache_bn_hash"] = cur_bn_hash

    g = st.session_state[cache_key]
    pct       = int(g["display_prob"] * 100)
    posterior = g["posterior"]
    bar_color = "#2d8a4e" if pct >= 70 else ("#f97316" if pct >= 50 else "#ef4444")
    bar_label = "✅ 承認圏内" if pct >= 70 else ("⚠️ 要審議" if pct >= 50 else "❌ 再考必要")

    # ── セクションヘッダ ───────────────────────────────────────────────────
    st.markdown(
        '<div style="background:linear-gradient(90deg,#1e3a5f,#0d1b2a);'
        'border-radius:10px;padding:.6rem 1.2rem .4rem 1.2rem;margin-bottom:.5rem;">'
        '<span style="color:#fbbf24;font-size:1.1rem;font-weight:700;">⚔️ 軍師AIコメント</span>'
        '<span style="color:#94a3b8;font-size:.8rem;margin-left:.8rem;">'
        '逐次ベイズ学習 × 承認奪取AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 承認確率バー（コンパクト版） ──────────────────────────────────────
    st.markdown(
        f'<div style="margin:.3rem 0 .1rem 0">'
        f'<strong>⚔️ ベイズ推定 承認確率：</strong>'
        f'<span style="font-size:1.4rem;font-weight:700;color:{bar_color}">{pct}%</span>'
        f'&nbsp;<span style="color:{bar_color};font-size:.95rem">{bar_label}</span>'
        f'</div>'
        f'<div class="gunshi-prob-bar-wrap" style="margin-bottom:.3rem">'
        f'<div class="gunshi-prob-bar-fill" style="width:{pct}%;background:{bar_color}">{pct}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    vehicle_label = (
        f" / 車種ブースト: {g['vehicle_type']} +{int(g.get('vehicle_boost', 0)*100)}%"
        if g.get("vehicle_boost", 0) > 0 else ""
    )
    st.caption(
        f"事前確率 {g['prior']*100:.1f}% → 証拠統合後 {posterior*100:.1f}% → フレーズ込み {pct}%　"
        f"（物件: {g['asset_name'] or '不明'} / リセール: {g['resale']} / リピート: {g['repeat_cnt']}回{vehicle_label}）"
    )

    # ── 役員車警告バナー（役員車が検出された場合のみ表示） ─────────────────
    if g.get("exec_car"):
        st.markdown(
            '<div style="background:#7f1d1d;border-left:5px solid #fca5a5;'
            'border-radius:8px;padding:.8rem 1.2rem;margin:.4rem 0;color:#fef2f2;">'
            '<span style="font-size:1.05rem;font-weight:700;">🚨 役員車・高級輸入車リース — 要注意</span><br>'
            '<span style="font-size:.88rem;">'
            'お金を産まない投資です。审査部から使用目的を厳しく問われます。<br>'
            '稟議書に使用目的を明記し、期間短縮または前受金の積み増しを必ず提案してください。'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── トップ推薦フレーズ（上位2件） ─────────────────────────────────────
    st.markdown("**⚡ 最強推薦フレーズ（TOP 2）**")
    for i, phrase in enumerate(g["top_phrases"][:2], 1):
        boost_pct = int(phrase["prob_boost"] * 100)
        # 役員車警告フレーズは赤ベースで表示
        card_style = (
            'background:#7f1d1d;border-left:4px solid #fca5a5;color:#fef2f2;'
            if phrase.get("category") == "役員車警告" else
            ''
        )
        st.markdown(
            f'<div class="gunshi-phrase-card" style="margin:.25rem 0;{card_style}">'
            f'<strong>#{i} [{phrase["category"]}]</strong> '
            f'<span style="font-size:.8rem;">承認確率 +{boost_pct}%</span><br>'
            f'{phrase["text"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── 業界動向サマリー（300文字・常時表示） ──────────────────────────────
    _raw_trend = st.session_state.get("_gunshi_trend_info", "")
    if _raw_trend and _raw_trend.strip():
        # ネット補足は除外し、jsicデータ部分を先頭300文字に絞る
        _trend_core = _raw_trend.split("\n\n【ネットで補足】")[0].strip()
        _trend_300  = (_trend_core[:300] + "…") if len(_trend_core) > 300 else _trend_core
        # 300文字版をセッションに保存（LLMプロンプトで使用）
        st.session_state["_gunshi_trend_300"] = _trend_300
        st.markdown(
            f'<div style="background:#0c2340;border-left:4px solid #38bdf8;'
            f'border-radius:8px;padding:.6rem 1rem;margin:.4rem 0 .5rem 0;color:#bae6fd;font-size:.87rem;">'
            f'<span style="font-size:.8rem;font-weight:700;color:#38bdf8;">📊 業界動向サマリー</span><br>'
            f'<span style="line-height:1.6;">{_trend_300}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── LLM 推薦文（キャッシュ表示 ＋ 生成ボタン） ────────────────────────
    st.markdown("**🧠 軍師の推薦文（LLM生成）**")
    cached_text = st.session_state.get("gunshi_llm_text", "")

    if cached_text:
        # すでに生成済み → スタイル付きで表示
        st.markdown(
            f'<div style="background:#0f2744;border-left:4px solid #fbbf24;'
            f'border-radius:8px;padding:1rem 1.2rem;color:#e2e8f0;'
            f'font-size:.93rem;line-height:1.7;white-space:pre-wrap;margin:.3rem 0;">'
            f'{cached_text}'
            f'</div>',
            unsafe_allow_html=True,
        )
        # 再生成ボタン（小さく）
        if st.button(
            "🔄 推薦文を再生成",
            key="gunshi_ai_comment_regen",
            help="Ollamaが起動中の場合のみ動作します",
        ):
            st.session_state.pop("gunshi_llm_text", None)
            st.rerun()
    else:
        # 未生成 → 生成ボタン
        col_btn, col_info = st.columns([2, 3])
        with col_btn:
            run_llm = st.button(
                "🤖 軍師AIに推薦文を生成させる",
                key="gunshi_ai_comment_generate",
                width='stretch',
                type="primary",
            )
        with col_info:
            _has_key = bool(_get_gemini_key())
            st.caption("✅ Gemini で生成します" if _has_key else "⚠️ APIキー未設定。サイドバーで入力してください（Ollamaにフォールバック）")

        if run_llm:
            # 業界動向は300文字サマリー版を使用（UIに表示済みのものと同一）
            _trend = st.session_state.get("_gunshi_trend_300", "")
            _comp  = (res or {}).get("comparison", "")
            prompt = build_gunshi_prompt(
                industry=g["industry_cat"],
                score=g["score"],
                pd_pct=g["pd_pct"],
                resale=g["resale"],
                repeat_cnt=g["repeat_cnt"],
                subsidy=g["subsidy"],
                bank=g["bank"],
                intuition=g["intuition"],
                posterior=posterior,
                success_patterns=g["patterns"],
                top_phrases=g["top_phrases"],
                asset_name=g.get("asset_name", ""),
                vehicle_type=g.get("vehicle_type", ""),
                trend_info=_trend,      # ← 業界動向（実データ）
                comparison_text=_comp,  # ← 財務比較（実データ）
                humor_style=st.session_state.get("humor_style", "standard"),
                asset_market_context=_get_asset_market_ctx(),
            )
            full_text = ""
            try:
                full_text = st.write_stream(_gemini_first_stream(prompt, model_name))
            except Exception as e:
                full_text = (
                    f"【定型文フォールバック】\n\n"
                    f"本案件（{g['industry_cat']}）は承認を強く推奨します。\n"
                    + g["top_phrases"][0]["text"] + "\n\n"
                    + g["top_phrases"][1]["text"] + "\n\n"
                    f"⚠️ LLM接続エラー: {e}\n"
                    f"（Gemini APIキーを確認するか、Ollamaを起動してください）"
                )
                st.markdown(full_text)
            st.session_state["gunshi_llm_text"] = full_text
            st.rerun()
