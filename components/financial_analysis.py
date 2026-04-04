# -*- coding: utf-8 -*-
"""
components/financial_analysis.py
=================================
「📊 3期財務分析」ページ — TimesFM リース審査支援。

機能:
    1. 3期分の売上・営業利益・純資産を入力するフォーム
    2. YoY成長率・営業利益率・純資産比率の自動計算
    3. FastAPI バックエンド（backend.py）に予測リクエストを送信
    4. 実績（実線）＋予測（点線）の Plotly グラフを表示
    5. Ollama ローカルLLMによる審査コメント生成（3行）
    6. 将来予測フェーズのプレースホルダ関数（TimesFM 拡張用）
"""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── 定数 ─────────────────────────────────────────────────────────────────────
# バックエンド API の接続先
BACKEND_URL = os.environ.get("FINANCIAL_BACKEND_URL", "http://localhost:8000")

# 業種選択肢（backend.py の SEASONAL_INDICES と揃える）
INDUSTRY_OPTIONS = [
    "建設業",
    "小売業",
    "製造業",
    "卸売業",
    "医療・福祉",
    "飲食・宿泊業",
    "サービス業",
    "不動産業",
    "情報通信業",
    "運輸・物流",
]

# 期ラベル
PERIOD_LABELS = ["3期前", "2期前", "直近期"]


# ── 指標計算 ──────────────────────────────────────────────────────────────────

def calc_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    売上・営業利益・純資産から以下の財務指標を計算して返す。
        - YoY 売上成長率（%）: (当期 - 前期) / 前期 × 100
        - 営業利益率（%）: 営業利益 / 売上 × 100
        - 純資産比率（%）: 純資産 / 売上 × 100（総資産が不明なため売上対比）

    Args:
        df: 3行（3期分）のデータフレーム。列: ["売上高", "営業利益", "純資産"]

    Returns:
        指標列を追加した新しいデータフレーム
    """
    result = df.copy()

    # YoY 売上成長率（最初の期は計算不可のため NaN）
    yoy: list[Optional[float]] = [None]
    for i in range(1, len(df)):
        prev = df["売上高"].iloc[i - 1]
        curr = df["売上高"].iloc[i]
        if prev and prev != 0:
            yoy.append((curr - prev) / abs(prev) * 100)
        else:
            yoy.append(None)
    result["売上成長率(%)"] = yoy

    # 営業利益率
    result["営業利益率(%)"] = df.apply(
        lambda r: r["営業利益"] / r["売上高"] * 100 if r["売上高"] != 0 else None,
        axis=1,
    )

    # 純資産比率（売上高対比）
    result["純資産比率(%)"] = df.apply(
        lambda r: r["純資産"] / r["売上高"] * 100 if r["売上高"] != 0 else None,
        axis=1,
    )

    return result


# ── グラフ描画 ────────────────────────────────────────────────────────────────

def _build_forecast_chart(
    api_resp: dict,
    metric_key_hist: str,
    metric_key_fore: str,
    label: str,
    color: str,
) -> go.Figure:
    """
    実績（実線）＋ 予測（点線）の Plotly 折れ線グラフを作成する。

    Args:
        api_resp:         /forecast エンドポイントのレスポンス dict
        metric_key_hist:  過去データのキー（例: "sales_history"）
        metric_key_fore:  予測データのキー（例: "sales_forecast"）
        label:            凡例・タイトルに使う指標名
        color:            ラインの色（例: "royalblue"）

    Returns:
        Plotly Figure
    """
    months_hist = api_resp["months_history"]
    months_fore = api_resp["months_forecast"]
    hist_vals   = api_resp[metric_key_hist]
    fore_vals   = api_resp[metric_key_fore]

    fig = go.Figure()

    # 実績ライン（実線）
    fig.add_trace(go.Scatter(
        x=months_hist,
        y=hist_vals,
        mode="lines+markers",
        name=f"{label}（実績）",
        line=dict(color=color, width=2),
        marker=dict(size=4),
    ))

    # 予測ライン（点線）— 実績の末尾と接続するため最後の実績点を先頭に追加
    fig.add_trace(go.Scatter(
        x=[months_hist[-1]] + months_fore,
        y=[hist_vals[-1]] + fore_vals,
        mode="lines+markers",
        name=f"{label}（予測）",
        line=dict(color=color, width=2, dash="dot"),
        marker=dict(size=4, symbol="diamond"),
    ))

    # 実績と予測の境界を縦線で示す
    fig.add_vline(
        x=months_hist[-1],
        line_dash="dash",
        line_color="gray",
        annotation_text="予測開始",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"{label} 推移（実績 + 12ヶ月予測）",
        xaxis_title="月",
        yaxis_title="千円",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def _build_metrics_chart(df_metrics: pd.DataFrame) -> go.Figure:
    """
    営業利益率・純資産比率の3期推移グラフを作成する。

    Args:
        df_metrics: calc_metrics() の出力

    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    periods = df_metrics["期"].tolist()

    fig.add_trace(go.Scatter(
        x=periods, y=df_metrics["営業利益率(%)"].tolist(),
        mode="lines+markers+text",
        name="営業利益率(%)",
        line=dict(color="tomato", width=2),
        marker=dict(size=8),
        text=[f"{v:.1f}%" if v is not None else "" for v in df_metrics["営業利益率(%)"]],
        textposition="top center",
    ))

    fig.add_trace(go.Scatter(
        x=periods, y=df_metrics["純資産比率(%)"].tolist(),
        mode="lines+markers+text",
        name="純資産比率(%)",
        line=dict(color="mediumseagreen", width=2),
        marker=dict(size=8),
        text=[f"{v:.1f}%" if v is not None else "" for v in df_metrics["純資産比率(%)"]],
        textposition="bottom center",
    ))

    fig.update_layout(
        title="収益性・純資産比率の推移",
        xaxis_title="期",
        yaxis_title="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ── Gemini 審査コメント ────────────────────────────────────────────────────────

def _get_gemini_api_key() -> str:
    """
    Gemini APIキーを優先順位に従って取得する。
    ai_chat.py と同じ取得ロジック。

    優先順位:
        1. 環境変数 GEMINI_API_KEY
        2. .streamlit/secrets.toml の GEMINI_API_KEY
        3. st.session_state["fin_gemini_api_key"]（手動入力済み）

    Returns:
        APIキー文字列。未設定の場合は空文字列。
    """
    # 1. 環境変数
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    # 2. secrets.toml
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # 3. セッション（手動入力済み）
    return st.session_state.get("fin_gemini_api_key", "")


def _call_gemini(prompt: str, api_key: str, timeout: int = 90) -> str:
    """
    Gemini API にプロンプトを送信して審査コメントを取得する。
    ai_chat.py の _gemini_chat と同じパターンを使用。

    Args:
        prompt:  送信するプロンプト文字列
        api_key: Gemini APIキー
        timeout: タイムアウト秒数

    Returns:
        LLM の回答テキスト。エラー時はエラーメッセージ文字列。
    """
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        return "[Gemini] google-generativeai がインストールされていません。`pip install google-generativeai` を実行してください。"

    if not api_key:
        return "[Gemini] APIキーが設定されていません。"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        )
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1024},
            request_options={"timeout": timeout},
        )
        return response.text
    except Exception as e:
        return f"[Gemini] エラー: {e}"


def _build_gemini_prompt(
    df_input: pd.DataFrame,
    api_resp: dict,
    industry: str,
) -> str:
    """
    Ollama に渡すプロンプトを組み立てる。
    TimesFM の12ヶ月後予測値を含めた審査コメントを依頼する。

    Args:
        df_input: 入力3期データ
        api_resp: /forecast レスポンス dict
        industry: 業種名

    Returns:
        プロンプト文字列
    """
    # 直近期の実績値
    latest_sales      = df_input["売上高"].iloc[-1]
    latest_profit     = df_input["営業利益"].iloc[-1]
    latest_net_assets = df_input["純資産"].iloc[-1]

    # 12ヶ月後の予測着地値
    sales_fore_12m      = api_resp["sales_forecast"][-1]
    profit_fore_12m     = api_resp["profit_forecast"][-1]
    net_assets_fore_12m = api_resp["net_assets_forecast"][-1]

    # 売上成長率（直近実績→12ヶ月予測）
    sales_growth = (
        (sales_fore_12m - latest_sales) / abs(latest_sales) * 100
        if latest_sales else 0
    )
    profit_margin_fore = (
        profit_fore_12m / sales_fore_12m * 100
        if sales_fore_12m else 0
    )

    prompt = f"""あなたはリース審査の専門家です。以下の企業財務データと将来予測を分析し、
審査担当者向けに日本語で正確に3行のコメントを出力してください。

【業種】{industry}

【3期財務実績（千円）】
{df_input[['期','売上高','営業利益','純資産']].to_string(index=False)}

【TimesFMによる12ヶ月後の予測着地見込み（千円）】
- 売上高:  {latest_sales:,.0f} → {sales_fore_12m:,.0f}（増減率: {sales_growth:+.1f}%）
- 営業利益: {latest_profit:,.0f} → {profit_fore_12m:,.0f}
- 純資産:  {latest_net_assets:,.0f} → {net_assets_fore_12m:,.0f}
- 予測営業利益率: {profit_margin_fore:.1f}%

【出力形式（必ず以下の3行で回答すること）】
■成長性：売上の成長性と市場ポジションの推察
■収益性：収益性の変化（利益率の改善・悪化）とその要因
■与信：純資産と予測値から見た支払い余力と総合的な与信判断の示唆
"""
    return prompt


# ── 将来予測フェーズ プレースホルダ ──────────────────────────────────────────

def forecast_placeholder(df: pd.DataFrame) -> None:
    """
    将来予測フェーズのプレースホルダ（TimesFM 等の高度なモデル組み込み用）。

    TODO:
        - Google TimesFM / Prophet / N-BEATS 等を直接組み込む場合はここに実装
        - df から時系列データを構築し、次期以降の予測値を返す形に変更する
        - 現在は backend.py の /forecast エンドポイントが同等の処理を担当している

    Args:
        df: 3期分の財務データ（pandas DataFrame）
    """
    pass  # 将来実装予定


# ── メインレンダリング関数 ────────────────────────────────────────────────────

def render_financial_analysis() -> None:
    """
    「📊 3期財務分析」ページ全体を描画する。
    lease_logic_sumaho12.py からルーティングされて呼ばれる。
    """
    import httpx  # type: ignore

    st.title("📊 3期財務分析 — TimesFM リース審査支援")
    st.caption(
        "3期分の財務データを入力し、業種別季節性を加味した月次補完と "
        "TimesFM による12ヶ月予測・Gemini 審査コメントを生成します。"
    )

    # ── ① データ入力フォーム ───────────────────────────────────────────────
    st.subheader("① 財務データ入力")
    st.info(
        "数値はすべて **千円単位** で入力してください。"
        "　営業利益が赤字の場合はマイナス値を入力してください。"
    )

    # 表ヘッダー行
    col_label, col_y3, col_y2, col_y1 = st.columns([2, 2, 2, 2])
    col_label.markdown("**指標**")
    col_y3.markdown(f"**{PERIOD_LABELS[0]}**")
    col_y2.markdown(f"**{PERIOD_LABELS[1]}**")
    col_y1.markdown(f"**{PERIOD_LABELS[2]}**")

    # 売上高入力
    c0, c1, c2, c3 = st.columns([2, 2, 2, 2])
    c0.markdown("売上高（千円）")
    sales_y3 = c1.number_input("売上_3期前", value=500_000, step=1_000, label_visibility="collapsed", key="sales_y3")
    sales_y2 = c2.number_input("売上_2期前", value=520_000, step=1_000, label_visibility="collapsed", key="sales_y2")
    sales_y1 = c3.number_input("売上_直近",  value=550_000, step=1_000, label_visibility="collapsed", key="sales_y1")

    # 営業利益入力
    d0, d1, d2, d3 = st.columns([2, 2, 2, 2])
    d0.markdown("営業利益（千円）")
    profit_y3 = d1.number_input("利益_3期前", value=30_000, step=1_000, label_visibility="collapsed", key="profit_y3")
    profit_y2 = d2.number_input("利益_2期前", value=35_000, step=1_000, label_visibility="collapsed", key="profit_y2")
    profit_y1 = d3.number_input("利益_直近",  value=38_000, step=1_000, label_visibility="collapsed", key="profit_y1")

    # 純資産入力
    e0, e1, e2, e3 = st.columns([2, 2, 2, 2])
    e0.markdown("純資産（千円）")
    net_y3 = e1.number_input("純資産_3期前", value=120_000, step=1_000, label_visibility="collapsed", key="net_y3")
    net_y2 = e2.number_input("純資産_2期前", value=145_000, step=1_000, label_visibility="collapsed", key="net_y2")
    net_y1 = e3.number_input("純資産_直近",  value=170_000, step=1_000, label_visibility="collapsed", key="net_y1")

    # 業種選択
    industry = st.selectbox("業種", INDUSTRY_OPTIONS, index=0, key="fin_industry")

    # ── ② 指標自動計算テーブル ─────────────────────────────────────────────
    st.divider()
    st.subheader("② 財務指標（自動計算）")

    # データフレーム構築
    df_input = pd.DataFrame({
        "期":    PERIOD_LABELS,
        "売上高": [sales_y3, sales_y2, sales_y1],
        "営業利益": [profit_y3, profit_y2, profit_y1],
        "純資産": [net_y3, net_y2, net_y1],
    })

    df_metrics = calc_metrics(df_input)

    # 表示用フォーマット
    def fmt_num(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:,.0f}"

    def fmt_pct(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:+.1f}%"

    display_df = pd.DataFrame({
        "期":           df_metrics["期"],
        "売上高(千円)":   df_metrics["売上高"].map(fmt_num),
        "営業利益(千円)": df_metrics["営業利益"].map(fmt_num),
        "純資産(千円)":   df_metrics["純資産"].map(fmt_num),
        "売上成長率":     df_metrics["売上成長率(%)"].map(fmt_pct),
        "営業利益率":     df_metrics["営業利益率(%)"].map(fmt_pct),
        "純資産比率":     df_metrics["純資産比率(%)"].map(fmt_pct),
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # 利益率推移グラフ（入力直後に即表示）
    with st.expander("📈 利益率・純資産比率の推移グラフ", expanded=True):
        st.plotly_chart(_build_metrics_chart(df_metrics), use_container_width=True)

    # ── ③ TimesFM 予測グラフ ───────────────────────────────────────────────
    st.divider()
    st.subheader("③ TimesFM 12ヶ月予測グラフ")
    st.caption(
        "「予測を実行」をクリックすると、FastAPI バックエンド (`backend.py`) に "
        "リクエストを送信し、業種別季節性補完 + TimesFM による12ヶ月予測を表示します。"
    )

    run_btn = st.button("🔮 予測を実行", type="primary", key="run_forecast_btn")

    if run_btn:
        payload = {
            "sales":      [float(sales_y3), float(sales_y2), float(sales_y1)],
            "profit":     [float(profit_y3), float(profit_y2), float(profit_y1)],
            "net_assets": [float(net_y3), float(net_y2), float(net_y1)],
            "industry":   industry,
        }
        try:
            with st.spinner("TimesFM で予測中...（初回はモデル読み込みに時間がかかります）"):
                resp = httpx.post(
                    f"{BACKEND_URL}/forecast",
                    json=payload,
                    timeout=120.0,
                )
            if resp.status_code != 200:
                st.error(f"バックエンドエラー: {resp.status_code} — {resp.text}")
            else:
                api_resp = resp.json()
                st.session_state["fin_api_resp"] = api_resp
                method = "TimesFM" if api_resp.get("timesfm_available") else "GBM（フォールバック）"
                st.success(f"予測完了 ✅ 使用モデル: {method}")

        except httpx.ConnectError:
            st.error(
                f"バックエンドに接続できません。以下を確認してください。\n\n"
                f"```\nuvicorn backend:app --reload --port 8000\n```"
            )
        except Exception as e:
            st.error(f"予測中にエラーが発生しました: {e}")

    # セッションに保存済みのレスポンスがあれば常に表示
    if "fin_api_resp" in st.session_state:
        api_resp = st.session_state["fin_api_resp"]

        # 3指標のグラフを縦に並べる
        chart_configs = [
            ("sales_history",      "sales_forecast",      "売上高",   "royalblue"),
            ("profit_history",     "profit_forecast",     "営業利益", "tomato"),
            ("net_assets_history", "net_assets_forecast", "純資産",   "mediumseagreen"),
        ]
        for hist_key, fore_key, label, color in chart_configs:
            fig = _build_forecast_chart(api_resp, hist_key, fore_key, label, color)
            st.plotly_chart(fig, use_container_width=True)

        # ── ④ Gemini 審査コメント ──────────────────────────────────────────
        st.divider()
        st.subheader("④ Gemini 審査コメント（AI自動生成）")
        st.caption(
            "Gemini APIが予測値を踏まえた3行の審査コメントを生成します。"
            f"　使用モデル: `{os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')}`"
        )

        # APIキー取得（環境変数 / secrets.toml → なければ手動入力）
        api_key = _get_gemini_api_key()
        if not api_key:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                key="fin_gemini_api_key_input",
                help="環境変数 GEMINI_API_KEY または .streamlit/secrets.toml でも設定できます",
            )
            if api_key_input:
                st.session_state["fin_gemini_api_key"] = api_key_input
                api_key = api_key_input
        else:
            st.caption("✅ Gemini APIキーを検出済み")

        if st.button("🤖 審査コメントを生成", key="gen_comment_btn"):
            if not api_key:
                st.warning("Gemini APIキーを入力してください。")
            else:
                prompt = _build_gemini_prompt(df_input, api_resp, industry)
                with st.spinner("Gemini で審査コメント生成中..."):
                    comment = _call_gemini(prompt, api_key)
                st.session_state["fin_gemini_comment"] = comment

        if "fin_gemini_comment" in st.session_state:
            comment = st.session_state["fin_gemini_comment"]
            if comment.startswith("[Gemini]"):
                # エラーメッセージ
                st.warning(comment)
            else:
                st.info(comment)

    # 将来予測フェーズのプレースホルダ（実装はここに追加）
    forecast_placeholder(df_input)
