"""
components/quantum_risk_view.py
────────────────────────────────────────────────────────────
⚛️ Q_risk 分析 (β) — 実験的機能

量子財務干渉モデル・物理動態分析・ベンフォード則による
財務構造の矛盾検知UIコンポーネント。

【重要】このモジュールは参考表示のみ。
実際のスコアリング・DB保存・係数更新には一切影響しない。
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np


def render_quantum_risk_view() -> None:
    """Q_risk分析βページのメイン描画関数。"""

    st.title("⚛️ Q_risk 分析")
    st.caption("🔬 実験的機能 (β) — 参考表示のみ。実際のスコアリングには影響しません。")

    st.markdown("""
    ```
    [ Quantum Finance Core v4.0 — GHZ Entanglement Protocol ]
    
    q[0] (売上)   : ──H──●───────────────────── 🎛️ 状態ベクトル: |ψ_sales⟩
                        │
    q[1] (純資産) : ──────X──●───────────────── 🎛️ 状態ベクトル: |ψ_equity⟩
                            │
    q[2] (利益)   : ──────────X──●───────────── 🎛️ 状態ベクトル: |ψ_profit⟩
                                │
    q[3] (負債)   : ──────────────X──[観測]─── 🎛️ 状態ベクトル: |ψ_debt⟩
    
    ※ 4つの財務指標がGHZ状態（|0000⟩ + |1111⟩）で量子もつれを起こしています。
    ```
    """, unsafe_allow_html=True)

    # β免責事項バナー
    st.warning(
        "⚠️ **β版 免責事項**\n\n"
        "このページは研究目的の実験的機能です。\n"
        "- **実際の審査スコアには影響しません**\n"
        "- 表示される Q_risk は参考値です\n"
        "- 係数再学習・DB保存は発生しません",
        icon="🧪"
    )

    st.divider()

    # ── アナライザーのロード ──────────────────────────────────────────────────
    try:
        from quantum_finance_analyzer import QuantumFinanceAnalyzer
        from quantum_finance_engine import QuantumFinanceEngine
        analyzer = QuantumFinanceAnalyzer()
        q_engine = QuantumFinanceEngine()
    except ImportError as e:
        st.error(f"分析モジュールが見つかりません: {e}")
        return

    # ── モード選択 ────────────────────────────────────────────────────────────
    mode = st.radio(
        "分析対象",
        ["現在の審査案件", "過去案件DB（一括）"],
        horizontal=True,
    )

    st.divider()
    
    # β機能内でのアプローチ切り替え（タブ）
    tab_classic, tab_wave = st.tabs([
        "📊 財務構造・物理動態分析 (既存)", 
        "🔮 業界基準・複素波動分析 (新規β)"
    ])

    with tab_classic:
        if mode == "現在の審査案件":
            _render_single_case(analyzer)
        else:
            _render_batch_analysis(analyzer)

    with tab_wave:
        if mode == "現在の審査案件":
            _render_single_wave_case(q_engine)
        else:
            st.info("💡 複素波動分析の一括処理は現在ロードマップ上です。単件分析をお試しください。")



# ─────────────────────────────────────────────────────────────────────────────
# 現在案件の単件分析
# ─────────────────────────────────────────────────────────────────────────────

def _render_single_case(analyzer) -> None:
    """session_state の現在審査案件をQ_riskで分析して表示。"""

    # last_result から財務インプットを復元（最優先は保存された financials、次に直接のキー）
    last_result = st.session_state.get("last_result") or {}
    q_inputs    = last_result.get("quantum_inputs") or {}
    financials  = last_result.get("financials") or {}

    inputs = None
    if financials or q_inputs or st.session_state.get("nenshu"):
        inputs = {
            "nenshu":       _sf(financials.get("nenshu")       or st.session_state.get("nenshu")),
            "gross_profit": _sf(financials.get("gross_profit") or st.session_state.get("item9_gross")),
            "op_profit":    _sf(financials.get("rieki")        or financials.get("op_profit") or q_inputs.get("op_profit") or st.session_state.get("rieki")),
            "ord_profit":   _sf(financials.get("ord_profit")   or q_inputs.get("ord_profit")  or st.session_state.get("item4_ord_profit")),
            "net_income":   _sf(financials.get("net_income")   or q_inputs.get("net_income")  or st.session_state.get("item5_net_income")),
            "net_assets":   _sf(financials.get("net_assets")   or st.session_state.get("net_assets")),
            "total_assets": _sf(financials.get("assets")       or financials.get("total_assets") or st.session_state.get("total_assets")),
            "machines":     _sf(financials.get("machines")     or q_inputs.get("machines")    or st.session_state.get("item6_machine")),
            "other_assets": _sf(financials.get("other_assets") or st.session_state.get("item7_other")),
            "depreciation": _sf(financials.get("depreciation") or q_inputs.get("depreciation")or st.session_state.get("item10_dep")),
            "bank_credit":  _sf(financials.get("bank_credit")  or st.session_state.get("bank_credit")),
            "lease_credit": _sf(financials.get("lease_credit") or st.session_state.get("lease_credit")),
        }
        # 総資産が0の場合、割り算エラーを防ぐため最低限1とする
        if inputs["total_assets"] <= 0:
            inputs["total_assets"] = 1.0
            
        # 全てゼロなら未入力と判断
        if all(v == 0.0 for v in inputs.values() if v != 1.0):
            inputs = None

    if not inputs:
        st.info("👈 先に個別審査を実行してください。審査後にここで追加分析できます。")

        with st.expander("📋 デモデータで試す"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("デモ実行（健全企業）", key="demo_healthy"):
                    st.session_state["_qrisk_demo"] = {
                        "nenshu": 500000, "gross_profit": 120000, "op_profit": 25000,
                        "ord_profit": 22000, "net_income": 15000, "net_assets": 80000,
                        "total_assets": 250000, "machines": 50000, "other_assets": 30000,
                        "depreciation": 5000, "bank_credit": 100000, "lease_credit": 30000,
                    }
                    st.rerun()
            with col2:
                if st.button("デモ実行（財務崩壊企業）", key="demo_crisis"):
                    st.session_state["_qrisk_demo"] = {
                        "nenshu": 50000, "gross_profit": 5000, "op_profit": -3000,
                        "ord_profit": -5000, "net_income": -8000, "net_assets": -20000,
                        "total_assets": 300000, "machines": 250000, "other_assets": 10000,
                        "depreciation": 30000, "bank_credit": 280000, "lease_credit": 50000,
                    }
                    st.rerun()

        inputs = st.session_state.get("_qrisk_demo")
        if not inputs:
            return

    df = pd.DataFrame([inputs])
    _display_q_risk_results(analyzer, df, show_company_name=False)


def _render_single_wave_case(q_engine) -> None:
    """複素波動型 Q_risk 分析（新規）の画面描画"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    last_result = st.session_state.get("last_result") or {}
    q_inputs    = last_result.get("quantum_inputs") or {}
    financials  = last_result.get("financials") or {}

    inputs = None
    if financials or q_inputs or st.session_state.get("nenshu"):
        inputs = {
            "nenshu":       _sf(financials.get("nenshu")       or st.session_state.get("nenshu")),
            "gross_profit": _sf(financials.get("gross_profit") or st.session_state.get("item9_gross")),
            "op_profit":    _sf(financials.get("rieki")        or financials.get("op_profit") or q_inputs.get("op_profit") or st.session_state.get("rieki")),
            "ord_profit":   _sf(financials.get("ord_profit")   or q_inputs.get("ord_profit")  or st.session_state.get("item4_ord_profit")),
            "net_income":   _sf(financials.get("net_income")   or q_inputs.get("net_income")  or st.session_state.get("item5_net_income")),
            "net_assets":   _sf(financials.get("net_assets")   or st.session_state.get("net_assets")),
            "total_assets": _sf(financials.get("assets")       or financials.get("total_assets") or st.session_state.get("total_assets")),
            "machines":     _sf(financials.get("machines")     or q_inputs.get("machines")    or st.session_state.get("item6_machine")),
            "other_assets": _sf(financials.get("other_assets") or st.session_state.get("item7_other")),
            "depreciation": _sf(financials.get("depreciation") or q_inputs.get("depreciation")or st.session_state.get("item10_dep")),
            "bank_credit":  _sf(financials.get("bank_credit")  or st.session_state.get("bank_credit")),
            "lease_credit": _sf(financials.get("lease_credit") or st.session_state.get("lease_credit")),
        }
        if inputs["total_assets"] <= 0:
            inputs["total_assets"] = 1.0
        if all(v == 0.0 for v in inputs.values() if v != 1.0):
            inputs = None

    if not inputs:
        st.info("👈 個別審査実行後に、ここで波の干渉分析が行えます。")
        inputs = st.session_state.get("_qrisk_demo")
        if not inputs:
            return

    # 初めてのユーザー向け解説エキスパンダー
    with st.expander("💡 【初めての方へ】複素波動分析（Q_risk）とは？", expanded=True):
        st.markdown("""
        この機能は、企業の財務データを **「波（波動ベクトル）」** として捉え、財務構造のバランスの崩れを視覚化する実験的な分析手法です。

        1. **業種標準を「基準（0度）」とする**:
           業種ごとに健全とされる「売上高/総資産」や「自己資本比率」のバランスを定義します。
        2. **実績を矢印（ベクトル）化する**:
           申請された数字が業界基準より大きければプラス方向、小さければマイナス方向に矢印が回転します。
        3. **「波の打ち消し合い」で矛盾を暴く**:
           - **正常**: 売上と純資産が基準通りであれば、矢印が同じ方向を向き、波が強め合います。
           - **矛盾（異常）**: 「売上だけ極端に高いが財務基盤が脆い」などバランスが崩れると、矢印が真逆を向き、互いに**打ち消し合って（干渉）**しまいます。
        
        この打ち消し合いによって失われたエネルギーの割合を **「Q_risk」** として算出し、企業の「数字の不自然さ」を見抜きます。
        """)

    # 業種選択
    selected_major = st.session_state.get("major_industry", "DEFAULT")
    
    # 分析実行
    row = pd.Series(inputs)
    analysis = q_engine.analyze_interference(row, selected_major)

    
    # ── 本物の量子コンピュータ (IBM Quantum) 連携セクション ─────────────────────
    st.divider()
    st.markdown("### 🚀 IBM Quantum 実機 / シミュレータ連携")
    st.caption("オープンソースの量子プログラミングフレームワーク **Qiskit** を使用し、本物の量子回路をバックエンドで自動構成します。")
    
    col_tk1, col_tk2 = st.columns([3, 2])
    with col_tk1:
        api_token = st.text_input(
            "IBM Quantum APIトークン (任意)", 
            type="password", 
            help="IBM Quantumのアカウントページから取得できるAPIトークンを入力してください。"
        )
    with col_tk2:
        check_job_id = st.text_input(
            "Job ID を入力して結果確認", 
            placeholder="例: crp0xxxxxxxxxxxx",
            help="実機へ送信したジョブのIDを入力して結果を取得します。"
        )
        if st.button("🔍 結果を取得", key="btn_fetch_job"):
            if not api_token or not check_job_id:
                st.warning("APIトークンとJob IDを入力してください。")
            else:
                with st.spinner("IBM Quantum より結果をポーリング中..."):
                    fetch_res = q_engine.retrieve_ibm_quantum_result(check_job_id, api_token)
                if "error" in fetch_res:
                    st.error(fetch_res["error"])
                else:
                    if fetch_res.get("status") == "DONE":
                        st.success(fetch_res["msg"])
                        
                        # 理論値（シミュレータ）も同時に再計算
                        sim_res = q_engine.run_on_ibm_quantum(row, selected_major, api_token=None)
                        
                        col_comp1, col_comp2 = st.columns(2)
                        with col_comp1:
                            st.markdown("**🚀 本物の実機 (IBM Quantum)**")
                            st.metric("実機 Q_risk", f"{fetch_res['qrisk_quantum']}%")
                            st.json(fetch_res["probabilities"])
                        with col_comp2:
                            st.markdown("**🤖 シミュレータ (理論値)**")
                            st.metric("理論値 Q_risk", f"{sim_res['qrisk_quantum']}%")
                            st.json(sim_res["probabilities"])
                    else:
                        st.info(fetch_res["msg"])
                        
    if st.button("⚛️ Qiskit 量子回路を生成・実行"):
        with st.spinner("量子状態をエンコード中..."):
            q_res = q_engine.run_on_ibm_quantum(row, selected_major, api_token)
        
        if "error" in q_res:
            st.error(q_res["error"])
        else:
            st.success(q_res["msg"])
            
            col_q1, col_q2 = st.columns([3, 2])
            with col_q1:
                st.markdown("**🤖 生成された量子回路 (4-Qubits)**")
                st.caption("Q0:売上, Q1:純資産, Q2:利益, Q3:負債。Ryで振幅、Pで位相を埋め込み、HとCXで相互干渉を生成。")
                st.code(q_res["circuit_str"])
            with col_q2:
                st.markdown("**📊 観測された確率分布**")
                st.caption("状態 '1111' (全指標が最悪の矛盾を起こした状態) の観測確率。")
                st.json(q_res["probabilities"])
                st.metric("📈 Qiskit推計 4次元矛盾リスク", f"{q_res['qrisk_quantum']}%")
                
            st.divider()
            with st.expander("🔍 【解説】量子回路と確率分布の読み解き方", expanded=True):
                st.markdown("""
                #### 1. 量子回路（Circuit）の構成要素
                - **`q_0`〜`q_3`**: 4つの量子ビット（Qubit）。それぞれ「売上」「純資産」「営業利益」「総負債」の波動を格納しています。
                - **`Ry` (回転ゲート)**: 財務数値の大きさに応じて量子を傾け、情報を記憶させています。
                - **`P` (位相ゲート)**: 業界平均からのズレ（角度）を与えています。
                - **`H` / `cx`**: 量子を重ね合わせ、連鎖的にもつれ（CNOT）させることで干渉効果を発生させます。
                
                #### 2. 確率分布（0000〜1111）の意味
                量子力学の世界では、1024回測定した際に `0000` から `1111` までの16パターンのうち、どれが出現したかの割合を見ます。
                特に **`'1111'`** の状態は、全指標の波が逆向きに矛盾したときに最も発生しやすくなるため、これを4次元版の危険度としています。
                """)
                
    st.divider()


    st.markdown("### 🔮 業界基準型・複素波動分析")
    st.caption("「売上」「純資産」「利益」「負債」が、業界基準に対してどのように打ち消し合っているか（論理的矛盾）を検知します。")
    st.info("""
    **📌 波動干渉 Q_risk の判定基準:**
    - **🟢 0% 〜 35%【健全】**: 財務バランスが非常に安定しており、理想的です。
    - **🟡 35% 〜 50%【要警戒】**: バランスの歪みが生じ始めており、無理のある経営のサインです。
    - **🔴 50% 以上【問題あり】**: 致命的な論理的矛盾を検知。事業実態の再精査を強く推奨します。
    """)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("⚛️ 波動干渉 Q_risk", f"{analysis['q_risk']}%")
    c2.metric("⚡ 理論最大強度 (I_max)", f"{analysis['i_max']}")
    c3.metric("🌊 実測強度 (I_actual)", f"{analysis['i_actual']}")
    
    # ペナルティシミュレーション表示
    st.divider()
    st.markdown("**💡 スコア補正シミュレーション（参考）**")
    p_factor = analysis['penalty_factor']
    if p_factor < 1.0:
        st.warning(f"⚠️ 財務位相のズレ（Q_risk > 35%）を検知したため、成約率に対し **{p_factor:.2f} 倍** の減衰ペナルティがシミュレートされます。")
    else:
        st.success("🟢 財務干渉は安定圏内です（ペナルティなし）。")
        
    # アルガン図プロットと解説
    st.divider()
    col_arg1, col_arg2 = st.columns([1, 1])
    with col_arg1:
        st.markdown("**📈 複素平面（アルガン図）**")
        fig = q_engine.visualize_argand_diagram(row, selected_major)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    with col_arg2:
        st.markdown("#### 🔍 4次元ベクトル図の読み解き方")
        st.caption("各財務指標を「矢印（位相ベクトル）」に変換し、繋ぎ合わせています。")
        st.markdown("""
        - **シアンの矢印（売上）**
        - **紫の矢印（純資産）**
        - **緑の矢印（営業利益）**
        - **黄色の矢印（負債）**：※負債は引く向きで合成されます
        - **ピンクの矢印（最終合成）**：4つの矢印を全て掛け合わせた「企業の総合強度」です。
        - **外側の点線円**：4指標が完全に協調している場合の最大理論値。ピンクの先端が円から遠ざかっているほど、財務パワーが失われています。
        """)

    # 波動干渉シミュレーションと解説
    st.divider()
    col_wave1, col_wave2 = st.columns([1, 1])
    with col_wave1:
        st.markdown("**🌊 財務波動の干渉（タイムドメイン表示）**")
        fig_wave = q_engine.visualize_wave_interference(row, selected_major)
        st.pyplot(fig_wave, use_container_width=True)
        plt.close()
    with col_wave2:
        st.markdown("#### 🔍 波動グラフの読み解き方")
        st.caption("ベクトルを時間軸上の「波（サイン波）」として展開したものです。")
        st.markdown("""
        - **各波のズレ（シアン・紫）**: 波の「山と谷」のタイミングが揃っているほど正常です。
        - **ピンクの点線（合成波）**: タイミングが大きくズレる（逆位相になる）と、お互いの数値を消し去るように平坦な波へと近づいていきます。
        """)






# ─────────────────────────────────────────────────────────────────────────────
# 過去案件DB一括分析
# ─────────────────────────────────────────────────────────────────────────────

def _render_batch_analysis(analyzer) -> None:
    """DB上の全過去案件をQ_riskで一括分析して可視化。"""
    try:
        from data_cases import load_all_cases
    except ImportError:
        st.error("data_cases.py が見つかりません")
        return

    with st.spinner("過去案件をDBから読み込み中..."):
        all_cases = load_all_cases()

    if not all_cases:
        st.info("過去案件データがありません。先にバッチ登録または個別審査を実施してください。")
        return

    # 財務データをDFに変換
    rows = []
    labels = []
    for c in all_cases:
        inp = c.get("inputs") or {}
        fin = c.get("financials") or {}
        row = {
            "nenshu":       _sf(inp.get("nenshu") or fin.get("nenshu")),
            "gross_profit": _sf(inp.get("gross_profit") or fin.get("gross_profit")),
            "op_profit":    _sf(inp.get("op_profit")    or fin.get("op_profit")),
            "ord_profit":   _sf(inp.get("ord_profit")   or fin.get("ord_profit")),
            "net_income":   _sf(inp.get("net_income")   or fin.get("net_income")),
            "net_assets":   _sf(inp.get("net_assets")   or fin.get("net_assets")),
            "total_assets": _sf(inp.get("total_assets") or fin.get("assets")),
            "machines":     _sf(inp.get("machines")     or fin.get("machines")),
            "other_assets": _sf(inp.get("other_assets") or fin.get("other_assets")),
            "depreciation": _sf(inp.get("depreciation") or fin.get("depreciation")),
            "bank_credit":  _sf(inp.get("bank_credit")  or fin.get("bank_credit")),
            "lease_credit": _sf(inp.get("lease_credit") or fin.get("lease_credit")),
        }
        rows.append(row)
        company = c.get("company_name") or c.get("borrower_name") or "不明"
        final   = c.get("final_status") or "未登録"
        labels.append({"企業名": company, "最終結果": final, "case_id": c.get("id", "")})

    if not rows:
        st.warning("財務データが空です。")
        return

    df = pd.DataFrame(rows)
    df_labels = pd.DataFrame(labels)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("分析対象件数", len(df))
    with col2:
        top_n = st.slider("表示件数（Q_risk上位）", 5, min(50, len(df)), min(20, len(df)))

    _display_q_risk_results(analyzer, df, df_labels=df_labels, top_n=top_n, show_company_name=True)


# ─────────────────────────────────────────────────────────────────────────────
# 共通: 結果表示
# ─────────────────────────────────────────────────────────────────────────────

def _display_q_risk_results(
    analyzer,
    df: pd.DataFrame,
    df_labels: pd.DataFrame | None = None,
    top_n: int = 20,
    show_company_name: bool = False,
) -> None:
    """Q_risk分析結果をStreamlitで表示する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    with st.spinner("⚛️ Q_risk 計算中..."):
        result = analyzer.full_analysis(df)

    # ── 単件表示 ──────────────────────────────────────────────────────────────
    if len(result) == 1:
        row = result.iloc[0].copy()
        risk_level = row["quantum_risk_level"]

        # ── last_result の正確な計算済み比率で上書き ──────────────────────────
        last_res = st.session_state.get("last_result") or {}
        if last_res.get("user_asset_turnover") is not None:
            row["asset_turnover"] = float(last_res["user_asset_turnover"])
        if last_res.get("user_roa") is not None:
            row["roa"] = float(last_res["user_roa"]) / 100  # % → 小数に変換

        # 総資産の未入力や単位異常による極端な異常値の丸め・補正
        if row["asset_turnover"] > 50.0 or abs(row["roa"]) > 5.0:
            st.error(
                "⚠️ **財務データ（総資産）の入力漏れ、または単位の不整合を検知しました。**\n\n"
                "総資産が 0 や未入力の状態で審査を実行すると、比率計算が天文学的な数字になります。\n"
                "個別審査フォームで **「総資産」** を正しく入力した上で、再度「審査実行」をしてください。",
                icon="🚨"
            )
            # 安全のため表示用の値を 0 に丸める
            row["asset_turnover"] = 0.0
            row["roa"] = 0.0
            stall_flag = False
        else:
            stall_flag = (row["asset_turnover"] < 0.3) and ((row["roa"] * 100) < 0)

        # ── 概要説明 ──────────────────────────────────────────────────────────
        st.markdown("""
> **このページについて**
> 財務比率の「組み合わせの矛盾」を3つの指標で検出します。
> 個々の数字が正常でも、複数の比率が同時におかしい場合に高スコアになります。
""")

        # ── メトリクス（ツールチップ付き） ─────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "⚛️ Q_risk",
            f"{row['q_risk']:.1f} / 100",
            help="財務比率10ペアの位相ズレをRMS合成したスコア。"
                 "40以上で要注意・60以上で財務構造の矛盾あり。"
                 "個々の指標は正常でも複数の組み合わせがおかしい場合に上昇します。",
        )
        c2.metric(
            "🔄 動態スコア",
            f"{row['dynamics_score']:.1f} / 100",
            help="資産回転率（売上÷総資産）とROAで計算する「企業の動きやすさ」。"
                 "大きな資産を持ちながら売上が少ない（失速状態）と低くなります。"
                 "目安：60以上が健全・40未満は要注意。",
        )
        c3.metric(
            "📊 エントロピーリスク",
            f"{row['entropy_risk']:.1f} / 100",
            help="財務数値の先頭桁分布がベンフォードの法則からどれだけ外れているかを示します。"
                 "自然な数値は1始まりが30%と最も多い法則があり、"
                 "操作・粉飾された数字はこの分布が崩れます。80以上で要注意。",
        )
        c4.metric(
            "総合判定",
            risk_level,
            help="Q_risk・動態スコア・エントロピーリスクを統合した判定。"
                 "🟢正常 / 🟡要注意（Q_risk≥30 or 動態スコア<40）/ 🔴高リスク（Q_risk≥60 or 失速 or エントロピー異常）",
        )

        # ── 詳細数値 ──────────────────────────────────────────────────────────
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔄 物理動態指標**")
            st.caption("企業を「回転する機械」として捉えた効率性の指標です。")
            at = row["asset_turnover"]
            at_comment = "✅ 良好" if at >= 0.8 else ("⚠️ やや低い" if at >= 0.3 else "🔴 低い（失速リスク）")
            st.write(f"- 資産回転率: **{at:.3f}** {at_comment}（目安: 0.8〜1.5）")
            st.caption("　売上高 ÷ 総資産。資産を使って売上を生み出す効率。低いほど資産が遊休状態。")

            roa_pct = row["roa"] * 100
            roa_comment = "✅ 良好" if roa_pct >= 2 else ("⚠️ 低め" if roa_pct >= 0 else "🔴 マイナス（資産を食いつぶし中）")
            st.write(f"- ROA（総資産利益率）: **{roa_pct:.2f}%** {roa_comment}（目安: 2〜5%）")
            st.caption("　純利益 ÷ 総資産。資産全体に対する稼ぐ力。マイナスは損失拡大中。")

            stall_flag = (at < 0.3) and (roa_pct < 0)
            stall_label = "🔴 失速中（資産回転が低くROAがマイナス）" if stall_flag else "🟢 正常"
            st.write(f"- 失速フラグ: {stall_label}")
            st.caption("　資産回転率<0.3 かつ ROA<0 の両方を満たす場合に点灯します。")

        with col_b:
            st.markdown("**⚛️ 量子干渉スコア補正（参考）**")
            st.caption("実際のスコアには反映されません。将来的な参考値として表示しています。")
            base_score = float(last_res.get("score") or last_res.get("hantei_score") or 75.0)
            corr = analyzer.apply_quantum_correction(
                base_score=base_score,
                q_risk=float(row["q_risk"]),
                dynamics_score=float(row["dynamics_score"]),
                entropy_risk=float(row["entropy_risk"]),
            )
            st.write(f"- 現在の実スコア: **{base_score:.1f}** ← これは変わりません")
            st.write(f"- β参考補正後スコア: **{corr['corrected_score']:.1f}**")
            st.caption("　Q_risk≥40のとき exp(-0.8×Q_risk/100) で減衰させた仮の値です。")
            st.write(f"- 補正係数: {corr['correction_factor']:.3f}")
            st.caption("　1.0=補正なし・0.45=最大50%減衰（Q_risk=100のとき）")
            if corr.get("warning"):
                st.warning(f"⚠️ 検知した異常: {corr['warning']}")

        # ── グラフ表示（横並びでコンパクトに） ─────────────────────────────────
        st.divider()
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            st.markdown("**📈 フェーズ・ポートレート**")
            st.caption("企業の「動態空間」における現在案件の位置。")
            fig2, ax2 = plt.subplots(figsize=(5, 3.5))
            fig2.patch.set_facecolor("#0E1117")
            ax2.set_facecolor("#1A1C24")

            color_node = "#FF007A" if stall_flag else "#00F5FF"
            ax2.scatter([at], [roa_pct], color=color_node, s=150, zorder=5, 
                        edgecolors="#FFFFFF", linewidth=1.5, alpha=0.9)
            ax2.scatter([at], [roa_pct], color=color_node, s=400, zorder=4, alpha=0.3)

            ax2.axvline(0.3, color="#FFB800", linestyle="--", alpha=0.7)
            ax2.axhline(0.0, color="#FF007A", linestyle="--", alpha=0.5)

            x_min = min(0.0, at - 0.2)
            x_max = max(2.0, at + 0.5)
            y_min = min(-10.0, roa_pct - 2.0)
            y_max = max(10.0, roa_pct + 5.0)
            
            ax2.fill_betweenx([y_min, 0], 0, 0.3, alpha=0.15, color="#FF007A")
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_xlabel("資産回転率", color="#FFFFFF", fontsize=8)
            ax2.set_ylabel("ROA（%）", color="#FFFFFF", fontsize=8)
            ax2.tick_params(colors="#FFFFFF", labelsize=8)
            ax2.grid(True, color="#31363F", alpha=0.6)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        with col_g2:
            st.markdown("**📡 財務リスクレーダー**")
            st.caption("3指標のバランス（低いほど良 / 動態は高いほど良）。")
            _plot_radar(row)
        return

    # ── 複数件表示（DB一括） ──────────────────────────────────────────────────
    if df_labels is not None:
        result = pd.concat([result.reset_index(drop=True), df_labels.reset_index(drop=True)], axis=1)

    # Q_risk上位N件を抽出
    result_sorted = result.sort_values("q_risk", ascending=False).head(top_n)

    # サマリーメトリクス
    c1, c2, c3 = st.columns(3)
    high_risk = (result["q_risk"] >= 60).sum()
    med_risk  = ((result["q_risk"] >= 40) & (result["q_risk"] < 60)).sum()
    stall     = result["stall_flag"].sum()
    c1.metric("🔴 高リスク（Q_risk≥60）", f"{high_risk} 件")
    c2.metric("🟡 要注意（Q_risk≥40）",   f"{med_risk} 件")
    c3.metric("⚡ 失速状態",              f"{stall} 件")

    # グラフ横並び（DB一括）
    st.divider()
    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.markdown("#### Q_risk 分布")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1A1C24")
        
        ax.hist(result["q_risk"], bins=20, color="#00F5FF", edgecolor="#1A1C24", alpha=0.8)
        ax.axvline(40, color="#FFB800", linestyle="--", label="要注意(40)")
        ax.axvline(60, color="#FF007A", linestyle="--", label="高リスク(60)")
        ax.set_xlabel("Q_risk スコア", color="#FFFFFF", fontsize=8)
        ax.set_ylabel("件数", color="#FFFFFF", fontsize=8)
        ax.tick_params(colors="#FFFFFF", labelsize=8)
        ax.grid(True, color="#31363F", alpha=0.6)
        
        legend = ax.legend(facecolor="#0E1117", edgecolor="#31363F", fontsize=7)
        plt.setp(legend.get_texts(), color="#FFFFFF")
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_g4:
        st.markdown("#### フェーズ・ポートレート")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor("#0E1117")
        ax2.set_facecolor("#1A1C24")
        
        colors_plot = ["#FF007A" if s else "#00F5FF" for s in result["stall_flag"]]
        ax2.scatter(result["asset_turnover"], result["roa"] * 100,
                    c=colors_plot, s=40, alpha=0.8, edgecolors="#FFFFFF", linewidth=0.5, zorder=5)
        ax2.axvline(0.3, color="#FFB800", linestyle="--", alpha=0.7)
        ax2.axhline(0.0, color="#FF007A", linestyle="--", alpha=0.5)
        
        ax2.fill_betweenx(
            [result["roa"].min() * 100 - 1, 0], 0, 0.3, alpha=0.15, color="#FF007A"
        )
        ax2.set_xlabel("資産回転率", color="#FFFFFF", fontsize=8)
        ax2.set_ylabel("ROA（%）", color="#FFFFFF", fontsize=8)
        ax2.tick_params(colors="#FFFFFF", labelsize=8)
        ax2.grid(True, color="#31363F", alpha=0.6)
        
        red_p  = mpatches.Patch(color="#FF007A", label="失速")
        blue_p = mpatches.Patch(color="#00F5FF", label="正常")
        legend2 = ax2.legend(handles=[red_p, blue_p], facecolor="#0E1117", edgecolor="#31363F", fontsize=7)
        plt.setp(legend2.get_texts(), color="#FFFFFF")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # 上位テーブル
    st.divider()
    st.markdown(f"#### ⚠️ Q_risk 上位 {top_n} 件（参考）")
    display_cols = ["q_risk", "dynamics_score", "entropy_risk", "stall_flag",
                    "data_integrity_flag", "quantum_risk_level"]
    if "企業名" in result_sorted.columns:
        display_cols = ["企業名", "最終結果"] + display_cols
    st.dataframe(
        result_sorted[display_cols].reset_index(drop=True),
        use_container_width=True,
    )

    st.caption("※ このデータは参考表示です。実際の審査判定・係数・DBに影響しません。")


def _plot_radar(row: pd.Series) -> None:
    """単件のレーダーチャート（Q_risk・動態・エントロピーの三角形）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["Q_risk\n(低いほど良)", "動態スコア\n(高いほど良)", "エントロピーリスク\n(低いほど良)"]
    values = [
        row["q_risk"],
        row["dynamics_score"],
        row["entropy_risk"],
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#1A1C24")
    
    # チャートの描画
    ax.plot(angles_plot, values_plot, "o-", linewidth=2, color="#00F5FF", markersize=6)
    ax.fill(angles_plot, values_plot, alpha=0.3, color="#00F5FF")
    
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=9, color="#FFFFFF")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#888888")
    ax.grid(True, color="#31363F", alpha=0.6)
    ax.spines['polar'].set_color('#31363F')
    
    ax.set_title("財務リスクレーダー（β参考）", pad=20, color="#FFFFFF")
    st.pyplot(fig, use_container_width=True)
    plt.close()


def _sf(val, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None and str(val).strip() not in ("", "nan") else default
    except (TypeError, ValueError):
        return default
