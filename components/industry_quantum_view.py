"""
業種別量子解析ビュー
各業種の「財務矛盾パターン」を量子干渉スコアで可視化する。
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "data" / "lease_data.db"
QMODEL_PATH = str(PROJECT_ROOT / "data" / "quantum_model.joblib")
FEEDBACK_PATH = PROJECT_ROOT / "data" / "quantum_feedback.jsonl"

# 業種小分類コード → 大分類コードの表示名マップ
_MAJOR_LABELS: dict[str, str] = {
    "D": "🏗️ 建設業",
    "E": "🏭 製造業",
    "H": "🚛 運輸業",
    "P": "🏥 医療・福祉",
    "K": "🏢 不動産・物品賃貸",
    "I": "🛒 卸売・小売業",
    "G": "💻 情報通信業",
    "J": "🏦 金融・保険業",
}

_PAIR_LABELS: dict[str, str] = {
    "op_profit_x_depreciation":  "利益 vs 減価償却",
    "op_profit_x_trend_val":     "利益 vs トレンド",
    "net_income_x_ord_profit":   "純利益 vs 経常利益",
    "op_profit_x_machines":      "利益 vs 機械装置",
    "op_profit_x_equip_total":   "利益 vs 設備合計",
    "qualit_score_x_op_profit":  "定性スコア vs 利益",
    "machines_x_op_profit":      "機械装置 vs 利益",
    "depreciation_x_machines":   "減価償却 vs 機械装置",
    "depreciation_x_op_profit":  "減価償却 vs 利益",
    "net_income_x_op_profit":    "純利益 vs 営業利益",
    "ord_profit_x_op_profit":    "経常利益 vs 営業利益",
    "machines_x_net_income":     "機械装置 vs 純利益",
}


@st.cache_resource
def _load_qgate():
    import os
    if not os.path.exists(QMODEL_PATH):
        return None
    try:
        from quantum_analysis_module import QuantumGate
        return QuantumGate.load(QMODEL_PATH)
    except Exception:
        return None


def _load_past_cases() -> list[dict]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT data, score, final_status, industry_sub FROM past_cases"
        ).fetchall()
    finally:
        conn.close()
    results = []
    for raw, score, status, industry_sub_col in rows:
        try:
            d = json.loads(raw)
            d["_score"] = float(score or 0)
            d["_status"] = status or "未登録"
            inp = d.setdefault("inputs", {})
            if not inp.get("industry_sub") and industry_sub_col:
                inp["industry_sub"] = industry_sub_col
            results.append(d)
        except Exception:
            pass
    return results


def _get_major_code(case: dict) -> str:
    from quantum_analysis_module import _infer_major_code
    inp = case.get("inputs", {})
    major = str(inp.get("industry_major") or "")
    code = major.split(" ")[0].strip() if major else ""
    if not code:
        sub = str(inp.get("industry_sub") or "")
        code = _infer_major_code(sub)
    return code


def _render_radar(pair_anomalies: dict[str, float], industry_name: str) -> None:
    labels = [_PAIR_LABELS.get(k, k) for k in pair_anomalies]
    values = list(pair_anomalies.values())
    if not labels:
        st.info("ペアデータなし")
        return

    # レーダーは閉じる必要がある
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(99, 110, 250, 0.25)",
        line=dict(color="rgb(99, 110, 250)", width=2),
        name=industry_name,
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(t=30, b=30, l=40, r=40),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)


def _write_feedback(case: dict, label: str) -> None:
    """フィードバックを quantum_feedback.jsonl に追記する"""
    import datetime
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "label": label,       # "妥当" | "要確認"
        "inputs": case.get("inputs", {}),
        "q_risk": case.get("_q_risk", 0),
        "q_verdict": case.get("_q_verdict", ""),
        "status": case.get("_status", ""),
        "score": case.get("_score", 0),
    }
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _render_case_table(records: list[dict]) -> None:
    import pandas as pd
    rows = []
    for r in records:
        inp = r.get("inputs", {})
        rows.append({
            "業種": inp.get("industry_sub", "不明"),
            "スコア": r.get("_score", 0),
            "結果": r.get("_status", ""),
            "Q_risk": r.get("_q_risk", 0),
            "判定": r.get("_q_verdict", ""),
            "主因": r.get("_q_top_pair", ""),
        })
    if not rows:
        st.info("案件データなし")
        return
    df = pd.DataFrame(rows).sort_values("Q_risk", ascending=False)

    def _color(val):
        if val >= 60:
            return "background-color: #ffcccc"
        if val >= 35:
            return "background-color: #fff3cc"
        return ""

    styled = df.style.applymap(_color, subset=["Q_risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # フィードバックボタン（上位 Q_risk 案件を対象）
    top_cases = sorted(records, key=lambda c: c.get("_q_risk", 0), reverse=True)[:5]
    if top_cases:
        st.markdown("##### Q_risk 上位案件へのフィードバック")
        st.caption("判定の妥当性をフィードバックすると、次回学習で重みに反映されます。")
        for i, c in enumerate(top_cases):
            inp = c.get("inputs", {})
            label = f"Q_risk={c.get('_q_risk', 0):.0f}  {inp.get('industry_sub', '?')}  {c.get('_status', '')}"
            col_lbl, col_ok, col_ng = st.columns([4, 1, 1])
            col_lbl.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            if col_ok.button("✅ 妥当", key=f"fbk_ok_{i}", use_container_width=True):
                _write_feedback(c, "妥当")
                st.toast("フィードバックを記録しました（妥当）")
            if col_ng.button("⚠️ 要確認", key=f"fbk_ng_{i}", use_container_width=True):
                _write_feedback(c, "要確認")
                st.toast("フィードバックを記録しました（要確認）")


def _render_explanation() -> None:
    with st.expander("💡 このモジュールが表しているもの（クリックで展開）", expanded=False):
        st.markdown("""
#### 一言で言うと
**「財務変数どうしの期待される関係が崩れている度合い」** を 0〜100 のスコアで表します。

---

#### 直感的なイメージ

例えば建設業なら「利益が高ければ設備（機械・車両）も大きいはず」という期待される整合性があります。

| 状態 | 意味 |
|---|---|
| 利益も高い ＋ 設備も大きい | 整合 → **リスク低** |
| 利益は高い ＋ 設備がほぼゼロ | 矛盾 → **リスク高** |

この「どのくらい矛盾しているか」を数値化したのが **Q_risk** です。

---

#### なぜ既存モデルで捕捉できないか

| モデル | 何を見ている |
|---|---|
| LightGBM | 個々の変数の絶対値（利益が何円か）でスコアを決める |
| マハラノビス距離 | 全変数を一括で「全体的な異常さ」として見る |
| **量子解析（本モジュール）** | **変数間の関係性**（利益と設備の比率・整合性）を見る |

「利益が高い」こと自体は LightGBM が高スコアとして評価します。でも
**「利益が高いのに設備がない」という矛盾の組み合わせ** は既存モデルが見落としやすい構造にあります。

---

#### Q_risk の解釈

| スコア | 判定 | 意味 |
|---|---|---|
| 0 〜 34 | 正常 | 財務変数間の関係に矛盾なし |
| 35 〜 59 | 要再審 | 1つ以上のペアに有意な矛盾あり。追加確認を推奨 |
| 60 〜 100 | 高リスク | 複数ペアで強い矛盾。高スコアでも失注・否決を示唆 |

---

#### 技術的な仕組み（概略）

財務変数を **Bloch 球面（単位球）上の点** に変換し、2変数の球面上の距離（干渉乖離度）を計算します。
業種ごとに「整合しているはずのペア」を定義し、そのペアの乖離度を重み付き集約したものが Q_risk です。

---

#### ⚠️ 限界・注意事項

- 「矛盾あり = 必ず失注」ではありません。業種特性・M&A後の一時的歪みが原因の場合もあります
- データが少ない（現在 40件程度）ため、重みの精度は限定的です。**案件が増えるほど精度が上がります**
- あくまで既存スコアの補助シグナルとして活用してください
        """)


def render_industry_quantum_view() -> None:
    st.title("⚛️ 業種別量子解析")
    st.caption("各業種の財務矛盾パターンを量子干渉スコアで可視化します")

    _render_explanation()

    gate = _load_qgate()
    if gate is None:
        st.error("量子モデル未生成。ターミナルで `python3 train_quantum.py` を実行してください。")
        return

    cases = _load_past_cases()
    if not cases:
        st.warning("過去案件データがありません。")
        return

    # 業種分類
    from quantum_analysis_module import INDUSTRY_RISK_DESCRIPTIONS, _extract_features

    case_by_major: dict[str, list[dict]] = {}
    for c in cases:
        code = _get_major_code(c)
        case_by_major.setdefault(code or "?", []).append(c)

    # Q_risk 計算（全案件）
    for c in cases:
        try:
            r = gate.predict(c)
            c["_q_risk"] = r["quantum_risk"]
            c["_q_verdict"] = r["verdict"]
            c["_q_pairs"] = r["pair_anomalies"]
            top = max(r["pair_anomalies"].items(), key=lambda x: x[1], default=("", 0))
            c["_q_top_pair"] = _PAIR_LABELS.get(top[0], top[0])
        except Exception:
            c["_q_risk"] = 0.0
            c["_q_verdict"] = "エラー"
            c["_q_pairs"] = {}
            c["_q_top_pair"] = ""

    # ── サイドバー: 業種選択 ────────────────────────────────────────────────
    available_codes = sorted([k for k in case_by_major if k != "?"])
    if not available_codes:
        st.warning("業種コードを特定できる案件がありません。")
        return

    options = {
        _MAJOR_LABELS.get(c, f"その他({c})"): c
        for c in available_codes
    }
    selected_label = st.selectbox(
        "業種を選択",
        list(options.keys()),
        key="industry_quantum_select",
    )
    selected_code = options[selected_label]
    industry_cases = case_by_major.get(selected_code, [])

    desc = INDUSTRY_RISK_DESCRIPTIONS.get(selected_code)

    # ── 業種概要 ─────────────────────────────────────────────────────────────
    if desc:
        st.markdown(f"### {desc['name']} の特徴的リスクパターン")
        st.info(desc["risk_summary"])
        st.markdown("**監視ペア:** " + " / ".join(desc["key_pairs"]))
    else:
        st.markdown(f"### {selected_label}")

    st.divider()

    # ── KPI ─────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total = len(industry_cases)
    high_risk = sum(1 for c in industry_cases if c.get("_q_risk", 0) >= 60)
    review = sum(1 for c in industry_cases if 35 <= c.get("_q_risk", 0) < 60)
    lost = sum(1 for c in industry_cases if c.get("_status") == "失注")
    col1.metric("案件数", total)
    col2.metric("高リスク (≥60)", high_risk, delta=f"{high_risk/total*100:.0f}%" if total else "0%")
    col3.metric("要再審 (35-59)", review)
    col4.metric("失注件数", lost)

    st.divider()

    # ── レーダーチャート + 案件テーブル ─────────────────────────────────────
    col_radar, col_table = st.columns([1, 1])

    with col_radar:
        st.markdown("#### 平均ペア乖離度レーダー")
        # 業種内全案件の平均 pair_anomalies
        all_pairs: dict[str, list[float]] = {}
        for c in industry_cases:
            for k, v in c.get("_q_pairs", {}).items():
                all_pairs.setdefault(k, []).append(v)
        avg_pairs = {k: sum(vs) / len(vs) for k, vs in all_pairs.items() if vs}
        if avg_pairs:
            _render_radar(avg_pairs, desc["name"] if desc else selected_label)
        else:
            st.info("レーダーデータなし")

    with col_table:
        st.markdown("#### 案件一覧 (Q_risk 降順)")
        _render_case_table(industry_cases)

    st.divider()

    # ── 全業種サマリー ────────────────────────────────────────────────────────
    with st.expander("📊 全業種 Q_risk サマリー"):
        import pandas as pd
        summary_rows = []
        for code, cs in sorted(case_by_major.items()):
            q_vals = [c.get("_q_risk", 0) for c in cs]
            n_lost = sum(1 for c in cs if c.get("_status") == "失注")
            n_high = sum(1 for q in q_vals if q >= 60)
            n_review = sum(1 for q in q_vals if 35 <= q < 60)
            summary_rows.append({
                "業種": _MAJOR_LABELS.get(code, f"その他({code})"),
                "件数": len(cs),
                "平均Q_risk": round(sum(q_vals) / len(q_vals), 1) if q_vals else 0,
                "高リスク": n_high,
                "要再審": n_review,
                "失注": n_lost,
            })
        df_sum = pd.DataFrame(summary_rows).sort_values("平均Q_risk", ascending=False)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
