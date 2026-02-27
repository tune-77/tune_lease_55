"""
credit_limit.py — 与信枠自動提案モジュール

【重要な前提】
  fin["bank_credit"]  = 銀行与信の現在残高（与信枠全体ではない）
  fin["lease_credit"] = リース与信の現在残高（与信枠全体ではない）
  → これらは「現在の借入残高」であり「与信限度額」ではない点に注意。
  → 本モジュールでは「残高から推計した実態」として活用する。

expose:
  suggest_credit_limit(res) -> CreditLimitResult
  render_credit_limit_ui(res)   ← Streamlit UI
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ────────────────────────────────────────────
# 係数テーブル（スコアに応じた倍率）
# ────────────────────────────────────────────

# スコア → 純資産に対する与信上限倍率
_SCORE_TO_MULT = [
    (90, 1.5),
    (80, 1.2),
    (70, 1.0),
    (60, 0.7),
    (50, 0.45),
    (0,  0.25),
]

# 自己資本比率 → 補正係数
_EQ_ADJUST = [
    (50, 1.20),
    (40, 1.10),
    (30, 1.00),
    (20, 0.88),
    (10, 0.72),
    (0,  0.55),
]

# 営業利益率 → 補正係数
_OP_ADJUST = [
    (10, 1.15),
    (5,  1.05),
    (0,  1.00),
    (-5, 0.85),
    (-99, 0.65),
]

# 残高 → 推計与信限度額の換算比（残高はざっくり限度額の 50〜80% が使われると仮定）
_BALANCE_TO_LIMIT_FACTOR = 1.6   # 残高 × 1.6 ≒ 推定限度額


def _lookup(table, val: float) -> float:
    """テーブルを上から検索して最初に条件を満たした係数を返す"""
    for threshold, coeff in table:
        if val >= threshold:
            return coeff
    return table[-1][1]


@dataclass
class CreditLimitResult:
    """与信枠提案の計算結果"""
    # ─ インプット
    net_assets: float        # 純資産（万円）
    nenshu: float            # 売上高（万円）
    bank_credit_bal: float   # 銀行与信残高（万円）
    lease_credit_bal: float  # リース与信残高（万円）
    score: float             # 審査スコア
    user_eq: float           # 自己資本比率（%）
    user_op: float           # 営業利益率（%）

    # ─ 計算結果
    base_limit: float = 0.0          # 純資産基準の上限
    sales_limit: float = 0.0         # 売上基準の上限
    gross_limit: float = 0.0         # 純資産+売上の統合上限
    eq_adj: float = 1.0              # 自己資本補正
    op_adj: float = 1.0              # 利益率補正
    score_mult: float = 1.0          # スコア倍率
    est_total_obligation: float = 0.0  # 推計総与信負担
    available: float = 0.0           # 利用可能推計額
    suggested: float = 0.0           # 提案与信枠
    tier: str = ""                   # ランク（A〜D）
    tier_label: str = ""             # ランクラベル
    remarks: list[str] = field(default_factory=list)  # 注記

    def as_dict(self) -> dict:
        return self.__dict__


def suggest_credit_limit(res: dict) -> CreditLimitResult:
    """
    審査結果 res から与信枠を試算して CreditLimitResult を返す。

    計算フロー：
      1. 純資産基準  = net_assets × score_mult
      2. 売上基準    = nenshu × 0.07
      3. 統合上限    = min(純資産基準, 売上基準) × eq_adj × op_adj
      4. 推計総負担  = (bank_bal + lease_bal) × 残高係数
      5. 利用可能額  = 統合上限 - 推計総負担
      6. 提案与信枠  = max(0, 利用可能額)
                        → 最低保証 (score>=70 なら 100万)
    """
    fin   = res.get("financials", {}) or {}
    score = float(res.get("score", 0))
    eq    = float(res.get("user_eq", 0))
    op    = float(res.get("user_op", 0))

    net_assets   = float(fin.get("net_assets", 0) or 0)
    nenshu       = float(fin.get("nenshu", 0) or 0)
    bank_bal     = float(fin.get("bank_credit", 0) or 0)
    lease_bal    = float(fin.get("lease_credit", 0) or 0)

    remarks = []

    # ── 各係数 ──────────────────────────────────────────────
    score_mult = _lookup(_SCORE_TO_MULT, score)
    eq_adj     = _lookup(_EQ_ADJUST, eq)
    op_adj     = _lookup(_OP_ADJUST, op)

    # ── 基準値計算 ──────────────────────────────────────────
    base_limit  = net_assets * score_mult           # 純資産ベース（万円）
    sales_limit = nenshu * 0.07                     # 売上高の 7%

    if net_assets <= 0:
        gross_limit = sales_limit * score_mult
        remarks.append("純資産が0以下のため売上基準を主軸に算出しています。")
    else:
        gross_limit = min(base_limit, sales_limit) if sales_limit > 0 else base_limit

    gross_limit = gross_limit * eq_adj * op_adj

    # ── 推計総与信負担 ───────────────────────────────────────
    # 残高は実際の限度額の一部。× 1.6 で推計限度額に換算して保守的に評価
    est_total = (bank_bal + lease_bal) * _BALANCE_TO_LIMIT_FACTOR
    if bank_bal > 0 or lease_bal > 0:
        remarks.append(
            f"銀行与信残高 {bank_bal:,.0f}万・リース与信残高 {lease_bal:,.0f}万は"
            "「現在の借入残高」です。限度額全体ではないため推計換算（×1.6）で負担を評価しています。"
        )

    # ── 利用可能額・提案値 ──────────────────────────────────
    available  = gross_limit - est_total
    suggested  = max(0.0, available)

    # スコア70以上なら最低保証100万
    if score >= 70 and suggested < 100:
        suggested = 100.0
        remarks.append("スコア70以上のため最低保証（100万円）を適用しました。")

    # スコア60未満は慎重判定
    if score < 60:
        suggested = min(suggested, 500.0)
        remarks.append("スコア60未満のため上限を500万円に制限しています。")

    # ── 警告 ────────────────────────────────────────────────
    if op < 0:
        remarks.append("⚠ 営業利益率がマイナスです。収益改善まで追加与信は慎重に。")
    if eq < 20:
        remarks.append("⚠ 自己資本比率20%未満。担保・保証等の補完措置を検討してください。")
    total_bal = bank_bal + lease_bal
    if total_bal > net_assets and net_assets > 0:
        remarks.append("⚠ 既存与信残高合計が純資産を超えています。返済能力に注意。")

    # ── ランク ──────────────────────────────────────────────
    if suggested >= 3000:
        tier, tier_label = "A", "上位ランク（3,000万円以上）"
    elif suggested >= 1000:
        tier, tier_label = "B", "標準ランク（1,000〜3,000万円）"
    elif suggested >= 300:
        tier, tier_label = "C", "小口ランク（300〜1,000万円）"
    elif suggested > 0:
        tier, tier_label = "D", "限定ランク（300万円未満）"
    else:
        tier, tier_label = "—", "与信余力なし"

    return CreditLimitResult(
        net_assets=net_assets,
        nenshu=nenshu,
        bank_credit_bal=bank_bal,
        lease_credit_bal=lease_bal,
        score=score,
        user_eq=eq,
        user_op=op,
        base_limit=base_limit,
        sales_limit=sales_limit,
        gross_limit=gross_limit,
        eq_adj=eq_adj,
        op_adj=op_adj,
        score_mult=score_mult,
        est_total_obligation=est_total,
        available=available,
        suggested=suggested,
        tier=tier,
        tier_label=tier_label,
        remarks=remarks,
    )


# ────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────

def render_credit_limit_ui(res: dict):
    """Streamlit 用 UI レンダリング"""
    import streamlit as st

    st.subheader("💳 与信枠自動提案")
    st.caption(
        "審査スコア・財務指標をもとに推奨リース与信枠を試算します。"
        "なお、入力されている銀行与信・リース与信は**現在の残高**（限度額ではない）であるため、"
        "内部的に限度額への換算補正を行っています。"
    )

    clr = suggest_credit_limit(res)
    fin = res.get("financials", {}) or {}

    # ── カード形式でサマリー表示 ──────────────────────────────
    tier_colors = {"A": "#00b09b", "B": "#2c5f8f", "C": "#ff8c00", "D": "#d62428", "—": "#888"}
    tc = tier_colors.get(clr.tier, "#888")

    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {tc}22, {tc}11);
    border-left: 5px solid {tc};
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
">
    <div style="font-size:0.78rem; color:#666; margin-bottom:4px;">推奨リース与信枠</div>
    <div style="font-size:2.2rem; font-weight:bold; color:{tc}; line-height:1.1;">
        {clr.suggested:,.0f} <span style="font-size:1rem;">万円</span>
    </div>
    <div style="font-size:0.85rem; color:#444; margin-top:6px;">
        ランク <strong>{clr.tier}</strong> — {clr.tier_label}
    </div>
</div>
""", unsafe_allow_html=True)

    # ── 計算根拠 ──────────────────────────────────────────────
    with st.expander("📐 計算根拠を見る", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**インプット値**")
            rows = {
                "純資産":        f"{clr.net_assets:,.0f} 万円",
                "売上高":        f"{clr.nenshu:,.0f} 万円",
                "銀行与信残高":  f"{clr.bank_credit_bal:,.0f} 万円",
                "リース与信残高":f"{clr.lease_credit_bal:,.0f} 万円",
                "審査スコア":    f"{clr.score:.1f}",
                "自己資本比率":  f"{clr.user_eq:.1f}%",
                "営業利益率":    f"{clr.user_op:.1f}%",
            }
            for k, v in rows.items():
                st.markdown(f"- **{k}**: {v}")
        with col2:
            st.markdown("**計算ステップ**")
            st.markdown(f"""
| ステップ | 値 |
|---|---|
| ① 純資産基準 | {clr.base_limit:,.0f} 万円 |
| ② 売上基準 (×7%) | {clr.sales_limit:,.0f} 万円 |
| ③ 統合上限（小さい方） | {clr.gross_limit / clr.eq_adj / clr.op_adj:,.0f} 万円 |
| ④ 自己資本補正（×{clr.eq_adj:.2f}） | → {clr.gross_limit / clr.op_adj:,.0f} 万円 |
| ⑤ 利益率補正（×{clr.op_adj:.2f}） | → {clr.gross_limit:,.0f} 万円 |
| ⑥ 推計総与信負担 | {clr.est_total_obligation:,.0f} 万円 |
| ⑦ 利用可能額 | {clr.available:,.0f} 万円 |
| **⑧ 提案与信枠** | **{clr.suggested:,.0f} 万円** |
""")

    # ── 注記 ──────────────────────────────────────────────────
    if clr.remarks:
        st.markdown("**⚠ 注記**")
        for r in clr.remarks:
            st.info(r, icon="ℹ️")

    # ── シミュレーター ──────────────────────────────────────
    st.divider()
    st.markdown("**🎚 パラメータ調整シミュレーター**")
    st.caption("値を変えると提案額がリアルタイムで再計算されます。")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sim_score = st.slider("審査スコア", 0, 100, int(clr.score), key="cl_sim_score")
        sim_eq    = st.slider("自己資本比率(%)", 0, 100, int(clr.user_eq), key="cl_sim_eq")
    with col_b:
        sim_op    = st.slider("営業利益率(%)", -30, 30, int(clr.user_op), key="cl_sim_op")
        sim_net   = st.number_input("純資産（万円）", value=int(clr.net_assets), step=100, key="cl_sim_net")
    with col_c:
        sim_bank  = st.number_input("銀行与信残高（万円）", value=int(clr.bank_credit_bal), step=100, key="cl_sim_bank")
        sim_lease = st.number_input("リース与信残高（万円）", value=int(clr.lease_credit_bal), step=100, key="cl_sim_lease")

    sim_fin = dict(fin)
    sim_fin.update({
        "net_assets":   sim_net,
        "nenshu":       clr.nenshu,
        "bank_credit":  sim_bank,
        "lease_credit": sim_lease,
    })
    sim_res = dict(res)
    sim_res.update({"score": sim_score, "user_eq": sim_eq, "user_op": sim_op, "financials": sim_fin})
    sim_clr = suggest_credit_limit(sim_res)

    tc2 = tier_colors.get(sim_clr.tier, "#888")
    delta = sim_clr.suggested - clr.suggested
    delta_str = f"{'▲' if delta >= 0 else '▼'} {abs(delta):,.0f}万円"
    st.markdown(f"""
<div style="
    background:{tc2}18; border:1px solid {tc2}55;
    border-radius:6px; padding:10px 16px; margin-top:8px;
">
    <span style="font-size:0.75rem;color:#666;">シミュレーション結果</span><br>
    <span style="font-size:1.6rem;font-weight:bold;color:{tc2};">{sim_clr.suggested:,.0f} 万円</span>
    &nbsp;<span style="font-size:0.85rem;color:{'#00b09b' if delta>=0 else '#d62428'};">{delta_str}</span>
    &nbsp;ランク <strong>{sim_clr.tier}</strong>
</div>
""", unsafe_allow_html=True)
