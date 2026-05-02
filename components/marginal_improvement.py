"""
限界改善シミュレーター — ボーダーラインケースに対し「最小変化量で承認圏内に到達できる指標」を提示する。
新規モデル学習不要。既存の run_quick_scoring() を反復呼び出しして実現。
"""
import streamlit as st
import copy

# ── 数値改善シナリオ定義 ────────────────────────────────────────────────────
# 正の係数を持つ（または業務上の改善方向が明確な）指標のみ対象とする。
# machines, net_income, lease_credit は現行モデルで負の係数を持つため除外。
# key: run_quick_scoring() に渡す inputs のキー（千円単位）
# steps: 試す改善量（percent なら割合、additive なら千円加算）
NUMERIC_SCENARIOS = [
    {
        "key": "op_profit",
        "label": "営業利益",
        "type": "additive",
        "steps": [500, 1000, 2000, 3000, 5000, 8000, 15000, 30000],
    },
    {
        "key": "nenshu",
        "label": "売上高",
        "type": "percent",
        "steps": [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0],
    },
    {
        "key": "gross_profit",
        "label": "粗利益",
        "type": "additive",
        "steps": [500, 1000, 2000, 5000, 10000, 20000],
    },
    {
        "key": "contracts",
        "label": "取引実績件数",
        "type": "additive_int",
        "steps": [1, 2, 3, 5, 10, 20],
    },
    {
        "key": "bank_credit",
        "label": "銀行与信枠",
        "type": "percent",
        "steps": [0.10, 0.20, 0.50, 1.0, 2.0],
    },
]

GRADE_OPTIONS = ["1-3", "4-6", "要注意", "無格付"]

GRADE_LABELS = {
    "1-3": "格付1〜3（優良）",
    "4-6": "格付4〜6（中位）",
    "要注意": "要注意",
    "無格付": "無格付",
}


def _build_base_inputs(res: dict) -> dict:
    """res (last_result) と session_state から run_quick_scoring() 用 inputs dict を復元する。"""
    fin = res.get("financials", {})
    return {
        "nenshu":      float(fin.get("nenshu") or 0),
        "op_profit":   float(fin.get("op_profit") or fin.get("rieki") or 0),
        "ord_profit":  float(fin.get("ord_profit") or 0),
        "net_income":  float(fin.get("net_income") or 0),
        "net_assets":  float(fin.get("net_assets") or 0),
        "total_assets": float(fin.get("assets") or fin.get("total_assets") or 0),
        "gross_profit": float(fin.get("gross_profit") or 0),
        "machines":    float(fin.get("machines") or 0),
        "other_assets": float(fin.get("other_assets") or 0),
        "bank_credit": float(fin.get("bank_credit") or 0),
        "lease_credit": float(fin.get("lease_credit") or 0),
        "depreciation": float(fin.get("depreciation") or 0),
        "dep_expense":  float(fin.get("dep_expense") or 0),
        "rent_expense": float(fin.get("rent_expense") or 0),
        "contracts":   int(st.session_state.get("contracts") or 0),
        "grade":       st.session_state.get("grade") or "1-3",
        "industry_major": res.get("industry_major") or st.session_state.get("select_major") or "D 建設業",
        "industry_sub":   res.get("industry_sub")   or st.session_state.get("select_sub")   or "06 総合工事業",
        "customer_type":  st.session_state.get("customer_type") or "既存先",
        "main_bank":      st.session_state.get("main_bank") or "非メイン先",
        "competitor":     st.session_state.get("competitor_status") or "競合なし",
        "intuition_score": 0.0,  # シミュレーション時は直感補正なし
        "asset_score":    float((res.get("_ts_result") or {}).get("asset_score") or 50.0),
    }


def _fmt_delta(scenario: dict, step: float, base_val: float) -> str:
    """シナリオとステップから表示文字列を生成。"""
    key = scenario["type"]
    if key == "percent":
        added = base_val * step
        return f"+{step*100:.0f}% (約{added/1000:,.0f}百万円増)"
    elif key in ("additive", "additive_int"):
        return f"+{step/1000:,.1f}百万円" if key == "additive" else f"+{int(step)}件"
    return str(step)


def compute_improvements(res: dict) -> tuple[list[dict], list[dict]]:
    """
    res (last_result) をもとに改善シナリオを評価して返す。

    Returns:
        (reachable, partial):
            reachable — 単独で承認圏内に到達できる改善案（gain 昇順）
            partial   — 承認圏内に届かないが最大改善できる案（gain 降順）
    """
    from scoring_core import run_quick_scoring, APPROVAL_LINE

    target = float(APPROVAL_LINE)
    base_inputs = _build_base_inputs(res)

    # シミュレーターは run_quick_scoring 内で一貫したスコアを使う。
    # res["score"] はフルパイプライン（定性・物件補正込み）のため基準が違う場合がある。
    try:
        current_score = float(run_quick_scoring(base_inputs).get("score", 0))
    except Exception:
        current_score = float(res.get("score") or 0)

    reachable: list[dict] = []
    partial: list[dict] = []

    # ── 数値特徴量の改善 ────────────────────────────────────────────────────
    for scenario in NUMERIC_SCENARIOS:
        key = scenario["key"]
        base_val = float(base_inputs.get(key) or 0)
        best_gain = 0.0
        best_entry: dict | None = None

        for step in scenario["steps"]:
            modified = copy.copy(base_inputs)
            if scenario["type"] == "percent":
                modified[key] = base_val * (1.0 + step)
            elif scenario["type"] == "additive":
                modified[key] = base_val + step
            elif scenario["type"] == "additive_int":
                modified[key] = int(base_val) + int(step)
            else:
                continue

            try:
                new_score = float(run_quick_scoring(modified).get("score", current_score))
            except Exception:
                continue

            gain = round(new_score - current_score, 1)
            if gain > best_gain and gain > 0:
                best_gain = gain
                best_entry = {
                    "label": scenario["label"],
                    "delta_label": _fmt_delta(scenario, step, base_val),
                    "new_score": round(new_score, 1),
                    "gain": gain,
                }

            if new_score >= target:
                # 最小ステップで達成 → これ以上は不要
                reachable.append(best_entry)  # type: ignore[arg-type]
                break
        else:
            # どのステップでも target 未達
            if best_entry is not None:
                partial.append(best_entry)

    # ── 格付改善 ── 実務的に可能な改善（格上げのみ）を提案 ──────────────────
    # 信用力順: 1-3 > 4-6 > 要注意 > 無格付
    # 格下げは提案しない（1-3 企業への「4-6 にせよ」は避ける）
    CREDIT_QUALITY_ORDER = ["1-3", "4-6", "要注意", "無格付"]
    current_grade = base_inputs.get("grade", "1-3")
    current_grade_idx = CREDIT_QUALITY_ORDER.index(current_grade) if current_grade in CREDIT_QUALITY_ORDER else 3

    grade_best: dict | None = None
    # 現在の格付より信用力が高い（= index が小さい）グレードのみ提案
    for candidate_grade in CREDIT_QUALITY_ORDER[:current_grade_idx]:
        modified = copy.copy(base_inputs)
        modified["grade"] = candidate_grade
        try:
            new_score = float(run_quick_scoring(modified).get("score", current_score))
        except Exception:
            continue

        gain = round(new_score - current_score, 1)
        if gain <= 0:
            continue

        entry = {
            "label": "信用格付",
            "delta_label": f"{GRADE_LABELS.get(current_grade, current_grade)} → {GRADE_LABELS.get(candidate_grade, candidate_grade)}",
            "new_score": round(new_score, 1),
            "gain": gain,
        }
        if new_score >= target:
            reachable.append(entry)
            grade_best = entry
            break
        elif grade_best is None or gain > grade_best["gain"]:
            grade_best = entry

    if grade_best is not None and grade_best not in reachable:
        partial.append(grade_best)

    # ── ソート ─────────────────────────────────────────────────────────────
    reachable.sort(key=lambda x: x["gain"])         # 改善幅が小さい順（実現しやすい）
    partial.sort(key=lambda x: x["gain"], reverse=True)  # 最大改善が上

    return reachable, partial, round(current_score, 1)


def render_improvement_advisor(res: dict) -> None:
    """
    スコアがボーダーライン付近の案件に対し、改善提案パネルを Streamlit で描画する。
    スコアが承認圏内 + 2 以上なら何も表示しない。
    """
    from scoring_core import APPROVAL_LINE

    current_score = float(res.get("score") or 0)
    hantei = res.get("hantei", "")

    # 表示条件: 要審議 or ギリギリ承認（+2点以内）
    if current_score >= APPROVAL_LINE + 2:
        return

    gap = max(0.0, APPROVAL_LINE - current_score)
    with st.expander(
        f"📈 **改善提案シミュレーター** — 承認ライン {APPROVAL_LINE} まで あと {gap:.1f} 点",
        expanded=(hantei == "要審議"),
    ):
        st.caption("各指標を単独で改善した場合の推定スコアを表示します。複数の改善を組み合わせると効果が高まります。")

        with st.spinner("シミュレーション計算中..."):
            reachable, partial, quant_base_score = compute_improvements(res)

        # ── 定量スコアが既に高い場合（定性要因が主因）────────────────────
        if quant_base_score >= APPROVAL_LINE:
            st.info(
                f"📊 **定量スコアは既に {quant_base_score:.0f} 点（承認ライン以上）です。**\n\n"
                "定量的な財務数値は十分な水準にあります。合計スコアを上げるには "
                "**定性評価の改善**（顧客安定性・返済履歴・主力銀行との関係等）が効果的です。"
            )
            return

        if not reachable and not partial:
            st.info("シミュレーション結果がありません（入力データが不足している可能性があります）。")
            return

        # 定量ベーススコアを注釈として表示
        if abs(quant_base_score - current_score) > 3:
            st.caption(
                f"※ 定量モデル（LR+LGB 数値のみ）での推定ベーススコア: {quant_base_score:.1f} 点。"
                f"表示スコア（{current_score:.1f} 点）は定性・物件補正を含みます。"
            )

        # ── 承認圏内に届く改善案 ──────────────────────────────────────────
        if reachable:
            st.markdown(
                f"#### ✅ 単独改善で承認ライン（{APPROVAL_LINE}点）に到達できる案"
            )
            _render_table(reachable, quant_base_score, APPROVAL_LINE, reached=True)

        # ── 部分改善（目標未達だが改善方向） ─────────────────────────────
        if partial:
            st.markdown("#### 🔧 目標未達だが改善方向（複数組み合わせを検討）")
            _render_table(partial, quant_base_score, APPROVAL_LINE, reached=False)

        # ── 補足説明 ───────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "※ 各シミュレーションは他の条件を固定した上での「単独改善」の試算です。"
            "実際のスコアは複合的な要因で変動します。直感スコア補正（±3点）は含まれていません。"
        )


def _render_table(items: list[dict], current_score: float, target: float, reached: bool) -> None:
    """改善案テーブルを HTML で描画する。"""
    rows_html = ""
    for item in items:
        gain = item["gain"]
        new_score = item["new_score"]

        if reached:
            score_color = "#16a34a"
            badge = f'<span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75rem;">承認圏内</span>'
        else:
            score_color = "#d97706"
            badge = f'<span style="background:#d97706;color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75rem;">+{gain:.1f}点</span>'

        rows_html += f"""
        <tr>
          <td style="padding:8px 12px;font-weight:600;">{item['label']}</td>
          <td style="padding:8px 12px;color:#475569;">{item['delta_label']}</td>
          <td style="padding:8px 12px;font-weight:700;color:{score_color};">{new_score:.1f}</td>
          <td style="padding:8px 12px;">{badge}</td>
        </tr>
        """

    st.markdown(
        f"""
        <table style="width:100%;border-collapse:collapse;font-size:0.875rem;">
          <thead>
            <tr style="background:#f1f5f9;color:#64748b;font-size:0.8rem;">
              <th style="padding:8px 12px;text-align:left;">指標</th>
              <th style="padding:8px 12px;text-align:left;">改善量</th>
              <th style="padding:8px 12px;text-align:left;">推定スコア</th>
              <th style="padding:8px 12px;text-align:left;">効果</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")  # spacing
