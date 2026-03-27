"""
components/asset_score_detail.py

物件スコア詳細評価コンポーネント。

- Gemini Search Grounding でリース満了時の中古価格・残存価値率を調査
- 残存価値率 → スコアへ機械変換
- カテゴリ別スライダーで担当者調整可能
- asset_scorer.py の補正発動条件のルールベース事前警告
- get_asset_context_for_ai(): 軍師AI・AIチャット等が呼べるコンテキスト生成器

セッションステートキー（asd_ プレフィックス）:
  asd_use_detail    : bool  - 詳細スコアを判定に使用するか
  asd_detail_score  : float - 詳細評価の加重平均スコア（0-100）
  asd_residual      : float - 残存価値率（%）
  asd_current_price : int   - 現在中古相場（万円）
  asd_estimated_price: int  - 満了時推定価格（万円）
  asd_search_rationale: str - 調査根拠テキスト
  asd_scores        : dict  - {item_id: score} スライダー値
"""

import json
import os
import re

import requests
import streamlit as st

from category_config import CATEGORY_SCORE_ITEMS, SCORE_GRADES

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)


# ── 残存価値率 → スコア変換（ステップ関数） ──────────────────────────────────
def _residual_to_score(residual_pct: float) -> int:
    """残存価値率（%）をスコア（0-100）に変換する。
    SCORE_GRADES の境界（S:90/A:80/B:65/C:50/D:0）に合わせて設定。
    注: 境界付近で段差が生じる。後から連続補間に改善可能。
    """
    if residual_pct >= 70:
        return 95   # S相当
    if residual_pct >= 55:
        return 82   # A相当
    if residual_pct >= 40:
        return 68   # B相当
    if residual_pct >= 25:
        return 52   # C相当
    if residual_pct >= 10:
        return 35   # D相当
    return 15


# ── Gemini Search Grounding による市場調査 ───────────────────────────────────
def _search_scores(
    category: str,
    asset_name: str,
    lease_term_here: int,
    acost_here: int,
    gemini_key: str,
) -> dict:
    """
    Gemini Search Grounding で物件の中古市場価格・残存価値率を調査する。

    Parameters
    ----------
    category       : "車両" | "IT機器" | "産業機械" | "医療機器"
    asset_name     : 物件名・型番（例: "ハイエース バン 2022年式"）
    lease_term_here: リース期間（月）
    acost_here     : 取得価格（千円）
    gemini_key     : Gemini API キー

    Returns
    -------
    dict with keys:
      current_price   : int   現在中古相場（万円）
      estimated_price : int   満了時推定価格（万円）
      residual_pct    : float 残存価値率（%）
      rationale       : str   調査根拠
      scores          : dict  {item_id: score} 非価格項目の評価
      error           : str   エラーメッセージ（正常時は空文字）
    """
    items = CATEGORY_SCORE_ITEMS.get(category, [])
    # 価格連動項目のID（残存価値率から機械変換するので除外）
    price_driven_ids = {"market_price", "resale_market", "market_liquidity"}
    non_price_items = [it for it in items if it["id"] not in price_driven_ids]

    items_description = "\n".join(
        f'  "{it["id"]}": {it["label"]}（{it["help"]}）'
        for it in non_price_items
    )

    acost_man_yen = acost_here // 100  # 千円 → 万円
    prompt = f"""あなたはリース会社の物件評価専門家です。以下の物件について中古市場価格をウェブ検索して調査し、結果をJSONのみで返してください。

物件: {asset_name if asset_name.strip() else f"{category}（型番不明）"}
リース契約期間: {lease_term_here}ヶ月後の推定中古価格を調査
取得価格: {acost_man_yen}万円

以下のJSONフォーマットのみで回答してください（マークダウンのコードブロックなし）：
{{
  "current_price_man_yen": <現在の中古相場（万円、整数）。不明なら0>,
  "estimated_price_man_yen": <{lease_term_here}ヶ月後の推定中古価格（万円、整数）。不明なら0>,
  "rationale": "<調査根拠・出典（100字以内）>",
  "scores": {{
{chr(10).join(f'    "{it["id"]}": <{it["label"]}の評価（0-100の整数）>' for it in non_price_items)}
  }}
}}

注意:
- estimated_price_man_yen は {lease_term_here}ヶ月（{lease_term_here//12}年{lease_term_here%12}ヶ月）後の中古市場価格を推定すること
- 検索結果が見つからない場合はカテゴリの一般的な相場を使用すること
- residual_pct は計算しないでください（システム側で計算します）"""

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
    }

    empty_result = {
        "current_price": 0,
        "estimated_price": 0,
        "residual_pct": 0.0,
        "rationale": "",
        "scores": {},
        "error": "",
    }

    try:
        resp = requests.post(
            f"{_GEMINI_URL}?key={gemini_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        # JSON 抽出（コードブロックに包まれている場合も対応）
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        if not json_match:
            empty_result["error"] = f"JSONが見つかりません: {raw_text[:200]}"
            return empty_result

        data = json.loads(json_match.group())
        current_price = int(data.get("current_price_man_yen", 0) or 0)
        estimated_price = int(data.get("estimated_price_man_yen", 0) or 0)

        # 残存価値率を計算
        residual_pct = 0.0
        if acost_man_yen > 0 and estimated_price > 0:
            residual_pct = min(100.0, (estimated_price / acost_man_yen) * 100)
        elif current_price > 0 and estimated_price > 0:
            # 取得価格不明時は現在相場を基準にする（精度は下がる）
            residual_pct = min(100.0, (estimated_price / current_price) * 100)

        return {
            "current_price": current_price,
            "estimated_price": estimated_price,
            "residual_pct": round(residual_pct, 1),
            "rationale": str(data.get("rationale", ""))[:200],
            "scores": {k: max(0, min(100, int(v or 50))) for k, v in data.get("scores", {}).items()},
            "error": "",
        }

    except json.JSONDecodeError as e:
        empty_result["error"] = f"JSONパースエラー: {e}"
        return empty_result
    except Exception as e:
        empty_result["error"] = f"調査エラー: {e}"
        return empty_result


# ── asset_scorer.py 補正条件の逆用: 事前警告 ────────────────────────────────
def _get_adjustment_warnings(
    category: str,
    scores: dict,
    lease_term_here: int,
) -> list[str]:
    """
    asset_scorer._adjust_weights() が発動する条件を事前に検出し、警告テキストを返す。
    スコア入力時にリアルタイムで表示することで、担当者が入力を再考できる。
    """
    warnings = []

    if category == "産業機械":
        custom = scores.get("customization_level", 50)
        resale = scores.get("resale_market", 50)
        if custom < 40 and resale > 60:
            warnings.append(
                "⚠️ **カスタマイズ度が低い（高カスタマイズ品）** にもかかわらず再販市場スコアが高めです。"
                " 実際の審査計算では再販市場スコアが50%に補正されます（asset_scorer.py の補正ロジック）。"
            )

    elif category == "IT機器":
        obsolescence = scores.get("tech_obsolescence", 50)
        if obsolescence > 80 and lease_term_here > 48:
            warnings.append(
                "⚠️ **技術陳腐化リスクが低い評価** ですが、リース期間が長め（48ヶ月超）です。"
                " 陳腐化系項目の重みが1.3倍に引き上げられる補正が発動します。"
            )

    elif category == "車両":
        ev_risk = scores.get("ev_tech_risk", 50)
        if ev_risk < 50 and lease_term_here > 48:
            warnings.append(
                "⚠️ **EV技術変化リスクあり** かつリース期間48ヶ月超です。"
                " EV技術リスク項目の重みが1.5倍に引き上げられる補正が発動します（バッテリー残価リスク反映）。"
            )

    return warnings


# ── 加重平均スコアを計算してセッションに保存 ────────────────────────────────
def _calc_and_store_score(category: str, scores: dict) -> float:
    """category の CATEGORY_SCORE_ITEMS 重みでスコアを加重平均し asd_detail_score に保存。"""
    items = CATEGORY_SCORE_ITEMS.get(category, [])
    if not items:
        return 50.0
    total_w = sum(it["weight"] for it in items)
    if total_w == 0:
        return 50.0
    weighted = sum(scores.get(it["id"], 50) * it["weight"] for it in items)
    result = round(weighted / total_w, 1)
    st.session_state["asd_detail_score"] = result
    return result


# ── メインUI ─────────────────────────────────────────────────────────────────
def render_asset_score_detail(
    asset_category: str,
    selected_asset_id: str,
    asset_name: str,
) -> None:
    """
    物件スコア詳細評価エキスパンダーを描画する。
    フォームの外側（st.form("shinsa_form") の前）に配置すること。
    """
    if not asset_category:
        return

    items = CATEGORY_SCORE_ITEMS.get(asset_category, [])
    if not items:
        return

    # APIキー確認
    gemini_key = (
        st.session_state.get("gemini_api_key", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )

    with st.expander("🔬 物件スコア詳細評価（任意）", expanded=False):
        st.caption(
            "物件の市場価値をAIがネット調査し、残存価値率からスコアを算出します。"
            "「この詳細スコアを判定に使用する」をオンにすると固定スコアを置き換えます。"
        )

        # ── 入力行: 機種名 ─────────────────────────────────────────────────
        model_name = st.text_input(
            "機種名・型番（任意）",
            value=st.session_state.get("asd_model_name", ""),
            placeholder="例: ハイエース バン 2022年式 / HP ProBook 450 G9",
            key="asd_model_name",
        )
        search_asset_name = f"{asset_name} {model_name}".strip() if model_name else asset_name

        # ── 入力行: 契約期間 + 取得価格 ────────────────────────────────────
        col_lt, col_ac = st.columns(2)
        with col_lt:
            # フォーム送信済みなら実値を初期値にセット
            _lt_init = st.session_state.get("lease_term", 36)
            lease_term_here = st.number_input(
                "契約期間（ヶ月）",
                min_value=1, max_value=120,
                value=int(_lt_init),
                step=1,
                key="asd_lease_term",
            )
        with col_ac:
            _ac_init = int(st.session_state.get("acquisition_cost", 0) or 0)
            acost_here = st.number_input(
                "取得価格（千円）",
                min_value=0, max_value=90_000_000,
                value=_ac_init,
                step=100,
                key="asd_acost",
            )

        # ── AI調査ボタン ───────────────────────────────────────────────────
        if gemini_key:
            if st.button("🌐 AIでネット調査してスコアを算出", key="asd_search_btn"):
                with st.spinner("Gemini がネット検索中..."):
                    result = _search_scores(
                        asset_category, search_asset_name,
                        lease_term_here, acost_here, gemini_key,
                    )
                if result["error"]:
                    st.error(f"調査失敗: {result['error']}")
                else:
                    st.session_state["asd_current_price"]    = result["current_price"]
                    st.session_state["asd_estimated_price"]  = result["estimated_price"]
                    st.session_state["asd_residual"]         = result["residual_pct"]
                    st.session_state["asd_search_rationale"] = result["rationale"]
                    # 価格連動項目のスコアをセット
                    price_driven = {"market_price", "resale_market", "market_liquidity"}
                    residual_score = _residual_to_score(result["residual_pct"])
                    existing = dict(st.session_state.get("asd_scores", {}))
                    for item in items:
                        if item["id"] in price_driven:
                            existing[item["id"]] = residual_score
                        elif item["id"] in result["scores"]:
                            existing[item["id"]] = result["scores"][item["id"]]
                    st.session_state["asd_scores"] = existing
                    st.rerun()
        else:
            st.caption("💡 Geminiキーをサイドバーで設定するとAI調査機能が使えます")

        # ── 調査結果表示 ───────────────────────────────────────────────────
        _residual = st.session_state.get("asd_residual", 0)
        if _residual:
            c1, c2, c3 = st.columns(3)
            c1.metric("現在中古相場", f"{st.session_state.get('asd_current_price', 0):,}万円")
            c2.metric(
                f"満了時推定（{lease_term_here}ヶ月後）",
                f"{st.session_state.get('asd_estimated_price', 0):,}万円",
            )
            c3.metric("残存価値率", f"{_residual:.1f}%")
            _rat = st.session_state.get("asd_search_rationale", "")
            if _rat:
                st.caption(f"📋 調査根拠: {_rat}")

        # ── カテゴリ別スライダー ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("**各評価項目（スライダーで調整可）**")
        current_scores = dict(st.session_state.get("asd_scores", {}))
        new_scores = {}
        for item in items:
            default_val = current_scores.get(item["id"], 50)
            val = st.slider(
                f"{item['label']} （重み {item['weight']}%）",
                min_value=0, max_value=100,
                value=int(default_val),
                key=f"asd_slider_{item['id']}",
                help=item.get("help", ""),
            )
            new_scores[item["id"]] = val

        # スライダー変更をセッションに反映
        st.session_state["asd_scores"] = new_scores

        # ── 計算プレビュー ─────────────────────────────────────────────────
        detail_score = _calc_and_store_score(asset_category, new_scores)
        grade_info = next(
            (g for g in SCORE_GRADES if detail_score >= g["min"]),
            SCORE_GRADES[-1],
        )
        st.markdown(
            f"**計算プレビュー: {detail_score:.1f}点 → "
            f"<span style='color:{grade_info['color']}'>グレード{grade_info['label']} "
            f"（{grade_info['text']}）</span>**",
            unsafe_allow_html=True,
        )

        # ── ルールベース警告 ───────────────────────────────────────────────
        warnings = _get_adjustment_warnings(asset_category, new_scores, lease_term_here)
        for w in warnings:
            st.warning(w)

        # ── 判定への適用チェックボックス ───────────────────────────────────
        st.markdown("---")
        use_detail = st.checkbox(
            "✅ この詳細スコアを審査判定に使用する",
            value=st.session_state.get("asd_use_detail", False),
            key="asd_use_detail",
            help="オンにすると固定スコアを詳細評価スコアで置き換えます。残存価値率が取得済みの場合は担保ウェイトも動的に算出されます。",
        )
        if use_detail:
            st.success(
                f"判定スコア: **{detail_score:.1f}点**"
                + (f"（残存価値率 {_residual:.1f}% から担保ウェイトも動的算出）" if _residual else "")
            )


# ── 他AIコンポーネント向けコンテキスト生成 ──────────────────────────────────
def get_asset_context_for_ai() -> str:
    """
    asd_* セッションステートから物件市場データのコンテキスト文字列を生成する。
    軍師AI・AIチャット・エージェントチームなどが呼び出すだけで使える。

    Returns
    -------
    str  AI調査済みの場合は市場データ文字列、未調査の場合は空文字
    """
    residual = st.session_state.get("asd_residual", 0)
    if not residual:
        return ""
    current   = st.session_state.get("asd_current_price", 0)
    estimated = st.session_state.get("asd_estimated_price", 0)
    rationale = st.session_state.get("asd_search_rationale", "")
    detail    = st.session_state.get("asd_detail_score")
    lease_term = st.session_state.get("asd_lease_term", st.session_state.get("lease_term", "-"))

    lines = [
        "【物件市場データ（AI調査済み）】",
        f"現在中古相場: {current:,}万円 /"
        f" 満了時推定（{lease_term}ヶ月後）: {estimated:,}万円 /"
        f" 残存価値率: {residual:.1f}%",
    ]
    if rationale:
        lines.append(f"調査根拠: {rationale}")
    if detail is not None:
        lines.append(f"詳細評価スコア（加重平均）: {detail:.1f}点")
    return "\n".join(lines)
