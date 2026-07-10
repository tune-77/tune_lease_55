"""
📋 事業計画チェック（簡易版）— 提出された事業計画の妥当性を検証する

2つの入口を提供する:
  1. 専用ページ（/business-plan-check → POST /api/business-plan/validate）
  2. チャット統合（紫苑が対話の中で相談に応じる。build_business_plan_chat_block）

審査担当者向け: 直近実績と計画値（売上・営業利益率）、リース条件から
機械的な整合性チェックを行い、Gemini が利用可能なら講評と
顧客への確認質問を付ける（LLM 不通でも機械チェックだけで応答する）。

出典（Cite the Source）:
  - static_data/required_documents.json: 要審議案件の追加書類は
    「事業計画書または売上見通し」
  - data/pdca_ai_rules.json: 「年商1億円以下の企業については、
    事業計画の実現可能性を詳細に検討すること」
  - knowledge_base/lease_review_knowledge/02_決算書分析と財務指標.md:
    新規事業は事業計画書を確認する
  - FAQ REV-040（frontend/src/app/faq/page.tsx）: 事業計画の合理性は
    売上・費用の根拠で判断し、「楽観シナリオのみ」は評価を下げる

注意: 数値閾値（成長率・利益率ジャンプ・リース料負担率）は Vault に
出典のない簡易版の暫定基準（推測）。環境変数で調整できるようにしてある。
"""
from __future__ import annotations

import json
import os
import re

import requests

# ── 閾値（簡易版の暫定基準。出典なしのため環境変数で調整可能） ─────────────────
_GROWTH_WATCH_PCT = float(os.environ.get("BPLAN_GROWTH_WATCH_PCT", "30"))
_GROWTH_WARN_PCT = float(os.environ.get("BPLAN_GROWTH_WARN_PCT", "50"))
_MARGIN_JUMP_WATCH_PT = float(os.environ.get("BPLAN_MARGIN_JUMP_WATCH_PT", "3"))
_MARGIN_JUMP_WARN_PT = float(os.environ.get("BPLAN_MARGIN_JUMP_WARN_PT", "5"))
_LEASE_BURDEN_WATCH_PCT = float(os.environ.get("BPLAN_LEASE_BURDEN_WATCH_PCT", "50"))

# data/pdca_ai_rules.json: 年商1億円以下は事業計画の実現可能性を詳細に検討
_SMALL_COMPANY_NENSHU_MM = 100.0  # 百万円

_LEVEL_ORDER = {"ok": 0, "info": 1, "watch": 2, "warning": 3}

_THRESHOLDS_NOTE = (
    "成長率・利益率・リース料負担率の閾値は簡易版の暫定基準です"
    "（審査基準としての出典はありません）。最終判断は担当者が行ってください。"
)


def _check(code: str, level: str, title: str, message: str) -> dict:
    return {"code": code, "level": level, "title": title, "message": message}


def validate_business_plan(params: dict) -> dict:
    """事業計画の簡易検証を実行する。

    Args:
        params: {
            industry_major, company_name, plan_basis,
            nenshu, op_margin_pct,             # 直近実績（売上=百万円）
            plan_nenshu, plan_op_margin_pct,   # 計画値
            lease_amount（百万円）, lease_months（回）,
            has_conservative_scenario（保守シナリオの提示有無）,
        }

    Returns:
        {"verdict", "summary_level", "checks": [...], "metrics": {...},
         "thresholds_note", "ai_review"?: {...}, "ai_available": bool}
    """
    nenshu = float(params.get("nenshu") or 0)
    op_margin = float(params.get("op_margin_pct") or 0)
    plan_nenshu = float(params.get("plan_nenshu") or 0)
    plan_op_margin = float(params.get("plan_op_margin_pct") or 0)
    lease_amount = float(params.get("lease_amount") or 0)
    lease_months = float(params.get("lease_months") or 0)
    has_conservative = bool(params.get("has_conservative_scenario"))

    checks: list[dict] = []
    metrics: dict = {}

    # ── 1. 売上成長率 ─────────────────────────────────────────────────────
    if nenshu <= 0:
        # knowledge_base/lease_review_knowledge/02_決算書分析と財務指標.md:
        # 新規事業は事業計画書で判断する
        checks.append(_check(
            "no_actual_revenue", "watch", "実績売上なし（新規事業の可能性）",
            "直近実績が未入力です。新規事業の場合は事業計画書の売上根拠・"
            "自己資金・保証の有無を重点的に確認してください。",
        ))
    elif plan_nenshu > 0:
        growth_pct = (plan_nenshu / nenshu - 1) * 100
        metrics["growth_pct"] = round(growth_pct, 1)
        if growth_pct > _GROWTH_WARN_PCT:
            checks.append(_check(
                "growth_rate", "warning", f"計画成長率 {growth_pct:.0f}%（楽観的な可能性）",
                f"計画売上が直近実績の {plan_nenshu / nenshu:.2f} 倍です。"
                "受注残・契約書など成長根拠の裏付け資料を確認してください。",
            ))
        elif growth_pct > _GROWTH_WATCH_PCT:
            checks.append(_check(
                "growth_rate", "watch", f"計画成長率 {growth_pct:.0f}%（要根拠確認）",
                "高めの成長計画です。売上根拠（受注見込み・販路）を確認してください。",
            ))
        else:
            checks.append(_check(
                "growth_rate", "ok", f"計画成長率 {growth_pct:.0f}%",
                "直近実績からの成長幅は現実的な範囲です。",
            ))

    # ── 2. 営業利益率のジャンプ ───────────────────────────────────────────
    if plan_op_margin and (nenshu > 0 or op_margin):
        margin_jump = plan_op_margin - op_margin
        metrics["margin_jump_pt"] = round(margin_jump, 1)
        if margin_jump > _MARGIN_JUMP_WARN_PT:
            checks.append(_check(
                "margin_jump", "warning", f"営業利益率が実績比 +{margin_jump:.1f}pt",
                "利益率の大幅改善計画です。費用計画（原価・人件費・販管費）の"
                "根拠を確認してください。FAQ REV-040 のとおり、売上・費用の"
                "根拠が示されない計画は評価を下げます。",
            ))
        elif margin_jump > _MARGIN_JUMP_WATCH_PT:
            checks.append(_check(
                "margin_jump", "watch", f"営業利益率が実績比 +{margin_jump:.1f}pt",
                "利益率改善の要因（値上げ・効率化・構成変化）を確認してください。",
            ))
        else:
            checks.append(_check(
                "margin_jump", "ok", f"営業利益率は実績比 {margin_jump:+.1f}pt",
                "利益率計画は実績と整合的です。",
            ))

    # ── 3. 年間リース料負担 vs 営業利益 ───────────────────────────────────
    if lease_amount > 0 and lease_months > 0:
        annual_lease = lease_amount / (lease_months / 12)
        metrics["annual_lease_mm"] = round(annual_lease, 1)
        plan_op_profit = plan_nenshu * plan_op_margin / 100
        actual_op_profit = nenshu * op_margin / 100
        metrics["plan_op_profit_mm"] = round(plan_op_profit, 1)
        metrics["actual_op_profit_mm"] = round(actual_op_profit, 1)

        if plan_op_profit <= 0:
            checks.append(_check(
                "lease_burden", "warning", "計画営業利益がゼロ以下",
                "計画上も営業利益が出ておらず、リース料の返済原資が"
                "確認できません。返済原資の説明を求めてください。",
            ))
        else:
            burden_plan_pct = annual_lease / plan_op_profit * 100
            metrics["lease_burden_plan_pct"] = round(burden_plan_pct, 1)
            if actual_op_profit < annual_lease <= plan_op_profit:
                checks.append(_check(
                    "lease_burden", "warning", "返済原資が計画利益に依存",
                    f"年間リース料 約{annual_lease:.1f}百万円は実績営業利益"
                    f"（{actual_op_profit:.1f}百万円）では賄えず、計画達成が"
                    "前提になります。計画未達時の返済原資（自己資金・他収益）を"
                    "確認してください。",
                ))
            elif burden_plan_pct > _LEASE_BURDEN_WATCH_PCT:
                checks.append(_check(
                    "lease_burden", "watch",
                    f"年間リース料が計画営業利益の {burden_plan_pct:.0f}%",
                    "計画が少し下振れすると返済負担が重くなります。"
                    "資金繰り・他の借入返済との合算負担を確認してください。",
                ))
            else:
                checks.append(_check(
                    "lease_burden", "ok",
                    f"年間リース料は計画営業利益の {burden_plan_pct:.0f}%",
                    "計画利益に対するリース料負担は許容範囲です。",
                ))
    elif lease_amount > 0:
        checks.append(_check(
            "lease_burden", "info", "リース期間未入力",
            "リース期間（回数）が未入力のため、年間リース料負担の"
            "チェックを省略しました。",
        ))

    # ── 4. 小規模企業の計画依存（data/pdca_ai_rules.json） ────────────────
    if 0 < nenshu <= _SMALL_COMPANY_NENSHU_MM and metrics.get("growth_pct", 0) > _GROWTH_WATCH_PCT / 2:
        checks.append(_check(
            "small_company", "watch", "年商1億円以下 × 成長計画",
            "年商1億円以下の企業は事業計画の実現可能性を詳細に検討する方針です"
            "（PDCA AIルール）。計画の前提条件を個別に確認してください。",
        ))

    # ── 5. 保守シナリオの有無（FAQ REV-040: 楽観シナリオのみは評価を下げる） ──
    if not has_conservative:
        checks.append(_check(
            "optimism_only", "watch", "保守シナリオ未提示",
            "楽観シナリオのみの計画は評価を下げる方針です（FAQ REV-040）。"
            "売上8割時の資金繰りなど、保守シナリオの提出を依頼してください。",
        ))
    else:
        checks.append(_check(
            "optimism_only", "ok", "保守シナリオあり",
            "保守シナリオが提示されています。前提の妥当性を確認してください。",
        ))

    summary_level = "ok"
    for c in checks:
        if _LEVEL_ORDER.get(c["level"], 0) > _LEVEL_ORDER.get(summary_level, 0):
            summary_level = c["level"]
    verdict = {
        "ok": "概ね現実的な範囲",
        "info": "概ね現実的な範囲（一部未確認）",
        "watch": "要確認（根拠資料の確認を推奨）",
        "warning": "楽観的な可能性（根拠の裏付けが必要）",
    }[summary_level]

    result: dict = {
        "verdict": verdict,
        "summary_level": summary_level,
        "checks": checks,
        "metrics": metrics,
        "thresholds_note": _THRESHOLDS_NOTE,
        "ai_available": False,
    }

    ai_review = _ai_review(params, checks, verdict)
    if ai_review:
        result["ai_review"] = ai_review
        result["ai_available"] = True
    return result


# ── Gemini 講評（任意。不通でも機械チェックだけで応答する） ─────────────────────

def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


_AI_SYS = (
    "あなたはリース審査の事業計画検証アシスタントです。"
    "売上・費用の根拠が示されない計画や楽観シナリオのみの計画は評価を下げる方針です。"
    "必ず有効なJSONのみで回答してください。"
)


def _ai_review(params: dict, checks: list[dict], verdict: str) -> dict | None:
    try:
        from api.secret_access import get_gemini_api_key
        api_key = get_gemini_api_key()
    except Exception:
        return None
    if not api_key:
        return None

    check_lines = "\n".join(f"- [{c['level']}] {c['title']}: {c['message']}" for c in checks)
    prompt = f"""## 審査案件
- 企業名: {params.get('company_name') or '（未設定）'}
- 業種: {params.get('industry_major') or '未設定'}
- 直近売上高: {params.get('nenshu') or 0}百万円 / 営業利益率: {params.get('op_margin_pct') or 0}%
- 計画売上高: {params.get('plan_nenshu') or 0}百万円 / 計画営業利益率: {params.get('plan_op_margin_pct') or 0}%
- リース金額: {params.get('lease_amount') or 0}百万円 / 期間: {params.get('lease_months') or 0}回
- 計画の根拠（担当者メモ）: {params.get('plan_basis') or '（記載なし）'}

## 機械チェック結果（暫定判定: {verdict}）
{check_lines}

上記の事業計画を審査担当者の視点で講評し、短いJSONのみで回答せよ。
{{
  "verdict": "現実的" | "やや楽観" | "過大",
  "comments": ["30字以内の講評を最大3個"],
  "questions": ["顧客への確認質問を最大3個"]
}}"""

    payload = {
        "system_instruction": {"parts": [{"text": _AI_SYS}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }
    try:
        resp = requests.post(
            _gemini_url(),
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if not raw.startswith("{"):
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                raw = m.group()
        data = json.loads(raw)
        return {
            "verdict": str(data.get("verdict") or ""),
            "comments": [str(c) for c in (data.get("comments") or [])[:3]],
            "questions": [str(q) for q in (data.get("questions") or [])[:3]],
        }
    except Exception:
        return None
# ── チャット統合: 事業計画相談モード ─────────────────────────────────────────
# 「ラーメン屋をやりたい」のような相談をチャットで受けたとき、紫苑が対話の中で
# 前提を質問し概算を提示するためのシステムプロンプト注入ブロック。
# /api/chat の intent 分岐（改善/通常/軍師AI）には触れず、追加ブロックとしてのみ作用する。

_STRONG_HINTS = (
    "開業", "起業", "創業", "出店", "開店", "新規事業", "新事業",
    "事業計画", "事業を始", "独立し", "脱サラ",
)
_DESIRE_HINTS = ("やりたい", "始めたい", "はじめたい", "やってみたい", "開きたい", "出したい", "持ちたい")
_BUSINESS_NOUNS = (
    "屋", "店", "事業", "ビジネス", "サロン", "工房", "工場", "クリニック",
    "教室", "カフェ", "レストラン", "ジム", "美容室", "医院", "薬局",
)


def is_business_plan_consult(message: str) -> bool:
    """メッセージが開業・新規事業計画の相談らしいかを判定する。"""
    msg = (message or "").strip()
    if not msg:
        return False
    if any(h in msg for h in _STRONG_HINTS):
        return True
    return any(d in msg for d in _DESIRE_HINTS) and any(n in msg for n in _BUSINESS_NOUNS)


def build_business_plan_chat_block(message: str) -> str:
    """事業計画相談モードのプロンプトブロックを返す。相談でなければ空文字。"""
    if not is_business_plan_consult(message):
        return ""
    return f"""

【事業計画相談モード】
ユーザーは開業・新規事業の相談をしている可能性が高い（無関係な話題ならこのモードは無視してよい）。
リース審査の知見を持つ相談相手として、チャットの対話の中で事業計画づくりを支援せよ。

進め方:
1. 概算に必要な前提を質問する。一度に最大3つまでにして対話で埋めていく。
   例: 業態と立地 / 規模（席数・坪数など）/ 想定客単価・回転数・営業日数 /
   初期投資（設備・内装・保証金）/ 自己資金と借入予定
2. 前提が揃ったら、チャット内で概算を段階的に提示する:
   - 月商の概算（例: 飲食なら 客単価×席数×回転数×営業日数。使った計算式を必ず明示）
   - 営業利益の概算（業態の目安利益率を使う場合は「仮置き」と明言する）
   - 設備をリースにした場合の月額の目安と、年間リース料÷営業利益の負担率。
     負担率が{int(_LEASE_BURDEN_WATCH_PCT)}%を超えるなら「計画が下振れすると返済が重くなる」と伝える
3. 楽観シナリオだけでなく、売上8割の保守シナリオを必ず併記する
   （楽観シナリオのみの事業計画は審査上評価を下げる方針）。
4. 数字はすべて概算・仮置きであることを明示し、正式な審査には決算書または
   事業計画書（売上・費用の根拠つき）が必要であることを案内する。
5. 承認・否決を断定・約束せず、次に集めるべき資料・数字を具体的に示して締める。"""
