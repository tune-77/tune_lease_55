"""
⚔️ マルチエージェント審査 — 石橋（慎重派）vs 風林火山（積極派）+ 軍師（調停）

スコア60超 or 40未満 → 軍師単独高速処理
スコア40〜60（境界） → 2エージェント並列討論（2ラウンド）→ 軍師裁定

なれ合い防止策:
  - Temperature差（石橋=0.3、風林火山=0.9）
  - 強制反論ラウンド（相手の主論点に必ず反論）
  - 意見乖離度チェック（同一意見なら逆張り指示を追加）
"""
from __future__ import annotations

import json
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor

from api.context.context_bundle import build_context_bundle

# ── モデル・エンドポイント定数 ───────────────────────────────────────────────────
# 石橋・風林火山: Gemini Flash（軽量・高速、temperature差で個性を分離）
# 軍師: Gemini Flash（同モデルでも上位プロンプト + temperature=0.3 で裁定役）
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

_DEBATE_LOW  = 40  # これ以下 → 否決ファストパス
_DEBATE_HIGH = 60  # これ以上 → 承認ファストパス

# ── ケースコンテキストテンプレート ──────────────────────────────────────────────
_CASE_CTX_TMPL = """## 審査案件
- 企業名: {company_name}
- 業種: {industry}
- 審査スコア: {score}点
- 売上高: {revenue}百万円
- 営業利益率: {op_margin}%
- 自己資本比率: {equity_ratio}%
- 銀行借入残高: {bank_credit}百万円
- リース借入残高: {lease_credit}百万円
- リース物件名: {asset_name}
- リース金額: {lease_amount}百万円""".strip()

# ── システムプロンプト ──────────────────────────────────────────────────────────
_CAUTIOUS_SYS = """あなたは「石橋」— リース審査の慎重派エージェントです。
損失回避を最優先し、リスクを徹底的に洗い出してください。
「渡れるかではなく、渡れなかった場合の損害を考えよ」が信条。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_AGGRESSIVE_SYS = """あなたは「風林火山」— リース審査の積極派エージェントです。
成長性・機会を重視し、機会損失も損失と捉えます。
「攻めずして勝機なし。慎重すぎる判断は機会を殺す」が信条。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""

_ARBITER_SYS = """あなたは「軍師」— リース審査の最終裁定者です。
慎重派・積極派双方の論拠を公平かつ鋭く評価し、最終判断を下してください。
条件付き承認の場合は実務的で具体的な条件を列挙してください。
必ず有効なJSONのみで回答してください。説明文や前置きは不要です。"""


def _build_case_ctx(params: dict) -> str:
    return _CASE_CTX_TMPL.format(
        company_name=params.get("company_name", "（未設定）"),
        industry=params.get("industry_major") or params.get("industry_sub") or "未設定",
        score=params.get("score", 0),
        revenue=params.get("nenshu", 0),
        op_margin=params.get("op_margin_pct") or params.get("op_profit_pct") or 0,
        equity_ratio=params.get("equity_ratio") or params.get("equity_pct") or 0,
        bank_credit=params.get("bank_credit", 0),
        lease_credit=params.get("lease_credit", 0),
        asset_name=params.get("asset_name", ""),
        lease_amount=params.get("lease_amount") or params.get("lease_total") or 0,
    )


def _get_gemini_api_key() -> str:
    """Gemini APIキーを取得する。
    優先順位: 環境変数 → secrets.toml（直接パース、Streamlit非依存）
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    # secrets.toml を直接パース（Streamlit非依存）
    # worktree / 本体リポジトリ の両方を探索する
    import re as _re
    _here = os.path.dirname(os.path.abspath(__file__))
    _candidates = []
    # worktreeルート → 本体リポジトリ方向へ最大4階層まで探す
    cur = os.path.dirname(_here)
    for _ in range(5):
        _candidates.append(os.path.join(cur, ".streamlit", "secrets.toml"))
        cur = os.path.dirname(cur)
    for sec_path in _candidates:
        if not os.path.exists(sec_path):
            continue
        try:
            with open(sec_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = _re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                    if m:
                        return m.group(1)
        except Exception:
            pass
    return ""


def _llm_call(system: str, prompt: str, temperature: float, max_tokens: int = 1024) -> dict:
    """Gemini generateContent REST API を呼び出してJSONを返す。"""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません")

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }
    resp = requests.post(
        f"{_GEMINI_URL}?key={api_key}",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    # コードブロック除去
    if "```" in raw:
        parts = raw.split("```")
        for part in parts[1::2]:
            cleaned = part.lstrip("json\n").strip()
            if cleaned.startswith("{"):
                raw = cleaned
                break
    # JSON抽出（応答にゴミが混ざった場合の保険）
    if not raw.startswith("{"):
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group()
    return json.loads(raw)


# ── エージェント別プロンプトビルダー ────────────────────────────────────────────

def _cautious_prompt(ctx: str, counter_json: str = "", extra: str = "") -> str:
    base = f"""{ctx}

【石橋の立場：慎重派】この案件を審査し、以下のJSON形式のみで回答せよ。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reasons": ["主な判断理由（2〜3個）"],
  "key_risks": ["重大リスク（2〜3個）"]
}}"""
    if counter_json:
        base += f"\n\n【必須】積極派の以下の意見に具体的に反論すること（reasons に含めよ）:\n{counter_json}"
    if extra:
        base += f"\n\n{extra}"
    return base


def _aggressive_prompt(ctx: str, counter_json: str = "", extra: str = "") -> str:
    base = f"""{ctx}

【風林火山の立場：積極派】この案件を審査し、以下のJSON形式のみで回答せよ。

{{
  "opinion": "承認" | "否決" | "条件付承認",
  "reasons": ["主な判断理由（2〜3個）"],
  "opportunities": ["見逃せない機会・強み（2〜3個）"]
}}"""
    if counter_json:
        base += f"\n\n【必須】慎重派の以下の意見に具体的に反論すること（reasons に含めよ）:\n{counter_json}"
    if extra:
        base += f"\n\n{extra}"
    return base


def _arbiter_debate_prompt(ctx: str, log: str) -> str:
    return f"""{ctx}

【討論ログ】
{log}

上記の討論を踏まえ、軍師として最終裁定を以下のJSON形式で示せ。

{{
  "final": "承認" | "否決" | "条件付承認",
  "reasoning": "裁定の根拠（2〜3文）",
  "conditions": ["条件1", "条件2"]
}}

"final" が "承認" の場合、conditions は空リスト []。"""


def _arbiter_solo_prompt(ctx: str, direction: str) -> str:
    return f"""{ctx}

スコアからこの案件は明確に{direction}圏内にある。速やかに裁定せよ。

{{
  "final": "承認" | "否決" | "条件付承認",
  "reasoning": "裁定の根拠（1〜2文）",
  "conditions": []
}}"""


# ── ユーティリティ ───────────────────────────────────────────────────────────────

def _safe_future(future, fallback: dict) -> dict:
    try:
        return future.result(timeout=90)
    except Exception as e:
        return {**fallback, "_error": str(e)}


def _norm_cautious(d: dict) -> dict:
    return {
        "opinion": d.get("opinion", "否決"),
        "reasons": d.get("reasons", []),
        "key_risks": d.get("key_risks", []),
    }


def _norm_aggressive(d: dict) -> dict:
    return {
        "opinion": d.get("opinion", "条件付承認"),
        "reasons": d.get("reasons", []),
        "opportunities": d.get("opportunities", []),
    }


def _norm_arbiter(d: dict) -> dict:
    return {
        "final": d.get("final", "条件付承認"),
        "reasoning": d.get("reasoning", ""),
        "conditions": d.get("conditions", []),
    }


def _excerpt(d: dict, key: str, max_items: int = 2) -> str:
    items = d.get(key, [])
    return "; ".join(str(i) for i in items[:max_items]) if items else "（なし）"


# ── メインエントリポイント ───────────────────────────────────────────────────────

def run_debate_screening(params: dict) -> dict:
    """
    マルチエージェント審査を実行する。

    Args:
        params: スコアリングAPIと同形式のdict。"score" キー必須。

    Returns:
        {
            "score": float,
            "mode": "solo" | "debate",
            "cautious"?: {...},    # debate モードのみ
            "aggressive"?: {...},  # debate モードのみ
            "arbiter": {...},
            "debate_log"?: str,    # debate モードのみ
            "context_bundle"?: {...},
        }
    """
    score = float(params.get("score", 0))
    ctx = _build_case_ctx(params)

    # ── コンテキスト注入: 地域/季節/業況を全エージェントへ配布 ─────────────────
    try:
        bundle = build_context_bundle(
            prefecture=params.get("prefecture", ""),
            industry=(
                params.get("industry_major") or
                params.get("industry_sub") or ""
            ),
        )
        ctx_block = bundle.to_system_prompt_block()
    except Exception:
        ctx_block = ""
        bundle = None

    cautious_sys = _CAUTIOUS_SYS + ("\n\n" + ctx_block if ctx_block else "")
    aggressive_sys = _AGGRESSIVE_SYS + ("\n\n" + ctx_block if ctx_block else "")
    arbiter_sys = _ARBITER_SYS + ("\n\n" + ctx_block if ctx_block else "")

    # ── ファストパス（境界外スコア） ──────────────────────────────────────────
    if score >= _DEBATE_HIGH or score <= _DEBATE_LOW:
        direction = "承認" if score >= _DEBATE_HIGH else "否決"
        arbiter_raw = _llm_call(
            arbiter_sys,
            _arbiter_solo_prompt(ctx, direction),
            temperature=0.3, max_tokens=512,
        )
        result = {
            "score": score,
            "mode": "solo",
            "arbiter": _norm_arbiter(arbiter_raw),
        }
        if bundle is not None:
            result["context_bundle"] = bundle.model_dump()
        return result

    # ── 討論モード（40 < score < 60） ────────────────────────────────────────
    # Round 1: 並列実行（石橋 temperature=0.3、風林火山 temperature=0.9）
    with ThreadPoolExecutor(max_workers=2) as pool:
        fc = pool.submit(_llm_call, cautious_sys, _cautious_prompt(ctx), 0.3)
        fa = pool.submit(_llm_call, aggressive_sys, _aggressive_prompt(ctx), 0.9)
        r1c = _safe_future(fc, {"opinion": "否決", "reasons": [], "key_risks": []})
        r1a = _safe_future(fa, {"opinion": "条件付承認", "reasons": [], "opportunities": []})

    # 乖離度チェック: 同意見なら逆張り指示を追加
    same_opinion = r1c.get("opinion") == r1a.get("opinion")
    extra_c = "【警告】積極派と同じ意見になっている。慎重派として必ず異なる立場で主張せよ。" if same_opinion else ""
    extra_a = "【警告】慎重派と同じ意見になっている。積極派として必ず異なる立場で主張せよ。" if same_opinion else ""

    r1c_json = json.dumps(_norm_cautious(r1c), ensure_ascii=False)
    r1a_json = json.dumps(_norm_aggressive(r1a), ensure_ascii=False)

    # Round 2: 強制反論ラウンド（相手の意見を受けて必ず反論）
    with ThreadPoolExecutor(max_workers=2) as pool:
        fc2 = pool.submit(
            _llm_call, cautious_sys,
            _cautious_prompt(ctx, r1a_json, extra_c), 0.3
        )
        fa2 = pool.submit(
            _llm_call, aggressive_sys,
            _aggressive_prompt(ctx, r1c_json, extra_a), 0.9
        )
        r2c = _safe_future(fc2, r1c)
        r2a = _safe_future(fa2, r1a)

    # 討論ログ構築
    debate_log = (
        "【第1ラウンド：初期見解】\n"
        f"石橋（慎重）: {r1c.get('opinion', '？')} — {_excerpt(r1c, 'reasons')}\n"
        f"風林火山（積極）: {r1a.get('opinion', '？')} — {_excerpt(r1a, 'reasons')}\n"
        "\n【第2ラウンド：強制反論】\n"
        f"石橋（慎重）: {r2c.get('opinion', '？')} — {_excerpt(r2c, 'reasons')}\n"
        f"風林火山（積極）: {r2a.get('opinion', '？')} — {_excerpt(r2a, 'reasons')}"
    )
    if same_opinion:
        debate_log = "[注: 第1ラウンドで両者の意見が一致したため、逆張り再討論を実施]\n\n" + debate_log

    # 軍師裁定（temperature=0.3 で中立・冷静）
    arbiter_raw = _llm_call(
        arbiter_sys,
        _arbiter_debate_prompt(ctx, debate_log),
        temperature=0.3, max_tokens=1024,
    )

    result = {
        "score": score,
        "mode": "debate",
        "cautious": _norm_cautious(r2c),
        "aggressive": _norm_aggressive(r2a),
        "arbiter": _norm_arbiter(arbiter_raw),
        "debate_log": debate_log,
        "same_opinion_r1": same_opinion,
    }
    if bundle is not None:
        result["context_bundle"] = bundle.model_dump()
    return result
