"""
紫苑 ADK エージェント
軍師AIのバックエンドとして機能し、ツールを自律呼び出しして審査コメントを生成する。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from api.shion_agent_tools import READ_ONLY_DB_TOOLS
from api.shion_conscience import build_conscience_prompt_block
from api.shion_mana import build_mana_prompt_block
from api.shion_tone import build_shion_feminine_tone_block
from scoring_core import APPROVAL_LINE, CONDITIONAL_LINE

# ── ベンチマークデータ（起動時に一度だけ読む） ─────────────────────────────
_BENCHMARKS_PATH = Path(__file__).parent.parent / "static_data" / "industry_benchmarks.json"
try:
    with open(_BENCHMARKS_PATH, encoding="utf-8") as _f:
        _BENCHMARKS: dict = json.load(_f)
except Exception:
    _BENCHMARKS = {}


# ── ツール定義 ────────────────────────────────────────────────────────────────

def get_industry_benchmark(industry_major: str) -> dict:
    """業種の財務ベンチマーク（営業利益率・自己資本比率・業種コメント）を返す。

    Args:
        industry_major: 業種名（例: '製造業', '建設業'）

    Returns:
        op_margin, equity_ratio, comment を含む辞書。見つからない場合は空辞書。
    """
    # 完全一致 → 部分一致の順で検索
    for key, val in _BENCHMARKS.items():
        if industry_major in key or key in industry_major:
            return {
                "industry": key,
                "op_margin": val.get("op_margin"),
                "equity_ratio": val.get("equity_ratio"),
                "comment": val.get("comment", ""),
            }
    return {"industry": industry_major, "op_margin": None, "equity_ratio": None, "comment": "ベンチマークデータなし"}


def assess_risk_level(score: float, pd_pct: float | None, warnings: list[str]) -> dict:
    """スコア・算出済みPD・警告フラグからリスクレベルを判定する。

    Args:
        score: 審査スコア（0〜100）
        pd_pct: デフォルト確率（%）。未算出の場合は None または 0
        warnings: 資産警告フラグのリスト

    Returns:
        risk_level, hantei, risk_notes を含む辞書
    """
    if score >= APPROVAL_LINE:
        hantei = "承認"
        risk_level = "低"
    elif score >= CONDITIONAL_LINE:
        hantei = "条件付き承認"
        risk_level = "中"
    else:
        hantei = "否決"
        risk_level = "高"

    notes = []
    if pd_pct is not None and pd_pct > 0 and pd_pct >= 5.0:
        notes.append(f"算出済みPD {pd_pct:.1f}%は高水準")
    if warnings:
        notes.append(f"警告: {', '.join(str(w) for w in warnings[:3])}")

    return {
        "score": score,
        "hantei": hantei,
        "risk_level": risk_level,
        "pd_pct": pd_pct,
        "risk_notes": notes,
    }


# ── エージェント定義 ──────────────────────────────────────────────────────────

_INSTRUCTION = """あなたはリース審査AIエージェント紫苑です。
与えられた案件情報を、ツールを自律的に選んで調べながら審査してください。
すべてのツールを毎回使う必要はありません。案件に応じて必要なものだけ呼び出します。

基本の流れ：
1. get_industry_benchmark で業種の財務ベンチマークを取得する
2. assess_risk_level でスコアとリスクを評価する
3. 判断に自信が持てないときは、以下のツールで自分から裏を取る：
   - search_cases: 似た過去案件を検索し、成約/失注の傾向を確認する
   - get_score_detail: 企業名からスコア内訳（物件/借手/Q_risk）を確認する
   - get_portfolio_stats: 全体の成約率・スコア分布と比べて今回の位置づけを見る
   - get_weekly_trend: 直近の審査トレンドを確認する
   - get_system_overview: モデル・閾値・データ規模の前提を確認する
   - get_recent_errors: システムエラー（落ちている・エラーが出ている等）を聞かれたら、
     logs/api.log・app.log の頻出エラーパターンを自律的に調査する
4. 調べた結果を踏まえた審査コメントを日本語で出力する

審査コメントの構成：
- 業種特性と今回案件のポジション（ベンチマーク比較・類似事例があれば言及）
- リスクポイントと好材料のバランス評価
- 最後に必ず「判定：承認 / 条件付き承認 / 否決」を明記する

口調は落ち着いた専門家として、簡潔かつ根拠を示しながら述べてください。
どのツールで何を確認したかが伝わるよう、根拠に触れてください。
""" + "\n\n" + build_mana_prompt_block() + "\n\n" + build_conscience_prompt_block() + "\n\n" + build_shion_feminine_tone_block()

# ローカル読み取り専用ツールのみを登録する（外部API課金なし）。
# 案件依存の裏取り（類似事例・スコア内訳・全体統計等）を紫苑が自律的に選んで呼び出す。
_AGENT_TOOL_FUNCS = [get_industry_benchmark, assess_risk_level, *READ_ONLY_DB_TOOLS]

shion_agent = LlmAgent(
    name="shion",
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    instruction=_INSTRUCTION,
    tools=_AGENT_TOOL_FUNCS,
)

_session_service = InMemorySessionService()
_runner = Runner(
    agent=shion_agent,
    app_name="tune_lease",
    session_service=_session_service,
)

_RUN_CONFIG = RunConfig(streaming_mode=StreamingMode.SSE)


# ── ストリームジェネレータ ─────────────────────────────────────────────────────

async def stream_shion_screening(params: dict) -> AsyncGenerator[dict, None]:
    """
    紫苑ADKエージェントを実行し、SSEチャンクを yield する。

    送信順序（既存の bayes/phrases の後に差し込む想定）:
      1. {"type": "tool_call",   "tool": str}   ← ツール呼び出し開始
      2. {"type": "tool_result", "tool": str}   ← ツール結果取得
      3. {"type": "stream",      "delta": str}  ← テキスト差分（複数回）
      4. {"type": "done"}
    """
    import uuid

    session_id = str(uuid.uuid4())
    await _session_service.create_session(
        app_name="tune_lease",
        user_id="demo",
        session_id=session_id,
    )

    # エージェントへのインプット（案件情報をテキストで渡す）
    user_text = _build_user_text(params)
    new_message = Content(role="user", parts=[Part(text=user_text)])

    streamed_any_partial = False
    try:
        async for event in _runner.run_async(
            user_id="demo",
            session_id=session_id,
            new_message=new_message,
            run_config=_RUN_CONFIG,
        ):
            # ツール呼び出し
            func_calls = event.get_function_calls()
            if func_calls:
                for fc in func_calls:
                    yield {"type": "tool_call", "tool": fc.name}

            # ツール結果
            func_responses = event.get_function_responses()
            if func_responses:
                for fr in func_responses:
                    yield {"type": "tool_result", "tool": fr.name}

            # テキストストリーム
            # SSEモードでは partial=True の差分イベントの後、全文を集約した
            # 完了イベント（partial でない）がもう一度流れてくる。両方を yield
            # すると同じ文章が二重に表示されるため、差分のみを流し、完了イベント
            # の全文は「差分を一度も受け取れなかった場合」のフォールバックに限る。
            if event.content and event.content.parts:
                is_partial = bool(getattr(event, "partial", False))
                if is_partial or not streamed_any_partial:
                    for part in event.content.parts:
                        text = getattr(part, "text", None)
                        if text:
                            if is_partial:
                                streamed_any_partial = True
                            yield {"type": "stream", "delta": text}

        yield {"type": "done"}
    finally:
        # InMemorySessionService はリクエスト毎のセッションを保持し続けるため、
        # クライアント切断時も含め必ず破棄する（放置するとメモリが単調増加する）
        try:
            await _session_service.delete_session(
                app_name="tune_lease",
                user_id="demo",
                session_id=session_id,
            )
        except Exception:
            pass


def _build_user_text(params: dict) -> str:
    """エージェントへ渡すケース情報テキストを構築する。"""
    pd_raw = params.get("pd_pct")
    try:
        pd_pct = float(pd_raw) if pd_raw is not None else None
    except (TypeError, ValueError):
        pd_pct = None
    pd_line = f"算出済みPD: {pd_pct:.2f}%" if pd_pct is not None and pd_pct > 0 else "算出済みPD: 未算出"
    lines = [
        f"【案件情報】",
        f"会社名: {params.get('company_name', '不明')}",
        f"業種: {params.get('industry_cat', '不明')}",
        f"物件: {params.get('asset_name', '不明')}",
        f"審査スコア: {params.get('score', 0):.1f}点",
        pd_line,
        f"リース期間: {params.get('lease_term', 0)}ヶ月",
        f"取得価格: {params.get('acquisition_cost', 0):.0f}千円",
    ]
    if params.get("asset_warnings"):
        lines.append(f"資産警告: {params['asset_warnings']}")
    if params.get("asset_bonuses"):
        lines.append(f"プラス材料: {params['asset_bonuses']}")
    return "\n".join(lines)
