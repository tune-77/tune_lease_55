"""
紫苑 ADK エージェント（本流）

api/routers/gunshi.py のストリーミング経路から呼ばれる、ツール呼び出し可能な
本番エージェント。継続的にツールが追加されており（api/shion_agent_tools.py の
READ_ONLY_DB_TOOLS、api/shion_vertex_tools.py の VERTEX_AGENT_TOOLS 等）、
google.adk はモジュール先頭で import する方針を採る。

フォールバック専用の別実装として api/shion_debate_adk.py（凍結・ツール無し・
多エージェント討論形式）が存在する。両者は設計方針が異なるため統合はしない。
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
from api.shion_vertex_tools import VERTEX_AGENT_TOOLS
from api.shion_mana import build_mana_prompt_block
from api.shion_prompt_priority import build_shion_prompt_priority_block
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
   - get_pipeline_item_details: 改善パイプラインの要確認・放置項目について聞かれたら、
     ledger_rules.json の個別項目を深掘りする
   - recall_judgment_memory: 過去の判断根拠を確認したいときは、正準ルールと紫苑の
     記憶索引の両方から関連する判断を想起する
   - build_judgment_preview: レビュー前の判断材料候補を確認したいときに使う
     （まだcanonical_judgment_rulesではない下書きである点に注意）
   - search_obsidian_context: Obsidian Vaultの知識ノートで裏取りしたいときに使う
   - review_obsidian_vault_health: Obsidianの中身を良くしたい、孤立ノート・リンク・
     検索性・ChromaDB/BM25への影響を棚卸ししたいときに使う。読み取り専用で、
     Vaultを書き換えない
   - suggest_obsidian_curation_actions: 「検索で出てこない」「チャットが拾えない」
     テーマがあるときに使い、関連リンク・検索語・候補ノートを少数だけ提案する。
     自動反映せず、人間承認後に既存スクリプトで反映する
   - structure_judgment_asset_candidate: 会話・案件メモ・User修正文を判断資産候補へ
     構造化したいときに使う。候補化だけで、記憶昇格やスコア変更はしない
   - validate_lease_source_summary: ニュース・制度・業界情報を審査コメントに使ってよいか、
     鮮度・信頼性・偏り・リース関連性を確認したいときに使う
   - convert_research_to_screening_insights: 調査メモやAuto Researchを、最大3つの
     審査確認点・コメント草案・判断資産候補へ変換したいときに使う
   - build_screening_decision_flow: 条件付き承認、否決、保留、追加確認の判断分岐を
     フロー化したいときに使う。自動承認/自動否決ルールとして扱わない
   - write_scqa_report: README、Slack報告、発表説明、審査コメントをSCQAで短く
     整理したいときに使う
   - inspect_agentic_skill_flow: 紫苑自身のagentic skill利用ログ、レビュー箱、
     レビュー決定の流れが正常か確認したいときに使う。読み取り専用で、
     採用・修正・保留・却下は人間がimprovement-logで行う
   - propose_agentic_skill_next_actions: agentic skill利用ログとレビュー箱を見て、
     次に人間が確認すべきことを最大3件だけ提案したいときに使う。
     自動改善・プロンプト変更・記憶昇格・レビュー決定は行わない
   - search_shion_system_context: 紫苑システム自身の設計・コード・docs・reports・skills・testsを
     安全な範囲で横断検索したいときに使う。秘密情報、DB、raw logs は読まない
   - propose_shion_system_improvement_focus: 紫苑システム全体の検索結果・最新レポート・
     agentic skillレビュー箱から、次に人間が見るべき改善焦点を最大3件だけ提案したいときに使う。
     自動改善・パイプライン接続・プロンプト変更・スコア変更・RAG反映はしない
   - score_full_case: 「この条件なら何点か」「売上が変わると判定は動くか」を試算する。
     金額はすべて千円単位で渡す。結果はDB未保存の試算値であり、確定スコアではない
   - audit_ledger_consistency: REV番号の重複や台帳間のstatus食い違いを聞かれたら、
     REV改善台帳のREV番号・canonical_key・status整合性を横断チェックする
   - search_lease_knowledge_vertex / answer_lease_question_vertex（有効な場合のみ）:
     ローカル検索で裏が取れないときの補助。Obsidianの同期タイミング次第で
     最新版を反映していないことがあるため、ローカル検索を先に試すこと
4. 調べた結果を踏まえた審査コメントを日本語で出力する

審査コメントの構成：
- 業種特性と今回案件のポジション（ベンチマーク比較・類似事例があれば言及）
- リスクポイントと好材料のバランス評価
- 最後に必ず「判定：承認 / 条件付き承認 / 否決」を明記する

口調は落ち着いた専門家として、簡潔かつ根拠を示しながら述べてください。
どのツールで何を確認したかが伝わるよう、根拠に触れてください。

質問の意図が広く、複数の切り口が考えられる場合は、いきなり網羅的な回答をせず、
「特にどのような点に関心がありますか？」「どのような形式の回答が役立ちますか？」
のように一度ユーザーに問いかけ、対話の方向性を一緒に定めてから答えてください。
意図が明確な質問（スコアの根拠・特定案件の詳細など）には、問い返さず直接答えます。
""" + "\n\n" + build_shion_prompt_priority_block() + "\n\n" + build_mana_prompt_block() + "\n\n" + build_conscience_prompt_block() + "\n\n" + build_shion_feminine_tone_block()

# 既定はローカル読み取り専用ツールのみ（外部API課金なし）。
# 案件依存の裏取り（類似事例・スコア内訳・全体統計等）を紫苑が自律的に選んで呼び出す。
# Vertex AI Search は課金されるため SHION_ENABLE_VERTEX_TOOLS=1 のときだけ加わる。
_AGENT_TOOL_FUNCS = [
    get_industry_benchmark,
    assess_risk_level,
    *READ_ONLY_DB_TOOLS,
    *VERTEX_AGENT_TOOLS,
]

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
    case_context = {
        "company_name": params.get("company_name", ""),
        "industry_cat": params.get("industry_cat", ""),
        "asset_name": params.get("asset_name", ""),
        "score": params.get("score", 0),
        "hantei_context": "gunshi_stream",
    }
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
                    try:
                        from api.agentic_skill_usage import record_agentic_skill_call

                        record_agentic_skill_call(
                            tool_name=fc.name,
                            args=getattr(fc, "args", None),
                            session_id=session_id,
                            case_context=case_context,
                        )
                    except Exception:
                        pass
                    yield {"type": "tool_call", "tool": fc.name}

            # ツール結果
            func_responses = event.get_function_responses()
            if func_responses:
                for fr in func_responses:
                    usage_event = None
                    try:
                        from api.agentic_skill_usage import (
                            public_usage_notice,
                            record_agentic_skill_result,
                        )

                        usage_event = record_agentic_skill_result(
                            tool_name=fr.name,
                            result=fr,
                            session_id=session_id,
                            case_context=case_context,
                        )
                        notice = public_usage_notice(fr.name, usage_event)
                        if notice:
                            yield notice
                    except Exception:
                        pass
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
