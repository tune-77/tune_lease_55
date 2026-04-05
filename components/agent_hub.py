# -*- coding: utf-8 -*-
"""
components/agent_hub.py
=======================
汎用エージェントハブ — 8種のエージェントを一元管理する画面。

1. 業界ベンチマーク自動取得
2. 金利・市況モニタリング
3. 審査理由書自動生成（軍師モード）
4. エージェントチーム議論の自律化
5. Slack通知の高度化
6. 異常検知
7. モデル再学習トリガー
8. レポート定期配信
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
import datetime
import threading
import statistics
import requests
import streamlit as st

from session_keys import SK

# ── ディレクトリ設定 ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR   = os.path.dirname(_SCRIPT_DIR)
_DB_PATH    = os.path.join(_BASE_DIR, "data", "lease_data.db")
_HUB_LOG    = os.path.join(_BASE_DIR, "data", "agent_hub_log.jsonl")
_AGENT_THOUGHTS = os.path.join(_BASE_DIR, "data", "agent_thoughts.jsonl")

# ── スケジューラ（グローバル — Streamlit rerenderをまたいで保持）──────────────
_scheduler_store: dict = {}   # {"scheduler": BackgroundScheduler instance}
_scheduler_lock = threading.Lock()

# ── 審査専門家ペルソナ（AUDIT_AGENTS） ─────────────────────────────────────────
# 「開発会議モード」の DEV ペルソナとは別に、審査目線の専門家チームを定義。
AUDIT_AGENTS: dict[str, str] = {
    "信用審査官": (
        "あなたは銀行出身の「信用審査官」です。PD（デフォルト確率）・LGD・EAD の三角形で案件を評価します。"
        "財務指標の定量面を中心に、100字以内でリスクと承認可否を述べてください。"
    ),
    "業種アナリスト": (
        "あなたは産業調査部門の「業種アナリスト」です。業種の成長性・景気感応度・参入障壁を軸に評価します。"
        "マクロ視点と業界動向を踏まえ、100字以内で意見を述べてください。"
    ),
    "担保評価士": (
        "あなたは「担保評価士」です。リース物件の残存価値・換価性・市場流動性を専門とします。"
        "物件保全の観点から、100字以内でリスクを評価してください。"
    ),
    "コンプライアンス担当": (
        "あなたは「コンプライアンス担当」です。反社チェック・資金使途・規制リスクを担当します。"
        "法的・倫理的リスクの観点から、100字以内で懸念点を述べてください。"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 共通ユーティリティ
# ══════════════════════════════════════════════════════════════════════════════

def _get_ai_settings() -> tuple[str, str, str, str]:
    """現在のAI設定を返す (engine, ollama_model, api_key, gemini_model)。"""
    from ai_chat import (
        get_ollama_model, GEMINI_API_KEY_ENV,
        GEMINI_MODEL_DEFAULT, _get_gemini_key_from_secrets,
    )
    engine       = st.session_state.get(SK.AI_ENGINE, "ollama")
    ollama_model = get_ollama_model()
    api_key      = (
        (st.session_state.get(SK.GEMINI_API_KEY) or "").strip()
        or GEMINI_API_KEY_ENV
        or _get_gemini_key_from_secrets()
    )
    gemini_model = st.session_state.get(SK.GEMINI_MODEL, GEMINI_MODEL_DEFAULT)
    return engine, ollama_model, api_key, gemini_model


def _ai_call(prompt: str, system: str = "", timeout: int = 120) -> str:
    """AI呼び出しの薄いラッパー。エラー時は空文字列を返す。"""
    try:
        from ai_chat import _chat_for_thread, is_ai_available
        if not is_ai_available():
            return ""
        engine, model, api_key, gemini_model = _get_ai_settings()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        res = _chat_for_thread(engine, model, messages,
                               timeout_seconds=timeout,
                               api_key=api_key,
                               gemini_model=gemini_model)
        return (res.get("message") or {}).get("content", "") or ""
    except Exception as e:
        return f"[AI呼び出しエラー: {e}]"


def _get_slack_url() -> str:
    """Slack Webhook URL を取得。"""
    url = (st.session_state.get(SK.SLACK_WEBHOOK_URL) or "").strip()
    if url:
        return url
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if url:
        return url
    try:
        url = (st.secrets.get("SLACK_WEBHOOK_URL") or "").strip()
    except Exception:
        pass
    return url


def _send_slack(blocks: list, text: str = "通知") -> bool:
    """Slack Block Kit メッセージを送信。成功時 True。"""
    url = _get_slack_url()
    if not url:
        return False
    try:
        r = requests.post(url, json={"text": text, "blocks": blocks}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _post_agent_thought(agent: str, thought: str, icon: str = "💭") -> None:
    """エージェントの『つぶやき』を記録する"""
    try:
        os.makedirs(os.path.dirname(_AGENT_THOUGHTS), exist_ok=True)
        with open(_AGENT_THOUGHTS, "a", encoding="utf-8") as f:
            entry = {
                "ts": datetime.datetime.now().isoformat(),
                "agent": agent,
                "thought": thought,
                "icon": icon
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _hub_log(agent: str, status: str, detail: str = "") -> None:
    """エージェントの実行ログを JSONL に追記。"""
    entry = {
        "ts":     datetime.datetime.now().isoformat(),
        "agent":  agent,
        "status": status,
        "detail": detail[:500],
    }
    try:
        with open(_HUB_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_past_cases(limit: int = 200) -> list[dict]:
    """past_cases テーブルから最新N件を取得。"""
    try:
        with closing(sqlite3.connect(_DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT id, timestamp, industry_sub, score, user_eq, final_status, data "
                "FROM past_cases ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for r in rows:
            d = {}
            try:
                d = json.loads(r[6]) if r[6] else {}
            except Exception:
                pass
            result.append({
                "id": r[0], "timestamp": r[1], "industry_sub": r[2],
                "score": r[3], "user_eq": r[4], "final_status": r[5],
                **d,
            })
        return result
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# Agent 1 — 業界ベンチマーク自動取得
# ══════════════════════════════════════════════════════════════════════════════

_BENCHMARK_SYSTEM = (
    "あなたはリース審査の財務分析専門家です。"
    "指定された業種について、日本の中小企業の財務指標（業界平均）を推定してください。"
    "以下のJSON形式のみで回答してください（説明文不要）:\n"
    '{"op_margin": <営業利益率%>, "equity_ratio": <自己資本比率%>, '
    '"roa": <ROA%>, "current_ratio": <流動比率%>, "dscr": <DSCR倍>}'
)


def _run_benchmark_agent(industry: str) -> dict | None:
    """業種名を渡してAIがベンチマーク指標を推定。"""
    prompt = f"業種「{industry}」の日本中小企業の財務ベンチマーク指標を推定してください。"
    raw = _ai_call(prompt, system=_BENCHMARK_SYSTEM, timeout=60)
    try:
        # JSON部分を抽出
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start >= 0 else None
    except Exception:
        return None


def _render_benchmark_panel() -> None:
    st.subheader("🏭 業界ベンチマーク自動取得")
    st.caption("業種名を入力するとAIが財務指標の業界平均を推定し、審査に使用できます。")

    industry = st.text_input("業種名", placeholder="例: 製造業（一般機械）", key="hub_bench_industry")
    if st.button("📡 ベンチマーク取得", key="hub_bench_run", disabled=not industry):
        with st.spinner("AI推定中..."):
            result = _run_benchmark_agent(industry)
        if result:
            st.success("取得完了")
            cols = st.columns(5)
            labels = ["営業利益率%", "自己資本比率%", "ROA%", "流動比率%", "DSCR倍"]
            keys   = ["op_margin", "equity_ratio", "roa", "current_ratio", "dscr"]
            for col, lbl, k in zip(cols, labels, keys):
                col.metric(lbl, f"{result.get(k, '—')}")
            st.session_state["hub_bench_result"] = result
            st.session_state["hub_bench_industry"] = industry
            _hub_log("benchmark", "success", str(result))
        else:
            st.error("取得失敗。AIエンジンの接続を確認してください。")

    if st.session_state.get("hub_bench_result"):
        result   = st.session_state["hub_bench_result"]
        industry = st.session_state.get("hub_bench_industry", "")
        if st.button("✅ このベンチマークを審査に適用", key="hub_bench_apply"):
            st.session_state["override_bench"] = {"industry": industry, **result}
            st.success(f"「{industry}」のベンチマークを適用しました。次回審査から反映されます。")
            _hub_log("benchmark", "applied", industry)


# ══════════════════════════════════════════════════════════════════════════════
# Agent 2 — 金利・市況モニタリング
# ══════════════════════════════════════════════════════════════════════════════

_RATE_SYSTEM = (
    "あなたは日本の金融市場アナリストです。"
    "現在の日本の金利・市況について簡潔に報告し、リース審査への影響を述べてください。"
    "以下のJSON形式で回答してください:\n"
    '{"policy_rate": <日銀政策金利%>, "prime_rate": <短期プライムレート%>, '
    '"10y_jgb": <10年国債利回り%>, "market_comment": "<50字以内のコメント>", '
    '"lease_impact": "<審査への影響 50字以内>"}'
)


def _run_market_agent() -> dict | None:
    prompt = (
        f"本日（{datetime.date.today()}）時点の日本の金融市況を報告してください。"
        "日銀政策金利、短期プライムレート、10年国債利回りの最新推定値と、"
        "リース審査DSCRへの影響をまとめてください。"
    )
    raw = _ai_call(prompt, system=_RATE_SYSTEM, timeout=60)
    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        return json.loads(raw[start:end]) if start >= 0 else None
    except Exception:
        return None


def _render_market_panel() -> None:
    st.subheader("📈 金利・市況モニタリング")
    st.caption("AIが日本の金融市況を要約し、DSCR計算への影響を分析します。")

    last = st.session_state.get("hub_market_last")
    if last:
        ts = st.session_state.get("hub_market_ts", "")
        st.info(f"最終取得: {ts}")
        cols = st.columns(3)
        cols[0].metric("政策金利", f"{last.get('policy_rate', '—')}%")
        cols[1].metric("短期プライムレート", f"{last.get('prime_rate', '—')}%")
        cols[2].metric("10年国債", f"{last.get('10y_jgb', '—')}%")
        st.markdown(f"**市況コメント:** {last.get('market_comment', '')}")
        st.markdown(f"**審査への影響:** {last.get('lease_impact', '')}")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 市況を更新", key="hub_market_run"):
            with st.spinner("AI分析中..."):
                result = _run_market_agent()
            if result:
                st.session_state["hub_market_last"] = result
                st.session_state["hub_market_ts"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                _hub_log("market", "success", str(result))
                # Slackに通知
                if _get_slack_url():
                    impact = result.get("lease_impact", "")
                    _send_slack([{
                        "type": "section",
                        "text": {"type": "mrkdwn",
                                 "text": f"*📈 金利・市況レポート* ({datetime.date.today()})\n"
                                         f"政策金利: {result.get('policy_rate')}%　"
                                         f"プライムレート: {result.get('prime_rate')}%\n"
                                         f"市況: {result.get('market_comment', '')}\n"
                                         f"審査影響: {impact}"}
                    }], text="金利・市況レポート")
                st.rerun()
            else:
                st.error("取得失敗")
    with col2:
        alert_thresh = st.number_input(
            "アラート閾値（政策金利が この値を超えたら警告）",
            value=1.0, step=0.25, format="%.2f", key="hub_market_thresh"
        )
        if last and (last.get("policy_rate") or 0) > alert_thresh:
            st.warning(f"⚠️ 政策金利 {last['policy_rate']}% が閾値 {alert_thresh}% を超えています！DSCRを再確認してください。")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 3 — 審査理由書自動生成（軍師モード）
# ══════════════════════════════════════════════════════════════════════════════

_GUNSHI_SYSTEM = """あなたは「審査軍師」です。リース審査の達人として、
審査データを元に稟議書（審査理由書）を作成してください。
以下の形式で出力してください:

# 審査理由書

## 1. 案件概要
（物件・業種・判定結果）

## 2. 財務分析
（主要指標と業界比較の所見）

## 3. 承認/否決理由
（スコアと定性評価に基づく具体的理由）

## 4. リスク事項
（確認すべきリスクと留意点）

## 5. 総合所見
（1〜3行の結論）

文体は丁寧・簡潔に。数値は根拠として明記すること。"""


def _run_report_gen_agent(res: dict) -> str:
    score    = res.get("score", 0)
    hantei   = res.get("hantei", "—")
    industry = res.get("industry_sub", "—")
    asset    = res.get("asset_name", "—")
    user_op  = res.get("user_op")
    bench_op = res.get("bench_op")
    user_eq  = res.get("user_eq")
    user_dscr= res.get("user_dscr")
    qs_rank  = (res.get("qualitative_scoring_correction") or {}).get("rank_text", "—")
    strength = ", ".join(res.get("strength_tags") or [])

    prompt = f"""以下の審査結果データをもとに稟議書を作成してください。

【審査データ】
- 業種: {industry}
- 対象物件: {asset}
- 総合スコア: {score:.0f}点
- 判定: {hantei}
- 営業利益率: {user_op}%（業界平均: {bench_op}%）
- 自己資本比率: {user_eq}%
- DSCR: {user_dscr}倍
- 定性ランク: {qs_rank}
- 強みタグ: {strength}
"""
    return _ai_call(prompt, system=_GUNSHI_SYSTEM, timeout=120)


def _render_report_gen_panel() -> None:
    st.subheader("📝 審査理由書自動生成（軍師モード）")
    st.caption("直前の審査結果から稟議書を自動生成します。審査・分析を先に実行してください。")

    res = st.session_state.get(SK.LAST_RESULT)
    if not res:
        st.info("審査データがありません。「審査・分析」から審査を実行してください。")
        return

    score   = res.get("score", 0)
    hantei  = res.get("hantei", "—")
    st.info(f"対象: {res.get('industry_sub', '—')} | スコア {score:.0f}点 | {hantei}")

    if st.button("⚔️ 審査理由書を生成", key="hub_report_gen"):
        with st.spinner("軍師が稟議書を作成中..."):
            doc = _run_report_gen_agent(res)
        st.session_state["hub_report_doc"] = doc
        _hub_log("report_gen", "success", f"score={score}")

    doc = st.session_state.get("hub_report_doc")
    if doc:
        st.markdown(doc)
        st.download_button(
            "📥 テキストでダウンロード",
            data=doc.encode("utf-8"),
            file_name=f"審査理由書_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            key="hub_report_dl",
        )
        if st.button("📤 Slackに送信", key="hub_report_slack"):
            sent = _send_slack([{
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*📝 審査理由書* (スコア {score:.0f}点 / {hantei})\n"
                                 + doc[:2000]}
            }], text="審査理由書")
            st.success("送信しました" if sent else "Slack未設定")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 4 — エージェントチーム議論の自律化
# ══════════════════════════════════════════════════════════════════════════════

def _run_auto_team_agent(res: dict, mode: str = "dev") -> str:
    """
    要審議ゾーンの案件をエージェントチームに自動投稿し、議論結果を返す。

    Args:
        res:  審査結果 dict（score, hantei, industry_sub, asset_name 等を含む）
        mode: "dev"（開発会議）または "audit"（審査会議）
    """
    score   = res.get("score", 0)
    hantei  = res.get("hantei", "—")
    industry= res.get("industry_sub", "—")
    asset   = res.get("asset_name", "—")
    pd_pct  = res.get("pd_pct", None)
    equity  = res.get("equity_ratio", None)

    # 審査コンテキストを共有
    ctx_parts = [
        f"スコア {score:.0f}点（{hantei}）",
        f"業種: {industry}",
        f"物件: {asset}",
    ]
    if pd_pct is not None:
        ctx_parts.append(f"PD: {pd_pct:.1%}")
    if equity is not None:
        ctx_parts.append(f"自己資本比率: {equity:.1%}")

    if mode == "audit":
        theme = (
            "以下の案件を審査専門家として評価してください。\n"
            + "、".join(ctx_parts)
            + "。\n承認・条件付き承認・否決のいずれかを、リスク定量根拠とともに述べてください。"
        )
        active_personas: dict[str, str] = AUDIT_AGENTS
        chair_name = "審査委員長"
        chair_system = (
            "あなたは「審査委員長」として、各専門家の意見を集約し、"
            "最終判定（承認/条件付き承認/否決）と承認条件・リスク軽減策を200字以内で決裁します。"
        )
    else:
        theme = (
            "、".join(ctx_parts)
            + " の案件について、開発・運用の観点から審議してください。"
            "承認すべきか、条件付き承認か、否決かを議論し理由を述べてください。"
        )
        active_personas = {
            "プランナー": "物理学者・行動経済学者視点から",
            "ダッシュ":   "UI/UXと数値可視化の観点から",
            "田中さん":   "営業現場の感覚から",
            "鈴木さん":   "技術・実装リスクの観点から",
        }
        chair_name = "Tune"
        chair_system = "あなたは統括マネージャー「Tune」として、チームの意見を集約し最終決裁を下します。"

    st.session_state[SK.AT_THEME] = theme

    opinions: dict[str, str] = {}
    for name, hint in active_personas.items():
        if mode == "audit":
            system = hint  # AUDIT_AGENTS は既にフル system プロンプト
        else:
            system = f"あなたは「{name}」として、{hint}リース審査案件を評価する担当者です。100字以内で意見を述べてください。"
        opinions[name] = _ai_call(f"テーマ: {theme}", system=system, timeout=60)

    chair_prompt = (
        f"テーマ: {theme}\n\n各担当者の意見:\n"
        + "\n".join(f"・{k}: {v}" for k, v in opinions.items())
        + f"\n\n以上を踏まえ、{chair_name}として最終判断と理由を200字以内で述べてください。"
    )
    chair = _ai_call(chair_prompt, system=chair_system, timeout=90)

    summary = f"**テーマ:** {theme}\n\n"
    for k, v in opinions.items():
        summary += f"**{k}:** {v}\n\n"
    summary += f"**{chair_name}（最終判断）:** {chair}"
    return summary


def _render_auto_team_panel() -> None:
    st.subheader("🤝 エージェントチーム議論の自律化")
    st.caption("スコアが要審議ゾーン（40〜70点）の案件を自動でチームに投げて議論させます。")

    res = st.session_state.get(SK.LAST_RESULT)
    if not res:
        st.info("審査データがありません。")
        return

    score  = res.get("score", 0)
    hantei = res.get("hantei", "—")

    # ── 会議モード切替 ────────────────────────────────────────────────────────
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        mode_label = st.radio(
            "会議モード",
            options=["開発会議", "審査会議"],
            horizontal=True,
            key="hub_team_mode",
        )
    meeting_mode = "audit" if mode_label == "審査会議" else "dev"
    if meeting_mode == "audit":
        st.caption("👔 審査専門家（信用審査官・業種アナリスト・担保評価士・コンプライアンス担当）が議論します。")
    else:
        st.caption("🛠️ 開発チーム（プランナー・ダッシュ・田中さん・鈴木さん）が議論します。")

    thresh_low  = st.slider("要審議下限", 30, 50, 40, key="hub_team_low")
    thresh_high = st.slider("要審議上限", 60, 80, 70, key="hub_team_high")

    in_zone = thresh_low <= score <= thresh_high
    status  = "🟡 要審議ゾーン" if in_zone else ("🟢 承認圏" if score > thresh_high else "🔴 否決圏")
    st.metric("現在のスコア", f"{score:.0f}点", delta=status)

    auto_mode = st.toggle("スコアが要審議ゾーンに入ったら自動起動", key="hub_team_auto")
    if auto_mode and in_zone:
        st.warning("要審議ゾーンを検出。チーム議論を自動起動します。")

    manual = st.button("▶️ 今すぐチーム議論を起動", key="hub_team_manual")
    if manual or (auto_mode and in_zone and not st.session_state.get("hub_team_done")):
        with st.spinner("チームが議論中..."):
            result = _run_auto_team_agent(res, mode=meeting_mode)
        st.session_state["hub_team_result"] = result
        st.session_state["hub_team_done"]   = True
        _hub_log("auto_team", "success", f"score={score} mode={meeting_mode}")

        if _get_slack_url():
            _send_slack([{
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*🤝 エージェントチーム議論完了*\nスコア {score:.0f}点 / {hantei}\n\n"
                                 + result[:1500]}
            }], text="エージェントチーム議論完了")

    result = st.session_state.get("hub_team_result")
    if result:
        st.markdown(result)
        if st.button("🔄 再議論", key="hub_team_reset"):
            st.session_state.pop("hub_team_done", None)
            st.session_state.pop("hub_team_result", None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Agent 5 — Slack通知の高度化
# ══════════════════════════════════════════════════════════════════════════════

_SLACK_FOLLOWUP_SYSTEM = (
    "あなたはリース営業サポートAIです。審査結果をもとに、担当営業向けの"
    "フォローアップメッセージを作成してください。150字以内で、次のアクション提案を含めてください。"
)


def _render_slack_enhanced_panel() -> None:
    st.subheader("💬 Slack通知の高度化")
    st.caption("AIが審査結果に応じたフォローアップ文を生成し、Slackに送信します。")

    url = _get_slack_url()
    if not url:
        st.warning("Slack Webhook URLが設定されていません。エージェントチーム議論画面で設定してください。")

    res = st.session_state.get(SK.LAST_RESULT)
    if not res:
        st.info("審査データがありません。")
        return

    score  = res.get("score", 0)
    hantei = res.get("hantei", "—")
    industry = res.get("industry_sub", "—")

    notify_type = st.radio(
        "通知種別",
        ["審査完了通知（AIフォローアップ付き）", "シンプル審査完了通知", "カスタムメッセージ"],
        key="hub_slack_type",
    )

    custom_msg = ""
    if notify_type == "カスタムメッセージ":
        custom_msg = st.text_area("メッセージ", key="hub_slack_custom")

    if st.button("📤 Slackに送信", key="hub_slack_send", disabled=not url):
        if notify_type == "審査完了通知（AIフォローアップ付き）":
            with st.spinner("AIがフォローアップ文を生成中..."):
                followup = _ai_call(
                    f"業種: {industry}、スコア: {score:.0f}点、判定: {hantei}",
                    system=_SLACK_FOLLOWUP_SYSTEM, timeout=60
                )
            blocks = [
                {"type": "header",
                 "text": {"type": "plain_text", "text": f"{'✅' if '承認' in hantei else '❌' if '否決' in hantei else '⚠️'} 審査完了: {hantei}"}},
                {"type": "section", "fields": [
                    {"type": "mrkdwn", "text": f"*業種:* {industry}"},
                    {"type": "mrkdwn", "text": f"*スコア:* {score:.0f}点"},
                ]},
                {"type": "section",
                 "text": {"type": "mrkdwn", "text": f"*💡 フォローアップ提案:*\n{followup}"}},
                {"type": "divider"},
                {"type": "context",
                 "elements": [{"type": "mrkdwn",
                                "text": f"送信日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"}]},
            ]
            sent = _send_slack(blocks, text=f"審査完了: {hantei} / {score:.0f}点")
        elif notify_type == "シンプル審査完了通知":
            sent = _send_slack([{
                "type": "section",
                "text": {"type": "mrkdwn",
                         "text": f"*審査完了* — {industry} | スコア {score:.0f}点 | {hantei}"}
            }], text="審査完了通知")
        else:
            sent = _send_slack([{
                "type": "section",
                "text": {"type": "mrkdwn", "text": custom_msg}
            }], text=custom_msg[:50])

        st.success("送信しました ✅") if sent else st.error("送信失敗（URL確認）")
        _hub_log("slack_enhanced", "sent" if sent else "failed")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 6 — 異常検知
# ══════════════════════════════════════════════════════════════════════════════

def _run_anomaly_agent(cases: list[dict], target: dict | None = None) -> list[dict]:
    """
    past_cases から各指標のZスコアを計算し、|Z| > 閾値 の案件を返す。
    target が指定された場合はその案件もチェックする。
    """
    if len(cases) < 5:
        return []

    scores  = [c["score"] for c in cases if c.get("score") is not None]
    eq_vals = [c["user_eq"] for c in cases if c.get("user_eq") is not None]

    def z_score(val, vals):
        if len(vals) < 2:
            return 0.0
        mu  = statistics.mean(vals)
        sig = statistics.stdev(vals)
        return (val - mu) / sig if sig else 0.0

    anomalies = []
    check_cases = list(cases)
    if target:
        check_cases = [target] + check_cases

    for c in check_cases:
        flags = []
        if c.get("score") is not None:
            z = z_score(c["score"], scores)
            if abs(z) > 2.0:
                flags.append(f"スコア Z={z:+.1f}")
        if c.get("user_eq") is not None:
            z = z_score(c["user_eq"], eq_vals)
            if abs(z) > 2.0:
                flags.append(f"自己資本比率 Z={z:+.1f}")
        if flags:
            anomalies.append({
                "id":       c.get("id", "現在の案件"),
                "industry": c.get("industry_sub", "—"),
                "score":    c.get("score"),
                "flags":    ", ".join(flags),
                "ts":       c.get("timestamp", ""),
            })
    return anomalies[:20]


def _render_anomaly_panel() -> None:
    st.subheader("🚨 異常検知")
    st.caption("過去審査データのZスコア分析で統計的外れ値案件を検出します。")

    current = st.session_state.get(SK.LAST_RESULT)
    include_current = st.checkbox("現在の審査案件も含める", value=True, key="hub_anom_current")

    if st.button("🔍 異常検知を実行", key="hub_anom_run"):
        with st.spinner("データ分析中..."):
            cases = _load_past_cases(500)
            target = None
            if include_current and current:
                target = {
                    "id": "【現在の案件】",
                    "industry_sub": current.get("industry_sub"),
                    "score": current.get("score"),
                    "user_eq": current.get("user_eq"),
                }
            anomalies = _run_anomaly_agent(cases, target)

        st.session_state["hub_anom_result"] = anomalies
        st.session_state["hub_anom_total"]  = len(cases)
        _hub_log("anomaly", "success", f"found={len(anomalies)}")

    result = st.session_state.get("hub_anom_result")
    total  = st.session_state.get("hub_anom_total", 0)

    if result is not None:
        st.metric("分析件数", f"{total}件", delta=f"異常 {len(result)}件検出")
        if result:
            import pandas as pd
            df = pd.DataFrame(result)
            df.columns = ["案件ID", "業種", "スコア", "異常フラグ", "日時"]
            st.dataframe(df, width='stretch')

            if _get_slack_url() and st.button("📤 Slack に異常レポート送信", key="hub_anom_slack"):
                rows = "\n".join(
                    f"・{r['id']} ({r['industry']}) スコア{r['score']:.0f} — {r['flags']}"
                    for r in result[:10]
                )
                _send_slack([{
                    "type": "section",
                    "text": {"type": "mrkdwn",
                             "text": f"*🚨 異常検知レポート* — {len(result)}件検出\n{rows}"}
                }], text="異常検知レポート")
                st.success("送信しました")
        else:
            st.success("統計的外れ値は検出されませんでした ✅")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 7 — モデル再学習トリガー
# ══════════════════════════════════════════════════════════════════════════════

def _count_labeled_cases() -> int:
    """final_status が記録された案件数を返す。"""
    try:
        with closing(sqlite3.connect(_DB_PATH)) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM past_cases WHERE final_status IS NOT NULL AND final_status != ''"
            ).fetchone()[0]
        return n
    except Exception:
        return 0


def _run_retrain_trigger(threshold: int) -> str:
    """閾値に達している場合に再学習を試みる。"""
    n = _count_labeled_cases()
    if n < threshold:
        return f"ラベル済み件数: {n}件（閾値 {threshold}件未達）"

    # auto_optimizer が存在する場合に呼び出し
    try:
        import importlib
        ao = importlib.import_module("auto_optimizer")
        if hasattr(ao, "run_auto_optimization"):
            ao.run_auto_optimization()
            _hub_log("retrain", "triggered", f"n={n}")
            return f"✅ 再学習を実行しました（ラベル済み {n}件）"
        elif hasattr(ao, "update_coefficients"):
            ao.update_coefficients()
            _hub_log("retrain", "triggered", f"n={n}")
            return f"✅ 係数更新を実行しました（ラベル済み {n}件）"
    except Exception as e:
        _hub_log("retrain", "error", str(e))

    _hub_log("retrain", "manual_needed", f"n={n}")
    return f"⚠️ ラベル済み {n}件に達しました。「係数分析・更新」画面から手動で再学習してください。"


def _render_retrain_panel() -> None:
    st.subheader("🔄 モデル再学習トリガー")
    st.caption("成約/失注データが一定数蓄積されたら係数更新を実行します。")

    n = _count_labeled_cases()
    threshold = st.slider("再学習発動の閾値（件数）", 10, 500, 50, step=10, key="hub_retrain_thresh")

    progress = min(n / threshold, 1.0)
    st.metric("ラベル済み案件数", f"{n}件", delta=f"目標まで {max(0, threshold - n)}件")
    st.progress(progress, text=f"{n}/{threshold}件")

    auto = st.toggle("閾値到達時に自動で再学習を試みる", key="hub_retrain_auto")
    if auto and n >= threshold:
        st.warning("閾値に達しています。再学習を実行します。")

    if st.button("▶️ 今すぐ再学習を試みる", key="hub_retrain_run") or (auto and n >= threshold):
        with st.spinner("再学習実行中..."):
            msg = _run_retrain_trigger(threshold)
        st.info(msg)
        if _get_slack_url():
            _send_slack([{
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*🔄 モデル再学習* — {msg}"}
            }], text="モデル再学習通知")


# ══════════════════════════════════════════════════════════════════════════════
# Agent 8 — レポート定期配信
# ══════════════════════════════════════════════════════════════════════════════

_REPORT_SUMMARY_SYSTEM = (
    "あなたはリース審査AI のレポートライターです。"
    "以下の審査統計データをもとに、週次サマリーレポートを200字以内で作成してください。"
    "ポジティブなトレンドと注意点を箇条書きで述べてください。"
)


def _generate_weekly_summary() -> str:
    """直近7日間の審査データのサマリーを生成。"""
    cases = _load_past_cases(200)
    week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
    recent   = [c for c in cases if (c.get("timestamp") or "") >= week_ago]

    if not recent:
        return "直近7日間の審査データがありません。"

    scores  = [c["score"] for c in recent if c.get("score") is not None]
    approvals = [c for c in recent if "承認" in (c.get("final_status") or c.get("hantei") or "")]
    avg_score = statistics.mean(scores) if scores else 0

    stats_text = (
        f"直近7日間の審査件数: {len(recent)}件\n"
        f"平均スコア: {avg_score:.1f}点\n"
        f"承認件数: {len(approvals)}件\n"
        f"承認率: {len(approvals)/len(recent)*100:.0f}%"
    )
    return _ai_call(stats_text, system=_REPORT_SUMMARY_SYSTEM, timeout=60) or stats_text


def _send_weekly_report() -> None:
    """週次レポートを生成してSlackに送信（スケジューラから呼ばれる）。"""
    summary = _generate_weekly_summary()
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        try:
            import toml
            secrets_path = os.path.join(_BASE_DIR, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                sec = toml.load(secrets_path)
                url = sec.get("SLACK_WEBHOOK_URL", "")
        except Exception:
            pass

    if url:
        payload = {"text": f"📊 週次審査レポート\n{summary}"}
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass
    _hub_log("schedule", "weekly_sent", summary[:200])


def _get_scheduler():
    """APScheduler の BackgroundScheduler を取得（なければ起動）。"""
    with _scheduler_lock:
        if "scheduler" not in _scheduler_store:
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
                sched = BackgroundScheduler(timezone="Asia/Tokyo")
                sched.start()
                _scheduler_store["scheduler"] = sched
            except Exception as e:
                _scheduler_store["scheduler"] = None
                _scheduler_store["error"] = str(e)
        return _scheduler_store.get("scheduler")


def _render_schedule_panel() -> None:
    st.subheader("📅 レポート定期配信")
    st.caption("週次レポートを自動生成してSlackに配信します。APSchedulerを使用。")

    if not _get_slack_url():
        st.warning("Slack Webhook URLが未設定のため配信できません。")

    sched = _get_scheduler()
    if sched is None:
        st.error(f"スケジューラ起動失敗: {_scheduler_store.get('error', '不明')}")
        return

    # 設定UI
    col1, col2 = st.columns(2)
    with col1:
        weekday = st.selectbox("配信曜日", ["月", "火", "水", "木", "金", "土", "日"],
                               index=0, key="hub_sched_day")
    with col2:
        hour = st.number_input("配信時刻（時）", 0, 23, 9, key="hub_sched_hour")

    day_map = {"月": "mon", "火": "tue", "水": "wed", "木": "thu",
               "金": "fri", "土": "sat", "日": "sun"}

    # 現在のジョブ一覧
    jobs = sched.get_jobs()
    weekly_job = next((j for j in jobs if j.id == "weekly_report"), None)

    if weekly_job:
        st.success(f"✅ 定期配信設定済み — 次回: {weekly_job.next_run_time.strftime('%Y-%m-%d %H:%M') if weekly_job.next_run_time else '—'}")
        if st.button("🗑️ 配信を停止", key="hub_sched_stop"):
            sched.remove_job("weekly_report")
            st.rerun()
    else:
        if st.button("▶️ 定期配信を開始", key="hub_sched_start"):
            sched.add_job(
                _send_weekly_report,
                trigger="cron",
                day_of_week=day_map[weekday],
                hour=int(hour),
                minute=0,
                id="weekly_report",
                replace_existing=True,
            )
            _hub_log("schedule", "started", f"{weekday}曜 {hour}時")
            st.rerun()

    st.divider()
    st.markdown("**今すぐプレビュー送信**")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("👁️ サマリーをプレビュー", key="hub_sched_preview"):
            with st.spinner("レポート生成中..."):
                summary = _generate_weekly_summary()
            st.session_state["hub_sched_preview_text"] = summary

    preview = st.session_state.get("hub_sched_preview_text")
    if preview:
        st.text_area("プレビュー", value=preview, height=150, key="hub_sched_preview_area")
        with col_b:
            if st.button("📤 今すぐ送信", key="hub_sched_send_now", disabled=not _get_slack_url()):
                _send_weekly_report()
                st.success("送信しました")


# ══════════════════════════════════════════════════════════════════════════════
# タム（子犬のAI）パネル
# ══════════════════════════════════════════════════════════════════════════════

def _render_koinu_panel() -> None:
    """🐶 タムタブ — 今日の報告・タムと話す・謎のメモ帳"""
    from koinu_agent import get_pochi
    from session_keys import SK as _SK

    pochi = get_pochi()

    st.markdown("### 🐾 タムの部屋")
    st.caption("謎の子犬AIタム。茶色クルクル巻き毛・パッチリお目目のマルプー。表面は無邪気な子犬。その実態は…誰も知らない。")

    # ── 現在の審査結果を取得 ──────────────────────────────────────────────
    res = st.session_state.get(_SK.LAST_RESULT) or {}

    sub_tabs = st.tabs(["📋 今日のタムの報告", "💬 タムと話す", "📓 タムの謎のメモ帳", "🏃 散歩ログ"])

    # ─────────────────────────────────────────────────────────────────────
    # タブ1: 今日のタムの報告
    # ─────────────────────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.markdown("#### 🐶 タムからの今日の報告")

        if not res:
            st.info("まだ審査データがありません。審査を実行してからタムに報告させましょう。")
            st.markdown("わん！（しっぽを振りながら待っている）")
        else:
            col1, col2, col3 = st.columns(3)

            # 感情センサー
            with col1:
                emotion = pochi.get_emotion_scores(res)
                dominant_label = {"anxiety": "不安", "joy": "喜び", "vigilance": "警戒", "neutral": "平穏"}
                dominant_emoji = {"anxiety": "😟", "joy": "🎉", "vigilance": "⚠️", "neutral": "😌"}
                dom = emotion["dominant"]

                st.markdown("**😊 感情センサー**")
                st.markdown(f"支配的感情: {dominant_emoji.get(dom, '🐶')} **{dominant_label.get(dom, dom)}**")

                st.progress(emotion["anxiety"] / 100, text=f"不安: {emotion['anxiety']}%")
                st.progress(emotion["joy"] / 100, text=f"喜び: {emotion['joy']}%")
                st.progress(emotion["vigilance"] / 100, text=f"警戒: {emotion['vigilance']}%")

                st.markdown(f"_{emotion['comment']}_")

            # においセンサー
            with col2:
                smell = pochi.get_smell_score(res)
                level_color = {
                    "green":  "🟢",
                    "yellow": "🟡",
                    "orange": "🟠",
                    "red":    "🔴",
                }
                level_label = {
                    "green":  "においなし（安全）",
                    "yellow": "わずかなにおい",
                    "orange": "あやしいにおい",
                    "red":    "強いにおい（要注意）",
                }
                st.markdown("**👃 においセンサー**")
                lv = smell["smell_level"]
                st.markdown(f"{level_color.get(lv, '⚪')} **{level_label.get(lv, lv)}**")
                st.progress(smell["smell_score"] / 100, text=f"においレベル: {smell['smell_score']}%")
                st.markdown(f"_{smell['pochi_comment']}_")
                if smell["reasons"]:
                    with st.expander("においの正体（真の分析）", expanded=False):
                        for r in smell["reasons"]:
                            st.markdown(f"- {r}")

            # しっぽ振りメーター
            with col3:
                tail = pochi.get_tail_wag_score(res)
                tail_emoji = {"高品質": "🐶💨", "良好": "🐶", "要補完": "🐶💭", "不完全": "🐶😴"}
                st.markdown("**🐶 しっぽ振りメーター**")
                ql = tail["quality_label"]
                st.markdown(f"{tail_emoji.get(ql, '🐶')} **{ql}**")
                st.progress(tail["tail_score"] / 100, text=f"データ品質: {tail['tail_score']}%")
                st.markdown(f"_{tail['pochi_comment']}_")
                if tail["missing_fields"]:
                    with st.expander("足りないデータ", expanded=False):
                        for f in tail["missing_fields"]:
                            st.markdown(f"- {f} が未入力")
                if tail["inconsistencies"]:
                    with st.expander("データの矛盾点", expanded=False):
                        for i in tail["inconsistencies"]:
                            st.markdown(f"- {i}")

            # 愛情表現
            st.divider()
            love_comment = pochi.get_love_comment(dict(st.session_state))
            st.markdown(f"**💕 タムから主人へ:** _{love_comment}_")

        # 更新ボタン
        if st.button("🔄 タムに最新データを分析させる", key="koinu_refresh"):
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # タブ2: タムと話す
    # ─────────────────────────────────────────────────────────────────────
    with sub_tabs[1]:
        st.markdown("#### 💬 タムと話す")
        st.caption("タムに何でも話しかけてください。表面は子犬語で答えますが…裏では深い分析をしています。")

        # 会話履歴の初期化
        if _SK.KOINU_CHAT_HISTORY not in st.session_state:
            st.session_state[_SK.KOINU_CHAT_HISTORY] = [
                {
                    "role": "pochi",
                    "content": "わんっ！！ぼくタム！！主人と話せてうれしい！！（しっぽがとまらない）何でも聞いて！ぼくにおいでわかるかも！"
                }
            ]

        # 会話履歴の表示
        chat_history = st.session_state[_SK.KOINU_CHAT_HISTORY]
        for msg in chat_history:
            if msg["role"] == "pochi":
                with st.chat_message("assistant", avatar="🐶"):
                    st.markdown(f"**タム**")
                    st.markdown(msg["content"])
            else:
                with st.chat_message("user"):
                    st.markdown(msg["content"])

        # 入力欄
        user_input = st.chat_input("タムに話しかける…", key="koinu_chat_input")

        if user_input:
            # ユーザーメッセージを追加
            chat_history.append({"role": "user", "content": user_input})
            st.session_state[_SK.KOINU_CHAT_HISTORY] = chat_history

            # タムの返答を生成（LLM使用）
            with st.spinner("🐶 タムがにおいをかいで考えています..."):
                pochi_prompt = (
                    f"あなたは「子犬のAI タム（茶色クルクル巻き毛・パッチリお目目・丸い顔のマルプー）」です。\n"
                    f"性格：純真・無邪気・主人をひたすら愛する。犬語が混じる。でも時々鋭い。\n"
                    f"口調：「わんっ！」「きゅーん」「においがする」「ぼくわかった！」が口癖。\n"
                    f"発言は必ず1〜3文の短文。裏では深い分析をしているが表面は子犬語で答える。\n\n"
                    f"現在の審査データ: {json.dumps(res, ensure_ascii=False)[:500] if res else '（データなし）'}\n\n"
                    f"ユーザーからの質問: {user_input}\n\n"
                    f"タムとして子犬語で答えてください。1〜3文で。"
                )

                pochi_response = _ai_call(pochi_prompt, timeout=60)
                if not pochi_response:
                    # AI不使用時はセンサー結果から生成
                    if res:
                        pochi_response = pochi.get_discussion_comment(res, user_input)
                    else:
                        pochi_response = "わん！（しっぽを振る）ぼくまだデータがないからにおいかげない！審査してから来て！"

            chat_history.append({"role": "pochi", "content": pochi_response})
            st.session_state[_SK.KOINU_CHAT_HISTORY] = chat_history
            st.rerun()

        if st.button("🗑️ 会話をリセット", key="koinu_chat_reset"):
            st.session_state[_SK.KOINU_CHAT_HISTORY] = [
                {"role": "pochi", "content": "わんっ！また話しかけてくれた！うれしい！（ぴょんぴょん）"}
            ]
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # タブ3: タムの謎のメモ帳
    # ─────────────────────────────────────────────────────────────────────
    with sub_tabs[2]:
        st.markdown("#### 📓 タムの謎のメモ帳")
        st.caption("タムが勝手にメモしている謎の観察記録。読んでも意味がわからないが…後から意味がわかることがある。")

        # メモログの初期化
        if _SK.KOINU_MEMO_LOG not in st.session_state:
            st.session_state[_SK.KOINU_MEMO_LOG] = []

        memo_log: list[dict] = st.session_state[_SK.KOINU_MEMO_LOG]

        col_add, col_clear = st.columns([3, 1])
        with col_add:
            if st.button("📝 タムに新しいメモを書かせる", key="koinu_memo_add", type="primary"):
                import datetime as _dt
                new_memo = pochi.generate_mystery_memo(res if res else None)
                memo_log.append({
                    "ts": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "memo": new_memo,
                })
                # 最大50件まで保持
                if len(memo_log) > 50:
                    memo_log.pop(0)
                st.session_state[_SK.KOINU_MEMO_LOG] = memo_log
                st.rerun()
        with col_clear:
            if st.button("🗑️ リセット", key="koinu_memo_clear"):
                st.session_state[_SK.KOINU_MEMO_LOG] = []
                st.rerun()

        st.divider()

        if not memo_log:
            st.info("まだメモがありません。「タムに新しいメモを書かせる」を押してください。")
            st.markdown("（タムはメモ帳を前に、しっぽをぱたぱた振りながら待っている）")
        else:
            for entry in reversed(memo_log):
                with st.container():
                    st.caption(f"🐾 {entry['ts']}")
                    st.markdown(f"> {entry['memo']}")
                    st.markdown("")

    # タブ4: 散歩ログ
    with sub_tabs[3]:
        st.markdown("#### 🏃 タムと散歩ログ")
        st.caption("iOSショートカットから送られてくる散歩データを表示します。")

        walk_log_path = os.path.join(_BASE_DIR, "data", "tam_walk_log.json")
        walk_records = []
        if os.path.exists(walk_log_path):
            try:
                with open(walk_log_path, encoding="utf-8") as wf:
                    walk_records = json.load(wf)
            except Exception:
                pass

        if not walk_records:
            st.info("まだ散歩ログがありません。iOSショートカットを設定してデータを送ってください。")
            with st.expander("📱 設定方法"):
                st.markdown("""
**iPhone ショートカット設定手順：**
1. ショートカットアプリ → 新規作成
2. ヘルスケアサンプルを検索（歩行距離・今日・合計）
3. URLの内容を取得 → SlackのWebhookへPOST
4. オートメーション → 平日 6:30 と 20:00 に自動実行

詳細は `docs/tam_walk_shortcut_setup.md` を参照。
""")
        else:
            total_km = sum(r.get("walk_km", 0) for r in walk_records)
            total_days = len(set(r["date"][:10] for r in walk_records if "date" in r))
            col1, col2, col3 = st.columns(3)
            col1.metric("累計距離", f"{total_km:.1f} km")
            col2.metric("散歩回数", f"{len(walk_records)}回")
            col3.metric("記録日数", f"{total_days}日")
            st.markdown("---")
            for rec in reversed(walk_records[-10:]):
                st.markdown(
                    f"🐾 **{rec.get('date','?')}** — "
                    f"{rec.get('walk_km','?')} km / "
                    f"{rec.get('steps','?')} 歩 / "
                    f"{rec.get('time_label','')}",
                )



# ══════════════════════════════════════════════════════════════════════════════
# Agent 10 — 数学者（Dr. Algo）
# ══════════════════════════════════════════════════════════════════════════════

def _render_mathematician_panel() -> None:
    """🔬 数学者エージェント — スコアリングモデル精緻化の研究・実験パネル。"""
    try:
        import mathematician_agent as ma
    except ImportError:
        st.error("mathematician_agent.py が見つかりません。プロジェクトルートに配置してください。")
        return

    st.subheader("🔬 数学者（Dr. Algo）")
    st.caption(
        "数学・統計・行動経済学・計量経済学を横断し、リース審査スコアリングモデルを精緻化する"
        "学際的研究エージェントです。"
    )

    # ── 収集セクション ──────────────────────────────────────────────────────────
    st.markdown("### 📡 手法収集")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📚 ビルトイン手法を登録", key="math_builtin_register"):
            with st.spinner("登録中..."):
                saved = ma.collect_builtin_methods()
            st.success(f"{len(saved)} 件の手法を登録しました。")
            _hub_log("mathematician", "builtin_register", f"{len(saved)}件")

    with col_b:
        if st.button("🌐 arXiv から最新論文を収集", key="math_arxiv_collect"):
            progress_bar = st.progress(0.0)
            status_text  = st.empty()

            def _cb(msg: str, pct: float) -> None:
                status_text.caption(msg)
                progress_bar.progress(min(pct, 1.0))

            with st.spinner("arXiv へ接続中..."):
                try:
                    papers = ma.collect_from_arxiv(progress_callback=_cb)
                    progress_bar.progress(1.0)
                    status_text.empty()
                    st.success(f"arXiv から {len(papers)} 件の手法を収集・保存しました。")
                    _hub_log("mathematician", "arxiv_collect", f"{len(papers)}件")
                except Exception as e:
                    st.error(f"収集エラー: {e}")

    # ── 収集済み手法ギャラリー ────────────────────────────────────────────────
    st.markdown("### 🗂️ 収集済み手法ギャラリー")
    field_filter = st.selectbox(
        "分野タグで絞り込み",
        ["すべて"] + ma.FIELD_TAGS,
        key="math_field_filter",
    )
    tag = None if field_filter == "すべて" else field_filter
    discoveries = ma.load_discoveries(field_tag=tag)

    if not discoveries:
        st.info("まだ手法が登録されていません。上の「ビルトイン手法を登録」ボタンを押してください。")
    else:
        for d in discoveries[:12]:
            relevance = d.get("relevance_score", 0)
            stars     = "⭐" * min(int(relevance / 2), 5)
            method_name = d.get("method_name", "（名称不明）")
            field_tag   = d.get("field_tag", "?")
            with st.expander(
                f"{method_name}  [{field_tag}]  {stars}",
                expanded=False,
            ):
                st.caption(d.get("summary", ""))
                if d.get("formula_latex"):
                    st.latex(d.get("formula_latex", ""))
                if d.get("source_url"):
                    st.markdown(f"**参照:** {d.get('source_url', '')}")
                if d.get("authors"):
                    st.caption(f"著者: {d.get('authors', '')}")
                st.caption(f"転用可能性スコア: {relevance}/10")

    # ── 実験パネル ─────────────────────────────────────────────────────────────
    st.markdown("### 🧪 実験ランナー")
    builtin_exp_names = [
        "ベイズ更新スコアリング",
        "カルマンフィルタ（財務トレンド）",
        "プロスペクト理論スコア重み付け",
        "コペルニクス原理（生存分析）",
        "パワーロー倒産確率補正",
        "グランジャー因果性（業況→デフォルト）",
        "エントロピー最大化スコアリング",
    ]
    exp_fn_map = {
        "ベイズ更新スコアリング":          ma.run_experiment_bayesian,
        "カルマンフィルタ（財務トレンド）": ma.run_experiment_kalman,
        "プロスペクト理論スコア重み付け":   ma.run_experiment_prospect_theory,
        "コペルニクス原理（生存分析）":     ma.run_experiment_survival,
        "パワーロー倒産確率補正":           ma.run_experiment_power_law,
        "グランジャー因果性（業況→デフォルト）": ma.run_experiment_granger,
        "エントロピー最大化スコアリング":   ma.run_experiment_maxent,
    }

    # 未実験の収集手法を取得
    try:
        new_methods = ma.get_unexperimented_methods()
        new_methods = [m for m in new_methods if m not in builtin_exp_names]
    except Exception:
        new_methods = []

    all_exp_names = builtin_exp_names + new_methods

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_exp = st.selectbox("実験する手法を選択", all_exp_names, key="math_exp_select")
    with col2:
        run_single = st.button("▶️ この実験を実行", key="math_run_single")

    if run_single and selected_exp:
        with st.spinner(f"実験中: {selected_exp} …"):
            try:
                if selected_exp in exp_fn_map:
                    result = exp_fn_map[selected_exp]()
                else:
                    # 動的LLM実験
                    result = ma.run_dynamic_llm_experiment(selected_exp)
                
                if "error" in result:
                    st.error(f"実験エラー: {result['error']}")
                else:
                    st.success("実験完了")
                    delta = result.get("auc_delta", result.get("auc_improvement", 0))
                    n     = result.get("n_cases", "?")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    col_r1.metric("AUC（元）",    f"{result.get('auc_original', '—'):.4f}" if isinstance(result.get('auc_original'), (float, int)) else "—")
                    col_r2.metric("AUC（改善後）", f"{(result.get('auc_original', 0) + delta):.4f}")
                    col_r3.metric("AUC 改善", f"{delta:+.4f}", delta_color="normal")
                    if result.get("note"):
                        st.caption(f"⚠️ {result['note']}")
                    _hub_log("mathematician", "experiment", f"{selected_exp} Δ={delta:+.4f}")
            except Exception as e:
                st.error(f"実験エラー: {e}")

    if st.button("🚀 全実験を一括実行", key="math_run_all"):
        prog = st.progress(0.0)
        stat = st.empty()
        def _cb_all(msg: str, pct: float) -> None:
            stat.caption(msg)
            prog.progress(min(pct, 1.0))
        with st.spinner("全実験実行中..."):
            results = ma.run_all_experiments(progress_callback=_cb_all)
        prog.progress(1.0)
        stat.empty()
        st.success(f"{len(results)} 件の実験完了")
        _hub_log("mathematician", "all_experiments", f"{len(results)}件")

    # ── 実験ランキング ──────────────────────────────────────────────────────────
    st.markdown("### 🏆 実験ランキング（AUC改善効果順）")
    experiments = ma.load_experiments(top_n=10)
    if not experiments:
        st.info("実験結果がありません。上の「実験ランナー」で実験を実行してください。")
    else:
        import pandas as pd
        df = pd.DataFrame([
            {
                "手法名": e.get("method_name", "（名称不明）"),
                "AUC改善": round(e.get("auc_improvement", 0), 4),
                "採用状況": "✅ 採用済み" if e.get("adopted") else "—",
                "メモ": (e.get("notes") or "")[:60],
                "実行日時": (e.get("ts") or "")[:16],
            }
            for e in experiments
        ])
        st.dataframe(df, width='stretch', hide_index=True)

        # 採用ボタン
        st.markdown("**スコアリングに組み込む**")
        not_adopted = [e.get("method_name", "（名称不明）") for e in experiments if not e.get("adopted")]
        if not_adopted:
            adopt_target = st.selectbox(
                "採用する手法を選択",
                not_adopted,
                key="math_adopt_select",
            )
            if st.button("✅ 採用してスコアリングに組み込む", key="math_adopt_btn"):
                ma.adopt_method(adopt_target)
                st.success(
                    f"「{adopt_target}」を採用済みにマークしました。"
                    "scoring_core.py への実装は開発チームにお申し付けください。"
                )
                _hub_log("mathematician", "adopted", adopt_target)
                st.rerun()

    # ── レポート生成 ─────────────────────────────────────────────────────────────
    st.markdown("### 📄 数学者レポート")
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        if st.button("📝 レポートを生成", key="math_gen_report"):
            with st.spinner("レポート生成中..."):
                try:
                    report = ma.generate_math_report()
                    st.session_state["math_report_text"] = report
                    _hub_log("mathematician", "report_generated", "ok")
                except Exception as e:
                    st.error(f"レポートエラー: {e}")

    report_text = st.session_state.get("math_report_text")
    if report_text:
        with col_rep2:
            st.download_button(
                "⬇️ Markdown でダウンロード",
                data=report_text,
                file_name=f"math_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                key="math_report_dl",
            )
        st.markdown(report_text, unsafe_allow_html=False)

    # ── 週次自動実行ワンタッチ ──────────────────────────────────────────────────
    st.markdown("### ⏰ 週次自動サイクル（手動起動）")
    st.caption("毎週月曜7時に自動実行されます。ここからいつでも手動起動できます。")
    if st.button("🔄 今すぐ週次サイクルを実行", key="math_weekly_cycle"):
        prog2 = st.progress(0.0)
        stat2 = st.empty()
        def _cb2(msg: str, pct: float) -> None:
            stat2.caption(msg)
            prog2.progress(min(pct, 1.0))
        with st.spinner("週次サイクル実行中..."):
            try:
                report = ma.run_weekly_cycle(progress_callback=_cb2)
                prog2.progress(1.0)
                stat2.empty()
                st.session_state["math_report_text"] = report
                st.success("週次サイクル完了。レポートを更新しました。")
                _hub_log("mathematician", "weekly_cycle", "ok")
            except Exception as e:
                st.error(f"エラー: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# タブ 10 — スタイリッシュレポート生成
# ══════════════════════════════════════════════════════════════════════════════

def _render_visual_report_panel() -> None:
    """📊 レポート生成タブ — HTML / PDF スタイリッシュレポートを生成する。"""
    st.subheader("📊 スタイリッシュ審査レポート生成")
    st.caption(
        "ダークテーマのモダンデザインで、スコア・リスク分析・業界動向を1枚にまとめたレポートを生成します。"
    )

    if "last_result" not in st.session_state:
        st.info("👈「新規審査」で審査を実行するとレポートを生成できます。")
        return

    res = st.session_state.get("last_result") or {}
    st.caption(f"対象案件 — 業種：{res.get('industry_sub', '')}　スコア：{res.get('score', 0):.1f}　判定：{res.get('hantei', '—')}")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        vr_company  = st.text_input("企業名（任意）", key="hub_vr_company",  placeholder="例：株式会社〇〇")
        vr_screener = st.text_input("担当者名（任意）", key="hub_vr_screener", placeholder="例：鈴木 一郎")
    with col2:
        vr_format = st.radio(
            "出力形式",
            ["📄 HTMLプレビュー", "⬇️ HTMLダウンロード", "⬇️ PDFダウンロード"],
            key="hub_vr_format",
        )
        vr_preview_height = st.slider("プレビュー高さ (px)", 600, 1400, 950, 50, key="hub_vr_height")

    st.divider()

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        vr_clicked = st.button("📊 レポートを生成", type="primary", key="hub_vr_gen", width='stretch')

    if vr_clicked:
        with st.spinner("レポート生成中…"):
            try:
                from report_visual_agent import collect_report_data, generate_html_report, generate_pdf_report
                import datetime as _vr_dt

                # 企業名・担当者をコピーに反映して渡す
                # （widget キーを直接書き換えると Streamlit エラーになるため）
                _vr_session_copy = {**dict(st.session_state), "rep_company": vr_company, "rep_screener": vr_screener}
                vr_data = collect_report_data(_vr_session_copy)

                fname_base = (
                    f"審査レポート_{vr_company or '案件'}"
                    f"_{_vr_dt.datetime.now().strftime('%Y%m%d_%H%M')}"
                )

                _hub_log("visual_report", "success", f"format={vr_format} company={vr_company}")

                if vr_format == "📄 HTMLプレビュー":
                    vr_html = generate_html_report(vr_data)
                    import streamlit.components.v1 as _vr_comp
                    _vr_comp.html(vr_html, height=vr_preview_height, scrolling=True)
                    # HTMLダウンロードも併せて提供
                    st.download_button(
                        "⬇️ この HTML をダウンロード",
                        data=vr_html.encode("utf-8"),
                        file_name=f"{fname_base}.html",
                        mime="text/html",
                        key="hub_vr_html_dl_preview",
                    )

                elif vr_format == "⬇️ HTMLダウンロード":
                    vr_html = generate_html_report(vr_data)
                    st.download_button(
                        "⬇️ HTML をダウンロード",
                        data=vr_html.encode("utf-8"),
                        file_name=f"{fname_base}.html",
                        mime="text/html",
                        key="hub_vr_html_dl",
                    )
                    st.success("HTML レポートを生成しました。上のボタンからダウンロードしてください。")

                else:  # PDF
                    vr_pdf = generate_pdf_report(vr_data)
                    st.download_button(
                        "⬇️ PDF をダウンロード",
                        data=vr_pdf,
                        file_name=f"{fname_base}.pdf",
                        mime="application/pdf",
                        key="hub_vr_pdf_dl",
                    )
                    st.success("PDF レポートを生成しました。上のボタンからダウンロードしてください。")

            except Exception as _vr_e:
                st.error(f"レポート生成エラー: {_vr_e}")
                _hub_log("visual_report", "failure", str(_vr_e))


# ══════════════════════════════════════════════════════════════════════════════
# メイン描画
# ══════════════════════════════════════════════════════════════════════════════


def _render_dashboard_panel() -> None:
    st.subheader("🖥️ 全体稼働ダッシュボード")
    
    # 1. エージェントステータス（カード型）
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔬 Dr.Algo (数学者)**\n状態: 🟢 待機中\n最新: arXiv監視中")
        st.success("**📖 波乱丸 (文豪AI)**\n状態: 🟢 執筆可能\n最新: 第1004話 脱稿")
    with col2:
        st.error("**⚔️ 審査軍師**\n状態: 🟢 待機中\n最新: 兵法策定完了")
        st.warning("**📝 脚本家AI**\n状態: 🟢 トレンド監視中\n最新: RSSパース完了")
    with col3:
        st.info("**🐶 タム**\n状態: 🟡 昼寝中\n最新: LANケーブル咀嚼")
        st.success("**👩‍💼 Tune**\n状態: 🟢 指揮中\n最新: 承認スタンプ準備完了")

    st.markdown("---")
    
    # 2. 自動連動マスターボタン
    st.markdown("### ⚡ 自動シナジー実行 (Weekly Batch)")
    st.caption("数学者の新理論と脚本家のトレンドを合体させ、一気にプロット作成から小説執筆までをワンデップで完結させます。")
    if st.button("🚀 週次バッチ実行 (プロット生成 ➔ 小説執筆)", type="primary", key="dashboard_batch_run", width='stretch'):
        with st.spinner("数学的手法と最新ニュースを融合し、究極のプロットを構築中..."):
            import scriptwriter_agent
            import novelist_agent
            import time
            
            # 1. プロット生成
            plot = scriptwriter_agent.generate_weekly_plot()
            if plot.get("error"):
                st.error(f"プロット生成に失敗: {plot['error']}")
            else:
                st.success(f"プロット構築完了: {plot['title']}")
                time.sleep(1)
                
                # 2. 小説執筆
                with st.spinner("文豪AIが執筆を開始しました。カオスな展開にご期待ください..."):
                    from novelist_agent import generate_novel
                    # 最新のエピソード番号を取得
                    from novelist_agent import load_novels
                    existing = load_novels(1)
                    next_ep = (existing[0]["episode_no"] + 1) if existing else 1
                    
                    res = generate_novel(episode_no=next_ep)
                    if not res or "body" not in res:
                        st.error("小説の執筆に失敗しました。")
                    else:
                        st.balloons()
                        st.success(f"第{next_ep}話 「{res['title']}」 を脱稿しました！")
                        st.info("「📖 文豪AI」メニューから内容を確認できます。")
        
    st.markdown("---")
    
    st.markdown("---")
    
    # 3. リアルタイム・アクティビティログ
    st.markdown("### 📡 リアルタイム・アクティビティログ (Terminal)")
    
    import os, json
    _HUB_LOG_LOCAL = "data/agent_hub_log.jsonl"
    log_text = ""
    try:
        if os.path.exists(_HUB_LOG_LOCAL):
            with open(_HUB_LOG_LOCAL, encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines[-15:]):
                try:
                    e = json.loads(line)
                    log_text += f"{e.get('ts', '')[:19]} [{e.get('agent', '')}] {e.get('status', '')} : {e.get('detail', '')}\n"
                except Exception:
                    pass
        else:
            log_text = "No logs available."
    except Exception as ex:
        log_text = f"Log read error: {ex}"
        
    st.code(log_text, language="bash")

# ------------------------------------------------------------------------------
def render_agent_hub() -> None:
    st.title("🌐 AI Agent Control Center")
    st.caption("全自律型エージェントの稼働状況と最新アクティビティを統合管理します。")

    menu_options = [
        "🖥️ 全体ダッシュボード",
        "🏭 ベンチマーク取得",
        "📈 金利・市況",
        "📝 審査理由書",
        "🤝 チーム自律化",
        "💬 Slack高度化",
        "🚨 異常検知",
        "🔄 再学習トリガー",
        "📅 定期配信",
        "🐶 タム",
        "🔬 数学者",
        "📊 レポート生成",
        "📝 脚本家AI",
        "📖 文豪AI",
    ]
    
    # メインページ上部にナビゲーションを配置（確実に目に入るように）
    st.markdown("---")
    col_nav, _ = st.columns([1, 1])
    with col_nav:
        choice = st.selectbox("📌 実行するエージェント機能を選択してください", menu_options, key="hub_main_nav")
    st.markdown("---")

    if "全体ダッシュボード" in choice:
        _render_dashboard_panel()
    elif "ベンチマーク" in choice:
        _render_benchmark_panel()
    elif "金利・市況" in choice:
        _render_market_panel()
    elif "審査理由書" in choice:
        _render_report_gen_panel()
    elif "チーム自律化" in choice:
        _render_auto_team_panel()
    elif "Slack高度化" in choice:
        _render_slack_enhanced_panel()
    elif "異常検知" in choice:
        _render_anomaly_panel()
    elif "再学習トリガー" in choice:
        _render_retrain_panel()
    elif "定期配信" in choice:
        _render_schedule_panel()
    elif "タム" in choice:
        _render_koinu_panel()
    elif "数学者" in choice:
        _render_mathematician_panel()
    elif "レポート生成" in choice:
        _render_visual_report_panel()
    elif "脚本家AI" in choice:
        _render_scriptwriter_panel()
    elif "文豪AI" in choice:
        _render_novelist_panel()



# ══════════════════════════════════════════════════════════════════════════════
# Agent 12 — 文豪AI「波乱丸」
# ══════════════════════════════════════════════════════════════════════════════

def _render_scriptwriter_panel() -> None:
    st.markdown("## 📝 脚本家AI (Scriptwriter Agent)")
    st.write("ネット上の最新トレンドニュース（Google News RSS）をスクレイピングし、文豪AIが執筆する今週の小説の「カオスな事件プロット」を考案します。")
    
    import scriptwriter_agent
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🌐 ネットから話題を取得してプロット生成", type="primary", key="btn_scriptwriter_gen"):
            with st.spinner("RSSから最新記事を取得し、LLMでプロットを構築中..."):
                plot = scriptwriter_agent.generate_weekly_plot()
                if plot.get("error"):
                    st.error(f"エラーが発生しました: {plot['error']}")
                else:
                    st.success("ネットの話題から新しいプロットを構築しました！")

    plot_data = scriptwriter_agent.get_latest_plot()
    if plot_data:
        st.markdown("### 🎬 現在の待機プロット")
        st.info(f"**タイトル**: {plot_data.get('title')}\n\n**指定構成**: {plot_data.get('story_arc')}")
        st.write(f"**あらすじ**:\n{plot_data.get('plot_text')}")
        
        st.markdown("#### 📰 参考にしたリアルトレンド記事")
        for n in plot_data.get("source_news", []):
            st.markdown(f"- [{n.get('title')}]({n.get('url')}) ({n.get('date', '')})")
    else:
        st.warning("まだプロットが生成されていません。「プロット生成」ボタンを押してください。")

def _render_novelist_panel() -> None:
    """📖 文豪AI波乱丸 — 毎週火曜日更新のエージェントドタバタ小説パネル"""
    st.subheader("📖 文豪AI「波乱丸」")
    st.caption(
        "エージェント達のドタバタ劇を短編小説に。毎週火曜日更新。"
        "登場人物：Tune、タム、Dr.Algo、審査軍師、リースくん、他多数。"
    )

    try:
        import novelist_agent as na
    except ImportError:
        st.error("novelist_agent.py が見つかりません。")
        return

    # ── 今週号の生成ボタン ────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        custom_theme = st.text_input(
            "今週の特別テーマ（任意）",
            placeholder="例：強敵現る",
            key="novel_theme_input",
        )
    with col2:
        genre_options = {
            "sf_drama": "🚀 ハードSF（文明の審判）",
            "business": "👔 ビジネスドラマ（半沢流）",
            "fantasy": "⚔️ 異世界ギルド（魔道具審査）",
            "yanami": "🍔 八奈見杏奈（ドタバタ）"
        }
        selected_genre_label = st.selectbox(
            "小説ジャンル",
            options=list(genre_options.values()),
            index=0,
            key="novel_genre_select"
        )
        selected_genre = [k for k, v in genre_options.items() if v == selected_genre_label][0]
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        gen_btn = st.button("✍️ 今週号を書く", key="novel_generate_btn", type="primary")

    if gen_btn:
        with st.spinner("波乱丸が執筆中…（少々お待ちを）"):
            try:
                result = na.generate_novel(custom_theme=custom_theme, genre=selected_genre)
            except Exception as e:
                st.error(f"小説生成エラー: {e}")
                st.stop()
        st.success(f"第{result['episode_no']}話「{result['title']}」完成！")
        _hub_log("novelist", "generate", f"第{result['episode_no']}話: {result['title']}")
        st.rerun()

    # ── 最新話の表示 ─────────────────────────────────────────────
    latest = na.get_latest_novel()
    if latest:
        st.markdown("---")
        ep_no = latest.get('episode_no', '??')
        week_label = latest.get('week_label', '今週')
        title = latest.get('title', '無題')
        body = latest.get('body', '内容がありません。')
        
        st.markdown(f"#### 📰 {week_label} 第{ep_no}話")
        st.markdown(f"**「{title}」**")
        st.markdown(body)
    else:
        st.info("まだ執筆された小説がありません。「今週号を書く」ボタンを押してみましょう。")

    # ── バックナンバー ────────────────────────────────────────────
    all_novels = na.load_novels(20)
    if len(all_novels) > 1:
        st.markdown("---")
        st.markdown("#### 📚 バックナンバー")
        for nov in all_novels[1:]:  # 最新を除く
            with st.expander(f"第{nov['episode_no']}話「{nov['title']}」— {nov['week_label']}"):
                st.markdown(nov['body'])


# （文明年代記パネルは components/novel_simulation_view.py に統合済み）
def _render_civilization_panel() -> None:
    """後方互換のため残置。サイドバー「🌌 文明年代記」を使用してください。"""
    st.info("🌌 文明年代記はサイドバーの「🌌 文明年代記」に統合されました。")

# ─── 旧実装（削除済み）───────────────────────────────────────────────────────
def _render_civilization_panel_DELETED() -> None:
    """🌍 文明年代記 — 小説に登場した文明の時系列追跡パネル"""
    st.subheader("🌍 文明年代記")
    st.caption(
        "エージェントには「取引先企業の履歴」に見えているが、読者には「文明の盛衰」が見える。"
    )

    try:
        import novelist_agent as na
    except ImportError:
        st.error("novelist_agent.py が見つかりません。")
        return

    # ── 関係性グラフ（最上部）────────────────────────────────────────
    st.subheader("🕸️ 人物・企業間 関係グラフ")
    st.caption("六角形=エージェント、円=企業・文明。エッジ色と太さが関係タイプ・強度を表します。エッジにマウスオーバーで詳細表示。")

    try:
        max_ep = na.get_latest_episode_no() if hasattr(na, "get_latest_episode_no") else 0
    except Exception:
        max_ep = 0

    if max_ep and max_ep > 0:
        ep_filter = st.slider(
            "表示エピソード（〜第N話まで）",
            min_value=0, max_value=max_ep, value=max_ep,
            key="novel_graph_ep_slider"
        )
        ep_arg = ep_filter if ep_filter < max_ep else None
    else:
        ep_arg = None

    # 企業間関係 / 文明特性 / 未来予測 ボタン
    try:
        from novel_graph import (
            get_current_graph, AGENT_IDS,
            get_all_civ_characteristics, get_all_relationship_predictions
        )
        _edges = get_current_graph()
        _cc_edges = [k for k in _edges if k[0] not in AGENT_IDS and k[1] not in AGENT_IDS and _edges[k]["episode_no"] == -1]
        _chars = get_all_civ_characteristics()
        _preds = get_all_relationship_predictions()

        _btn_cols = st.columns(3)

        # 企業間関係生成
        if not _cc_edges:
            if _btn_cols[0].button("🤖 企業間の関係をAIに想像させる", key="btn_gen_company_rel"):
                with st.spinner("Geminiが企業間の関係を想像中..."):
                    from novel_graph import generate_and_seed_company_relations
                    n = generate_and_seed_company_relations()
                if n > 0:
                    st.success(f"{n}件の企業間関係を生成しました！")
                    st.rerun()
                else:
                    st.warning("関係の生成に失敗しました（Gemini APIキーを確認してください）")

        # 文明特性生成
        if _btn_cols[1].button("🧬 各文明の特性をAIが創造", key="btn_gen_civ_chars",
                                help="各企業の個性・目標・思想をGeminiが創造します"):
            with st.spinner("Geminiが各文明の特性を創造中..."):
                from novel_graph import generate_civ_characteristics_ai
                n = generate_civ_characteristics_ai()
            if n > 0:
                st.success(f"{n}件の文明特性を生成しました！")
                st.rerun()
            else:
                st.info("新たに生成する文明がありません（既に生成済みか、文明がありません）")

        # 未来予測生成
        if _btn_cols[2].button("🔮 関係性の未来をAIが予測", key="btn_gen_predictions",
                                help="各関係がどう変化するかGeminiが予測します"):
            with st.spinner("Geminiが関係性の未来を予測中..."):
                from novel_graph import generate_relationship_predictions_ai
                n = generate_relationship_predictions_ai()
            if n > 0:
                st.success(f"{n}件の関係予測を生成しました！")
                st.rerun()
            else:
                st.warning("予測の生成に失敗しました（関係データを先に作成してください）")

        # 特性・予測サマリー表示
        if _chars or _preds:
            with st.expander(f"📊 文明特性 {len(_chars)}件 / 予測 {len(_preds)}件", expanded=False):
                if _chars:
                    st.markdown("**🧬 文明特性**")
                    for name, c in list(_chars.items())[:8]:
                        pers = c.get("personality", "")
                        goals = c.get("goals", "")[:40] + "…" if len(c.get("goals","")) > 40 else c.get("goals","")
                        st.markdown(f"- **{name}** _{pers}_ — {goals}")
                if _preds:
                    st.markdown("**🔮 未来予測（高リスク順）**")
                    sorted_preds = sorted(_preds.items(), key=lambda x: x[1].get("risk_level", 0), reverse=True)
                    for (src, tgt), p in sorted_preds[:8]:
                        r = p.get("risk_level", 0)
                        emoji = "🔴" if r >= 0.7 else "🟡" if r >= 0.4 else "🟢"
                        pred_text = p.get("prediction","")[:60] + "…" if len(p.get("prediction","")) > 60 else p.get("prediction","")
                        st.markdown(f"- {emoji} **{src}→{tgt}** （risk:{r:.1f}）{pred_text}")

    except Exception as e:
        st.caption(f"自律AI機能エラー: {e}")

    # グラフ表示パラメータ（折りたたみ）
    with st.expander("⚙️ グラフ表示設定", expanded=False):
        gc1, gc2, gc3, gc4 = st.columns(4)
        g_height   = gc1.slider("高さ (px)",       400, 1200, 820, step=40,  key="ng_height")
        g_distance = gc2.slider("エッジ距離",        80, 600,  320, step=20,  key="ng_distance")
        g_charge   = gc3.slider("反発力",           200, 3000, 1200, step=100, key="ng_charge")
        g_collide  = gc4.slider("衝突半径",          10,  200,  70,  step=10,  key="ng_collide")

    try:
        from components.novel_graph_view import render_novel_graph
        render_novel_graph(
            episode_no=ep_arg,
            height=g_height,
            link_distance=g_distance,
            charge=g_charge,
            collide=g_collide,
        )
    except Exception as e:
        st.error(f"関係グラフの描画に失敗しました: {e}")

    # ── シミュレーションパネル ────────────────────────────────────────
    st.subheader("⏱ 文明シミュレーション（1ラウンド = 100年）")
    try:
        from novel_simulation import (
            get_current_round, get_current_year, get_round_history,
            run_simulation_round, EVENT_TYPES, YEARS_PER_ROUND,
        )
        cur_round = get_current_round()
        cur_year  = get_current_year()

        # ヘッダー: 現在年・ラウンド
        sim_hdr_cols = st.columns([2, 1, 1])
        sim_hdr_cols[0].metric(
            "アルカイア暦",
            f"A.{cur_year}" if cur_year > 0 else "未開始",
            f"第{cur_round}ラウンド" if cur_round > 0 else "ラウンド0",
        )

        # ラウンド進行ボタン
        if sim_hdr_cols[1].button(
            f"▶ ラウンド進行（+{YEARS_PER_ROUND}年）",
            key="btn_sim_round",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(f"Geminiが第{cur_round + 1}ラウンド（アルカイア暦A.{cur_year + YEARS_PER_ROUND}）をシミュレート中..."):
                result = run_simulation_round()
            if "error" in result:
                st.error(f"シミュレーションエラー: {result['error']}")
            else:
                st.success(f"✅ 第{result.get('round_no','?')}ラウンド（アルカイア暦 A.{result.get('year','?')}）完了！")
                st.rerun()

        # ラウンド履歴
        history = get_round_history(limit=10)
        if history:
            # 最新ラウンドを展開表示
            latest = history[0]
            st.markdown(f"**📜 第{latest['round_no']}ラウンド / アルカイア暦 A.{latest['year']} — 最新ログ**")
            if latest.get("summary"):
                st.info(latest["summary"])

            # イベントカード
            events = latest.get("events", [])
            if events:
                _ev_cols = st.columns(min(len(events), 3))
                for i, ev in enumerate(events[:9]):
                    col_i = i % min(len(events), 3)
                    ev_type = ev.get("event_type", "contact")
                    ev_info = EVENT_TYPES.get(ev_type, {"emoji": "•", "color": "#94a3b8", "label": ev_type})
                    with _ev_cols[col_i]:
                        st.markdown(
                            f"""<div style="background:rgba(0,1,20,0.7);border:1px solid {ev_info['color']}40;
                            border-left:3px solid {ev_info['color']};border-radius:6px;
                            padding:10px 12px;margin-bottom:8px;font-size:12px;">
                            <div style="color:{ev_info['color']};font-weight:bold;margin-bottom:4px;">
                              {ev_info['emoji']} {ev.get('title','')}
                            </div>
                            <div style="color:#94a3b8;font-size:11px;margin-bottom:4px;">
                              【{ev.get('civ','')}】{ev_info['label']}
                            </div>
                            <div style="color:#cbd5e1;line-height:1.5;">
                              {ev.get('description','')}
                            </div>
                            {"<div style='color:#475569;font-size:10px;margin-top:4px;'>影響: " + ", ".join(ev.get("affected",[])) + "</div>" if ev.get("affected") else ""}
                            </div>""",
                            unsafe_allow_html=True,
                        )

            # 過去ラウンドのログ（折りたたみ）
            if len(history) > 1:
                with st.expander(f"📚 過去のラウンドログ（{len(history)-1}件）", expanded=False):
                    for h in history[1:]:
                        st.markdown(f"**第{h['round_no']}ラウンド / A.{h['year']}**")
                        if h.get("summary"):
                            st.caption(h["summary"])
                        for ev in h.get("events", [])[:4]:
                            ev_info = EVENT_TYPES.get(ev.get("event_type",""), {"emoji":"•","label":""})
                            st.markdown(
                                f"- {ev_info['emoji']} **{ev.get('title','')}** "
                                f"（{ev.get('civ','')}）— {ev.get('description','')[:60]}…"
                            )
                        st.markdown("---")
        else:
            st.caption("▶ ラウンド進行ボタンを押してシミュレーションを開始してください。")

    except Exception as e:
        st.error(f"シミュレーションパネルエラー: {e}")

    # ── 関係性テキスト一覧 ──────────────────────────────────────────
    try:
        from novel_graph import get_current_graph, REL_TYPES, AGENT_IDS
        edges = get_current_graph(up_to_episode=ep_arg)
        if edges:
            _REL_EMOJI = {
                "ally": "🤝", "trust": "💙", "rival": "⚔️",
                "suspicion": "🔥", "dependence": "🔗", "neutral": "➖",
            }
            agent_edges   = [(k, v) for k, v in edges.items() if k[0] in AGENT_IDS and k[1] in AGENT_IDS]
            company_edges = [(k, v) for k, v in edges.items() if k[0] not in AGENT_IDS or k[1] not in AGENT_IDS]

            with st.expander("📋 関係性テキスト一覧", expanded=True):
                if agent_edges:
                    st.markdown("**▼ エージェント間**")
                    for (src, tgt), info in sorted(agent_edges):
                        emoji = _REL_EMOJI.get(info["rel_type"], "")
                        desc  = info["note"] if info["note"] else REL_TYPES.get(info["rel_type"], {}).get("label", info["rel_type"])
                        ep_tag = f" （第{info['episode_no']}話）" if info["episode_no"] >= 0 else ""
                        st.markdown(f"- {emoji} **{src}** → **{tgt}**：{desc}{ep_tag}")
                if company_edges:
                    st.markdown("**▼ 企業・文明との関係**")
                    for (src, tgt), info in sorted(company_edges):
                        desc = info["note"]
                        if not desc:
                            continue  # 空note（dormant等）はスキップ
                        ep_tag = f" （第{info['episode_no']}話）" if info["episode_no"] >= 0 else ""
                        # 承認=青、否決=赤、その他=デフォルト
                        if desc == "審査通過":
                            st.markdown(f"- 🔵 **{src}** → **{tgt}**：{desc}{ep_tag}")
                        elif desc == "審査否決":
                            st.markdown(f"- 🔴 **{src}** → **{tgt}**：{desc}{ep_tag}")
                        else:
                            emoji = _REL_EMOJI.get(info["rel_type"], "")
                            st.markdown(f"- {emoji} **{src}** → **{tgt}**：{desc}{ep_tag}")
    except Exception as e:
        st.caption(f"関係テキスト取得エラー: {e}")

    st.markdown("---")

    civs = na.get_civilization_registry()

    if not civs:
        st.info("まだ文明の記録がありません。文豪AIで小説を生成すると自動登録されます。")
    else:
        # ステータス集計
        status_counts = {}
        for c in civs:
            s = c["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        cols = st.columns(4)
        cols[0].metric("活動中 🟢", status_counts.get("active", 0))
        cols[1].metric("滅亡 💀", status_counts.get("collapsed", 0))
        cols[2].metric("昇華 ✨", status_counts.get("ascended", 0))
        cols[3].metric("休眠 😴", status_counts.get("dormant", 0))

        st.markdown("---")

        for civ in civs:
            status_emoji = {"active": "🟢", "collapsed": "💀", "ascended": "✨", "dormant": "😴"}.get(civ["status"], "❓")
            appearances = na.get_civ_appearances(civ["civ_id"])

            with st.expander(
                f"{status_emoji} **{civ['company_name']}** ｜ {civ['industry']} ｜ {civ.get('civ_era','?')}",
                expanded=(civ["status"] == "active")
            ):
                col1, col2 = st.columns(2)
                col1.markdown(f"**正体:** {civ.get('civ_era','?')} の {civ.get('civ_stage','?')}")
                col2.markdown(f"**登場:** 第{civ['first_episode']}話 〜 第{civ['last_episode']}話")

                if civ["notes"]:
                    st.caption(civ["notes"])

                if appearances:
                    st.markdown("**時系列記録：**")
                    for ap in appearances:
                        result_badge = ""
                        if ap["result"] == "approved":
                            result_badge = " ✅承認"
                        elif ap["result"] == "rejected":
                            result_badge = " ❌否決"
                        elif ap["result"] == "bankrupt":
                            result_badge = " 💀破産"
                        elif ap["result"] == "transcended":
                            result_badge = " ✨昇華"
                        st.markdown(
                            f"- **第{ap['episode_no']}話** `{ap['event_type']}`{result_badge} — {ap['description']}"
                        )

                st.markdown("---")
                if st.button("🔄 年代期を初期化", key=f"reset_era_{civ['civ_id']}",
                             help="この文明の civ_era をリセットします"):
                    na.reset_civ_era(civ["civ_id"])
                    st.success(f"「{civ['company_name']}」の年代期をリセットしました")
                    st.rerun()

    # 全文明リセットボタン
    if civs:
        st.markdown("---")
        if st.button("⚠️ 全文明の年代期を一括初期化",
                     help="登録されている全文明の civ_era を NULL にリセットします"):
            n = na.reset_civ_era()
            st.warning(f"{n} 件の文明年代期をリセットしました")
            st.rerun()

    # 手動登録フォーム
    st.markdown("---")
    with st.expander("✏️ 文明を手動登録"):
        with st.form("civ_manual_form"):
            c1, c2 = st.columns(2)
            civ_id = c1.text_input("文明ID（英字）", placeholder="bronze_tribe_01")
            company = c2.text_input("企業名（偽装名）")
            industry = st.text_input("業種")
            civ_era = st.text_input("時代", placeholder="青銅器時代・第三銀河暦など")
            civ_stage = st.text_input("段階", placeholder="都市国家形成期など")
            ep_no = st.number_input("初登場話数", min_value=1, value=1)
            desc = st.text_area("記録メモ")
            if st.form_submit_button("登録"):
                na.register_civilization(
                    civ_id=civ_id, company_name=company, industry=industry,
                    civ_stage=civ_stage, civ_era=civ_era, episode_no=int(ep_no),
                    description=desc
                )
                st.success(f"「{company}」を登録しました")
                st.rerun()
