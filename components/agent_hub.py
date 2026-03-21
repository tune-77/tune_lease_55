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
        chair_name = "つね"
        chair_system = "あなたは統括マネージャー「つね」として、チームの意見を集約し最終決裁を下します。"

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
            st.dataframe(df, use_container_width=True)

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
# メイン描画
# ══════════════════════════════════════════════════════════════════════════════

def render_agent_hub() -> None:
    st.title("🤖 汎用エージェントハブ")
    st.caption("8種のエージェントを使って審査プロセスを自動化・高度化します。")

    tabs = st.tabs([
        "🏭 ベンチマーク取得",
        "📈 金利・市況",
        "📝 審査理由書",
        "🤝 チーム自律化",
        "💬 Slack高度化",
        "🚨 異常検知",
        "🔄 再学習トリガー",
        "📅 定期配信",
    ])

    with tabs[0]: _render_benchmark_panel()
    with tabs[1]: _render_market_panel()
    with tabs[2]: _render_report_gen_panel()
    with tabs[3]: _render_auto_team_panel()
    with tabs[4]: _render_slack_enhanced_panel()
    with tabs[5]: _render_anomaly_panel()
    with tabs[6]: _render_retrain_panel()
    with tabs[7]: _render_schedule_panel()

    # ── 実行ログ（折りたたみ）────────────────────────────────────────────────
    with st.expander("📋 エージェント実行ログ", expanded=False):
        try:
            if os.path.exists(_HUB_LOG):
                with open(_HUB_LOG, encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines[-30:]):
                    try:
                        e = json.loads(line)
                        st.caption(f"{e['ts'][:19]} | {e['agent']} | {e['status']} | {e['detail']}")
                    except Exception:
                        pass
            else:
                st.caption("ログなし")
        except Exception as e:
            st.caption(f"ログ読み込みエラー: {e}")
