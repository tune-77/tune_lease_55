"""スコアリング実行時のリアルタイム異常値検知・即時ログ/アラート。

背景:
    scoring_core.py はリクエストごとに Q_risk・信用リスク群スコア・
    総合/借手スコア差などを既に計算しているが、これまでは結果を
    UI へ返すだけで、閾値超過（強警戒帯）や想定外の欠損値が出ても
    誰も即座には気づけなかった（バッチ監査 system_guardrails.py は
    成約/失注済み案件のみを対象とし、審査実行の瞬間は見ていない）。

    本モジュールは審査APIレスポンス確定直後（api/main.py から
    BackgroundTasks 経由で呼ぶ想定）に result dict を検査し、
    「おかしな挙動の兆候が出た瞬間」にログとSlack通知キューへ即時に残す。
    scoring_core.py 自体の計算ロジックには一切手を入れない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app_logger import log_warning
from constants import Q_RISK_ATTENTION_LINE, Q_RISK_STRONG_WARNING_LINE

# 総合スコアと借手スコアの差がこれ以上なら「数値の乖離」として警戒する。
# 根拠: scoring_core.py 内で診断推奨(diagnostic_recommendations)がこの差 20 点を
# 目安にしているため合わせる（マハラノビス/UMAP診断のトリガー基準と同一の値）。
SCORE_GAP_WARNING = 20.0
SCORE_GAP_ERROR = 35.0

ERROR = "error"
WARNING = "warning"

# 同一案件・同一種別の通知を連続で送らないためのクールダウン（秒）。
# 審査フォームは値を微調整しながら何度も再計算されるため、無しだと
# Slack通知キューが同じ警告で埋まってしまう。
_ALERT_COOLDOWN_SECONDS = 600.0
_last_alert_at: dict[tuple[str, str], float] = {}


@dataclass
class Finding:
    level: str
    kind: str
    message: str


def _is_missing_or_wrong_type(value: Any, expected_types: tuple[type, ...]) -> bool:
    return value is None or not isinstance(value, expected_types)


def detect_scoring_anomalies(result: dict[str, Any]) -> list[Finding]:
    """審査結果 dict から異常値・未定義の状態変化を検知する。"""
    findings: list[Finding] = []

    score = result.get("score")
    score_borrower = result.get("score_borrower")
    hantei = result.get("hantei")

    # ── 未定義の状態変化: 常に埋まるはずのフィールドが欠損/型不正 ──
    if _is_missing_or_wrong_type(score, (int, float)):
        findings.append(
            Finding(ERROR, "undefined_state", f"score が未定義/不正な型です（値: {score!r}）")
        )
    if _is_missing_or_wrong_type(score_borrower, (int, float)):
        findings.append(
            Finding(
                ERROR,
                "undefined_state",
                f"score_borrower が未定義/不正な型です（値: {score_borrower!r}）",
            )
        )
    if hantei not in ("承認圏内", "要審議"):
        findings.append(
            Finding(ERROR, "undefined_state", f"hantei が想定外の値です（値: {hantei!r}）")
        )

    # ── 数値の乖離: 総合スコアと借手スコアが大きく離れている ──
    if isinstance(score, (int, float)) and isinstance(score_borrower, (int, float)):
        gap = abs(float(score) - float(score_borrower))
        if gap >= SCORE_GAP_ERROR:
            findings.append(
                Finding(
                    ERROR,
                    "score_gap",
                    f"総合スコアと借手スコアの乖離が大きすぎます（差 {gap:.1f}点、"
                    f"score={score} / score_borrower={score_borrower}）",
                )
            )
        elif gap >= SCORE_GAP_WARNING:
            findings.append(
                Finding(
                    WARNING,
                    "score_gap",
                    f"総合スコアと借手スコアに乖離があります（差 {gap:.1f}点）",
                )
            )

    # ── Q_risk（量子干渉リスクスコア）の強警戒 ──
    quantum_risk = result.get("quantum_risk")
    if isinstance(quantum_risk, (int, float)):
        if quantum_risk >= Q_RISK_STRONG_WARNING_LINE:
            findings.append(
                Finding(ERROR, "quantum_risk", f"Q_risk が強警戒帯です（値 {quantum_risk:.1f}）")
            )
        elif quantum_risk >= Q_RISK_ATTENTION_LINE:
            findings.append(
                Finding(WARNING, "quantum_risk", f"Q_risk が要注意帯です（値 {quantum_risk:.1f}）")
            )

    # ── 信用リスク群 × Q_risk の複合強警戒フラグ ──
    if result.get("credit_quantum_strong_warning"):
        findings.append(
            Finding(
                ERROR,
                "credit_quantum_strong_warning",
                "信用リスク群スコアとQ_riskが同時に強警戒水準です"
                f"（credit_risk_group_score={result.get('credit_risk_group_score')}, "
                f"quantum_risk={quantum_risk}）",
            )
        )

    return findings


def _should_alert(case_label: str, kind: str, now: float) -> bool:
    key = (case_label, kind)
    last = _last_alert_at.get(key)
    if last is not None and (now - last) < _ALERT_COOLDOWN_SECONDS:
        return False
    _last_alert_at[key] = now
    return True


def record_scoring_anomalies(result: dict[str, Any], case_label: str = "") -> list[Finding]:
    """異常値を検知し、即時ログ + Slack通知キューへ積む（ベストエフォート）。

    api/main.py の BackgroundTasks から呼ぶ想定（レスポンス確定後に非同期実行、
    審査リクエスト自体のレイテンシには影響しない）。
    """
    findings = detect_scoring_anomalies(result)
    if not findings:
        return findings

    now = time.monotonic()
    label = case_label or str(result.get("company_no") or result.get("company_name") or "unknown")

    for finding in findings:
        log_warning(finding.message, context=f"scoring_anomaly:{finding.kind}")

        if finding.level != ERROR:
            continue
        if not _should_alert(label, finding.kind, now):
            continue
        try:
            from slack_notify import push_notification

            push_notification(f"【審査リアルタイム異常検知】{label}: {finding.message}")
        except Exception:
            pass

    return findings
