#!/usr/bin/env python3
"""Vertex AI Search（Discovery Engine）の利用額を推定し、クレジットの消費を見張る。

クレジット残額は CLI/API で読めず、請求の BigQuery エクスポート（cloud_billing_export）は
2026-10 時点でテーブルが無い。そこで Cloud Monitoring の API リクエスト数
（serviceruntime.googleapis.com/api/request_count、service=discoveryengine.googleapis.com）を
メソッド別に数え、単価を掛けて推定する。

精度の限界（朝報・報告にも書く）:
- 単価は公開価格の上振れ側の固定値。無料枠（月1万クエリ）とインデックス保管料は数えない。
  → 推定は実額より高めに出る（自動 off が遅れる方向には外れにくい）。
- 為替は固定（JPY_PER_USD）。Monitoring の反映は数分遅れ、保持は約6週間。
- 前回実行からの差分を累計に足していく。初期値は 2026-10-03 時点の消費額（¥116）。

推定累計がクレジット額の90%で朝報警告、100%で data/vertex_credit_state.json に auto_off を書き、
api/vertex_credit_mode が拡張を止める。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.vertex_credit_mode import CREDIT_JPY, STATE_PATH, WARN_RATIO, load_state  # noqa: E402

PROJECT_ID = "gen-lang-client-0420497423"
JPY_PER_USD = 160.0
INITIAL_USED_JPY = 116.0  # ¥152,846 − 残 ¥152,730（2026-10-03 確認）
# USD / 1000 リクエスト（上振れ側）。Search=Enterprise、Answer=Enterprise＋LLM追加機能、Rank=Ranking API
UNIT_USD_PER_1K = {
    "SearchService.Search": 4.0,
    "ConversationalSearchService.AnswerQuery": 10.0,
    "ConversationalSearchService.ConverseConversation": 10.0,
    "RankService.Rank": 1.0,
}

CountFetcher = Callable[[dt.datetime, dt.datetime], dict[str, int]]


def _token() -> str:
    return subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True, timeout=30).strip()


def fetch_request_counts(start: dt.datetime, end: dt.datetime, project_id: str = PROJECT_ID) -> dict[str, int]:
    """期間中の Discovery Engine のメソッド別リクエスト数。"""
    seconds = max(60, int((end - start).total_seconds()))
    params = {
        "filter": 'metric.type="serviceruntime.googleapis.com/api/request_count" AND resource.type="consumed_api" '
        'AND resource.label.service="discoveryengine.googleapis.com"',
        "interval.startTime": start.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "interval.endTime": end.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregation.alignmentPeriod": f"{seconds}s",
        "aggregation.perSeriesAligner": "ALIGN_SUM",
        "aggregation.crossSeriesReducer": "REDUCE_SUM",
        "aggregation.groupByFields": "resource.label.method",
    }
    url = f"https://monitoring.googleapis.com/v3/projects/{project_id}/timeSeries?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read().decode("utf-8"))
    counts: dict[str, int] = {}
    for series in data.get("timeSeries") or []:
        method = str(series.get("resource", {}).get("labels", {}).get("method") or "")
        total = sum(int(p.get("value", {}).get("int64Value") or 0) for p in series.get("points") or [])
        counts[method] = counts.get(method, 0) + total
    return counts


def estimate_jpy(counts: dict[str, int]) -> float:
    usd = 0.0
    for method, count in counts.items():
        for suffix, unit in UNIT_USD_PER_1K.items():
            if method.endswith(suffix):
                usd += count * unit / 1000
                break
    return round(usd * JPY_PER_USD, 1)


def update_usage(
    state: dict[str, Any],
    *,
    now: dt.datetime,
    fetch: CountFetcher = fetch_request_counts,
) -> dict[str, Any]:
    usage = dict(state.get("usage") or {})
    last_end = dt.datetime.fromisoformat(usage["last_checked_end"]) if usage.get("last_checked_end") else now - dt.timedelta(days=1)
    delta_counts = fetch(last_end, now) if now > last_end else {}
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    month_counts = fetch(month_start, now)
    cumulative = float(usage.get("cumulative_jpy", INITIAL_USED_JPY)) + estimate_jpy(delta_counts)
    usage.update(
        {
            "last_checked_end": now.isoformat(timespec="seconds"),
            "cumulative_jpy": round(cumulative, 1),
            "month": month_start.strftime("%Y-%m"),
            "month_jpy": estimate_jpy(month_counts),
            "month_counts": month_counts,
            "ratio": round(cumulative / CREDIT_JPY, 4),
            "method": "cloud_monitoring_request_count",
        }
    )
    state["usage"] = usage
    if usage["ratio"] >= 1.0 and not state.get("auto_off"):
        state["auto_off"] = True
        state["auto_off_reason"] = f"推定累計 ¥{cumulative:,.0f} がクレジット ¥{CREDIT_JPY:,} に到達"
        state["auto_off_at"] = now.isoformat(timespec="seconds")
    return state


def write_state(state: dict[str, Any], path: Path | None = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def morning_report_lines(state_path: Path | None = None, *, today: dt.date | None = None) -> list[str]:
    """朝報用: 推定利用額とモードの1行、品質比較の1行。90%・自動 off・見張り失敗は警告行を先頭に。"""
    from api.vertex_credit_mode import CREDIT_EXPIRES, credit_mode_status
    from scripts.eval_vertex_vs_chroma import morning_line

    state = load_state(state_path)
    usage = state.get("usage") or {}
    mode = credit_mode_status(today=today, state_path=state_path)
    lines: list[str] = []
    ratio = float(usage.get("ratio") or 0.0)
    if state.get("auto_off"):
        lines.append(f"> [!warning] Vertex クレジットモードを自動 off（{state.get('auto_off_reason', '')}）。拡張は止まり従来動作に戻った")
    elif ratio >= WARN_RATIO:
        lines.append(f"> [!warning] Vertex クレジットの推定消費が {ratio:.0%}（100%で自動 off）")
    if usage.get("last_error"):
        lines.append(f"> [!warning] Vertex 利用額の見張りに失敗: {usage['last_error']}")
    counts = usage.get("month_counts") or {}
    short = {"Search": "SearchService.Search", "Answer": "AnswerQuery", "Rank": "RankService.Rank"}
    breakdown = " / ".join(
        f"{label} {sum(v for k, v in counts.items() if k.endswith(suffix))}" for label, suffix in short.items()
    )
    if usage:
        lines.append(
            f"- 💳 Vertex 推定利用（Monitoring件数×上振れ単価）: 今月 ¥{float(usage.get('month_jpy') or 0):,.0f}（{breakdown}）"
            f"・累計 ¥{float(usage.get('cumulative_jpy') or 0):,.0f}＝クレジットの {ratio:.1%}"
            f"・モード {'on' if mode['active'] else 'off'}（{mode['reason']}）・期限 {CREDIT_EXPIRES.isoformat()}"
        )
    else:
        lines.append(f"- 💳 Vertex 推定利用: まだ計測なし・モード {'on' if mode['active'] else 'off'}（{mode['reason']}）")
    lines.append(morning_line(state))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=None)
    args = parser.parse_args()
    state = load_state(args.state)
    try:
        state = update_usage(state, now=dt.datetime.now().astimezone())
    except Exception as exc:  # noqa: BLE001 - 見張りの失敗は記録して朝報に出す
        state.setdefault("usage", {})["last_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        write_state(state, args.state)
        print(f"vertex_credit_monitor: 失敗 {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    state["usage"].pop("last_error", None)
    write_state(state, args.state)
    usage = state["usage"]
    print(
        f"vertex_credit_monitor: month={usage['month']} ¥{usage['month_jpy']:,.0f} "
        f"cumulative=¥{usage['cumulative_jpy']:,.0f} ({usage['ratio']:.1%}) auto_off={bool(state.get('auto_off'))}"
    )
    if usage["ratio"] >= WARN_RATIO:
        print(f"警告: Vertex クレジットの推定消費が {usage['ratio']:.0%} に達しました", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
