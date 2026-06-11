"""daily_knowledge_feed.py — 毎朝6時に実行される知識フィード収集スクリプト（Phase 1-B）

既存スクリプトを順番に呼び出すだけのオーケストレーター。
各タスクは独立していて、1つが失敗しても他は続行する。

実行方法:
    python3 daily_knowledge_feed.py              # 全フィード実行
    python3 daily_knowledge_feed.py --only estat # e-Stat のみ
    python3 daily_knowledge_feed.py --dry-run    # 実行せず確認のみ

launchd から:
    launchd/com.tunelease.daily-knowledge-feed.plist を読み込むと毎朝6:00に自動実行
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily_feed")

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

LOG_FILE = _DIR / "data" / "daily_feed_log.jsonl"


# ──────────────────────────────────────────────────────────────────────────────
# タスク定義
# ──────────────────────────────────────────────────────────────────────────────

def task_estat_annual() -> dict:
    """e-Stat から法人企業統計（年次）を取得し industry_capex_lease.json を更新する。"""
    from fetch_estat_annual import main as fetch_annual
    fetch_annual()
    out = _DIR / "data" / "industry_capex_lease.json"
    return {"output": str(out), "exists": out.exists()}


def task_estat_benchmarks() -> dict:
    """e-Stat から業種別財務指標を取得し industry_benchmarks.json を更新する。"""
    from fetch_estat_benchmarks import main as fetch_bench
    fetch_bench()
    out = _DIR / "data" / "industry_benchmarks.json"
    return {"output": str(out), "exists": out.exists()}


def task_boj_rate() -> dict:
    """日銀 API から短期・長期金利を取得し base_rate_master テーブルを更新する。

    日銀の統計データ API (e-Stat 経由) から無担保コール翌日物・10年国債利回りを取得する。
    API が取得できない場合は既存値を維持する（フォールバック）。
    """
    import urllib.request
    import sqlite3

    DB_PATH = str(_DIR / "data" / "lease_data.db")
    today = datetime.now().strftime("%Y-%m")

    # 日銀統計 API: e-Stat statsDataId=0003343024（短期金利）
    # 実際の APIキーは fetch_estat_annual.py と共有
    APP_ID = "5d55cf528a66dc1ded12484f09cfe9e62a1522c7"
    url = (
        "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData"
        f"?appId={APP_ID}&statsDataId=0003343024&metaGetFlg=N&cntGetFlg=N"
        "&cdCat01=0010000&startPosition=1&limit=1&sectionHeaderFlg=1"
    )

    rate_value: float | None = None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "tunelease/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        values = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {}).get("DATA_INF", {}).get("VALUE", [])
        if values:
            rate_value = float(values[-1].get("$", 0))
    except Exception as e:
        log.warning("日銀API取得失敗（フォールバック）: %s", e)
        return {"rate": None, "skipped": True, "reason": str(e)}

    if rate_value is not None:
        try:
            conn = sqlite3.connect(DB_PATH)
            # base_rate_master テーブルが存在すれば今月の金利を記録
            conn.execute(
                "INSERT OR IGNORE INTO base_rate_master (month) VALUES (?)", (today,)
            )
            # 短期金利（2年リース基準）を更新
            conn.execute(
                "UPDATE base_rate_master SET r_2y = ?, r_3y = ? WHERE month = ?",
                (round(rate_value + 0.5, 3), round(rate_value + 0.8, 3), today)
            )
            conn.commit()
            conn.close()
            return {"rate": rate_value, "month": today, "updated": True}
        except Exception as e:
            return {"rate": rate_value, "db_error": str(e)}

    return {"rate": None, "skipped": True}


def task_lease_news_rss() -> dict:
    """リース業界ニュースを収集して Obsidian に書き出す。

    既存の scripts/run_lease_news_collection.sh に相当する処理を Python から呼ぶ。
    """
    # 既存のニュース収集スクリプトが存在すれば実行
    script = _DIR / "scripts" / "run_lease_news_collection.sh"
    if script.exists():
        import subprocess
        result = subprocess.run(
            ["/bin/zsh", str(script), "--limit", "10"],
            cwd=str(_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            env={
                **os.environ,
                "PYTHONPATH": str(_DIR),
            }
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-500:] if result.stdout else "",
            "stderr": result.stderr[-300:] if result.stderr else "",
        }

    # スクリプトがなければ lease_news_digest.py で直接取得
    try:
        from lease_news_digest import LeaseNewsFocus
        focus = LeaseNewsFocus(available=False)
        return {"available": focus.available, "skipped": not focus.available}
    except Exception as e:
        return {"error": str(e)}


def task_lease_judgment_research() -> dict:
    """審査判断に直接使う知識をWeb調査し、通常のObsidian Vaultへ保存する。"""
    import subprocess

    script = _DIR / "scripts" / "auto_research_lease_judgment.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(_DIR),
        capture_output=True,
        text=True,
        timeout=180,
        env={
            **os.environ,
            "PYTHONPATH": str(_DIR),
        },
    )
    payload: dict = {
        "returncode": result.returncode,
        "stdout": result.stdout[-1000:] if result.stdout else "",
        "stderr": result.stderr[-500:] if result.stderr else "",
    }
    if result.returncode != 0:
        raise RuntimeError(payload["stderr"] or "lease judgment auto research failed")
    return payload


def task_machinery_orders_to_vault() -> dict:
    """機械受注統計分析結果を lease-wiki-vault に Markdown ノートとして書き出す。"""
    analysis_path = _DIR / "data" / "external" / "estat_machinery_orders" / "machinery_orders_analysis.json"
    if not analysis_path.exists():
        return {"skipped": True, "reason": "machinery_orders_analysis.json が存在しない"}

    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    core = data["core_indicator"]
    latest = data["latest_values_100m_yen"]

    latest_month = meta["latest_month"]  # e.g. "2026-03"
    signal = core["macro_signal"]

    icloud_docs = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
    vault_path = icloud_docs / "lease-wiki-vault"
    if not vault_path.exists():
        return {"skipped": True, "reason": f"lease-wiki-vault が見つかりません: {vault_path}"}

    output_dir = vault_path / "10_Industry_Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"機械受注統計_{latest_month}.md"

    content = f"""# 機械受注統計_{latest_month}

> e-Stat 内閣府 機械受注統計 | 自動生成: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## マクロ判定: {signal}

## 最新値（{latest_month}、季節調整値・億円）

| 項目 | 値（億円） |
|------|-----------|
| 受注額合計 | {latest.get('受注額合計', '-'):,.1f} |
| 船舶・電力を除く民需 | {latest.get('民間需要（船舶・電力を除く）', '-'):,.1f} |
| 製造業 | {latest.get('民間需要_製造業計', '-'):,.1f} |
| 非製造業（除船・電） | {latest.get('民間需要_非製造業（船舶・電力を除く）', '-'):,.1f} |
| 海外需要 | {latest.get('海外需要', '-'):,.1f} |
| 官公需 | {latest.get('官公需計', '-'):,.1f} |

## 変化率

- 前月比（季節調整）: **{core['latest_mom_pct']:+.1f}%**
- 前年同月比（原系列）: **{core['latest_yoy_pct_original']:+.1f}%**
- 3か月平均の3か月前比: **{core['three_month_average_change_vs_3m_ago_pct']:+.1f}%**

## 傾向

- 3か月移動平均: {core['latest_3m_average_100m_yen']:,.1f}億円
- 12か月移動平均: {core['latest_12m_average_100m_yen']:,.1f}億円
- 期間線形傾向: 1か月当たり{core['monthly_linear_trend_100m_yen']:+.1f}億円

## リース審査への活用メモ

- マクロ判定が「改善」の場合は設備投資需要の増加局面として審査コメントへ反映可能
- 「減速」の場合は中古流通性・再販価格の確認を通常より厳しく行う
- 製造業/非製造業を分け、顧客業種に合う系列を参照すること

## 出典

- e-Stat 統計表ID: {meta['stats_data_id']}
- 観測期間: {meta['period']}
- データ取得日: {meta['retrieved_at']}
"""

    output_path.write_text(content, encoding="utf-8")
    return {
        "output": str(output_path),
        "latest_month": latest_month,
        "macro_signal": signal,
    }


def task_macro_drift_check() -> dict:
    """コンセプトドリフト検知を実行し、異常時に Slack 通知する。"""
    from macro_drift_monitor import check_concept_drift
    result = check_concept_drift()
    if result.get("is_drift"):
        try:
            from slack_notify import send_slack_message
            send_slack_message(
                f"⚠️ [daily_feed] コンセプトドリフト検知\n{result.get('message', '')}"
            )
        except Exception:
            pass
    return result


def task_fluid_pipeline_status() -> dict:
    """FluidPipeline の状態をチェックし、再学習条件が揃っていれば起動する。"""
    from fluid_pipeline import FluidPipeline
    fp = FluidPipeline()
    status = fp.status()
    retrain = status.get("retraining", {})
    if retrain.get("needed"):
        fp._spawn_retraining("daily_feed_scheduled")
        return {**retrain, "retraining_spawned": True}
    return {**retrain, "retraining_spawned": False}


# ──────────────────────────────────────────────────────────────────────────────
# タスクレジストリ
# ──────────────────────────────────────────────────────────────────────────────

TASKS: dict[str, tuple[str, Callable]] = {
    "estat":    ("e-Stat 法人企業統計（年次）",              task_estat_annual),
    "bench":    ("e-Stat 業種別財務指標",                    task_estat_benchmarks),
    "boj":      ("日銀金利 API",                            task_boj_rate),
    "machinery":("機械受注統計 → lease-wiki-vault",          task_machinery_orders_to_vault),
    "research": ("リース判断 Auto Research",                 task_lease_judgment_research),
    "news":     ("リース業界ニュース RSS",                    task_lease_news_rss),
    "drift":    ("コンセプトドリフト検知",                    task_macro_drift_check),
    "pipeline": ("FluidPipeline 再学習チェック",             task_fluid_pipeline_status),
}


# ──────────────────────────────────────────────────────────────────────────────
# 実行エンジン
# ──────────────────────────────────────────────────────────────────────────────

def _log_result(task_key: str, status: str, result: dict, duration_s: float) -> None:
    entry = {
        "ts": datetime.now().isoformat(),
        "task": task_key,
        "status": status,
        "duration_s": round(duration_s, 2),
        **result,
    }
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def run(only: list[str] | None = None, dry_run: bool = False) -> dict[str, str]:
    """全タスクを順番に実行する。Returns: {task_key: "ok"|"error"|"skipped"}"""
    import time

    targets = only if only else list(TASKS.keys())
    summary: dict[str, str] = {}

    log.info("=" * 55)
    log.info("🌊 daily_knowledge_feed 開始 (%s)", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("実行タスク: %s", ", ".join(targets))
    if dry_run:
        log.info("🔍 DRY RUN モード")
    log.info("=" * 55)

    for key in targets:
        if key not in TASKS:
            log.warning("未知のタスク: %s — スキップ", key)
            summary[key] = "unknown"
            continue

        label, fn = TASKS[key]
        log.info("▶ [%s] %s", key, label)

        if dry_run:
            log.info("  → DRY RUN: スキップ")
            summary[key] = "dry_run"
            continue

        t0 = time.monotonic()
        try:
            result = fn()
            duration = time.monotonic() - t0
            log.info("  ✅ 完了 (%.1fs) %s", duration, json.dumps(result, ensure_ascii=False)[:120])
            _log_result(key, "ok", result if isinstance(result, dict) else {}, duration)
            summary[key] = "ok"
        except Exception as e:
            duration = time.monotonic() - t0
            tb = traceback.format_exc()[-400:]
            log.error("  ❌ エラー (%.1fs): %s", duration, e)
            log.debug(tb)
            _log_result(key, "error", {"error": str(e), "traceback": tb}, duration)
            summary[key] = "error"

    log.info("=" * 55)
    ok = sum(1 for v in summary.values() if v == "ok")
    err = sum(1 for v in summary.values() if v == "error")
    log.info("🏁 完了: %d成功 / %d失敗 / %d件合計", ok, err, len(targets))
    log.info("=" * 55)
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="毎朝の知識フィード収集")
    parser.add_argument("--only", nargs="+", choices=list(TASKS.keys()),
                        help=f"実行するタスクを指定 (選択肢: {', '.join(TASKS.keys())})")
    parser.add_argument("--dry-run", action="store_true", help="実行せず確認のみ")
    args = parser.parse_args()

    results = run(only=args.only, dry_run=args.dry_run)
    # エラーがあれば終了コード1
    sys.exit(1 if "error" in results.values() else 0)
