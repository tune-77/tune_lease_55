#!/usr/bin/env python3
"""aurion daily state_*.json を読み、DB異常・スコアリングドリフトフラグを検出する。

パイプラインの早期ステップで実行し:
  1. 異常があれば EXPORT_FILE に追記（改善パイプラインへ問題を伝える）
  2. DAILY-BRIEF.md に aurion アラートセクションを追記

終了コード: 0（正常 / 異常あり両方）、1（読み込み失敗）
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AURION_DIR = PROJECT_ROOT / "data" / "aurion_daily"
EXPORT_FILE = Path(os.environ.get("EXPORT_FILE", "/tmp/obsidian_improvements_export.txt"))

# Vault パス（write_daily_brief.py と同一定義）
_VAULT_PATH = Path.home() / "Documents" / "Obsidian Vault"
_ICLOUD_DOCS = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
_ICLOUD_VAULT_PATH = _ICLOUD_DOCS / "lease-wiki-vault"
_ICLOUD_MAIN_VAULT_PATH = _ICLOUD_DOCS / "Obsidian Vault"   # reindex_obsidian._DEFAULT_VAULT と同一


def find_latest_state() -> Path | None:
    if not AURION_DIR.exists():
        return None
    candidates = sorted(AURION_DIR.glob("state_????-??-??.json"))
    return candidates[-1] if candidates else None


def detect_anomalies(state: dict) -> list[str]:
    alerts: list[str] = []

    # errors フィールドが空でない場合
    errors = state.get("errors", [])
    if errors:
        alerts.append(f"aurion errors: {errors}")

    # DB 異常チェック
    db = state.get("db", {})
    if db.get("status") != "completed":
        alerts.append(f"DB 同期未完了 (status={db.get('status')})")

    # Q_risk の平均が 0.0 かつ全案件が 0 → 計算停止の可能性
    q_risk = db.get("q_risk", {})
    if q_risk.get("n", 0) > 0 and q_risk.get("max_q", 0) == 0.0:
        alerts.append(f"Q_risk が全件 0.0（計算停止の可能性, n={q_risk['n']}）")

    # RAG インデックス異常
    rag = state.get("vault_b_rag", {})
    if rag.get("status") != "completed":
        alerts.append(f"RAG インデックス未完了 (status={rag.get('status')})")
    elif rag.get("returncode", 0) != 0:
        alerts.append(f"RAG インデックス終了コード異常 ({rag.get('returncode')})")

    # 同期異常
    sync = state.get("sync", {})
    if sync.get("status") != "completed":
        alerts.append(f"Obsidian 同期未完了 (status={sync.get('status')})")

    # スコアリングドリフト: スコア帯別の成約率チェック
    # 60-80帯の win_pct が 40-60帯を下回ったらドリフト兆候
    score_bands = db.get("score_bands", [])
    band_map = {b["band"]: b["win_pct"] for b in score_bands}
    if "40-60" in band_map and "60-80" in band_map:
        if band_map["60-80"] < band_map["40-60"]:
            alerts.append(
                f"スコアリングドリフト兆候: 60-80帯 win_pct({band_map['60-80']:.1f}%) < "
                f"40-60帯({band_map['40-60']:.1f}%)"
            )

    return alerts


def append_to_export(alerts: list[str], state_path: Path) -> None:
    if not alerts:
        return
    lines = [
        "[改善] aurion 自動診断アラート（要確認）",
        f"  出典: {state_path.name}",
    ] + [f"  - {a}" for a in alerts]
    try:
        with open(EXPORT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")
        print(f"  EXPORT_FILE へ追記: {EXPORT_FILE}")
    except OSError as e:
        print(f"  警告: EXPORT_FILE 書き込み失敗: {e}")


def append_to_daily_brief(alerts: list[str], state_path: Path, state: dict) -> None:
    """DAILY-BRIEF.md が既存の場合、aurion アラートセクションを末尾に追記する。"""
    if not alerts:
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    started_at = state.get("started_at", "不明")
    section = f"""
## ⚠️ aurion 自動診断アラート

> 生成: {now} | `check_aurion_state.py` | 診断ファイル: `{state_path.name}` (started: {started_at})

"""
    section += "\n".join(f"- {a}" for a in alerts) + "\n"

    for vault in [_VAULT_PATH, _ICLOUD_VAULT_PATH, _ICLOUD_MAIN_VAULT_PATH]:
        brief = vault / "DAILY-BRIEF.md"
        if brief.exists():
            try:
                existing = brief.read_text(encoding="utf-8")
                if "aurion 自動診断アラート" in existing:
                    marker = "## ⚠️ aurion 自動診断アラート"
                    existing = existing.split(marker)[0].rstrip()
                brief.write_text(existing + "\n" + section, encoding="utf-8")
                print(f"  DAILY-BRIEF.md にアラート追記: {brief}")
            except OSError as e:
                print(f"  警告: DAILY-BRIEF.md 書き込み失敗 ({brief}): {e}")


def save_alert_file(alerts: list[str], state_path: Path, state: dict) -> None:
    """iCloud メインVault の Projects/tune_lease_55/Alerts/ にアラートファイルを保存する。"""
    if not alerts or not _ICLOUD_MAIN_VAULT_PATH.exists():
        return

    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    started_at = state.get("started_at", "不明")

    alert_dir = _ICLOUD_MAIN_VAULT_PATH / "Projects" / "tune_lease_55" / "Alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    alert_path = alert_dir / f"aurion_alert_{today}.md"

    content = f"""# aurion 自動診断アラート — {today}

> 生成: {now} | 診断ファイル: `{state_path.name}` (started: {started_at})

## 検出された異常

"""
    content += "\n".join(f"- {a}" for a in alerts) + "\n"

    conclusions = state.get("reasoning", {}).get("conclusions", [])
    if conclusions:
        content += "\n## aurion 推論サマリ\n\n"
        content += "\n".join(f"- {c}" for c in conclusions) + "\n"

    try:
        alert_path.write_text(content, encoding="utf-8")
        print(f"  アラートファイル保存: {alert_path}")
    except OSError as e:
        print(f"  警告: アラートファイル保存失敗: {e}")


def main() -> int:
    state_path = find_latest_state()
    if state_path is None:
        print("[check_aurion_state] state_*.json が見つかりません → スキップ")
        return 0

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[check_aurion_state] 読み込み失敗: {e}")
        return 1

    print(f"[check_aurion_state] 診断ファイル: {state_path.name}")
    alerts = detect_anomalies(state)

    if alerts:
        print(f"  ⚠️  異常 {len(alerts)} 件検出:")
        for a in alerts:
            print(f"    - {a}")
        append_to_export(alerts, state_path)
        append_to_daily_brief(alerts, state_path, state)
        save_alert_file(alerts, state_path, state)
    else:
        print("  ✅ 異常なし")

    # reasoning conclusions があれば EXPORT_FILE に追記（パイプラインプロンプト強化）
    conclusions = state.get("reasoning", {}).get("conclusions", [])
    if conclusions:
        lines = ["[改善] aurion 推論サマリ（参考情報）"] + [
            f"  - {c}" for c in conclusions
        ]
        try:
            with open(EXPORT_FILE, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n\n")
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
