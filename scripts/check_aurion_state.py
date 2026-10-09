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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_paths import resolve_lease_wiki_vault, resolve_obsidian_vault  # noqa: E402

AURION_DIR = PROJECT_ROOT / "data" / "aurion_daily"
EXPORT_FILE = Path(os.environ.get("EXPORT_FILE", "/tmp/obsidian_improvements_export.txt"))

# Vault パス（env → iCloud の解決順は runtime_paths が唯一の正）
_VAULT_PATH = resolve_obsidian_vault()
_ICLOUD_VAULT_PATH = resolve_lease_wiki_vault()
_ICLOUD_MAIN_VAULT_PATH = _VAULT_PATH


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
    # 出典ファイル名（state_YYYY-MM-DD.json）には日付が入るため、ここに含めると
    # 同じ異常が翌日も続いただけで改善パイプラインの重複防止キー
    # （canonical_key = title+description のハッシュ、step1_extract_and_structure.py）
    # が毎回変わり、既存REVとして畳み込まれず日毎に新規REVが発行され続けてしまう
    # （READMEにあるREV-230/237・REV-292と同種の重複発行）。日付付きの出典は
    # save_alert_file / append_to_daily_brief 側にのみ残す。
    lines = ["[改善] aurion 自動診断アラート（要確認）"] + [f"  - {a}" for a in alerts]
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

    # REV-585: DAILY-BRIEF はメインVaultのルート1か所だけ（lease-wiki-vault 側は更新を止めた）
    for vault in [_VAULT_PATH]:
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


ALERT_REMINDER_DAYS = 7


def _alert_state_path() -> Path:
    """前回 Alerts に書いた警告の一覧（REV-585）。検証時は DATA_DIR で本番 data/ から切り離す。"""
    data_dir = Path(os.environ.get("DATA_DIR") or PROJECT_ROOT / "data")
    return data_dir / "aurion_alert_state.json"


def _load_alert_state() -> dict:
    try:
        value = json.loads(_alert_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_alert_state(alerts: list[str], today: str, first_seen: str) -> None:
    path = _alert_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"alerts": alerts, "written": today, "since": first_seen}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError as e:
        print(f"  警告: アラート状態の保存失敗: {e}")


def diff_alerts(previous: list[str], current: list[str]) -> dict[str, list[str]]:
    """前回との差分。文面が変われば（数値の悪化も含め）新規＋解消として扱う。"""
    prev, curr = set(previous), set(current)
    return {
        "new": [a for a in current if a not in prev],
        "resolved": [a for a in previous if a not in curr],
        "ongoing": [a for a in current if a in prev],
    }


def save_alert_file(alerts: list[str], state_path: Path, state: dict, *, today: str | None = None) -> str | None:
    """Projects/tune_lease_55/Alerts/ に、警告が変わった時だけ書く（REV-585）。

    新規・解消・文面の変化（悪化を含む）があれば必ず書く。前回と同じなら書かない。
    同じ警告が続いても ALERT_REMINDER_DAYS 日ごとに「継続」として書き直し、出なくならないようにする。
    戻り値は書いた理由（new / resolved / reminder）か None。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    previous_state = _load_alert_state()
    previous = list(previous_state.get("alerts") or [])
    diff = diff_alerts(previous, alerts)
    since = previous_state.get("since") or today
    days_since_written = 0
    try:
        days_since_written = (
            datetime.fromisoformat(today) - datetime.fromisoformat(previous_state.get("written") or today)
        ).days
    except ValueError:
        days_since_written = ALERT_REMINDER_DAYS

    if diff["new"] or diff["resolved"]:
        reason = "new" if diff["new"] else "resolved"
        since = today if diff["new"] or not previous else since
    elif alerts and days_since_written >= ALERT_REMINDER_DAYS:
        reason = "reminder"
    else:
        if alerts:
            print(f"  アラートは前回（{previous_state.get('written')}）と同じ → Alerts への書き出しは省略")
        return None
    if not _ICLOUD_MAIN_VAULT_PATH.exists():
        return None

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    started_at = state.get("started_at", "不明")
    alert_dir = _ICLOUD_MAIN_VAULT_PATH / "Projects" / "tune_lease_55" / "Alerts"
    alert_dir.mkdir(parents=True, exist_ok=True)
    alert_path = alert_dir / f"aurion_alert_{today}.md"

    content = f"""# aurion 自動診断アラート — {today}

> 生成: {now} | 診断ファイル: `{state_path.name}` (started: {started_at}) | 前回と変化した時だけ書き出し（REV-585）

"""
    if diff["new"]:
        content += "## 新しく出た・変化した異常\n\n" + "\n".join(f"- {a}" for a in diff["new"]) + "\n\n"
    if diff["resolved"]:
        content += "## 解消した・変化した異常\n\n" + "\n".join(f"- {a}" for a in diff["resolved"]) + "\n\n"
    if diff["ongoing"]:
        label = f"継続中の異常（{since} から）" if reason == "reminder" else "継続中の異常"
        content += f"## {label}\n\n" + "\n".join(f"- {a}" for a in diff["ongoing"]) + "\n\n"
    if not alerts:
        content += "現在、検出されている異常はありません。\n"

    conclusions = state.get("reasoning", {}).get("conclusions", [])
    if conclusions and alerts:
        content += "\n## aurion 推論サマリ\n\n" + "\n".join(f"- {c}" for c in conclusions) + "\n"

    try:
        alert_path.write_text(content, encoding="utf-8")
        print(f"  アラートファイル保存（{reason}）: {alert_path}")
    except OSError as e:
        print(f"  警告: アラートファイル保存失敗: {e}")
        return None
    _save_alert_state(alerts, today, since if alerts else today)
    return reason


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
    else:
        print("  ✅ 異常なし")
    # 異常なしの日も呼ぶ（前回あった異常の「解消」を書くため）
    save_alert_file(alerts, state_path, state)

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
