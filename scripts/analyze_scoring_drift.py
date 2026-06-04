"""
data/lease_data.db からスコア帯別成約率の逆転を検出し、
異常検出時のみ [改善] タグ付きテキストを stdout に出力する。
正常時は何も出力しない（パイプラインへのノイズを最小化）。

run_daily_improvement_pipeline.sh から >> EXPORT_FILE でキャプチャされる。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "lease_data.db"

# 判定する勝率: 低帯の成約率 > 高帯の成約率 なら逆転異常
# (低帯ラベル, 低下限, 低上限, 高帯ラベル, 高下限, 高上限)
_BAND_PAIRS = [
    ("40-60帯", 40, 60, "60-80帯", 60, 80),
    ("60-80帯", 60, 80, "80-100帯", 80, 100),
]

_WIN_STATUSES = ("成約", "検収完了")
_MIN_CASES = 50  # 信頼性確保のための最低件数
_THRESHOLD_PT = 1.0  # 逆転と判定する最低差（%pt）


def _get_band_win_rate(
    conn: sqlite3.Connection, lo: float, hi: float
) -> tuple[int, float] | None:
    """スコア帯の (件数, 成約率) を返す。件数が閾値未満なら None。"""
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                SUM(CASE WHEN final_status IN (?, ?) THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS win_rate
            FROM past_cases
            WHERE score >= ? AND score < ?
              AND final_status IN ('成約', '失注', '検収完了')
            """,
            (*_WIN_STATUSES, lo, hi),
        ).fetchone()
    except sqlite3.Error as e:
        print(f"警告: DB クエリ失敗: {e}", file=sys.stderr)
        return None
    if not row or row[0] < _MIN_CASES:
        return None
    return int(row[0]), float(row[1])


def main() -> None:
    if not _DB_PATH.exists():
        print(f"警告: DB が見つかりません: {_DB_PATH}", file=sys.stderr)
        return

    try:
        conn = sqlite3.connect(str(_DB_PATH))
    except sqlite3.Error as e:
        print(f"警告: DB 接続失敗: {e}", file=sys.stderr)
        return

    anomalies: list[str] = []

    try:
        for low_label, low_lo, low_hi, high_label, high_lo, high_hi in _BAND_PAIRS:
            low_result = _get_band_win_rate(conn, low_lo, low_hi)
            high_result = _get_band_win_rate(conn, high_lo, high_hi)

            if low_result is None or high_result is None:
                continue

            low_n, low_rate = low_result
            high_n, high_rate = high_result

            # 低帯の成約率が高帯を上回っている（かつ差が閾値以上）
            if low_rate > high_rate + _THRESHOLD_PT / 100:
                diff_pt = round((low_rate - high_rate) * 100, 1)
                anomalies.append(
                    f"[改善] スコア{high_label}の成約率逆転：モデルキャリブレーション見直し\n"
                    f"理由：analyze_scoring_drift — {high_label}"
                    f"({high_n}件, {high_rate * 100:.1f}%) が"
                    f"{low_label}({low_n}件, {low_rate * 100:.1f}%) を"
                    f"{diff_pt}pt 下回る逆転を検出\n"
                )
    finally:
        conn.close()

    if anomalies:
        header = "# スコアリングドリフト自動分析\n\n"
        print(header + "\n".join(anomalies))
        print(
            f"analyze_scoring_drift: {len(anomalies)}件の逆転異常を検出",
            file=sys.stderr,
        )
    else:
        print(
            "analyze_scoring_drift: 異常なし（スコア帯別成約率は正常範囲）",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
