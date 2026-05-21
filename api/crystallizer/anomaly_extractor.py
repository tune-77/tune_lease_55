"""
screening_records から「学びになる案件」を抽出する。
- スコアが平均±15点以上の外れ値案件
- エージェント間で意見が割れた案件（debate_log に「否決」と「承認」が混在）
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field

# past_cases テーブルの DB パス（data_cases.py と同じ）
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
_DB_PATH = os.path.join(_DATA_DIR, "lease_data.db")

_LOOKBACK_COUNT = 30    # 直近 N 件を対象
_ANOMALY_SIGMA = 15.0   # 平均±N点を外れ値とする


@dataclass
class AnomalyCase:
    """抽出した外れ値/意見割れ案件。"""
    case_id: str
    score: float
    industry: str
    judgment: str
    reason: str
    data: dict = field(default_factory=dict)

    def to_summary(self) -> str:
        return (
            f"[{self.case_id}] 業種={self.industry} スコア={self.score:.1f} "
            f"結果={self.judgment} 理由={self.reason}"
        )


def extract_anomalies(db_path: str = _DB_PATH) -> list[AnomalyCase]:
    """
    直近30件から外れ値・意見割れ案件を抽出して返す。
    DB が存在しない場合は空リストを返す。
    """
    if not os.path.exists(db_path):
        return []

    try:
        with closing(sqlite3.connect(db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, score, industry_sub,
                       json_extract(data, '$.judgment') AS judgment,
                       json_extract(data, '$.debate_log') AS debate_log,
                       data
                FROM past_cases
                WHERE score IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (_LOOKBACK_COUNT,),
            ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    scores = [r["score"] for r in rows if r["score"] is not None]
    if not scores:
        return []

    avg = sum(scores) / len(scores)
    anomalies: list[AnomalyCase] = []

    for r in rows:
        score = r["score"]
        if score is None:
            continue

        try:
            data = json.loads(r["data"] or "{}")
        except Exception:
            data = {}

        reason = ""

        # 外れ値チェック
        if abs(score - avg) >= _ANOMALY_SIGMA:
            direction = "高スコア" if score > avg else "低スコア"
            reason = f"平均({avg:.1f}点)から{abs(score - avg):.1f}点乖離の{direction}案件"

        # 意見割れチェック（debate_log に承認と否決が両方含まれる）
        debate_log = r["debate_log"] or ""
        if not reason and debate_log:
            has_approve = "承認" in debate_log
            has_reject = "否決" in debate_log
            if has_approve and has_reject:
                reason = "エージェント間で承認/否決が割れた案件"

        if not reason:
            continue

        anomalies.append(AnomalyCase(
            case_id=str(r["id"]),
            score=score,
            industry=r["industry_sub"] or "不明",
            judgment=r["judgment"] or "不明",
            reason=reason,
            data=data,
        ))

    return anomalies
