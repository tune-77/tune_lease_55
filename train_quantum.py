"""
量子解析モジュール学習スクリプト
data/lease_data.db の past_cases から成約/失注案件を取得して QuantumGate を学習・保存。
使い方:
    python3 train_quantum.py
    python3 train_quantum.py --backtest   # 評価指標も表示
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = PROJECT_ROOT / "data" / "lease_data.db"
MODEL_PATH = str(PROJECT_ROOT / "data" / "quantum_model.joblib")
MIN_CASES = 5  # 学習に必要な最小件数


def _load_cases(status: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT data, score FROM past_cases WHERE final_status = ?", (status,)
        ).fetchall()
    finally:
        conn.close()
    result = []
    for raw, score in rows:
        try:
            d = json.loads(raw)
            d["_score"] = float(score or 0)
            result.append(d)
        except Exception:
            pass
    return result


def train(backtest: bool = False) -> None:
    seiyaku = _load_cases("成約")
    lost = _load_cases("失注")
    logger.info("成約: %d件, 失注: %d件", len(seiyaku), len(lost))

    if len(seiyaku) < MIN_CASES:
        logger.warning("成約件数が少ない(%d件)。合成データで補完", len(seiyaku))
        seiyaku = _synth_cases(MAX_N=30)

    from quantum_analysis_module import QuantumGate
    gate = QuantumGate()
    gate.fit(seiyaku, lost if lost else None)
    gate.save(MODEL_PATH)
    logger.info("モデル保存: %s", MODEL_PATH)

    if backtest:
        _run_backtest(gate, lost)


def _synth_cases(MAX_N: int = 30) -> list[dict]:
    """成約案件が少ない場合の合成データ（正常ケース = 均衡した財務）"""
    import random
    cases = []
    for _ in range(MAX_N):
        profit = random.uniform(5, 50)
        cases.append({"inputs": {
            "op_profit": profit * 1000,
            "depreciation": profit * 0.3 * 1000,
            "machines": profit * 0.8 * 1000,
            "net_income": profit * 0.8 * 1000,
            "ord_profit": profit * 0.9 * 1000,
            "grade": "②B格",
            "industry_major": "D 建設業",
            "qualitative": {"strength_tags": [], "onehot": {}},
        }})
    return cases


def _run_backtest(gate, lost: list[dict]) -> None:
    import numpy as np

    high_score_lost = [c for c in lost if c.get("_score", 0) >= 80]
    if not high_score_lost:
        logger.info("高スコア失注案件なし。全失注案件(%d件)でバックテスト", len(lost))
        high_score_lost = lost

    if not high_score_lost:
        logger.warning("失注案件ゼロ。バックテスト不可")
        return

    results = [gate.predict(c) for c in high_score_lost]
    q_risks = [r["quantum_risk"] for r in results]

    threshold_flag = 35.0   # "要再審" 以上 → secondary review 誘導
    threshold_high = 60.0   # "高リスク"
    tp_flag = sum(1 for q in q_risks if q >= threshold_flag)
    tp_high = sum(1 for q in q_risks if q >= threshold_high)
    total = len(q_risks)
    recall_flag = tp_flag / total if total else 0.0
    recall_high = tp_high / total if total else 0.0

    logger.info("=== バックテスト結果 ===")
    logger.info("対象: 高スコア失注 %d件", total)
    logger.info("Q_risk>=35 (要再審+) recall: %.2f (目標 ≥ 0.40)", recall_flag)
    logger.info("Q_risk>=60 (高リスク) recall: %.2f", recall_high)
    logger.info("Q_risk 分布: min=%.1f mean=%.1f max=%.1f",
                min(q_risks), float(np.mean(q_risks)), max(q_risks))

    # マハラノビスとの独立性確認
    try:
        maha_path = str(PROJECT_ROOT / "data" / "mahalanobis_model.joblib")
        from mahalanobis_engine import MahalanobisScorer
        maha = MahalanobisScorer.load(maha_path)
        from quantum_analysis_module import _extract_features
        m_scores = []
        for c in high_score_lost:
            inp = c.get("inputs", c)
            feat = _extract_features(inp)
            if feat is None:
                continue
            try:
                # MahalanobisScorer は DataFrame 入力なので pandas で包む
                import pandas as pd
                df_row = pd.DataFrame([inp])
                score_m, *_ = maha.get_analysis(df_row)
                m_scores.append(float(score_m))
            except Exception:
                pass
        if len(m_scores) >= 3:
            q_sub = q_risks[:len(m_scores)]
            corr = float(np.corrcoef(q_sub, m_scores)[0, 1])
            logger.info("マハラノビスとの相関 r=%.3f (目標 |r|<0.70)", corr)
    except Exception as e:
        logger.debug("マハラノビス相関チェック スキップ: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    args = parser.parse_args()
    train(backtest=args.backtest)
