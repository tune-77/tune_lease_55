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

def _load_training_config() -> dict:
    p = PROJECT_ROOT / "data" / "quantum_config.json"
    try:
        import json as _j
        return _j.loads(p.read_text(encoding="utf-8")).get("training", {})
    except Exception:
        return {}

_TRAIN_CFG = _load_training_config()
MIN_CASES: int = int(_TRAIN_CFG.get("min_cases", 5))
_SYNTH_N: int  = int(_TRAIN_CFG.get("synth_fallback_n", 30))


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
        seiyaku = _synth_cases(MAX_N=_SYNTH_N)

    from quantum_analysis_module import QuantumGate
    gate = QuantumGate()
    gate.fit(seiyaku, lost if lost else None)
    gate.save(MODEL_PATH)
    logger.info("モデル保存: %s", MODEL_PATH)

    if backtest:
        _run_backtest(gate, lost)


def _synth_cases(industry: str | None = None, MAX_N: int = 30) -> list[dict]:
    """
    成約案件が少ない場合の合成データ（正常ケース = 財務的に均衡した健全企業）。

    industry=None の場合、主要業種をラウンドロビンで生成する。
    各テンプレートは業種固有の財務比率（設備/利益/減価償却）を反映する。
    """
    import random

    # ── 業種別財務テンプレート ───────────────────────────────────────────────
    # 各エントリ: (profit_range, machines_ratio, depr_ratio, net_ratio, ord_ratio, grade, tags)
    # profit は百万円。千円に変換してから格納。
    _TEMPLATES: dict[str, dict] = {
        "D 建設業": {
            "profit_range": (5.0, 40.0),
            "machines_ratio": (0.5, 1.5),   # 設備は利益の 0.5〜1.5倍（適正範囲）
            "depr_ratio":     (0.08, 0.20),
            "net_ratio":      (0.75, 0.90),
            "ord_ratio":      (0.85, 0.95),
            "grades": ["②B格", "③C格"],
            "tags": ["技術力"],
        },
        "E 製造業": {
            "profit_range": (10.0, 80.0),
            "machines_ratio": (2.0, 5.0),   # 製造業は設備が利益の数倍
            "depr_ratio":     (0.15, 0.30),
            "net_ratio":      (0.70, 0.88),
            "ord_ratio":      (0.80, 0.95),
            "grades": ["①A格", "②B格"],
            "tags": ["技術力", "設備充実"],
        },
        "H 運輸業": {
            "profit_range": (3.0, 25.0),
            "machines_ratio": (3.0, 7.0),   # トラック等で高設備
            "depr_ratio":     (0.20, 0.35),
            "net_ratio":      (0.65, 0.85),
            "ord_ratio":      (0.75, 0.92),
            "grades": ["②B格", "③C格"],
            "tags": ["安定顧客"],
        },
        "P 医療・福祉": {
            "profit_range": (5.0, 50.0),
            "machines_ratio": (0.3, 1.0),   # 設備は比較的低い
            "depr_ratio":     (0.05, 0.15),
            "net_ratio":      (0.80, 0.95),  # 純利益と営業利益が近い
            "ord_ratio":      (0.90, 0.98),
            "grades": ["①A格", "②B格"],
            "tags": ["安定顧客"],
        },
        "K 不動産業": {
            "profit_range": (8.0, 60.0),
            "machines_ratio": (0.1, 0.5),   # 機械装置は少ない
            "depr_ratio":     (0.03, 0.10),
            "net_ratio":      (0.72, 0.90),
            "ord_ratio":      (0.82, 0.95),
            "grades": ["②B格", "③C格"],
            "tags": [],
        },
    }
    _INDUSTRY_ORDER = list(_TEMPLATES.keys())

    def _make_case(ind: str) -> dict:
        t = _TEMPLATES[ind]
        profit = random.uniform(*t["profit_range"])
        return {"inputs": {
            "op_profit":    profit * 1000,
            "depreciation": profit * random.uniform(*t["depr_ratio"]) * 1000,
            "machines":     profit * random.uniform(*t["machines_ratio"]) * 1000,
            "net_income":   profit * random.uniform(*t["net_ratio"]) * 1000,
            "ord_profit":   profit * random.uniform(*t["ord_ratio"]) * 1000,
            "grade":        random.choice(t["grades"]),
            "industry_major": ind,
            "qualitative":  {"strength_tags": list(t["tags"]), "onehot": {k: 1 for k in t["tags"]}},
        }}

    if industry is not None:
        if industry not in _TEMPLATES:
            logger.warning("業種テンプレートなし: %s → デフォルト(D 建設業)を使用", industry)
            industry = "D 建設業"
        return [_make_case(industry) for _ in range(MAX_N)]

    # industry=None: ラウンドロビンで全業種を均等に生成
    cases = []
    for i in range(MAX_N):
        ind = _INDUSTRY_ORDER[i % len(_INDUSTRY_ORDER)]
        cases.append(_make_case(ind))
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

    from quantum_analysis_module import THRESHOLD_SECONDARY_REVIEW as threshold_flag, THRESHOLD_HIGH_RISK as threshold_high
    tp_flag = sum(1 for q in q_risks if q >= threshold_flag)
    tp_high = sum(1 for q in q_risks if q >= threshold_high)
    total = len(q_risks)
    recall_flag = tp_flag / total if total else 0.0
    recall_high = tp_high / total if total else 0.0

    logger.info("=== バックテスト結果 ===")
    logger.info("対象: 高スコア失注 %d件", total)
    logger.info("Q_risk>=%.0f (要再審+) recall: %.2f (目標 ≥ 0.40)", threshold_flag, recall_flag)
    logger.info("Q_risk>=%.0f (高リスク) recall: %.2f", threshold_high, recall_high)
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
