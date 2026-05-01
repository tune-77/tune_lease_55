import numpy as np
import pandas as pd

from mahalanobis_engine import MahalanobisScorer, summarize_distribution_drift


def _make_df(seed: int = 42, n: int = 120, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "nenshu": rng.normal(1000 + shift * 400, 120, n),
            "op_profit": rng.normal(150 + shift * 60, 35, n),
            "total_assets": rng.normal(800 + shift * 250, 90, n),
            "equity_ratio": rng.normal(0.35 + shift * 0.08, 0.04, n),
        }
    )


def test_drift_summary_stable_distribution_is_not_drift():
    features = ["nenshu", "op_profit", "total_assets", "equity_ratio"]
    train_df = _make_df(seed=1, shift=0.0)
    recent_df = _make_df(seed=2, shift=0.02)

    scorer = MahalanobisScorer(features)
    scorer.fit(train_df)

    summary = summarize_distribution_drift(scorer, recent_df[features].values)

    assert summary["is_drift"] is False
    assert set(summary["metrics"].keys()) == {"kl", "js", "wasserstein2", "fisher_rao_proxy"}
    assert all(v >= 0 for v in summary["metrics"].values())


def test_drift_summary_shifted_distribution_is_drift():
    features = ["nenshu", "op_profit", "total_assets", "equity_ratio"]
    train_df = _make_df(seed=10, shift=0.0)
    recent_df = _make_df(seed=11, shift=1.0)

    scorer = MahalanobisScorer(features)
    scorer.fit(train_df)

    summary = summarize_distribution_drift(scorer, recent_df[features].values)

    assert summary["is_drift"] is True
    assert len(summary["triggered_metrics"]) >= 2
