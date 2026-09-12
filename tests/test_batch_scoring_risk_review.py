import sys
from types import SimpleNamespace

import components.batch_scoring as batch_scoring


def test_asset_weighting_preserves_forced_review(monkeypatch):
    scorer_result = {
        "score": 95.0,
        "score_base": 95.0,
        "score_borrower": 95.0,
        "approval_line": 71,
        "hantei": "要審議",
        "risk_review_required": True,
        "risk_review_reasons": ["Q_risk 強警戒（65.0）"],
        "user_equity_ratio": 20.0,
        "user_op_margin": 5.0,
    }
    monkeypatch.setitem(
        sys.modules,
        "scoring_core",
        SimpleNamespace(run_quick_scoring=lambda _inputs: scorer_result.copy()),
    )
    monkeypatch.setitem(
        sys.modules,
        "category_config",
        SimpleNamespace(
            ASSET_ID_TO_CATEGORY={"vehicle": "車両"},
            ASSET_WEIGHT={"車両": {"asset_w": 0.4, "obligor_w": 0.6}},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "asset_scorer",
        SimpleNamespace(calc_asset_score=lambda *_args: {"total_score": 80.0, "grade": "A"}),
    )
    monkeypatch.setattr(
        batch_scoring,
        "_compute_auto_bench_ind_scores",
        lambda _inputs: (50.0, 50.0, "test"),
    )

    output = batch_scoring._score_one(
        {
            "格付": "4-6",
            "物件ID（任意）": "vehicle",
            "リース期間(月)": 60,
            "取得価格(百万円)": 10,
            "総資産(百万円)": 100,
        }
    )

    assert output["UI表示用"]["総合スコア"] >= 71
    assert output["UI表示用"]["判定"] == "要審議"
    stored = output["DB保存用"]["result"]
    assert stored["score_based_hantei"] == "承認圏内"
    assert stored["risk_review_required"] is True
