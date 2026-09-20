"""asset_score のAPI境界テスト。

`api/schemas.py` の `asset_score` を `default=50.0` にすると、クライアントが
省略した場合と明示的に 50 を入力した場合が境界で同一になり、
`scoring_core.py:755` の `used_default_asset_score` が決して立たなくなる。
このフラグに依存する4箇所（`scoring_core.py:991` の警告文、
`lease_intelligence_mind.py:1633`、`api/prediction_snapshot.py:72`、
`scoring-auditor` エージェント）が沈黙するため、その退行を防ぐ。
"""
from __future__ import annotations

import pytest

from api.schemas import ScoringRequest


def _used_default_flag(inputs: dict) -> bool:
    """`scoring_core.py:755` と同一の未入力判定式。"""
    raw = inputs.get("asset_score")
    return raw is None or str(raw).strip() == ""


def _resolved_score(inputs: dict) -> float:
    """`api/scoring_full.py` の asset_score 解決と同一式。

    `dict.get` の default はキーが無い時しか効かないため、値が None の場合は
    明示的に既定値へ落とす必要がある（`float(None)` は TypeError）。
    """
    raw = inputs.get("asset_score")
    return 50.0 if raw is None else float(raw)


@pytest.mark.parametrize(
    ("kwargs", "expected_flag", "expected_score"),
    [
        pytest.param({}, True, 50.0, id="未入力"),
        pytest.param({"asset_score": 50.0}, False, 50.0, id="明示的に50"),
        pytest.param({"asset_score": 0}, False, 0.0, id="0は正当なスコア"),
    ],
)
def test_asset_score_boundary_preserves_unset(kwargs, expected_flag, expected_score):
    inputs = ScoringRequest(company_name="テスト工業", **kwargs).model_dump()

    # 契約そのものを固定する。ここが 50.0 に戻ると未入力判定が復元不能になる。
    if not kwargs:
        assert inputs["asset_score"] is None, (
            "ScoringRequest.asset_score の default が None でなくなっている。"
            "センチネル値を入れると「未入力」と「その値を入力」が境界で潰れ、"
            "used_default_asset_score に依存する4箇所が無言で機能しなくなる。"
        )

    assert _used_default_flag(inputs) is expected_flag, (
        f"used_default_asset_score が {expected_flag} になるべきところで "
        f"{_used_default_flag(inputs)} になった（asset_score={inputs['asset_score']!r}）。"
        "api/schemas.py の default か scoring_core.py:755 の判定式を確認すること。"
    )

    assert _resolved_score(inputs) == expected_score, (
        f"scoring へ渡る値が {expected_score} になるべきところで "
        f"{_resolved_score(inputs)} になった。未入力は 50.0 へ補完し、"
        "0.0 は正当なスコアとしてそのまま通すこと（`or` で潰さない）。"
    )
