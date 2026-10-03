"""借手スコアが RF からロジスティック式へ切り替わったことが返り値と記録で分かること（スコア値は従来どおり）。"""
import json

import silent_failure_log as sfl

INPUTS = {
    "nenshu": 500, "total_assets": 400, "net_assets": 100, "op_profit": 20, "ord_profit": 18,
    "net_income": 10, "industry_major": "D 建設業", "customer_type": "既存先", "lease_term": 60,
    "acquisition_cost": 30, "asset_score": 60,
}


def test_rf_failure_falls_back_visibly(tmp_path, monkeypatch):
    import sys

    import scoring.predict_one  # noqa: F401 - パッケージが同名関数を公開しているのでモジュールは sys.modules から取る
    import scoring_core as sc

    po = sys.modules["scoring.predict_one"]

    sfl._last_write.clear()
    path = tmp_path / "sf.jsonl"
    monkeypatch.setenv("SILENT_FAILURE_LOG_PATH", str(path))

    def boom(**_kwargs):
        raise RuntimeError("model missing")

    assert sc.run_quick_scoring(dict(INPUTS))["borrower_model"] in {"rf", "logistic"}
    monkeypatch.setattr(po, "predict_one", boom)
    result = sc.run_quick_scoring(dict(INPUTS))
    assert result["borrower_model"] == "logistic"
    assert isinstance(result["score"], (int, float))
    components = [json.loads(line)["component"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert "scoring.core.rf_model" in components
