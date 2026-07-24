"""審査ドメインのシーム（DomainProvider 契約・Thresholds・ツールレジストリ）のテスト。

重いモジュール（scoring_core / numpy）に依存しない。契約の形と配線だけを検証する。
"""

from screening_domain import DomainProvider, Thresholds, build_tool_registry


def test_thresholds_band():
    t = Thresholds(approval=71, conditional=60, review=40)
    assert t.band(72) == "approved"
    assert t.band(71) == "approved"
    assert t.band(65) == "conditional"
    assert t.band(45) == "review"
    assert t.band(39) == "rejected"
    assert t.band_label_ja(72) == "承認"
    assert t.band_label_ja(65) == "条件付き承認"
    assert t.band_label_ja(10) == "否決"


class _FakeProvider:
    """契約を満たす軽量スタブ（呼ばれた引数を記録）。"""

    name = "fake"

    def __init__(self):
        self.calls: list[tuple] = []

    def thresholds(self):
        return Thresholds(70, 55, 35)

    def score(self, applicant, subject):
        self.calls.append(("score", applicant, subject))
        return {"score": 80, "band": "approved", "reasons": [], "risk_flags": []}

    def search_cases(self, query, limit=5):
        self.calls.append(("search_cases", query, limit))
        return [{"id": "c1"}]

    def inspect_policy(self):
        self.calls.append(("inspect_policy",))
        return {"ok": True}

    def coefficients(self):
        return {"w": 1}

    def lookup_rules(self, query):
        self.calls.append(("lookup_rules", query))
        return [{"rule": "r1"}]

    def search_knowledge(self, query, limit=3):
        self.calls.append(("search_knowledge", query, limit))
        return {"results": []}

    def benchmark(self, segment):
        self.calls.append(("benchmark", segment))
        return {"segment": segment}

    def portfolio_stats(self):
        return {"total": 0}


def test_fake_provider_satisfies_protocol():
    # runtime_checkable Protocol はメソッド存在を検査する
    assert isinstance(_FakeProvider(), DomainProvider)


def test_tool_registry_routes_to_provider():
    provider = _FakeProvider()
    registry = build_tool_registry(provider)

    # ドメイン依存ツールが揃っている
    for name in ("get_score_detail", "search_cases", "inspect_scoring_policy",
                 "lookup_judgment_rules", "search_knowledge", "get_industry_benchmark"):
        assert name in registry

    assert registry["search_cases"]({"query": "残価", "limit": 3}) == [{"id": "c1"}]
    assert ("search_cases", "残価", 3) in provider.calls

    registry["get_industry_benchmark"]({"segment": "建設業"})
    assert ("benchmark", "建設業") in provider.calls

    out = registry["get_score_detail"]({"applicant": {"a": 1}, "subject": {"s": 2}})
    assert out["band"] == "approved"
    assert ("score", {"a": 1}, {"s": 2}) in provider.calls


def test_lease_provider_thresholds_come_from_constants():
    # 参照実装の thresholds() は constants.py の単一ソースを返す（値の複製なし）
    import constants
    from screening_domain.lease_provider import LeaseDomainProvider

    t = LeaseDomainProvider().thresholds()
    assert t.approval == constants.APPROVAL_LINE
    assert t.conditional == constants.CONDITIONAL_LINE
    assert t.review == constants.REVIEW_LINE
