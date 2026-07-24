"""リース審査の DomainProvider 参照実装。

既存のリース実装（constants / scoring_core / lease_intelligence_tools 等）を薄く
アダプトするだけで、審査ロジックは複製・変更しない。重いモジュール（scoring_core /
numpy）は各メソッド内で遅延 import し、この層自体は import-light に保つ。

「別業種へ載せ替える」とは、このファイルに相当する Provider を業種ごとに書くこと。
"""
from __future__ import annotations

from typing import Any

from screening_domain.contract import Thresholds


class LeaseDomainProvider:
    """リースを審査ドメインの 1 インスタンスとして束ねた参照実装。"""

    name = "lease"

    def thresholds(self) -> Thresholds:
        # 承認ライン集約の単一ソース（constants.py）から供給。値の複製をしない。
        from constants import APPROVAL_LINE, CONDITIONAL_LINE, REVIEW_LINE

        return Thresholds(
            approval=int(APPROVAL_LINE),
            conditional=int(CONDITIONAL_LINE),
            review=int(REVIEW_LINE),
        )

    def score(self, applicant: dict[str, Any], subject: dict[str, Any]) -> dict[str, Any]:
        # 参照実装は既存のクイックスコアリングへ委譲する薄いアダプタ。
        # 入力スキーマの正規化は業種側の責務（ここではそのまま渡す）。
        from scoring_core import run_quick_scoring

        payload = {**subject, **applicant}
        result = run_quick_scoring(payload)
        score = result.get("score") if isinstance(result, dict) else None
        band = self.thresholds().band(score) if isinstance(score, (int, float)) else "unknown"
        return {
            "score": score,
            "band": band,
            "reasons": result.get("reasons", []) if isinstance(result, dict) else [],
            "risk_flags": result.get("risk_flags", []) if isinstance(result, dict) else [],
            "raw": result,
        }

    def search_cases(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        from lease_intelligence_tools import search_cases as _search_cases

        result = _search_cases(query, limit)
        if isinstance(result, dict):
            return result.get("cases", []) or []
        return list(result or [])

    def inspect_policy(self) -> dict[str, Any]:
        from lease_intelligence_tools import inspect_scoring_policy

        return inspect_scoring_policy()

    def coefficients(self) -> dict[str, Any]:
        from lease_intelligence_tools import get_scoring_coefficients

        return get_scoring_coefficients()

    def lookup_rules(self, query: str) -> list[dict[str, Any]]:
        from lease_intelligence_tools import lookup_judgment_rules

        result = lookup_judgment_rules(query)
        if isinstance(result, dict):
            return result.get("rules", []) or []
        return list(result or [])

    def search_knowledge(self, query: str, limit: int = 3) -> dict[str, Any]:
        from lease_intelligence_tools import search_lease_wiki

        return search_lease_wiki(query, limit)

    def benchmark(self, segment: str) -> dict[str, Any]:
        from lease_intelligence_tools import get_industry_benchmark

        return get_industry_benchmark(segment)

    def portfolio_stats(self) -> dict[str, Any]:
        from lease_intelligence_tools import get_portfolio_stats

        return get_portfolio_stats()


# 現状のアクティブドメイン。将来は環境変数 SCREENING_DOMAIN 等で切り替える。
def get_active_provider() -> LeaseDomainProvider:
    """現在アクティブな審査ドメインの Provider を返す（現状はリース固定）。"""
    return LeaseDomainProvider()
