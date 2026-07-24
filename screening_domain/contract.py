"""審査ドメインの最小契約（DomainProvider）とツールレジストリ雛形。

紫苑のドメイン依存ツールは、すべてこの契約のメソッド呼び出しに還元できる。
新業種はこの Protocol を実装するだけで紫苑が「住める」ようになり、リース実装は
その参照実装（1 インスタンス）に格下げされる。

この層は import-light に保つ（重い scoring_core / numpy を読み込まない）。実装側の
Provider がメソッド内で遅延 import する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Thresholds:
    """審査の判定ライン。値の出所はドメインごと（リースは constants.py）。

    人格側・レポート側はこの 1 オブジェクト経由で帯を判定し、業種ごとに散らばった
    数値ハードコードを避ける（承認ライン集約の受け皿）。
    """

    approval: int
    conditional: int
    review: int

    def band(self, score: float) -> str:
        """スコアを判定帯に変換する。approved / conditional / review / rejected。"""
        if score >= self.approval:
            return "approved"
        if score >= self.conditional:
            return "conditional"
        if score >= self.review:
            return "review"
        return "rejected"

    def band_label_ja(self, score: float) -> str:
        return {
            "approved": "承認",
            "conditional": "条件付き承認",
            "review": "要審査",
            "rejected": "否決",
        }[self.band(score)]


@runtime_checkable
class DomainProvider(Protocol):
    """紫苑が住む「審査ドメイン」が提供すべき最小契約。

    メソッドの戻り値はプレーンな dict/list（人格側は業種の型に依存しない）。
    """

    name: str

    def thresholds(self) -> Thresholds:
        """判定ライン（承認 / 条件付き / 要審査）。"""
        ...

    def score(self, applicant: dict[str, Any], subject: dict[str, Any]) -> dict[str, Any]:
        """申込主体＋対象を採点。{score, band, reasons[], risk_flags[]} を返す。"""
        ...

    def search_cases(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """類似の過去案件を検索。"""
        ...

    def inspect_policy(self) -> dict[str, Any]:
        """スコアリング方針の透明化（どの経路でどう決まるか）。"""
        ...

    def coefficients(self) -> dict[str, Any]:
        """モデル係数・重み。"""
        ...

    def lookup_rules(self, query: str) -> list[dict[str, Any]]:
        """判断ルール・閾値の参照。"""
        ...

    def search_knowledge(self, query: str, limit: int = 3) -> dict[str, Any]:
        """業種ナレッジの RAG 検索。"""
        ...

    def benchmark(self, segment: str) -> dict[str, Any]:
        """業種・セグメントのベンチマーク。"""
        ...

    def portfolio_stats(self) -> dict[str, Any]:
        """ポートフォリオ集計。"""
        ...


# 紫苑のツール名 → DomainProvider メソッドへの写像。
# execute_tool（現状 20 分岐の if/elif）を将来ここ経由に差し替える差し替え口の雛形。
# ドメイン非依存ツール（get_pipeline_status / search_obsidian / git 系など）は
# 対象外＝人格側に残す。ここに載るのは「ドメインの中身に触れるツール」だけ。
def build_tool_registry(provider: DomainProvider) -> dict[str, Callable[[dict[str, Any]], Any]]:
    """provider を束ねたツールレジストリを返す（name → 呼び出し可能）。"""
    return {
        "get_score_detail": lambda args: provider.score(
            args.get("applicant", {}) or {}, args.get("subject", {}) or {}
        ),
        "search_cases": lambda args: provider.search_cases(
            str(args.get("query", "")), int(args.get("limit", 5) or 5)
        ),
        "compare_similar_cases": lambda args: provider.search_cases(
            str(args.get("query", "")), int(args.get("limit", 5) or 5)
        ),
        "inspect_scoring_policy": lambda args: provider.inspect_policy(),
        "get_scoring_coefficients": lambda args: provider.coefficients(),
        "lookup_judgment_rules": lambda args: provider.lookup_rules(str(args.get("query", ""))),
        "search_knowledge": lambda args: provider.search_knowledge(
            str(args.get("query", "")), int(args.get("limit", 3) or 3)
        ),
        "get_industry_benchmark": lambda args: provider.benchmark(str(args.get("segment", ""))),
        "get_portfolio_stats": lambda args: provider.portfolio_stats(),
    }


# レジストリが対象とする「ドメイン依存」ツール名。人格側に残す中立ツールと区別する。
DOMAIN_TOOL_NAMES = frozenset(
    {
        "get_score_detail",
        "search_cases",
        "compare_similar_cases",
        "inspect_scoring_policy",
        "get_scoring_coefficients",
        "lookup_judgment_rules",
        "search_knowledge",
        "get_industry_benchmark",
        "get_portfolio_stats",
    }
)
