"""審査ドメインのシーム（差し替え口）。

紫苑（人格エンジン）を業種横断で使い回すための境界。人格側は「審査ドメインが
何を提供するか」だけに依存し、リース固有の実装（scoring_core / constants /
lease_intelligence_tools 等）は `LeaseDomainProvider` という 1 インスタンスに束ねる。
別業種へ載せ替えるときは、この契約（DomainProvider）を実装した新しい Provider を
差し込むだけでよい。

現状はスケルトン（契約＋リース参照実装＋ツールレジストリの雛形）であり、ライブの
`lease_intelligence_tools.execute_tool` はまだ差し替えていない。次段でそこを
`build_tool_registry(get_active_provider())` に配線する。
"""

from screening_domain.contract import (
    DomainProvider,
    Thresholds,
    build_tool_registry,
)

__all__ = ["DomainProvider", "Thresholds", "build_tool_registry"]
