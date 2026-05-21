"""
ContextBundle — 地域/季節/業況コンテキストをまとめた Pydantic モデル
"""
from __future__ import annotations

import datetime
from pydantic import BaseModel, Field

from api.context.geo_enricher import get_geo_context
from api.context.season_enricher import get_season_context
from api.context.sentiment_enricher import get_sentiment_context


class ContextBundle(BaseModel):
    """討論エージェントへ注入するコンテキストのまとめ。"""

    geo_context: str = Field(default="", description="地域経済圏コンテキスト")
    season_context: str = Field(default="", description="季節性キャッシュフローコンテキスト")
    sentiment_context: str = Field(default="", description="業種別業況感コンテキスト")
    generated_at: str = Field(default="", description="生成日時（ISO 8601）")

    def to_system_prompt_block(self) -> str:
        """各エージェントのシステムプロンプトに追記するブロックを返す。"""
        parts = [
            "## 📊 審査コンテキスト情報（自動配布）",
            self.geo_context,
            self.season_context,
            self.sentiment_context,
            "---",
        ]
        return "\n".join(p for p in parts if p)


def build_context_bundle(
    prefecture: str = "",
    industry: str = "",
    month: int | None = None,
) -> ContextBundle:
    """
    都道府県・業種・月から ContextBundle を生成する。

    Args:
        prefecture: 都道府県名（例: "東京都"）
        industry:   業種名（例: "建設業"）
        month:      月（1〜12）。None の場合は今月。

    Returns:
        ContextBundle インスタンス
    """
    if month is None:
        month = datetime.date.today().month

    return ContextBundle(
        geo_context=get_geo_context(prefecture=prefecture, industry=industry),
        season_context=get_season_context(month=month, industry=industry),
        sentiment_context=get_sentiment_context(industry=industry),
        generated_at=datetime.datetime.now().isoformat(timespec="seconds"),
    )
