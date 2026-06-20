"""
改善ルールのスキーマ定義。

REVパターン分類（台帳調査 2026-06-20）:
  patch_json      JSONファイルの特定要素を直接書き換え（耐用年数・マスタデータ）
  add_api_field   Pydantic/SQLAlchemy モデルへのフィールド追加（コード生成）
  ui_text         フロントエンド UI 文言・ラベルの修正
  scoring_weight  スコアリング係数・閾値の変更
  endpoint_add    FastAPI エンドポイントの追加
  config_value    定数・設定値（constants.py 等）の変更
  llm_diff        自然文指示を LLM に渡して diff を生成（複雑な変更のフォールバック）
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from dataclasses import dataclass, field


RuleType = Literal[
    "patch_json",
    "add_api_field",
    "ui_text",
    "scoring_weight",
    "endpoint_add",
    "config_value",
    "llm_diff",
]


@dataclass
class ImprovementRule:
    """改善台帳の 1 エントリを表す。JSON から dict として渡される想定。"""

    rev_id: str
    type: RuleType
    description: str

    # patch_json 用
    target: Optional[str] = None       # プロジェクトルートからの相対パス
    match: Optional[dict] = None       # 対象要素の特定条件 {field: value, ...}
    patch: Optional[dict] = None       # 上書きする {field: new_value, ...}
    match_path: Optional[str] = None   # JSONPath 風パス指定（例: "categories[*].items[*]"）

    # add_api_field / config_value 用
    model_name: Optional[str] = None   # 対象モデル名
    field_name: Optional[str] = None   # 追加/変更フィールド名
    field_type: Optional[str] = None   # 型文字列（例: "Optional[float] = None"）

    # llm_diff 用
    instruction: Optional[str] = None  # LLM へ渡す自然文の指示

    # 共通メタ
    affected_files: list[str] = field(default_factory=list)
    risk: str = "low"                  # low / medium / high
    auto_fix_allowed: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "ImprovementRule":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class ApplyResult:
    rev_id: str
    success: bool
    message: str
    changed_file: Optional[str] = None
    diff_summary: Optional[str] = None
