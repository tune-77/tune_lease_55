"""AI自動修正を小規模・低リスクに限定するポリシー."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_ALLOWED_KEYWORDS = [
    "文言", "ラベル", "説明文", "placeholder", "tooltip", "表示名",
    "faq", "ヘルプ", "ボタン文言", "max_tokens", "timeout", "表示件数",
    "誤字", "タイポ",
]

_DENY_KEYWORDS = [
    "スコアリング", "score", "auc", "モデル", "係数", "閾値", "randomforest",
    "lightgbm", "lgbm", "学習", "再学習", "db", "sqlite", "schema",
    "スキーマ", "migration", "api連携", "api変更", "レスポンス", "認証",
    "セキュリティ", "権限", "外部api", "edinet", "帝国データバンク",
    "ocr", "kubernetes", "docker", "インフラ", "データ移行", "削除",
    "複数ファイル", "横断", "ポートフォリオ", "公平性", "バイアス",
]

_SAFE_FILE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json"}
_DANGEROUS_PATH_PARTS = {
    "models", "data", "migrations", "alembic", ".github", "launchd",
}


def _text(improvement: dict[str, Any]) -> str:
    return " ".join(
        str(improvement.get(key, ""))
        for key in ("title", "description", "reason", "target_module", "canonical_key")
        if improvement.get(key)
    ).lower()


def _body_text(improvement: dict[str, Any]) -> str:
    return " ".join(
        str(improvement.get(key, ""))
        for key in ("title", "description", "reason", "canonical_key")
        if improvement.get(key)
    ).lower()


def _referenced_files(text: str) -> list[str]:
    # tsx/jsx を ts/js より先に置く（ts が tsx にマッチして拡張子が切れるのを防ぐ）
    return re.findall(r"[\w./-]+\.(?:py|tsx|ts|jsx|js|md|json|yaml|yml|toml|plist)", text)


def evaluate_auto_fix_policy(
    improvement: dict[str, Any],
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    AI自動修正を許可するか判定する。

    許可は「明示的に安全な小規模変更」だけ。迷う場合は needs_review。
    """
    text = _text(improvement)
    target_module = str(improvement.get("target_module") or "")
    files = set(_referenced_files(_body_text(improvement)))
    if target_module:
        files.add(target_module)

    reasons: list[str] = []
    required_checks = ["py_compile", "targeted_test"]

    deny_hits = [kw for kw in _DENY_KEYWORDS if kw in text]
    if deny_hits:
        return {
            "auto_fix_allowed": False,
            "reason": "禁止キーワードに該当: " + ", ".join(deny_hits[:5]),
            "risk": "high",
            "max_files": 0,
            "required_checks": required_checks,
        }

    if not files:
        return {
            "auto_fix_allowed": False,
            "reason": "対象ファイル未特定のため手動確認",
            "risk": "medium",
            "max_files": 1,
            "required_checks": required_checks,
        }

    if len(files) > 1:
        return {
            "auto_fix_allowed": False,
            "reason": f"複数ファイル参照のため手動確認: {len(files)} files",
            "risk": "medium",
            "max_files": 1,
            "required_checks": required_checks,
        }

    if files:
        file_name = next(iter(files))
        path = Path(file_name)
        if path.suffix.lower() not in _SAFE_FILE_SUFFIXES:
            return {
                "auto_fix_allowed": False,
                "reason": f"自動修正対象外の拡張子: {path.suffix}",
                "risk": "medium",
                "max_files": 1,
                "required_checks": required_checks,
            }
        if any(part in _DANGEROUS_PATH_PARTS for part in path.parts):
            return {
                "auto_fix_allowed": False,
                "reason": f"重要パス配下のため手動確認: {file_name}",
                "risk": "high",
                "max_files": 1,
                "required_checks": required_checks,
            }

    allowed_hit = any(kw in text for kw in _ALLOWED_KEYWORDS)
    category = str((improvement.get("implementation") or {}).get("category", ""))
    if category == "quick_ui":
        allowed_hit = True
        reasons.append("quick_ui分類")

    if not allowed_hit:
        return {
            "auto_fix_allowed": False,
            "reason": "小規模UI/文言/FAQ/設定値変更として明示されていない",
            "risk": "medium",
            "max_files": 1,
            "required_checks": required_checks,
        }

    reasons.append("小規模・低リスク変更")
    return {
        "auto_fix_allowed": True,
        "reason": "; ".join(reasons),
        "risk": "low",
        "max_files": 1,
        "required_checks": required_checks,
    }


if __name__ == "__main__":
    samples = [
        {"title": "ボタン文言を修正", "target_module": "frontend/src/app/page.tsx"},
        {"title": "スコアリング閾値を変更", "target_module": "scoring_core.py"},
    ]
    for sample in samples:
        print(sample["title"], "=>", evaluate_auto_fix_policy(sample))
