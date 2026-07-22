"""AI自動修正を小規模・低リスクに限定するポリシー."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_ALLOWED_KEYWORDS = [
    "文言", "ラベル", "説明文", "placeholder", "tooltip", "表示名",
    "faq", "ヘルプ", "ボタン文言", "max_tokens", "timeout", "表示件数",
    "誤字", "タイポ", "表示",
]

_DENY_KEYWORDS = [
    "スコアリング", "score", "auc", "モデル", "係数", "閾値", "randomforest",
    "lightgbm", "lgbm", "学習", "再学習", "db", "sqlite", "schema",
    "スキーマ", "migration", "api連携", "api変更", "レスポンス", "認証",
    "セキュリティ", "権限", "外部api", "edinet", "帝国データバンク",
    "ocr", "kubernetes", "docker", "インフラ", "データ移行", "削除",
    "複数ファイル", "横断", "ポートフォリオ", "公平性", "バイアス",
    "動いていない", "機能がない", "接続エラー", "通信エラー", "500",
    "登録ボタンがない", "リース期間",
]

_SAFE_FILE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".json"}
_DANGEROUS_PATH_PARTS = {
    "models", "data", "migrations", "alembic", ".github", "launchd",
}

_TARGET_INFERENCE_RULES: list[tuple[tuple[str, ...], str]] = [
    (("自律改善フロー",), "frontend/src/app/system-overview/page.tsx"),
    (("紫苑の記憶システム",), "frontend/src/app/shion-memory-system/page.tsx"),
    (("ホーム画面",), "frontend/src/app/home/page.tsx"),
    (("リースニュースの注目論点",), "frontend/src/app/home/page.tsx"),
    # 主要ページへの自然文からの対象推定（文言・タイポ等の微修正を発火可能にするため）
    (("faqページ",), "frontend/src/app/faq/page.tsx"),
    (("よくある質問",), "frontend/src/app/faq/page.tsx"),
    (("ヘルプページ",), "frontend/src/app/help/page.tsx"),
    (("ヘルプ画面",), "frontend/src/app/help/page.tsx"),
    (("チャット画面",), "frontend/src/app/chat/page.tsx"),
    (("案件一覧",), "frontend/src/app/cases/page.tsx"),
    (("業界統計",), "frontend/src/app/industry-stats/page.tsx"),
    (("改善ログ",), "frontend/src/app/improvement-log/page.tsx"),
    (("競合分析",), "frontend/src/app/competitor/page.tsx"),
    (("金利画面",), "frontend/src/app/interest/page.tsx"),
    (("財務画面",), "frontend/src/app/finance/page.tsx"),
    (("事業計画チェック",), "frontend/src/app/business-plan-check/page.tsx"),
]


# canonical_key は不透明な識別ハッシュ（例: misc_a92f18c9bdb3）であり、
# 禁止/許可キーワードやファイル参照の走査対象に含めると、ハッシュ中の
# 偶然の部分文字列（"db"/"api"/"500" 等）に誤マッチして正当な候補まで
# DENY されてしまう。キーワード走査には意味のあるテキストだけを使う。
def _text(improvement: dict[str, Any]) -> str:
    return " ".join(
        str(improvement.get(key, ""))
        for key in ("title", "description", "reason", "target_module")
        if improvement.get(key)
    ).lower()


def _body_text(improvement: dict[str, Any]) -> str:
    return " ".join(
        str(improvement.get(key, ""))
        for key in ("title", "description", "reason")
        if improvement.get(key)
    ).lower()


def _referenced_files(text: str) -> list[str]:
    # tsx/jsx を ts/js より先に置く（ts が tsx にマッチして拡張子が切れるのを防ぐ）
    return re.findall(r"[\w./-]+\.(?:py|tsx|ts|jsx|js|md|json|yaml|yml|toml|plist)", text)


def infer_target_module(
    improvement: dict[str, Any],
    workspace_root: str | Path | None = None,
) -> str | None:
    """自然文の改善案から、低リスク候補の対象ファイルだけを控えめに推定する."""
    if improvement.get("target_module"):
        return str(improvement.get("target_module"))

    text = _text(improvement)
    root = Path(workspace_root) if workspace_root else None
    for keywords, target in _TARGET_INFERENCE_RULES:
        if not all(keyword.lower() in text for keyword in keywords):
            continue
        if root is not None and not (root / target).is_file():
            continue
        return target
    return None


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
    inferred_target_module = infer_target_module(improvement, workspace_root)
    if not target_module and inferred_target_module:
        target_module = inferred_target_module
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
    result = {
        "auto_fix_allowed": True,
        "reason": "; ".join(reasons),
        "risk": "low",
        "max_files": 1,
        "required_checks": required_checks,
    }
    if inferred_target_module and not improvement.get("target_module"):
        result["inferred_target_module"] = inferred_target_module
    return result


def classify_quick_fix(
    improvement: dict[str, Any],
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """改善候補が「対象ファイルの明確な小規模修正（quick_ui）」かを判定する供給用ヘルパー。

    トリアージやチャットが、抽象的な要望を自動修正パイプラインが発火できる形へ
    整形するために使う。auto_fix_policy を単一の真実源として再利用し、判定が通れば
    そのまま候補源へ追記できる `candidate` を返す。

    戻り値:
        {
          "is_quick_fix": bool,          # 自動修正の発火対象か
          "target_module": str | None,   # 特定/推定された単一対象ファイル
          "risk": str,                   # low / medium / high
          "reason": str,
          "candidate": dict | None,      # is_quick_fix のとき、追記可能な候補dict
        }
    """
    policy = evaluate_auto_fix_policy(improvement, workspace_root)
    target = (
        str(improvement.get("target_module") or "").strip()
        or policy.get("inferred_target_module")
        or None
    )
    is_quick_fix = bool(policy.get("auto_fix_allowed") and target)

    candidate: dict[str, Any] | None = None
    if is_quick_fix:
        candidate = {
            "title": str(improvement.get("title") or "").strip(),
            "description": str(improvement.get("description") or improvement.get("detail") or "").strip(),
            "reason": str(improvement.get("reason") or "").strip(),
            "target_module": target,
            "implementation": {"category": "quick_ui"},
        }

    return {
        "is_quick_fix": is_quick_fix,
        "target_module": target,
        "risk": policy.get("risk", "medium"),
        "reason": policy.get("reason", ""),
        "candidate": candidate,
    }


if __name__ == "__main__":
    samples = [
        {"title": "ボタン文言を修正", "target_module": "frontend/src/app/page.tsx"},
        {"title": "スコアリング閾値を変更", "target_module": "scoring_core.py"},
    ]
    for sample in samples:
        print(sample["title"], "=>", evaluate_auto_fix_policy(sample))
