"""改善案の規模を判定して auto / approval / manual に分類する."""

from __future__ import annotations

import re
from typing import Literal

ChangeSize = Literal["auto", "approval", "manual"]

# ── 自動マージ対象（auto）のシグナル ─────────────────────────────────────

_AUTO_TITLE_PATTERNS: list[str] = [
    r"文言",
    r"ラベル",
    r"テキスト変更",
    r"ui.{0,10}変更",
    r"表示.{0,10}変更",
    r"タイポ",
    r"typo",
    r"コメント",
    r"ログ",          # ログ出力・ログ変更を含む軽微な変更
    r"パラメータ.{0,10}調整",
    r"定数.{0,10}変更",
    r"閾値.{0,8}微調整",
    r"timeout.{0,10}変更",
    r"短縮",
    r"置き換え",
    r"置換",
    r"リネーム",
    r"rename",
    r"print.{0,10}(変更|置換|logger)",
    r"冗長.{0,15}(削除|短縮|修正)",
    r"軽微",
    r"minor",
    r"小さな",
]

_AUTO_DESCRIPTION_PATTERNS: list[str] = [
    r"1ファイル",
    r"単一ファイル",
    r"50行以内",
    r"数行の",
    r"軽微",
]

# ── 手動が必要（manual）のシグナル ───────────────────────────────────────

_MANUAL_PATTERNS: list[str] = [
    r"新規.{0,10}機能",
    r"新機能",
    r"新しい.{0,10}モジュール",
    r"外部api",
    r"外部 api",
    r"third.{0,5}party",
    r"インフラ",
    r"docker",
    r"kubernetes",
    r"k8s",
    r"terraform",
    r"ci.{0,5}cd",
    r"デプロイ.{0,10}変更",
    r"大規模",
    r"リアーキテクチャ",
    r"re.?architect",
    r"全面的な書き直し",
    r"マイクロサービス",
    r"新規.{0,5}db",
    r"新しい.{0,5}テーブル.{0,5}追加",
    r"新規.*api.*エンドポイント.*追加",
]

# ── 承認待ち（approval）のシグナル ───────────────────────────────────────

_APPROVAL_KEYWORDS: list[str] = [
    "スコアリング",
    "scoring",
    "lgbm",
    "lightgbm",
    "quantum",
    "quantum_risk",
    "auc",
    "モデル",
    "model weight",
    "閾値",
    "threshold",
    "スキーマ",
    "migration",
    "マイグレーション",
    "alter table",
    "create table",
    "drop table",
    "sqlite",
    "db schema",
    "api変更",
    "引数変更",
    "レスポンス構造",
    "インターフェース変更",
    "i/f変更",
    "エンドポイント変更",
    "rest api",
    "認証",
    "auth",
    "セキュリティ",
    "security",
    "権限",
    "permission",
    "slack",
    "webhook",
]

_MULTI_FILE_THRESHOLD = 3  # .py ファイルが N 個以上参照で approval

_SCORING_FILES: frozenset[str] = frozenset({
    "quantum_analysis_module.py",
    "scoring_core.py",
    "total_scorer.py",
    "asset_scorer.py",
    "category_config.py",
    "industry_hybrid_model.py",
    "rule_manager.py",
    "coeff_definitions.py",
})


def classify_change_size(improvement_text: str) -> ChangeSize:
    """
    改善案テキスト（title + description をまとめた文字列）から変更規模を判定する。

    Returns:
        "auto"     : 自動マージ対象（1ファイル・UI文言・パラメータ調整・50行以内）
        "approval" : 承認待ち（複数ファイル・ロジック・DB/API・スコアリング変更）
        "manual"   : 手動実装が必要（新規大規模機能・外部API・インフラ変更）
    """
    text_lower = improvement_text.lower()

    # ── manual チェック（最優先）─────────────────────────────────────────
    for pat in _MANUAL_PATTERNS:
        if re.search(pat, text_lower):
            return "manual"

    # ── approval チェック ─────────────────────────────────────────────────
    for kw in _APPROVAL_KEYWORDS:
        if kw in text_lower:
            return "approval"

    py_file_count = len(re.findall(r'\b\w+\.py\b', text_lower))
    if py_file_count >= _MULTI_FILE_THRESHOLD:
        return "approval"

    for scoring_file in _SCORING_FILES:
        if scoring_file.lower() in text_lower:
            return "approval"

    # ── auto チェック ────────────────────────────────────────────────────
    for pat in _AUTO_TITLE_PATTERNS + _AUTO_DESCRIPTION_PATTERNS:
        if re.search(pat, text_lower):
            return "auto"

    # 50行以内の数字が含まれる場合
    if re.search(r'[1-4][0-9]行|50行|数行|[1-9]行', text_lower):
        return "auto"

    # デフォルト: 複数ファイル（2ファイル）でも approval 側へ
    if py_file_count >= 2:
        return "approval"

    # 単一ファイル変更と推定できる場合は auto
    if py_file_count == 1:
        return "auto"

    # 判断できない場合は安全側（approval）
    return "approval"


def classify_improvement(improvement: dict) -> ChangeSize:
    """
    improvement dict（title + description + source_file を含む）から規模を判定する。

    improvement dict のキー:
        title       : 改善案タイトル
        description : 改善案詳細
        source_file : 元ファイルパス（Obsidian ノートなど）
        target_module: 対象モジュール名
    """
    parts = [
        improvement.get("title", ""),
        improvement.get("description", ""),
        improvement.get("target_module", "") or "",
    ]
    combined = " ".join(str(p) for p in parts if p)
    return classify_change_size(combined)


# ── CLI テスト ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases: list[tuple[str, ChangeSize]] = [
        ("UI文言を「保存」から「登録」に変更する", "auto"),
        ("ログ出力の冗長な文字列を短縮する", "auto"),
        ("スコアリングの閾値を35から32に変更する", "approval"),
        ("スキーマに新しいカラムを追加するマイグレーション", "approval"),
        ("SQLiteのデータベース定義を変更してAPI I/Fを更新する", "approval"),
        ("Slack連携の新規APIエンドポイントを追加する大規模機能", "manual"),
        ("外部APIとの連携モジュールを新規作成する", "manual"),
        ("Dockerのインフラ設定を変更する", "manual"),
    ]
    print("improvement_classifier self-test\n" + "=" * 40)
    all_ok = True
    for text, expected in test_cases:
        result = classify_change_size(text)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_ok = False
        print(f"{status} [{result:>8}] expected={expected:>8} | {text[:50]}")
    print("\n" + ("✅ 全テスト通過" if all_ok else "❌ テスト失敗あり"))
