"""改善案の規模を判定して auto / approval / manual に分類する."""

from __future__ import annotations

import re
from typing import Literal

ChangeSize = Literal["auto", "approval", "manual"]

# ── auto: ホワイトリスト（明示的に一致した場合のみ auto）────────────────
# UI文言・ラベル・テキスト系（大文字小文字不変の日本語もカバー）
_AUTO_TEXT_KEYWORDS: list[str] = [
    "文言", "表示", "ラベル", "テキスト", "説明文", "tooltip", "placeholder",
]

# パラメータ・定数・設定値系
_AUTO_PARAM_KEYWORDS: list[str] = [
    "パラメータ", "定数", "設定値",
]

# 「既存の〜を〜に変更」「〜の誤字」パターン
_AUTO_EXPRESSION_PATTERNS: list[str] = [
    r"既存の.{0,30}を.{0,30}に変更",
    r".{0,20}の誤字",
    r"\btypo\b",
    r"タイポ",
]

# description が 50 文字以内なら auto
_AUTO_MAX_DESCRIPTION_LEN = 50

# ── manual: 新規機能・外部API・インフラ（最優先）────────────────────────
_MANUAL_PATTERNS: list[str] = [
    r"新規.{0,10}機能",
    r"新機能",
    r"新しい.{0,10}モジュール",
    r"外部.{0,3}api",
    r"外部 api",
    r"外部.{0,5}連携",
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
    r"動的.{0,10}エンジン",
    r"金利.{0,10}(エンジン|算出|計算)",
]

# ── approval: 複数ファイル・DB/APIスキーマ・ロジック変更（Dispatchに通知のみ）
_APPROVAL_KEYWORDS: list[str] = [
    "スコアリング", "scoring",
    "lgbm", "lightgbm",
    "quantum", "quantum_risk",
    "auc", "モデル", "model weight",
    "スキーマ", "migration", "マイグレーション",
    "alter table", "create table", "drop table",
    "sqlite", "db schema",
    "api変更", "引数変更", "レスポンス構造",
    "インターフェース変更", "i/f変更",
    "エンドポイント変更", "rest api",
    "認証", "auth", "セキュリティ", "security",
    "権限", "permission",
    "slack", "webhook",
    "ロジック変更", "ロジック", "アルゴリズム変更", "アルゴリズム", "algorithm",
]

# .py ファイルが N 個以上参照 → 複数ファイル変更として approval
_MULTI_FILE_THRESHOLD = 2

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
        "auto"     : ホワイトリスト一致（UI文言・パラメータ・50文字以内の短い修正のみ）
        "approval" : Dispatch通知のみ（複数ファイル・ロジック・DB/API・スコアリング変更）
        "manual"   : 手動実装が必要（新規機能・外部API・インフラ・設計変更）
    """
    text_lower = improvement_text.lower()

    # ── 1. manual チェック（最優先）──────────────────────────────────────
    for pat in _MANUAL_PATTERNS:
        if re.search(pat, text_lower):
            return "manual"

    # ── 2. approval チェック ──────────────────────────────────────────────
    for kw in _APPROVAL_KEYWORDS:
        if kw in text_lower:
            return "approval"

    py_file_count = len(re.findall(r'\b\w+\.py\b', text_lower))
    if py_file_count >= _MULTI_FILE_THRESHOLD:
        return "approval"

    for scoring_file in _SCORING_FILES:
        if scoring_file.lower() in text_lower:
            return "approval"

    # ── 3. auto チェック（ホワイトリスト、明示的な一致のみ）─────────────
    # UI文言・ラベル系（日本語は lower() 後も変わらないので text_lower で検索可）
    for kw in _AUTO_TEXT_KEYWORDS:
        if kw in text_lower:
            return "auto"

    # パラメータ・定数系
    # ただし「閾値」は approval_keywords にも含まれるため、この時点では approval 未該当のもののみ
    for kw in _AUTO_PARAM_KEYWORDS:
        if kw in text_lower:
            return "auto"

    # 「既存の〜を〜に変更」「誤字」表現
    for pat in _AUTO_EXPRESSION_PATTERNS:
        if re.search(pat, text_lower):
            return "auto"

    # 50文字以内の短い修正
    if len(improvement_text.strip()) <= _AUTO_MAX_DESCRIPTION_LEN:
        return "auto"

    # ── 4. デフォルト: manual（安全側）──────────────────────────────────
    return "manual"


def classify_improvement(improvement: dict) -> ChangeSize:
    """
    improvement dict（title + description + target_module を含む）から規模を判定する。

    improvement dict のキー:
        title        : 改善案タイトル
        description  : 改善案詳細
        source_file  : 元ファイルパス（Obsidian ノートなど）
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
        # auto: 文言・ラベル系
        ("UI文言を「保存」から「登録」に変更する", "auto"),
        ("ラベルのテキストを修正する", "auto"),
        ("tooltipの説明文を短縮する", "auto"),
        ("placeholderを変更する", "auto"),
        # auto: パラメータ・定数系
        ("定数MAX_RETRYを5から3に変更する", "auto"),
        ("設定値のデフォルトを変更する", "auto"),
        # auto: 「既存の〜を〜に変更」パターン
        ("既存のエラーメッセージを日本語に変更する", "auto"),
        ("送信ボタンの誤字を修正する", "auto"),
        # auto: 50文字以内の短い修正
        ("送信ボタンを「送信」に統一", "auto"),
        # approval: ロジック変更・複数ファイル
        ("スコアリングのロジックを変更する", "approval"),
        ("スキーマに新しいカラムを追加するマイグレーション", "approval"),
        ("scoring_core.pyのアルゴリズムを修正する", "approval"),
        ("scoring_core.py と total_scorer.py を変更する", "approval"),
        # approval: DB/API/スコアリング
        ("スコアリングの閾値を35から32に変更する", "approval"),
        ("SQLiteのデータベース定義を変更してAPI I/Fを更新する", "approval"),
        # manual: 新規機能・外部API
        ("動的金利エンジンを新規作成する", "manual"),
        ("外部APIとの連携モジュールを新規作成する", "manual"),
        ("新規機能としてSlack連携を追加する大規模な変更", "manual"),
        ("Dockerのインフラ設定を変更する", "manual"),
        ("新機能：レポート自動生成モジュールの追加", "manual"),
    ]
    print("improvement_classifier self-test\n" + "=" * 50)
    all_ok = True
    for text, expected in test_cases:
        result = classify_change_size(text)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_ok = False
        print(f"{status} [{result:>8}] expected={expected:>8} | {text[:55]}")
    print("\n" + ("✅ 全テスト通過" if all_ok else "❌ テスト失敗あり"))
