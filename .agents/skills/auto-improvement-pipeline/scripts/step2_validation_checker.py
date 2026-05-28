"""Step 2: 改善案の妥当性を検証するチェッカーエージェント."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


def validate_improvement(
    improvement: dict[str, Any],
    rag_context: str = "",
    source_code_check: bool = True,
) -> dict[str, Any]:
    """
    改善案を妥当性検証する。
    
    チェック項目：
    1. 論理的整合性（既存モジュールとの衝突検査）
    2. エッジケース・例外処理
    3. アーキテクチャ複雑性の判定
    
    Args:
        improvement: Step 1 から来た改善案JSON
        rag_context: Obsidian RAGから取得したコンテキスト
        source_code_check: 実際のソースコード検証を行うか
    
    Returns:
        検証結果JSON: { status: APPROVED|REJECTED, verification_report, critical_flaws, alternative_suggestion }
    """
    
    target_module = improvement.get("target_module")
    description = improvement.get("description", "")
    
    critical_flaws: list[str] = []
    issues: list[str] = []
    
    # 1. ファイルの存在確認
    if target_module and source_code_check:
        file_check = _check_target_file_exists(target_module)
        if not file_check["exists"]:
            issues.append(f"ターゲットファイルが見つかりません: {target_module}")
    
    # 2. RAGコンテキストとの競合検査
    if rag_context:
        rag_issues = _check_rag_conflicts(improvement, rag_context)
        issues.extend(rag_issues)
    
    # 3. データ型・ロジック矛盾の検査
    logic_flaws = _check_logic_flaws(description)
    if logic_flaws:
        critical_flaws.extend(logic_flaws)
    
    # 4. スコープ外（エスケープ）の改善か確認
    if _is_out_of_scope(description):
        issues.append("この改善はスコープが不明確か、複数モジュールに影響する可能性があります")
    
    # 5. 既存テストへの影響評価
    test_impact = _evaluate_test_impact(improvement)
    if test_impact["has_breaking_changes"]:
        critical_flaws.append(f"テスト破壊の可能性: {test_impact['reason']}")
    
    # 承認判定
    status = "APPROVED" if len(critical_flaws) == 0 else "REJECTED"
    
    # 代替案の提示
    alternative = _suggest_alternative(improvement, critical_flaws)
    
    verification_report = _build_verification_report(
        improvement,
        issues,
        critical_flaws,
        test_impact,
    )
    
    return {
        "status": status,
        "verification_report": verification_report,
        "critical_flaws": critical_flaws,
        "alternative_suggestion": alternative,
        "metadata": {
            "issues_count": len(issues),
            "flaws_count": len(critical_flaws),
            "test_breaking": test_impact.get("has_breaking_changes", False),
        },
    }


def _check_target_file_exists(target_module: str) -> dict[str, Any]:
    """ターゲットファイルの存在確認."""
    
    # .py拡張子がない場合、追加
    if not target_module.endswith(".py"):
        target_module = f"{target_module}.py"
    
    # ワークスペースのルートを仮定
    workspace_root = Path(__file__).parent.parent.parent.parent.parent  # tune_lease_55まで遡る
    
    possible_paths = [
        workspace_root / target_module,
        workspace_root / "components" / target_module,
        workspace_root / "scoring" / target_module,
        workspace_root / f"{target_module.replace('.py', '')}" / "main.py",
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_file():
            return {
                "exists": True,
                "path": str(path),
                "absolute": path.resolve(),
            }
    
    return {
        "exists": False,
        "path": target_module,
        "suggestion": "検索パス: " + " | ".join(str(p) for p in possible_paths),
    }


def _check_rag_conflicts(improvement: dict[str, Any], rag_context: str) -> list[str]:
    """
    Obsidian（RAG）から取得した既存仕様との競合を検査。
    
    競合例：
    - 既に対応済みのバグ
    - 既存ルール・仕様と矛盾する提案
    """
    
    issues: list[str] = []
    
    description = improvement.get("description", "").lower()
    
    # 簡易版：RAGコンテキストが「既に対応」「実装済」を示唆していないか
    already_done_keywords = ["実装済", "対応済", "完了", "resolved", "fixed", "done"]
    
    if rag_context:
        rag_lower = rag_context.lower()
        for kw in already_done_keywords:
            if kw in rag_lower and kw in description:
                issues.append(
                    f"この改善は既に対応済みの可能性があります（'{kw}' が見つかりました）"
                )
                break
    
    return issues


def _check_logic_flaws(description: str) -> list[str]:
    """
    ロジック上の穴・エッジケース未処理をチェック。
    """
    
    flaws: list[str] = []
    
    # パターン1: 数値操作でゼロ除算の可能性
    if re.search(r'分割|÷|除算|/\s*\w+|threshold|閾値', description):
        if not re.search(r'ゼロ|空|null|チェック|確認', description):
            flaws.append("数値操作がある場合、ゼロ除算やNULL値のチェックが必要です")
    
    # パターン2: リスト・配列操作でインデックスエラー
    if re.search(r'リスト|配列|\[.+?\]|要素|item', description):
        if not re.search(r'長さ|len\(|サイズ|isEmpty|空チェック', description):
            flaws.append("リスト・配列操作では、長さチェックやemptyチェックが必要です")
    
    # パターン3: 並行処理やグローバル状態の変更
    if re.search(r'グローバル|global|共有|lock|mutex|thread', description):
        if not re.search(r'ロック|同期|atomic|thread-safe', description):
            flaws.append("グローバル状態や共有リソースを変更する場合、スレッドセーフティを確保してください")
    
    # パターン4: マジックナンバー（根拠なき数値）
    if re.search(r'閾値を(\d+)に[変下]げ|値を(\d+)に設定', description):
        if not re.search(r'理由|根拠|データ|テスト|実績', description):
            flaws.append("マジックナンバー（閾値・設定値の数値）には、データ根拠やテスト結果の記載が必要です")
    
    return flaws


def _is_out_of_scope(description: str) -> bool:
    """
    改善のスコープが不明確か、複数モジュールにまたがっていないか確認。
    """
    
    # 複数ファイル・モジュール言及
    module_count = len(re.findall(r'\.py|モジュール|関数|クラス', description))
    
    # スコープ不明確な表現
    vague_keywords = ["全体的", "いろいろ", "あちこち", "いろいろなところ", "～的な"]
    vague_count = sum(1 for kw in vague_keywords if kw in description)
    
    return module_count > 2 or vague_count > 0


def _evaluate_test_impact(improvement: dict[str, Any]) -> dict[str, Any]:
    """
    改善によるテスト破壊の可能性を評価。
    """
    
    description = improvement.get("description", "").lower()
    
    has_breaking = False
    reason = ""
    
    # テスト破壊的な変更の兆候
    breaking_keywords = [
        ("API変更", "public関数・メソッドの引数・戻り値型を変更"),
        ("削除", "既存関数・変数を削除する"),
        ("型変更", "データ型を変更（int→str等）"),
        ("順序変更", "配列・リストの順序を変更"),
        ("スキーマ変更", "DBスキーマを変更"),
    ]
    
    for keyword, impact in breaking_keywords:
        if keyword.lower() in description:
            has_breaking = True
            reason = impact
            break
    
    return {
        "has_breaking_changes": has_breaking,
        "reason": reason,
    }


def _suggest_alternative(improvement: dict[str, Any], critical_flaws: list[str]) -> str | None:
    """代替案を提示."""
    
    if not critical_flaws:
        return None
    
    description = improvement.get("description", "")
    
    # よくある脆い改善への代替案
    if "マジックナンバー" in critical_flaws[0]:
        return (
            "改善案：数値に対して、설정값として constants.py に定数を定義し、"
            "その根拠（データ分析結果やテストケース）をドキュメント化してください。"
        )
    
    if "テスト破壊" in critical_flaws[0]:
        return (
            "改善案：APIを変更する代わりに、新しい関数を追加し、"
            "既存の呼び出し元を段階的に移行させることを検討してください。"
        )
    
    if "スレッドセーフ" in critical_flaws[0]:
        return (
            "改善案：グローバル状態を避け、コンテキストオブジェクト経由で"
            "状態を管理することを検討してください。"
        )
    
    return None


def _build_verification_report(
    improvement: dict[str, Any],
    issues: list[str],
    critical_flaws: list[str],
    test_impact: dict[str, Any],
) -> str:
    """検証レポートの文字列化."""
    
    lines = [
        f"改善案ID: {improvement.get('id', 'N/A')}",
        f"対象: {improvement.get('target_module', '未指定')}",
        f"優先度: {improvement.get('priority', 'N/A')}",
        "",
        "【検証結果】",
    ]
    
    if critical_flaws:
        lines.append(f"⚠️ 致命的な懸念 ({len(critical_flaws)}件):")
        for flaw in critical_flaws:
            lines.append(f"  - {flaw}")
    else:
        lines.append("✅ 致命的な懸念なし")
    
    if issues:
        lines.append(f"\n⚠️ 確認事項 ({len(issues)}件):")
        for issue in issues:
            lines.append(f"  - {issue}")
    
    if test_impact.get("has_breaking_changes"):
        lines.append(f"\n❌ テスト破壊の可能性: {test_impact.get('reason', '')}")
    
    return "\n".join(lines)


def validate_improvements_batch(improvements: list[dict[str, Any]], rag_context: str = "") -> list[dict[str, Any]]:
    """
    複数の改善案をバッチ検証する。
    
    Returns:
        各改善案の検証結果リスト
    """
    
    results: list[dict[str, Any]] = []
    
    for improvement in improvements:
        result = validate_improvement(improvement, rag_context=rag_context)
        results.append(result)
    
    return results


if __name__ == "__main__":
    # テスト用サンプル
    sample_improvement = {
        "id": "REV-001",
        "target_module": "quantum_analysis_module.py",
        "title": "quantum_risk の閾値を35から32に下げる",
        "description": "現在は quantum_risk >= 35 で要注意フラグが立ちます。テストデータを見ると閾値が甘い。32に下げるべき。",
        "reason": "最近のテストケースで、実際のリスク案件が MEDIUM で判定されている",
        "priority": "HIGH",
    }
    
    result = validate_improvement(sample_improvement)
    print(json.dumps(result, ensure_ascii=False, indent=2))
