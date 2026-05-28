"""Step 1: チャットログから改善点を抽出・構造化するスクリプト."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_improvements_from_chat_log(chat_log: str) -> list[dict[str, Any]]:
    """
    チャットログから改善点を抽出し、JSON形式の配列に構造化する。
    
    改善点のトリガー：
    - [改善], [TODO], [FIX], [REFACTOR] などのキーワード
    - または明示的な修正要求文
    
    Args:
        chat_log: チャット全体のテキスト
    
    Returns:
        改善案のJSON配列（最小限の情報のみ）
    """
    
    improvements: list[dict[str, Any]] = []
    improvement_id = 1
    
    # トリガーパターン：[改善], [TODO] など
    trigger_pattern = r'\[(?:改善|TODO|FIX|REFACTOR|バグ|問題)\]\s*(.+?)(?=\n\n|\Z)'
    
    # 明示的な修正要求パターン
    explicit_fix_patterns = [
        r'(?:～を修正|～を改善|～をリファクタ|～の\s*問題を|～を対応)し',
        r'(?:修正すべき|改善すべき|対応すべき)[:：]?\s*(.+?)(?=\n|$)',
    ]
    
    # トリガーパターンで検索
    for match in re.finditer(trigger_pattern, chat_log, re.DOTALL):
        text = match.group(1).strip()
        
        improvement = _parse_improvement_text(text, f"REV-{improvement_id:03d}")
        if improvement:
            improvements.append(improvement)
            improvement_id += 1
    
    # まだヒットがない場合、より柔軟に検索
    if not improvements:
        # より緩いパターン：「～が必要」「～を～へ」など
        lines = chat_log.split('\n')
        for line in lines:
            if any(kw in line for kw in ['修正', '改善', 'バグ', '問題', 'リファクタ', '対応']):
                improvement = _parse_improvement_text(line, f"REV-{improvement_id:03d}")
                if improvement:
                    improvements.append(improvement)
                    improvement_id += 1
    
    return improvements


def _parse_improvement_text(text: str, rev_id: str) -> dict[str, Any] | None:
    """
    単一の改善テキストをパースしてJSON構造化する。
    
    Args:
        text: 改善内容のテキスト（複数行可）
        rev_id: 改善案のID
    
    Returns:
        構造化された改善案、またはNone（パース失敗時）
    """
    
    text = text.strip()
    if not text or len(text) < 10:
        return None
    
    # モジュール名抽出（「～.py」「～モジュール」「～関数」など）
    module_match = re.search(r'([a-zA-Z0-9_.-]+\.py|[a-zA-Z0-9_]+(?:モジュール|関数|クラス))', text)
    target_module = module_match.group(1) if module_match else None
    
    # タイトル抽出：最初の50文字
    title = text.split('\n')[0][:50].strip()
    if not title:
        return None
    
    # 優先度判定
    priority = "MEDIUM"
    if any(kw in text for kw in ['致命的', 'クリティカル', '緊急', 'セキュリティ']):
        priority = "HIGH"
    elif any(kw in text for kw in ['後でいい', 'いずれ', 'オプション', '将来']):
        priority = "LOW"
    
    return {
        "id": rev_id,
        "target_module": target_module,
        "title": title,
        "description": text,
        "reason": _extract_reason(text),
        "priority": priority,
    }


def _extract_reason(text: str) -> str:
    """テキストから「理由」部分を抽出."""
    
    # 理由セクションを検索
    reason_match = re.search(
        r'(?:理由|背景|目的|なぜなら|理由:\s*)(.+?)(?=\n|$)',
        text,
        re.IGNORECASE,
    )
    if reason_match:
        return reason_match.group(1).strip()
    
    # 見つからない場合、最後の1-2文を理由として使用
    sentences = re.split(r'[。\.!！?？\n]+', text.strip())
    if len(sentences) > 1:
        return sentences[-1].strip()
    
    return ""


def validate_extraction_output(improvements: list[dict[str, Any]]) -> bool:
    """抽出結果の基本的な妥当性をチェック."""
    
    if not isinstance(improvements, list):
        return False
    
    for item in improvements:
        required_fields = ["id", "target_module", "title", "description", "reason", "priority"]
        if not all(field in item for field in required_fields):
            return False
        if item["priority"] not in ["HIGH", "MEDIUM", "LOW"]:
            return False
    
    return True


def extract_improvements_as_json(chat_log: str) -> str:
    """
    チャットログを改善案のJSON配列に変換し、JSON文字列として返す。
    （他のエージェント・スクリプトのインプット用）
    
    Args:
        chat_log: チャット全体
    
    Returns:
        JSON配列の文字列（[...] 形式）
    """
    
    improvements = extract_improvements_from_chat_log(chat_log)
    
    # バリデーション
    if not validate_extraction_output(improvements):
        return json.dumps([], ensure_ascii=False)
    
    return json.dumps(improvements, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # テスト用のサンプルチャットログ
    sample_chat = """
    ユーザー: スコアリングロジックの精度が落ちているように感じます。
    
    [改善] quantum_analysis_module.py の金融矛盾検出をもっと厳しくする
    現在は quantum_risk >= 35 で要注意フラグが立ちますが、データを見ると
    閾値が甘すぎるようです。32 に下げるべきです。
    
    理由：最近のテストケースで、実際のリスク案件が MEDIUM で判定されている
    
    ユーザー: 次に、grade_normalizer.py で格付けの正規化ロジックに問題があります。
    [TODO] 「無格付」を正しくハンドルする
    
    現在は例外を落としていますが、本来はデフォルト値を適用すべきです。
    """
    
    result = extract_improvements_as_json(sample_chat)
    print(result)
    
    # JSON妥当性確認
    parsed = json.loads(result)
    print(f"\n抽出件数: {len(parsed)}")
    for item in parsed:
        print(f"  - {item['id']}: {item['title']}")
