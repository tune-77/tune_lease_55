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
    
    # モジュール名抽出（.py/.tsx/.ts/.js/.jsx/.json や「～モジュール」「～関数」など）
    module_match = re.search(r'([a-zA-Z0-9_./-]+\.(?:py|tsx|ts|jsx|js|json)|[a-zA-Z0-9_]+(?:モジュール|関数|クラス))', text)
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


def _run_demo_mode(demo_dir: "Path") -> None:
    """--demo フラグ時: demo_chat_logs.json から改善点を抽出して demo_step1_output.json に書き出す."""
    import sys
    from pathlib import Path as _Path

    chat_log_path = demo_dir / "demo_chat_logs.json"
    output_path = demo_dir / "demo_step1_output.json"

    if not chat_log_path.exists():
        print(f"[Step1-DEMO] エラー: {chat_log_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    with chat_log_path.open(encoding="utf-8") as f:
        chats = json.load(f)

    improvements: list[dict[str, Any]] = []
    for i, chat in enumerate(chats, start=1):
        issue = chat.get("detected_issue", {})
        demo_id = f"DEMO-{i:03d}"
        title = chat.get("title", "")
        description = (
            f"{title}\n"
            f"資産: {issue.get('asset') or issue.get('asset_category', '不明')}\n"
            f"問題フィールド: {issue.get('field', '不明')}\n"
            f"誤った値: {issue.get('wrong_value')} → 正しい値: {issue.get('correct_value')}\n"
            f"根拠: {issue.get('source', '不明')}\n"
            f"影響: {issue.get('impact', '不明')}"
        )
        target = issue.get("field", "")
        # フィールドからターゲットモジュールを推定
        if "lease_term" in target or "useful_life" in target:
            target_module = "useful_life_equipment.json"
        elif "depreciation" in target or "coefficient" in target.lower():
            target_module = "coeff_auto.json"
        elif "liquidity" in target or "scoring" in target:
            target_module = "scoring_core.py"
        else:
            target_module = "static_data/knowledge_base.json"

        improvements.append({
            "id": demo_id,
            "target_module": target_module,
            "title": title,
            "description": description,
            "reason": issue.get("impact", ""),
            "priority": "HIGH" if issue.get("type") in ("data_error", "missing_data") else "MEDIUM",
            "issue_type": issue.get("type", "unknown"),
            "source": issue.get("source", ""),
            "wrong_value": issue.get("wrong_value"),
            "correct_value": issue.get("correct_value"),
            "chat_id": chat.get("id", ""),
        })

    output_path.write_text(json.dumps(improvements, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Step1-DEMO] {len(improvements)} 件の改善案を抽出しました → {output_path}")
    for imp in improvements:
        print(f"  {imp['id']}: {imp['title']} ({imp['priority']})")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Step1: チャットログから改善点を抽出")
    parser.add_argument("--demo", action="store_true", help="デモモード: demo_chat_logs.json を入力とする")
    args = parser.parse_args()

    if args.demo:
        # スクリプトの場所から workspace root を特定して demo ディレクトリを解決
        _script_dir = Path(__file__).resolve().parent
        _root = _script_dir
        while _root != _root.parent:
            if (_root / "CLAUDE.md").exists():
                break
            _root = _root.parent
        _demo_dir = _root / "scripts" / "demo"
        _run_demo_mode(_demo_dir)
    else:
        # 通常モード: サンプルチャットログでテスト
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
        parsed = json.loads(result)
        print(f"\n抽出件数: {len(parsed)}")
        for item in parsed:
            print(f"  - {item['id']}: {item['title']}")
