"""Step 1: チャットログから改善点を抽出・構造化するスクリプト."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
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


# ── target_module 推定用キーワード辞書（フォールバック） ──────────────────
# 改善テキストにファイル名が明示されていない場合（実際のチャットログ由来の
# 改善案はほぼ常にこちら）でも、頻出テーマのキーワードから対象モジュールを
# 推定する。複数候補がある場合はカンマ区切りで列挙する。
# 具体的なキーワードを先頭に、汎用的なキーワード（「紫苑」等）を末尾に置くこと
# （リスト順で最初にマッチしたものが採用されるため）。
_TARGET_MODULE_KEYWORD_MAP: list[tuple[tuple[str, ...], str]] = [
    (("音声入力", "音声応答", "音声で", "音声回答", "voice"), "frontend/src/app/voice-chat/page.tsx"),
    (("千円単位", "千円", "金額の表示単位", "金額入力", "百万単位"), "frontend/src/app/financial/page.tsx, scoring_core.py"),
    (("スコアリング", "AIスコア", "Q_risk", "ドリフト", "キャリブレーション"), "scoring_core.py, total_scorer.py"),
    (("リース負担", "判定ロジック"), "scoring_core.py, rule_manager.py"),
    (("記憶参照", "記憶精度", "過去の会話", "chromaDB", "chroma"), "scripts/build_shion_memory_vector_index.py, mobile_app/rag_daily_maintenance.py"),
    (("マルチエージェント討論", "議論停滞", "意思疎通"), "lease_intelligence_dialogue.py, frontend/src/app/debate/page.tsx"),
    (("lease-wiki-vault", "obsidian-vault", "Vault名"), "mobile_app/obsidian_bridge.py, lease_intelligence_tools.py"),
    (("法定対応年数", "法定耐用年数"), "static_data/useful_life_equipment.json, useful_life_lookup.py"),
    (("画像やファイルを添付", "ファイル添付"), "frontend/src/app/chat/page.tsx"),
    (("自律改善フロー", "自動実行元", "codex"), "scripts/build_codex_auto_queue.py, scripts/execute_codex_queue.py"),
    (("紫苑", "めぶきちゃん", "八奈見さん"), "lease_intelligence_dialogue.py, mobile_app/chat_assistant.py"),
]


def _infer_target_module_from_keywords(text: str) -> str | None:
    """テキストにファイル名が明示されていない場合、頻出テーマのキーワードから
    対象モジュールを推定するフォールバック。"""
    for keywords, module in _TARGET_MODULE_KEYWORD_MAP:
        if any(kw in text for kw in keywords):
            return module
    return None


# ── target_module 推定用リポジトリ全文検索（フォールバック2） ────────────
# キーワード辞書でもカバーできないテーマについて、改善テキストから
# キーワード候補（漢字・カタカナの連続、英数字トークン）を抽出し、リポジトリを
# grepしてヒット数の多いファイルを対象モジュール候補として推定する。
# ひらがなの助詞・活用語尾で自然に区切られるため、簡易的な名詞抽出として機能する。
_KEYWORD_TOKEN_RE = re.compile(r'[一-龠ァ-ヶー]{2,}|[A-Za-z0-9_]{3,}')

_SEARCH_EXTENSIONS = (".py", ".tsx", ".ts", ".jsx", ".js", ".json")
_SEARCH_EXCLUDE_DIRS = (
    ".venv", "node_modules", "__pycache__", ".git", ".next",
    "tests", "test", ".agents", "reports", "knowledge_base",
    ".vector_index", ".cloudrun_bundle", "dist", "build", "static_data",
)
# 改善パイプライン自身のメタ管理スクリプト（REV/改善案テキストを大量に扱うため
# キーワード検索で誤ヒットしやすい）はターゲット候補から除外する
_EXCLUDE_PATH_SUBSTRINGS = (
    "improvement", "codex_queue", "pipeline_ledger", "dispatch_notifier",
)
_MAX_KEYWORDS_FOR_SEARCH = 5
_MAX_CANDIDATE_MODULES = 2
_MAX_FILES_PER_KEYWORD = 30
_GREP_TIMEOUT_SEC = 10


def _find_repo_root() -> Path | None:
    """CLAUDE.md を目印にリポジトリルートを探索する."""
    root = Path(__file__).resolve().parent
    while root != root.parent:
        if (root / "CLAUDE.md").exists():
            return root
        root = root.parent
    return None


def _extract_search_keywords(text: str) -> list[str]:
    """漢字/カタカナの連続・英数字トークンをキーワード候補として抽出する。
    長いトークンほど固有性が高いとみなし、長い順に上位N件を返す。"""
    tokens = _KEYWORD_TOKEN_RE.findall(text)
    seen: set[str] = set()
    unique = [t for t in tokens if not (t in seen or seen.add(t))]
    unique.sort(key=len, reverse=True)
    return unique[:_MAX_KEYWORDS_FOR_SEARCH]


def _infer_target_module_from_repo_search(text: str) -> str | None:
    """キーワード辞書でも解決できない場合の最終フォールバック。
    リポジトリ全文検索でヒット数の多いファイルを対象モジュール候補として返す
    （複数候補はカンマ区切り。あくまで推定であり要確認扱いは変わらない）。"""
    repo_root = _find_repo_root()
    if repo_root is None:
        return None

    keywords = _extract_search_keywords(text)
    if not keywords:
        return None

    scores: dict[str, int] = {}
    for keyword in keywords:
        cmd = ["grep", "-rlF"]
        for ext in _SEARCH_EXTENSIONS:
            cmd.append(f"--include=*{ext}")
        for exclude_dir in _SEARCH_EXCLUDE_DIRS:
            cmd.append(f"--exclude-dir={exclude_dir}")
        cmd.append(keyword)
        cmd.append(str(repo_root))
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=_GREP_TIMEOUT_SEC,
            )
        except Exception:
            continue

        files = [
            f for f in result.stdout.splitlines()
            if f and not any(sub in f.lower() for sub in _EXCLUDE_PATH_SUBSTRINGS)
        ][:_MAX_FILES_PER_KEYWORD]

        weight = len(keyword)  # 長い（＝固有性の高い）キーワードほど重く加点
        for f in files:
            try:
                rel = str(Path(f).relative_to(repo_root))
            except ValueError:
                rel = f
            scores[rel] = scores.get(rel, 0) + weight

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [f for f, _ in ranked[:_MAX_CANDIDATE_MODULES]]
    return ", ".join(top)


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
    if module_match:
        target_module = module_match.group(1)
    else:
        target_module = (
            _infer_target_module_from_keywords(text)
            or _infer_target_module_from_repo_search(text)
        )
    
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
