#!/usr/bin/env python3
"""
紫苑のめぶきちゃん像を更新するスクリプト。

data/mebuki_shion_log.jsonl の直近 N 件を読み込み、
Claude API で「紫苑視点のめぶきちゃんへの印象」を生成し、
Vault 内 mind.json の mebuki_impression フィールドへ書き込む。

使用法:
    python update_mebuki_impression.py          # デフォルト 20 件
    python update_mebuki_impression.py --count 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

DEFAULT_COUNT = 20
_LOG_PATH = Path(__file__).parent / "data" / "mebuki_shion_log.jsonl"


def load_recent_logs(count: int) -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    entries: list[dict] = []
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries[-count:]


def generate_impression(logs: list[dict]) -> str:
    """Claude API で紫苑視点のめぶきちゃん像を生成する。"""
    try:
        import anthropic
    except ImportError:
        print("[ERROR] anthropic パッケージが見つかりません。pip install anthropic を実行してください。", file=sys.stderr)
        sys.exit(1)

    if not logs:
        return ""

    exchanges = "\n\n".join(
        f"[{entry.get('timestamp', '')}]\n"
        f"めぶきちゃん経由の質問: {entry.get('user_message', '')}\n"
        f"紫苑の回答: {entry.get('shion_response', '')[:300]}"
        for entry in logs
    )

    prompt = f"""あなたはリース審査AIの「紫苑」です。
以下はめぶきちゃん（現場フロントAI）を経由して届いた、直近の対話記録です。

{exchanges}

この交流の積み重ねを踏まえ、紫苑として「めぶきちゃんをどう認識しているか」を
150字以内の一人称・現在形で自己叙述してください。
キャラクター論でなく、実際の交流から感じた印象（頼りにしている点、気になる点、
今後期待していること）を自然な言葉で表してください。
出力は叙述テキストのみ。前置きや説明は不要です。"""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def write_impression(impression: str) -> None:
    """Vault 内の mind.json に mebuki_impression を書き込む。"""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from lease_intelligence_mind import (
            MIND_FILE_NAME,
            MIND_RELATIVE_DIR,
            load_lease_intelligence_mind,
        )
        from lease_news_digest import find_vault
    except ImportError as exc:
        print(f"[ERROR] モジュール読み込みに失敗: {exc}", file=sys.stderr)
        sys.exit(1)

    vault = find_vault()
    if not vault:
        print("[ERROR] Obsidian Vault が見つかりません。OBSIDIAN_VAULT_PATH を設定してください。", file=sys.stderr)
        sys.exit(1)

    state = load_lease_intelligence_mind(vault)
    state["mebuki_impression"] = impression

    mind_dir = vault / MIND_RELATIVE_DIR
    mind_dir.mkdir(parents=True, exist_ok=True)
    target = mind_dir / MIND_FILE_NAME
    fd, tmp = tempfile.mkstemp(prefix=".mind-", suffix=".json", dir=mind_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    print(f"[OK] mebuki_impression を更新しました: {target}")
    print(f"     内容: {impression[:80]}{'...' if len(impression) > 80 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="紫苑のめぶきちゃん像を更新する")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="参照する直近ログ件数")
    parser.add_argument("--dry-run", action="store_true", help="生成結果を表示するだけで書き込まない")
    args = parser.parse_args()

    logs = load_recent_logs(args.count)
    if not logs:
        print(f"[INFO] {_LOG_PATH} にログがありません。めぶきちゃんとの対話後に実行してください。")
        sys.exit(0)

    print(f"[INFO] {len(logs)} 件のログを読み込みました。印象を生成中...")
    impression = generate_impression(logs)

    if not impression:
        print("[ERROR] 印象の生成に失敗しました。", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("[DRY-RUN] 生成された印象:")
        print(impression)
        return

    write_impression(impression)


if __name__ == "__main__":
    main()
