#!/usr/bin/env python3
"""
今日のGitコミット・マージPRからDispatch作業ログを生成してObsidianに保存する。
Gemini APIが利用可能な場合は冒頭に作業要約（## 概要）を追加する。
"""
import subprocess
import json
import sys
import os
import re
import requests
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mobile_app.obsidian_bridge import find_vault

_REPO_ROOT = Path(__file__).parent.parent


def _get_gemini_api_key() -> str:
    """Gemini APIキーを取得。環境変数 → secrets.toml の順で探す。"""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    cur = _REPO_ROOT
    for _ in range(5):
        sec = cur / ".streamlit" / "secrets.toml"
        if sec.exists():
            try:
                for line in sec.read_text(encoding="utf-8").splitlines():
                    m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                    if m:
                        return m.group(1)
            except Exception:
                pass
        cur = cur.parent
    return ""


def _gemini_url() -> str:
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )


def summarize_with_gemini(prs: list, commits: list) -> str | None:
    """今日の作業内容をGeminiで3〜5行に要約する。失敗時はNoneを返す。"""
    api_key = _get_gemini_api_key()
    if not api_key:
        print("GEMINI_API_KEY が未設定のため要約をスキップします")
        return None

    pr_lines = "\n".join(
        f"- PR #{pr['number']} {pr['title']}" for pr in prs
    ) if prs else "（なし）"

    seen: set[str] = set()
    commit_lines_list = []
    for c in commits:
        if "|" not in c:
            continue
        subject = c.split("|", 2)[1]
        if subject not in seen:
            seen.add(subject)
            commit_lines_list.append(f"- {subject}")
    commit_lines = "\n".join(commit_lines_list) if commit_lines_list else "（なし）"

    prompt = f"""以下はリースAI開発プロジェクト（tune_lease_55）の今日の作業内容です。
開発者向けに3〜5行で要約してください。箇条書きは使わず、自然な日本語の文章で書いてください。

【マージしたPR】
{pr_lines}

【コミット】
{commit_lines}
"""
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 512,
        },
    }
    try:
        resp = requests.post(
            f"{_gemini_url()}?key={api_key}",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini要約に失敗しました（スキップ）: {e}")
        return None


def get_todays_commits():
    """今日のgitコミット一覧を取得（マージコミット含む）"""
    today = date.today().strftime("%Y-%m-%d")
    result = subprocess.run(
        [
            "git", "log",
            f"--after={today}T00:00:00",
            f"--before={today}T23:59:59",
            "--pretty=format:%H|%s|%an",
        ],
        capture_output=True, text=True,
        cwd=_REPO_ROOT,
    )
    if not result.stdout.strip():
        return []
    return [line for line in result.stdout.strip().split("\n") if line]


def get_todays_merged_prs():
    """今日マージされたPRのタイトルと番号をgh CLIで取得"""
    result = subprocess.run(
        [
            "gh", "pr", "list",
            "--state", "merged",
            "--limit", "20",
            "--json", "number,title,mergedAt,url",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    try:
        prs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    today = date.today().strftime("%Y-%m-%d")
    return [pr for pr in prs if pr.get("mergedAt", "").startswith(today)]


def generate_log_content(
    prs: list, commits: list, date_str: str, summary: str | None = None
) -> str:
    """Obsidianノートの本文を生成。summaryがあれば冒頭に ## 概要 を追加。"""
    lines = [f"# Dispatch 作業ログ {date_str}", ""]

    if summary:
        lines.append("## 概要")
        lines.append(summary)
        lines.append("")

    if prs:
        lines.append("## マージしたPR")
        for pr in prs:
            url = pr.get("url", "")
            lines.append(f"- [PR #{pr['number']} {pr['title']}]({url})")
        lines.append("")

    if commits:
        lines.append("## コミット")
        seen: set[str] = set()
        for c in commits:
            if "|" not in c:
                continue
            parts = c.split("|", 2)
            subject = parts[1] if len(parts) > 1 else c
            if subject not in seen:
                seen.add(subject)
                lines.append(f"- {subject}")
        lines.append("")

    if not prs and not commits:
        lines.append("_今日の作業記録なし_")
        lines.append("")

    lines.append(f"_自動生成: {date_str}_")
    return "\n".join(lines)


def save_to_obsidian(vault_path: Path, content: str, date_str: str) -> Path:
    """Obsidianの作業ログノートに保存（既存ファイルは追記）"""
    log_dir = vault_path / "Projects" / "tune_lease_55" / "作業ログ"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{date_str}.md"

    if log_file.exists():
        existing = log_file.read_text(encoding="utf-8")
        log_file.write_text(existing + "\n\n---\n\n" + content, encoding="utf-8")
    else:
        log_file.write_text(content, encoding="utf-8")

    return log_file


def main() -> None:
    today = date.today().strftime("%Y-%m-%d")

    vault = find_vault()
    if not vault:
        print("Error: Obsidian Vault が見つかりません", file=sys.stderr)
        sys.exit(1)

    print(f"Vault: {vault}")
    prs = get_todays_merged_prs()
    commits = get_todays_commits()
    print(f"マージPR: {len(prs)}件  コミット: {len(commits)}件")

    print("Geminiで要約中...")
    summary = summarize_with_gemini(prs, commits)
    if summary:
        print(f"要約完了: {summary[:60]}...")

    content = generate_log_content(prs, commits, today, summary)
    saved = save_to_obsidian(Path(vault), content, today)
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
