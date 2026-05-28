"""Claude Code CLI を使って改善案を実装し、PRを作成・マージするランナー."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _find_claude_bin() -> str | None:
    """which claude でパスを取得する."""
    try:
        result = subprocess.run(
            ["which", "claude"],
            capture_output=True, text=True, timeout=5,
        )
        path = result.stdout.strip()
        return path if path else None
    except Exception:
        return None


def _find_workspace_root() -> Path:
    """CLAUDE.md または tune_lease_55.py が存在する祖先ディレクトリを返す."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "CLAUDE.md").exists() or (current / "tune_lease_55.py").exists():
            return current
        current = current.parent
    return Path.cwd()


def _extract_pr_number(text: str) -> int | None:
    """gh 出力や URL から PR 番号を抽出する."""
    m = re.search(r'/pull/(\d+)', text)
    if m:
        return int(m.group(1))
    m = re.search(r'#(\d+)', text)
    if m:
        return int(m.group(1))
    return None


def _build_claude_prompt(improvement: dict[str, Any]) -> str:
    """improvement dict から claude --print に渡すプロンプトを組み立てる."""
    title = improvement.get("title", "改善案")
    description = improvement.get("description", "")
    source_file = improvement.get("source_file", "")
    target_module = improvement.get("target_module", "")
    reason = improvement.get("reason", "")

    parts = [
        f"以下の改善案を実装してください。",
        f"",
        f"## タイトル",
        f"{title}",
        f"",
        f"## 詳細",
        f"{description}",
    ]
    if reason:
        parts += ["", "## 理由", reason]
    if target_module:
        parts += ["", "## 対象モジュール", target_module]
    if source_file:
        parts += ["", "## 出典ファイル", source_file]
    parts += [
        "",
        "## 実装上の注意",
        "- セキュリティの脆弱性（SQLインジェクション・XSS等）を絶対に導入しないでください",
        "- 既存のテストを壊さないでください",
        "- 変更は最小限に留め、指定された改善のみを実施してください",
        "- 実装が完了したら変更したファイルを git add してコミットしてください",
    ]
    return "\n".join(parts)


def run_claude_agent(improvement: dict[str, Any], size: str) -> dict[str, Any]:
    """
    Claude Code CLI で改善案を実装し、PRを作成する。

    Args:
        improvement: 改善案 dict（title, description, source_file, target_module を含む）
        size:        "auto" | "approval" | "manual"

    Returns:
        {
            "success": bool,
            "pr_number": int | None,
            "pr_url": str | None,
            "merged": bool,
            "message": str,
        }
    """
    claude_bin = _find_claude_bin()
    if not claude_bin:
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "claude CLI が見つかりません（which claude が空）",
        }

    workspace = _find_workspace_root()
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_slug = re.sub(r'[^\w-]', '-', improvement.get("title", "improvement")[:40])
    branch_name = f"feature/agent-{date_str}-{branch_slug}"

    # ── ブランチ作成 ─────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["git", "checkout", "-b", branch_name, "master"],
            cwd=workspace, capture_output=True, text=True, check=True,
        )
        logger.info("ブランチ作成: %s", branch_name)
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"ブランチ作成失敗: {e.stderr[:200]}",
        }

    # ── claude --print で実装 ────────────────────────────────────────────
    prompt = _build_claude_prompt(improvement)
    try:
        result = subprocess.run(
            [claude_bin, "--print", prompt],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("claude 実行エラー (rc=%d): %s", result.returncode, result.stderr[:300])
    except subprocess.TimeoutExpired:
        logger.warning("claude 実行タイムアウト (300s)")
    except Exception as e:
        logger.warning("claude 実行例外: %s", e)

    # ── git status で変更があるか確認 ────────────────────────────────────
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace, capture_output=True, text=True,
    )
    has_changes = bool(status_result.stdout.strip())

    if not has_changes:
        subprocess.run(
            ["git", "checkout", "master"],
            cwd=workspace, capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            cwd=workspace, capture_output=True,
        )
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "claude が変更を生成しませんでした（git status が空）",
        }

    # ── コミット ─────────────────────────────────────────────────────────
    title = improvement.get("title", "改善")
    commit_msg = (
        f"feat: {title}\n\n"
        f"自動改善パイプライン（claude-agent-runner）による実装\n\n"
        "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    )
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=workspace, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=workspace, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("コミット失敗: %s", e.stderr[:200] if hasattr(e, "stderr") else str(e))

    # ── Push ─────────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=workspace, capture_output=True, text=True, check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        subprocess.run(
            ["git", "checkout", "master"],
            cwd=workspace, capture_output=True,
        )
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"push 失敗: {e}",
        }

    # ── PR 作成 ──────────────────────────────────────────────────────────
    pr_body = (
        f"## 概要\n{improvement.get('description', '')}\n\n"
        f"## 実装理由\n{improvement.get('reason', '')}\n\n"
        f"**規模分類**: `{size}`\n\n"
        "🤖 Generated by claude-agent-runner (auto-improvement-pipeline)"
    )
    pr_title = f"feat: {title}"
    if size == "auto":
        pr_title = f"[auto-merge] {pr_title}"

    gh_create_args = [
        "gh", "pr", "create",
        "--title", pr_title,
        "--body", pr_body,
        "--base", "master",
        "--head", branch_name,
    ]
    if size == "approval":
        gh_create_args.append("--draft")

    pr_url: str | None = None
    pr_number: int | None = None
    try:
        pr_result = subprocess.run(
            gh_create_args,
            cwd=workspace, capture_output=True, text=True, check=True,
        )
        pr_url = pr_result.stdout.strip()
        pr_number = _extract_pr_number(pr_url)
        logger.info("PR 作成: %s", pr_url)
    except subprocess.CalledProcessError as e:
        logger.warning("PR 作成失敗: %s", e.stderr[:300] if hasattr(e, "stderr") else str(e))
        subprocess.run(["git", "checkout", "master"], cwd=workspace, capture_output=True)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"PR 作成失敗: {e.stderr[:200] if hasattr(e, 'stderr') else str(e)}",
        }

    # ── auto: 自動マージ ─────────────────────────────────────────────────
    merged = False
    if size == "auto" and pr_number:
        try:
            merge_result = subprocess.run(
                ["gh", "pr", "merge", str(pr_number), "--merge", "--auto"],
                cwd=workspace, capture_output=True, text=True, timeout=60,
            )
            if merge_result.returncode == 0:
                merged = True
                logger.info("PR #%d 自動マージ完了", pr_number)
            else:
                logger.warning("自動マージ失敗 (rc=%d): %s", merge_result.returncode, merge_result.stderr[:200])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("自動マージ例外: %s", e)

    # ── approval: ドラフト → 公開状態に ─────────────────────────────────
    elif size == "approval" and pr_number:
        try:
            subprocess.run(
                ["gh", "pr", "ready", str(pr_number)],
                cwd=workspace, capture_output=True, text=True, timeout=30,
            )
            logger.info("PR #%d を公開状態にしました", pr_number)
        except Exception as e:
            logger.warning("gh pr ready 失敗: %s", e)

    # 元のブランチに戻る
    subprocess.run(
        ["git", "checkout", "master"],
        cwd=workspace, capture_output=True,
    )

    return {
        "success": True,
        "pr_number": pr_number,
        "pr_url": pr_url,
        "merged": merged,
        "message": (
            f"PR 作成完了（{'自動マージ済み' if merged else size}）: {pr_url}"
        ),
    }


if __name__ == "__main__":
    # 簡易動作確認（dry run）
    if "--check" in sys.argv:
        claude_bin = _find_claude_bin()
        print(f"claude CLI: {claude_bin or '見つかりません'}")
        print(f"workspace: {_find_workspace_root()}")
