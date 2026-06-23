"""Gemini API を使って改善案を実装し、PRを作成・マージするランナー."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GEMINI_TIMEOUT_SECONDS = 90


def _get_gemini_api_key(workspace: Path | None = None) -> str | None:
    """環境変数 → .streamlit/secrets.toml の順で Gemini API キーを取得する."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    if workspace is None:
        workspace = _find_workspace_root()
    secrets_path = workspace / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            for line in secrets_path.read_text(encoding="utf-8").splitlines():
                if "GEMINI_API_KEY" in line and "=" in line:
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
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


def _call_gemini(api_key: str, prompt: str) -> str | None:
    """Gemini REST API を呼び出す（gemini-2.5-flash → gemini-2.5-pro フォールバック）."""
    try:
        import requests  # type: ignore[import-untyped]
        for model_name in ("gemini-2.5-flash", "gemini-2.5-pro"):
            url = (
                "https://generativelanguage.googleapis.com/v1beta"
                f"/models/{model_name}:generateContent"
            )
            resp = requests.post(
                f"{url}?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=_GEMINI_TIMEOUT_SECONDS,
            )
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            logger.warning("Gemini %s: HTTP %s", model_name, resp.status_code)
    except Exception as e:
        logger.warning("Gemini API エラー: %s", e)
    return None


def _find_target_file(improvement: dict[str, Any], workspace: Path) -> Path | None:
    """improvement の target_module からファイルを特定する."""
    target_module = improvement.get("target_module")
    if not target_module:
        return None
    target = Path(target_module)
    if target.is_absolute() and target.is_file():
        return target
    name = target.name if target.suffix else f"{target_module}.py"
    candidates = [
        workspace / target,
        workspace / name,
        workspace / "api" / name,
        workspace / "scripts" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _apply_diff(diff_text: str, target_file: Path, current_code: str) -> str | None:
    """unified diff を適用して新しいファイル内容を返す."""
    fd, tmp_path_str = tempfile.mkstemp(suffix=".py")
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(current_code)
        dry = subprocess.run(
            ["patch", "--dry-run", str(tmp_path)],
            input=diff_text, text=True, capture_output=True, timeout=10,
        )
        if dry.returncode != 0:
            return None
        apply_r = subprocess.run(
            ["patch", str(tmp_path)],
            input=diff_text, text=True, capture_output=True, timeout=10,
        )
        if apply_r.returncode != 0:
            return None
        return tmp_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("diff 適用中に例外: %s", e)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)
        Path(tmp_path_str + ".orig").unlink(missing_ok=True)


def _extract_new_code(raw_output: str, target_file: Path, current_code: str) -> str | None:
    """LLM 出力から新しいコードを抽出する（diff または全文コードに対応）."""
    diff_match = re.search(r"```diff\n(.*?)```", raw_output, re.DOTALL)
    if diff_match:
        result = _apply_diff(diff_match.group(1), target_file, current_code)
        if result:
            return result

    stripped = raw_output.strip()
    if stripped.startswith(("--- ", "diff ")):
        result = _apply_diff(stripped, target_file, current_code)
        if result:
            return result

    py_match = re.search(r"```python\n(.*?)```", raw_output, re.DOTALL)
    if py_match:
        return py_match.group(1)

    if stripped.startswith(("import ", "from ", "#!", '"""', "class ", "def ")):
        return stripped

    return None


def _build_gemini_prompt(improvement: dict[str, Any], target_file: Path, current_code: str) -> str:
    """Gemini に渡すコード改善プロンプトを構築する."""
    title = improvement.get("title", "改善案")
    description = improvement.get("description", "")
    reason = improvement.get("reason", "")

    _code_limit = 24000
    code_snippet = current_code[:_code_limit]
    is_truncated = len(current_code) > _code_limit

    return (
        "あなたは Python コード改善の専門家です。\n"
        "以下のPythonファイルに対して、指定された改善を unified diff 形式で実施してください。\n\n"
        f"## 対象ファイル\n{target_file.name}"
        + (f"（先頭{_code_limit}文字のみ表示）\n\n" if is_truncated else "\n\n")
        + f"## 現在のコード\n```python\n{code_snippet}\n```\n\n"
        "## 実施すべき改善\n"
        f"タイトル: {title}\n"
        f"詳細: {description}\n"
        f"理由: {reason}\n\n"
        "## 出力ルール\n"
        "変更箇所のみの unified diff を返してください。以下の形式で厳密に返すこと：\n"
        "```diff\n"
        f"--- a/{target_file.name}\n"
        f"+++ b/{target_file.name}\n"
        "@@ ... @@\n"
        "...\n"
        "```\n"
        "セキュリティの脆弱性（SQLインジェクション・XSS等）を絶対に導入しないでください。\n"
    )


def _cleanup_branch(workspace: Path, branch_name: str) -> None:
    """現在のブランチをmaster に戻し、作業ブランチを削除する."""
    subprocess.run(["git", "checkout", "master"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "branch", "-D", branch_name], cwd=workspace, capture_output=True)


def _has_file_changes(workspace: Path) -> bool:
    """git diff/status でファイル変更を確認する."""
    diff_result = subprocess.run(
        ["git", "diff", "--stat", "HEAD"],
        cwd=workspace, capture_output=True, text=True,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace, capture_output=True, text=True,
    )
    return bool(diff_result.stdout.strip()) or bool(status_result.stdout.strip())


def run_claude_agent(improvement: dict[str, Any], size: str) -> dict[str, Any]:
    """
    Gemini API で改善案を実装し、PRを作成する。

    auto のみ実行。approval / manual は即リターンする。

    Args:
        improvement: 改善案 dict（title, description, target_module 等を含む）
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
    if size != "auto":
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"size={size!r} は自動実装の対象外です（auto のみ対応）",
        }

    workspace = _find_workspace_root()
    api_key = _get_gemini_api_key(workspace)
    if not api_key:
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "GEMINI_API_KEY が設定されていません",
        }

    target_file = _find_target_file(improvement, workspace)
    if not target_file:
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"対象ファイルが特定できません: {improvement.get('target_module')}",
        }

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_slug = re.sub(r'[^\w-]', '-', improvement.get("title", "improvement")[:40])
    branch_name = f"feature/agent-{date_str}-{branch_slug}"

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

    current_code = target_file.read_text(encoding="utf-8")
    prompt = _build_gemini_prompt(improvement, target_file, current_code)
    raw_output = _call_gemini(api_key, prompt)

    if not raw_output:
        _cleanup_branch(workspace, branch_name)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "Gemini API からの応答が空でした",
        }

    new_code = _extract_new_code(raw_output, target_file, current_code)
    if not new_code:
        _cleanup_branch(workspace, branch_name)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "Gemini 出力からコードを抽出できませんでした",
        }

    target_file.write_text(new_code, encoding="utf-8")

    if not _has_file_changes(workspace):
        _cleanup_branch(workspace, branch_name)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": "Gemini が変更を生成しませんでした（git diff が空）",
        }

    title = improvement.get("title", "改善")
    commit_msg = (
        f"feat: {title}\n\n"
        f"自動改善パイプライン（gemini-agent-runner）による実装\n\n"
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
        logger.warning(
            "コミット失敗: %s", e.stderr[:200] if hasattr(e, "stderr") else str(e)
        )

    try:
        subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=workspace, capture_output=True, text=True, check=True, timeout=60,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _cleanup_branch(workspace, branch_name)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"push 失敗: {e}",
        }

    pr_body = (
        f"## 概要\n{improvement.get('description', '')}\n\n"
        f"## 実装理由\n{improvement.get('reason', '')}\n\n"
        f"**規模分類**: `{size}`\n\n"
        "🤖 Generated by gemini-agent-runner (auto-improvement-pipeline)"
    )
    pr_title = f"[auto-merge] feat: {title}"

    try:
        pr_result = subprocess.run(
            [
                "gh", "pr", "create",
                "--title", pr_title,
                "--body", pr_body,
                "--base", "master",
                "--head", branch_name,
            ],
            cwd=workspace, capture_output=True, text=True, check=True,
        )
        pr_url = pr_result.stdout.strip()
        pr_number = _extract_pr_number(pr_url)
        logger.info("PR 作成: %s", pr_url)
    except subprocess.CalledProcessError as e:
        _cleanup_branch(workspace, branch_name)
        return {
            "success": False,
            "pr_number": None,
            "pr_url": None,
            "merged": False,
            "message": f"PR 作成失敗: {e.stderr[:200] if hasattr(e, 'stderr') else str(e)}",
        }

    merged = False
    if pr_number:
        try:
            merge_result = subprocess.run(
                ["gh", "pr", "merge", str(pr_number), "--merge", "--auto"],
                cwd=workspace, capture_output=True, text=True, timeout=60,
            )
            if merge_result.returncode == 0:
                merged = True
                logger.info("PR #%d 自動マージ完了", pr_number)
            else:
                logger.warning(
                    "自動マージ失敗 (rc=%d): %s",
                    merge_result.returncode, merge_result.stderr[:200],
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning("自動マージ例外: %s", e)

    subprocess.run(["git", "checkout", "master"], cwd=workspace, capture_output=True)

    return {
        "success": True,
        "pr_number": pr_number,
        "pr_url": pr_url,
        "merged": merged,
        "message": f"PR 作成完了（{'自動マージ済み' if merged else '未マージ'}）: {pr_url}",
    }


if __name__ == "__main__":
    if "--check" in sys.argv:
        workspace = _find_workspace_root()
        api_key = _get_gemini_api_key(workspace)
        print(f"Gemini API key: {'設定済み' if api_key else '未設定'}")
        print(f"workspace: {workspace}")
