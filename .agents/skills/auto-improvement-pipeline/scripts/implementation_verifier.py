"""Independent checker for code produced by the auto-improvement implementer."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from loop_constraints import load_loop_constraints


EXCLUDE_DIRS = frozenset({"pydeps", "_archive", "node_modules", ".venv", "__pycache__", ".git"})


@dataclass
class VerificationResult:
    passed: bool | None
    summary: str
    verifier_role: str
    verification_id: str
    isolation: str

    def as_dict(self) -> dict:
        return asdict(self)


class IndependentImprovementVerifier:
    """Verify candidate code in a detached git worktree, never in the maker tree."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.constraints = load_loop_constraints()
        roles = self.constraints["roles"]
        if roles["must_differ"] and roles["implementer"] == roles["verifier"]:
            raise ValueError("implementer and verifier roles must differ")
        self.role = str(roles["verifier"])

    @staticmethod
    def sanity_check(original: str, candidate: str) -> tuple[bool, str]:
        original_lines = len(original.splitlines())
        candidate_lines = len(candidate.splitlines())
        if original_lines and candidate_lines < original_lines * 0.5:
            return False, f"行数激減: {original_lines} → {candidate_lines}"
        try:
            original_tree = ast.parse(original)
            candidate_tree = ast.parse(candidate)
        except SyntaxError as exc:
            return False, f"AST解析失敗: {exc}"
        original_functions = {node.name for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef)}
        candidate_functions = {node.name for node in ast.walk(candidate_tree) if isinstance(node, ast.FunctionDef)}
        lost = original_functions - candidate_functions
        if len(lost) > 2:
            return False, f"関数消失: {sorted(lost)}"
        return True, "OK"

    def verify(
        self,
        target_file: Path,
        original: str,
        candidate: str,
        *,
        allow_syntax_only: bool = False,
        timeout: int = 60,
    ) -> VerificationResult:
        verification_id = uuid.uuid4().hex[:12]
        if target_file.suffix.lower() == ".py":
            sane, reason = self.sanity_check(original, candidate)
        else:
            original_lines = len(original.splitlines())
            candidate_lines = len(candidate.splitlines())
            sane = not original_lines or candidate_lines >= original_lines * 0.5
            reason = "OK" if sane else f"行数激減: {original_lines} → {candidate_lines}"
        if not sane:
            return VerificationResult(False, reason, self.role, verification_id, "pre-write")

        try:
            relative = target_file.resolve().relative_to(self.workspace_root.resolve())
        except ValueError:
            return VerificationResult(False, "対象ファイルがworkspace外", self.role, verification_id, "rejected")

        worktree_parent = Path(tempfile.mkdtemp(prefix="auto-improvement-verify-"))
        worktree = worktree_parent / "checkout"
        added = False
        try:
            add = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
                cwd=self.workspace_root, capture_output=True, text=True, timeout=30,
            )
            if add.returncode != 0:
                return VerificationResult(None, f"検証worktree作成失敗: {add.stderr[-300:]}", self.role, verification_id, "unavailable")
            added = True
            isolated_target = worktree / relative
            isolated_target.parent.mkdir(parents=True, exist_ok=True)
            isolated_target.write_text(candidate, encoding="utf-8")
            return self._run_checks(isolated_target, worktree, allow_syntax_only, timeout, verification_id)
        finally:
            if added:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=self.workspace_root, capture_output=True, timeout=30,
                )
            shutil.rmtree(worktree_parent, ignore_errors=True)

    def _run_checks(
        self,
        target: Path,
        worktree: Path,
        allow_syntax_only: bool,
        timeout: int,
        verification_id: str,
    ) -> VerificationResult:
        python_bin = self.workspace_root / ".venv/bin/python"
        pytest_bin = self.workspace_root / ".venv/bin/pytest"
        if not python_bin.exists():
            python_bin = Path(shutil.which("python3") or "python3")

        suffix = target.suffix.lower()
        if suffix == ".py":
            syntax = subprocess.run(
                [str(python_bin), "-m", "py_compile", str(target)],
                cwd=worktree, capture_output=True, text=True, timeout=30,
            )
            if syntax.returncode != 0:
                return VerificationResult(False, f"SyntaxError: {syntax.stderr[-500:]}", self.role, verification_id, "git-worktree")
        elif suffix == ".json":
            try:
                json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                return VerificationResult(False, f"JSON syntax error: {exc}", self.role, verification_id, "git-worktree")
        elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
            tsc = self.workspace_root / "frontend/node_modules/.bin/tsc"
            frontend = worktree / "frontend"
            if not tsc.exists() or not frontend.exists():
                return VerificationResult(None, "隔離TypeScript checker未導入", self.role, verification_id, "git-worktree")
            syntax = subprocess.run(
                [str(tsc), "--noEmit"],
                cwd=frontend, capture_output=True, text=True, timeout=timeout,
            )
            if syntax.returncode != 0:
                return VerificationResult(False, (syntax.stdout + syntax.stderr)[-1500:], self.role, verification_id, "git-worktree")
            return VerificationResult(True, "TypeScript check passed", self.role, verification_id, "git-worktree")
        elif suffix == ".md" and allow_syntax_only:
            return VerificationResult(True, "Markdown low-risk verification passed", self.role, verification_id, "git-worktree")

        stem = target.stem
        tests = [
            path for path in worktree.glob(f"**/test_*{stem}.py")
            if not any(part in EXCLUDE_DIRS for part in path.parts)
        ]
        if not pytest_bin.exists() or not tests:
            if allow_syntax_only:
                return VerificationResult(True, "syntaxのみ確認（low-risk変更）", self.role, verification_id, "git-worktree")
            reason = "pytest未導入" if not pytest_bin.exists() else f"対象テストなし: {stem}"
            return VerificationResult(None, reason, self.role, verification_id, "git-worktree")

        test = subprocess.run(
            [str(pytest_bin), *[str(path) for path in tests], "-q", "--tb=short"],
            cwd=worktree, capture_output=True, text=True, timeout=timeout,
        )
        output = (test.stdout + test.stderr)[-1500:]
        return VerificationResult(test.returncode == 0, output, self.role, verification_id, "git-worktree")
