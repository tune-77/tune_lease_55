"""Step 3: 承認された改善案の自動適用・Git コミット・PR 作成・Cowork 報告（本実装）."""

from __future__ import annotations

import fnmatch
import json
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


# ────────────────────────────────────────────────────────────────────────────
# NEEDS_REVIEW 判定用キーワード
# ────────────────────────────────────────────────────────────────────────────

_SECURITY_KEYWORDS = [
    "sql", "xss", "injection", "auth", "セキュリティ", "認証", "権限", "csrf",
    "command injection", "password", "パスワード", "token", "トークン",
]
_DB_KEYWORDS = [
    "スキーマ", "migration", "マイグレーション", "alter table", "create table",
    "drop table", "db schema", "sqlite", "データベース定義",
]
_API_KEYWORDS = [
    "api変更", "引数変更", "レスポンス構造", "インターフェース変更", "i/f変更",
    "エンドポイント変更", "rest api",
]
_SCORING_KEYWORDS = [
    "スコアリング", "閾値", "threshold", "lgbm", "lightgbm", "quantum_risk",
    "auc", "モデル係数", "model weight", "score_calculation",
]

# 変更に慎重を要するスコアリング重要ファイル
_SCORING_FILES: frozenset[str] = frozenset({
    "quantum_analysis_module.py", "scoring_core.py", "total_scorer.py",
    "asset_scorer.py", "category_config.py", "industry_hybrid_model.py",
    "rule_manager.py", "coeff_definitions.py",
})

# テスト探索から除外するディレクトリ名
EXCLUDE_DIRS: frozenset[str] = frozenset({
    "pydeps", "_archive", "node_modules", ".venv", "__pycache__", ".git",
})

# 自動書き換え禁止ファイル（denylist）
WRITE_DENYLIST: list[str] = [
    ".streamlit/secrets.toml",
    "data/",
    "*.plist",
    "scoring_core.py",
    "retraining_pipeline.py",
    "migrate",
    "alembic",
]

# ── パイプライン台帳（optional）────────────────────────────────────────────
_LEDGER_AVAILABLE: bool = False
_ledger_module_path = str(Path(__file__).parent.parent)
if _ledger_module_path not in sys.path:
    sys.path.insert(0, _ledger_module_path)
try:
    import pipeline_ledger as _ledger  # type: ignore[import-untyped]
    _LEDGER_AVAILABLE = True
except ImportError:
    _ledger = None  # type: ignore[assignment]


def _result(action: str, reason: str, **extra: Any) -> dict[str, Any]:
    """apply_improvement の戻り値ヘルパー."""
    return {"action": action, "reason": reason, **extra}


# ────────────────────────────────────────────────────────────────────────────
# Step3AutoApplier
# ────────────────────────────────────────────────────────────────────────────

class Step3AutoApplier:
    """承認済み改善案の自動適用・デプロイエンジン（本実装）."""

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root: Path = (
            Path(workspace_root) if workspace_root else self._detect_workspace_root()
        )
        self.date_str: str = datetime.now().strftime("%Y%m%d")
        self.auto_branch: str = f"auto-improve/{self.date_str}"
        self.patches_dir: Path = Path("/tmp/patches")
        self.patches_dir.mkdir(parents=True, exist_ok=True)

        # 適用結果リスト
        self._applied: list[dict[str, Any]] = []
        self._needs_review: list[dict[str, Any]] = []
        self._rejected: list[dict[str, Any]] = []
        self._pr_url: str | None = None

        # テスト済みコードを保持（ブランチ切り替え後に再適用するため）
        self._pending_patches: list[tuple[Path, str]] = []

    # ── 外部インターフェース ──────────────────────────────────────────────

    def apply_improvement(
        self,
        improvement: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        一件の改善案を処理する。

        実際のファイル書き換えはここでは行わず、テスト通過後に
        _pending_patches へ追加する。ファイルへの書き込みは
        git_commit_and_push() 内でブランチ切り替え後に実施。

        Returns:
            {"action": "applied"|"needs_review"|"skipped", "reason": str, ...}
        """
        imp_id = improvement.get("id", "?")
        title = improvement.get("title", "")

        # [#4] 台帳重複チェック（処理済みならスキップ）
        _ledger_key: str | None = None
        if _LEDGER_AVAILABLE:
            _ledger_key = _ledger.compute_key(title, improvement.get("description", ""))
            already, done_status = _ledger.is_processed(_ledger_key)
            if already:
                logger.info("skip: already processed (%s)", done_status)
                return _result("skipped", f"台帳: 処理済み ({done_status})")

        # Step 2 で REJECTED → 即スキップ
        if validation_result.get("status") != "APPROVED":
            reason = "; ".join(validation_result.get("critical_flaws", ["REJECTED by Step2"]))
            self._rejected.append({"id": imp_id, "title": title, "reason": reason})
            return _result("skipped", reason)

        # NEEDS_REVIEW 判定
        needs_review, review_reason = self._check_needs_review(improvement, validation_result)
        if needs_review:
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": review_reason,
                "detail": improvement.get("description", "")[:300],
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason=review_reason)
            return _result("needs_review", review_reason)

        # 対象ファイル特定
        target_file = self._find_target_file(improvement.get("target_module"))
        if not target_file:
            patch_file = self._save_patch_markdown(improvement, validation_result)
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": "対象ファイルが特定できないため手動実装が必要",
                "detail": str(patch_file),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason="対象ファイル不明")
            return _result("needs_review", "対象ファイル不明", patch_file=str(patch_file))

        # [#5] 書き換え禁止リストチェック
        if self._is_denylisted(target_file):
            denylist_reason = f"denylist: 自動適用禁止ファイル ({target_file.name})"
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": denylist_reason,
                "detail": str(target_file.relative_to(self.workspace_root)),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason=denylist_reason)
            return _result("needs_review", denylist_reason)

        # コード生成（Codex → Claude → Gemini フォールバック）
        current_code = target_file.read_text(encoding="utf-8")
        prompt = self._build_diff_prompt(target_file, improvement, current_code)
        raw_output = self._generate_code_with_fallback(prompt, current_code, improvement)
        new_code = self._extract_new_code(raw_output, target_file, current_code) if raw_output else None
        if not new_code:
            patch_file = self._save_patch_markdown(improvement, validation_result, target_file)
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": "コード生成失敗（全バックエンド）— パッチファイルを手動確認してください",
                "detail": str(patch_file),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason="コード生成失敗")
            return _result("needs_review", "コード生成失敗", patch_file=str(patch_file))

        # ローカルテスト（現ブランチ上で試験的に書き換え → テスト → 元に戻す）
        original_code = current_code

        # [#2-C] 健全性チェック（書き込み前に行数・関数消失を検証）
        sanity_ok, sanity_reason = self._sanity_check(original_code, new_code, target_file)
        if not sanity_ok:
            patch_file = self._save_patch_markdown(improvement, validation_result, target_file)
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": f"健全性チェック失敗: {sanity_reason}",
                "detail": str(patch_file),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason=f"健全性チェック失敗: {sanity_reason}")
            return _result("needs_review", f"健全性チェック失敗: {sanity_reason}", patch_file=str(patch_file))


        target_file.write_text(new_code, encoding="utf-8")
        test_ok, test_output = self._run_local_tests(target_file)
        target_file.write_text(original_code, encoding="utf-8")  # 必ずロールバック

        if test_ok is None:
            # テスト環境未整備（pytest なし / テストファイル0件）→ 自動適用しない
            patch_file = self._save_patch_markdown(improvement, validation_result, target_file)
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": f"テスト環境未整備: {test_output[:200]}",
                "detail": str(patch_file),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason="テスト環境未整備")
            return _result("needs_review", "テスト環境未整備", patch_file=str(patch_file))

        if not test_ok:
            patch_file = self._save_patch_markdown(improvement, validation_result, target_file)
            self._needs_review.append({
                "id": imp_id,
                "title": title,
                "reason": f"テスト失敗によりロールバック: {test_output[:200]}",
                "detail": str(patch_file),
            })
            if _LEDGER_AVAILABLE and _ledger_key:
                _ledger.record(_ledger_key, "needs_review", title, reason="テスト失敗")
            return _result("needs_review", "テスト失敗", patch_file=str(patch_file))

        # テスト通過 → 保留リストへ追加（ブランチ切り替え後に実適用）
        self._pending_patches.append((target_file, new_code))
        self._applied.append({
            "id": imp_id,
            "file": str(target_file.relative_to(self.workspace_root)),
            "title": title,
            "pr_url": None,  # git_commit_and_push 後に更新
        })
        if _LEDGER_AVAILABLE and _ledger_key:
            _ledger.record(_ledger_key, "applied", title)
        return _result("applied", "テスト通過・適用予定")

    def git_commit_and_push(self) -> dict[str, Any]:
        """
        保留中のパッチを auto-improve/YYYYMMDD ブランチへ適用・コミット・Push・PR 作成。
        """
        if not self._pending_patches:
            return {"success": False, "commit_hash": None, "pr_url": None,
                    "message": "コミット対象なし（pending_patches が空）"}

        original_branch: str | None = None

        try:
            # 現在のブランチを保存
            original_branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.workspace_root, capture_output=True, text=True, check=True,
            ).stdout.strip()

            # master を最新化してから auto-improve ブランチを用意
            _fetch_ok = True
            try:
                subprocess.run(
                    ["git", "fetch", "origin", "master"],
                    cwd=self.workspace_root, capture_output=True, check=True, timeout=30,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as fetch_err:
                _fetch_ok = False
                logger.warning("ネットワーク不達、ローカルmasterから続行: %s", fetch_err)

            branch_list = subprocess.run(
                ["git", "branch", "--list", self.auto_branch],
                cwd=self.workspace_root, capture_output=True, text=True,
            ).stdout

            if self.auto_branch in branch_list:
                subprocess.run(
                    ["git", "checkout", self.auto_branch],
                    cwd=self.workspace_root, capture_output=True, check=True,
                )
            elif _fetch_ok:
                subprocess.run(
                    ["git", "checkout", "-b", self.auto_branch, "origin/master"],
                    cwd=self.workspace_root, capture_output=True, check=True,
                )
            else:
                # fetch 失敗時はローカル master を基点にブランチ作成
                subprocess.run(
                    ["git", "checkout", "-b", self.auto_branch, "master"],
                    cwd=self.workspace_root, capture_output=True, check=True,
                )

            # ブランチ切り替え後に保留パッチを実際に書き込む
            for file_path, new_content in self._pending_patches:
                file_path.write_text(new_content, encoding="utf-8")

            # ステージング
            for entry in self._applied:
                subprocess.run(
                    ["git", "add", entry["file"]],
                    cwd=self.workspace_root, capture_output=True, check=True,
                )

            # コミット
            titles_str = "\n".join(
                f"  - {e['id']}: {e['title']}" for e in self._applied
            )
            commit_msg = (
                f"auto-improve: {len(self._applied)}件の改善を自動適用 ({self.date_str})\n\n"
                f"{titles_str}\n\n"
                "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
            )
            commit_result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.workspace_root, capture_output=True, text=True, check=True,
            )
            commit_hash = commit_result.stdout.strip().split()[-1] if commit_result.stdout else "unknown"

            # Push
            try:
                subprocess.run(
                    ["git", "push", "-u", "origin", self.auto_branch],
                    cwd=self.workspace_root, capture_output=True, check=True, timeout=60,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as push_err:
                logger.warning("push保留: リモートへの push 失敗: %s", push_err)
                if _LEDGER_AVAILABLE:
                    for entry in self._applied:
                        _ledger_key_push = _ledger.compute_key(
                            entry["title"], entry.get("file", "")
                        )
                        _ledger.record(_ledger_key_push, "push_pending", entry["title"],
                                       reason="push失敗: ネットワーク不達")
                # PR 作成はスキップして commit_hash のみ返す
                if original_branch:
                    subprocess.run(
                        ["git", "checkout", original_branch],
                        cwd=self.workspace_root, capture_output=True, check=False,
                    )
                return {
                    "success": False,
                    "commit_hash": commit_hash,
                    "pr_url": None,
                    "message": f"push保留: {push_err}",
                }

            # PR 作成
            pr_url = self._create_pull_request()
            self._pr_url = pr_url

            # 元のブランチに戻る
            if original_branch:
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=self.workspace_root, capture_output=True, check=False,
                )

            return {
                "success": True,
                "commit_hash": commit_hash,
                "pr_url": pr_url,
                "message": f"コミット・PR 作成完了（{len(self._applied)}件適用）",
            }

        except subprocess.CalledProcessError as e:
            # 失敗しても元のブランチへ戻れるよう試みる
            if original_branch:
                subprocess.run(
                    ["git", "checkout", original_branch],
                    cwd=self.workspace_root, capture_output=True, check=False,
                )
            return {
                "success": False,
                "commit_hash": None,
                "pr_url": None,
                "message": f"Git エラー: {e.stderr or str(e)}",
            }

    def generate_report(self) -> Path:
        """日次レポートを ~/Library/Logs/tunelease/reports/ に出力する."""
        for entry in self._applied:
            entry["pr_url"] = self._pr_url

        report: dict[str, Any] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "applied": self._applied,
            "needs_review": self._needs_review,
            "rejected": self._rejected,
            "pr_url": self._pr_url,
            "summary": {
                "applied_count": len(self._applied),
                "needs_review_count": len(self._needs_review),
                "rejected_count": len(self._rejected),
            },
        }

        reports_dir = Path.home() / "Library" / "Logs" / "tunelease" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"improvement_report_{self.date_str}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # latest.json シンボリックリンクを常に最新レポートへ向ける
        latest_link = reports_dir / "latest.json"
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(report_path)

        return report_path

    # ── NEEDS_REVIEW 判定 ─────────────────────────────────────────────────

    def _check_needs_review(
        self,
        improvement: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> tuple[bool, str]:
        """NEEDS_REVIEW かを判定し (bool, reason_str) を返す."""
        reasons: list[str] = []
        text = (
            improvement.get("description", "")
            + " "
            + improvement.get("title", "")
        ).lower()

        if any(kw in text for kw in _SECURITY_KEYWORDS):
            reasons.append("セキュリティ関連の変更")

        if any(kw in text for kw in _DB_KEYWORDS):
            reasons.append("DBスキーマ変更・マイグレーション")

        if any(kw in text for kw in _API_KEYWORDS):
            reasons.append("API I/F変更")

        py_file_count = len(re.findall(r'\b\w+\.py\b', text))
        if py_file_count >= 3:
            reasons.append(f"複数ファイルにまたがる変更（{py_file_count}ファイル参照）")

        if any(kw in text for kw in _SCORING_KEYWORDS):
            reasons.append("スコアリングロジック・モデル閾値変更")

        target_module = improvement.get("target_module", "") or ""
        if Path(target_module).name in _SCORING_FILES:
            reasons.append(f"スコアリング重要ファイルへの変更（{target_module}）")

        confidence = self._compute_confidence(validation_result)
        if confidence < 0.7:
            reasons.append(f"信頼度スコア不足（confidence={confidence:.2f}）")

        return bool(reasons), "; ".join(reasons)

    def _compute_confidence(self, validation_result: dict[str, Any]) -> float:
        """validation_result から confidence_score を算出する."""
        if "confidence_score" in validation_result:
            return float(validation_result["confidence_score"])

        meta = validation_result.get("metadata", {})
        issues = int(meta.get("issues_count", 0))
        flaws = int(meta.get("flaws_count", 0))
        breaking = 1 if meta.get("test_breaking", False) else 0
        score = 1.0 - (issues * 0.1) - (flaws * 0.2) - (breaking * 0.3)
        return max(0.0, min(1.0, score))

    # ── API キー取得（共通）─────────────────────────────────────────────────

    def _get_api_key(self, key_name: str) -> str:
        """環境変数 → .streamlit/secrets.toml の順で API キーを取得する."""
        key = os.environ.get(key_name, "")
        if key:
            return key
        secrets_path = self.workspace_root / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            try:
                for line in secrets_path.read_text(encoding="utf-8").splitlines():
                    if key_name in line and "=" in line:
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
            except OSError:
                pass
        return ""

    def _get_gemini_api_key(self) -> str:
        return self._get_api_key("GEMINI_API_KEY")

    # ── プロンプト構築 ────────────────────────────────────────────────────

    def _build_diff_prompt(
        self,
        target_file: Path,
        improvement: dict[str, Any],
        current_code: str,
    ) -> str:
        """unified diff 形式でのコード変更を要求するプロンプトを構築する."""
        code_snippet = current_code[:8000]
        is_truncated = len(current_code) > 8000
        return (
            "あなたは Python コード改善の専門家です。\n"
            "以下のPythonファイルに対して、指定された改善を実施してください。\n\n"
            f"## 対象ファイル\n{target_file.name}"
            + ("（先頭8000文字のみ表示）\n\n" if is_truncated else "\n\n")
            + f"## 現在のコード\n```python\n{code_snippet}\n```\n\n"
            "## 実施すべき改善\n"
            f"タイトル: {improvement.get('title', '')}\n"
            f"詳細: {improvement.get('description', '')}\n"
            f"理由: {improvement.get('reason', '')}\n\n"
            "## 出力ルール\n"
            "変更箇所のみの unified diff を返してください。ファイル全体の出力は不要。"
            "以下の形式で厳密に返すこと：\n"
            "```diff\n"
            f"--- a/{target_file.name}\n"
            f"+++ b/{target_file.name}\n"
            "@@ ... @@\n"
            "...\n"
            "```\n"
            "セキュリティの脆弱性（SQLインジェクション・XSS等）を絶対に導入しないでください。\n"
        )

    # ── Gemini（旧来実装を維持）──────────────────────────────────────────

    def _call_gemini_api(self, prompt: str) -> str | None:
        """Gemini REST API を呼び出す。"""
        api_key = self._get_gemini_api_key()
        if not api_key:
            logger.warning("GEMINI_API_KEY 未設定のため Gemini をスキップ")
            return None
        try:
            import requests  # type: ignore[import-untyped]
            url = (
                "https://generativelanguage.googleapis.com/v1beta"
                "/models/gemini-2.0-flash:generateContent"
            )
            resp = requests.post(
                f"{url}?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.warning("Gemini API エラー: %s", e)
            return None

    def _generate_code_with_gemini(
        self,
        target_file: Path,
        improvement: dict[str, Any],
    ) -> str | None:
        """後方互換: Gemini で完全なコードを生成する（旧来のプロンプト形式）."""
        current_code = target_file.read_text(encoding="utf-8")
        code_snippet = current_code[:8000]
        is_truncated = len(current_code) > 8000
        prompt = (
            "あなたは Python コード改善の専門家です。\n"
            "以下のPythonファイルに対して、指定された改善を実施してください。\n\n"
            f"## 対象ファイル\n{target_file.name}"
            + ("（先頭8000文字のみ表示）\n\n" if is_truncated else "\n\n")
            + f"## 現在のコード\n```python\n{code_snippet}\n```\n\n"
            "## 実施すべき改善\n"
            f"タイトル: {improvement.get('title', '')}\n"
            f"詳細: {improvement.get('description', '')}\n"
            f"理由: {improvement.get('reason', '')}\n\n"
            "## 出力ルール\n"
            "1. 変更後の完全な Python コードのみを出力してください\n"
            "2. コードブロック（```python ... ```）で必ず囲んでください\n"
            "3. 説明文・変更点の解説は一切不要です\n"
            "4. 元のコードの構造・コメント・インポートを維持してください\n"
            "5. セキュリティの脆弱性（SQLインジェクション・XSS等）を絶対に導入しないでください\n"
            "6. ファイルが長くて先頭しか見えていない場合は、先頭部分の改善のみ実施し残りは省略せず維持してください\n"
        )
        raw = self._call_gemini_api(prompt)
        if raw is None:
            return None
        return self._extract_new_code(raw, target_file, current_code)

    # ── コード生成バックエンド（Codex / Claude / Gemini フォールバック）───

    def _try_codex(self, prompt: str, file_content: str) -> str | None:
        """OpenAI Codex (gpt-4.1 → o4-mini) でコードを生成する."""
        api_key = self._get_api_key("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY 未設定のため Codex をスキップ")
            return None
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("openai パッケージ未インストール。インストールを試みます")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "openai",
                     "--break-system-packages"],
                    check=True, capture_output=True, timeout=60,
                )
                import openai  # type: ignore[import-untyped]  # noqa: F811
            except Exception as install_err:
                logger.warning("openai インストール失敗: %s", install_err)
                return None

        client = openai.OpenAI(api_key=api_key)
        for model in ("gpt-4.1", "o4-mini"):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=90,
                )
                text = resp.choices[0].message.content or ""
                if text.strip():
                    logger.info("Codex (%s) でコード生成成功", model)
                    return text
            except openai.RateLimitError:
                logger.warning("Codex %s: 429 Rate Limit — フォールバック", model)
                return None
            except openai.APITimeoutError:
                logger.warning("Codex %s: タイムアウト — フォールバック", model)
                return None
            except openai.NotFoundError:
                logger.warning("Codex: モデル %s が見つかりません。次を試します", model)
                continue
            except Exception as e:
                logger.warning("Codex %s エラー: %s", model, e)
                return None
        return None

    def _try_claude(self, prompt: str, file_content: str) -> str | None:
        """Anthropic Claude (claude-sonnet-4-6) でコードを生成する."""
        api_key = self._get_api_key("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY 未設定のため Claude をスキップ")
            return None
        try:
            import anthropic  # type: ignore[import-untyped]
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text if msg.content else ""
            if text.strip():
                logger.info("Claude (claude-sonnet-4-6) でコード生成成功")
                return text
            return None
        except Exception as e:
            logger.warning("Claude API エラー: %s", e)
            return None

    def _try_gemini(self, prompt: str, file_content: str) -> str | None:
        """Gemini でコードを生成する（_call_gemini_api のラッパー）."""
        raw = self._call_gemini_api(prompt)
        if raw is None:
            return None
        logger.info("Gemini でコード生成成功")
        return raw

    def _generate_code_with_fallback(
        self,
        prompt: str,
        file_content: str,
        item: dict[str, Any],
    ) -> str | None:
        """Codex → Claude → Gemini の順でコードを生成する."""
        result = self._try_codex(prompt, file_content)
        if result:
            return result
        logger.warning("Codex failed, falling back to Claude")
        result = self._try_claude(prompt, file_content)
        if result:
            return result
        logger.warning("Claude failed, falling back to Gemini")
        result = self._try_gemini(prompt, file_content)
        return result

    # ── diff 適用・コード抽出 ────────────────────────────────────────────

    def _apply_diff(
        self,
        diff_text: str,
        target_file: Path,
        current_code: str,
    ) -> str | None:
        """unified diff を一時ファイルに適用して新しいファイル内容を返す."""
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
                logger.warning(
                    "patch --dry-run 失敗 (%s): %s", target_file.name, dry.stderr[:200]
                )
                return None

            apply_r = subprocess.run(
                ["patch", str(tmp_path)],
                input=diff_text, text=True, capture_output=True, timeout=10,
            )
            if apply_r.returncode != 0:
                logger.warning(
                    "patch 適用失敗 (%s): %s", target_file.name, apply_r.stderr[:200]
                )
                return None

            return tmp_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("diff 適用中に例外: %s", e)
            return None
        finally:
            tmp_path.unlink(missing_ok=True)
            Path(tmp_path_str + ".orig").unlink(missing_ok=True)

    def _extract_new_code(
        self,
        raw_output: str,
        target_file: Path,
        current_code: str,
    ) -> str | None:
        """LLM 出力から新しいコードを抽出する（diff または全文コードに対応）."""
        # ```diff ... ``` ブロック
        diff_match = re.search(r"```diff\n(.*?)```", raw_output, re.DOTALL)
        if diff_match:
            result = self._apply_diff(diff_match.group(1), target_file, current_code)
            if result:
                return result

        # 裸の unified diff
        stripped = raw_output.strip()
        if stripped.startswith(("--- ", "diff ")):
            result = self._apply_diff(stripped, target_file, current_code)
            if result:
                return result

        # ```python ... ``` ブロック
        py_match = re.search(r"```python\n(.*?)```", raw_output, re.DOTALL)
        if py_match:
            return py_match.group(1)

        # コードブロックなし・Python らしければ採用
        if stripped.startswith(("import ", "from ", "#!", '"""', "class ", "def ")):
            return stripped

        logger.warning("LLM 出力からコードを抽出できませんでした（%s）", target_file.name)
        return None

    # ── Git / PR ──────────────────────────────────────────────────────────

    def _create_pull_request(self) -> str | None:
        """gh コマンドで PR を作成する."""
        applied_lines = "\n".join(
            f"- {e['id']}: {e['title']} (`{e['file']}`)" for e in self._applied
        )
        review_lines = "\n".join(
            f"- {e['id']}: {e['title']} → _{e['reason']}_" for e in self._needs_review
        )

        body = (
            f"## 自動改善パイプライン実行結果（{self.date_str}）\n\n"
            f"### ✅ 自動適用された改善（{len(self._applied)}件）\n"
            + (applied_lines or "なし")
            + f"\n\n### 👀 要確認の改善（{len(self._needs_review)}件）\n"
            + (review_lines or "なし")
            + f"\n\n詳細レポート: `~/Library/Logs/tunelease/reports/improvement_report_{self.date_str}.json`\n\n"
            "🤖 Generated by auto-improvement-pipeline"
        )

        try:
            result = subprocess.run(
                [
                    "gh", "pr", "create",
                    "--title", f"auto-improve: {len(self._applied)}件の改善を自動適用 ({self.date_str})",
                    "--body", body,
                    "--base", "master",
                    "--head", self.auto_branch,
                ],
                cwd=self.workspace_root, capture_output=True, text=True, check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  PR 作成失敗: {e.stderr}", file=sys.stderr)
            return None

    # ── パッチファイル保存（手動確認用）────────────────────────────────────

    def _save_patch_markdown(
        self,
        improvement: dict[str, Any],
        validation_result: dict[str, Any],
        target_file: Path | None = None,
    ) -> Path:
        """改善案を /tmp/patches/ に Markdown 形式で保存する（手動確認用）."""
        patch_name = f"{self.date_str}_{improvement.get('id', 'AUTO')}.md"
        patch_path = self.patches_dir / patch_name

        lines = [
            f"# 改善案パッチ: {improvement.get('title', 'N/A')}",
            "",
            f"**ID**: {improvement.get('id', 'N/A')}",
            f"**対象ファイル**: {improvement.get('target_module') or '未特定'}",
            f"**優先度**: {improvement.get('priority', 'N/A')}",
            f"**作成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 改善内容",
            improvement.get("description", ""),
            "",
            "## 改善理由",
            improvement.get("reason", ""),
            "",
            "## Step2 検証結果",
            validation_result.get("verification_report", ""),
        ]
        if target_file:
            lines += ["", "## 対象ファイル絶対パス", str(target_file)]

        patch_path.write_text("\n".join(lines), encoding="utf-8")
        return patch_path

    # ── テスト実行 ────────────────────────────────────────────────────────

    def _run_local_tests(
        self, target_file: Path, timeout: int = 60
    ) -> tuple[bool | None, str]:
        """
        pytest / py_compile でコード修正の妥当性を確認する。
        戻り値: True=通過, False=失敗, None=テスト環境未整備(needs_review へ降格)
        """
        python_bin = self.workspace_root / ".venv" / "bin" / "python"
        pytest_bin = self.workspace_root / ".venv" / "bin" / "pytest"

        # 足切り: SyntaxError があれば即 False
        try:
            syn = subprocess.run(
                [str(python_bin), "-m", "py_compile", str(target_file)],
                cwd=self.workspace_root, capture_output=True, text=True, timeout=30,
            )
            if syn.returncode != 0:
                return False, f"SyntaxError: {syn.stderr}"
        except subprocess.TimeoutExpired:
            return False, "py_compile timeout"
        except Exception as e:
            return False, str(e)

        # pytest バイナリ確認（なければ needs_review）
        if not pytest_bin.exists():
            return None, "pytest が .venv に見つかりません — テスト無しは適用しない"

        # テストファイル探索（EXCLUDE_DIRS をフィルタリング）
        stem = target_file.stem
        raw_files = list(self.workspace_root.glob(f"test_*{stem}.py"))
        raw_files += list(self.workspace_root.glob(f"**/test_*{stem}.py"))
        test_files = [
            p for p in raw_files
            if not any(ex in p.parts for ex in EXCLUDE_DIRS)
        ]

        if not test_files:
            return None, f"テストファイル 0 件 ({stem}) — テスト無しは適用しない"

        try:
            result = subprocess.run(
                [str(pytest_bin)] + [str(f) for f in test_files] + ["-v", "--tb=short"],
                cwd=self.workspace_root, capture_output=True, text=True, timeout=timeout,
            )
            output = (result.stdout + result.stderr)[-1500:]
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Test timeout (>{timeout}s)"
        except Exception as e:
            return False, str(e)

    # ── 健全性チェック・denylist ─────────────────────────────────────────

    @staticmethod
    def _sanity_check(
        original_content: str, new_content: str, file_path: Path
    ) -> tuple[bool, str]:
        """生成コードの健全性チェック（行数激減・関数消失を検出）."""
        import ast as _ast

        orig_lines = len(original_content.splitlines())
        new_lines = len(new_content.splitlines())
        if orig_lines > 0 and new_lines < orig_lines * 0.5:
            return False, f"行数激減: {orig_lines} → {new_lines}"

        try:
            orig_tree = _ast.parse(original_content)
            new_tree = _ast.parse(new_content)
        except SyntaxError as e:
            return False, f"新コードのAST解析失敗: {e}"

        orig_funcs = {n.name for n in _ast.walk(orig_tree) if isinstance(n, _ast.FunctionDef)}
        new_funcs = {n.name for n in _ast.walk(new_tree) if isinstance(n, _ast.FunctionDef)}
        lost = orig_funcs - new_funcs
        if len(lost) > 2:
            return False, f"関数消失: {lost}"

        return True, "OK"

    def _is_denylisted(self, target_file: Path) -> bool:
        """対象ファイルが WRITE_DENYLIST に該当するかチェック。"""
        try:
            rel = str(target_file.relative_to(self.workspace_root))
        except ValueError:
            rel = str(target_file)

        for pattern in WRITE_DENYLIST:
            if pattern.endswith("/"):
                if rel.startswith(pattern) or f"/{pattern[:-1]}/" in f"/{rel}":
                    return True
            elif "*" in pattern:
                if fnmatch.fnmatch(target_file.name, pattern):
                    return True
            else:
                if pattern in rel or target_file.name == pattern:
                    return True
        return False

    # ── ファイル検索 ──────────────────────────────────────────────────────

    def _find_target_file(self, target_module: str | None) -> Path | None:
        """ターゲットモジュール名から実ファイルを検索する."""
        if not target_module:
            return None

        name = target_module if target_module.endswith(".py") else f"{target_module}.py"

        candidates = [
            self.workspace_root / name,
            self.workspace_root / "components" / name,
            self.workspace_root / "scoring" / name,
            self.workspace_root / "api" / name,
            self.workspace_root / "scripts" / name,
        ]
        for path in candidates:
            if path.is_file():
                return path

        return None

    # ── Obsidian 同期（後方互換）─────────────────────────────────────────

    def sync_obsidian_knowledge(
        self,
        improvement: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        """改善ログを Obsidian 改善ログフォルダへ保存する（後方互換シム）."""
        vault_candidates = [
            Path("/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"),
            Path("/Users/kobayashiisaoryou/Documents/Obsidian Vault"),
        ]
        vault = next((v for v in vault_candidates if v.exists()), None)
        if not vault:
            return {"success": False, "note_path": None, "error": "Vault not found"}

        try:
            note_dir = vault / "改善ログ"
            note_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{self.date_str}_{improvement.get('id', 'AUTO')}.md"
            note_path = note_dir / filename
            content = "\n".join([
                f"# {improvement.get('title', 'N/A')}",
                "",
                f"**ID**: {improvement.get('id', 'N/A')}",
                f"**対象**: {improvement.get('target_module', 'N/A')}",
                f"**日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## 改善内容",
                improvement.get("description", ""),
                "",
                "## 検証結果",
                validation_result.get("verification_report", ""),
            ])
            note_path.write_text(content, encoding="utf-8")
            return {"success": True, "note_path": str(note_path)}
        except Exception as e:
            return {"success": False, "note_path": None, "error": str(e)}

    # ── ユーティリティ ────────────────────────────────────────────────────

    def _detect_workspace_root(self) -> Path:
        """スクリプト位置からプロジェクトルート（CLAUDE.md のある場所）を検出する."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / "CLAUDE.md").exists() or (current / "tune_lease_55.py").exists():
                return current
            current = current.parent
        return Path.cwd()


# ────────────────────────────────────────────────────────────────────────────
# pipeline_runner.py から呼ばれるエントリポイント
# ────────────────────────────────────────────────────────────────────────────

def apply_improvements_pipeline(
    improvements: list[dict[str, Any]],
    validation_results: list[dict[str, Any]],
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    複数の改善案を一括処理する（pipeline_runner.py の呼び出しインターフェース）.

    Returns:
        pipeline_runner.py が期待するキー群を含む dict
    """
    applier = Step3AutoApplier(workspace_root)

    # 各改善案を処理
    for improvement, validation in zip(improvements, validation_results):
        result = applier.apply_improvement(improvement, validation)
        action_label = {
            "applied":      "✅ APPLIED",
            "needs_review": "👀 NEEDS_REVIEW",
            "skipped":      "⏭  SKIPPED",
        }.get(result["action"], result["action"])
        print(f"  [{improvement.get('id')}] {action_label}: {result['reason'][:70]}")

    # Git コミット・Push・PR 作成
    commit_result = applier.git_commit_and_push()
    if commit_result["success"]:
        print(f"✅ Git コミット・PR 作成: {commit_result.get('pr_url', 'PR URL 不明')}")
    elif applier._applied:
        print(f"⚠️  Git 操作失敗: {commit_result.get('message')}", file=sys.stderr)

    # 適用済みに対して Obsidian 同期
    applied_ids = {e["id"] for e in applier._applied}
    for imp, val in zip(improvements, validation_results):
        if imp.get("id") in applied_ids:
            applier.sync_obsidian_knowledge(imp, val)

    # Cowork 報告レポート生成
    report_path = applier.generate_report()
    print(f"📄 Cowork 報告レポート: {report_path}")

    return {
        "applied_count": len(applier._applied),
        "failed_count": 0,  # ロールバック済みは needs_review に振り分け済み
        "needs_review_count": len(applier._needs_review),
        "commit_result": commit_result,
        "applied_improvements": applier._applied,
        "needs_review": applier._needs_review,
        "rejected": applier._rejected,
        "test_results": {},
        "report_path": str(report_path),
    }


# ────────────────────────────────────────────────────────────────────────────
# 単体テスト用エントリポイント
# ────────────────────────────────────────────────────────────────────────────

def _run_self_test() -> None:
    """NEEDS_REVIEW 判定と confidence 計算の単体テスト."""
    applier = Step3AutoApplier()

    # ケース1: スコアリングファイル → NEEDS_REVIEW 期待
    imp_scoring = {
        "id": "REV-T01",
        "target_module": "quantum_analysis_module.py",
        "title": "quantum_risk の閾値を35から32に下げる",
        "description": "スコアリング閾値を32に変更する",
        "reason": "テストケースで甘い",
        "priority": "HIGH",
    }
    val_ok = {
        "status": "APPROVED",
        "verification_report": "OK",
        "critical_flaws": [],
        "metadata": {"issues_count": 0, "flaws_count": 0, "test_breaking": False},
    }
    needs_r, reason = applier._check_needs_review(imp_scoring, val_ok)
    print(f"[T01] NEEDS_REVIEW={needs_r}, reason='{reason}'")
    assert needs_r, "スコアリングファイルは NEEDS_REVIEW になるべき"

    # ケース2: 通常の改善 → NEEDS_REVIEW 不要
    imp_normal = {
        "id": "REV-T02",
        "target_module": "utils.py",
        "title": "ログ出力の冗長な文字列を短縮する",
        "description": "print文をlogger.info に置き換える",
        "reason": "標準化",
        "priority": "LOW",
    }
    needs_r2, reason2 = applier._check_needs_review(imp_normal, val_ok)
    print(f"[T02] NEEDS_REVIEW={needs_r2}, reason='{reason2}'")

    # ケース3: confidence_score 計算
    val_low = {
        "status": "APPROVED",
        "verification_report": "",
        "critical_flaws": [],
        "metadata": {"issues_count": 3, "flaws_count": 1, "test_breaking": False},
    }
    conf = applier._compute_confidence(val_low)
    print(f"[T03] confidence={conf:.2f}  (expected < 0.7)")
    assert conf < 0.7, f"confidence={conf} は 0.7 未満になるべき"

    print("\n✅ 全テスト通過")


if __name__ == "__main__":
    _run_self_test()
