"""Machine-enforced safety policy shared by every auto-improvement path."""

from __future__ import annotations

import fnmatch
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


CONFIG_PATH = Path(__file__).resolve().parent.parent / "loop_constraints.json"
ATTEMPT_STATE_RELATIVE_PATH = Path(".claude/state/auto_improvement_attempts.json")


def load_loop_constraints(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"denylist", "human_gate", "limits", "roles"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"loop constraints missing sections: {sorted(missing)}")
    if data["roles"].get("must_differ") and data["roles"].get("implementer") == data["roles"].get("verifier"):
        raise ValueError("implementer and verifier roles must differ")
    return data


def normalize_repo_path(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return PurePosixPath(normalized).as_posix()


def matches_any_path(path: str | Path, patterns: Iterable[str]) -> str | None:
    normalized = normalize_repo_path(path)
    for pattern in patterns:
        if fnmatch.fnmatchcase(normalized, pattern) or PurePosixPath(normalized).match(pattern):
            return pattern
    return None


def candidate_text(improvement: dict[str, Any]) -> str:
    return " ".join(
        str(improvement.get(key) or "")
        for key in ("title", "description", "reason", "target_module")
    ).lower()


def evaluate_execution_constraints(
    improvement: dict[str, Any],
    paths: Iterable[str | Path],
    *,
    attempt_count: int = 0,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one binding decision for denylist, human gates and retry limits."""
    cfg = constraints or load_loop_constraints()
    normalized_paths = sorted({normalize_repo_path(path) for path in paths if str(path).strip()})
    limits = cfg["limits"]

    if attempt_count >= int(limits["max_attempts_per_item"]):
        return {
            "allowed": False,
            "decision": "max_attempts_exceeded",
            "human_gate": True,
            "reason": f"最大試行回数 {limits['max_attempts_per_item']} 回に到達",
        }

    if len(normalized_paths) > int(limits["max_files_per_change"]):
        return {
            "allowed": False,
            "decision": "human_gate",
            "human_gate": True,
            "reason": f"変更対象が上限を超過: {len(normalized_paths)} files",
        }

    for path in normalized_paths:
        matched = matches_any_path(path, cfg["denylist"])
        if matched:
            return {
                "allowed": False,
                "decision": "denylist",
                "human_gate": True,
                "reason": f"denylist path: {path} ({matched})",
            }
        matched = matches_any_path(path, cfg["human_gate"]["paths"])
        if matched:
            return {
                "allowed": False,
                "decision": "human_gate",
                "human_gate": True,
                "reason": f"human-gated path: {path} ({matched})",
            }

    text = candidate_text(improvement)
    keyword = next((word for word in cfg["human_gate"]["keywords"] if str(word).lower() in text), None)
    if keyword:
        return {
            "allowed": False,
            "decision": "human_gate",
            "human_gate": True,
            "reason": f"human-gated keyword: {keyword}",
        }

    return {
        "allowed": True,
        "decision": "auto",
        "human_gate": False,
        "reason": "機械制約を通過",
        "max_attempts": int(limits["max_attempts_per_item"]),
        "roles": dict(cfg["roles"]),
    }


class AttemptLedger:
    """Persistent per-improvement attempt budget with atomic state writes."""

    def __init__(self, workspace_root: str | Path, constraints: dict[str, Any] | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.constraints = constraints or load_loop_constraints()
        self.path = self.workspace_root / ATTEMPT_STATE_RELATIVE_PATH

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        now = datetime.now(timezone.utc)
        retention = timedelta(days=int(self.constraints["limits"]["attempt_retention_days"]))
        kept: dict[str, Any] = {}
        for key, entry in data.items():
            try:
                updated = datetime.fromisoformat(str(entry["updated_at"]))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
            except (KeyError, TypeError, ValueError):
                continue
            if now - updated <= retention:
                kept[str(key)] = entry
        return kept

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="attempts-", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
            Path(temp_name).replace(self.path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def count(self, key: str) -> int:
        return int(self._load().get(key, {}).get("count", 0))

    def begin(self, key: str) -> dict[str, Any]:
        with self._lock():
            state = self._load()
            previous = int(state.get(key, {}).get("count", 0))
            maximum = int(self.constraints["limits"]["max_attempts_per_item"])
            if previous >= maximum:
                return {"allowed": False, "count": previous, "max_attempts": maximum}
            count = previous + 1
            state[key] = {
                "count": count,
                "status": "running",
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            self._save(state)
            return {"allowed": True, "count": count, "max_attempts": maximum}

    def finish(self, key: str, status: str, detail: str = "") -> None:
        with self._lock():
            state = self._load()
            entry = dict(state.get(key, {}))
            entry.update({
                "status": status,
                "detail": detail[:300],
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            state[key] = entry
            self._save(state)
