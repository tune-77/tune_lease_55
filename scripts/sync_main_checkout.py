#!/usr/bin/env python3
"""定期処理の前に、メインチェックアウトが最新の master かを確かめ、安全なら fast-forward する。

launchd の定期処理と API はメインチェックアウトのコードで動く。作業ブランチや古い master の
ままだと古いコードで動く（2026-10 に2回発生: feat/pageindex-evaluation-pilot・
fix/agent-hub-datetime-context・遅れた master）。日次パイプラインの最初に1回走る。

- master かつ origin/master より遅れていて、ローカルコミットが無い → ff する。
  未コミット変更（data/ は日常的にある）と上流の変更ファイルが重なる場合、作業ツリーの内容が
  上流と同一のファイルだけ HEAD に揃えてから ff（中身は ff 後も同じ）。1つでも違えば何もしない。
- master 以外 → 作業中の可能性があるので切り替えない。朝報の警告ブロックに出す。
- fetch 失敗などで判定できない → 何も変えず警告のみ（パイプラインは止めない）。

終了コード: 0 = 最新 / ff 済み、1 = 要確認（朝報に出る）。結果は data/main_checkout_status.json。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATUS = ROOT / "data" / "main_checkout_status.json"
MAIN_BRANCH = "master"
UPSTREAM = f"origin/{MAIN_BRANCH}"


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=timeout)


def _out(repo: Path, *args: str) -> str:
    proc = _git(repo, *args)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {(proc.stderr or proc.stdout).strip()[:200]}")
    return proc.stdout.strip()


def dirty_paths(repo: Path) -> list[str]:
    """未コミット（ステージ済み・未ステージ・未追跡）のパス。リネームは新旧両方。"""
    proc = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if proc.returncode != 0:
        raise RuntimeError(f"git status: {proc.stderr.strip()[:200]}")
    # strip しない（先頭エントリの " M path" の空白が XY 列の一部）
    entries = proc.stdout.split("\0") if proc.stdout else []
    paths: list[str] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        if len(entry) > 3:
            paths.append(entry[3:])
            if entry[0] in "RC":  # 次の要素がリネーム元
                i += 1
                if i < len(entries) and entries[i]:
                    paths.append(entries[i])
        i += 1
    return paths


def inspect(repo: Path, *, fetch: bool = True) -> dict[str, Any]:
    """ブランチ・遅れ・先行・未コミット件数を読む（変更はしない）。fetch=False ならローカルの origin/master で判定。"""
    info: dict[str, Any] = {"checked_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    if fetch:
        try:
            proc = _git(repo, "fetch", "--quiet", "origin", MAIN_BRANCH, timeout=90)
        except subprocess.TimeoutExpired:
            proc = None
        if proc is None or proc.returncode != 0:
            detail = "timeout" if proc is None else (proc.stderr or proc.stdout).strip()[:200]
            info.update({"status": "undetermined", "error": f"git fetch 失敗: {detail}"})
            fetch_failed = True
        else:
            fetch_failed = False
    else:
        fetch_failed = False
    try:
        info["branch"] = _out(repo, "rev-parse", "--abbrev-ref", "HEAD")
        info["uncommitted"] = len(dirty_paths(repo))
        info["behind"] = int(_out(repo, "rev-list", "--count", f"HEAD..{UPSTREAM}"))
        info["ahead"] = int(_out(repo, "rev-list", "--count", f"{UPSTREAM}..HEAD"))
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        info.update({"status": "undetermined", "error": str(exc)[:300]})
        return info
    if fetch_failed:
        return info
    if info["branch"] != MAIN_BRANCH:
        info["status"] = "not_main_branch"
    elif info["behind"] == 0:
        info["status"] = "current"
    elif info["ahead"] > 0:
        info["status"] = "diverged"
    else:
        info["status"] = "behind"
    return info


def _same_as_upstream(repo: Path, path: str) -> bool:
    """作業ツリーの内容が origin/master の同じパスと完全一致するか（どちらかに無ければ不一致）。"""
    file = repo / path
    if not file.is_file():
        return False
    upstream = _git(repo, "rev-parse", "--verify", "--quiet", f"{UPSTREAM}:{path}")
    if upstream.returncode != 0:
        return False
    return _out(repo, "hash-object", "--", path) == upstream.stdout.strip()


def fast_forward(repo: Path, info: dict[str, Any]) -> dict[str, Any]:
    """status=behind の時だけ呼ぶ。重なりを確かめ、安全なら揃えて ff する。"""
    changed = set(filter(None, _out(repo, "diff", "--name-only", "HEAD", UPSTREAM).splitlines()))
    overlap = sorted(changed & set(dirty_paths(repo)))
    differing = [path for path in overlap if not _same_as_upstream(repo, path)]
    if differing:
        info.update({"status": "blocked_by_local_changes", "conflicting_files": differing[:20]})
        return info
    tracked = set(filter(None, _out(repo, "ls-files", "--", *overlap).splitlines())) if overlap else set()
    for path in overlap:
        # 中身は上流と同一なので、ff 後も同じ内容に戻る。未追跡なら消して ff に作らせる
        if path in tracked:
            _out(repo, "checkout", "HEAD", "--", path)
        else:
            (repo / path).unlink()
    before = _out(repo, "rev-parse", "--short", "HEAD")
    proc = _git(repo, "merge", "--ff-only", UPSTREAM)
    if proc.returncode != 0:
        for path in overlap:  # 揃えた分を上流の内容で書き戻す（揃える前と同一）
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_bytes(_git_show_bytes(repo, f"{UPSTREAM}:{path}"))
        info.update({"status": "ff_failed", "error": (proc.stderr or proc.stdout).strip()[:300]})
        return info
    info.update(
        {
            "status": "fast_forwarded",
            "from": before,
            "to": _out(repo, "rev-parse", "--short", "HEAD"),
            "aligned_files": overlap,
            "behind": 0,
        }
    )
    return info


def _git_show_bytes(repo: Path, spec: str) -> bytes:
    return subprocess.run(["git", "show", spec], cwd=repo, capture_output=True, check=True).stdout


def sync(repo: Path, *, apply: bool = True) -> dict[str, Any]:
    info = inspect(repo, fetch=True)
    if info.get("status") == "behind" and apply:
        try:
            info = fast_forward(repo, info)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            info.update({"status": "ff_failed", "error": str(exc)[:300]})
    return info


OK_STATUSES = {"current", "fast_forwarded"}


def warning_line(info: dict[str, Any]) -> str | None:
    """朝報の警告ブロック用。問題が無ければ None。"""
    status = info.get("status")
    if status in OK_STATUSES or not status:
        return None
    branch = info.get("branch", "?")
    counts = f"未コミット{info.get('uncommitted', '?')}件、{UPSTREAM} より{info.get('behind', '?')}コミット遅れ"
    if status == "not_main_branch":
        return f"🌿 メインチェックアウトが master 以外: `{branch}`（{counts}）。定期処理と API はこのブランチのコードで動く"
    if status == "behind":
        return f"🌿 メインチェックアウトの master が古い（{counts}）。次回の日次パイプラインで自動 ff 予定"
    if status == "diverged":
        return f"🌿 master にローカルコミット{info.get('ahead')}件があり自動 ff できない（{counts}）"
    if status == "blocked_by_local_changes":
        files = ", ".join(info.get("conflicting_files") or [])
        return f"🌿 master が古いが、未コミット変更と上流の変更が重なり自動 ff を見送り（{counts}）: {files}"
    return f"🌿 メインチェックアウトの鮮度を判定できず（{status}）: {info.get('error', '')}"


def write_status(path: Path, info: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"[main-checkout] status write failed: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="判定だけして ff しない")
    args = parser.parse_args()
    repo = args.repo.resolve()
    info = sync(repo, apply=not args.dry_run)
    write_status(args.status or repo / "data" / "main_checkout_status.json", info)
    line = warning_line(info)
    summary = {k: info.get(k) for k in ("status", "branch", "behind", "ahead", "uncommitted", "from", "to", "aligned_files")}
    print(f"main_checkout: {json.dumps(summary, ensure_ascii=False)}")
    if line:
        print(f"警告: {line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
