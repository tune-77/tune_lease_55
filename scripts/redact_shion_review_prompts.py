#!/usr/bin/env python3
"""保存済みの紫苑レビュー依頼文（企業名・営業メモ入り）を伏字に書き換える（2026-10-03, PR #1229 の後始末）。

PR #1229 より前は、審査分析画面の紫苑レビュー依頼文が /api/chat 経由で蒸留ノート・改善ログ・会話ログ・
Cloud Run 入力・言語素材・各種ログに伏字なしで残り、Vault → ChromaDB / Vertex 同期にも載っていた。
削除はせず、依頼文を含む記録だけを伏字化し、レビューの論点・判断の要点は残す。

- 対象: 依頼文の見出しを含む Vault の Markdown、data/ の jsonl・json・Vertex エクスポート、lease_data.db の会話履歴
  （shion_screening_reviews.prompt_text は案件データ本体と同じ行にある正規の記録なので対象外）
- 伏字: 依頼文の「企業名:」行から社名を拾い、同じ記録（Markdown は1ファイル・jsonl は1行・json は1文字列）の中で
  〈企業〉に置き換えたうえで api.vertex_query_mask.mask_for_vertex を通す（人物・金額・営業メモ欄・ラベル付きの値）
- 退避: 書き換え前の原本（DB は該当行）を、scripts/backup_case_data.py と同じ AES-256-GCM・キーチェーンの鍵で暗号化して
  iCloud のバックアップ領域へ置く。平文はディスクに書かない（メモリ上で tar にまとめて暗号化）。復号して照合してから書き換える
- 既定はドライラン（件数だけ）。--apply で退避と書き換えを行う

  .venv/bin/python scripts/redact_shion_review_prompts.py            # 件数の確認
  .venv/bin/python scripts/redact_shion_review_prompts.py --apply    # 退避＋書き換え
  復元: python scripts/restore_case_data_backup.py <archive> --out <dir>（展開して中身を確かめてから手で戻す）
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import sqlite3
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from api.vertex_query_mask import mask_for_vertex  # noqa: E402
from backup_case_data import DEFAULT_BACKUP_ROOT, decrypt_bytes, encrypt_bytes, load_key  # noqa: E402

MARKERS = ("【審査分析画面からの紫苑レビュー依頼】", "この案件を、審査担当者の横にいる紫苑としてレビューしてください")
COMPANY_TOKEN = "〈企業〉"
BACKUP_DIR = DEFAULT_BACKUP_ROOT.parent / "redaction"
ARCHIVE_PREFIX = "shion_review_redaction"
DATA_GLOBS = ("data/**/*.jsonl", "data/**/*.json", "data/agent_search/lease_knowledge_export/*.txt")
# 依頼文の「・企業名: X」。JSON 文字列の生の形（\n がエスケープされたまま）でも値だけを取る
_COMPANY_LINE_RE = re.compile(r"企業名\s*[:：]\s*([^\s\\・「」\"]{2,40})")
# 依頼文の金額欄。プレビューで途中が切れた値（「55百…」）は mask_for_vertex の金額規則に掛からないのでラベルで伏せる
_AMOUNT_LABEL_RE = re.compile(r"(取得価額|銀行与信残高|リース与信残高(?:（他社含む）)?)\s*[:：]\s*[^\s\\、。,，\"]+")
_HIRAGANA_RE = re.compile(r"^[ぁ-ゖー]+$")
_KATAKANA_RE = re.compile(r"^[ァ-ヺー]+$")
_YAML_STRING_LINE_RE = re.compile(r'^(\s*[\w-]+:\s*)(".*")\s*$')


def has_marker(text: str) -> bool:
    return any(marker in text for marker in MARKERS)


def company_names(text: str) -> set[str]:
    names = {m.group(1).strip() for m in _COMPANY_LINE_RE.finditer(text)}
    return {name for name in names if len(name) >= 2 and name not in {"未入力", "〈伏字〉", "<伏>"} and "〈" not in name and "[" not in name}


def _name_pattern(name: str) -> re.Pattern[str] | None:
    """短い仮名の社名は語の一部（「わかりにくい」の「くい」等）を壊さないよう境界を付ける。

    ひらがな2文字以下はテスト入力（「うう」等）で、語の一部と区別できないので置換しない
    （依頼文の「企業名:」行自体はラベル付きの値として mask_for_vertex が伏せる）。
    """
    if _HIRAGANA_RE.match(name):
        return None if len(name) <= 2 else re.compile(rf"(?<![ぁ-ゖ]){re.escape(name)}(?![ぁ-ゖー]{{2}})")
    if _KATAKANA_RE.match(name) and len(name) <= 3:
        return re.compile(rf"(?<![ァ-ヺー]){re.escape(name)}(?![ァ-ヺー])")
    return re.compile(re.escape(name))


def mask_text(text: str, names: set[str]) -> str:
    """社名を〈企業〉に置き換えてから mask_for_vertex。エスケープされた改行（\\n）の区切りごとに処理して構造を保つ。"""
    for name in sorted(names, key=len, reverse=True):
        pattern = _name_pattern(name)
        if pattern is not None:
            text = pattern.sub(COMPANY_TOKEN, text)
    return "\\n".join(
        _AMOUNT_LABEL_RE.sub(lambda m: f"{m.group(1)}: 〈金額〉", mask_for_vertex(segment)) for segment in text.split("\\n")
    )


def mask_markdown(text: str, names: set[str]) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        yaml = _YAML_STRING_LINE_RE.match(line)
        if yaml:  # frontmatter の JSON 文字列（question: "..."）は復号→伏字→再エンコードで引用符を壊さない
            try:
                value = json.loads(yaml.group(2))
            except json.JSONDecodeError:
                value = None
            if isinstance(value, str):
                out.append(yaml.group(1) + json.dumps(mask_text(value, names).replace("\\n", "\n"), ensure_ascii=False))
                continue
        out.append(mask_text(line, names))
    return "\n".join(out)


def _walk_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, list):
        return [_walk_strings(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _walk_strings(item, fn) for key, item in value.items()}
    return value


def mask_jsonl(text: str) -> str:
    """依頼文を含む行だけ、その行の文字列を全部伏字にする（返答・user_id の社名も同じ行で消す）。他の行はそのまま。"""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if not has_marker(line):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            lines[index] = mask_text(line, company_names(line))
            continue
        names = company_names(json.dumps(record, ensure_ascii=False))
        lines[index] = json.dumps(_walk_strings(record, lambda s: mask_text(s, names)), ensure_ascii=False)
    return "\n".join(lines)


def mask_json(text: str) -> str:
    """json は記録の単位が決まらないので、依頼文を含む文字列と、そこから拾った社名を含む文字列を伏字にする。"""
    document = json.loads(text)
    found: list[str] = []
    _walk_strings(document, lambda s: found.append(s) or s)
    names = set().union(*(company_names(s) for s in found if has_marker(s))) if found else set()
    masked = _walk_strings(document, lambda s: mask_text(s, names) if has_marker(s) or any(n in s for n in names) else s)
    indent = 2 if text.lstrip().startswith(("{\n", "[\n")) else None
    return json.dumps(masked, ensure_ascii=False, indent=indent) + ("\n" if text.endswith("\n") else "")


def redact_file_text(path: Path, text: str) -> str:
    if path.suffix == ".jsonl":
        return mask_jsonl(text)
    if path.suffix == ".json":
        return mask_json(text)
    return mask_markdown(text, company_names(text))


@dataclass
class Change:
    kind: str  # vault / data / sqlite
    path: Path
    rel: str
    original: bytes
    redacted: bytes
    rows: list[dict] = field(default_factory=list)  # sqlite の該当行（原本）


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def find_file_changes(vault: Path | None, repo_root: Path = REPO_ROOT) -> tuple[list[Change], list[str]]:
    changes: list[Change] = []
    marker_only: list[str] = []  # 依頼文の見出しはあるが伏字にする部分が無い（MOC のリンク等）
    candidates: list[tuple[str, Path, Path]] = []
    if vault and vault.exists():
        candidates += [("vault", vault, p) for p in vault.rglob("*.md") if not p.name.endswith(".icloud")]
    for pattern in DATA_GLOBS:
        candidates += [("data", repo_root, p) for p in repo_root.glob(pattern) if p.is_file()]
    seen: set[Path] = set()
    for kind, root, path in candidates:
        if path in seen:
            continue
        seen.add(path)
        text = _read(path)
        if text is None or not has_marker(text):
            continue
        rel = str(path.relative_to(root))
        redacted = redact_file_text(path, text)
        if redacted == text:
            marker_only.append(f"{kind}:{rel}")
            continue
        changes.append(Change(kind, path, rel, text.encode("utf-8"), redacted.encode("utf-8")))
    return changes, marker_only


def find_sqlite_change(db: Path) -> Change | None:
    if not db.exists():
        return None
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "select id, user_id, content from chat_messages where " + " or ".join("content like ?" for _ in MARKERS),
            [f"%{m}%" for m in MARKERS],
        )]
        # 同じ紫苑レビュー会話（user_id）の返答にも社名が出るので、その会話の全行を対象にする
        user_ids = sorted({r["user_id"] for r in rows})
        if user_ids:
            rows = [dict(r) for r in conn.execute(
                f"select id, user_id, content from chat_messages where user_id in ({','.join('?' * len(user_ids))})", user_ids
            )]
    if not rows:
        return None
    names = set().union(*(company_names(r["content"] or "") for r in rows))
    redacted_rows = [{**r, "user_id": mask_text(r["user_id"], names), "content": mask_text(r["content"] or "", names)} for r in rows]
    changed = [new for old, new in zip(rows, redacted_rows) if old != new]
    if not changed:
        return None
    original = json.dumps(rows, ensure_ascii=False, indent=1).encode("utf-8")
    return Change("sqlite", db, "lease_data.db#chat_messages", original, json.dumps(changed, ensure_ascii=False).encode("utf-8"), rows=changed)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive(changes: list[Change]) -> tuple[bytes, dict]:
    """原本だけをメモリ上の tar.gz にまとめる（平文をディスクに書かない）。"""
    manifest = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "紫苑レビュー依頼文の伏字化前の原本（scripts/redact_shion_review_prompts.py）",
        "files": [{"kind": c.kind, "rel": c.rel, "path": str(c.path), "sha256": _sha(c.original), "redacted_sha256": _sha(c.redacted)} for c in changes],
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = int(dt.datetime.now().timestamp())
            tar.addfile(info, io.BytesIO(data))

        add("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        for c in changes:
            add(f"{c.kind}/{c.rel}" if c.kind != "sqlite" else "sqlite/lease_data.db.chat_messages.json", c.original)
    return buf.getvalue(), manifest


def verify_archive(blob: bytes, key: bytes, changes: list[Change]) -> None:
    with tarfile.open(fileobj=io.BytesIO(decrypt_bytes(key, blob)), mode="r:gz") as tar:
        for c in changes:
            name = f"{c.kind}/{c.rel}" if c.kind != "sqlite" else "sqlite/lease_data.db.chat_messages.json"
            member = tar.extractfile(name)
            if member is None or _sha(member.read()) != _sha(c.original):
                raise RuntimeError(f"退避アーカイブの照合に失敗: {name}")


def write_redacted(change: Change) -> None:
    if change.kind == "sqlite":
        with sqlite3.connect(change.path) as conn:
            conn.executemany("update chat_messages set user_id = ?, content = ? where id = ?", [(r["user_id"], r["content"], r["id"]) for r in change.rows])
        return
    # 稼働中の API が追記するログがあるので、原本を読んだ後に増えた分は伏字にして後ろへ足す
    current = change.path.read_bytes()
    tail = current[len(change.original):] if current.startswith(change.original) else b""
    if not current.startswith(change.original):
        raise RuntimeError(f"読み取り後にファイルが書き換わった（追記以外）: {change.rel}")
    if tail:
        tail = redact_file_text(change.path, tail.decode("utf-8")).encode("utf-8") if change.path.suffix == ".jsonl" else tail
    tmp = change.path.with_name(change.path.name + ".redact.tmp")
    tmp.write_bytes(change.redacted + tail)
    tmp.replace(change.path)


def run(apply: bool, vault: Path | None, repo_root: Path = REPO_ROOT) -> dict:
    changes, marker_only = find_file_changes(vault, repo_root)
    sqlite_change = find_sqlite_change(repo_root / "data" / "lease_data.db")
    if sqlite_change:
        changes.append(sqlite_change)
    summary: dict[str, Any] = {
        "vault_files": sum(1 for c in changes if c.kind == "vault"),
        "data_files": sum(1 for c in changes if c.kind == "data"),
        "sqlite_rows": len(sqlite_change.rows) if sqlite_change else 0,
        "marker_only_unchanged": marker_only,
        "by_dir": {},
        "applied": False,
    }
    for c in changes:
        top = c.kind + ":" + ("/".join(Path(c.rel).parts[:-1]) or ".")
        summary["by_dir"][top] = summary["by_dir"].get(top, 0) + 1
    if not apply or not changes:
        return summary
    key = load_key()  # 鍵が無ければ何も書かずにここで失敗する
    blob_plain, manifest = build_archive(changes)
    blob = encrypt_bytes(key, blob_plain)
    del blob_plain
    verify_archive(blob, key, changes)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    archive = BACKUP_DIR / f"{ARCHIVE_PREFIX}_{dt.datetime.now():%Y%m%d_%H%M%S}.tar.gz.enc"
    part = archive.with_name(archive.name + ".part")
    part.write_bytes(blob)
    part.replace(archive)
    for change in changes:
        write_redacted(change)
    summary.update({"applied": True, "archive": str(archive), "archived_files": len(manifest["files"])})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="保存済みの紫苑レビュー依頼文を伏字に書き換える（原本は暗号化して退避）")
    parser.add_argument("--apply", action="store_true", help="退避と書き換えを行う（既定は件数の確認だけ）")
    parser.add_argument("--vault", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="data/ を持つチェックアウト（worktree から実行する時はメイン）")
    args = parser.parse_args()
    vault = args.vault
    if vault is None:
        from runtime_paths import resolve_obsidian_vault

        vault = resolve_obsidian_vault()
    print(json.dumps(run(args.apply, vault, args.repo_root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
