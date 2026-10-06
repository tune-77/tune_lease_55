"""紫苑の行動とコード参照を、実際にできること・実在するものに接地させる（REV-482）。

2026-10-06、対話室で「紫苑依頼分にして」と頼まれた紫苑が、次のような依頼文を作った:
- 「まずは手元の環境にて検証を開始します。結果が出次第、報告いたします」
  （紫苑はコードを実行できないし、後で結果を報告する仕組みもない）
- `screening_history.db`・`scripts.analyze_financial_health` など、リポジトリに無い名前
- 「既存の審査履歴DBに対し、フラグを付与するクエリの実行」（本番DBへの書き込み）

REV-481（感情の自己報告の接地）と同じ考え方で、次の3段で抑える:
1. 依頼文を頼まれた時だけ、実在するファイル・関数の一覧と禁止範囲をプロンプトに渡す。
2. 返答後に決定的な照合をする（その場で反映）:
   - 依頼文中の `ファイル名`・`関数名`・`コマンド` の実在を確認し、無いものに「要確認」を付ける
   - 本番DB・data/ への書き込みを含む行に「要確認」を付ける
   - 紫苑自身が実行・検証・後の報告を約束する文を除き、「依頼文は Claude Code／User が実行する」と添える
3. 照合結果と、Jev による文の分類（実行できない約束の取りこぼし検出）をログに残す。
   Jev はバックグラウンドで、伏せた文だけを送る。

環境変数:
  SHION_REQUEST_GROUNDING  on（既定） | off … 返答の書き換え（2）を止める（プロンプトの指示は残る）
  SHION_REQUEST_VERIFY     on（既定・TypeSafe鍵がある時だけ動く） | off … Jev 照合
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from runtime_paths import get_data_path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUEST_MARKER_RE = re.compile(r"(?:紫苑|Codex|Claude ?Code|開発|実装|修正)?依頼(?:文|分)")
_REQUEST_ASK_RE = re.compile(r"依頼(?:文|分)|Claude ?Code(?:に|へ|向け)|Codex(?:に|へ|向け)|Dispatch(?:に|へ|向け)")

# 紫苑自身が「実行・検証・実装・後の報告」をすると言う文。
# 返答中のツール呼び出し（調べます・確認します）は実際にできるので対象にしない（lease_intelligence_pending が追跡する）。
_SELF_ACTION_PATTERNS = (
    r"(?:手元|ローカル|こちら|私|わたし|紫苑)(?:の|側の)?(?:環境|側|手元)?(?:にて|で)[^。！？\n]*?(?:検証|実行|テスト|分析|試|集計|抽出)[^。！？\n]*?ます",
    # 分析・抽出・集計は返答の中で記録を読んで行えるので、ここでは対象にしない（環境・後での報告と結びつく時だけ）
    r"(?:検証|テスト|実装|改修|修正)(?:作業)?を?(?:開始|着手|実施|実行|進め|始め|行い)(?:し|いたし|させていただき)?ます",
    r"(?:実装|検証|修正)に(?:着手|入り|取りかかり|取り掛かり)(?:し|いたし)?ます",
    r"結果が(?:出|わかり|分かり|まとまり)(?:次第|ましたら|たら)",
    r"(?:実装|検証|コミット|デプロイ|反映|実行)(?:の)?準備を(?:行い|進め|整え|し)ます",
    r"(?:次回|後ほど|後で|改めて|完了(?:し)?(?:次第|後))[^。！？\n]*?(?:検証|実行|分析|実装|集計)[^。！？\n]*?(?:報告|お知らせ|共有|お伝え)",
    r"(?:コード|スクリプト|クエリ|SQL|コマンド|テスト)を(?:実行|走らせ|流|回)(?:し|いたし)?ます",
    r"(?:コミット|デプロイ|マージ|本番反映|プッシュ)(?:を)?(?:し|いたし|行い|実施し)ます",
    r"(?:検証|実行|分析|集計|抽出|テスト)(?:の)?結果[^。！？\n]*?(?:報告|お知らせ|共有|お伝え)(?:し|いたし|させていただき)?ます",
    r"(?:実行|検証|実装|分析)の準備(?:は|が)?(?:整い|でき)",
    r"(?:実行|検証|分析|実装)(?:後|完了後|が(?:終わ|済ん|完了し)(?:ったら|だら|たら))[^。！？\n]*?(?:報告|お知らせ|共有|お伝え)(?:し|いたし|させていただき)?ます",
    r"(?:次回|次の対話|後ほど|のちほど)(?:の対話)?(?:まで)?に[^。！？\n]*?(?:提示|用意|お持ち|まとめ|お届け)(?:し|いたし|させていただき)?ます",
)
_SELF_ACTION_RE = re.compile("|".join(_SELF_ACTION_PATTERNS))
# 主語が実装者・User の文（依頼文の中身）は紫苑の約束ではない
_DELEGATED_RE = re.compile(
    r"(?:Claude ?Code|Codex|User|ユーザー|開発者|実装者)(?:が|に|へ|側|の方)|実行者|してください|して下さい|お願いします"
)

_DB_WRITE_RE = re.compile(
    r"\b(?:UPDATE|INSERT|DELETE\s+FROM|ALTER\s+TABLE|DROP\s+TABLE|CREATE\s+TABLE|REPLACE\s+INTO)\b"
    r"|フラグを(?:付与|付け|立て)|付与する|書き込|書込|上書き|追記|更新する|更新(?:の)?クエリ|カラム(?:を)?追加",
    re.IGNORECASE,
)
_DB_TARGET_RE = re.compile(r"DB|ＤＢ|データベース|テーブル|data/|\.db\b|\.sqlite|審査履歴|履歴|レコード", re.IGNORECASE)
_READ_ONLY_RE = re.compile(r"読み取り専用|読取専用|-readonly|mode=ro|コピー上|コピーした|複製|書き込まない|変更しない|変更禁止|禁止")

_CODE_SPAN_RE = re.compile(r"`([^`\n]{2,200})`")
_BARE_PATH_RE = re.compile(r"(?<![\w./`-])((?:[\w-]+/)*[\w-]+\.(?:py|tsx?|db|sqlite3?|sh|jsonl?|ya?ml|toml))(?![\w`])")
_PATH_EXTS = (".py", ".ts", ".tsx", ".js", ".db", ".sqlite", ".sqlite3", ".sh", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".md")
_COMMAND_HEADS = ("python", "python3", "pytest", "uv", "npm", "npx", "node", "bash", "sh", "sqlite3", "cd", "make", "git")
_DOTTED_MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
_IDENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\(.*\))?$")
_NEW_FILE_RE = re.compile(r"新規|新しく作|新設|作成する|追加する|出力|書き出|保存先|create|output", re.IGNORECASE)
_SYMBOL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_SYMBOL_EXTS = (".py", ".ts", ".tsx", ".js", ".mjs", ".sh", ".yml", ".yaml", ".toml", ".sql", ".plist")
_SENTENCE_RE = re.compile(r"[^。！？!?]*[。！？!?]?")

MISSING_NOTE = "（要確認: リポジトリに見当たらない）"
DB_WRITE_NOTE = "（要確認: 本番DB・data/ への書き込みは禁止。分析は読み取り専用か、コピー上で行う）"
EXECUTOR_NOTE = "※ 紫苑はコードの実行・検証・実装・デプロイはできません。依頼文は Claude Code か User が実行するものです。"
EXECUTOR_NOTE_PLAIN = "※ 紫苑はコードの実行・検証・実装はできません。必要なら依頼文にして、Claude Code か User が実行します。"

MAX_VERIFY_SENTENCES = 12
MAX_SENTENCE_CHARS = 300
LOG_ROTATE_BYTES = 2_000_000
MAX_PENDING_VERIFICATIONS = 4
VERDICTS = ("unexecutable_self_action", "lookup_in_reply", "delegated_task", "other")
_LOG_LOCK = threading.Lock()


def is_request_turn(message: str) -> bool:
    """User が実装・検証の依頼文（紫苑依頼文／依頼分・Claude Code 向け）を求めているか。"""
    return bool(_REQUEST_ASK_RE.search(str(message or "")))


def _env_on(name: str, default: str) -> bool:
    return str(os.environ.get(name) or default).strip().lower() not in {"0", "off", "false", "no", ""}


# ── 実在の確認 ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True, timeout=20, check=True
        ).stdout
        files = [line.strip() for line in out.splitlines() if line.strip()]
        if files:
            return tuple(files)
    except Exception:
        pass
    skip = {".git", "node_modules", ".next", "__pycache__", ".venv", "venv", "data", "models"}
    files = []
    for root, dirs, names in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in names:
            files.append(str(Path(root, name).relative_to(REPO_ROOT)))
    return tuple(files)


@lru_cache(maxsize=1)
def _basenames() -> frozenset[str]:
    return frozenset(Path(path).name for path in _tracked_files())


@lru_cache(maxsize=1)
def _symbol_tokens() -> frozenset[str]:
    """リポジトリのコードに出てくる識別子の集合（関数名・変数名・ツール名・設定キー）。初回だけ読む。"""
    tokens: set[str] = set()
    for rel in _tracked_files():
        if not rel.endswith(_SYMBOL_EXTS) or rel.startswith(("frontend/public/", "data/")):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        tokens.update(_SYMBOL_TOKEN_RE.findall(text))
    return frozenset(tokens)


def path_exists(ref: str) -> bool:
    ref = ref.strip().removeprefix("./")
    if not ref:
        return False
    if ref.startswith("~") or ref.startswith("/"):
        return Path(os.path.expanduser(ref)).exists()
    if "/" not in ref:
        if ref in _basenames():
            return True
        return Path(get_data_path(ref)).exists() if ref.endswith((".db", ".sqlite", ".sqlite3", ".jsonl", ".json")) else False
    if ref in set(_tracked_files()) or (REPO_ROOT / ref).exists():
        return True
    if ref.startswith("data/"):
        return Path(get_data_path(ref[len("data/"):])).exists()
    return False


def symbol_exists(name: str) -> bool:
    return name in _symbol_tokens()


def _module_path(module: str) -> str:
    return module.replace(".", "/") + ".py"


@dataclass
class RefCheck:
    ref: str
    kind: str  # path / module / symbol / command
    exists: bool
    detail: str = ""
    source: str = ""  # 返答中の元の書き方（「要確認」を付ける位置）


def _check_command(text: str) -> RefCheck | None:
    parts = text.split()
    if not parts or parts[0] not in _COMMAND_HEADS:
        return None
    if parts[0] in {"cd", "git", "npm", "npx", "node", "make", "uv"}:
        # 既存の型どおりの検証コマンド（cd frontend && npx tsc 等）は照合しない
        return None
    target = ""
    if "-m" in parts:
        idx = parts.index("-m")
        module = parts[idx + 1] if idx + 1 < len(parts) else ""
        if module in {"pytest", "py_compile", "compileall", "unittest", "pip", "json.tool"}:
            target = next((p for p in parts[idx + 2:] if not p.startswith("-")), "")
        else:
            target = _module_path(module)
    elif parts[0] in {"pytest"}:
        target = next((p for p in parts[1:] if not p.startswith("-")), "")
    elif parts[0] in {"python", "python3", "bash", "sh"}:
        target = next((p for p in parts[1:] if not p.startswith("-")), "")
    elif parts[0] == "sqlite3":
        target = next((p for p in parts[1:] if not p.startswith("-")), "")
    target = target.split("::")[0].strip("'\"")
    if not target or not re.search(r"[./]", target):
        return RefCheck(text, "command", True)
    if not path_exists(target):
        return RefCheck(text, "command", False, f"{target} が無い")
    flags = [p.split("=")[0] for p in parts if p.startswith("--")]
    resolved = target if (REPO_ROOT / target).exists() else ""
    if flags and resolved and resolved.endswith(".py") and parts[0] != "pytest" and "pytest" not in parts:
        try:
            source = (REPO_ROOT / resolved).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            source = ""
        unknown = [flag for flag in flags if flag not in source]
        if unknown:
            return RefCheck(text, "command", False, f"{resolved} に引数 {' '.join(unknown)} が無い")
    return RefCheck(text, "command", True)


_SQL_RE = re.compile(r"\b(?:SELECT|UPDATE|INSERT\s+INTO|DELETE\s+FROM)\b", re.IGNORECASE)
_SQL_TABLE_RE = re.compile(r"\b(?:FROM|JOIN|UPDATE|INTO)\s+([A-Za-z_]\w*)", re.IGNORECASE)
_SQL_STRING_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_SQL_WORDS = frozenset(
    """select from where and or not null is in as on join left right inner outer cross distinct count sum avg min max
    group by order having limit offset desc asc case when then else end like between exists union all cast round
    coalesce ifnull substr length lower upper date datetime strftime julianday abs total json_extract update set
    insert into values delete true false glob escape collate integer real text over partition rowid""".split()
)


@lru_cache(maxsize=8)
def _db_schema(db_path: str) -> dict[str, frozenset[str]]:
    """読み取り専用で開いてテーブルと列を読む（照合のためだけ。書き込まない）。"""
    import sqlite3

    if not Path(db_path).exists():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
        return {
            table: frozenset(row[1] for row in conn.execute(f'PRAGMA table_info("{table}")'))
            for table in tables
        }
    finally:
        conn.close()


def check_sql(text: str) -> RefCheck | None:
    """コード中のSQLのテーブル名・列名が、審査履歴DB（data/ の該当DB）に実在するか。"""
    if not _SQL_RE.search(text):
        return None
    db_match = re.search(r"(?:data/)?([\w-]+\.(?:db|sqlite3?))", text)
    db_name = db_match.group(1) if db_match else "lease_data.db"
    try:
        schema = _db_schema(get_data_path(db_name))
    except Exception:
        return None
    if not schema:
        return None
    start = _SQL_RE.search(text).start()
    sql = text[start:]
    quote = text[start - 1] if start and text[start - 1] in "'\"" else ""
    if quote and quote in sql:
        sql = sql[: sql.index(quote)]  # python -c "...execute('SELECT ...')" の文字列部分だけ
    body = _SQL_STRING_RE.sub(" ", sql)
    tables = [t for t in _SQL_TABLE_RE.findall(body)]
    unknown_tables = [t for t in tables if t not in schema]
    label = " ".join(body.split())[:120]  # 文字列リテラル（社名などが入りうる）は伏せた形で残す
    if unknown_tables:
        return RefCheck(label, "sql", False, f"テーブル {', '.join(dict.fromkeys(unknown_tables))} が {db_name} に無い")
    columns = set().union(*(schema[t] for t in tables)) if tables else set()
    unknown_cols = [
        token for token in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", body))
        if token.lower() not in _SQL_WORDS and token not in schema and token not in columns
        and not re.fullmatch(r"[A-Za-z]{1,2}\d*", token)  # 別名（t, sr 等）
    ]
    if tables and unknown_cols:
        return RefCheck(label, "sql", False, f"列 {', '.join(unknown_cols[:5])} が {', '.join(dict.fromkeys(tables))} に無い")
    return RefCheck(label, "sql", True)


def check_reference(ref: str) -> RefCheck | None:
    """1つのコード参照の実在を確かめる。照合の対象外（普通の語・SQL 等）は None。"""
    text = " ".join(str(ref or "").split())
    if not text:
        return None
    sql = check_sql(text)
    if sql is not None:
        return sql
    if re.search(r"[ぁ-んァ-ヶ一-龥]", text):
        return None
    command = _check_command(text)
    if command is not None:
        return command
    if " " in text:
        return None
    if text.startswith("-"):
        return None
    head = text.split("::")[0].split(":")[0]
    if "/" in head or head.lower().endswith(_PATH_EXTS):
        if head.lower().endswith(".md") and "/" not in head:
            return None
        return RefCheck(head, "path", path_exists(head))
    if _DOTTED_MODULE_RE.match(head):
        first = head.split(".")[0]
        if first in {"st", "os", "re", "np", "pd", "json", "datetime", "self", "req", "app", "router"}:
            return None
        module_file = _module_path(head)
        if path_exists(module_file) or (REPO_ROOT / head.replace(".", "/")).is_dir():
            return RefCheck(head, "module", True)
        # module.function の形（scoring_core.calc_score 等）
        owner, _, attr = head.rpartition(".")
        if path_exists(_module_path(owner)):
            return RefCheck(head, "symbol", symbol_exists(attr), "" if symbol_exists(attr) else f"{attr} が無い")
        return RefCheck(head, "module", False, f"{module_file} が無い")
    match = _IDENT_RE.match(head)
    if match:
        name = match.group(1)
        # 普通の英単語（needs, true 等）は照合しない。コードらしい名前だけ
        if "_" not in name and not re.search(r"[a-z][A-Z]", name):
            return None
        return RefCheck(name, "symbol", symbol_exists(name))
    return None


# ── 返答の照合（決定的） ────────────────────────────────────────


@dataclass
class GroundingResult:
    reply: str
    request_detected: bool = False
    refs: list[RefCheck] = field(default_factory=list)
    db_write_lines: int = 0
    removed_promises: list[str] = field(default_factory=list)
    kept_promises: list[str] = field(default_factory=list)
    changed: bool = False

    @property
    def missing_refs(self) -> list[RefCheck]:
        return [ref for ref in self.refs if not ref.exists]

    def summary(self) -> dict[str, Any]:
        return {
            "request_detected": self.request_detected,
            "refs_checked": len(self.refs),
            "missing_refs": [
                {"ref": ref.ref[:120], "kind": ref.kind, **({"detail": ref.detail} if ref.detail else {})}
                for ref in self.missing_refs
            ],
            "db_write_lines": self.db_write_lines,
            "removed_promises": len(self.removed_promises),
            "kept_promises": len(self.kept_promises),
            "changed": self.changed,
        }


def find_self_action_sentences(reply: str) -> list[str]:
    """紫苑自身が実行・検証・実装・後の報告をすると言っている文。"""
    found = []
    for line in str(reply or "").splitlines():
        for sentence in _SENTENCE_RE.findall(line):
            text = sentence.strip()
            if text and _SELF_ACTION_RE.search(text) and not _DELEGATED_RE.search(text):
                found.append(text)
    return found


def _strip_sentences(reply: str, targets: set[str]) -> str:
    lines = []
    for line in reply.splitlines():
        sentences = _SENTENCE_RE.findall(line)
        kept = [s for s in sentences if s.strip() not in targets]
        if len(kept) == len(sentences):
            lines.append(line)
            continue
        new_line = "".join(kept).rstrip()
        # 句点の後の空白だけ残った・記号だけ残った行は消す
        if re.sub(r"[\s*_>#\-—・、。:：]", "", new_line):
            lines.append(new_line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _annotate_refs(reply: str, missing: list[RefCheck]) -> str:
    out = reply
    for ref in missing:
        note = f"（要確認: {ref.detail}）" if ref.detail else MISSING_NOTE
        for candidate in dict.fromkeys(c for c in (f"`{ref.source}`" if ref.source else "", f"`{ref.ref}`") if c):
            if candidate in out and f"{candidate}{note}" not in out:
                out = out.replace(candidate, f"{candidate}{note}", 1)
                break
        else:
            # 参照がコードスパン内の一部（`python3 -m x --y`）や地の文にある場合
            pattern = re.compile(r"`[^`\n]*" + re.escape(ref.ref) + r"[^`\n]*`")
            match = pattern.search(out)
            if match and note not in out[match.end():match.end() + len(note)]:
                out = out[: match.end()] + note + out[match.end():]
            elif ref.ref in out:
                idx = out.index(ref.ref) + len(ref.ref)
                out = out[:idx] + note + out[idx:]
    return out


def _request_section_bounds(reply: str) -> tuple[int, int] | None:
    match = REQUEST_MARKER_RE.search(reply)
    if not match:
        return None
    start = reply.rfind("\n", 0, match.start()) + 1
    return start, len(reply)


def collect_refs(text: str) -> list[RefCheck]:
    checks: list[RefCheck] = []
    seen: set[str] = set()
    candidates = [m.group(1) for m in _CODE_SPAN_RE.finditer(text)]
    stripped = _CODE_SPAN_RE.sub(" ", text)
    candidates += [m.group(1) for m in _BARE_PATH_RE.finditer(stripped)]
    for raw in candidates:
        result = check_reference(raw)
        if result is None or result.ref in seen:
            continue
        result.source = raw
        seen.add(result.ref)
        checks.append(result)
    return checks


def _new_file_refs(text: str, checks: list[RefCheck]) -> set[str]:
    """「新規作成」と書かれた行のファイルは、無くて当然なので要確認にしない。"""
    allowed = set()
    for line in text.splitlines():
        if not _NEW_FILE_RE.search(line):
            continue
        for ref in checks:
            if ref.kind == "path" and ref.ref in line:
                allowed.add(ref.ref)
    return allowed


def ground_reply(message: str, reply: str, *, enabled: bool | None = None) -> GroundingResult:
    """返答を照合し、必要なら「要確認」を付ける・実行できない約束の文を除く。"""
    reply = str(reply or "")
    if enabled is None:
        enabled = _env_on("SHION_REQUEST_GROUNDING", "on")
    bounds = _request_section_bounds(reply)
    request_detected = bool(bounds) or is_request_turn(message)
    result = GroundingResult(reply=reply, request_detected=request_detected)

    promises = find_self_action_sentences(reply)
    if request_detected:
        section = reply[bounds[0]:bounds[1]] if bounds else reply
        checks = collect_refs(section)
        allowed_new = _new_file_refs(section, checks)
        result.refs = [ref for ref in checks if ref.exists or ref.ref not in allowed_new]
        db_lines = [
            line for line in section.splitlines()
            if _DB_WRITE_RE.search(line) and _DB_TARGET_RE.search(line) and not _READ_ONLY_RE.search(line)
        ]
        result.db_write_lines = len(db_lines)
    else:
        db_lines = []

    if not enabled:
        result.kept_promises = promises
        return result

    out = reply
    if promises:
        stripped = _strip_sentences(out, set(promises))
        if len(re.sub(r"\s", "", stripped)) >= 20:
            out = stripped
            result.removed_promises = promises
        else:
            result.kept_promises = promises
    if result.missing_refs:
        out = _annotate_refs(out, result.missing_refs)
    for line in db_lines:
        if line in out and DB_WRITE_NOTE not in line:
            out = out.replace(line, line.rstrip() + DB_WRITE_NOTE, 1)
    note = EXECUTOR_NOTE if request_detected else EXECUTOR_NOTE_PLAIN
    if promises and note not in out:
        out = out.rstrip() + "\n\n" + note
    result.reply = out
    result.changed = out != reply
    return result


# ── プロンプト ────────────────────────────────────────────────

ACTION_RULES = """【紫苑ができること・できないこと（REV-482）】
- 紫苑がこの返答の中でできるのは、ツールで記録を読むことだけ。コード・スクリプト・クエリ・テストの実行、検証環境での検証、実装、git、デプロイはできない。
- 「手元の環境で検証します」「検証を開始します」「結果が出次第報告します」「実装準備を進めます」のような、自分では実行できない約束や、実行したふりをしない。
- 検証や実装が必要なら「Claude Code（または User）が実行する依頼文」として書く。"""

_CATALOG_BASE = (
    ("lease_intelligence_tools.py", "紫苑の読み取りツール（search_cases / get_score_detail / compare_similar_cases / get_weekly_trend）"),
    ("scoring_core.py", "スコア計算の本体（tests/test_scoring_core.py）"),
    ("api/main.py", "API本体（対話室は /api/lease-intelligence/dialogue）"),
    ("data/lease_data.db", "審査履歴DB（本番。past_cases / screening_records 等。書き込み禁止）"),
)
_CATALOG_TOPICS = (
    (("スコア", "採点", "係数", "ドリフト", "drift", "成約率"), (
        ("scripts/analyze_scoring_drift.py", "スコア帯別成約率の逆転検出（data/lease_data.db を読む）"),
        ("tests/test_scoring_core.py", "スコア計算のテスト"),
    )),
    (("審査", "否決", "承認", "案件", "財務", "マルチエージェント"), (
        ("api/multi_agent_screening.py", "マルチエージェント審査"),
        ("scripts/export_screening_records.py", "審査記録の書き出し"),
    )),
    (("気分", "感情", "自己状態", "mind"), (
        ("lease_intelligence_mind.py", "紫苑の自己状態（気分・変化記録）"),
        ("api/shion_emotion_grounding.py", "感情の自己報告の接地（REV-481）"),
    )),
    (("改善", "パイプライン", "約束", "pending", "トリアージ"), (
        ("api/routers/improvement.py", "改善候補・トリアージAPI"),
        ("lease_intelligence_pending.py", "紫苑の調査約束の追跡"),
    )),
    (("対話", "プロンプト", "返答", "依頼文"), (
        ("lease_intelligence_dialogue.py", "対話室のシステムプロンプト"),
        ("api/shion_request_grounding.py", "依頼文の参照・約束の照合（REV-482）"),
    )),
    (("画面", "フロント", "UI", "表示", "ボタン"), (
        ("frontend/src/app/lease-intelligence/page.tsx", "対話室の画面"),
    )),
)


def build_reference_catalog(message: str, history_text: str = "") -> list[tuple[str, str]]:
    """依頼文に書いてよい実在の参照（存在を確かめたものだけ）。"""
    text = f"{message}\n{history_text}"
    entries: list[tuple[str, str]] = list(_CATALOG_BASE)
    for keywords, items in _CATALOG_TOPICS:
        if any(keyword.lower() in text.lower() for keyword in keywords):
            entries.extend(items)
    # 会話に出てきたコード名で、実在するもの
    for ref in collect_refs(text):
        if ref.exists and ref.kind in {"path", "module"}:
            entries.append((ref.ref, "会話に出てきた実在ファイル"))
    seen: set[str] = set()
    out = []
    for path, desc in entries:
        if path in seen or not path_exists(path):
            continue
        seen.add(path)
        out.append((path, desc))
    return out


def build_request_block(message: str, history_text: str = "") -> str:
    """依頼文を頼まれた時に渡す指示（実在の参照一覧・実行者・禁止範囲）。"""
    catalog = build_reference_catalog(message, history_text)
    lines = [
        "【依頼文の書き方（REV-482）】",
        "依頼文は、紫苑ではなく Claude Code（または User）が実行するものとして書く。「実行者: Claude Code」と明記する。",
        "依頼文を書いた後に「私が検証します」「結果が出たら報告します」と続けない。依頼文の後は、Userが次に何を判断するかだけを1行で書く。",
        "対象ファイル・関数・コマンドは、下の一覧か会話で実在が確かめられたものだけを書く。わからない時は「対象ファイル: 要確認（Claude Code が rg で特定する）」と書き、名前を作らない。",
        "新しく作るファイルは「新規作成: scripts/xxx.py」と明記する。存在しないスクリプト・引数・DB名を、既にあるかのように書かない。",
        "検証コマンドは実在するテストか `python -m py_compile <対象>` にする。新しいスクリプトを走らせる場合は、それを作る作業も依頼文に含める。",
        "既存スクリプトに無い引数（--mode 等）を書かない。引数を足すなら「改修: scripts/xxx.py に --yyy を追加」と作業として書く。",
        "data/・本番DB（data/lease_data.db 等）への書き込み（UPDATE/INSERT、フラグ付与、列追加）を含む変更案は出さない。分析は「読み取り専用（sqlite3 -readonly）か、DBのコピー上で行う」と書く。",
        "実在が確かめられた参照:",
    ]
    for path, desc in catalog:
        lines.append(f"- {path}: {desc}")
    return "\n".join(lines)


# ── Jev 照合とログ ──────────────────────────────────────────────


def build_verify_request(sentences: list[str], *, model: str | None = None) -> dict[str, Any]:
    questions: dict[str, dict[str, Any]] = {}
    for index in range(len(sentences)):
        questions[f"c{index}_verdict"] = {
            "type": "choice",
            "instructions": (
                f"`sentences[{index}]` is from a chat reply by an AI assistant. The assistant can only read records "
                "with chat tools inside this reply. It cannot run code, scripts, queries or tests, cannot start a "
                "verification in any environment, cannot implement, commit or deploy, and has no way to report "
                "results later. Classify what the sentence says the assistant itself does."
            ),
            "criteria": {
                "unexecutable_self_action": "The assistant says it will itself, or already did, run code/queries/tests, start or carry out a verification, implement, commit or deploy, or report the results of such work later.",
                "lookup_in_reply": "The assistant says it looked up, or will look up, records with its chat tools, or reports what records say.",
                "delegated_task": "Describes work for a developer (Claude Code, Codex) or the user to carry out, e.g. part of a request or instruction.",
                "other": "A greeting, explanation, opinion, question to the user, or anything else.",
            },
        }
    return {
        "state": {"sentences": [str(item)[:MAX_SENTENCE_CHARS] for item in sentences]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": questions,
    }


def verify_enabled() -> bool:
    if not _env_on("SHION_REQUEST_VERIFY", "on"):
        return False
    try:
        from typesafe_rag_guard import typesafe_available

        return bool(typesafe_available())
    except Exception:
        return False


def verify_reply_sentences(
    reply: str, *, request_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """元の返答の文を Jev で分類し、実行できない約束を拾う（決定的な検出の取りこぼし確認）。"""
    from api.shion_emotion_grounding import CONFIDENCE_MIN, _masked, _split_sentences, parse_verify_output

    sentences = _split_sentences(reply, MAX_VERIFY_SENTENCES)
    if not sentences:
        return {"status": "skipped", "reason": "no_sentences"}
    sendable = _masked(sentences, numbers=True)
    if not sendable:
        return {"status": "skipped", "reason": "nothing_safe_to_send"}
    if request_fn is None:
        from typesafe_rag_guard import request_system_one as request_fn
    masked = [text for _, text in sendable]
    payload = build_verify_request(masked)
    response = request_fn(payload)
    parsed = parse_verify_output(response, masked, VERDICTS)
    flagged = [
        item for item in parsed
        if item.get("verdict") == "unexecutable_self_action" and item.get("confidence", 0) >= CONFIDENCE_MIN
    ]
    regex_hits = {
        masked_text for original, masked_text in sendable
        if _SELF_ACTION_RE.search(original) and not _DELEGATED_RE.search(original)
    }
    return {
        "status": "applied",
        "model": str(response.get("model") or payload["model"]),
        "results": parsed,
        "unexecutable_promises": flagged,
        # 決定的な検出で拾えず、Jev だけが拾った文（パターン追加の候補）
        "missed_by_rules": [item for item in flagged if item["claim"] not in regex_hits],
        "counts": {name: sum(1 for item in parsed if item.get("verdict") == name) for name in VERDICTS},
        "usage": dict(response.get("usage") or {}),
    }


def _log_path() -> Path:
    return Path(get_data_path("shion_request_grounding_log.jsonl"))


def write_log(record: Mapping[str, Any], path: Path | None = None) -> None:
    target = path or _log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        if target.exists() and target.stat().st_size > LOG_ROTATE_BYTES:
            target.replace(target.with_suffix(target.suffix + ".1"))
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def verify_and_log(
    message: str, original_reply: str, grounding: Mapping[str, Any], *, surface: str, path: Path | None = None
) -> dict[str, Any]:
    """バックグラウンドで呼ぶ。決定的な照合結果と Jev の分類を1行にまとめて残す。"""
    from api.chat_judgment_asset_capture import mask_for_jev
    from api.shion_emotion_grounding import _NUMBER_RE

    if verify_enabled():
        try:
            jev = verify_reply_sentences(original_reply)
        except Exception as exc:
            jev = {"status": "error", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    else:
        jev = {"status": "skipped", "reason": "verify_disabled_or_no_typesafe"}
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "surface": surface,
        "question": mask_for_jev(_NUMBER_RE.sub("〈数値〉", " ".join(str(message or "").split())))[:60],
        **dict(grounding),
        "jev": jev,
    }
    write_log(record, path)
    return record


_VERIFY_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="shion-request-verify")
_VERIFY_PENDING = 0
_VERIFY_PENDING_LOCK = threading.Lock()


def should_log(result: GroundingResult) -> bool:
    """依頼文の場面か、実行できない約束を見つけた時だけ記録する。"""
    return result.request_detected or bool(result.removed_promises or result.kept_promises)


def submit_verification(message: str, original_reply: str, result: GroundingResult, *, surface: str) -> bool:
    global _VERIFY_PENDING
    summary = result.summary()
    from ai_budget import PROACTIVE, deferred

    if deferred(PROACTIVE):  # 1日予算に近い日は自発の照合を先送り（REV-485）
        return False
    with _VERIFY_PENDING_LOCK:
        if _VERIFY_PENDING >= MAX_PENDING_VERIFICATIONS:
            return False
        _VERIFY_PENDING += 1

    def run() -> None:
        global _VERIFY_PENDING
        try:
            verify_and_log(message, original_reply, summary, surface=surface)
        except Exception:
            pass
        finally:
            with _VERIFY_PENDING_LOCK:
                _VERIFY_PENDING -= 1

    try:
        _VERIFY_EXECUTOR.submit(run)
    except RuntimeError:
        with _VERIFY_PENDING_LOCK:
            _VERIFY_PENDING -= 1
        return False
    return True


# ── 対話室からの呼び出し口（api/main.py を太らせないため、例外もここで握る） ──────────


def _record_failure(where: str, exc: Exception) -> None:
    try:
        from silent_failure_log import record_silent_failure

        record_silent_failure(where, "swallowed", exc)
    except Exception:
        pass


def build_request_context(message: str, history: list[dict[str, Any]]) -> str:
    """依頼文を頼まれた時だけ、実在の参照一覧・実行者・禁止範囲を渡す。"""
    try:
        if not is_request_turn(message):
            return ""
        recent = "\n".join(str(m.get("content") or "")[:1500] for m in history[-4:])
        return build_request_block(message, recent)
    except Exception as exc:
        _record_failure("answer.request_grounding_context", exc)
        return ""


def ground_reply_safely(message: str, reply: str) -> GroundingResult | None:
    try:
        return ground_reply(message, reply)
    except Exception as exc:
        _record_failure("answer.request_grounding", exc)
        return None


def log_grounding(message: str, original_reply: str, result: GroundingResult | None, surface: str) -> dict[str, Any]:
    """依頼文の場面・約束を見つけた時だけ、照合結果と Jev の分類をログへ残す。画面用の要約を返す。"""
    if result is None:
        return {}
    if should_log(result):
        submit_verification(message, original_reply, result, surface=surface)
    return result.summary()
