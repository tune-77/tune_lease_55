#!/usr/bin/env python3
"""REV-591: 2026-10-09 の検証の会話に「検証・本人の会話ではない」印を付け、想起・内省の材料から外す。

Claude Code の検証（同じ質問「運送業や建設業の倒産が…」の繰り返し、「こんにちは、今日は寒いね」
「ありがとう、助かったよ」「お疲れさま、少し休憩しよう」）が本人の会話として記録され、
10/10 04:07 の Private Reflection が「ユーザーは思考停止」「同じ質問が3回続いたら心理状態を確認する」
などの内省・行動変更・判断資産候補を作った。

データは削除しない。各ファイルをバックアップしてから、
- 会話ログ・予想ログ・経験ログ・相手の様子の行に origin=verification を足す
- 会話履歴（chat_messages）の user_id を <id>:verification に分ける（画面・次回履歴・週1イラストから外れる）
- 対話ノートの該当見出しに〔検証・本人の会話ではない〕を付ける
- mind.json の該当する記憶・会話の要点・内省の持ち越しは、同じファイル内の verification_excluded へ移す
- 10/10 の内省ノートに訂正の節を足し、紫苑への訂正を記録として残す
- 内省アクション候補（判断資産候補）は検証由来・要確認のまま保留にする（承認しない）

検証時間帯（本人の発言は無いことを会話ログで確認済み）:
- UTC 2026-10-08 21:29:30〜21:31:00（verify_rev532）
- UTC 2026-10-08 22:20:00〜2026-10-09 00:00:00（JST 10/9 07:20〜09:00）
同じ秒に10件入った 22:07:01 の5往復は音声セッションの文字起こし（本人）なので対象外。

使い方: python scripts/mark_verification_conversations_20261009.py           # 確認だけ
        python scripts/mark_verification_conversations_20261009.py --apply   # バックアップして印を付ける
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_paths import get_data_dir, get_db_path, resolve_obsidian_vault  # noqa: E402
from shion_verification_origin import (  # noqa: E402
    VERIFICATION_CORRECTION_HEADING,
    VERIFICATION_HISTORY_SUFFIX,
    VERIFICATION_NOTE_MARK,
)

JST = timezone(timedelta(hours=9))
WINDOWS_UTC = (
    (datetime(2026, 10, 8, 21, 29, 30, tzinfo=timezone.utc), datetime(2026, 10, 8, 21, 31, 0, tzinfo=timezone.utc)),
    (datetime(2026, 10, 8, 22, 20, 0, tzinfo=timezone.utc), datetime(2026, 10, 9, 0, 0, 0, tzinfo=timezone.utc)),
)
MARK = {
    "origin": "verification",
    "origin_marked_by": "REV-591",
    "origin_note": "Claude Code の検証テストの会話。ユーザー本人の発言・様子ではない（2026-10-10 印付け）",
}
CORRECTION = (
    "2026-10-09〜10 の同じ質問（運送業や建設業の倒産が増えてるけど、リース審査で気をつけることは？）の繰り返しと、"
    "「今日ちょっと疲れた」「お疲れさま、少し休憩しよう」「ありがとう、助かったよ」等は Claude Code の検証テストで、"
    "ユーザーの実際の発言ではない。ユーザーが同じ質問を繰り返したのでも、思考停止していたのでもない。"
    "ただしユーザー本人も、実際にこの頃は疲れていたと話している（2026-10-10、ユーザー本人より）。"
)
KEYPOINT_CORRECTION = (
    "10/9〜10の同じ質問の繰り返しと「疲れた」「休憩しよう」等はClaude Codeの検証テストで本人の発言ではない。"
    "ただし本人もこの頃は実際に疲れていたと話している"
)
REFLECTION_DATE = "2026-10-10"
VERIFICATION_MESSAGES = (
    "運送業や建設業の倒産が増えてるけど",
    "こんにちは、今日は寒いね",
    "ありがとう、助かったよ",
    "お疲れさま、少し休憩しよう",
)
# 検証の答えから mind.json に入った会話の要点（10/9・運送/建設の質問への紫苑の答え由来）
KEYPOINT_TOPICS = ("運送", "建設", "価格転嫁", "補助金", "人手に依存", "燃料", "省力化", "転嫁不可", "現状維持の投資")
REFLECTION_LESSON_WORDS = ("同じ質問", "繰り返", "心理", "思考の停止", "論理の追撃")


def _parse(value: Any, *, naive_tz: timezone) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00").replace(" ", "T")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=naive_tz)


def in_window(value: Any, *, naive_tz: timezone = timezone.utc) -> bool:
    ts = _parse(value, naive_tz=naive_tz)
    return bool(ts) and any(start <= ts < end for start, end in WINDOWS_UTC)


class Marker:
    def __init__(self, *, apply: bool, backup_dir: Path) -> None:
        self.apply = apply
        self.backup_dir = backup_dir
        self.inventory: list[tuple[str, str]] = []

    def note(self, store: str, item: str) -> None:
        self.inventory.append((store, item))

    def backup(self, path: Path) -> None:
        if not self.apply or not path.exists():
            return
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        target = self.backup_dir / (path.name if path.parent.name != "Lease Intelligence" else f"vault_{path.name}")
        if target.exists():
            target = self.backup_dir / f"{path.parent.name.replace(' ', '_')}_{path.name}"
        if not target.exists():
            shutil.copy2(path, target)

    def rewrite_text(self, path: Path, transform: Callable[[str], str]) -> bool:
        """書き換え中に追記が入ったらやり直す（本番サーバーが同じファイルへ追記するため）。"""
        if not path.exists():
            return False
        for _ in range(5):
            before = path.read_text(encoding="utf-8")
            after = transform(before)
            if after == before:
                return False
            if not self.apply:
                return True
            self.backup(path)
            if path.read_text(encoding="utf-8") != before:
                continue
            tmp = path.with_name(path.name + ".rev591.tmp")
            tmp.write_text(after, encoding="utf-8")
            os.replace(tmp, path)
            return True
        raise RuntimeError(f"{path} が書き換え中に変わり続けたので止めました")

    # ── data/*.jsonl ───────────────────────────────────────────────
    def mark_jsonl(self, path: Path, label: str, ts_key: str, *, naive_tz: timezone, describe: Callable[[dict], str]) -> None:
        def transform(text: str) -> str:
            lines = text.splitlines(keepends=True)
            out = []
            for line in lines:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if isinstance(row, dict) and in_window(row.get(ts_key), naive_tz=naive_tz) and row.get("origin") != "verification":
                    row.update(MARK)
                    out.append(json.dumps(row, ensure_ascii=False, sort_keys="event_id" in row) + "\n")
                    self.note(label, describe(row))
                else:
                    out.append(line)
            return "".join(out)

        self.rewrite_text(path, transform)

    # ── data/user_affect_state.json ────────────────────────────────
    def mark_affect(self, path: Path) -> None:
        def transform(text: str) -> str:
            data = json.loads(text)
            for uid, user in dict(data.get("users") or {}).items():
                for obs in list(user.get("observations") or []):
                    if in_window(obs.get("at"), naive_tz=JST) and obs.get("origin") != "verification":
                        obs.update(MARK)
                        self.note("相手の様子（#1283 user_affect_state.json）", f"{uid} {obs.get('at')} {obs.get('label')}")
            return json.dumps(data, ensure_ascii=False, indent=2) + "\n"

        self.rewrite_text(path, transform)

    # ── chat_messages（会話履歴） ──────────────────────────────────
    def mark_chat_messages(self, db_path: Path) -> None:
        if not db_path.exists():
            return
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT id, user_id, role, created_at, content FROM chat_messages ORDER BY id").fetchall()
            hits = [
                r for r in rows
                if in_window(r[3]) and not str(r[1]).endswith(VERIFICATION_HISTORY_SUFFIX)
            ]
            for row_id, uid, role, created, content in hits:
                if role == "user":
                    self.note("会話履歴（chat_messages）", f"#{row_id} {uid} {created} {str(content)[:30]}")
            if self.apply and hits:
                self.backup(db_path)
                with conn:
                    conn.executemany(
                        "UPDATE chat_messages SET user_id = ? WHERE id = ?",
                        [(f"{uid}{VERIFICATION_HISTORY_SUFFIX}", row_id) for row_id, uid, *_ in hits],
                    )
        finally:
            conn.close()

    # ── Vault: 対話ノート ───────────────────────────────────────────
    def mark_dialogue_note(self, path: Path) -> None:
        def transform(text: str) -> str:
            out = []
            for line in text.splitlines(keepends=True):
                stripped = line.rstrip("\n")
                if stripped.startswith("## ") and VERIFICATION_NOTE_MARK not in stripped:
                    clock = stripped[3:].strip()[:8]
                    if in_window(f"2026-10-09T{clock}", naive_tz=JST):
                        self.note("対話ノート（Dialogue/2026-10-09.md）", f"{clock} の往復")
                        line = f"{stripped} {VERIFICATION_NOTE_MARK}\n"
                out.append(line)
            return "".join(out)

        self.rewrite_text(path, transform)

    # ── Vault: 日次の記憶ノート（追記式）に残った内省の学びの行へ印 ──────────
    def mark_memory_note_lines(self, path: Path) -> None:
        def transform(text: str) -> str:
            out = []
            for line in text.splitlines(keepends=True):
                stripped = line.rstrip("\n")
                if (
                    stripped.startswith("- Private Reflectionからの学び")
                    and any(word in stripped for word in REFLECTION_LESSON_WORDS)
                    and VERIFICATION_NOTE_MARK not in stripped
                ):
                    self.note("Memory/2026-10-10.md（会話サマリーの行）", stripped[2:50])
                    line = f"{stripped} {VERIFICATION_NOTE_MARK}\n"
                out.append(line)
            return "".join(out)

        self.rewrite_text(path, transform)

    # ── Vault: 内省ノートへ訂正の節 ─────────────────────────────────
    def add_correction_section(self, path: Path, label: str) -> None:
        section = f"{VERIFICATION_CORRECTION_HEADING}\n\n{CORRECTION}\n\n"

        def transform(text: str) -> str:
            if VERIFICATION_CORRECTION_HEADING in text:
                return text
            if text.startswith("---\n") and "\nmaterial_origin:" not in text.split("\n---", 1)[0]:
                text = text.replace("\n---\n", "\nmaterial_origin: verification\nmaterial_origin_marked_by: REV-591\n---\n", 1)
            lines = text.splitlines(keepends=True)
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    lines.insert(i + 1, "\n" + section.rstrip("\n") + "\n")
                    break
            else:
                lines.insert(0, section)
            self.note(label, "冒頭に訂正の節を追加（本文はそのまま）")
            return "".join(lines)

        self.rewrite_text(path, transform)

    # ── Vault: mind.json ────────────────────────────────────────────
    def mark_mind(self, vault: Path) -> None:
        from lease_intelligence_mind import MIND_FILE_NAME, _mind_locked, _write_state, load_lease_intelligence_mind, mind_directory

        path = mind_directory(vault) / MIND_FILE_NAME
        with _mind_locked(vault):
            state = load_lease_intelligence_mind(vault)
            excluded = list(state.get("verification_excluded") or [])
            moved = 0

            def quarantine(field: str, entry: Any) -> None:
                nonlocal moved
                excluded.append({**MARK, "from": field, "entry": entry})
                moved += 1

            kept = []
            for item in state.get("memories") or []:
                summary = str(item.get("summary") or "")
                if item.get("date") == "2026-10-09" and any(f"「{m}" in summary for m in VERIFICATION_MESSAGES):
                    quarantine("memories", item)
                    self.note("mind.json 記憶（memories）", summary[:50])
                else:
                    kept.append(item)
            state["memories"] = kept

            kept = []
            for item in state.get("conversation_keypoints") or []:
                content = str(item.get("content") or "")
                from_verification_answer = (
                    item.get("date") == "2026-10-09"
                    and item.get("session_id") == "lease-intelligence-dialogue"
                    and any(word in content for word in KEYPOINT_TOPICS)
                )
                from_reflection = (
                    item.get("date") == REFLECTION_DATE
                    and content.startswith("Private Reflectionからの学び")
                    and any(word in content for word in REFLECTION_LESSON_WORDS)
                )
                if from_verification_answer or from_reflection:
                    quarantine("conversation_keypoints", item)
                    self.note("mind.json 会話の要点（conversation_keypoints）", content[:50])
                else:
                    kept.append(item)
            state["conversation_keypoints"] = kept

            for entry in state.get("mood_change_log") or []:
                if in_window(entry.get("ts"), naive_tz=JST) and entry.get("origin") != "verification":
                    entry.update(MARK)
                    moved += 1
                    self.note("mind.json 気分の変化記録（mood_change_log）", f"{entry.get('ts')} {str(entry.get('trigger'))[:20]}")

            reflection = dict(state.get("private_reflection") or {})
            if reflection.get("last_reflected_date") == REFLECTION_DATE and not reflection.get("verification_correction"):
                carried = {k: reflection.get(k) for k in ("text", "last_feedback", "last_reusable_lessons", "next_context")}
                quarantine("private_reflection", carried)
                reflection.update(
                    {
                        "text": CORRECTION,
                        "last_reusable_lessons": [],
                        "next_context": "",
                        "verification_correction": CORRECTION,
                        "last_feedback": {**dict(reflection.get("last_feedback") or {}), "reusable_lessons": [], "next_context": "", **MARK},
                    }
                )
                state["private_reflection"] = reflection
                self.note("mind.json 内省の持ち越し（private_reflection）", "10/10 内省の学び・次回への持ち越しを外し、訂正に置き換え")
            question = str(state.get("current_question") or "")
            if any(word in question for word in REFLECTION_LESSON_WORDS):
                quarantine("current_question", question)
                state["current_question"] = ""
                self.note("mind.json 持ち越す問い（current_question）", question[:50])

            state["verification_excluded"] = excluded
            if self.apply and moved:
                self.backup(path)
                _write_state(vault, state)

    # ── 判断資産候補（内省アクション候補） ─────────────────────────
    def hold_reflection_candidates(self, path: Path) -> None:
        def transform(text: str) -> str:
            out = []
            for line in text.splitlines(keepends=True):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if str(row.get("target_date")) == REFLECTION_DATE and row.get("origin_verification") is not True:
                    row.update(
                        {
                            "origin_verification": True,
                            "origin_marked_by": "REV-591",
                            "review_note": "検証の会話を本人の会話と読んだ 10/10 の内省から作られた候補。要確認のまま保留（承認しない）",
                        }
                    )
                    self.note("判断資産候補（reflection_action_candidates.jsonl・保留）", str(row.get("proposed_change"))[:60])
                    out.append(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    out.append(line)
            return "".join(out)

        self.rewrite_text(path, transform)

    def record_correction_keypoint(self, vault: Path) -> None:
        from lease_intelligence_mind import load_lease_intelligence_mind, save_conversation_keypoints

        existing = [str(i.get("content") or "") for i in load_lease_intelligence_mind(vault).get("conversation_keypoints") or []]
        if KEYPOINT_CORRECTION in existing:
            return
        self.note("紫苑への訂正（mind.json 会話の要点）", KEYPOINT_CORRECTION)
        if self.apply:
            save_conversation_keypoints(
                vault,
                session_id="user_correction_rev591",
                keypoints=[KEYPOINT_CORRECTION],
                date_str=REFLECTION_DATE,
                content_source="user",
            )


def run(*, apply: bool) -> Marker:
    data = get_data_dir()
    vault = resolve_obsidian_vault()
    stamp = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    marker = Marker(apply=apply, backup_dir=data / "backups" / f"rev591_verification_mark_{stamp}")
    marker.mark_jsonl(
        data / "cloudrun_chat_log.jsonl", "会話ログ（cloudrun_chat_log.jsonl）", "ts", naive_tz=timezone.utc,
        describe=lambda r: f"{r.get('ts', '')[:19]}Z {r.get('surface')} {r.get('user_id')} {str(r.get('user_message'))[:30]}",
    )
    marker.mark_jsonl(
        data / "shion_experience_events.jsonl", "経験ログ（shion_experience_events.jsonl）", "ts", naive_tz=timezone.utc,
        describe=lambda r: f"{r.get('ts', '')[:19]}Z {r.get('category')} {str(r.get('message_preview'))[:30]}",
    )
    marker.mark_jsonl(
        data / "shion_prediction_log.jsonl", "予想と答え合わせ（#1291 shion_prediction_log.jsonl）", "at", naive_tz=JST,
        describe=lambda r: f"{r.get('at')} {r.get('user_id')} 予想={r.get('expected_affect')} 実際={r.get('actual_affect')}",
    )
    marker.mark_affect(data / "user_affect_state.json")
    marker.mark_chat_messages(Path(get_db_path()))
    marker.hold_reflection_candidates(data / "reflection_action_candidates.jsonl")
    if vault is not None:
        li = vault / "Projects" / "tune_lease_55" / "Lease Intelligence"
        # mind.json の書き込みは Memory/<今日>.md を作り直す（外した要点は消え、訂正の要点が載る）
        marker.mark_mind(vault)
        marker.record_correction_keypoint(vault)
        marker.mark_dialogue_note(li / "Dialogue" / "2026-10-09.md")
        marker.add_correction_section(li / "Private Reflection" / f"{REFLECTION_DATE}.md", "Private Reflection/2026-10-10.md")
        marker.add_correction_section(li / "Reflection" / f"{REFLECTION_DATE}_reflection.md", "Reflection/2026-10-10_reflection.md")
        marker.mark_memory_note_lines(li / "Memory" / f"{REFLECTION_DATE}.md")
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="バックアップしてから印を付ける（無ければ確認だけ）")
    args = parser.parse_args(argv)
    marker = run(apply=args.apply)
    current = ""
    for store, item in marker.inventory:
        if store != current:
            print(f"\n## {store}")
            current = store
        print(f"- {item}")
    print(f"\n{'印を付けました' if args.apply else '確認のみ（--apply で実行）'}: {len(marker.inventory)} 件")
    if args.apply:
        print(f"バックアップ: {marker.backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
