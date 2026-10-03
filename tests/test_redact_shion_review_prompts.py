"""保存済みの紫苑レビュー依頼文の伏字化（scripts/redact_shion_review_prompts.py）の検査。"""
from __future__ import annotations

import json
import secrets
import sqlite3
import tarfile
import io

from scripts import redact_shion_review_prompts as red
from scripts.backup_case_data import decrypt_bytes

PROMPT = (
    "【審査分析画面からの紫苑レビュー依頼】\nこの案件を、審査担当者の横にいる紫苑としてレビューしてください。\n前提:\n"
    "・企業名: サセ\n・業種: 24 生産用機械器具製造業\n・取得価額: 5百万円\n・営業メモ: 山田社長と銀行担当者の関係で多少高くても受注\n"
)
REPLY = "・違和感: サセは銀行紹介案件で、格付×利益率の矛盾に注目します。\n・稟議に残す一文: 返済原資の具体性を確認する。"


def _assert_redacted(text: str) -> None:
    assert "サセ" not in text and "山田" not in text and "多少高くても" not in text and "5百万円" not in text
    assert "格付×利益率の矛盾" in text and "返済原資の具体性を確認する" in text  # 学び（論点・判断の要点）は残す


def test_markdown_note_masks_company_memo_amount_and_keeps_learning():
    text = red.mask_markdown(PROMPT + "\n" + REPLY, red.company_names(PROMPT))
    _assert_redacted(text)
    assert "〈企業〉は銀行紹介案件" in text and "営業メモ:〈伏字〉" in text


def test_frontmatter_json_string_stays_valid_yaml_string():
    note = "---\nquestion: " + json.dumps(PROMPT, ensure_ascii=False) + "\nsource: vertex_answer_api\n---\n## 要約\n" + REPLY
    text = red.mask_markdown(note, red.company_names(note))
    question_line = next(line for line in text.splitlines() if line.startswith("question: "))
    value = json.loads(question_line[len("question: "):])
    _assert_redacted(value + text)
    assert value.startswith("【審査分析画面からの紫苑レビュー依頼】")


def test_jsonl_masks_only_records_with_the_prompt_and_keeps_json_valid():
    other = {"user_message": "サセという言葉だけの別の会話", "assistant_reply": "ok"}
    review = {"user_id": "screening-shion-review:サセ", "user_message": PROMPT, "assistant_reply": REPLY}
    text = json.dumps(other, ensure_ascii=False) + "\n" + json.dumps(review, ensure_ascii=False) + "\n"
    lines = red.mask_jsonl(text).splitlines()
    assert json.loads(lines[0]) == other  # 依頼文を含まない行は触らない
    masked = json.loads(lines[1])
    _assert_redacted(json.dumps(masked, ensure_ascii=False))
    assert masked["user_id"] == "screening-shion-review:〈企業〉"


def test_sqlite_rows_of_the_review_conversation_are_masked(tmp_path):
    db = tmp_path / "lease_data.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table chat_messages (id integer primary key, user_id text, role text, content text)")
        conn.executemany("insert into chat_messages (user_id, role, content) values (?, ?, ?)", [
            ("screening-shion-review:サセ", "user", PROMPT),
            ("screening-shion-review:サセ", "assistant", REPLY),
            ("default", "user", "普通の会話"),
        ])
    change = red.find_sqlite_change(db)
    assert change and len(change.rows) == 2
    red.write_redacted(change)
    with sqlite3.connect(db) as conn:
        rows = conn.execute("select user_id, content from chat_messages order by id").fetchall()
    _assert_redacted(" ".join(u + c for u, c in rows[:2]))
    assert rows[2] == ("default", "普通の会話")


def test_apply_archives_encrypted_originals_then_rewrites(tmp_path, monkeypatch):
    key = secrets.token_bytes(32)
    monkeypatch.setattr(red, "load_key", lambda: key)
    monkeypatch.setattr(red, "BACKUP_DIR", tmp_path / "backup")
    vault, repo = tmp_path / "vault", tmp_path / "repo"
    (vault / "Research").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    note = vault / "Research" / "note.md"
    note.write_text(PROMPT + REPLY, encoding="utf-8")
    moc = vault / "MOC.md"
    moc.write_text("- [[【審査分析画面からの紫苑レビュー依頼】この案件を]]", encoding="utf-8")
    log = repo / "data" / "chat_log.jsonl"
    log.write_text(json.dumps({"user_message": PROMPT, "reply": REPLY}, ensure_ascii=False) + "\n", encoding="utf-8")

    dry = red.run(False, vault, repo)
    assert dry["vault_files"] == 1 and dry["data_files"] == 1 and not dry["applied"]
    assert dry["marker_only_unchanged"] == ["vault:MOC.md"]
    assert "サセ" in note.read_text(encoding="utf-8")  # ドライランは書き換えない

    result = red.run(True, vault, repo)
    _assert_redacted(note.read_text(encoding="utf-8") + log.read_text(encoding="utf-8"))
    archives = list((tmp_path / "backup").glob("*.tar.gz.enc"))
    assert len(archives) == 1 and result["archive"] == str(archives[0])
    blob = archives[0].read_bytes()
    assert "サセ".encode() not in blob  # 退避は暗号文だけ
    with tarfile.open(fileobj=io.BytesIO(decrypt_bytes(key, blob)), mode="r:gz") as tar:
        assert tar.extractfile("vault/Research/note.md").read().decode() == PROMPT + REPLY
    assert not list(tmp_path.rglob("*.redact.tmp"))


def test_short_kana_names_do_not_break_ordinary_words():
    text = red.mask_text("サセは分かりにくい。サセボ市のくいの話。", {"サセ", "くい"})
    assert text == "〈企業〉は分かりにくい。サセボ市のくいの話。"  # 「サセボ」「にくい」「くい」（ひらがな2文字）は触らない


def test_truncated_amount_preview_is_masked():
    assert red.mask_text("・取得価額: 55百…", set()) == "・取得価額: 〈金額〉"


def test_company_name_with_spaces_is_captured_to_end_of_field():
    prompt = PROMPT.replace("サセ", "株式会社 山田製作所")
    names = red.company_names(prompt)
    assert names == {"株式会社 山田製作所"}
    masked = red.mask_text(prompt + "回答: 株式会社 山田製作所を確認。", names)
    assert "山田製作所" not in masked


def test_sqlite_write_rescans_rows_added_after_initial_scan(tmp_path):
    db = tmp_path / "lease_data.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table chat_messages (id integer primary key, user_id text, role text, content text)")
        conn.execute("insert into chat_messages (user_id, role, content) values (?, ?, ?)", ("screening:サセ", "user", PROMPT))
    change = red.find_sqlite_change(db)
    assert change is not None
    with sqlite3.connect(db) as conn:
        conn.execute("insert into chat_messages (user_id, role, content) values (?, ?, ?)", ("screening:株式会社 山田製作所", "user", PROMPT.replace("サセ", "株式会社 山田製作所")))
    assert red.write_redacted(change) == 2
    with sqlite3.connect(db) as conn:
        text = " ".join(" ".join(row) for row in conn.execute("select user_id, content from chat_messages"))
    assert "サセ" not in text and "山田製作所" not in text
