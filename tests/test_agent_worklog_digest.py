from __future__ import annotations

import json
import datetime as dt


def test_agent_worklog_digest_extracts_public_summary_sections(tmp_path):
    import scripts.build_agent_worklog_digest as digest

    vault = tmp_path / "Vault"
    daily = vault / "Daily"
    daily.mkdir(parents=True)
    note_date = dt.date.today().isoformat()
    note = daily / f"{note_date}.md"
    note.write_text(
        """
# {note_date}

## 10:15 Codex Work Log

### Summary
- FAQ導線をサポートハブへ寄せた

### Chat Summary
- Userは低利用を不要ではなく文脈不足として見たいと判断した

### Decisions
- FAQは削除せず、審査画面と紫苑対話室から導線を追加する

### Changes
- frontend/src/app/help/page.tsx

### Verification
- npm run typecheck

### Git
- commit: abc

## 11:30 Claude Work Log

### Summary
- 自己提案の判定軸をレビュー

### Private
- これは出してはいけない
""".format(note_date=note_date).strip(),
        encoding="utf-8",
    )

    parsed = digest.parse_work_logs(note)
    result = digest.build_digest(vault, days=1, limit=5)

    assert len(parsed) == 2
    assert result["count"] == 2
    first = result["items"][0]
    assert first["agent"] == "Claude"
    assert first["summary"] == ["自己提案の判定軸をレビュー"]
    second = result["items"][1]
    assert second["agent"] == "Codex"
    assert "文脈不足" in second["chat_summary"][0]
    assert "これは出してはいけない" not in json.dumps(result, ensure_ascii=False)
    assert result["policy"]["raw_chat_logs_excluded"] is True
