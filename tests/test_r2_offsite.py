from __future__ import annotations

import datetime as dt
import json
import secrets
from pathlib import Path

import pytest

from scripts import backup_case_data as backup
from scripts import r2_offsite as r2

CREDS = r2.R2Credentials("acct", "id", "secret")


class FakeR2:
    """S3 互換 API の PUT/GET(list)/DELETE だけを真似る。最初はバケットが無い。"""

    def __init__(self):
        self.bucket_exists = False
        self.objects: dict[str, bytes] = {}

    def __call__(self, creds, method, path, query=None, body=b""):
        parts = path.lstrip("/").split("/", 1)
        if method == "PUT" and len(parts) == 1:
            self.bucket_exists = True
            return b""
        if not self.bucket_exists:
            raise r2.R2Error("R2 PUT HTTP 404 NoSuchBucket")
        if method == "PUT":
            self.objects[parts[1]] = body
        elif method == "DELETE":
            self.objects.pop(parts[1])
        elif method == "GET":
            items = "".join(
                f"<Contents><Key>{k}</Key><Size>{len(v)}</Size></Contents>"
                for k, v in sorted(self.objects.items()) if k.startswith(query["prefix"])
            )
            return f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{items}</ListBucketResult>'.encode()
        return b""


def _archive(tmp_path: Path, name: str, blob: bytes | None = None) -> Path:
    path = tmp_path / name
    path.write_bytes(blob if blob is not None else backup.MAGIC + secrets.token_bytes(32))
    return path


def test_sync_creates_bucket_keeps_n_per_profile_and_totals(tmp_path):
    fake = FakeR2()
    other = _archive(tmp_path, "judgment_daily_20261001_233000.tar.gz.enc")
    r2.sync_archive(other, "judgment_daily", keep=30, creds=CREDS, request=fake)
    names = ["case_data_20260920_013000", "case_data_20260927_013000", "case_data_20261004_013000", "case_data_20261004_013000_1"]
    for n in names:
        result = r2.sync_archive(_archive(tmp_path, n + ".tar.gz.enc"), "case_data", keep=2, creds=CREDS, request=fake)

    assert fake.bucket_exists
    assert sorted(fake.objects) == [
        "case_data/case_data_20261004_013000.tar.gz.enc",
        "case_data/case_data_20261004_013000_1.tar.gz.enc",
        "judgment_daily/judgment_daily_20261001_233000.tar.gz.enc",  # 他 profile の世代は消さない
    ]
    assert result["removed"] == ["case_data/case_data_20260927_013000.tar.gz.enc"]
    assert result["bucket_total_bytes"] == sum(len(v) for v in fake.objects.values())


@pytest.mark.parametrize("name,blob", [("x.tar.gz.enc", b"plain tar"), ("x.tar.gz", backup.MAGIC + b"z")])
def test_sync_refuses_anything_but_encrypted_archive(tmp_path, name, blob):
    fake = FakeR2()
    with pytest.raises(r2.R2Error, match="暗号化アーカイブではない"):
        r2.sync_archive(_archive(tmp_path, name, blob), "case_data", keep=2, creds=CREDS, request=fake)
    assert not fake.objects


def test_r2_status_is_separate_and_report_warns(tmp_path):
    status = tmp_path / "status.json"
    now = dt.datetime(2026, 10, 5, 7, 0).astimezone()
    backup.record_status("case_data", ok=True, summary=backup.BackupSummary("", "d", [], [], []), path=status)
    backup.record_r2_status("case_data", ok=False, error="R2Error: R2 PUT HTTP 403", path=status)
    data = json.loads(status.read_text())
    assert data["case_data"]["ok"] is True  # R2 失敗で iCloud 側を失敗にしない
    line = backup.r2_report_line(status_path=status, now=now)
    assert line.startswith("- ⚠️ R2 オフサイト") and "直近失敗 case_data" in line and "成功なし: judgment_daily" in line

    for profile in backup.PROFILES:
        backup.record_r2_status(profile, ok=True, path=status, result={
            "bucket": "b", "key": "k", "bytes": 1, "bucket_total_bytes": 420 * 10**6})
    assert backup.r2_report_line(status_path=status).startswith("- R2 オフサイト最終成功")
    assert "0.42GB/10GB" in backup.r2_report_line(status_path=status)

    data = json.loads(status.read_text())
    data["r2"]["bucket_total_bytes"] = 9 * 10**9
    status.write_text(json.dumps(data))
    assert "無料枠に接近" in backup.r2_report_line(status_path=status)
