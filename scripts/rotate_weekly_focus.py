"""
ledger.jsonl で 21 日以上 needs_review のままのエントリを parked に移行する。
parked_until フィールド（28日後）を追加し、期限後に自動復活させる。

既存パイプラインには影響を与えず、ledger.jsonl への追記のみを行う。
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"
_STALE_DAYS = 21
_PARKED_DAYS = 28
_TODAY = date.today()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[: len(fmt)], fmt).date()
        except ValueError:
            continue
    return None


def main() -> None:
    if not _LEDGER_PATH.exists():
        print(f"ledger.jsonl が見つかりません: {_LEDGER_PATH}", file=sys.stderr)
        return

    # タイトル → 最新ステータス・初回登録日を追跡
    latest_status: dict[str, str] = {}
    latest_entry: dict[str, dict] = {}
    first_seen: dict[str, date] = {}
    latest_parked_until: dict[str, date | None] = {}

    for raw in _LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        title = entry.get("title", "")
        if not title:
            continue

        recorded = _parse_date(entry.get("recorded_at"))

        if title not in first_seen or (recorded and recorded < first_seen[title]):
            first_seen[title] = recorded or _TODAY

        latest_status[title] = entry.get("status", "")
        latest_entry[title] = entry
        latest_parked_until[title] = _parse_date(entry.get("parked_until"))

    parked_until_date = _TODAY + timedelta(days=_PARKED_DAYS)
    to_park: list[dict] = []
    to_revive: list[dict] = []

    for title, status in latest_status.items():
        entry = latest_entry[title]

        if status == "needs_review":
            # parked_until が設定されている場合は既に処理済みの可能性があるのでスキップ
            if latest_parked_until.get(title):
                continue
            first = first_seen.get(title, _TODAY)
            stale_days = (_TODAY - first).days
            if stale_days >= _STALE_DAYS:
                to_park.append(
                    {
                        **entry,
                        "status": "parked",
                        "parked_until": parked_until_date.isoformat(),
                        "parked_reason": f"needs_review 滞留 {stale_days}日（自動 parking）",
                        "recorded_at": datetime.now().isoformat(),
                    }
                )

        elif status == "parked":
            pu = latest_parked_until.get(title)
            if pu and pu <= _TODAY:
                revived = {k: v for k, v in entry.items() if k != "parked_until" and k != "parked_reason"}
                revived["status"] = "needs_review"
                revived["recorded_at"] = datetime.now().isoformat()
                to_revive.append(revived)

    if not to_park and not to_revive:
        print(f"rotate_weekly_focus: 対象なし（park 0件、復活 0件）")
        return

    with _LEDGER_PATH.open("a", encoding="utf-8") as f:
        for e in to_park:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        for e in to_revive:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(
        f"rotate_weekly_focus: {len(to_park)}件を parked に移行"
        f"（{_STALE_DAYS}日以上滞留、復活予定: {parked_until_date}）、"
        f"{len(to_revive)}件を needs_review に復活"
    )


if __name__ == "__main__":
    main()
