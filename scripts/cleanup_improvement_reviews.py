#!/usr/bin/env python3
"""
改善パイプライン要確認リストのクリーンアップスクリプト。

GitHub PR 履歴から REV ステータスを判定し、台帳 (ledger.jsonl) を更新する。
reports/*.json は .gitignore 対象のため、台帳への直接書き込みで永続化する。

ステータス判定ロジック:
  - MERGED PR に REV-NNN が含まれる → applied
  - CLOSED (未マージ) PR のみ     → rejected
  - PR なし                       → deferred (--mark-deferred フラグ時のみ記録)

使い方:
  python3 scripts/cleanup_improvement_reviews.py           # dry-run
  python3 scripts/cleanup_improvement_reviews.py --apply   # 台帳に書き込み
  python3 scripts/cleanup_improvement_reviews.py --apply --mark-deferred  # deferred も記録

台帳パス: ~/Library/Logs/tunelease/ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent / ".agents" / "skills" / "auto-improvement-pipeline"
sys.path.insert(0, str(_PIPELINE_DIR))
sys.path.insert(0, str(_PIPELINE_DIR / "scripts"))

try:
    import pipeline_ledger as ledger  # type: ignore
except ImportError:
    ledger = None  # type: ignore

try:
    from improvement_identity import canonical_key as _canonical_key  # type: ignore
except ImportError:
    def _canonical_key(title: str, description: str = "") -> str:  # type: ignore[misc]
        return f"rev_{title.strip().lower()}"

LEDGER_PATH = Path(
    os.environ.get("LEDGER_PATH",
                   str(Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"))
)

# 過去の台帳不整合を個別に補正するための手打ちテーブル群
# （KNOWN_CODE_APPLIED / KNOWN_TITLE_APPLIED / KNOWN_PR_OVERRIDES /
#  KNOWN_APPLIED_NO_PR / REV_TITLES）は cleanup_improvement_reviews_data.json に
# 外部化している。各テーブルの役割は scripts/README_ledger.md を参照。
_DATA_PATH = Path(__file__).resolve().parent / "cleanup_improvement_reviews_data.json"
with _DATA_PATH.open(encoding="utf-8") as _f:
    _OVERRIDE_DATA = json.load(_f)

# PR タイトルに REV-NNN が明記されていないが確認済みのもの
# (PR 番号, status)
# PR を経ずにコミット直接で実装済みが確認されたチャット改善メモ
# (canonical_key, title, commit_or_reason)
# → --apply 実行時に台帳へ applied として追記する
KNOWN_CODE_APPLIED: list[tuple[str, str, str]] = [
    tuple(entry) for entry in _OVERRIDE_DATA["known_code_applied"]
]

# B群: 出荷済みだが、生メモの canonical_key（title+description 依存）が出荷 REV 番号と
# 繋がらないため、runtime 台帳で needs_review のまま churn し続ける項目。
# canonical_key は description 差異でブレるため固定キーで照合できない。代わりに
# runtime 台帳の未決着エントリを「正規化タイトル一致」で解決し、その実キーのまま
# applied 化する（_apply_title_matched_closures で処理）。
# 値は (タイトル, 出荷エビデンス)。タイトルは recursive_self_improvement レポートの表記に合わせる。
KNOWN_TITLE_APPLIED: list[tuple[str, str]] = [
    tuple(entry) for entry in _OVERRIDE_DATA["known_title_applied"]
]

KNOWN_PR_OVERRIDES: dict[str, tuple[int, str]] = {
    k: tuple(v) for k, v in _OVERRIDE_DATA["known_pr_overrides"].items()
}

# PR を経由せずコードレビューで実装確認済みの REV
# (commit_hash, 実装ファイル, 説明)
KNOWN_APPLIED_NO_PR: dict[str, tuple[str, str, str]] = {
    k: tuple(v) for k, v in _OVERRIDE_DATA["known_applied_no_pr"].items()
}

# REV ID → タイトル（reports から取得したマスタ）
REV_TITLES: dict[str, str] = _OVERRIDE_DATA["rev_titles"]


def _extract_revs(title: str) -> list[int]:
    """PR タイトルから REV 番号を全て抽出する（大文字小文字不問）。

    対応フォーマット:
      "feat: REV-039 タイトル"          → [39]
      "fix: REV-282 REV-267 タイトル"   → [282, 267]
      "chore: REV-035/036/040 タイトル" → [35, 36, 40]
    """
    revs: list[int] = []
    for m in re.finditer(r"REV-(\d+)", title, re.IGNORECASE):
        revs.append(int(m.group(1)))
        # スラッシュ区切りの追加番号 (REV-035/036/040 形式)
        slash_match = re.match(r"((?:/\d+)+)", title[m.end():])
        if slash_match:
            for num in re.findall(r"/(\d+)", slash_match.group(1)):
                revs.append(int(num))
    return revs


def _fetch_pr_rev_map() -> dict[str, dict]:
    """GitHub CLI で PR 一覧を取得し REV → {status, pr_num} のマップを返す。"""
    try:
        result = subprocess.run(
            ["gh", "pr", "list", "--state", "all", "--limit", "300",
             "--json", "number,title,state,mergedAt"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"エラー: gh コマンド失敗: {e}", file=sys.stderr)
        sys.exit(1)

    prs = json.loads(result.stdout)
    rev_map: dict[str, dict] = {}

    for pr in prs:
        rev_nums = _extract_revs(pr["title"])
        for num in rev_nums:
            rev_id = f"REV-{num:03d}"
            state = pr["state"]
            if state == "MERGED":
                # MERGED は常に applied で上書き
                rev_map[rev_id] = {"status": "applied", "pr_num": pr["number"]}
            elif state == "CLOSED" and rev_id not in rev_map:
                rev_map[rev_id] = {"status": "rejected", "pr_num": pr["number"]}

    # PR タイトル非タグの既知マッピングをマージ（MERGED > KNOWN_OVERRIDE > CLOSED）
    for rev_id, (pr_num, status) in KNOWN_PR_OVERRIDES.items():
        if rev_id not in rev_map or (status == "applied" and rev_map[rev_id]["status"] != "applied"):
            rev_map[rev_id] = {"status": status, "pr_num": pr_num}

    return rev_map


def _get_ledger_latest() -> dict[str, dict]:
    """台帳から key → {status, migrated} の最新エントリを返す。

    - migrated=True: 既に canonical_key 形式（新形式）で記録済み
    - misc_* / 任意キーも含めて返す（KNOWN_CODE_APPLIED の照合に使用）
    """
    if not LEDGER_PATH.exists():
        return {}
    status_by_key: dict[str, str] = {}
    migrated: set[str] = set()

    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            s = e.get("status", "")
            if not s:
                continue
            # 新形式: rev_id フィールドに REV ID を持つ
            rev_id = e.get("rev_id", "")
            if rev_id.startswith("REV-"):
                status_by_key[rev_id] = s
                migrated.add(rev_id)
                continue
            # canonical_key / misc_* を含む全キー
            k = e.get("key", "")
            if k:
                status_by_key[k] = s
                if k.startswith("REV-"):
                    migrated.add(k)
        except json.JSONDecodeError:
            continue

    return {
        k: {"status": s, "migrated": k in migrated}
        for k, s in status_by_key.items()
    }


def _get_rev_id_first_seen_key() -> dict[str, str]:
    """rev_id → 台帳にその REV 番号が最初に記録されたときの key を返す。

    canonical_key は本来 (title, description) から決定的に計算されるが、
    起票時（実タイトル・description あり）と本スクリプトの reconciliation時
    （REV_TITLES に無い REV 番号は rev_id 文字列自体をタイトル代わりに使う
    フォールバック、description なし）とで異なる入力から独立に計算されるため、
    同じ REV 番号でも別の canonical_key が生成されうる。結果、reconciliation の
    "applied" 書き込みが別キーの孤立エントリになり、本来のエントリが
    needs_review のまま取り残される事故が起きていた（例: REV-230, REV-237 —
    孤立キー misc_dfe021de8b2b / misc_665e5e075fed に "applied" が書かれ、
    元の misc_6be43f8b3dce / misc_64ce70442596 は needs_review のまま放置）。
    再発防止のため、その REV 番号が台帳に最初に出現したときの key を
    「本当の」canonical_key として扱い、以後の reconciliation 書き込みは
    必ずこのキーを再利用する。
    """
    if not LEDGER_PATH.exists():
        return {}
    result: dict[str, str] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        rev_id = e.get("rev_id", "")
        key = e.get("key", "")
        if rev_id.startswith("REV-") and key:
            result.setdefault(rev_id, key)
    return result


_OPEN_STATUSES = {"needs_review", "parked", "suppressed"}


def _norm_title(title: str) -> str:
    """タイトル一致用の軽量正規化（空白除去 + 小文字化）。

    canonical_key ほど強い正規化はかけず、レポート表記のブレ（前後空白）だけを吸収する。
    """
    return re.sub(r"\s+", "", str(title or "").strip().lower())


def _get_ledger_open_titles() -> dict[str, tuple[str, str]]:
    """runtime 台帳の「未決着（needs_review/parked/suppressed）」最新エントリを
    正規化タイトル → (実key, status) で返す。

    canonical_key が description 差異でブレても、runtime 台帳が保持する実 key を
    タイトルから解決できるようにするための逆引き表。
    """
    if not LEDGER_PATH.exists():
        return {}
    latest: dict[str, dict] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        k = e.get("key", "")
        if k:
            latest[k] = e  # last wins

    out: dict[str, tuple[str, str]] = {}
    for k, e in latest.items():
        if e.get("status") in _OPEN_STATUSES:
            nt = _norm_title(e.get("title", ""))
            if nt:
                out[nt] = (k, e.get("status", ""))
    return out


def _apply_title_matched_closures(now: str) -> list[dict]:
    """KNOWN_TITLE_APPLIED を runtime 台帳の未決着エントリにタイトル一致で当て、
    実 key のまま applied 追記する更新エントリ群を返す（副作用なし）。

    - 台帳に該当タイトルの未決着エントリが無ければ何もしない（no-op）。
    - 既に applied のものは _OPEN_STATUSES に含まれないため自然に対象外。
    """
    open_by_title = _get_ledger_open_titles()
    updates: list[dict] = []
    for title, reason in KNOWN_TITLE_APPLIED:
        hit = open_by_title.get(_norm_title(title))
        if not hit:
            continue
        real_key, cur_status = hit
        updates.append({
            "key": real_key,
            "rev_id": "",
            "status": "applied",
            "title": title,
            "canonical_key": real_key,
            "pr_url": "",
            "reason": f"{reason}（旧status={cur_status}→applied, title一致で解決）",
            "recorded_at": now,
        })
    return updates


def _append_ledger(entry: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="台帳に実際に書き込む（省略時は dry-run）")
    parser.add_argument("--mark-deferred", action="store_true", help="PR なしの REV も deferred として台帳に追加する")
    args = parser.parse_args()

    print("GitHub PR から REV マッピングを取得中...")
    pr_map = _fetch_pr_rev_map()
    ledger_latest = _get_ledger_latest()
    rev_id_first_key = _get_rev_id_first_seen_key()

    now = datetime.now().isoformat()

    updates: list[dict] = []

    # PR から判定した applied / rejected
    for rev_id, info in sorted(pr_map.items()):
        entry_info = ledger_latest.get(rev_id, {})
        current_status = entry_info.get("status", "")
        already_migrated = entry_info.get("migrated", False)
        new_status = info["status"]
        pr_num = info["pr_num"]

        # ステータスが同じかつ新形式で記録済みなら何もしない
        if current_status == new_status and already_migrated:
            continue

        reason = (
            f"PR #{pr_num} マージ済み" if new_status == "applied"
            else f"PR #{pr_num} クローズ（未マージ）"
        )
        if current_status == new_status and not already_migrated:
            reason += "（旧形式→canonical_key 移行）"
        title = REV_TITLES.get(rev_id, rev_id)
        # REV_TITLES に無い REV 番号は、rev_id 文字列そのものから canonical_key を
        # 計算すると起票時の実キーと食い違い孤立エントリを生む（詳細は
        # _get_rev_id_first_seen_key のdocstring）。台帳に記録済みの実キーが
        # あれば必ずそれを再利用する。
        ck = rev_id_first_key.get(rev_id) or _canonical_key(title)
        updates.append({
            "key": ck,          # pipeline が is_processed() で照合するキー
            "rev_id": rev_id,   # 人間可読の REV ID（参照用）
            "status": new_status,
            "title": title,
            "canonical_key": ck,
            "pr_url": f"https://github.com/kobayashiisaoryou/tune_lease_55/pull/{pr_num}",
            "reason": reason,
            "recorded_at": now,
        })

    # PR なし・コードレビュー確認済みの REV 項目 (KNOWN_APPLIED_NO_PR)
    for rev_id, (commit, src_file, desc) in sorted(KNOWN_APPLIED_NO_PR.items()):
        entry_info = ledger_latest.get(rev_id, {})
        if entry_info.get("status") == "applied":
            continue
        updates.append({
            "key": rev_id,
            "status": "applied",
            "title": REV_TITLES.get(rev_id, rev_id),
            "canonical_key": rev_id.lower(),
            "pr_url": None,
            "reason": f"コードレビュー確認済み: {src_file} (commit: {commit}) — {desc}",
            "recorded_at": now,
        })

    # PR #273 rawチャットメモ＋重複エントリのコード確認済み applied (KNOWN_CODE_APPLIED)
    for (key, title, reason) in KNOWN_CODE_APPLIED:
        entry_info = ledger_latest.get(key, {})
        if entry_info.get("status") == "applied":
            continue
        updates.append({
            "key": key,
            "rev_id": "",
            "status": "applied",
            "title": title,
            "canonical_key": key,
            "pr_url": "",
            "reason": reason,
            "recorded_at": now,
        })

    # B群: 出荷済みだが canonical_key が出荷REVと繋がらず needs_review で滞留している
    # 生メモを、runtime 台帳のタイトル一致で実キーごと applied 化する。
    updates.extend(_apply_title_matched_closures(now))

    # deferred（PR なし） - --mark-deferred フラグ時のみ
    if args.mark_deferred:
        all_rev_ids = set(REV_TITLES.keys())
        for rev_id in sorted(all_rev_ids):
            if rev_id in pr_map:
                continue  # PR あり → スキップ
            entry_info = ledger_latest.get(rev_id, {})
            current = entry_info.get("status", "")
            already_migrated = entry_info.get("migrated", False)
            if current in ("applied", "deferred") and already_migrated:
                continue
            deferred_title = REV_TITLES.get(rev_id, rev_id)
            deferred_ck = _canonical_key(deferred_title)
            updates.append({
                "key": deferred_ck,
                "rev_id": rev_id,
                "status": "deferred",
                "title": deferred_title,
                "canonical_key": deferred_ck,
                "pr_url": "",
                "reason": "PR なし・手動タグ群",
                "recorded_at": now,
            })

    if not updates:
        print("更新対象なし（台帳は既に最新）")
        return

    print(f"\n{'[DRY-RUN] ' if not args.apply else ''}更新件数: {len(updates)}\n")
    by_status: dict[str, list[str]] = {}
    for u in updates:
        label = u.get("rev_id") or u["key"]
        by_status.setdefault(u["status"], []).append(f"  {label}: {u['title'][:40]}")

    for st, lines in sorted(by_status.items()):
        print(f"=== {st} ({len(lines)}) ===")
        for line in lines:
            print(line)
        print()

    if args.apply:
        for u in updates:
            _append_ledger(u)
        print(f"台帳を更新しました: {LEDGER_PATH}")
    else:
        print("実際に書き込むには --apply を付けて実行してください。")


if __name__ == "__main__":
    main()
