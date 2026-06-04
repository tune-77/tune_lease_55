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
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".agents" / "skills" / "auto-improvement-pipeline"))

try:
    import pipeline_ledger as ledger  # type: ignore
except ImportError:
    ledger = None  # type: ignore

LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"

# PR タイトルに REV-NNN が明記されていないが確認済みのもの
# (PR 番号, status)
# PR を経ずにコミット直接で実装済みが確認されたチャット改善メモ
# (canonical_key, title, commit_or_reason)
# → --apply 実行時に台帳へ applied として追記する
KNOWN_CODE_APPLIED: list[tuple[str, str, str]] = [
    # PR #273 で取り込まれた raw チャットメモ（2026-06-01/02）のうち実装済みもの
    ("misc_3dd41b39e887", "結果登録で金利が入れられない",
     "e7e71ea: parseRateInput 追加・入力を text 型に変更"),
    ("misc_9ae8c238acbf", "最終結果登録時にエラーが発生する",
     "e7e71ea: api/main.py + register/page.tsx を安定化"),
    ("misc_7531d9ed95dd", "ビジュアルインサイト画面にAIコメントがない",
     "REV-109実装済み: visual/page.tsx L347 AI状況説明コメントパネル"),
    ("misc_3145ef0539d7", "ニュースの要約機能がない",
     "PR #265: リースニュース要約→Obsidian保存→ホーム掲示"),
    ("misc_a79737a7c5ae", "リースニュースの記事内容が薄い",
     "PR #265/266: Gemini要約＋新フォーマット統一により内容拡充"),
    # needs_review だが実装済み REV と同一タイトルの重複エントリ
    ("repeat_query",                 "同一クエリ繰り返し対応",              "REV-017 applied: chat_intent.py"),
    ("industry_question_clarification", "業界情報に関する質問の具体化支援", "REV-014 applied: chat_intent.py"),
    ("misc_08d6373ad7e9",            "業種別成約率の傾向情報提供",          "REV-023 applied（重複キー）"),
    ("misc_1c13f7a8dbd0",            "条件付き承認の推奨アクション自動提示","REV-011 applied（重複キー）"),
    ("misc_1dd0127850bb",            "業界別成約率データの提供",             "REV-041 applied（重複キー）"),
    ("misc_2182c9351393",            "ホーム画面への改善項目表示",           "REV-050 applied（重複キー）"),
    ("misc_26cd5d97058f",            "2段階モデル設計・実装",                "REV-002 applied: 動的金利提案エンジン"),
    ("misc_54f6bcf4b493",            "業種別成約率の傾向情報提供",           "REV-023 applied（重複）"),
    ("misc_5bd5ed432830",            "曖昧な質問「今日の」への対応強化",    "REV-013 applied: chat_intent.py"),
    ("misc_5cebef364a12",            "補助金関連情報の整理と参照性向上",    "REV-055 applied（重複キー）"),
    ("misc_6abc8ad3aaa1",            "アンサンブルモデル（CatBoost追加）",  "REV-004 applied（重複キー）"),
    ("misc_8bade82fc31d",            "数字入力の効率化とOCRの改善",          "REV-048 applied（重複キー）"),
    ("misc_b0d8041cab2b",            "期間指定データ集計機能の追加",         "REV-040 applied（重複キー）"),
    ("misc_bcc44e9f249e",            "条件付き承認の推奨アクション自動提示","REV-011 applied（重複）"),
    ("misc_bccd69403e88",            "知識宇宙マップの視覚化機能強化",       "REV-022 applied（重複キー）"),
    ("misc_c992757ce799",            "曖昧な質問「今日の」への対応強化",    "REV-013 applied（重複）"),
    ("misc_cecb0a5a3d79",            "知識宇宙マップの視覚化機能強化",       "REV-022 applied（重複）"),
    ("misc_d1199d9942d4",            "ホーム画面へのリースニュース表示",     "PR #265 実装済み"),
    ("misc_70ff923bdcdf",            "ホーム画面リースニュースまとめのコンテンツ拡充", "PR #265/266 実装済み"),
]

KNOWN_PR_OVERRIDES: dict[str, tuple[int, str]] = {
    "REV-001": (193, "rejected"),   # PR#193 CLOSED: EDINET API連携
    "REV-005": (202, "applied"),    # PR#202 MERGED: OCRモバイル入力機能
    "REV-007": (207, "rejected"),   # PR#207 CLOSED: ポートフォリオリスク管理
    "REV-010": (211, "applied"),    # PR#211 MERGED: 公平性・バイアス監査基盤
    "REV-011": (212, "applied"),    # PR#212 MERGED: 条件付き承認の推奨アクション
    "REV-016": (211, "applied"),    # PR#211 で同時マージ扱い
}

# REV ID → タイトル（reports から取得したマスタ）
REV_TITLES: dict[str, str] = {
    "REV-001": "EDINET連携（Phase2）",
    "REV-002": "動的金利提案エンジン / 2段階モデル設計",
    "REV-004": "デフォルト率モデルPD警告 / アンサンブルモデル",
    "REV-005": "OCRモバイル入力機能",
    "REV-007": "ポートフォリオリスク管理",
    "REV-009": "帝国データバンクAPI連携 / Counterfactual分析",
    "REV-010": "公平性・バイアス監査基盤",
    "REV-011": "条件付き承認の推奨アクション自動提示",
    "REV-016": "リース審査外の質問への対応",
    "REV-018": "詳細情報要求への対応強化",
    "REV-019": "物件名からの業種自動推測と更新",
    "REV-022": "知識宇宙マップの視覚化機能強化",
    "REV-023": "業種別成約率の傾向情報提供",
    "REV-025": "リースバック勉強会の理解度向上",
    "REV-026": "AI ChatのDB連携機能強化",
    "REV-027": "リース情報活用方法の検討",
    "REV-035": "業種別成約率タブ追加",
    "REV-040": "期間指定データ集計機能の追加",
    "REV-041": "業界別成約率データの提供",
    "REV-048": "数字入力の効率化とOCRの改善",
    "REV-050": "ホーム画面への改善項目表示",
    "REV-055": "補助金関連情報の整理と参照性向上",
    "REV-061": "PD表示の明確化",
    "REV-064": "フォームUX改善",
    "REV-067": "FAQ強化",
    "REV-068": "ナレッジベース整備",
    "REV-069": "リスク強調表示",
    "REV-072": "条件付承認リスク強調",
    "REV-073": "Q_risk解釈ガイド追加",
    "REV-079": "リスク表示改善",
    "REV-085": "PD表示色分け",
    "REV-089": "量子干渉リスクUIパネル",
    "REV-094": "業種別成約率タブ強化",
    "REV-095": "ビジュアルインサイト改善",
    "REV-098": "ナレッジベース拡充",
    "REV-102": "Q_risk解釈ガイド拡充",
    "REV-107": "知識宇宙マップ改善",
    "REV-108": "知識宇宙マップ詳細強化",
    "REV-109": "ビジュアルインサイト AIコメントパネル",
    "REV-113": "Q_riskバッジ表示",
    "REV-114": "量子干渉リスク詳細",
    "REV-121": "営業ガイド強化",
    "REV-122": "FAQ拡充",
    "REV-133": "知識宇宙マップ機能拡張",
    "REV-138": "知識宇宙マップUI改善",
}


def _extract_revs(title: str) -> list[int]:
    """PR タイトルから REV 番号を全て抽出する。

    "REV-040/050/064/068" のようなスラッシュ区切りも処理する。
    """
    revs: list[int] = []
    for m in re.finditer(r"REV-(\d+)", title, re.IGNORECASE):
        revs.append(int(m.group(1)))
        rest = title[m.end():]
        for extra in re.findall(r"^(?:/(\d+))+", rest):
            pass
        # スラッシュ区切りの追加番号を個別に取得
        slash_match = re.match(r"((?:/\d+)+)", rest)
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
    """台帳から key → {status, recorded_at} の最新エントリを返す。"""
    if not LEDGER_PATH.exists():
        return {}
    latest: dict[str, dict] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
            k = e.get("key", "")
            s = e.get("status", "")
            if k and s:
                latest[k] = {"status": s, "recorded_at": e.get("recorded_at", "")}
        except json.JSONDecodeError:
            continue
    return latest


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

    now = datetime.now().isoformat()

    updates: list[dict] = []

    # PR から判定した applied / rejected
    for rev_id, info in sorted(pr_map.items()):
        current_status = ledger_latest.get(rev_id, {}).get("status", "")
        new_status = info["status"]
        pr_num = info["pr_num"]

        if current_status == new_status:
            continue  # 変更なし

        reason = (
            f"PR #{pr_num} マージ済み" if new_status == "applied"
            else f"PR #{pr_num} クローズ（未マージ）"
        )
        title = REV_TITLES.get(rev_id, rev_id)
        updates.append({
            "key": rev_id,
            "status": new_status,
            "title": title,
            "canonical_key": rev_id.lower(),
            "pr_url": f"https://github.com/kobayashiisaoryou/tune_lease_55/pull/{pr_num}",
            "reason": reason,
            "recorded_at": now,
        })

    # コードコミット直接で実装済みが確認された非 PR 項目
    for (key, title, reason) in KNOWN_CODE_APPLIED:
        entry_info = ledger_latest.get(key, {})
        current_status = entry_info.get("status", "") if isinstance(entry_info, dict) else entry_info
        if current_status == "applied":
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

    # deferred（PR なし） - --mark-deferred フラグ時のみ
    if args.mark_deferred:
        all_rev_ids = set(REV_TITLES.keys())
        for rev_id in sorted(all_rev_ids):
            if rev_id in pr_map:
                continue  # PR あり → スキップ
            current = ledger_latest.get(rev_id, {}).get("status", "")
            if current in ("applied", "deferred"):
                continue
            updates.append({
                "key": rev_id,
                "status": "deferred",
                "title": REV_TITLES.get(rev_id, rev_id),
                "canonical_key": rev_id.lower(),
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
        by_status.setdefault(u["status"], []).append(f"  {u['key']}: {u['title'][:40]}")

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
