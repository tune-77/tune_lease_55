"""
check_integrity.py — データ・モデル整合性チェック

「特徴量を追加したら再学習が必要」のような依存関係を自動検出する。
開発時・デプロイ前に実行する。

使い方:
    python check_integrity.py
    python check_integrity.py --fix   # 自動修正可能なものを修正
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from contextlib import closing

_BASE = os.path.dirname(os.path.abspath(__file__))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

errors: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def fail(msg: str, fix: str = "") -> None:
    errors.append(msg)
    print(f"  {FAIL} {msg}")
    if fix:
        print(f"     → 修正: {fix}")


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"  {WARN} {msg}")


# ===========================================================================
# 1. Mahalanobis モデル特徴量チェック
# ===========================================================================
def check_mahalanobis_features() -> None:
    print("\n[1] Mahalanobis モデル特徴量")

    try:
        sys.path.insert(0, _BASE)
        from train_mahalanobis import FEATURES as CURRENT_FEATURES
    except Exception as e:
        fail(f"train_mahalanobis.py の FEATURES 読み込み失敗: {e}")
        return

    model_path = os.path.join(_BASE, "data", "mahalanobis_model.joblib")
    if not os.path.exists(model_path):
        fail(
            "data/mahalanobis_model.joblib が存在しません",
            fix="python train_mahalanobis.py",
        )
        return

    try:
        import joblib
        model = joblib.load(model_path)
        saved_features = list(model.feature_names)
    except Exception as e:
        fail(f"モデルの読み込み失敗: {e}", fix="python train_mahalanobis.py")
        return

    if saved_features == list(CURRENT_FEATURES):
        ok(f"特徴量一致 ({len(CURRENT_FEATURES)}次元): {CURRENT_FEATURES}")
    else:
        added   = [f for f in CURRENT_FEATURES if f not in saved_features]
        removed = [f for f in saved_features if f not in CURRENT_FEATURES]
        detail  = []
        if added:
            detail.append(f"追加された特徴量: {added}")
        if removed:
            detail.append(f"削除された特徴量: {removed}")
        fail(
            f"モデルの特徴量({len(saved_features)}次元) と FEATURES({len(CURRENT_FEATURES)}次元) が不一致。"
            + " / ".join(detail),
            fix="python train_mahalanobis.py",
        )


# ===========================================================================
# 2. 営業部選択肢の一致チェック
# ===========================================================================
def check_sales_dept_options() -> None:
    print("\n[2] 営業部選択肢の一致")

    # scoring_core.py（"未設定" を含まない）
    try:
        from scoring_core import SALES_DEPT_OPTIONS as CORE_OPTS
    except Exception as e:
        fail(f"scoring_core.SALES_DEPT_OPTIONS 読み込み失敗: {e}")
        return

    # form_apply.py（"未設定" を含む）
    try:
        from components.form_apply import SALES_DEPT_OPTIONS as FORM_OPTS
    except Exception as e:
        fail(f"components/form_apply.SALES_DEPT_OPTIONS 読み込み失敗: {e}")
        return

    # form_apply から "未設定" を除いたものが scoring_core と一致すべき
    form_real = [o for o in FORM_OPTS if o != "未設定"]
    if form_real == list(CORE_OPTS):
        ok(f"scoring_core ↔ form_apply 一致: {CORE_OPTS}")
    else:
        fail(
            f"scoring_core {list(CORE_OPTS)} ≠ form_apply（未設定除く）{form_real}",
            fix="どちらかに合わせて拠点リストを統一してください",
        )

    # train_mahalanobis._DEPT_MAP の値が CORE_OPTS と一致するか
    try:
        from train_mahalanobis import _DEPT_MAP
        dept_values = sorted(_DEPT_MAP.values())
        core_sorted = sorted(CORE_OPTS)
        if dept_values == core_sorted:
            ok(f"train_mahalanobis._DEPT_MAP ↔ scoring_core 一致")
        else:
            fail(
                f"_DEPT_MAP の値 {dept_values} ≠ SALES_DEPT_OPTIONS {core_sorted}",
                fix="train_mahalanobis.py の _DEPT_MAP を更新後、python train_mahalanobis.py を実行",
            )
    except Exception as e:
        fail(f"train_mahalanobis._DEPT_MAP 読み込み失敗: {e}")

    # chat_wizard.py は実行時定数なので grep でチェック
    wiz_path = os.path.join(_BASE, "components", "chat_wizard.py")
    try:
        content = open(wiz_path, encoding="utf-8").read()
        missing = [opt for opt in CORE_OPTS if opt not in content]
        if missing:
            fail(
                f"chat_wizard.py に以下の営業部拠点が見当たりません: {missing}",
                fix="components/chat_wizard.py の _SALES_DEPT_OPTS を更新してください",
            )
        else:
            ok("chat_wizard.py に全拠点名が存在")
    except Exception as e:
        warn(f"chat_wizard.py の読み取りエラー: {e}")


# ===========================================================================
# 3. DB テーブル・カラム存在チェック
# ===========================================================================
def check_db() -> None:
    print("\n[3] SQLite DB テーブル・カラム")

    db_path = os.path.join(_BASE, "data", "lease_data.db")
    if not os.path.exists(db_path):
        fail(
            "data/lease_data.db が存在しません",
            fix="python migrate_to_sqlite.py 等でDBを初期化してください",
        )
        return

    REQUIRED_TABLES = {
        "gunshi_cases": ["id", "industry", "result"],
        "past_cases":   ["id"],
        "phrase_weights": ["phrase_id", "industry"],
    }

    try:
        with closing(sqlite3.connect(db_path)) as conn:
            cur = conn.cursor()
            existing_tables = {
                r[0] for r in cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            for table, required_cols in REQUIRED_TABLES.items():
                if table not in existing_tables:
                    fail(f"テーブル '{table}' が存在しません")
                    continue
                existing_cols = {
                    r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()
                }
                missing_cols = [c for c in required_cols if c not in existing_cols]
                if missing_cols:
                    fail(f"'{table}' に必要カラムがありません: {missing_cols}")
                else:
                    ok(f"テーブル '{table}' OK（{len(existing_cols)}カラム）")
    except Exception as e:
        fail(f"DB接続エラー: {e}")


# ===========================================================================
# 4. Mahalanobis モデルの特徴量と DB 営業部カラムの一致（拡張チェック）
# ===========================================================================
def check_dept_features_vs_dept_map() -> None:
    print("\n[4] FEATURES の dept_* と _DEPT_MAP の整合")

    try:
        from train_mahalanobis import FEATURES, _DEPT_MAP
    except Exception as e:
        fail(f"train_mahalanobis 読み込み失敗: {e}")
        return

    dept_in_features = sorted([f for f in FEATURES if f.startswith("dept_")])
    dept_in_map      = sorted(_DEPT_MAP.keys())

    if dept_in_features == dept_in_map:
        ok(f"FEATURES の dept_* ↔ _DEPT_MAP キー 一致: {dept_in_features}")
    else:
        only_features = [f for f in dept_in_features if f not in dept_in_map]
        only_map      = [f for f in dept_in_map if f not in dept_in_features]
        detail = []
        if only_features:
            detail.append(f"FEATURES のみ: {only_features}")
        if only_map:
            detail.append(f"_DEPT_MAP のみ: {only_map}")
        fail(
            "FEATURES と _DEPT_MAP のキーが不一致。" + " / ".join(detail),
            fix="train_mahalanobis.py で FEATURES と _DEPT_MAP を同時に更新してください",
        )


# ===========================================================================
# 自動修正（--fix オプション）
# ===========================================================================
def auto_fix() -> None:
    print("\n🔧 自動修正: Mahalanobis モデルを再学習します...")
    result = subprocess.run(
        [sys.executable, os.path.join(_BASE, "train_mahalanobis.py")],
        cwd=_BASE,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("  ✅ 再学習完了")
        print(result.stdout.strip())
    else:
        print("  ❌ 再学習失敗")
        print(result.stderr[-500:])


# ===========================================================================
# メイン
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="データ・モデル整合性チェック")
    parser.add_argument("--fix", action="store_true", help="自動修正可能な問題を修正する")
    args = parser.parse_args()

    print("=" * 60)
    print("  check_integrity.py — 整合性チェック")
    print("=" * 60)

    check_mahalanobis_features()
    check_sales_dept_options()
    check_db()
    check_dept_features_vs_dept_map()

    print("\n" + "=" * 60)
    if errors:
        print(f"  {FAIL} {len(errors)} 件のエラーがあります")
        for e in errors:
            print(f"    • {e}")
        if args.fix:
            auto_fix()
        else:
            print("\n  ヒント: python check_integrity.py --fix で自動修正を試みます")
        print("=" * 60)
        return 1
    elif warnings:
        print(f"  {WARN} {len(warnings)} 件の警告（エラーなし）")
        print("=" * 60)
        return 0
    else:
        print(f"  {PASS} すべてのチェックが通過しました")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
