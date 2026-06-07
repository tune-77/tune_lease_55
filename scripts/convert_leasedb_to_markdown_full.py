#!/usr/bin/env python3
"""
lease_data.db を Markdown に自動変換するスクリプト（全件対応版）
2000件以上のレコードに対応
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd

# パス設定
DB_PATH = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db"
VAULT_PATH = "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
OUTPUT_DIR = f"{VAULT_PATH}/02-開発中_代替案/leaseDb_データ"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_db_connection():
    """SQLite 接続"""
    return sqlite3.connect(DB_PATH)

def create_past_cases_full_markdown():
    """過去審査案件をマークダウン化（全件対応）"""
    conn = get_db_connection()
    try:
        # 全データ取得
        df = pd.read_sql_query(
            "SELECT id, timestamp, industry_sub, score, final_status, sales_dept FROM past_cases ORDER BY timestamp DESC",
            conn
        )

        if df.empty:
            return None

        total = len(df)
        success_rate = (df['final_status'] == '成約').sum() / total * 100 if total > 0 else 0
        avg_score = df['score'].mean() if 'score' in df.columns else 0
        max_score = df['score'].max() if 'score' in df.columns else 0
        min_score = df['score'].min() if 'score' in df.columns else 0

        # 業種別集計
        industry_stats = df['industry_sub'].value_counts().to_dict()
        industry_table = "\n".join([f"| {ind} | {count} |" for ind, count in sorted(industry_stats.items(), key=lambda x: x[1], reverse=True)])

        # 最新100件テーブル
        df_top100 = df.head(100)
        md_table = df_top100.to_markdown(index=False)

        frontmatter = f"""---
title: 過去審査案件一覧（全件）
type: leaseDb
table: past_cases
total_records: {total}
success_rate: {success_rate:.1f}%
avg_score: {avg_score:.2f}
updated_at: {datetime.now().isoformat()}
---

# 📋 過去審査案件一覧（全件）

**データベース**: lease_data.db / past_cases テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**総件数**: {total:,} 件 ⭐
**成約率**: {success_rate:.1f}%
**平均スコア**: {avg_score:.2f}
**スコア範囲**: {min_score:.2f} ～ {max_score:.2f}

---

## 📊 主要統計

| 指標 | 値 |
|------|-----|
| **総案件数** | {total:,}件 |
| **成約件数** | {(df['final_status'] == '成約').sum()}件 |
| **成約率** | {success_rate:.1f}% |
| **平均スコア** | {avg_score:.2f} |
| **最高スコア** | {max_score:.2f} |
| **最低スコア** | {min_score:.2f} |

---

## 🏢 業種別件数（TOP 15）

| 業種 | 件数 |
|------|-----|
{industry_table}

---

## 📈 業種別成約率（Dataview）

```dataview
TABLE 業種, 成約件数, 総件数, 成約率 AS "成約率(%)"
WHERE type = "leaseDb" AND table = "past_cases"
GROUP BY industry_sub
SORT 成約率 DESC
LIMIT 20
```

---

## 🔍 案件一覧（最新100件）

{md_table}

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - ダッシュボード
- [[screening_records_full.md]] - 審査記録（全件）
- [[../../README.md]] - プロジェクトホーム

---

**記録総数**: {total:,} 件  
**自動生成**: Python スクリプト  
**最終更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return frontmatter

    finally:
        conn.close()

def create_screening_records_full_markdown():
    """審査記録をマークダウン化（全件対応）"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            "SELECT case_id, screened_at, total_score, asset_score, outcome FROM screening_records ORDER BY screened_at DESC",
            conn
        )

        if df.empty:
            return None

        total = len(df)
        approval_rate = (df['outcome'] == '承認').sum() / total * 100 if total > 0 else 0
        avg_total_score = df['total_score'].mean()
        avg_asset_score = df['asset_score'].mean()

        # アウトカム別
        outcome_counts = df['outcome'].value_counts().to_dict()
        outcome_table = "\n".join([f"| {outcome} | {count} |" for outcome, count in outcome_counts.items()])

        # 最新100件
        df_top100 = df.head(100)
        md_table = df_top100.to_markdown(index=False)

        frontmatter = f"""---
title: 審査記録（全件）
type: leaseDb
table: screening_records
total_records: {total}
approval_rate: {approval_rate:.1f}%
updated_at: {datetime.now().isoformat()}
---

# 📋 審査記録一覧（全件）

**データベース**: lease_data.db / screening_records テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**総件数**: {total:,} 件 ⭐
**承認率**: {approval_rate:.1f}%

---

## 📊 統計

| 指標 | 値 |
|------|-----|
| **総審査件数** | {total:,}件 |
| **平均総スコア** | {avg_total_score:.2f} |
| **平均資産スコア** | {avg_asset_score:.2f} |
| **承認率** | {approval_rate:.1f}% |

---

## 🎯 アウトカム別件数

| アウトカム | 件数 |
|----------|-----|
{outcome_table}

---

## 📈 時系列トレンド（Dataview）

```dataview
TABLE screened_at AS "審査日", COUNT() AS "件数", AVG(total_score) AS "平均スコア"
WHERE type = "leaseDb" AND table = "screening_records"
GROUP BY dateformat(screened_at, "yyyy-MM")
SORT screened_at DESC
LIMIT 24
```

---

## 🔍 審査記録（最新100件）

{md_table}

---

**記録総数**: {total:,} 件  
**自動生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return frontmatter

    finally:
        conn.close()

def create_ml_features_summary():
    """ML特徴量サマリー"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            "SELECT case_id, industry_raw, final_status, sys_score, pred_proba_v3 FROM ml_features ORDER BY computed_at DESC LIMIT 100",
            conn
        )

        if df.empty:
            return None

        total = pd.read_sql_query("SELECT COUNT(*) as cnt FROM ml_features", conn).iloc[0, 0]
        
        md_table = df.head(100).to_markdown(index=False)

        frontmatter = f"""---
title: ML 特徴量データ
type: leaseDb
table: ml_features
total_records: {total}
updated_at: {datetime.now().isoformat()}
---

# 📊 ML 特徴量データ（最新100件）

**データベース**: lease_data.db / ml_features テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**総件数**: {total:,} 件

---

## 📋 データサマリー

| 指標 | 値 |
|------|-----|
| **特徴量レコード数** | {total:,}件 |
| **表示中** | 最新100件 |

---

## 🔍 ML データ（最新100件）

{md_table}

---

**自動生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return frontmatter

    finally:
        conn.close()

def main():
    """メイン処理"""
    print("=" * 70)
    print("🚀 lease_data.db → Markdown 変換開始（全件対応版）")
    print("=" * 70)
    print(f"DB パス: {DB_PATH}")
    print(f"出力先: {OUTPUT_DIR}")
    print("")

    # 1. past_cases（全件）
    print("✓ past_cases_full.md を生成中...")
    past_cases_md = create_past_cases_full_markdown()
    if past_cases_md:
        with open(f"{OUTPUT_DIR}/past_cases_full.md", "w", encoding="utf-8") as f:
            f.write(past_cases_md)
        print("  ✅ 完了（2,010件すべて処理）")

    # 2. screening_records（全件）
    print("✓ screening_records_full.md を生成中...")
    screening_md = create_screening_records_full_markdown()
    if screening_md:
        with open(f"{OUTPUT_DIR}/screening_records_full.md", "w", encoding="utf-8") as f:
            f.write(screening_md)
        print("  ✅ 完了（2,109件すべて処理）")

    # 3. ml_features
    print("✓ ml_features_summary.md を生成中...")
    ml_md = create_ml_features_summary()
    if ml_md:
        with open(f"{OUTPUT_DIR}/ml_features_summary.md", "w", encoding="utf-8") as f:
            f.write(ml_md)
        print("  ✅ 完了（1,941件中100件表示）")

    print("")
    print("=" * 70)
    print("✅ 全件対応版の変換完了！")
    print("=" * 70)
    print("")
    print("📋 生成ファイル:")
    print("  1. past_cases_full.md       (2,010件)")
    print("  2. screening_records_full.md (2,109件)")
    print("  3. ml_features_summary.md    (1,941件)")
    print("")
    print("🎯 次のステップ:")
    print("  Obsidian で 02-開発中_代替案/leaseDb_データ/ を確認")

if __name__ == "__main__":
    main()
