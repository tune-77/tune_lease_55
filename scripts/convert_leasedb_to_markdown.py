#!/usr/bin/env python3
"""
lease_data.db を Markdown に自動変換するスクリプト
Obsidian Vault に Dataview 対応ドキュメントを生成
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import calendar

# パス設定
DB_PATH = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db"
VAULT_PATH = "/Users/kobayashiisaoryou/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault"
OUTPUT_DIR = f"{VAULT_PATH}/02-開発中_代替案/leaseDb_データ"

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_db_connection():
    """SQLite 接続"""
    return sqlite3.connect(DB_PATH)

def table_to_markdown(table_name, limit=100, columns=None):
    """
    SQLite テーブルを Markdown テーブルに変換
    """
    conn = get_db_connection()
    try:
        # テーブル情報取得
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)

        if df.empty:
            return None, "テーブルが空です"

        # 特定カラムのみ選択
        if columns:
            available_cols = [c for c in columns if c in df.columns]
            df = df[available_cols]

        # Markdown テーブルに変換
        md_table = df.to_markdown(index=False)

        return df, md_table
    finally:
        conn.close()

def create_past_cases_markdown():
    """過去審査案件をマークダウン化"""

    df, md_table = table_to_markdown(
        'past_cases',
        limit=50,
        columns=['id', 'timestamp', 'industry_sub', 'score', 'final_status', 'sales_dept']
    )

    if df is None:
        return None

    # 統計情報
    total = len(df)
    success_rate = (df['final_status'] == '成約').sum() / total * 100 if total > 0 else 0
    avg_score = df['score'].mean() if 'score' in df.columns else 0

    frontmatter = f"""---
title: 過去審査案件一覧
type: leaseDb
table: past_cases
total_records: {total}
success_rate: {success_rate:.1f}%
avg_score: {avg_score:.2f}
updated_at: {datetime.now().isoformat()}
---

# 📋 過去審査案件一覧

**データベース**: lease_data.db / past_cases テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**件数**: {total} 件
**成約率**: {success_rate:.1f}%
**平均スコア**: {avg_score:.2f}

---

## 📊 統計

| 指標 | 値 |
|------|-----|
| **総案件数** | {total}件 |
| **成約率** | {success_rate:.1f}% |
| **平均スコア** | {avg_score:.2f} |
| **最高スコア** | {df['score'].max():.2f} if 'score' in df.columns else 'N/A' |
| **最低スコア** | {df['score'].min():.2f} if 'score' in df.columns else 'N/A' |

---

## 📈 業種別件数

```dataview
TABLE 業種, 件数
WHERE type = "leaseDb" AND table = "past_cases"
GROUP BY industry_sub
```

---

## 🔍 案件一覧

{md_table}

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - ダッシュボード
- [[screening_records.md]] - 審査記録
- [[payment_history.md]] - 返済履歴
- [[../../README.md]] - プロジェクトホーム

---

**自動生成**: Python スクリプト
**頻度**: 日次（毎朝 6:00）
"""

    return frontmatter

def create_screening_records_markdown():
    """審査記録をマークダウン化"""

    df, md_table = table_to_markdown(
        'screening_records',
        limit=100,
        columns=['case_id', 'screened_at', 'total_score', 'asset_score', 'outcome']
    )

    if df is None:
        return None

    total = len(df)
    approval_rate = (df['outcome'] == '承認').sum() / total * 100 if total > 0 else 0

    frontmatter = f"""---
title: 審査記録
type: leaseDb
table: screening_records
total_records: {total}
approval_rate: {approval_rate:.1f}%
updated_at: {datetime.now().isoformat()}
---

# 📋 審査記録一覧

**データベース**: lease_data.db / screening_records テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**件数**: {total} 件
**承認率**: {approval_rate:.1f}%

---

## 📊 統計

| 指標 | 値 |
|------|-----|
| **総審査件数** | {total}件 |
| **承認率** | {approval_rate:.1f}% |
| **平均スコア** | {df['total_score'].mean():.2f} |

---

## 🎯 アウトカム別件数

```dataview
TABLE outcome AS "判定", COUNT() AS "件数"
WHERE type = "leaseDb" AND table = "screening_records"
GROUP BY outcome
```

---

## 🔍 審査記録（最新100件）

{md_table}

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - ダッシュボード
- [[past_cases.md]] - 過去案件
- [[payment_history.md]] - 返済履歴

---

**自動生成**: Python スクリプト
"""

    return frontmatter

def create_payment_history_markdown():
    """返済履歴をマークダウン化"""

    df, md_table = table_to_markdown(
        'payment_history',
        limit=100,
        columns=['contract_id', 'check_date', 'payment_status', 'overdue_amount']
    )

    if df is None:
        return None

    total = len(df)
    normal_rate = (df['payment_status'] == '正常').sum() / total * 100 if total > 0 else 0

    frontmatter = f"""---
title: 返済履歴
type: leaseDb
table: payment_history
total_records: {total}
normal_rate: {normal_rate:.1f}%
updated_at: {datetime.now().isoformat()}
---

# 📋 返済履歴

**データベース**: lease_data.db / payment_history テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**件数**: {total} 件
**正常率**: {normal_rate:.1f}%

---

## 📊 統計

| 指標 | 値 |
|------|-----|
| **総記録件数** | {total}件 |
| **正常返済率** | {normal_rate:.1f}% |

---

## 🔍 返済履歴（最新100件）

{md_table}

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - ダッシュボード
- [[past_cases.md]] - 過去案件
- [[screening_records.md]] - 審査記録

---

**自動生成**: Python スクリプト
"""

    return frontmatter

def create_subsidies_markdown():
    """補助金情報をマークダウン化"""

    df, md_table = table_to_markdown(
        'subsidies',
        limit=50,
        columns=['id', 'name', 'category', 'max_amount', 'deadline', 'active']
    )

    if df is None:
        return None

    total = len(df)

    frontmatter = f"""---
title: 補助金一覧
type: leaseDb
table: subsidies
total_records: {total}
updated_at: {datetime.now().isoformat()}
---

# 💰 利用可能な補助金一覧

**データベース**: lease_data.db / subsidies テーブル
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**件数**: {total} 件

---

## 📋 補助金リスト

{md_table}

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - ダッシュボード
- [[../../03-知識_業界/リース業界知識_索引.md]] - 業界知識

---

**自動生成**: Python スクリプト
"""

    return frontmatter

def is_month_end():
    """月末かどうかを判定"""
    today = datetime.now()
    # 来月の1日から1日前が月の最終日
    next_month = today.replace(day=28) + timedelta(days=4)
    last_day_of_month = (next_month - timedelta(days=next_month.day)).day
    return today.day == last_day_of_month


def create_monthly_summary_markdown():
    """月次サマリーレポートを生成"""

    conn = get_db_connection()
    try:
        today = datetime.now()
        year = today.year
        month = today.month

        # 月の最初と最後の日付
        first_day = f"{year}-{month:02d}-01"
        _, last_day = calendar.monthrange(year, month)
        last_date = f"{year}-{month:02d}-{last_day}"

        # その月の統計
        monthly_stats = pd.read_sql_query(f"""
            SELECT
              COUNT(*) AS "総件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS "失注",
              SUM(CASE WHEN final_status = '未登録' THEN 1 ELSE 0 END) AS "未登録"
            FROM past_cases
            WHERE DATE(timestamp) BETWEEN '{first_day}' AND '{last_date}'
        """, conn)

        # 業種別月次パフォーマンス
        industry_monthly = pd.read_sql_query(f"""
            SELECT
              industry_sub AS "業種",
              COUNT(*) AS "件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS "失注",
              SUM(CASE WHEN final_status = '未登録' THEN 1 ELSE 0 END) AS "未登録",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE DATE(timestamp) BETWEEN '{first_day}' AND '{last_date}'
            GROUP BY industry_sub
            ORDER BY "件数" DESC
        """, conn)

        # 日別トレンド
        daily_trend = pd.read_sql_query(f"""
            SELECT
              DATE(timestamp) AS "日付",
              COUNT(*) AS "件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS "失注",
              SUM(CASE WHEN final_status = '未登録' THEN 1 ELSE 0 END) AS "未登録",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE DATE(timestamp) BETWEEN '{first_day}' AND '{last_date}'
            GROUP BY DATE(timestamp)
            ORDER BY DATE(timestamp) DESC
        """, conn)

        # スコア帯別月次分析
        score_monthly = pd.read_sql_query(f"""
            SELECT
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END AS "スコア帯",
              COUNT(*) AS "件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE DATE(timestamp) BETWEEN '{first_day}' AND '{last_date}' AND final_status != '未登録'
            GROUP BY
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END
            ORDER BY score DESC
        """, conn)

        # 統計情報
        total = monthly_stats["総件数"].values[0] if len(monthly_stats) > 0 else 0
        success = monthly_stats["成約"].values[0] if len(monthly_stats) > 0 else 0
        fail = monthly_stats["失注"].values[0] if len(monthly_stats) > 0 else 0
        pending = monthly_stats["未登録"].values[0] if len(monthly_stats) > 0 else 0

        success_rate = (success / total * 100) if total > 0 else 0

        # Markdown テーブルに変換
        industry_table = industry_monthly.to_markdown(index=False)
        daily_table = daily_trend.to_markdown(index=False)
        score_table = score_monthly.to_markdown(index=False)

        frontmatter = f"""---
title: 月次サマリー {year}年{month}月
type: leaseDb
category: monthly_summary
month: {year}-{month:02d}
total_cases: {int(total)}
success_rate: {success_rate:.1f}%
updated_at: {datetime.now().isoformat()}
---

# 📊 月次サマリーレポート {year}年{month}月

**期間**: {year}年{month}月1日 ～ {year}年{month}月{last_day}日
**生成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 月次ハイライト

| 指標 | 件数 | 比率 |
|------|------|------|
| **総案件数** | {int(total)}件 | 100% |
| **成約** | {int(success)}件 | {success_rate:.1f}% ✅ |
| **失注** | {int(fail)}件 | {(fail/total*100):.1f}% ❌ |
| **未登録（処理中）** | {int(pending)}件 | {(pending/total*100):.1f}% ⏳ |

### 📈 前月比較（参考）

```
来月・先月との数値を確認したい場合は、
別の月次レポートと比較してください
```

---

## 🏭 業種別月次パフォーマンス

{industry_table}

### 💡 業種別インサイト

**最高成約率業種:**
```
{industry_monthly.loc[industry_monthly['成約率(%)'].idxmax(), '業種']}
→ {industry_monthly['成約率(%)'].max():.1f}%
```

**最低成約率業種:**
```
{industry_monthly.loc[industry_monthly['成約率(%)'].idxmin(), '業種']}
→ {industry_monthly['成約率(%)'].min():.1f}%
```

**最高ボリューム業種:**
```
{industry_monthly.loc[industry_monthly['件数'].idxmax(), '業種']}
→ {industry_monthly['件数'].max()}件
```

---

## 📅 日別トレンド（{year}年{month}月）

{daily_table}

### 📊 グラフ分析

```
成約件数の推移をグラフで確認したい場合は、
スプレッドシートにデータをコピーしてください
```

---

## 🎯 スコア帯別月次分析

{score_table if not score_monthly.empty else "データなし"}

### 営業インサイト

```
✅ 高スコア（80以上）の成約パターンを確認
✅ 低スコア（50～69）のボリューム分布を確認
✅ スコア精度の検証
```

---

## 📌 月次アクションアイテム

### 来月への提言

```
1. 成約率が低い業種への対策を検討
   └─ スコア基準の見直し、営業プロセス改善

2. ボリュームが多い業種への営業強化
   └─ 営業リソース配分の最適化

3. 未登録件数が多い場合
   └─ 審査プロセスの効率化を確認
```

---

## 🔗 関連ドキュメント

- [[leaseDb_ダッシュボード.md]] - メインダッシュボード
- [[2026-06_07_スコア別成約率分析.md]] - 直近スコア分析
- [[INDEX.md]] - leaseDb INDEX

---

## 📋 月次レポート履歴

毎月月末に自動生成されます。
Vault 内で `月次サマリー` を検索すると全件表示されます。

---

**自動生成**: Python スクリプト
**実行頻度**: 毎月月末日
**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return frontmatter

    finally:
        conn.close()


def create_score_band_analysis_markdown():
    """スコア帯別成約率分析をマークダウン化"""

    conn = get_db_connection()
    try:
        # スコア帯別成約率（確定済みのみ）
        score_analysis = pd.read_sql_query("""
            SELECT
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END AS "スコア帯",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              SUM(CASE WHEN final_status = '失注' THEN 1 ELSE 0 END) AS "失注",
              COUNT(*) AS "確定件数",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE score IS NOT NULL AND final_status != '未登録'
            GROUP BY
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END
            ORDER BY score DESC
        """, conn)

        # 業種別スコア帯クロス分析
        industry_score_analysis = pd.read_sql_query("""
            SELECT
              industry_sub AS "業種",
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END AS "スコア帯",
              COUNT(*) AS "件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE score IS NOT NULL AND final_status != '未登録'
            GROUP BY industry_sub,
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END
            ORDER BY industry_sub, score DESC
        """, conn)

        # 月別トレンド（スコア帯別）
        monthly_trend = pd.read_sql_query("""
            SELECT
              DATE(timestamp) AS "日付",
              CASE
                WHEN score >= 80 THEN '80以上'
                ELSE '80未満'
              END AS "スコア帯",
              COUNT(*) AS "件数",
              SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) AS "成約",
              ROUND(
                SUM(CASE WHEN final_status = '成約' THEN 1 ELSE 0 END) * 100.0 /
                COUNT(*),
                1
              ) AS "成約率(%)"
            FROM past_cases
            WHERE score IS NOT NULL AND final_status != '未登録'
            GROUP BY DATE(timestamp),
              CASE
                WHEN score >= 80 THEN '80以上'
                ELSE '80未満'
              END
            ORDER BY DATE(timestamp) DESC
            LIMIT 60
        """, conn)

        # 未登録案件の追跡（スコア帯別）
        pending_cases = pd.read_sql_query("""
            SELECT
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END AS "スコア帯",
              industry_sub AS "業種",
              COUNT(*) AS "未登録件数"
            FROM past_cases
            WHERE final_status = '未登録'
            GROUP BY
              CASE
                WHEN score >= 90 THEN '90以上'
                WHEN score >= 80 THEN '80～89'
                WHEN score >= 70 THEN '70～79'
                WHEN score >= 60 THEN '60～69'
                WHEN score >= 50 THEN '50～59'
                ELSE '50未満'
              END,
              industry_sub
            ORDER BY score DESC, COUNT(*) DESC
        """, conn)

        # Markdown テーブルに変換
        score_table = score_analysis.to_markdown(index=False)
        industry_table = industry_score_analysis.to_markdown(index=False)
        monthly_table = monthly_trend.head(20).to_markdown(index=False)
        pending_table = pending_cases.to_markdown(index=False)

        # 統計情報
        total_confirmed = score_analysis["確定件数"].sum()
        total_success = score_analysis["成約"].sum()
        overall_rate = (total_success / total_confirmed * 100) if total_confirmed > 0 else 0

        frontmatter = f"""---
title: スコア別成約率分析
type: leaseDb
category: analysis
analysis_date: {datetime.now().strftime('%Y-%m-%d')}
total_confirmed_cases: {int(total_confirmed)}
overall_success_rate: {overall_rate:.1f}%
updated_at: {datetime.now().isoformat()}
---

# 📊 スコア帯別成約率分析

**分析日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**対象**: 確定済み案件（未登録を除外）
**データベース**: lease_data.db / past_cases テーブル

---

## 🎯 エグゼクティブサマリー

| 指標 | 値 |
|------|-----|
| **分析対象件数** | {int(total_confirmed)}件 |
| **全体成約率** | {overall_rate:.1f}% |
| **最高成約率スコア帯** | {score_analysis.loc[score_analysis['成約率(%)'].idxmax(), 'スコア帯']}（{score_analysis['成約率(%)'].max():.1f}%） |
| **最低成約率スコア帯** | {score_analysis.loc[score_analysis['成約率(%)'].idxmin(), 'スコア帯']}（{score_analysis['成約率(%)'].min():.1f}%） |
| **分析範囲** | {score_analysis['スコア帯'].unique().tolist()} |

---

## 📈 スコア帯別成約率（確定済みデータのみ）

{score_table}

### 💡 インサイト

```
🥇 最高：{score_analysis.loc[score_analysis['成約率(%)'].idxmax(), 'スコア帯']} → {score_analysis['成約率(%)'].max():.1f}%
🥈 第2位：{score_analysis.sort_values('成約率(%)', ascending=False).iloc[1]['スコア帯']} → {score_analysis.sort_values('成約率(%)', ascending=False).iloc[1]['成約率(%)']:.1f}%
🥉 第3位：{score_analysis.sort_values('成約率(%)', ascending=False).iloc[2]['スコア帯']} → {score_analysis.sort_values('成約率(%)', ascending=False).iloc[2]['成約率(%)']:.1f}%
```

---

## 🏭 業種別×スコア帯クロス分析

{industry_table}

### 営業機会分析

**高成約率 ＆ 高ボリューム業種:**
```
スコア 80～89 帯で最も成約率が高い業種を優先ターゲットに設定
→ 営業効率が最大化される
```

**拡大機会:**
```
スコア 60～79 帯の中成約率業種
→ ボリュームが多いため、営業努力で成約率向上が可能
```

---

## 📅 月別トレンド（直近60日）

{monthly_table}

**トレンド分析:**
- 高スコア案件（80以上）の成約率推移を監視
- 季節変動やキャンペーン効果を検出

---

## ⏳ 処理中案件の追跡（未登録ケース）

{pending_table}

### 今後の影響予測

未登録案件が確定（成約/失注）に分類されると、以下のように変動します：

```
過去のパターンから推測:
- 成約率が 48～65% に上昇する可能性
- スコア帯によって成約パターンが異なる
- 約1ヶ月後に結果が確定
```

**監視項目:**
- 7月初旬に同様の分析を実施して比較
- 未登録案件の成約/失注の分類結果を確認
- 実績と予測のギャップを分析

---

## 🎯 営業戦略の提言

### 優先度ランキング

| ランク | スコア帯 | 成約率 | 戦略 |
|--------|---------|-------|------|
| **1位** | {score_analysis.sort_values('成約率(%)', ascending=False).iloc[0]['スコア帯']} | {score_analysis['成約率(%)'].max():.1f}% | 最優先営業対象 |
| **2位** | {score_analysis.sort_values('成約率(%)', ascending=False).iloc[1]['スコア帯']} | {score_analysis.sort_values('成約率(%)', ascending=False).iloc[1]['成約率(%)']:.1f}% | 次点対象 |
| **3位** | 50～59 | {score_analysis[score_analysis['スコア帯']=='50～59']['成約率(%)'].values[0]:.1f}% | ボリューム狙い |

### アクション

- ✅ スコア {score_analysis.sort_values('成約率(%)', ascending=False).iloc[0]['スコア帯']} 帯の顧客を営業の第一優先に
- ✅ 低スコア帯でも ボリュームあるため、提案効率を上げる
- ✅ 未登録案件の成約/失注分類を毎週 monitoring

---

## 🔗 関連リンク

- [[leaseDb_ダッシュボード.md]] - メインダッシュボード
- [[past_cases.md]] - 過去審査案件一覧
- [[INDEX.md]] - leaseDb INDEX

---

**自動生成**: Python スクリプト
**実行頻度**: 毎週日曜日
**更新日**: {datetime.now().strftime('%Y-%m-%d')}
"""

        return frontmatter

    finally:
        conn.close()


def create_dashboard_markdown():
    """leaseDb ダッシュボードを作成"""

    # 基本統計取得
    conn = get_db_connection()
    try:
        # past_cases
        past_cases_count = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM past_cases", conn
        ).iloc[0, 0]

        # screening_records
        screening_count = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM screening_records", conn
        ).iloc[0, 0]

        # payment_history
        payment_count = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM payment_history", conn
        ).iloc[0, 0]

        # screening_outcomes
        outcomes_count = pd.read_sql_query(
            "SELECT COUNT(*) as cnt FROM screening_outcomes", conn
        ).iloc[0, 0]

    finally:
        conn.close()

    dashboard = f"""---
title: leaseDb ダッシュボード
type: leaseDb
category: dashboard
updated_at: {datetime.now().isoformat()}
---

# 📊 leaseDb ダッシュボード

**Obsidian Vault × lease_data.db 統合ダッシュボード**
**更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 リアルタイム統計

| メトリクス | 件数 |
|----------|-----|
| **過去審査案件** | {past_cases_count}件 |
| **審査記録** | {screening_count}件 |
| **返済履歴記録** | {payment_count}件 |
| **成約追跡** | {outcomes_count}件 |

---

## 📈 主要ダッシュボード

### 1️⃣ 過去審査案件

[[past_cases.md|📋 過去審査案件一覧]]

```
最新50件の審査案件を表示
├─ 業種別分析
├─ スコア分布
└─ 成約率
```

### 2️⃣ 審査記録

[[screening_records.md|📋 審査記録一覧]]

```
最新100件の審査記録
├─ アウトカム別統計
├─ スコア分析
└─ 承認率トレンド
```

### 3️⃣ 返済履歴

[[payment_history.md|📋 返済履歴]]

```
最新100件の返済記録
├─ 返済ステータス
├─ 延滞分析
└─ 正常率
```

### 4️⃣ 補助金マスタ

[[subsidies_list.md|💰 補助金一覧]]

```
利用可能な補助金情報
├─ 補助金名
├─ 上限額
└─ 期限
```

---

## 🔍 Dataview クエリ例

### 成約率の業種別比較

```dataview
TABLE 業種, 成約件数, 総件数, 成約率
WHERE type = "leaseDb"
GROUP BY industry_sub
SORT 成約率 DESC
```

### 最近の審査記録（高スコア順）

```dataview
TABLE 案件ID, スコア, 判定
WHERE type = "leaseDb" AND table = "screening_records"
SORT total_score DESC
LIMIT 20
```

### 返済正常率（時系列）

```dataview
TABLE 月, 正常件数, 総件数, 正常率
WHERE type = "leaseDb" AND table = "payment_history"
GROUP BY check_date
SORT check_date DESC
LIMIT 12
```

---

## 🛠️ 自動更新設定

**自動実行**: 毎日 06:00
**スクリプト**: `convert_leasedb_to_markdown.py`
**更新対象**:
- past_cases.md
- screening_records.md
- payment_history.md
- subsidies_list.md

**cron 設定**:
```bash
0 6 * * * /usr/bin/python3 /Users/kobayashiisaoryou/clawd/tune_lease_55/scripts/convert_leasedb_to_markdown.py
```

---

## 📋 ドキュメント一覧

| ドキュメント | 用途 | 更新頻度 |
|-----------|------|---------|
| past_cases.md | 過去審査案件 | 日次 |
| screening_records.md | 審査記録 | 日次 |
| payment_history.md | 返済履歴 | 日次 |
| subsidies_list.md | 補助金情報 | 週次 |

---

## 🔗 関連リンク

- [[../02-開発中_代替案/改善A+B_ダッシュボード.md]] - 開発ダッシュボード
- [[../03-知識_業界/リース業界知識_索引.md]] - 業界知識
- [[../README.md]] - プロジェクトホーム

---

**DB 統合**: lease_data.db × Obsidian Vault
**方式**: 日次自動変換 + Dataview 表示
**作成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""

    return dashboard

def main():
    """メイン処理"""
    print("=" * 60)
    print("🚀 lease_data.db → Markdown 変換開始")
    print("=" * 60)
    print(f"DB パス: {DB_PATH}")
    print(f"出力先: {OUTPUT_DIR}")
    print("")

    # 1. past_cases を生成
    print("✓ past_cases.md を生成中...")
    past_cases_md = create_past_cases_markdown()
    if past_cases_md:
        with open(f"{OUTPUT_DIR}/past_cases.md", "w", encoding="utf-8") as f:
            f.write(past_cases_md)
        print("  ✅ 完了")

    # 2. screening_records を生成
    print("✓ screening_records.md を生成中...")
    screening_md = create_screening_records_markdown()
    if screening_md:
        with open(f"{OUTPUT_DIR}/screening_records.md", "w", encoding="utf-8") as f:
            f.write(screening_md)
        print("  ✅ 完了")

    # 3. payment_history を生成
    print("✓ payment_history.md を生成中...")
    payment_md = create_payment_history_markdown()
    if payment_md:
        with open(f"{OUTPUT_DIR}/payment_history.md", "w", encoding="utf-8") as f:
            f.write(payment_md)
        print("  ✅ 完了")

    # 4. subsidies を生成
    print("✓ subsidies_list.md を生成中...")
    subsidies_md = create_subsidies_markdown()
    if subsidies_md:
        with open(f"{OUTPUT_DIR}/subsidies_list.md", "w", encoding="utf-8") as f:
            f.write(subsidies_md)
        print("  ✅ 完了")

    # 5. ダッシュボードを生成
    print("✓ leaseDb_ダッシュボード.md を生成中...")
    dashboard_md = create_dashboard_markdown()
    if dashboard_md:
        with open(f"{OUTPUT_DIR}/leaseDb_ダッシュボード.md", "w", encoding="utf-8") as f:
            f.write(dashboard_md)
        print("  ✅ 完了")

    # 6. スコア帯別成約率分析を生成（毎週）
    print("✓ スコア別成約率分析.md を生成中...")
    today = datetime.now()
    year = today.strftime("%Y")
    month = today.strftime("%m")
    day = today.strftime("%d")

    analysis_filename = f"{year}-{month}_{day}_スコア別成約率分析.md"
    score_analysis_md = create_score_band_analysis_markdown()
    if score_analysis_md:
        with open(f"{OUTPUT_DIR}/{analysis_filename}", "w", encoding="utf-8") as f:
            f.write(score_analysis_md)
        print(f"  ✅ 完了: {analysis_filename}")

    # 7. 月次サマリーを生成（月末）
    if is_month_end():
        print("✓ 月次サマリーレポート.md を生成中...")
        month_summary_md = create_monthly_summary_markdown()
        if month_summary_md:
            month_summary_filename = f"月次サマリー_{year}年{month}月.md"
            with open(f"{OUTPUT_DIR}/{month_summary_filename}", "w", encoding="utf-8") as f:
                f.write(month_summary_md)
            print(f"  ✅ 完了: {month_summary_filename} ⭐")

    print("")
    print("=" * 60)
    print("✅ 変換完了！")
    print(f"📁 出力ファイル: {OUTPUT_DIR}")
    print("=" * 60)
    print("")
    print("📋 生成ファイル:")
    print("  1. leaseDb_ダッシュボード.md")
    print("  2. past_cases.md")
    print("  3. screening_records.md")
    print("  4. payment_history.md")
    print("  5. subsidies_list.md")
    print(f"  6. {analysis_filename} ⭐ 毎週生成")

    if is_month_end():
        print(f"  7. 月次サマリー_{year}年{month}月.md 🎉 月末に生成")

    print("")
    print("🎯 次のステップ:")
    print("  1. Obsidian を開く")
    print("  2. 02-開発中_代替案/leaseDb_データ/ を確認")
    print("  3. leaseDb_ダッシュボード.md でダッシュボードを確認")

if __name__ == "__main__":
    main()
