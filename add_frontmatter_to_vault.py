#!/usr/bin/env python3
"""
Obsidian Vault に Frontmatter を自動追加するスクリプト

優先度順：
1. Asset Knowledge/ - 物件知識（最重要）
2. Cases/ - 案件ログ
3. AI Chat/ - チャットログ
4. Asset Finance/ - 資産分析
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Vault パス
VAULT_ROOT = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents" / "Obsidian Vault"
LEASE_PROJECT = VAULT_ROOT / "Projects" / "tune_lease_55"

# フォルダごとの Frontmatter テンプレート
ASSET_KNOWLEDGE_TEMPLATE = """---
title: {title}
tags: [{tags}]
industry: {industry}
asset_type: {asset_type}
created: {created}
updated: {updated}
---
"""

CASES_TEMPLATE = """---
title: {title}
tags: [{tags}]
industry: {industry}
score_range: {score_range}
credit_rating: {credit_rating}
case_status: {case_status}
created: {created}
updated: {updated}
---
"""

CHAT_LOG_TEMPLATE = """---
title: {title}
tags: [ai-chat, {date}]
chat_date: {date}
created: {created}
---
"""

ASSET_FINANCE_TEMPLATE = """---
title: {title}
tags: [{tags}]
asset_type: {asset_type}
analysis_type: {analysis_type}
created: {created}
updated: {updated}
---
"""


def has_frontmatter(text: str) -> bool:
    """ファイルが既に frontmatter を持つかチェック。"""
    return text.startswith("---\n") and "\n---\n" in text[4:]


def extract_title_from_filename(path: Path) -> str:
    """ファイル名からタイトルを抽出。"""
    name = path.stem
    # スネークケース → タイトルケース
    return name.replace("_", " ").title()


def infer_industry_from_path(path: Path) -> Optional[str]:
    """パス・ファイル名から業種を推測。"""
    text = str(path).lower()

    industries = {
        "c 製造業": ["製造", "機械", "工作", "金属"],
        "d 建設業": ["建設", "工事", "建機"],
        "h 運輸業": ["車両", "運輸", "物流", "建機"],
        "i 卸売業": ["卸売", "卸"],
        "j 小売業": ["小売", "販売"],
        "l 不動産業": ["不動産", "土地"],
        "p 医療・福祉": ["医療", "医療機器", "福祉"],
    }

    for industry, keywords in industries.items():
        if any(kw in text for kw in keywords):
            return industry

    return None


def infer_asset_type(path: Path) -> str:
    """ファイル名から物件タイプを推測。"""
    name = str(path.stem).lower()

    asset_types = {
        "車両": ["車", "冷凍車", "キャラバン", "ハイエース"],
        "建機": ["建機", "pc200", "バックホウ", "パワーショベル"],
        "医療機器": ["医療", "mri", "ct"],
        "工作機械": ["機械", "cnc", "プレス", "射出成形"],
        "発電機": ["発電", "ディーゼル"],
        "フォークリフト": ["フォーク", "リフト"],
    }

    for atype, keywords in asset_types.items():
        if any(kw in name for kw in keywords):
            return atype

    return "その他"


def extract_score_range_from_content(text: str) -> Optional[str]:
    """ノート内容からスコア範囲を抽出。"""
    # パターン: "スコア 75", "score:75", etc.
    match = re.search(r"(?:スコア|score)[\s:]*(\d+)", text)
    if match:
        score = int(match.group(1))
        # スコア ± 5 の範囲を設定
        min_score = max(0, score - 5)
        max_score = min(100, score + 5)
        return f"[{min_score}, {max_score}]"
    return None


def extract_credit_rating(text: str) -> Optional[str]:
    """信用格付を抽出。"""
    # パターン: "4-6", "信用格付：4-6" etc.
    match = re.search(r"(?:信用格付|credit|rating)[\s:]*(\d-\d|\d)", text)
    if match:
        return match.group(1)
    return None


def infer_case_status(filename: str) -> str:
    """ファイル名からケースステータスを推測。"""
    lower = filename.lower()
    if any(w in lower for w in ["承認", "承認済", "ok", "pass"]):
        return "approved"
    elif any(w in lower for w in ["否決", "否認", "ng", "reject"]):
        return "rejected"
    elif any(w in lower for w in ["条件", "条件付", "pending"]):
        return "conditional"
    return "open"


def build_frontmatter_for_file(path: Path, folder_type: str) -> Optional[str]:
    """ファイルに応じた frontmatter を生成。"""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"  ⚠️  読み取り失敗: {path.name} ({e})")
        return None

    if has_frontmatter(content):
        return None  # 既に frontmatter がある

    title = extract_title_from_filename(path)
    now = datetime.now()
    created = now.strftime("%Y-%m-%d")
    updated = created

    # フォルダタイプ別処理
    if folder_type == "asset_knowledge":
        industry = infer_industry_from_path(path) or "unknown"
        asset_type = infer_asset_type(path)
        tags = f"物件ファイナンス, {asset_type}"

        fm = ASSET_KNOWLEDGE_TEMPLATE.format(
            title=title,
            tags=tags,
            industry=industry,
            asset_type=asset_type,
            created=created,
            updated=updated,
        )

    elif folder_type == "cases":
        industry = infer_industry_from_path(path) or "unknown"
        score_range = extract_score_range_from_content(content) or "[0, 100]"
        credit_rating = extract_credit_rating(content) or "unknown"
        case_status = infer_case_status(path.stem)
        tags = f"案件, {case_status}"

        fm = CASES_TEMPLATE.format(
            title=title,
            tags=tags,
            industry=industry,
            score_range=score_range,
            credit_rating=credit_rating,
            case_status=case_status,
            created=created,
            updated=updated,
        )

    elif folder_type == "chat_log":
        # ファイル名から日付を抽出（例: 2026-05-30.md）
        chat_date = path.stem if re.match(r"\d{4}-\d{2}-\d{2}", path.stem) else created

        fm = CHAT_LOG_TEMPLATE.format(
            title=title,
            date=chat_date,
            created=created,
        )

    elif folder_type == "asset_finance":
        industry = infer_industry_from_path(path) or "unknown"
        asset_type = infer_asset_type(path)
        # ファイル名から分析タイプを推測
        analysis_type = "residual_value" if "残価" in path.stem else "financial_analysis"
        tags = f"資産分析, {asset_type}"

        fm = ASSET_FINANCE_TEMPLATE.format(
            title=title,
            tags=tags,
            asset_type=asset_type,
            analysis_type=analysis_type,
            created=created,
            updated=updated,
        )

    else:
        return None

    return fm


def insert_frontmatter_to_file(path: Path, frontmatter: str) -> bool:
    """ファイルの冒頭に frontmatter を挿入。"""
    try:
        content = path.read_text(encoding="utf-8")
        new_content = frontmatter + content
        path.write_text(new_content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  ❌ 書き込み失敗: {path.name} ({e})")
        return False


def process_folder(folder_path: Path, folder_type: str, dry_run: bool = False) -> tuple[int, int]:
    """フォルダ内のすべての .md ファイルに frontmatter を追加。"""
    if not folder_path.exists():
        print(f"❌ フォルダが見つかりません: {folder_path}")
        return 0, 0

    md_files = list(folder_path.rglob("*.md"))
    processed = 0
    skipped = 0

    print(f"\n📁 {folder_path.name} ({len(md_files)} files)")

    for path in md_files:
        fm = build_frontmatter_for_file(path, folder_type)
        if fm is None:
            skipped += 1
            continue

        if dry_run:
            print(f"  [DRY RUN] ✓ {path.name}")
        else:
            if insert_frontmatter_to_file(path, fm):
                print(f"  ✓ {path.name}")
                processed += 1
            else:
                skipped += 1

    return processed, skipped


def main():
    """メイン処理。"""
    dry_run = "--dry-run" in sys.argv or "-n" in sys.argv

    if dry_run:
        print("🔍 ドライラン モード（ファイルは変更されません）\n")
    else:
        print("⚠️  本番モード（ファイルが変更されます）\n")
        input("続行するには Enter を押してください: ")

    print("=" * 60)
    print("Obsidian Vault Frontmatter 自動追加")
    print("=" * 60)

    total_processed = 0
    total_skipped = 0

    # 優先度順に処理
    folders = [
        (LEASE_PROJECT / "Asset Knowledge", "asset_knowledge", "物件ファイナンス知識 ⭐⭐⭐"),
        (LEASE_PROJECT / "Cases" / "2026-05", "cases", "2026年5月の案件ログ ⭐⭐"),
        (LEASE_PROJECT / "AI Chat", "chat_log", "AI チャットログ ⭐"),
        (LEASE_PROJECT / "Asset Finance", "asset_finance", "資産ファイナンス分析 ⭐"),
    ]

    for folder_path, folder_type, description in folders:
        print(f"\n📌 {description}")
        processed, skipped = process_folder(folder_path, folder_type, dry_run)
        total_processed += processed
        total_skipped += skipped

    print("\n" + "=" * 60)
    print(f"✅ 完了: {total_processed} ファイル処理、{total_skipped} ファイルスキップ")
    if dry_run:
        print("💡 ドライラン完了。実行するには --dry-run を削除してください。")
    print("=" * 60)


if __name__ == "__main__":
    main()
