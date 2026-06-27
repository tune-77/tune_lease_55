#!/usr/bin/env python3
"""Build a Cloud Run-safe chat memory and anonymized case pack.

The output is a small public Markdown note under the normal Obsidian Vault:

    Projects/tune_lease_55/Lease Intelligence/Public/Chat Memory/

It is designed for Cloud Run RAG. It contains yesterday's durable decisions and
anonymous past-case examples, but not raw chat logs, Private Reflection, Daily
notes, customer names, or sensitive details.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VAULT = (
    Path.home()
    / "Library"
    / "Mobile Documents"
    / "iCloud~md~obsidian"
    / "Documents"
    / "Obsidian Vault"
)
VAULT_PATH = Path(os.environ.get("OBSIDIAN_VAULT", os.environ.get("OBSIDIAN_VAULT_PATH", str(DEFAULT_VAULT)))).expanduser()
OUTPUT_DIR = Path("Projects/tune_lease_55/Lease Intelligence/Public/Chat Memory")
STATE_DIR = PROJECT_ROOT / "memory"
LONG_TERM_MEMORY = PROJECT_ROOT / "MEMORY.md"

CORE_IDENTITY_MEMORIES = [
    "紫苑は、Kobayashiさんのリース審査の経験・違和感・判断基準を、再利用できる判断資産として育てるAI。",
    "回答は一般論で終わらせず、稟議・回収・残価・保守・条件付き承認など、実務判断に戻して返す。",
    "Cloudflare版で感じられた「記憶が近い」「同じ紫苑が返している」手触りを、Cloud Run版でも守る。",
    "正しさだけでなく、過去の判断と地続きに感じられる返し方を重視する。",
]

JUDGMENT_PRINCIPLE_MEMORIES = [
    "残価リスクは、中古流通・保守期限・撤去費・再販制約まで見る。",
    "補助金案件は、採択有無だけでなく、入金時期・未採択時の返済余力・返還リスクを見る。",
    "業種リスクは単独で見ず、物件価値・用途・収益改善根拠と組み合わせて判断する。",
    "承認/否決の二択で終わらせず、条件付き承認・追加確認・保全条件へ落とす。",
    "過去事例は顧客名ではなく、業種・物件・論点・判断の型として再利用する。",
]


DECISION_KEYWORDS = (
    "Cloud Run",
    "Cloud SQL",
    "Obsidian",
    "GCS",
    "Vault",
    "正本",
    "要約",
    "同期",
    "セキュリティ",
    "Private Reflection",
    "Daily",
    "ハッカソン",
)
LONG_TERM_MEMORY_KEYWORDS = (
    "Core Motivation",
    "AI Chat",
    "Knowledge Loop",
    "Current Focus",
    "Knowledge KPI",
    "Operating Mode",
    "Core Principle",
    "Aspiration",
    "Obsidian Default",
    "Memory Hygiene",
    "Dependency Triage",
    "Cloud Run",
    "GCS",
    "RAG",
    "紫苑",
    "リース知性体",
    "判断資産",
    "知識",
    "正本",
)
PRIVATE_MEMORY_KEYWORDS = (
    "Private Reflection",
    "Mana",
    "妹",
    "亡くなった",
    "personal",
    "個人",
    "Kobayashi",
)
CURATED_LONG_TERM_MEMORIES = [
    "このシステムは、リース審査の経験・判断基準・違和感・成功/失敗事例を、再利用できる判断資産として育てる。",
    "AIチャットは単発回答ではなく、Obsidianに保存された知識・ニュース・改善ログ・過去事例をRAGで思い出して回答する。",
    "重要なのは機能を増やすことより、知識ループを止めずに回し続けること。判断基準は追加より継続、複雑化より持続性に置く。",
    "知識進化は、知識化率・再利用率・効果率・重複/陳腐化率で見る。作ったかではなく、使われたか、判断が良くなったかを見る。",
    "Obsidianはローカル/iCloud Vaultを正本とし、Cloud Runは入力受付・デモ実行・AIチャットを担当する。",
    "Cloud Runへ渡す記憶は、公開可能な選抜知識・要約済み方針・匿名過去事例に限定する。Daily全文、Private Reflection、生ログは渡さない。",
    "Cloud Runで発生した入力や会話はCloud SQL/GCSへ保存し、ローカルMacの日次同期で要約としてObsidian正本へ戻す。",
    "紫苑の記憶は固定された正解ではなく、根拠・確信度・作成日・適用条件を持つ更新可能な信念として扱う。",
]


def identity_memories() -> list[str]:
    return [_redact(item, 240) for item in CORE_IDENTITY_MEMORIES]


def judgment_principle_memories() -> list[str]:
    return [_redact(item, 240) for item in JUDGMENT_PRINCIPLE_MEMORIES]


def _database_url_from_secret() -> str:
    secret_name = os.environ.get("DATABASE_URL_SECRET_NAME", "").strip()
    if not secret_name:
        return ""
    try:
        result = subprocess.run(
            ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_name}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        print(f"警告: Secret Manager から DATABASE_URL を読めませんでした: {type(exc).__name__}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def _database_url() -> str:
    current = os.environ.get("DATABASE_URL", "").strip()
    if current:
        return current
    loaded = _database_url_from_secret()
    if loaded:
        os.environ["DATABASE_URL"] = loaded
    return loaded


def _redact(text: Any, limit: int = 180) -> str:
    value = str(text or "")
    value = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[email]", value)
    value = re.sub(r"\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b", "[phone]", value)
    value = re.sub(r"\b\d{6,}\b", "[number]", value)
    value = " ".join(value.replace("\n", " ").split())
    if len(value) > limit:
        return value[: limit - 1] + "..."
    return value


def _memory_dates(days: int) -> list[date]:
    today = datetime.now().date()
    return [today - timedelta(days=offset) for offset in range(max(1, days))]


def collect_decisions(days: int) -> list[str]:
    decisions: list[str] = []
    seen: set[str] = set()
    for day in _memory_dates(days):
        path = STATE_DIR / f"{day.isoformat()}.md"
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip().lstrip("- ").strip()
            if not line or len(line) < 16:
                continue
            if not any(keyword in line for keyword in DECISION_KEYWORDS):
                continue
            if "投入件数" in line or "Cloudflare URL" in line:
                continue
            line = _redact(line, 220)
            if line in seen:
                continue
            seen.add(line)
            decisions.append(line)
            if len(decisions) >= 8:
                return decisions
    return decisions


def collect_long_term_memory(limit: int = 8) -> list[str]:
    """Cloud Runチャットに持たせる公開可能な長期記憶を抽出する。"""
    curated = [_redact(item, 240) for item in CURATED_LONG_TERM_MEMORIES[:limit]]
    if len(curated) >= limit:
        return curated
    if not LONG_TERM_MEMORY.exists():
        return curated
    memories: list[str] = list(curated)
    seen: set[str] = set(curated)
    current_heading = ""
    for raw in LONG_TERM_MEMORY.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#"):
            current_heading = line.strip("# ").strip()
            continue
        if not line.startswith("-"):
            continue
        text = line.lstrip("- ").strip()
        if len(text) < 20:
            continue
        combined = f"{current_heading}: {text}" if current_heading else text
        if any(keyword in combined for keyword in PRIVATE_MEMORY_KEYWORDS):
            continue
        if not any(keyword in combined for keyword in LONG_TERM_MEMORY_KEYWORDS):
            continue
        item = _redact(combined, 240)
        if item in seen:
            continue
        seen.add(item)
        memories.append(item)
        if len(memories) >= limit:
            break
    return memories


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    try:
        return dict(row)
    except Exception:
        return {}


def _parse_json_blob(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _fetch_cases_postgres(limit: int) -> list[dict[str, Any]]:
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(_database_url(), cursor_factory=psycopg2.extras.DictCursor, connect_timeout=10)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, industry_sub, score, final_status, data::text AS data
            FROM past_cases
            WHERE final_status IS NOT NULL AND final_status <> ''
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (limit * 4,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _fetch_cases_sqlite(limit: int) -> list[dict[str, Any]]:
    db_path = Path(os.environ.get("DB_PATH", PROJECT_ROOT / "data" / "demo.db"))
    if not db_path.exists():
        db_path = PROJECT_ROOT / "data" / "lease_data.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, industry_sub, score, final_status, data
            FROM past_cases
            WHERE final_status IS NOT NULL AND final_status <> ''
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit * 4,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def collect_case_examples(limit: int) -> list[dict[str, str]]:
    try:
        rows = _fetch_cases_postgres(limit) if _database_url() else _fetch_cases_sqlite(limit)
    except Exception as exc:
        print(f"警告: 過去事例の取得に失敗しました: {type(exc).__name__}", file=sys.stderr)
        rows = []

    cases: list[dict[str, str]] = []
    seen_shapes: set[str] = set()
    for index, row in enumerate(rows, 1):
        data = _parse_json_blob(row.get("data"))
        inputs = data.get("inputs") if isinstance(data.get("inputs"), dict) else {}
        industry = _redact(row.get("industry_sub") or data.get("industry_sub") or data.get("industry_major") or "業種未設定", 80)
        asset = _redact(
            inputs.get("lease_asset_name")
            or inputs.get("asset_name")
            or data.get("asset_name")
            or data.get("asset_type")
            or data.get("asset_detail")
            or _asset_hint_for_industry(industry),
            80,
        )
        status = _redact(row.get("final_status") or data.get("final_status") or "結果未設定", 40)
        score = row.get("score")
        judgment = _redact(data.get("judgment") or data.get("comment") or data.get("memo") or "", 160)
        shape = f"{industry}|{asset}|{status}"
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        cases.append(
            {
                "label": f"匿名過去事例-{len(cases) + 1:02d}",
                "industry": industry,
                "asset": asset,
                "status": status,
                "score": "" if score is None else str(round(float(score), 1)),
                "lesson": judgment or _case_lesson(status=status, industry=industry, asset=asset),
            }
        )
        if len(cases) >= limit:
            break
    return cases


def _case_lesson(status: str, industry: str, asset: str) -> str:
    if "失" in status or "否" in status:
        return f"{industry}の{asset}では、導入効果・返済余力・既存借入の説明不足を重点確認する。"
    if "成" in status or "承" in status or "検収" in status:
        return f"{industry}の{asset}では、用途と収益改善根拠が揃うと条件付き承認の材料になる。"
    return f"{industry}の{asset}では、過去事例として物件用途・資金繰り・補助金前提を比較する。"


def _asset_hint_for_industry(industry: str) -> str:
    if "製造" in industry or "食料品" in industry or "金属" in industry or "機械" in industry:
        return "製造設備"
    if "工事" in industry or "建設" in industry:
        return "建設機械"
    if "運送" in industry or "物流" in industry:
        return "車両・物流設備"
    if "医療" in industry:
        return "医療機器"
    if "卸売" in industry or "小売" in industry:
        return "物流・店舗設備"
    return "事業用設備"


def build_markdown(
    target_date: str,
    decisions: list[str],
    long_term_memories: list[str],
    cases: list[dict[str, str]],
) -> str:
    lines = [
        "---",
        f"date: {target_date}",
        "source: local_curated_memory_pack",
        "public_knowledge: true",
        "cloud_run_safe: true",
        "tags: [cloud_run, chat_memory, past_cases, public_knowledge, 紫苑]",
        "---",
        "",
        f"# {target_date} Cloud Chat Memory Pack",
        "",
        "## 使い方",
        "- Cloud Run上の紫苑が、長期記憶・直近方針・匿名過去事例を思い出すための公開用メモリパック。",
        "- Daily / Private Reflection / 生チャット / 顧客名は含めない。",
        "- デモでは「過去に似た事例があった」と説明するために使う。",
        "",
        "## 長期記憶",
    ]
    if long_term_memories:
        lines.extend(f"- {item}" for item in long_term_memories)
    else:
        lines.append("- このシステムは、リース判断の経験・知識・改善ログを判断資産として再利用する。")
        lines.append("- AIチャットは、保存されたObsidian知識をRAGで参照し、案件文脈に戻す。")

    lines += [
        "",
        "## 継続中の方針",
    ]
    if decisions:
        lines.extend(f"- {item}" for item in decisions)
    else:
        lines.append("- Obsidianはローカル/iCloud Vaultを正本にし、Cloud Runは入力受付とデモ実行を担当する。")
        lines.append("- Cloud Runが読む知識は、選抜済みの公開可能なMarkdownに限定する。")

    lines += ["", "## 匿名過去事例", ""]
    if cases:
        for case in cases:
            lines += [
                f"### {case['label']}",
                f"- 業種: {case['industry']}",
                f"- 物件: {case['asset']}",
                f"- 結果: {case['status']}",
                f"- スコア: {case['score'] or '非表示'}",
                f"- デモでの使い方: {case['lesson']}",
                "",
            ]
    else:
        lines += [
            "### 匿名過去事例-01",
            "- 業種: 地域製造業",
            "- 物件: 工作機械",
            "- 結果: 条件付き承認",
            "- スコア: 非表示",
            "- デモでの使い方: 補助金採択前の設備更新では、未採択時の返済余力と導入効果の根拠を確認する。",
            "",
        ]

    lines += [
        "## 除外したもの",
        "- 顧客名、会社名、電話番号、メール、生チャット全文",
        "- Private Reflection、Daily全文、Codex作業ログ、個人的な長期記憶",
        "- Cloud SQL要約やCloud Run入力ログの再同期",
        "",
    ]
    return "\n".join(lines)


def build_layer_markdown(title: str, target_date: str, items: list[str], description: str) -> str:
    lines = [
        "---",
        f"date: {target_date}",
        "source: local_curated_memory_pack",
        "public_knowledge: true",
        "cloud_run_safe: true",
        "tags: [cloud_run, chat_memory, identity_memory, public_knowledge, 紫苑]",
        "---",
        "",
        f"# {title}",
        "",
        description,
        "",
        "## メモリ",
    ]
    lines.extend(f"- {item}" for item in items)
    lines += [
        "",
        "## 運用",
        "- このファイルはCloud Run版の /api/chat にRAGとは別枠で常時注入する。",
        "- 顧客名、会社名、Private Reflection、Daily全文、生チャットは含めない。",
        "",
    ]
    return "\n".join(lines)


def write_pack(markdown: str, target_date: str, dry_run: bool) -> Path:
    out_dir = VAULT_PATH / OUTPUT_DIR
    out_path = out_dir / f"{target_date}_cloud_chat_memory_pack.md"
    latest_path = out_dir / "latest_cloud_chat_memory_pack.md"
    if dry_run:
        print(markdown)
        return out_path
    if not (VAULT_PATH / ".obsidian").exists():
        raise SystemExit(f"Obsidian Vault が見つかりません: {VAULT_PATH}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    latest_path.write_text(markdown, encoding="utf-8")
    return out_path


def write_layer_packs(target_date: str, decisions: list[str], dry_run: bool) -> list[Path]:
    layers = [
        (
            "identity.md",
            build_layer_markdown(
                "Core Identity Memory",
                target_date,
                identity_memories(),
                "Cloud Run版でも「同じ紫苑がそこにいる」と感じるための公開安全な同一性メモリ。",
            ),
        ),
        (
            "judgment-principles.md",
            build_layer_markdown(
                "Judgment Memory",
                target_date,
                judgment_principle_memories(),
                "Kobayashiさんのリース判断資産として回答を返すための、常時参照する判断原則。",
            ),
        ),
        (
            "recent-continuity.md",
            build_layer_markdown(
                "Recent Continuity Memory",
                target_date,
                decisions
                or [
                    "Cloud Run版は、Cloudflare版で感じた記憶の近さ・返答の厚み・紫苑らしさを再現する方向で調整する。",
                    "回答品質は必須語スコアだけでなく、knowledge_refs、memory_recall.refs、同一性の手触りで評価する。",
                ],
                "直近の方針・関心・移行中の判断を短く保つ公開安全な継続メモリ。",
            ),
        ),
    ]
    out_dir = VAULT_PATH / OUTPUT_DIR
    paths = [out_dir / name for name, _ in layers]
    if dry_run:
        for name, content in layers:
            print(f"\n\n<!-- {name} -->\n")
            print(content)
        return paths
    if not (VAULT_PATH / ".obsidian").exists():
        raise SystemExit(f"Obsidian Vault が見つかりません: {VAULT_PATH}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in layers:
        (out_dir / name).write_text(content, encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud Run用の安全な短期記憶・匿名過去事例パックを作成")
    parser.add_argument("--date", default=datetime.now().date().isoformat())
    parser.add_argument("--days", type=int, default=2)
    parser.add_argument("--case-limit", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    decisions = collect_decisions(args.days)
    long_term_memories = collect_long_term_memory()
    cases = collect_case_examples(args.case_limit)
    markdown = build_markdown(args.date, decisions, long_term_memories, cases)
    path = write_pack(markdown, args.date, args.dry_run)
    layer_paths = write_layer_packs(args.date, decisions, args.dry_run)
    print(f"memory_pack: {path}")
    for layer_path in layer_paths:
        print(f"memory_layer: {layer_path}")


if __name__ == "__main__":
    main()
