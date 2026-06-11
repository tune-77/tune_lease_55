"""
DAILY-BRIEF.md を Obsidian Vault に書き出す。
reports/latest.json と static_data/macro_context.json を読み込んで
日次サマリを生成する。Vault が存在しない場合は graceful skip。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
_ICLOUD_DOCS = Path.home() / "Library" / "Mobile Documents" / "iCloud~md~obsidian" / "Documents"
ICLOUD_VAULT_PATH = _ICLOUD_DOCS / "lease-wiki-vault"          # RAG インデックス対象
ICLOUD_MAIN_VAULT_PATH = _ICLOUD_DOCS / "Obsidian Vault"       # メインVault（reindex_obsidian._DEFAULT_VAULT と同一）
VAULT_PATH = ICLOUD_MAIN_VAULT_PATH
LATEST_JSON = PROJECT_ROOT / "reports" / "latest.json"
MACRO_JSON = PROJECT_ROOT / "static_data" / "macro_context.json"
SIDECAR_BRIEF_MD = PROJECT_ROOT / "reports" / "agent_sidecar_brief.md"
OUTPUT_PATH = VAULT_PATH / "DAILY-BRIEF.md"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def format_macro_summary(macro: dict) -> str:
    if not macro:
        return "_マクロデータ取得不可_"

    lines = []
    if boj := macro.get("boj_policy_rate", {}):
        lines.append(f"- 日銀政策金利: **{boj.get('value', '-')}%** （前回: {boj.get('prev', '-')}%）")
    if cpi := macro.get("core_cpi", {}):
        lines.append(f"- コアCPI (前年比): **{cpi.get('value', '-')}%** （前回: {cpi.get('prev', '-')}%）")
    if lead := macro.get("leading_indicator", {}):
        lines.append(f"- 景気先行指数: **{lead.get('value', '-')}** （前回: {lead.get('prev', '-')}）")
    if unemp := macro.get("unemployment", {}):
        lines.append(f"- 失業率: **{unemp.get('value', '-')}%**")
    if assessment := macro.get("assessment", ""):
        if isinstance(assessment, dict):
            env = assessment.get("macro_env", "-")
            risk = assessment.get("macro_risk_score", "-")
            lines.append(f"- 環境評価: **{env}**（リスクスコア: {risk}）")
            for factor in assessment.get("factors", {}).values():
                comment = factor.get("comment", "")
                if comment:
                    lines.append(f"  - {comment}")
        else:
            lines.append(f"- 環境評価: {assessment}")

    fetched = macro.get("fetched_at", "")
    if fetched:
        try:
            dt = datetime.fromisoformat(fetched)
            lines.append(f"- データ取得日時: {dt.strftime('%Y-%m-%d %H:%M')}")
        except Exception:
            pass

    return "\n".join(lines) if lines else "_データなし_"


def format_applied_revs(report: dict) -> str:
    applied = report.get("applied_improvements", [])
    if not applied:
        return "_適用なし_"
    lines = []
    for item in applied[:10]:
        title = item.get("title") or item.get("id") or "不明"
        lines.append(f"- {title}")
    if len(applied) > 10:
        lines.append(f"- …他 {len(applied) - 10} 件")
    return "\n".join(lines)


def format_new_rev_candidates(report: dict) -> str:
    candidates = report.get("improvements", [])
    if not candidates:
        candidates = report.get("original_improvements", [])
    if not candidates:
        return "_候補なし_"
    lines = []
    for item in candidates[:5]:
        title = item.get("title") or item.get("id") or "不明"
        priority = item.get("priority", "")
        tag = f" ［{priority}］" if priority else ""
        lines.append(f"- {title}{tag}")
    if len(candidates) > 5:
        lines.append(f"- …他 {len(candidates) - 5} 件")
    return "\n".join(lines)


def format_agent_sidecar_notes(max_chars: int = 1800) -> str:
    """Read sidecar agent observations as advisory context only."""
    if not SIDECAR_BRIEF_MD.exists():
        return "_Sidecar agent brief 未生成_"
    try:
        text = SIDECAR_BRIEF_MD.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return "_Sidecar agent brief 読み込み不可_"
    if not text:
        return "_Sidecar agent brief 空_"
    marker = "## Reports"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    return text[:max_chars].strip()


def main() -> None:
    if not VAULT_PATH.exists():
        print(f"[write_daily_brief] iCloud Vault が見つかりません: {VAULT_PATH}。スキップします。")
        sys.exit(0)

    report = load_json(LATEST_JSON)
    macro = load_json(MACRO_JSON)
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    applied_count = report.get("applied_count", 0)
    failed_count = report.get("failed_count", 0)
    needs_review_count = report.get("needs_review_count", 0)

    original_improvements = report.get("original_improvements", report.get("improvements", []))
    new_rev_count = len(original_improvements)

    codex_queue_count = report.get("codex_auto_queue_count", 0)
    codex_safe_count = report.get("codex_auto_safe_count", 0)

    content = f"""# DAILY-BRIEF — {today}

> 自動生成: {now} | `write_daily_brief.py`

## パイプライン実行サマリ

| 項目 | 件数 |
|------|------|
| 適用済み改善 | **{applied_count}** 件 |
| 失敗 | {failed_count} 件 |
| 要レビュー待ち | {needs_review_count} 件 |
| 新規REV候補 | {new_rev_count} 件 |
| Codex 自動実行キュー | {codex_queue_count} 件（安全 {codex_safe_count} 件） |

## 適用された改善（直近）

{format_applied_revs(report)}

## 新規REV候補トップ5

{format_new_rev_candidates(report)}

## マクロ経済環境

{format_macro_summary(macro)}

## Sidecar Agent Notes（参考・本体非連動）

{format_agent_sidecar_notes()}

---

_次回更新: 翌 AM4:00 （run_daily_improvement_pipeline.sh）_
"""

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"[write_daily_brief] 書き出し完了: {OUTPUT_PATH}")

    for vault, label in [
        (ICLOUD_VAULT_PATH, "lease-wiki-vault"),
        (ICLOUD_MAIN_VAULT_PATH, "iCloud メインVault"),
    ]:
        if vault.exists():
            out = vault / "DAILY-BRIEF.md"
            out.write_text(content, encoding="utf-8")
            print(f"[write_daily_brief] {label} に書き出し: {out}")
        else:
            print(f"[write_daily_brief] {label} が見つかりません（スキップ）: {vault}")


if __name__ == "__main__":
    main()
