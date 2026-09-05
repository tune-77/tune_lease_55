"""Build a lightweight Shion memory index from existing local memory sources.

This is intentionally read-mostly and local: it does not call LLMs, does not
write to Obsidian, and does not alter the daily improvement pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.shion_memory_taxonomy import MEMORY_TYPES, RECALL_ROUTES, make_memory_record
from obsidian_query import list_vault_md_files

DEFAULT_OUTPUT = REPO_ROOT / "data" / "shion_memory_index.json"

_DOMAIN_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "scoring_model",
        (
            "AUC",
            "OOF",
            "RandomForest",
            "LightGBM",
            "LGBM",
            "LogisticRegression",
            "MLP",
            "QCL",
            "PD",
            "score_borrower",
            "bench_score",
            "ind_score",
            "モデル",
            "再学習",
        ),
        "モデル性能、スコア差分、再学習方針、PD表示を確認する時",
    ),
    (
        "lease_screening",
        (
            "審査",
            "承認",
            "否決",
            "条件付き",
            "条件付",
            "与信",
            "リスク",
            "稟議",
            "案件",
            "保証",
        ),
        "案件審査、承認条件、否決理由、稟議コメントを作る時",
    ),
    (
        "lease_contract",
        ("購入選択権", "買取", "残価", "契約", "満了", "再リース"),
        "契約条件、満了後対応、残価・買取可否を説明する時",
    ),
    (
        "asset_life",
        ("耐用年数", "期待使用期間", "リース期間", "厨房機器", "建機", "フォークリフト", "トラック", "医療機器"),
        "物件の期間妥当性、耐用年数、再リース可否を見る時",
    ),
    (
        "rag_memory_ops",
        ("RAG", "ChromaDB", "Obsidian", "Vault", "記憶", "想起", "インデックス", "昇格"),
        "記憶検索、RAG接続、Obsidian同期、記憶昇格の挙動を確認する時",
    ),
    (
        "system_ops",
        (
            "Cloud Run",
            "Cloudflare",
            "api/",
            "frontend/",
            "script",
            "scripts/",
            "pytest",
            "テスト",
            "デプロイ",
            "LaunchAgent",
            "Streamlit",
            "FastAPI",
        ),
        "実装、テスト、デプロイ、運用手順を決める時",
    ),
    (
        "data_quality",
        ("CSV", "OCR", "DB", "past_cases", "欠損", "データ", "json", "ログ", "バックフィル"),
        "入力データ、OCR、CSV、DB、ログ品質を確認する時",
    ),
    (
        "market_news",
        ("ニュース", "金利", "景気", "倒産", "補助金", "業界", "市場", "サプライヤー"),
        "外部環境や業界ニュースを審査観点へ落とす時",
    ),
    (
        "shion_identity",
        ("Mana", "良心", "紫苑", "内省", "Private Reflection", "上位規範", "価値観"),
        "紫苑の振る舞い、境界線、内省、人格的一貫性を確認する時",
    ),
    (
        "user_preference",
        ("User", "ユーザー", "好み", "方針", "覚えて", "Kobayashi"),
        "ユーザーの継続的な好み、依頼方針、会話上の前提を反映する時",
    ),
)

_TYPE_DOMAIN_DEFAULTS: dict[str, tuple[str, str]] = {
    "judgment_memory": ("lease_screening", "判断基準、確認質問、条件設定へ落とす時"),
    "technical_memory": ("system_ops", "実装、運用、検証手順を決める時"),
    "value_memory": ("shion_identity", "判断の優先順位や振る舞いの境界を確認する時"),
    "dialogue_memory": ("user_preference", "ユーザーの継続的な好みや依頼背景を反映する時"),
    "reflection_memory": ("shion_identity", "紫苑の内省や回答姿勢を整える時"),
    "factual_memory": ("general_knowledge", "関連する事実前提として回答や確認に使う時"),
}


def _safe_float(value: Any, default: float) -> float:
    """confidence等の数値フィールドを安全に変換する。

    mind.json / canonical_judgment_rules.json は他プロセスが書くため、
    想定外の型（文字列ラベル等）が混じっても索引構築全体を落とさない。
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _memory_bullets_from_markdown(path: Path, source: str, *, memory_layer: str | None = None) -> list[dict[str, Any]]:
    text = _read_text(path)
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        content = stripped[2:].strip()
        if _skip_markdown_bullet(content):
            continue
        private = "Private Reflection" in content or "私室" in content
        records.append(
            make_memory_record(
                content,
                source=source,
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_layer=memory_layer,  # type: ignore[arg-type]
                private=private,
            ).to_dict()
        )
    return records


def _promoted_memory_records(path: Path) -> list[dict[str, Any]]:
    """Read structured promoted memories, with legacy bullet fallback.

    Preferred format:
    - content: ...
      type: factual_memory
      domain: lease_contract
      confidence: user_taught
      use_when: ...
      judgment_asset_candidate: true
      source: promo_xxx
    """
    text = _read_text(path)
    records: list[dict[str, Any]] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        content = str(current.get("content") or "").strip()
        if content and not _skip_structured_promoted_content(content):
            confidence_label = str(current.get("confidence") or "")
            confidence = 0.9 if confidence_label == "user_taught" else 0.75
            record = make_memory_record(
                content,
                source="promoted_memory",
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_layer="long_term",
                memory_type=current.get("type") or None,  # type: ignore[arg-type]
                confidence=confidence,
            ).to_dict()
            for key, out_key in (
                ("domain", "domain"),
                ("use_when", "use_when"),
                ("source", "promotion_source_id"),
                ("kind", "promotion_kind"),
                ("promoted_at", "promoted_at"),
            ):
                if current.get(key):
                    record[out_key] = current[key]
            if current.get("confidence"):
                record["confidence_label"] = current["confidence"]
            if current.get("judgment_asset_candidate"):
                record["judgment_asset_candidate"] = current["judgment_asset_candidate"].lower() == "true"
            records.append(record)
        current = None

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- content:"):
            flush()
            current = {"content": stripped.split(":", 1)[1].strip()}
            continue
        if current is not None and line.startswith("  ") and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip()
            continue
        if stripped.startswith("- "):
            flush()
            content = _strip_legacy_promotion_suffix(stripped[2:].strip())
            if _skip_markdown_bullet(content):
                continue
            records.append(
                make_memory_record(
                    content,
                    source="promoted_memory",
                    source_path=str(path.relative_to(REPO_ROOT)),
                    memory_layer="long_term",
                ).to_dict()
            )
    flush()
    return records


def _strip_legacy_promotion_suffix(content: str) -> str:
    return content.split("（昇格 ", 1)[0].strip()


def _infer_domain_use_when(record: dict[str, Any]) -> tuple[str, str]:
    content = str(record.get("content") or "")
    source_path = str(record.get("source_path") or "")
    hay = f"{source_path}\n{content}".lower()
    for domain, terms, use_when in _DOMAIN_RULES:
        if any(term.lower() in hay for term in terms):
            return domain, use_when
    memory_type = str(record.get("memory_type") or "")
    return _TYPE_DOMAIN_DEFAULTS.get(
        memory_type,
        ("general_knowledge", "関連する前提知識として回答や確認に使う時"),
    )


def _enrich_long_term_metadata(records: list[dict[str, Any]]) -> None:
    """Fill domain/use_when for all long-term memories without overwriting curated values."""
    for record in records:
        if str(record.get("memory_layer") or "") != "long_term":
            continue
        domain, use_when = _infer_domain_use_when(record)
        if not str(record.get("domain") or "").strip():
            record["domain"] = domain
        if not str(record.get("use_when") or "").strip():
            record["use_when"] = use_when


def _skip_structured_promoted_content(content: str) -> bool:
    if "自動生成プレースホルダー" in content:
        return True
    return False


def _skip_markdown_bullet(content: str) -> bool:
    if len(content) < 12:
        return True
    if "自動生成プレースホルダー" in content:
        return True
    # Section labels such as "**Key Features**:" are headings, not memories.
    if content.startswith("**") and content.endswith(":") and len(content) < 80:
        return True
    if content in {"", "-"}:
        return True
    return False


def _mind_records(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    records: list[dict[str, Any]] = []

    upper = data.get("upper_authority")
    if isinstance(upper, dict):
        values = upper.get("values") or []
        content = (
            f"{upper.get('name', 'Mana')}: {upper.get('role', '')} / "
            f"{upper.get('boundary', '')} / values={'; '.join(map(str, values))}"
        )
        records.append(
            make_memory_record(
                content,
                source="mind.upper_authority",
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_type="value_memory",
                confidence=0.95,
            ).to_dict()
        )

    world_view = data.get("world_view")
    if isinstance(world_view, dict):
        summary = str(world_view.get("summary") or "").strip()
        if summary:
            records.append(
                make_memory_record(
                    summary,
                    source="mind.world_view",
                    source_path=str(path.relative_to(REPO_ROOT)),
                    memory_type="factual_memory",
                    confidence=0.75,
                ).to_dict()
            )
        for signal in world_view.get("key_signals") or []:
            records.append(
                make_memory_record(
                    str(signal),
                    source="mind.world_view.key_signal",
                    source_path=str(path.relative_to(REPO_ROOT)),
                    memory_type="factual_memory",
                    confidence=0.7,
                ).to_dict()
            )

    for kp in data.get("conversation_keypoints") or []:
        if not isinstance(kp, dict):
            continue
        content = str(kp.get("fact") or kp.get("content") or "").strip()
        if not content:
            continue
        records.append(
            make_memory_record(
                content,
                source=str(kp.get("source") or "mind.conversation_keypoint"),
                source_path=str(path.relative_to(REPO_ROOT)),
                memory_type=kp.get("memory_type") or None,
                confidence=_safe_float(kp.get("confidence"), 0.75),
            ).to_dict()
        )

    return records


def _knowledge_markdown_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    source_dirs = (REPO_ROOT / "knowledge_base" / "okf_lease_concepts",)
    for root in source_dirs:
        if not root.exists():
            continue
        for path in sorted(list_vault_md_files(root)):
            rel = str(path.relative_to(REPO_ROOT))
            text = _read_text(path)
            snippets = _markdown_snippets(text)
            mtype = "judgment_memory" if "/rules/" in rel else "factual_memory"
            # ノートタイトル（主題語）。content には含めない（id が変わり
            # created_at / last_used_at の引き継ぎが壊れるため）。ベクトル索引側が
            # 埋め込みテキストの前置に使う。
            title = _note_title(text)
            for snippet in snippets:
                record = make_memory_record(
                    snippet,
                    source="knowledge_base",
                    source_path=rel,
                    memory_type=mtype,  # type: ignore[arg-type]
                    confidence=0.82,
                ).to_dict()
                if title:
                    record["topic"] = title
                records.append(record)
    return records


def _canonical_judgment_rule_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = payload.get("rules") if isinstance(payload, dict) else []
    records: list[dict[str, Any]] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("private") is True or str(rule.get("status") or "") != "active":
            continue
        statement = str(rule.get("canonical_statement") or "").strip()
        if len(statement) < 12:
            continue
        record = make_memory_record(
            statement,
            source="canonical_judgment_rules",
            source_path=str(path.relative_to(REPO_ROOT)),
            memory_type="judgment_memory",
            confidence=_safe_float(rule.get("confidence"), 0.82),
        ).to_dict()
        record["topic"] = str(rule.get("concept") or "")
        record["evidence_count"] = _safe_int(rule.get("evidence_count"), 0)
        record["user_evidence_count"] = _safe_int(rule.get("user_evidence_count"), 0)
        record["evidence_paths"] = list(rule.get("evidence_paths") or [])[:6]
        records.append(record)
    return records


def _note_title(text: str) -> str:
    """frontmatter の title:、なければ最初の H1 見出しを返す。"""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            stripped = line.strip()
            if stripped == "---":
                break
            if stripped.startswith("title:"):
                return stripped[len("title:"):].strip().strip("\"'")
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _markdown_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    current_heading = ""
    lines = text.splitlines()
    start = 0
    # YAML frontmatter はノートのメタデータであり記憶ではないため索引に入れない
    # （"tags: [...]" や "confidence: medium" が記憶レコード化して想起枠を奪っていた）
    if lines and lines[0].strip() == "---":
        for j in range(1, len(lines)):
            if lines[j].strip() == "---":
                start = j + 1
                break
    for raw_line in lines[start:]:
        line = raw_line.strip()
        if not line or line == "---" or line.startswith("<!--"):
            continue
        if line.startswith("#"):
            current_heading = line.lstrip("#").strip()
            continue
        if line.startswith("- "):
            content = line[2:].strip()
        else:
            content = line
        if len(content) < 18 or content.startswith("```"):
            continue
        if current_heading:
            content = f"{current_heading}: {content}"
        snippets.append(content)
    return snippets[:24]


def _load_previous_fields(path: Path) -> dict[str, dict[str, str]]:
    """前回索引から、引き継ぐべきフィールド（初出日・最終使用日）をIDごとに読む。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    previous: dict[str, dict[str, str]] = {}
    for record in data.get("records") or []:
        if not isinstance(record, dict):
            continue
        rid = str(record.get("id") or "")
        if rid:
            previous[rid] = {
                "created_at": str(record.get("created_at") or ""),
                "last_used_at": str(record.get("last_used_at") or ""),
            }
    return previous


def _safe_records(label: str, factory: Any) -> list[dict[str, Any]]:
    """1つの記憶ソースの取り込み失敗が索引全体のビルドを落とさないようにする。

    各ソースの読み込み関数はファイルI/O・JSON解析の例外は個別に握っているが、
    想定外のフィールド型など未知の異常まで防げるとは限らない。ここで最後の
    セーフティネットとして広く捕捉し、当該ソースだけ空扱いで継続する。
    """
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001 - 単一ソースの異常で夜間パイプラインを止めない
        print(f"警告: 記憶ソース取り込みに失敗（スキップして継続）: {label}: {exc}")
        return []


def build_index(
    previous_index_path: Path | None = None, *, demo_safe: bool = False
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []

    persistent_path = REPO_ROOT / "PERSISTENT_MEMORY.md"
    if persistent_path.exists():
        records.extend(
            _safe_records(
                str(persistent_path.relative_to(REPO_ROOT)),
                lambda: _memory_bullets_from_markdown(
                    persistent_path,
                    "persistent_memory",
                    memory_layer="persistent",
                ),
            )
        )

    memory_path = REPO_ROOT / "MEMORY.md"
    if memory_path.exists():
        records.extend(
            _safe_records(
                str(memory_path.relative_to(REPO_ROOT)),
                lambda: _memory_bullets_from_markdown(
                    memory_path,
                    "long_term_memory",
                    memory_layer="long_term",
                ),
            )
        )

    memory_dir = REPO_ROOT / "memory"
    if memory_dir.exists():
        for path in sorted(memory_dir.glob("20*.md"))[-14:]:
            records.extend(
                _safe_records(
                    str(path.relative_to(REPO_ROOT)),
                    lambda path=path: _memory_bullets_from_markdown(path, "daily_memory", memory_layer="mid_term"),
                )
            )

    mind_path = REPO_ROOT / "data" / "mind.json"
    if mind_path.exists():
        records.extend(_safe_records(str(mind_path.relative_to(REPO_ROOT)), lambda: _mind_records(mind_path)))

    # 会話から承認を経て昇格した長期記憶（apply_shion_memory_promotions.py が追記）
    promoted_path = REPO_ROOT / "knowledge_base" / "shion_promoted_memories.md"
    if promoted_path.exists():
        records.extend(
            _safe_records(
                str(promoted_path.relative_to(REPO_ROOT)),
                lambda: _promoted_memory_records(promoted_path),
            )
        )

    records.extend(_safe_records("knowledge_base", _knowledge_markdown_records))

    canonical_rules_path = REPO_ROOT / "data" / "canonical_judgment_rules.json"
    if canonical_rules_path.exists():
        records.extend(
            _safe_records(
                str(canonical_rules_path.relative_to(REPO_ROOT)),
                lambda: _canonical_judgment_rule_records(canonical_rules_path),
            )
        )

    # Deduplicate by stable id, keeping the first occurrence.
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        rid = str(record.get("id") or "")
        if rid and rid not in deduped:
            deduped[rid] = record

    final_records = list(deduped.values())

    # 再生成のたびに created_at（初出日）が今日へリセットされると、鮮度更新の
    # 「作成から45日超かつ未使用 → stale」が永久に発火しないため、前回索引から
    # 初出日と最終使用日を引き継ぐ。
    previous = _load_previous_fields(
        previous_index_path or (REPO_ROOT / "data" / "shion_memory_index.json")
    )
    for record in final_records:
        prev = previous.get(str(record.get("id") or ""))
        if not prev:
            continue
        if prev.get("created_at"):
            record["created_at"] = prev["created_at"]
        if prev.get("last_used_at") and not record.get("last_used_at"):
            record["last_used_at"] = prev["last_used_at"]

    # 改訂宣言（data/shion_memory_revisions.jsonl）を再適用する。
    # 宣言ファイルが真実の源なので、索引を再生成しても revised / supersedes が消えない。
    from scripts.revise_shion_memory import apply_revisions, load_revisions

    revisions = load_revisions(REPO_ROOT / "data" / "shion_memory_revisions.jsonl")
    if revisions:
        holder: dict[str, Any] = {"records": final_records}
        try:
            apply_revisions(holder, revisions)
        except Exception as exc:  # noqa: BLE001 - 改訂宣言の異常で索引再構築全体を止めない
            print(f"警告: 改訂宣言の適用に失敗（スキップして継続）: {exc}")
        else:
            final_records = holder["records"]

    _enrich_long_term_metadata(final_records)

    if demo_safe:
        # 公開デモ環境には対話・内省・private の記憶を載せない
        final_records = [r for r in final_records if not _is_demo_unsafe(r)]

    counts = Counter(str(r.get("memory_type") or "unknown") for r in final_records)
    layer_counts = Counter(str(r.get("memory_layer") or "unknown") for r in final_records)
    status_counts = Counter(str(r.get("status") or "active") for r in final_records)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": 1,
        "taxonomy": MEMORY_TYPES,
        "recall_routes": RECALL_ROUTES,
        "summary": {
            "total_records": len(final_records),
            "by_type": dict(sorted(counts.items())),
            "by_layer": dict(sorted(layer_counts.items())),
            "by_status": dict(sorted(status_counts.items())),
        },
        "records": final_records,
    }


def _is_demo_unsafe(record: dict[str, Any]) -> bool:
    """公開デモバンドルへ載せてはいけない記憶か（対話・内省・private）。"""
    if record.get("private"):
        return True
    if str(record.get("status") or "") == "private":
        return True
    return str(record.get("memory_type") or "") in {"dialogue_memory", "reflection_memory"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Shion memory taxonomy index.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--demo-safe",
        action="store_true",
        help="公開デモ向けに dialogue_memory / reflection_memory / private を除外する",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 引き継ぎ元は出力先の既存索引。初回出力先（デモ用の別パス等）はローカル既定索引から引き継ぐ
    previous = args.output if args.output.exists() else None
    index = build_index(previous_index_path=previous, demo_safe=args.demo_safe)
    text = json.dumps(index, ensure_ascii=False, indent=2)
    if args.dry_run:
        print(text)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.",
        suffix=".tmp",
        dir=args.output.parent,
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(args.output)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"wrote={args.output}")
    print(f"total_records={index['summary']['total_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
