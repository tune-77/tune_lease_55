#!/usr/bin/env python3
"""
レシピ生成スクリプト。

reports/latest.json の needs_review から、auto_fix_candidates が
ledger.jsonl に記録されたエントリだけをレシピ化して
data/recipes/pending/REV-NNN.json に出力する。

エラーはすべてキャッチして例外を外に出さない（|| true 呼び出し想定）。
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

PROJECT_ROOT = PROJECT_ROOT_FOR_IMPORT
LATEST_JSON = PROJECT_ROOT / "reports" / "latest.json"
PENDING_DIR = PROJECT_ROOT / "data" / "recipes" / "pending"
APPLIED_DIR = PROJECT_ROOT / "data" / "recipes" / "applied"
LEDGER_PATH = Path.home() / "Library" / "Logs" / "tunelease" / "ledger.jsonl"

ALLOWED_CATEGORIES = {"ui", "label", "config", "tailwind"}
BLOCKED_PATH_KEYWORDS = {"scoring_core", "analysis_", "api/", "models/", "data/"}
BLOCKED_TITLE_KEYWORDS = {"scoring", "model", "api", "db", "schema"}
SAFE_PATH_PREFIX = "frontend/src/"


def _is_safe_path(path: str) -> bool:
    if not path.startswith(SAFE_PATH_PREFIX):
        return False
    for kw in BLOCKED_PATH_KEYWORDS:
        if kw in path:
            return False
    return True


def _load_latest() -> list[dict]:
    if not LATEST_JSON.exists():
        print(f"[generate_recipes] latest.json が見つかりません: {LATEST_JSON}")
        return []
    with LATEST_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get("needs_review", [])


def _load_ledger_candidates() -> dict[str, dict]:
    """ledger.jsonl から auto_fix_candidates を持つエントリを返す {rev_id: entry}"""
    candidates: dict[str, dict] = {}
    if not LEDGER_PATH.exists():
        return candidates
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "auto_fix_candidates" not in entry:
                continue
            rev_id = entry.get("rev_id") or entry.get("id") or ""
            if rev_id:
                candidates[rev_id] = entry
    return candidates


def _already_exists(rev: str) -> bool:
    pending = PENDING_DIR / f"{rev}.json"
    applied = APPLIED_DIR / f"{rev}.json"
    return pending.exists() or applied.exists()


def _is_policy_auto(policy: object) -> bool:
    if policy is None or policy == "" or policy == "auto":
        return True
    if isinstance(policy, dict):
        return bool(policy.get("auto_fix_allowed")) or policy.get("policy") == "auto"
    return False


def _count_similar_in_ledger(rev: str, title: str, candidates: dict[str, dict]) -> int:
    title_words = set(title.lower().split())
    count = 0
    for cand_rev, entry in candidates.items():
        if cand_rev == rev:
            continue
        cand_title = (entry.get("title", "") or "").lower()
        if any(w in cand_title for w in title_words if len(w) > 2):
            count += 1
    return count


def _build_intelligence_comment(rev: str, title: str, ledger_candidates: dict[str, dict]) -> str:
    try:
        from lease_intelligence_knowledge import build_lease_intelligence_knowledge
        knowledge = build_lease_intelligence_knowledge(theme=title, limit=3)
        top_note = ""
        if knowledge.source_paths:
            top_note = Path(knowledge.source_paths[0]).stem
        similar_count = _count_similar_in_ledger(rev, title, ledger_candidates)
        parts: list[str] = [f"過去の類似改善: {similar_count}件"]
        if top_note:
            parts.append(f"関連Obsidianノート: {top_note}")
        comment = " / ".join(parts)
        return comment[:100]
    except Exception:
        return ""


_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:generateContent"
)

_SHION_JUDGMENT_PROMPT = """\
以下の改善案について、自動修正できるか判断してください。

改善タイトル: {title}
変更対象ファイル: {target_files}
過去の類似改善: {similar_count}件
関連Obsidianノート: {obsidian_context}

判断基準:
- auto: フロントエンドの表示・スタイル変更のみ、影響範囲が小さい
- discuss: スコアリング・DB・APIロジック・モデルに触れる、影響範囲が大きい
- review: 判断が難しい・情報不足・リスク不明

「auto」「discuss」「review」のいずれかと、50字以内の理由を日本語で返してください。
必ずJSON形式のみで返答してください: {{"recommendation": "auto", "reason": "理由"}}
"""


def _call_gemini_for_shion(prompt: str) -> str | None:
    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            generation_config={"max_output_tokens": 256, "temperature": 0.2},
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() or None
    except Exception:
        pass
    try:
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.2},
        }).encode("utf-8")
        url = f"{_GEMINI_REST_URL}?key={api_key}"
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None


def _build_shion_recommendation(
    rev: str,
    title: str,
    target_files: list[str],
    ledger_candidates: dict[str, dict],
    intelligence_comment: str,
) -> tuple[str, str]:
    """紫苑の判断: (recommendation, reason) を返す。失敗時は ("review", ...)"""
    try:
        similar_count = _count_similar_in_ledger(rev, title, ledger_candidates)
        obsidian_context = intelligence_comment or "なし"
        files_str = ", ".join(target_files) if target_files else "不明"
        prompt = _SHION_JUDGMENT_PROMPT.format(
            title=title,
            target_files=files_str,
            similar_count=similar_count,
            obsidian_context=obsidian_context,
        )
        raw = _call_gemini_for_shion(prompt)
        if not raw:
            return ("review", "API呼び出し失敗のため判断不能")
        cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(cleaned)
        rec = str(parsed.get("recommendation", "review")).lower()
        if rec not in ("auto", "discuss", "review"):
            rec = "review"
        reason = str(parsed.get("reason", ""))[:50]
        return (rec, reason)
    except Exception as e:
        return ("review", f"判断エラー: {type(e).__name__}")


def _build_recipe(item: dict, ledger_entry: dict) -> dict | None:
    rev = item.get("id", "")
    title = item.get("title", "")
    raw_candidates: list[dict] = ledger_entry.get("auto_fix_candidates", [])

    file_map: dict[str, list[dict]] = {}
    for cand in raw_candidates:
        path = cand.get("file", "")
        if not _is_safe_path(path):
            print(f"[generate_recipes] {rev}: 安全でないパスをスキップ: {path}")
            continue
        change: dict = {"find": cand.get("find", ""), "replace": cand.get("replace", "")}
        if "occurrence" in cand:
            change["occurrence"] = cand["occurrence"]
        file_map.setdefault(path, []).append(change)

    if not file_map:
        return None

    # category 推定（tailwind クラスっぽければ tailwind_class、それ以外は find_replace）
    recipe_type = "find_replace"
    category = ledger_entry.get("category", item.get("category", ""))
    if category == "tailwind":
        recipe_type = "tailwind_class"
    elif category == "config":
        recipe_type = "config_value"

    safety = "none"
    if any(p.endswith(".tsx") or p.endswith(".ts") for p in file_map):
        safety = "tsc"

    return {
        "rev": rev,
        "title": title,
        "type": recipe_type,
        "files": [
            {"path": p, "changes": changes}
            for p, changes in file_map.items()
        ],
        "safety": safety,
        "max_lines_changed": 50,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    generated = 0
    skipped_no_candidates = 0
    skipped_exists = 0
    skipped_policy = 0
    skipped_blocked = 0
    errors = 0

    try:
        needs_review = _load_latest()
        ledger_candidates = _load_ledger_candidates()
    except Exception as e:
        print(f"[generate_recipes] 読み込みエラー: {e}")
        return

    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    for item in needs_review:
        rev = item.get("id", "")
        title = item.get("title", "")

        try:
            # 除外: タイトルにブロックキーワード
            title_lower = title.lower()
            if any(kw in title_lower for kw in BLOCKED_TITLE_KEYWORDS):
                skipped_blocked += 1
                continue

            # 除外: policy が auto でない
            policy = item.get("auto_fix_policy")
            if not _is_policy_auto(policy):
                skipped_policy += 1
                continue

            # 除外: ledger に auto_fix_candidates がない
            ledger_entry = ledger_candidates.get(rev)
            if ledger_entry is None:
                skipped_no_candidates += 1
                continue

            # 除外: category チェック
            category = ledger_entry.get("category", item.get("category", ""))
            if category and category not in ALLOWED_CATEGORIES:
                skipped_blocked += 1
                continue

            # 除外: 既存ファイル
            if _already_exists(rev):
                skipped_exists += 1
                continue

            recipe = _build_recipe(item, ledger_entry)
            if recipe is None:
                skipped_no_candidates += 1
                continue

            intelligence_comment = _build_intelligence_comment(rev, title, ledger_candidates)
            recipe["intelligence_comment"] = intelligence_comment

            target_files = [f["path"] for f in recipe.get("files", [])]
            shion_rec, shion_reason = _build_shion_recommendation(
                rev, title, target_files, ledger_candidates, intelligence_comment
            )
            recipe["shion_recommendation"] = shion_rec
            recipe["shion_reason"] = shion_reason

            out_path = PENDING_DIR / f"{rev}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)

            print(f"[generate_recipes] 生成: {out_path.name} ({title})")
            generated += 1

        except Exception as e:
            print(f"[generate_recipes] エラー ({rev}): {e}")
            errors += 1

    print(
        f"[generate_recipes] 完了 — 生成: {generated}件 / "
        f"スキップ(既存): {skipped_exists}件 / "
        f"スキップ(候補なし): {skipped_no_candidates}件 / "
        f"スキップ(policy): {skipped_policy}件 / "
        f"スキップ(対象外): {skipped_blocked}件 / "
        f"エラー: {errors}件"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[generate_recipes] 予期しないエラー: {e}", file=sys.stderr)
