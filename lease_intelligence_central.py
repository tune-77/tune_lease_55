"""Central synthesis engine for Shion (REV-154).

Reads conversation_keypoints from the Vault mind.json, detects recurring
patterns using Gemini, and updates world_view.commentary.
world_view.summary is NEVER modified.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

MIND_RELATIVE_DIR = Path("Projects") / "tune_lease_55" / "Lease Intelligence"


def _vault_mind_path(vault_path: str) -> Path:
    return Path(vault_path) / MIND_RELATIVE_DIR / "mind.json"


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    here = Path(__file__).parent
    for _ in range(5):
        sec = here / ".streamlit" / "secrets.toml"
        if sec.exists():
            for line in sec.read_text(encoding="utf-8").splitlines():
                m = re.match(r'^GEMINI_API_KEY\s*=\s*["\'](.+)["\']', line.strip())
                if m:
                    return m.group(1)
        here = here.parent
    return ""


def _call_gemini(prompt: str) -> str:
    import requests

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が見つかりません")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 2000},
    }
    resp = requests.post(
        url, json=payload, headers={"x-goog-api-key": api_key}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def _load_all_keypoints(vault_path: str) -> list[dict[str, Any]]:
    """Vault mind.json から全 keypoints を読み込む。role タグがないものは role='legacy' として扱う。"""
    mind_path = _vault_mind_path(vault_path)
    try:
        data = json.loads(mind_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    raw: list[Any] = data.get("conversation_keypoints") or []
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = str(item.get("fact") or item.get("content") or "").strip()
        if not content:
            continue
        role = str(item.get("role") or "legacy").strip() or "legacy"
        result.append({
            "content": content,
            "role": role,
            "date": str(item.get("date") or ""),
            "source": str(item.get("source") or item.get("type") or ""),
        })
    return result


def _group_by_role(keypoints: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """role ごとにグループ化。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kp in keypoints:
        groups[kp["role"]].append(kp)
    return dict(groups)


def _detect_patterns(keypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    複数の keypoints に共通するテーマを Gemini で検出する。
    Gemini が使えない場合は空リストを返す。

    確認済み信念の閾値:
    - 3つ以上の異なるロールで言及された論点
    - または同じロールで3回以上出現した論点
    """
    if not keypoints:
        return []

    kp_text = "\n".join(
        f"[{i + 1}] ({kp['role']}) {kp['content']}"
        for i, kp in enumerate(keypoints[:60])
    )

    prompt = f"""以下はリース審査AIシステム「紫苑」の複数ペルソナ（shion_skeptic=懐疑派・shion_optimist=楽観派・shion_arbiter=仲裁役・shion_innovator=革新派・legacy=旧来）が
過去の討論・対話から抽出した判断キーポイントの一覧です。

{kp_text}

この一覧を分析して、繰り返し登場する重要な論点・テーマを最大8個抽出してください。
以下のJSON配列形式のみで返してください（前後の説明テキストは不要）:

[
  {{
    "theme": "テーマの簡潔な説明（50字以内）",
    "count": 出現回数の推定（整数）,
    "supporting_roles": ["支持しているロール名のリスト"],
    "conflicting_roles": ["反対または懐疑的なロール名のリスト"],
    "status": "observation|emerging_pattern|confirmed_belief"
  }}
]

status の基準:
- confirmed_belief: 3つ以上の異なるロールで言及、または同一ロールで3回以上
- emerging_pattern: 2つのロールで言及、または2回出現
- observation: 1件のみ"""

    try:
        raw_response = _call_gemini(prompt)
        m = re.search(r"\[.*\]", raw_response, re.DOTALL)
        if not m:
            return []
        patterns = json.loads(m.group(0))
        if not isinstance(patterns, list):
            return []
        return [p for p in patterns if isinstance(p, dict) and p.get("theme")]
    except Exception as e:
        print(f"[central] Gemini パターン検出失敗: {e}")
        return []


def _update_world_view_commentary(vault_path: str, patterns: list[dict[str, Any]]) -> None:
    """
    mind.json の world_view に commentary フィールドを追記/更新する。
    - world_view.summary は絶対に変更しない（人間が書いたもの）
    - world_view.commentary のみ更新する
    - アトミック書き込み（一時ファイル経由）
    """
    mind_path = _vault_mind_path(vault_path)
    try:
        data = json.loads(mind_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    world_view = data.get("world_view")
    if not isinstance(world_view, dict):
        return

    confirmed = [p for p in patterns if p.get("status") == "confirmed_belief"]
    emerging = [p for p in patterns if p.get("status") == "emerging_pattern"]
    observations = [p for p in patterns if p.get("status") == "observation"]

    known_tradeoffs = [
        {
            "theme": p["theme"],
            "supporters": p.get("supporting_roles") or [],
            "opponents": p.get("conflicting_roles") or [],
        }
        for p in patterns
        if p.get("supporting_roles") and p.get("conflicting_roles")
    ]

    world_view["commentary"] = {
        "confirmed_beliefs": [p["theme"] for p in confirmed],
        "emerging_patterns": [p["theme"] for p in emerging],
        "observations": [p["theme"] for p in observations],
        "known_tradeoffs": known_tradeoffs,
        "last_updated": dt.datetime.now().isoformat(timespec="seconds"),
    }

    data["world_view"] = world_view

    tmp = mind_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(mind_path)


def run_central_synthesis(vault_path: str) -> dict[str, Any]:
    """
    セントラル統合処理のメイン関数。
    lease_intelligence_reflection.py の夜間バッチから呼ばれる。

    返り値:
    {
      "confirmed_beliefs": [...],
      "emerging_patterns": [...],
      "known_tradeoffs": [...],
      "processed_keypoints": N,
      "synthesis_date": "..."
    }
    """
    keypoints = _load_all_keypoints(vault_path)
    if not keypoints:
        return {
            "confirmed_beliefs": [],
            "emerging_patterns": [],
            "known_tradeoffs": [],
            "processed_keypoints": 0,
            "synthesis_date": dt.datetime.now().isoformat(timespec="seconds"),
            "note": "keypoints が見つかりませんでした",
        }

    by_role = _group_by_role(keypoints)
    print(f"[central] ロール別件数: { {r: len(v) for r, v in by_role.items()} }")

    patterns = _detect_patterns(keypoints)
    _update_world_view_commentary(vault_path, patterns)

    confirmed = [p for p in patterns if p.get("status") == "confirmed_belief"]
    emerging = [p for p in patterns if p.get("status") == "emerging_pattern"]
    known_tradeoffs = [
        {
            "theme": p["theme"],
            "supporters": p.get("supporting_roles") or [],
            "opponents": p.get("conflicting_roles") or [],
        }
        for p in patterns
        if p.get("supporting_roles") and p.get("conflicting_roles")
    ]

    return {
        "confirmed_beliefs": [p["theme"] for p in confirmed],
        "emerging_patterns": [p["theme"] for p in emerging],
        "known_tradeoffs": known_tradeoffs,
        "processed_keypoints": len(keypoints),
        "synthesis_date": dt.datetime.now().isoformat(timespec="seconds"),
    }


def main() -> None:
    try:
        from lease_news_digest import find_vault
        vault = find_vault()
    except Exception:
        vault = None
    if not vault:
        print("[central] Obsidian Vault が見つかりません")
        sys.exit(1)
    result = run_central_synthesis(str(vault))
    print(f"[central] 完了: {result}")


if __name__ == "__main__":
    main()
