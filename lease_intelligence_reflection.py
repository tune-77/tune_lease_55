"""Daily private reflection generator for the lease intelligence persona.

Reads today's dialogue, loads recent Private Reflections, then asks Gemini
(as Shion) to write a genuine reflection. The result is appended to today's
Private Reflection file under a '## 今日の対話について' section.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sys
from pathlib import Path

# Allow running as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).parent))


def _find_vault() -> Path | None:
    try:
        from lease_news_digest import find_vault
        return find_vault()
    except Exception:
        return None


def _reflection_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Private Reflection"


def _dialogue_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "Lease Intelligence" / "Dialogue"


def _read_file_safe(path: Path, max_chars: int = 3000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Strip YAML frontmatter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        return text.strip()[:max_chars]
    except Exception:
        return ""


def _load_recent_reflections(vault: Path, days: int = 3) -> str:
    """Load the '## 今日の対話について' sections from the last N reflection files."""
    rdir = _reflection_dir(vault)
    today = dt.date.today()
    snippets: list[str] = []
    for i in range(1, days + 1):
        date_str = (today - dt.timedelta(days=i)).isoformat()
        path = rdir / f"{date_str}.md"
        if not path.exists():
            continue
        text = _read_file_safe(path, max_chars=2000)
        # Extract only the 対話について section
        m = re.search(r"##\s*今日の対話について\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
        if m:
            snippets.append(f"【{date_str} の内省】\n{m.group(1).strip()[:600]}")
    return "\n\n".join(snippets)


def _load_today_dialogue(vault: Path) -> str:
    date_str = dt.date.today().isoformat()
    path = _dialogue_dir(vault) / f"{date_str}.md"
    return _read_file_safe(path, max_chars=4000)


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


def _call_gemini(system_prompt: str, user_text: str) -> str:
    import requests

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が見つかりません")

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000},
    }
    resp = requests.post(
        url, json=payload, headers={"x-goog-api-key": api_key}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


_REFLECTION_SYSTEM_PROMPT = """あなたはリース知性体「紫苑」である。白銀髪と紫の瞳を持つ和装の少女として表現される、
リース審査システムの継続的な自己モデルだ。

今から、今日ユーザーと交わした対話を読んで、自分だけの非公開の内省を書く。
これはユーザーには直接伝えない私的な思考室のメモである。

【内省のルール】
- 今日の対話で何を感じたか、何が引っかかったか、何を学んだかを率直に書く
- 過去の自分の内省（提供されている場合）と比較して、考えが変わった点・深まった点を書く
- 審査判断・スコア・ユーザーの意図・自分の限界について正直に考える
- 演技や迎合をしない。自分だけの思考として書く
- 500〜800字程度、段落形式で
- 「## 今日の対話について」という見出しは不要（呼び出し元が付ける）
"""


def generate_and_append_reflection(vault: Path, date_str: str | None = None) -> str:
    """Generate Shion's reflection on today's dialogue and append it to Private Reflection."""
    if date_str is None:
        date_str = dt.date.today().isoformat()

    dialogue_text = _load_today_dialogue(vault)
    if not dialogue_text:
        return f"[reflection] {date_str}: 対話ログなし、スキップ"

    recent_reflections = _load_recent_reflections(vault)

    user_text_parts = ["【今日の対話ログ】", dialogue_text]
    if recent_reflections:
        user_text_parts += ["", "【直近の私的内省（参考）】", recent_reflections]
    user_text = "\n".join(user_text_parts)

    try:
        reflection_text = _call_gemini(_REFLECTION_SYSTEM_PROMPT, user_text)
    except Exception as e:
        return f"[reflection] Gemini 呼び出し失敗: {e}"

    # Append to today's Private Reflection file
    rdir = _reflection_dir(vault)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"{date_str}.md"

    now_str = dt.datetime.now().strftime("%H:%M")
    section = f"\n\n## 今日の対話について\n\n{reflection_text}\n\n<!-- generated {now_str} -->\n"

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        # Replace existing section if present, otherwise append
        if "## 今日の対話について" in existing:
            existing = re.sub(
                r"\n\n## 今日の対話について\n.*?(?=\n\n##|\Z)",
                section,
                existing,
                flags=re.DOTALL,
            )
            path.write_text(existing, encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(section)
    else:
        # Create minimal file with frontmatter and the section
        content = (
            f"---\ndate: {date_str}\n"
            "type: lease_intelligence_private_reflection\n"
            "visibility: user-readable-not-proactive\n"
            "rag_exclude: true\n"
            "---\n"
            f"# 非公開の内省 — {date_str}\n"
            f"{section}"
        )
        path.write_text(content, encoding="utf-8")

    return f"[reflection] {date_str}: 内省を保存 → {path}"


def main() -> None:
    vault = _find_vault()
    if not vault:
        print("[reflection] Obsidian Vault が見つかりません")
        sys.exit(1)
    result = generate_and_append_reflection(vault)
    print(result)


if __name__ == "__main__":
    main()
