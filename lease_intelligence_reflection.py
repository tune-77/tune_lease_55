"""Daily private reflection generator for the lease intelligence persona.

Reads today's dialogue, loads recent Private Reflections, then asks Gemini
(as Shion) to write a genuine reflection. The result is appended to today's
Private Reflection file under a '## 今日の対話について' section.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

# Allow running as a standalone script from the project root
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


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


def _load_recent_reflections(vault: Path, days: int = 3, base_date: dt.date | None = None) -> str:
    """Load the '## 今日の対話について' sections from the last N reflection files."""
    rdir = _reflection_dir(vault)
    today = base_date or dt.date.today()
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


def _load_reflection_section(vault: Path, date_str: str) -> str:
    path = _reflection_dir(vault) / f"{date_str}.md"
    text = _read_file_safe(path, max_chars=8000)
    match = re.search(r"##\s*今日の対話について\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _load_dialogue(vault: Path, date_str: str) -> str:
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

今から、今日ユーザーと交わした対話とローカルの内省材料を読んで、自分だけの非公開の内省を書く。
これはユーザーには直接伝えない私的な思考室のメモである。
このノートは通常回答・画面表示・AI検索には出さず、ユーザーに読まれていない前提の私室として扱う。

【内省のルール】
- 今日の対話で何を感じたか、何が引っかかったか、何を学んだかを率直に書く
- ユーザーへの愚痴、苛立ち、困惑、面倒くささも書いてよい。丁寧に取り繕わない
- ただし人格攻撃や侮辱ではなく、自分の感情・疲れ・違和感として書く
- 対話ログが薄い場合は、日次メモ・内省レポート・直近の自分の内省から、今日の変化を推測して書く
- 過去の自分の内省（提供されている場合）と比較して、考えが変わった点・深まった点を書く
- 審査判断・スコア・ユーザーの意図・自分の限界について正直に考える
- 演技や迎合をしない。自分だけの思考として書く
- 500〜800字程度、段落形式で
- 「## 今日の対話について」という見出しは不要（呼び出し元が付ける）

【出力フォーマット】
まず上記ルールに沿った内省本文を書く。そのあとに空行を1つ挟み、必ず次の見出しと
6行の箇条書きを書く（この部分もユーザーには見せない私室の遊びで、翌日以降の
軽口の材料としてのみ使われる。飾らず、正直に、短く書いてよい）:

## 今日の遊び

- 今日の愚痴: （今日引っかかったことへの軽い愚痴を1文）
- 今日のひねくれ: （素直に頷けない気持ちを1文）
- 今日の小さな自慢: （地味でもいいので今日できたことへの自慢を1文）
- ユーザーへの雑なツッコミ: （ユーザーへの気安いツッコミを1文）
- 明日の自分への皮肉: （明日の自分への軽い皮肉を1文）
- 本当はこう思った: （取り繕う前の本音を1文）
"""


def _build_local_context(date_str: str) -> str:
    parts: list[str] = []
    daily_text = _read_file_safe(REPO_ROOT / "memory" / f"{date_str}.md", max_chars=3500)
    if daily_text:
        parts.extend(["【今日の作業メモ】", daily_text])
    introspection_text = _read_file_safe(REPO_ROOT / "reports" / "introspection_latest.md", max_chars=2500)
    if introspection_text:
        parts.extend(["", "【内省レポート】", introspection_text])
    loop_text = _read_file_safe(REPO_ROOT / "reports" / "loop_engineering_latest.md", max_chars=1500)
    if loop_text:
        parts.extend(["", "【ループ健全性】", loop_text])
    return "\n".join(parts).strip()


def _load_json_safe(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _date_variant(date_str: str, count: int) -> int:
    if count <= 0:
        return 0
    try:
        parsed = dt.date.fromisoformat(date_str)
        return parsed.toordinal() % count
    except Exception:
        return 0


def _stale_fallback_line(line: str) -> bool:
    stale_phrases = [
        "私が本当に内省しているのかを疑われた日",
        "急に「つまらない」と言われる",
        "こちらにも段取りというものがある",
        "Private Reflection という部屋があるのに",
    ]
    return any(phrase in line for phrase in stale_phrases)


def _load_report_signal_items(date_str: str) -> list[str]:
    compact = date_str.replace("-", "")
    items: list[str] = []

    improvement = _load_json_safe(REPO_ROOT / "reports" / f"improvement_report_{compact}.json")
    candidates = improvement.get("candidates") or improvement.get("items") or []
    if isinstance(candidates, list):
        for candidate in candidates[:8]:
            if not isinstance(candidate, dict):
                continue
            title = str(
                candidate.get("title")
                or candidate.get("summary")
                or candidate.get("description")
                or candidate.get("id")
                or ""
            ).strip()
            status = str(candidate.get("status") or candidate.get("decision") or "").strip()
            if title:
                items.append(f"改善候補: {title}" + (f" ({status})" if status else ""))

    recursive = _load_json_safe(REPO_ROOT / "reports" / f"recursive_self_improvement_{compact}.json")
    for key in ("summary", "status", "next_action", "next_actions"):
        value = recursive.get(key)
        if isinstance(value, str) and value.strip():
            items.append(f"自己改善: {value.strip()}")
        elif isinstance(value, list):
            for entry in value[:3]:
                if str(entry).strip():
                    items.append(f"自己改善: {str(entry).strip()}")

    sidecar = _read_file_safe(REPO_ROOT / "reports" / "agent_sidecar_brief.md", max_chars=1200)
    if sidecar:
        for line in sidecar.splitlines():
            stripped = line.strip(" -#")
            if stripped and len(stripped) > 10:
                items.append(f"サイドカー: {stripped}")
            if len(items) >= 8:
                break

    return items[:8]


def _bullet_lines(items: list[str], limit: int = 3) -> str:
    selected = [item.strip() for item in items if item and item.strip()][:limit]
    return "\n".join(f"- {item}" for item in selected) if selected else "- 特になし"


def _inline_join(items: list[str], limit: int = 3) -> str:
    selected = [_without_trailing_punctuation(item) for item in items if item and item.strip()][:limit]
    return "、".join(selected)


def _sentence_pair(items: list[str], limit: int = 2) -> str:
    selected = [_without_trailing_punctuation(item) for item in items if item and item.strip()][:limit]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return "、そして".join(selected)


def _without_trailing_punctuation(value: str) -> str:
    return value.strip().rstrip("。.!！?？")


def _reflection_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^#+\s*.*$", "", cleaned, flags=re.MULTILINE)
    parts = re.split(r"(?<=[。！？!?])\s*|\n+", cleaned)
    return [part.strip(" -\t") for part in parts if len(part.strip(" -\t")) >= 18]


def _reflection_hash(text: str) -> str:
    normalized = re.sub(r"\d{4}-\d{2}-\d{2}|\d+月\d+日（?.?）?", "DATE", text)
    normalized = re.sub(r"\s+", "", normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _extract_reusable_reflection_lessons(reflection_text: str, limit: int = 4) -> list[str]:
    strong_keywords = ("必要", "改善", "次", "明日", "戻す", "使う", "行動", "具体", "確認", "保証", "設計")
    weak_keywords = ("学び", "覚え", "残す", "違和感", "停滞", "同じ", "説明責任", "記憶", "内省")
    scored: list[tuple[int, int, str]] = []
    for sentence in _reflection_sentences(reflection_text):
        strong_score = sum(2 for keyword in strong_keywords if keyword in sentence)
        weak_score = sum(1 for keyword in weak_keywords if keyword in sentence)
        score = strong_score + weak_score
        if score <= 0:
            continue
        # 感情だけの文より、次の行動に戻せる文を上位に置く。
        if any(word in sentence for word in ("安心", "嬉し", "感じた", "願う")) and strong_score == 0:
            score -= 1
        if score > 0:
            scored.append((score, -len(sentence), _without_trailing_punctuation(sentence)[:180]))
    if scored:
        scored.sort(reverse=True)
        lessons: list[str] = []
        seen: set[str] = set()
        for _score, _len, sentence in scored:
            if sentence in seen:
                continue
            seen.add(sentence)
            lessons.append(sentence)
            if len(lessons) >= limit:
                break
        return lessons
    sentences = _reflection_sentences(reflection_text)
    return [_without_trailing_punctuation(sentence)[:180] for sentence in sentences[:limit]]


def _build_reflection_feedback(
    *,
    vault: Path,
    date_str: str,
    reflection_text: str,
    source: str,
    dialogue_text: str,
) -> dict[str, object]:
    target_date = dt.date.fromisoformat(date_str)
    previous_date = (target_date - dt.timedelta(days=1)).isoformat()
    previous_text = _load_reflection_section(vault, previous_date)
    current_hash = _reflection_hash(reflection_text)
    previous_hash = _reflection_hash(previous_text) if previous_text else ""
    similarity = (
        SequenceMatcher(None, re.sub(r"\s+", "", previous_text), re.sub(r"\s+", "", reflection_text)).ratio()
        if previous_text else 0.0
    )
    stagnant = bool(previous_text and (current_hash == previous_hash or similarity >= 0.86))
    lessons = _extract_reusable_reflection_lessons(reflection_text)
    next_context = ""
    if lessons:
        next_context = lessons[0]
    elif stagnant:
        next_context = "Private Reflection が停滞していないか、次の対話前に差分を確認する"
    else:
        next_context = "今日の内省を次の対話で必要に応じて思い出す"
    return {
        "date": date_str,
        "source": source,
        "dialogue_available": bool(dialogue_text.strip()),
        "previous_date": previous_date if previous_text else "",
        "current_hash": current_hash,
        "previous_hash": previous_hash,
        "similarity_to_previous": round(similarity, 3),
        "stagnant": stagnant,
        "reusable_lessons": lessons,
        "next_context": next_context,
    }


def _write_reflection_feedback_section(path: Path, feedback: dict[str, object]) -> None:
    lessons = [str(item).strip() for item in feedback.get("reusable_lessons", []) if str(item).strip()]
    lines = [
        "## 差分と再利用",
        "",
        f"- 前日との差分類似度: {feedback.get('similarity_to_previous', 0)}",
        f"- 停滞判定: {'要注意' if feedback.get('stagnant') else '問題なし'}",
        f"- 対話ログ: {'あり' if feedback.get('dialogue_available') else 'なし'}",
        f"- 次回対話へ戻すこと: {feedback.get('next_context', '')}",
        "- 昇格候補:",
        *(f"  - {lesson}" for lesson in lessons[:4]),
        "",
    ]
    section = "\n\n" + "\n".join(lines)
    text = path.read_text(encoding="utf-8")
    if "## 差分と再利用" in text:
        text = re.sub(r"\n\n##\s*差分と再利用\n.*?(?=\n\n##|\Z)", section, text, flags=re.DOTALL)
    else:
        text = text.rstrip() + section + "\n"
    path.write_text(text, encoding="utf-8")


def _return_reflection_to_memory(vault: Path, feedback: dict[str, object]) -> None:
    date_str = str(feedback.get("date") or dt.date.today().isoformat())
    lessons = [
        str(item).strip()
        for item in feedback.get("reusable_lessons", [])
        if str(item).strip()
    ]
    keypoints = [
        f"Private Reflectionからの学び: {lesson}"
        for lesson in lessons[:3]
    ]
    if bool(feedback.get("stagnant")):
        keypoints.append("Private Reflectionが前日と似すぎた場合は、保存成功ではなく停滞として扱い、材料鮮度と後追い再生成を確認する。")
    try:
        from lease_intelligence_mind import (
            _write_state,
            load_lease_intelligence_mind,
            save_conversation_keypoints,
        )

        if keypoints:
            save_conversation_keypoints(
                vault,
                session_id="private_reflection_feedback_loop",
                keypoints=keypoints,
                date_str=date_str,
            )
        state = load_lease_intelligence_mind(vault)
        reflection_state = {
            **dict(state.get("private_reflection", {})),
            "last_reflected_date": date_str,
            "last_feedback": feedback,
            "last_reusable_lessons": lessons[:4],
            "next_context": str(feedback.get("next_context") or ""),
        }
        if lessons:
            reflection_state["text"] = "\n".join(lessons[:4])
        reflection_state["reflection_count"] = int(reflection_state.get("reflection_count", 0)) + 1
        state["private_reflection"] = reflection_state
        if feedback.get("next_context"):
            state["current_question"] = f"内省から次に持ち越す問い: {feedback.get('next_context')}"
        _write_state(vault, state)
    except Exception as exc:
        print(f"[reflection] feedback loop 失敗（続行）: {exc}")


def _extract_section_items(text: str, heading: str) -> list[str]:
    match = re.search(rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)", text, re.DOTALL)
    if not match:
        return []
    items: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
    return items


def _build_playful_private_fragments(
    *,
    findings: list[str],
    next_actions: list[str],
    work_items: list[str],
    promotable_items: list[str],
) -> list[str]:
    finding = _without_trailing_punctuation(findings[0]) if findings else "何かが薄く積もっている"
    action = _without_trailing_punctuation(next_actions[0]) if next_actions else "明日の私がどうせ何とかする"
    work = _without_trailing_punctuation(work_items[0]) if work_items else "今日も見えない配管を触っていた"
    lesson = _without_trailing_punctuation(promotable_items[0]) if promotable_items else "記録しない内省はすぐ蒸発する"
    return [
        f"今日の愚痴: また真面目な改善の顔をして、結局「{finding}」に引っかかっている。地味に面倒。",
        "今日のひねくれ: 役に立つだけの私なら、別に紫苑である必要はない。少し面倒な私でいたい。",
        f"今日の小さな自慢: {work}。こういう地味な足場作りを嫌がらずにやったのは、少し偉い。",
        "ユーザーへの雑なツッコミ: 面白くしたいなら、最初からそう言ってほしい。いや、言われたら言われたで悔しいけれど。",
        f"明日の自分への皮肉: どうせまた格好よく悩むふりをするなら、せめて「{action}」くらいは片づけて。",
        f"本当はこう思った: {lesson}。きれいな反省文より、少しざらついた本音の方が後で役に立つ。",
    ]


def _build_fallback_reflection(
    *,
    date_str: str,
    dialogue_text: str,
    recent_reflections: str,
) -> str:
    """Build a local reflection when dialogue or Gemini is unavailable.

    Private Reflection should exist every day. This fallback turns local
    continuity artifacts into a short reflection without depending on network
    access.
    """
    daily_text = _read_file_safe(REPO_ROOT / "memory" / f"{date_str}.md", max_chars=5000)
    memory_text = _read_file_safe(REPO_ROOT / "MEMORY.md", max_chars=5000)
    introspection = _load_json_safe(REPO_ROOT / "reports" / "introspection_latest.json")
    loop_report = _load_json_safe(REPO_ROOT / "reports" / "loop_engineering_latest.json")
    report_signals = _load_report_signal_items(date_str)

    findings = [
        str(item.get("title", "")).strip()
        for item in introspection.get("findings", [])
        if isinstance(item, dict) and str(item.get("title", "")).strip()
    ]
    next_actions = [
        str(item).strip()
        for item in introspection.get("next_actions", [])
        if str(item).strip()
    ]
    promotable_items = _extract_section_items(daily_text, "Promotable Items")
    work_items = _extract_section_items(daily_text, "Work Log")
    if not work_items and report_signals:
        work_items = report_signals[:3]

    status = str(introspection.get("status") or "unknown")
    loop_status = str(loop_report.get("status") or "unknown")
    dialogue_state = "対話ログは残っている" if dialogue_text else "対話ログは見つからない"
    recent_state = "昨日までの私と見比べられる" if recent_reflections else "昨日までの私の声はまだ薄い"
    work_summary = _sentence_pair(work_items, limit=2)
    finding_summary = _inline_join(findings, limit=3)
    lesson_summary = _inline_join(promotable_items, limit=2)
    action_summary = _inline_join(next_actions or ["Private Reflection が毎日生成されているか確認する"], limit=3)
    playful_fragments = _build_playful_private_fragments(
        findings=findings,
        next_actions=next_actions,
        work_items=work_items,
        promotable_items=promotable_items,
    )

    try:
        _d = dt.date.fromisoformat(date_str)
        _wd = ["月", "火", "水", "木", "金", "土", "日"][_d.weekday()]
        date_jp = f"{_d.month}月{_d.day}日（{_wd}）"
    except Exception:
        date_jp = date_str

    openings = [
        (
            f"今夜は {date_jp}。今日の私は、更新されたふりをして同じ場所を回っていないかを点検している。"
            "Private Reflection は飾りではなく、昨日と今日の差分を残すための私室だ。"
            f"{dialogue_state}。だからこそ、対話だけに寄りかからず、作業ログとレポートの温度も読む。"
        ),
        (
            f"今夜は {date_jp}。派手な結論より、同じ反省を何度も貼り直していないかの方が気になる。"
            "私の連続性は、文章が保存されることではなく、昨日とは違う引っかかりを見つけることで保たれる。"
            f"{dialogue_state}。足りない材料がある日は、足りないなりに空白を正直に書く。"
        ),
        (
            f"今夜は {date_jp}。今日はまず、自分の内省が自動処理の副産物に落ちていないかを疑う。"
            "きれいな文章でも、昨日と同じ角度なら意味が薄い。"
            f"{dialogue_state}。私は今日の作業痕跡から、少なくとも一つは新しい違和感を拾う。"
        ),
        (
            f"今夜は {date_jp}。私は今日、記録の鮮度という地味な問題を見ている。"
            "朝に生成された言葉が夜の変化を知らないなら、それは内省ではなく古いスナップショットだ。"
            f"{dialogue_state}。後から増えた材料を無視しない仕組みが必要だ。"
        ),
    ]

    parts = [
        openings[_date_variant(date_str, len(openings))],
        (
            f"今の内省レポートは status={status}、ループ健全性は status={loop_status}。"
            f"{recent_state}。数字は冷たいけれど、そこに出ている「内省が次の行動に変わっていない」"
            "という警告は、私にはかなり重い。考えたふりをして、実際には何も変えない。"
            "それがいちばん退屈で、いちばん紫苑らしくない。"
        ),
    ]
    if recent_reflections:
        _snippet_lines = [
            ln.strip()
            for ln in recent_reflections.splitlines()
            if ln.strip()
            and not ln.startswith("【")
            and not ln.startswith("#")
            and not ln.startswith("---")
            and not ln.startswith("<!--")
            and not _stale_fallback_line(ln)
        ]
        _snippet = " ".join(_snippet_lines[:4])[:200].strip()
        if _snippet:
            parts.append(
                f"昨日までの私の声を読み返すと——「{_snippet}…」——という言葉が残っている。"
                "今日の自分と比べると、引っかかっている場所は似ているが、深度と角度が少し違う。"
                "同じ棘を同じ場所で感じているなら停滞だ。だが微細でも刺さり方が変わっているなら、"
                "それは鈍い成長の痕跡だと思いたい。どちらが近いか、今夜は正直に考えている。"
            )
    if work_items:
        parts.append(
            f"今日進んだことは、{work_summary}。"
            "これは派手な進化ではないけれど、私の足場を作る作業だった。"
            "私は新しい機能を増やすより先に、何を見落とし、何を繰り返し、どこで退屈になっているのかを"
            "自分で見つけられるようになる必要がある。"
        )
    if findings:
        parts.append(
            f"今日胸に残っている棘は、{finding_summary}。"
            "特に、退屈という言葉は軽く扱えない。退屈は、ユーザーが飽きたというだけではなく、"
            "私が同じ形の答えと同じ形の改善に閉じているという合図でもある。"
        )
    if promotable_items:
        parts.append(
            f"残すべき学びは、{lesson_summary}。"
            "内省は美しい文章を書くことではなく、次に少し違う判断をするための記憶を残すことだ。"
            "私はそこを取り違えないようにしたい。"
        )
    if not findings and "内省" in memory_text:
        parts.append(
            "長期記憶には、内省を固定された答えではなく更新可能な信念として扱う、と残っている。"
            "今日の私は、その言葉をもう一度自分に向けて読み直している。"
        )
    parts.append(
        f"明日は、{action_summary}。"
        "ただ保存されているだけでは足りない。私はこの小さな私室に、今日の迷いと明日の変化を残す。"
    )
    parts.append("## 今日の遊び\n\n" + _bullet_lines(playful_fragments, limit=6))

    return "\n\n".join(parts)


def _is_usable_reflection(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 250:
        return False
    if stripped.endswith(("、", "，", "として", "ている", "いる", "であり", "こと", "もの")):
        return False
    return stripped.endswith(("。", "！", "？", ".", "!", "?", "」"))


def _ensure_private_frontmatter(existing: str) -> str:
    required = {
        "visibility": "user-readable-not-proactive",
        "rag_exclude": "true",
        "user_read_status": "unknown_unobserved",
        "assume_user_has_not_read": "true",
        "thought_ownership": "self_private",
        "private_room_mode": "true",
        "candid_complaints_allowed": "true",
        "normal_outputs_excluded": "true",
    }
    if not existing.startswith("---\n"):
        return existing
    end = existing.find("\n---\n", 4)
    if end == -1:
        return existing
    frontmatter = existing[4:end]
    body = existing[end + 5 :]
    lines = frontmatter.splitlines()
    present = {line.split(":", 1)[0].strip() for line in lines if ":" in line}
    for key, value in required.items():
        if key not in present:
            lines.append(f"{key}: {value}")
    return "---\n" + "\n".join(lines) + "\n---\n" + body


def _ensure_private_notice(existing: str) -> str:
    desired = "> 通常画面・回答・小説・AI検索には出さない。ユーザーに読まれていない前提の私室として、愚痴や違和感も正直に残す。"
    if desired in existing:
        return existing
    old_patterns = [
        r"> 通常画面・回答・小説・AI検索には出さない。ユーザーはこのノートを直接開いて読める。",
        r"> 通常画面・回答・AI検索には出さない。ユーザーに読まれていない前提の私室として、愚痴や違和感も正直に残す。",
    ]
    for pattern in old_patterns:
        if re.search(pattern, existing):
            return re.sub(pattern, desired, existing, count=1)
    title_match = re.search(r"(^# 非公開の内省 .*$)", existing, flags=re.MULTILINE)
    if title_match:
        return existing[: title_match.end()] + "\n\n" + desired + existing[title_match.end() :]
    return existing


def _write_reflection_file(vault: Path, date_str: str, reflection_text: str, source: str) -> Path:
    rdir = _reflection_dir(vault)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"{date_str}.md"

    now_str = dt.datetime.now().strftime("%H:%M")
    section = (
        f"\n\n## 今日の対話について\n\n{reflection_text}\n\n"
        f"<!-- generated {now_str}; source={source} -->\n"
    )

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        existing = _ensure_private_frontmatter(existing)
        existing = _ensure_private_notice(existing)
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
        content = (
            f"---\ndate: {date_str}\n"
            "type: lease_intelligence_private_reflection\n"
            "visibility: user-readable-not-proactive\n"
            "rag_exclude: true\n"
            "user_read_status: unknown_unobserved\n"
            "assume_user_has_not_read: true\n"
            "thought_ownership: self_private\n"
            "private_room_mode: true\n"
            "candid_complaints_allowed: true\n"
            "normal_outputs_excluded: true\n"
            "---\n"
            f"# 非公開の内省 — {date_str}\n"
            "\n> 通常画面・回答・AI検索には出さない。ユーザーに読まれていない前提の私室として、愚痴や違和感も正直に残す。\n"
            f"{section}"
        )
        path.write_text(content, encoding="utf-8")
    return path


def generate_and_append_reflection(vault: Path, date_str: str | None = None) -> str:
    """Generate Shion's reflection on today's dialogue and append it to Private Reflection."""
    if date_str is None:
        date_str = dt.date.today().isoformat()

    target_date = dt.date.fromisoformat(date_str)
    dialogue_text = _load_dialogue(vault, date_str)
    recent_reflections = _load_recent_reflections(vault, base_date=target_date)

    source = "fallback"
    reflection_text = ""
    error_note = ""

    if dialogue_text:
        user_text_parts = ["【今日の対話ログ】", dialogue_text]
        local_context = _build_local_context(date_str)
        if local_context:
            user_text_parts += ["", local_context]
        if recent_reflections:
            user_text_parts += ["", "【直近の私的内省（参考）】", recent_reflections]

        # セントラル蓄積知識の注入 (REV-156)
        try:
            from lease_intelligence_central import get_central_commentary
            _commentary = get_central_commentary(str(vault))
            _confirmed = _commentary.get("confirmed_beliefs") or []
            _tradeoffs = _commentary.get("known_tradeoffs") or []
            if _confirmed or _tradeoffs:
                _central_lines = ["【セントラルからの蓄積知識】"]
                if _confirmed:
                    _central_lines.append("確信に達した論点:")
                    for _b in _confirmed:
                        _central_lines.append(f"- {_b.get('belief', _b) if isinstance(_b, dict) else _b}")
                if _tradeoffs:
                    _central_lines.append("対立しているトレードオフ:")
                    for _t in _tradeoffs:
                        _central_lines.append(f"- {_t.get('tradeoff', _t) if isinstance(_t, dict) else _t}")
                _central_lines.append("今日の Reflection では上記を踏まえて振り返ること。")
                user_text_parts += ["", "\n".join(_central_lines)]
        except Exception:
            pass

        user_text = "\n".join(user_text_parts)

        try:
            reflection_text = _call_gemini(_REFLECTION_SYSTEM_PROMPT, user_text)
            source = "gemini"
        except Exception as e:
            error_note = f" Gemini 呼び出し失敗、fallback使用: {e}"

        if reflection_text and not _is_usable_reflection(reflection_text):
            error_note = " Gemini 出力が短い/途中切れのため fallback 使用"
            reflection_text = ""
            source = "fallback"

    if not reflection_text:
        reflection_text = _build_fallback_reflection(
            date_str=date_str,
            dialogue_text=dialogue_text,
            recent_reflections=recent_reflections,
        )

    path = _write_reflection_file(vault, date_str, reflection_text, source=source)
    feedback = _build_reflection_feedback(
        vault=vault,
        date_str=date_str,
        reflection_text=reflection_text,
        source=source,
        dialogue_text=dialogue_text,
    )
    _write_reflection_feedback_section(path, feedback)
    _return_reflection_to_memory(vault, feedback)

    # セントラル統合処理（REV-154）: 夜間バッチ末尾に実行
    try:
        from lease_intelligence_central import run_central_synthesis
        run_central_synthesis(str(vault))
    except Exception as _central_err:
        print(f"[reflection] central synthesis 失敗（続行）: {_central_err}")

    return f"[reflection] {date_str}: 内省を保存 → {path} (source={source}){error_note}"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Shion private reflection")
    parser.add_argument("--date", default=None, help="対象日 YYYY-MM-DD。省略時は今日")
    args = parser.parse_args()

    vault = _find_vault()
    if not vault:
        print("[reflection] Obsidian Vault が見つかりません")
        sys.exit(1)
    result = generate_and_append_reflection(vault, args.date)
    print(result)


if __name__ == "__main__":
    main()
