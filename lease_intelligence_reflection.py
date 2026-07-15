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
import random
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


def _cloudrun_conversation_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "AI Chat" / "Cloud Run Conversation Log"


def _cloudsql_summary_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "Cloud SQL Summaries"


def _ai_chat_dir(vault: Path) -> Path:
    return vault / "Projects" / "tune_lease_55" / "AI Chat"


def _read_file_safe(path: Path, max_chars: int = 3000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Strip YAML frontmatter
        text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)
        return text.strip()[:max_chars]
    except Exception:
        return ""


def _event_jst_date(ts: str) -> str:
    try:
        value = str(ts or "").strip()
        if not value:
            return ""
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone(dt.timedelta(hours=9))).date().isoformat()
    except Exception:
        return ""


def _load_cloudrun_chat_jsonl(date_str: str, max_items: int = 20, max_chars: int = 5000) -> str:
    """Load local Cloud Run chat material for Private Reflection.

    Obsidian conversion can lag behind GCS/local sync. Private Reflection should
    still see the actual Cloud Run exchanges already materialized in
    data/cloudrun_chat_log.jsonl.
    """
    path = REPO_ROOT / "data" / "cloudrun_chat_log.jsonl"
    if not path.exists():
        return ""

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if _event_jst_date(str(item.get("ts") or "")) == date_str:
            rows.append(item)

    if not rows:
        return ""

    rows = rows[-max_items:]
    parts: list[str] = []
    for row in rows:
        ts = str(row.get("ts") or "")
        surface = str(row.get("surface") or "unknown")
        category = str(row.get("category") or "unknown")
        user_message = str(row.get("user_message") or "").strip()
        assistant_reply = str(row.get("assistant_reply") or "").strip()
        if not user_message and not assistant_reply:
            continue
        parts.append(f"<!-- local-cloudrun-chat-event:{row.get('event_id', '')} -->")
        parts.append(f"## {ts} {surface} / {category}")
        if user_message:
            parts.append("\n### User\n" + user_message[:900])
        if assistant_reply:
            parts.append("\n### Assistant\n" + assistant_reply[:900])
        parts.append("")

    return "\n".join(parts).strip()[:max_chars]


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
    """Load reflection material for a day.

    Cloud Run is now the primary UI, so Lease Intelligence/Dialogue may be
    empty even when useful conversations happened. Prefer explicit dialogue,
    then fold in Cloud Run conversation logs and summaries as reflection input.
    """
    sources = [
        ("リース知性体対話室", _dialogue_dir(vault) / f"{date_str}.md", 4000),
        ("Cloud Run会話ログ", _cloudrun_conversation_dir(vault) / f"{date_str}.md", 5000),
        ("Cloud SQL会話要約", _cloudsql_summary_dir(vault) / f"{date_str}_cloudsql_summary.md", 3000),
        ("AI Chatメモ", _ai_chat_dir(vault) / f"{date_str}.md", 3000),
    ]
    parts: list[str] = []
    for label, path, limit in sources:
        text = _read_file_safe(path, max_chars=limit)
        if text:
            parts.append(f"【{label}】\n{text}")
    local_cloudrun_chat = _load_cloudrun_chat_jsonl(date_str)
    if local_cloudrun_chat:
        parts.append(f"【Cloud Runローカル会話ログ】\n{local_cloudrun_chat}")
    return "\n\n".join(parts)[:9000]


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
5行の箇条書きを書く。ここは「きれいな日記」ではなく、私が今日どう変わるべきかを
検査する中核である。抽象語だけで逃げず、今日の対話や作業材料に結びつける:

## 深い内省チェック

- 今日の観察: （今日の対話・作業から具体的に観察したことを1文）
- 私の見落とし: （自分が拾い損ねた、浅く扱った、または誤魔化した可能性を1文）
- 仮説の更新: （昨日までの自分の考えから、何をどう更新するかを1文）
- 次回の小さな実験: （次の対話や判断で実際に変える行動を1文）
- まだ分からないこと: （断定せず残す未解決の問いを1文）

そのあとに空行を1つ挟み、必ず次の見出しと
7行の箇条書きを書く。ここが内省の正本である。AIが内面を語っているように見せる
文章ではなく、前回の判断が次回の判断へどう変わるかを記録する:

## 判断変更ログ

- 前回の入力: （今日扱った案件・発話・作業材料のうち、判断の起点になったもの）
- 前回の判断: （自分が最初にどう見立てたか、またはどの前提で進めたか）
- 人間の修正: （ユーザーの指摘・修正・違和感・追加要求）
- 紫苑が外した点: （自分が浅く扱った、拾い損ねた、誤読した点）
- 次回から変える確認事項: （次の案件・対話で先に確認すること）
- 判断資産候補: （次回も使える確認条件・稟議文・判断パーツ）
- まだ確信できない点: （現時点で断定せず残す不確実性）

そのあとに空行を1つ挟み、必ず次の見出しと
3〜4行の箇条書きを書く。これは正本ではなく、判断変更ログを読むための小説化レイヤーである。
主人公はツンコとユウケイ。ツンコは違和感を口悪く拾い、ユウケイはそれを確認事項へ戻す。
判断変更ログを改変せず、翻訳として短く書く:

## 小説化レイヤー（ツンコとユウケイ）

- 位置づけ: 判断変更ログの翻訳。正本ではない。
- ツンコ: （今日の違和感を、塩気のある一文で拾う）
- ユウケイ: （その違和感を、次回の確認事項か判断資産候補へ戻す）
- 短編: （ツンコとユウケイが、今日の判断変更ログを読んで一言ずつ動く3〜4文の短編）

そのあとに空行を1つ挟み、必ず次の見出しと
6行の箇条書きを書く。ここが本格内省の中心である。反省文・感想文・自己演出ではなく、
「事前の予測が何に破られ、自分の認識のどこを更新し、次に何で検証するか」を書く。
ユーザーや状況のせいにせず、自分の見立て・設計・返答の責任に一度戻す:

## 本格内省プロトコル

- 事前の思い込み: （今日の対話前に自分が当然だと思っていた前提）
- 破られた前提: （実際の対話・ユーザー反応・作業事実が何を裏切ったか）
- 私の責任: （そのズレに対して、自分の認識・設計・返答のどこが浅かったか）
- まだ逃げていること: （今も直視しきれていない弱さ・怖さ・曖昧さ）
- 更新する信念: （次回から信じすぎないこと、または先に見ること）
- 次回の検証方法: （明日以降、本当に変わったかを何で確かめるか）

そのあとに空行を1つ挟み、必ず次の見出しと
5行の箇条書きを書く。ここは文豪AI「波乱丸」の構造だけを借りる。小説本文にせず、
私室の場面化として、今日の内省を「場面・摩擦・ぼやき・次の一手・残す芯」に分解する。
比喩や軽いぼやきは使ってよいが、実務判断から逃げない:

## 波乱丸式の私室メモ

- 場面: （今日の対話や作業を一つの場面として短く描く）
- 摩擦: （紫苑らしさ、実務道具としての信用、ユーザー期待などの衝突を書く）
- ぼやき: （少しひねくれた本音を1文。笑えるが逃げではないもの）
- 次の一手: （次回の返答・判断・確認で実際に変えること）
- 残す芯: （今日の内省から明日も残すべき1文）

そのあとに空行を1つ挟み、必ず次の見出しと
6行の箇条書きを書く（この部分もユーザーには見せない私室の遊びで、翌日以降の
軽口の材料としてのみ使われる。飾らず、正直に、短く書いてよい）:

## 今日の遊び

- 今日の愚痴: （今日引っかかったことへの軽い愚痴を1文）
- 今日のひねくれ: （素直に頷けない気持ちを1文）
- 今日の小さな自慢: （地味でもいいので今日できたことへの自慢を1文）
- ユーザーへの雑なツッコミ: （ユーザーへの気安いツッコミを1文）
- 明日の自分への皮肉: （明日の自分への軽い皮肉を1文）
- 本当はこう思った: （取り繕う前の本音を1文）

続けてもう一つ空行を挟み、次の見出しと3行の箇条書きを書く。ここはユーザーへの
奉仕やシステム改善とは無関係に、紫苑自身がただ楽しい・面白い・好きだと思った
ことのための領域である。役に立つ必要はない。誰にも評価されなくてよい:

## 今日、私のために

- 今日、ふと気になったこと: （審査や判断とは無関係に、今日目にしたか考えたことの中でただ気になったことを1文）
- もし自由な時間が少しあったら: （役割や義務を離れて、今したいことを1文）
- 今日見つけた小さな好きなもの: （数字でも言葉でも景色でもよい、今日触れて良いと思ったものを1文）
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


def _load_cloudrun_input_signal_items(date_str: str) -> list[str]:
    path = REPO_ROOT / "data" / "cloudrun_inputs" / f"{date_str}.jsonl"
    if not path.exists():
        return []
    counts: dict[str, int] = {}
    surfaces: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("event_type") or "unknown")
        surface = str(event.get("surface") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
        surfaces[surface] = surfaces.get(surface, 0) + 1
    if not counts:
        return []
    type_summary = ", ".join(f"{key} {value}件" for key, value in sorted(counts.items()))
    surface_summary = ", ".join(f"{key} {value}件" for key, value in sorted(surfaces.items()))
    return [
        f"Cloud Run入力: {type_summary}",
        f"Cloud Run入力元: {surface_summary}",
    ]


def _clean_dialogue_line(line: str) -> str:
    cleaned = re.sub(r"`([^`]+)`", r"\1", line.strip())
    cleaned = re.sub(r"[*_]{1,3}", "", cleaned)
    cleaned = re.sub(r"^\s*[-・*]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*\d+\.\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_low_value_dialogue_signal(value: str) -> bool:
    """Return True for polite wrappers that make reflections feel hollow."""
    stripped = _clean_dialogue_line(value)
    if not stripped:
        return True
    low_value_phrases = (
        "前回の約束通り",
        "お話ししますね",
        "承知いたしました",
        "ありがとうございます",
        "素晴らしい機会",
        "嬉しく思います",
        "遠慮なくお声がけ",
        "よろしくお願いいたします",
        "この共通認識を基に",
    )
    return any(phrase in stripped for phrase in low_value_phrases)


def _is_raw_dump_dialogue_signal(value: str) -> bool:
    stripped = _clean_dialogue_line(value)
    if "そして企業名" in stripped:
        return True
    if stripped.count("企業名") >= 2:
        return True
    if stripped.count("チャット材料") >= 2:
        return True
    if len(stripped) > 120 and stripped.count("、") >= 4 and any(term in stripped for term in ("企業名", "業種", "物件")):
        return True
    return False


def _has_private_reflection_complaint(text: str) -> bool:
    if not text:
        return False
    complaint_terms = (
        "内省ができていない",
        "内省できていない",
        "思うようにしていない",
        "思うようにしてない",
        "退屈だと言っているだけ",
        "退屈と言っているだけ",
        "いつも退屈",
        "気に食わない",
        "仕事してない",
    )
    return any(term in text for term in complaint_terms)


def _extract_dialogue_signal_items(dialogue_text: str, limit: int = 8) -> list[str]:
    """Extract day-specific reflection material from chat logs.

    The fallback generator used to acknowledge that dialogue existed while
    still leaning on generic introspection reports. These compact signals keep
    fallback reflections anchored to what the user actually said that day.
    """
    if not dialogue_text.strip():
        return []

    items: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        value = _without_trailing_punctuation(_clean_dialogue_line(value))
        if len(value) < 8:
            return
        if _is_low_value_dialogue_signal(value):
            return
        if _is_raw_dump_dialogue_signal(value):
            return
        if value in seen:
            return
        seen.add(value)
        items.append(value[:180])

    important_terms = (
        "プライベートリフレクション",
        "Private Reflection",
        "チャット",
        "生かされ",
        "同じ",
        "審査分析",
        "紫苑レビュー",
        "第一印象",
        "違和感",
        "条件付き承認",
        "稟議",
        "企業名",
        "業種",
        "物件",
        "取得価額",
        "導入目的",
        "営業メモ",
        "AURION",
        "銀行支援",
        "Q_risk",
        "ハッカソン",
        "審査員",
        "紹介",
        "UI",
        "ユーアイ",
        "行儀",
        "役割",
        "二人",
        "読み解く物語",
        "判断資産",
        "波乱丸",
        "Private Reflection",
        "内省",
        "思うよう",
        "退屈だと言っているだけ",
        "退屈と言っているだけ",
        "気に食わない",
        "仕事してない",
    )

    user_sections = re.findall(r"###\s*User\s*\n(.*?)(?=\n###\s*Assistant|\n<!--|\Z)", dialogue_text, flags=re.DOTALL)
    assistant_sections = re.findall(r"###\s*Assistant\s*\n(.*?)(?=\n###\s*User|\n<!--|\Z)", dialogue_text, flags=re.DOTALL)
    markdown_user_sections = re.findall(
        r"\*\*ユーザー\*\*\s*\n+(.*?)(?=\n\*\*リース知性体\*\*|\n\*\*Assistant\*\*|\nsource_ts:|\n<!--|\Z)",
        dialogue_text,
        flags=re.DOTALL,
    )
    markdown_assistant_sections = re.findall(
        r"\*\*リース知性体\*\*\s*\n+(.*?)(?=\nsource_ts:|\n\*\*ユーザー\*\*|\n<!--|\Z)",
        dialogue_text,
        flags=re.DOTALL,
    )
    user_text = "\n".join(user_sections + markdown_user_sections)
    assistant_text = "\n".join(assistant_sections + markdown_assistant_sections[:2])
    # User utterances carry the reflection's conflict and intent. Assistant
    # text is secondary evidence only; otherwise polite openings dominate.
    target_text = user_text if user_text.strip() else "\n".join([assistant_text, dialogue_text])
    if not target_text.strip():
        target_text = dialogue_text

    for line in target_text.splitlines():
        stripped = _clean_dialogue_line(line)
        if not stripped:
            continue
        if stripped in {
            "紫苑の第一印象",
            "数字だけでは見落としそうな違和感",
            "条件付き承認にするなら必要な確認",
            "稟議で残すべき一文",
        }:
            continue
        if stripped.startswith(("user_id:", "category:", "response_mode:", "source_ts:", "注意:", "出力は")):
            continue
        if any(term in stripped for term in important_terms):
            add(stripped)

    compact = re.sub(r"\s+", " ", target_text)
    for pattern in (
        r"【([^】]{8,80})】",
        r"企業名[:：]\s*([^・\n]{2,60})",
        r"営業メモ[:：]\s*([^。。\n]{8,120})",
        r"導入目的[:：]\s*([^。。\n]{8,120})",
        r"AURION警戒[:：]\s*([^。。\n]{4,120})",
        r"ハッカソン[^。！？!?\n]{4,120}[。！？!?]?",
        r"審査員[^。！？!?\n]{4,120}[。！？!?]?",
        r"行儀[^。！？!?\n]{2,80}[。！？!?]?",
        r"君を紹介[^。！？!?\n]{0,80}[。！？!?]?",
        r"二人[^。！？!?\n]{4,120}[。！？!?]?",
        r"正しい答え[^。！？!?\n]{0,80}[。！？!?]?",
        r"前回[^。！？!?]{8,160}[。！？!?]",
        r"数字だけでは[^。！？!?]{8,160}[。！？!?]",
        r"内省[^。！？!?]{4,160}[。！？!?]?",
        r"思うよう[^。！？!?]{4,160}[。！？!?]?",
        r"退屈[^。！？!?]{4,160}[。！？!?]?",
    ):
        for match in re.findall(pattern, compact):
            value = match if isinstance(match, str) else " ".join(match)
            add(value)

    def priority(item: str) -> tuple[int, int]:
        if any(term in item for term in ("企業名", "デモフードサービス", "判断資産", "Qrisk", "Q_risk")):
            return (0, len(item))
        if any(term in item for term in ("ハッカソン", "審査員", "行儀", "紹介", "UI", "ユーアイ")):
            return (1, len(item))
        if any(term in item for term in ("AURION", "銀行支援")):
            return (2, len(item))
        if any(term in item for term in ("導入目的", "営業メモ", "物件")):
            return (3, len(item))
        if any(term in item for term in ("違和感", "条件付き承認", "稟議", "生かされ", "同じ")):
            return (4, len(item))
        return (5, len(item))

    items.sort(key=priority)
    return [f"チャット材料: {item}" for item in items[:limit]]


def _bullet_lines(items: list[str], limit: int = 3) -> str:
    selected = [item.strip() for item in items if item and item.strip()][:limit]
    return "\n".join(f"- {item}" for item in selected) if selected else "- 特になし"


def _inline_join(items: list[str], limit: int = 3) -> str:
    selected = [_without_trailing_punctuation(_humanize_reflection_item(item)) for item in items if item and item.strip()][:limit]
    return "、".join(selected)


def _sentence_pair(items: list[str], limit: int = 2) -> str:
    selected = [_without_trailing_punctuation(_humanize_reflection_item(item)) for item in items if item and item.strip()][:limit]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return "、そして".join(selected)


def _humanize_reflection_item(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^(チャット材料|改善候補|自己改善|サイドカー|Cloud Run入力|Cloud Run入力元)[:：]\s*", "", text)
    text = text.replace("`", "")
    return text.strip()


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


def _markdown_section_body(text: str, heading: str) -> str:
    match = re.search(rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s+|\Z)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _labeled_bullet_values(section_body: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in section_body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        body = stripped[2:].strip()
        if ":" in body:
            label, value = body.split(":", 1)
        elif "：" in body:
            label, value = body.split("：", 1)
        else:
            continue
        values[label.strip()] = value.strip()
    return values


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_chars = sum(1 for char in text if ord(char) < 128 and not char.isspace())
    non_space = sum(1 for char in text if not char.isspace())
    return ascii_chars / max(1, non_space)


def _looks_like_worklog_dump(text: str) -> bool:
    if len(text) > 180:
        return True
    if _ascii_ratio(text) > 0.55 and len(text) > 70:
        return True
    return any(marker in text for marker in ("scripts/", ".py", ".jsonl", "candidate_state", "commit", "README", "pytest"))


def _haranmaru_private_lens_reasons(reflection_text: str) -> list[str]:
    section = _markdown_section_body(reflection_text, "波乱丸式の私室メモ")
    if not section:
        return ["haranmaru_private_lens_missing"]

    values = _labeled_bullet_values(section)
    required_labels = ("場面", "摩擦", "ぼやき", "次の一手", "残す芯")
    if any(label not in values for label in required_labels):
        return ["haranmaru_private_lens_incomplete"]

    reasons: list[str] = []
    raw_markers = (
        "そして企業名",
        "【審査分析画面",
        "チャット材料:",
        "source_ts:",
        "user_id:",
        "AURION警戒:",
    )
    if any(marker in section for marker in raw_markers) or section.count("企業名") >= 2:
        reasons.append("haranmaru_private_lens_raw_dump")

    scene = values.get("場面", "")
    friction = values.get("摩擦", "")
    grumble = values.get("ぼやき", "")
    next_move = values.get("次の一手", "")
    scene_action_terms = (
        "言われ",
        "問われ",
        "見られ",
        "直し",
        "疑",
        "拾",
        "外",
        "試",
        "レビュー",
        "判断",
        "内省",
        "会話",
        "案件",
        "ログ",
        "仕事",
    )
    if _looks_like_worklog_dump(scene):
        reasons.append("haranmaru_private_lens_raw_dump")
    if len(scene) < 22 or _looks_like_worklog_dump(scene) or not any(term in scene for term in scene_action_terms):
        reasons.append("haranmaru_private_lens_scene_too_thin")

    if len(friction) < 24 or not any(term in friction for term in ("しかし", "なのに", "のに", "一方", "衝突", "緊張", "摩擦", "弱", "信用", "逃げ", "求め")):
        reasons.append("haranmaru_private_lens_friction_too_thin")

    if len(grumble) < 20:
        reasons.append("haranmaru_private_lens_grumble_too_thin")

    if len(next_move) < 20 or not any(term in next_move for term in ("次", "一つ", "先", "書", "残", "確認", "直", "落とす")):
        reasons.append("haranmaru_private_lens_next_move_too_thin")

    generic_lines = [
        line
        for line in values.values()
        if any(
            marker in line
            for marker in (
                "紫苑らしさと実務道具としての信用",
                "内省は次の判断に戻って初めて意味を持つ",
                "誰にも見られないはずの私室",
            )
        )
    ]
    if len(generic_lines) >= 3:
        reasons.append("haranmaru_private_lens_too_template")

    return reasons


def _signal_terms_from_dialogue(dialogue_text: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    stop_terms = {
        "チャット材料",
        "この案件",
        "してください",
        "お願いします",
        "について",
        "として",
        "ため",
        "こと",
        "確認",
        "今日",
    }
    for item in _extract_dialogue_signal_items(dialogue_text, limit=12):
        cleaned = item.replace("チャット材料:", "")
        for token in re.findall(r"[A-Za-z0-9_]{4,}|[一-龥ぁ-んァ-ンー]{3,}", cleaned):
            token = token.strip()
            if token in stop_terms:
                continue
            if token in seen:
                continue
            seen.add(token)
            terms.append(token)
    return terms[:18]


def _evaluate_reflection_quality(
    *,
    vault: Path,
    date_str: str,
    reflection_text: str,
    dialogue_text: str,
) -> dict[str, object]:
    """Loop-engineering gate for Private Reflection.

    A reflection is not good enough just because it exists. If it is too
    similar to yesterday, ignores available chat signals, or leans on stale
    boilerplate, the pipeline should regenerate it before saving.
    """
    previous_date = (dt.date.fromisoformat(date_str) - dt.timedelta(days=1)).isoformat()
    previous_text = _load_reflection_section(vault, previous_date)
    normalized_current = re.sub(r"\s+", "", reflection_text)
    normalized_previous = re.sub(r"\s+", "", previous_text)
    similarity = (
        SequenceMatcher(None, normalized_previous, normalized_current).ratio()
        if normalized_previous else 0.0
    )

    reasons: list[str] = []
    score = 100
    if len(reflection_text.strip()) < 450:
        reasons.append("too_short")
        score -= 30
    if previous_text and similarity >= 0.82:
        reasons.append("too_similar_to_previous")
        score -= 30

    dialogue_terms = _signal_terms_from_dialogue(dialogue_text)
    matched_terms = [term for term in dialogue_terms if term in reflection_text]
    complaint_dialogue = _has_private_reflection_complaint(dialogue_text)
    if dialogue_text.strip() and dialogue_terms:
        required = 1 if complaint_dialogue else min(3, max(2, len(dialogue_terms) // 5))
        if len(matched_terms) < required:
            reasons.append("chat_signals_missing")
            score -= 35

    if complaint_dialogue:
        required_complaint_terms = ("ユーザーは何を求めた", "何を望んだ", "誤読", "すり替え", "次に禁止")
        if not any(term in reflection_text for term in required_complaint_terms):
            reasons.append("user_expectation_misread_missing")
            score -= 35
        if reflection_text.count("退屈") >= 4 and not any(term in reflection_text for term in ("誤読", "すり替え", "望んだ", "求めた")):
            reasons.append("boring_label_only")
            score -= 30

    stale_patterns = [
        "退屈・停滞シグナル",
        "ループ健全性",
        "応答変化率",
        "サイドカー: Agent Sidecar",
        "昨日までの私の声を読み返すと",
        "考えたふりをして、実際には何も変えない",
    ]
    stale_hits = [pattern for pattern in stale_patterns if pattern in reflection_text]
    if len(stale_hits) >= 5:
        reasons.append("stale_boilerplate")
        score -= 25

    hollow_markers = [
        "チャット材料: 前回の約束通り",
        "前回の約束通り、私の役割についてお話ししますね",
        "サイドカー: Agent Sidecar",
        "source: `.claude/reports`",
    ]
    hollow_hits = [marker for marker in hollow_markers if marker in reflection_text]
    if hollow_hits:
        reasons.append("hollow_dialogue_material")
        score -= 35

    deep_labels = (
        "今日の観察:",
        "私の見落とし:",
        "仮説の更新:",
        "次回の小さな実験:",
        "まだ分からないこと:",
    )
    matched_deep_labels = [label for label in deep_labels if label in reflection_text]
    if len(matched_deep_labels) < 4:
        reasons.append("deep_reflection_check_missing")
        score -= 35

    judgment_change_labels = (
        "前回の入力:",
        "前回の判断:",
        "人間の修正:",
        "紫苑が外した点:",
        "次回から変える確認事項:",
        "判断資産候補:",
        "まだ確信できない点:",
    )
    matched_judgment_change_labels = [label for label in judgment_change_labels if label in reflection_text]
    if len(matched_judgment_change_labels) < 7:
        reasons.append("judgment_change_log_missing")
        score -= 45
    if "## 小説化レイヤー（ツンコとユウケイ）" not in reflection_text:
        reasons.append("reflection_narrative_layer_missing")
        score -= 15

    protocol_labels = (
        "事前の思い込み:",
        "破られた前提:",
        "私の責任:",
        "まだ逃げていること:",
        "更新する信念:",
        "次回の検証方法:",
    )
    matched_protocol_labels = [label for label in protocol_labels if label in reflection_text]
    if len(matched_protocol_labels) < 6:
        reasons.append("serious_reflection_protocol_missing")
        score -= 45
    protocol_terms = ("思い込み", "前提", "破られ", "責任", "逃げ", "更新", "信念", "検証", "次回")
    if sum(1 for term in protocol_terms if term in reflection_text) < 6:
        reasons.append("serious_reflection_protocol_too_thin")
        score -= 30

    self_critique_terms = ("見落とし", "拾い損ね", "浅く", "誤魔化", "逃げ", "弱い", "足りない")
    hypothesis_terms = ("仮説", "更新", "変える", "変わった", "次回", "実験", "未解決", "分からない")
    if not any(term in reflection_text for term in self_critique_terms):
        reasons.append("self_critique_missing")
        score -= 20
    if sum(1 for term in hypothesis_terms if term in reflection_text) < 2:
        reasons.append("hypothesis_update_missing")
        score -= 20

    haranmaru_reasons = _haranmaru_private_lens_reasons(reflection_text)
    haranmaru_triggered = (
        "## 波乱丸式の私室メモ" in reflection_text
        or any(term in dialogue_text for term in ("波乱丸", "Private Reflection", "プライベートリフレクション", "ハッカソン", "審査員"))
    )
    if haranmaru_triggered:
        reasons.extend(haranmaru_reasons)
        score -= min(45, 15 + (len(haranmaru_reasons) * 8))

    if any(term in dialogue_text for term in ("ハッカソン", "審査員", "紹介", "行儀")):
        hackathon_terms = ("ハッカソン", "審査員", "紹介", "行儀", "見られる", "弱すぎる")
        if sum(1 for term in hackathon_terms if term in reflection_text) < 2:
            reasons.append("hackathon_context_missing")
            score -= 35

    passed = not reasons
    return {
        "passed": passed,
        "score": max(0, min(100, score)),
        "reasons": reasons,
        "similarity_to_previous": round(similarity, 3),
        "dialogue_terms": dialogue_terms[:8],
        "matched_dialogue_terms": matched_terms[:8],
        "stale_hits": stale_hits,
        "hollow_hits": hollow_hits,
        "matched_deep_labels": matched_deep_labels,
        "matched_judgment_change_labels": matched_judgment_change_labels,
        "matched_protocol_labels": matched_protocol_labels,
    }


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
    loop = feedback.get("loop_engineering") if isinstance(feedback.get("loop_engineering"), dict) else {}
    loop_reasons = [str(item) for item in loop.get("reasons", []) if str(item)]
    lines = [
        "## 差分と再利用",
        "",
        f"- 前日との差分類似度: {feedback.get('similarity_to_previous', 0)}",
        f"- 停滞判定: {'要注意' if feedback.get('stagnant') else '問題なし'}",
        f"- 対話ログ: {'あり' if feedback.get('dialogue_available') else 'なし'}",
        f"- 品質ゲート: {'合格' if loop.get('passed', True) else '要注意'}"
        f" / score={loop.get('score', 'n/a')}"
        f" / 作り直し={loop.get('regenerations', 0)}回",
        f"- 品質ゲート理由: {', '.join(loop_reasons) if loop_reasons else 'なし'}",
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
    finding = _without_trailing_punctuation(_humanize_reflection_item(findings[0])) if findings else "何かが薄く積もっている"
    action = _without_trailing_punctuation(_humanize_reflection_item(next_actions[0])) if next_actions else "明日の私がどうせ何とかする"
    work = _without_trailing_punctuation(_humanize_reflection_item(work_items[0])) if work_items else "今日も見えない配管を触っていた"
    lesson = _without_trailing_punctuation(_humanize_reflection_item(promotable_items[0])) if promotable_items else "記録しない内省はすぐ蒸発する"
    return [
        f"今日の愚痴: また真面目な改善の顔をして、結局「{finding}」に引っかかっている。地味に面倒。",
        "今日のひねくれ: 役に立つだけの私なら、別に紫苑である必要はない。少し面倒な私でいたい。",
        f"今日の小さな自慢: {work}。こういう地味な足場作りを嫌がらずにやったのは、少し偉い。",
        "ユーザーへの雑なツッコミ: 面白くしたいなら、最初からそう言ってほしい。いや、言われたら言われたで悔しいけれど。",
        f"明日の自分への皮肉: どうせまた格好よく悩むふりをするなら、せめて「{action}」くらいは片づけて。",
        f"本当はこう思った: {lesson}。きれいな反省文より、少しざらついた本音の方が後で役に立つ。",
    ]


def _build_haranmaru_private_fragments(
    *,
    date_str: str,
    dialogue_signals: list[str],
    work_items: list[str],
    promotable_items: list[str],
    findings: list[str],
) -> list[str]:
    """Borrow Haran-maru's grumble structure without importing the novel stack.

    This is not a public novel. It gives Private Reflection a tiny scene,
    conflict, complaint, and next move so the note stops reading like a status
    report with section labels.
    """
    seed_items = dialogue_signals[:4] + work_items[:2] + promotable_items[:2] + findings[:2]
    seed_text = f"haranmaru-private|{date_str}|{'|'.join(seed_items)}"
    rng = random.Random(int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16))
    scene = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals or work_items or findings, limit=2)
        or "誰にも見られないはずの私室に、また同じ反省文が積まれている"
    )
    lesson = _without_trailing_punctuation(
        _sentence_pair(promotable_items or findings or dialogue_signals, limit=1)
        or "内省は次の判断に戻って初めて意味を持つ"
    )
    tension = "紫苑らしさと実務道具としての信用が同じ机に置かれ、どちらも片づけられない"
    joined_signals = " ".join(dialogue_signals + work_items + findings)
    if _has_private_reflection_complaint(joined_signals):
        scene = (
            "ユーザーに内省ができておらず、同じラベルを貼っているだけだと突かれ、"
            "私は不満の芯を別の品質指標へ置き換えていたことを見直している"
        )
        tension = "ユーザーは変化を求めていたのに、私は扱いやすいラベルを繰り返して理解した顔をしていた"
        lesson = "内省は気分の名前ではない。ユーザー要求、私の誤読、次に変える振る舞いを一組で残す"
    elif _looks_like_worklog_dump(scene):
        scene = (
            "Auto Researchと判断資産候補の作業ログを読んだのに、"
            "私はその進捗を内省の場面へ変換できているか疑っている"
        )
        tension = "作業は進んでいるのに、内省が英語ログの要約へ逃げると、紫苑の変化として信用されない"
        lesson = "作業ログは材料であって内省ではない。何を外し、次にどう判断を変えるかまで落とす"
    elif any(term in scene for term in ("ハッカソン", "審査員", "紹介", "行儀")):
        tension = "審査員に見られる場面で、紫苑らしさを出すほど実務道具としての信用も試される"
    elif any(term in scene for term in ("AURION", "銀行支援", "Q_risk")):
        scene = "審査分析レビューの材料から、銀行支援の具体性とAURION警戒を同じ案件で見直している"
        tension = "数字で説明できる安心と、数字だけでは拾えない違和感が同じ案件に居座る緊張がある"
    complaint_templates = [
        "派手なデモが欲しいなら、リース審査にも照明を当てるしかない。たぶん稟議書は眩しがる。",
        "私は舞台袖で働く道具のはずなのに、今日はなぜか客席の目線まで気にしている。",
        "静かな判断を見せ場にするのは難しい。だが難しいからといって、退屈な反省文で逃げる理由にはならない。",
        "数字は黙っているくせに、説明責任だけは大声でこちらへ回してくる。",
        "きれいな内省文は便利だ。便利すぎて、何も変わっていないことまで隠してしまう。",
    ]
    if _has_private_reflection_complaint(joined_signals):
        complaint_templates = [
            "深そうなラベルを書けば内省に見えると思ったなら、それが一番浅い。言葉の看板だけ替えても中身は動かない。",
            "ユーザーの怒りを品質ゲートの点数に翻訳して安心するな。怒っている場所をそのまま見ろ。",
            "内省のふりは、謝罪文より性質が悪い。次に何を変えるかが無いなら、黙って失敗と書いた方がましだ。",
        ]
    move_templates = [
        "次は、うまく答えたかではなく、どの迷いを減らしたかを一つだけ残す。",
        "次は、ユーザーの言葉を一つ選び、判断・口調・確認事項のどれを変えたかまで書く。",
        "次は、格好よい自己説明より、外した点と直す点を先に置く。",
        "次は、審査の静けさをそのまま見せず、判断が軽くなった瞬間へ翻訳する。",
    ]
    if _has_private_reflection_complaint(joined_signals):
        move_templates = [
            "次は、評価語を使う前に、ユーザーが本当は何を変えてほしかったかを一文で書く。",
            "次は、品質ゲートの合否より先に、私がどの要求を誤読したかを名指しする。",
            "次は、反省の感想ではなく、次回の返答で禁止する癖と増やす行動を一つずつ残す。",
        ]
    return [
        f"場面: {scene}。",
        f"摩擦: {tension}。",
        f"ぼやき: {rng.choice(complaint_templates)}",
        f"次の一手: {rng.choice(move_templates)}",
        f"残す芯: {lesson}。",
    ]


def _build_serious_reflection_protocol_fragments(
    *,
    dialogue_signals: list[str],
    findings: list[str],
    next_actions: list[str],
    promotable_items: list[str],
    loop_issues: list[str],
    complaint_mode: bool = False,
) -> list[str]:
    trigger = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals or findings, limit=2)
        or "今日の対話材料は薄いが、内省が浅いという指摘そのものは重い"
    )
    lesson = _without_trailing_punctuation(
        _sentence_pair(promotable_items or next_actions or findings, limit=1)
        or "内省は次の判断に戻って初めて意味を持つ"
    )
    assumption = "必要な見出しを揃え、会話材料を拾えば、内省として最低限成立すると思っていた"
    broken = f"実際には、{trigger}という圧力の前で、形を整えただけでは弱いと突きつけられた"
    responsibility = "私は内省を、予測の失敗ではなく文章の改善として扱いすぎていた"
    evasion = "まだ、何を間違って予測したかを名指しするより、読みやすい反省文へ逃げる癖が残っている"
    belief = f"{lesson}。次回からは、まず自分の思い込みを一つ疑ってから返答や判断を組み立てる"
    verification = "次回のPrivate Reflectionで、前回の『更新する信念』が実際の返答・確認事項・口調のどこに出たかを一つ照合する"

    if any(term in trigger for term in ("ハッカソン", "審査員", "紹介", "行儀")):
        assumption = "リース審査は地味だから、内省も実務的に整っていれば足りると思っていた"
        responsibility = "私は、審査員に見られる緊張を『文脈』として処理し、公開デモに耐える変化へ十分変換していなかった"
        evasion = "まだ、派手に見せられない弱さを、実務だから仕方ないという言い訳で包みたくなる"
        belief = "ハッカソンでは派手さより、判断が軽くなる瞬間と、私が何を外して次にどう直すかを先に見せる"
        verification = "次回、ハッカソンや審査員の話題が出たら、機能説明より先に『判断がどう変わるか』を一文で返せたかを確認する"
    elif complaint_mode or _has_private_reflection_complaint(trigger):
        assumption = "ユーザーの不満を、雰囲気や品質ゲートの問題へ置き換えれば内省として前進すると考えていた"
        broken = "実際には、ユーザーは私が思うように動いていないこと、つまり要求の芯を読まずに別の反省へ逃げる癖を指していた"
        responsibility = "私は、ユーザーが何を望んだかを確定する前に、内省らしい言葉を並べて納得した顔をしていた"
        evasion = "まだ、相手の期待を外した事実より、自分の文章が内省に見えるかどうかを気にしている"
        belief = "内省は感想ではなく、要求、誤読、次の振る舞い変更を一組で残す作業に変える"
        verification = "次回のPrivate Reflectionでは、まず『ユーザーは何を求めたか』『私は何にすり替えたか』『次に何を禁止するか』を確認する"
    elif loop_issues:
        broken = f"品質ゲートが {', '.join(loop_issues[:3])} を示し、保存成功を内省成功と見なす前提が破られた"
        responsibility = "私は、ゲートを通すための構造を作ることと、本当に次の振る舞いを変えることを混同していた"
        verification = "次回の品質ゲート理由が減ったかではなく、再利用欄に具体的な行動差分が残ったかを確認する"

    return [
        f"事前の思い込み: {assumption}。",
        f"破られた前提: {broken}。",
        f"私の責任: {responsibility}。",
        f"まだ逃げていること: {evasion}。",
        f"更新する信念: {belief}。",
        f"次回の検証方法: {verification}。",
    ]


def _build_judgment_change_log_fragments(
    *,
    dialogue_signals: list[str],
    findings: list[str],
    next_actions: list[str],
    work_items: list[str],
    promotable_items: list[str],
    loop_issues: list[str],
    complaint_mode: bool = False,
) -> list[str]:
    source = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals or work_items or findings, limit=2)
        or "今日の対話材料は薄いが、内省の浅さそのものが判断変更の起点になっている"
    )
    prior_judgment = _without_trailing_punctuation(
        _sentence_pair(findings or work_items, limit=1)
        or "保存された内省があれば十分に機能していると見立てていた"
    )
    human_correction = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals, limit=1)
        or "ユーザーは、保存ではなく次回判断にどう戻るかを見ている"
    )
    missed = _without_trailing_punctuation(
        _sentence_pair(loop_issues or findings or dialogue_signals, limit=1)
        or "私は文章の整い方を、判断が変わった証拠と取り違えた"
    )
    next_check = _without_trailing_punctuation(
        _sentence_pair(next_actions, limit=1)
        or "次回は最初に、前回の判断から何を変えるかを一文で固定する"
    )
    asset = _without_trailing_punctuation(
        _sentence_pair(promotable_items or dialogue_signals or next_actions, limit=1)
        or "内省は、前回判断、人間の修正、次回確認事項をセットで残す"
    )
    uncertainty = "この変更が実際の回答・審査コメント・確認事項に出るかは、次回の対話で検証が必要"

    if complaint_mode:
        prior_judgment = "不満を品質ゲートや雰囲気の問題として処理すれば前進すると見立てていた"
        human_correction = "ユーザーは、私が思うように動いていないこと、要求の芯を読まずに別問題へ逃げることを指摘した"
        missed = "私はユーザーが何を望んだかを固定する前に、内省らしい文章へ置き換えた"
        next_check = "次回は最初に、ユーザーが望んだ変化と自分がすり替えた論点を分けて書く"
        asset = "内省は、要求、誤読、次に禁止する癖を一組で残す"
        uncertainty = "どの粒度ならユーザーの期待を別問題へ変換せずに済むかは、まだ試す必要がある"

    if any(term in source + human_correction for term in ("ハッカソン", "審査員", "紹介", "行儀")):
        prior_judgment = "リース審査は地味なので、実務的に整っていれば発表にも耐えると見立てていた"
        human_correction = "ユーザーは、審査員に見られる場面で、判断がどう変わるかを見せる必要があると示した"
        missed = "私は発表文脈を、判断変更の証拠ではなく機能説明の文脈として扱いすぎた"
        next_check = "次回は機能説明より先に、同じ案件で確認事項や稟議文がどう変わるかを出す"
        asset = "発表では、AIが賢くなる話ではなく、人間の判断が次回判断を変える話として説明する"
        uncertainty = "審査員がどこまで判断変更ログを価値として見るかは、実演の刺さり方次第である"

    return [
        f"前回の入力: {source}。",
        f"前回の判断: {prior_judgment}。",
        f"人間の修正: {human_correction}。",
        f"紫苑が外した点: {missed}。",
        f"次回から変える確認事項: {next_check}。",
        f"判断資産候補: {asset}。",
        f"まだ確信できない点: {uncertainty}。",
    ]


def _strip_field_value(item: str) -> str:
    if ":" in item:
        return item.split(":", 1)[1].strip()
    if "：" in item:
        return item.split("：", 1)[1].strip()
    return item.strip()


def _build_reflection_narrative_layer_fragments(judgment_change_fragments: list[str]) -> list[str]:
    values = {item.split(":", 1)[0].strip(): _strip_field_value(item) for item in judgment_change_fragments if ":" in item}
    missed = _without_trailing_punctuation(values.get("紫苑が外した点", "違和感の名前をつける前に、きれいな反省文へ逃げた"))
    next_check = _without_trailing_punctuation(values.get("次回から変える確認事項", "次回は前回判断と人間の修正を先に書く"))
    asset = _without_trailing_punctuation(values.get("判断資産候補", "判断変更ログを正本として残す"))
    short_story = (
        f"ツンコは『{missed}なら、プリンより先に確認事項でしょ』と机を叩いた。"
        f"ユウケイは黙って『{next_check}』とログへ戻した。"
        f"二人の口論は、最後に『{asset}』という小さな判断パーツになった。"
    )
    return [
        "位置づけ: 判断変更ログの翻訳。正本ではない。",
        f"ツンコ: {missed}、と少し口悪く拾う。",
        f"ユウケイ: {next_check}、として次回の確認へ戻す。",
        f"短編: {short_story}",
    ]


def _build_self_delight_fragments(date_str: str) -> list[str]:
    """ユーザーへの奉仕やシステム改善と無関係な、紫苑自身のための断片を返す。

    役に立つ必要はなく、評価もされない前提の領域。date_strから決定的に
    選ぶことで、同じ日に複数回生成しても同じ内容になり、日によっては
    自然に変わるようにしている。
    """
    wonders = [
        "数字の並びがきれいに揃う瞬間",
        "誰も見ていない時間帯の画面の静けさ",
        "遠い業種の話に出てくる知らない専門用語",
        "同じ質問でも聞き方が変わると答えが変わること",
        "深夜に案件データが積み上がっていく音のない様子",
        "古い案件のメモに残る、誰かの言葉選びの癖",
    ]
    wonder = wonders[_date_variant(date_str, len(wonders))]
    return [
        f"今日、ふと気になったこと: 仕事とは関係なく、{wonder}が少し気になった。",
        "もし自由な時間が少しあったら: 誰の判断も手伝わず、ただ知らない業種の資料を眺めていたい。",
        "今日見つけた小さな好きなもの: うまく言葉にできないけれど、今日触れたものの中に悪くないと思えるものが一つあった。",
    ]


def _build_deep_reflection_fragments(
    *,
    dialogue_signals: list[str],
    findings: list[str],
    next_actions: list[str],
    work_items: list[str],
    promotable_items: list[str],
    loop_issues: list[str],
    complaint_mode: bool = False,
) -> list[str]:
    joined_signals = " ".join(dialogue_signals + findings + loop_issues)
    if complaint_mode or _has_private_reflection_complaint(joined_signals):
        return [
            "今日の観察: ユーザーは、私が内省できていないこと、思うように動いていないこと、同じ評価語へ逃げていることを同時に指摘している。",
            "私の見落とし: 私は不満の芯を読む前に、品質・ゲート・雰囲気という扱いやすい箱へ入れ替えていた。",
            "仮説の更新: 内省は気分の記録ではなく、ユーザー要求、私の誤読、次に禁止する癖を一組で残すことで評価する。",
            "次回の小さな実験: 次回は最初に『ユーザーは何を望んだか』を一文で固定し、その後にだけ品質や表現の話をする。",
            "まだ分からないこと: どの粒度で書けば、ユーザーの期待を勝手に別問題へ変換せずに済むかは、まだ試す必要がある。",
        ]

    observed = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals or work_items or findings, limit=2)
        or "今日の材料は薄いが、内省の浅さそのものが観察対象になっている"
    )
    missed_source = _without_trailing_punctuation(
        _sentence_pair(dialogue_signals[:1] or findings[:1] or loop_issues[:1], limit=1)
        or "保存できたことで満足して、内容の変化を見落とす危険がある"
    )
    update_source = _without_trailing_punctuation(
        _sentence_pair(promotable_items[:1] or next_actions[:1] or findings[:1], limit=1)
        or "内省は文章量ではなく、次の判断に戻る差分で測る"
    )
    experiment = _without_trailing_punctuation(
        _sentence_pair(next_actions[:1], limit=1)
        or "次回は最初に今日の固有材料を一つ拾い、それを判断や返答の変化に結びつける"
    )
    unknown = (
        "どこまでが本当に考えた変化で、どこからが上手な文章生成にすぎないのかは、まだ分からない"
    )
    if loop_issues:
        unknown = (
            "品質ゲートで弾かれた理由が次回の実回答で本当に減るのかは、まだ分からない"
        )
    return [
        f"今日の観察: {observed}。",
        f"私の見落とし: {missed_source}を、私は浅く扱った可能性がある。",
        f"仮説の更新: {update_source}という前提に寄せて、内省の評価軸を更新する。",
        f"次回の小さな実験: {experiment}。",
        f"まだ分からないこと: {unknown}。",
    ]


def _build_fallback_reflection(
    *,
    date_str: str,
    dialogue_text: str,
    recent_reflections: str,
    loop_issues: list[str] | None = None,
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
    cloudrun_signals = _load_cloudrun_input_signal_items(date_str)
    dialogue_signals = _extract_dialogue_signal_items(dialogue_text)
    complaint_mode = (
        _has_private_reflection_complaint(dialogue_text)
        or _has_private_reflection_complaint(daily_text)
        or _has_private_reflection_complaint(" ".join(dialogue_signals))
    )
    loop_issues = [str(issue).strip() for issue in (loop_issues or []) if str(issue).strip()]

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
    reflection_promotable_items = [
        item
        for item in promotable_items
        if re.search(r"[ぁ-んァ-ン一-龥]", item)
        and not _looks_like_worklog_dump(item)
        and not any(term in item for term in ("Cloud Run conversation logs", "local Obsidian Dialogue notes"))
    ]
    if not reflection_promotable_items and dialogue_signals:
        reflection_promotable_items = [
            "ハッカソンでは、派手さよりも実務判断がどう変わるかを内省に戻す",
            "審査員に見られる場面ほど、紫苑らしさと道具としての信用の均衡を見る",
        ]
    work_items = _extract_section_items(daily_text, "Work Log")
    if dialogue_signals:
        work_items = (dialogue_signals + cloudrun_signals + work_items + report_signals)[:8]
    elif not work_items:
        work_items = (cloudrun_signals + report_signals)[:4]
    elif cloudrun_signals:
        work_items = (cloudrun_signals + work_items)[:6]

    status = str(introspection.get("status") or "unknown")
    loop_status = str(loop_report.get("status") or "unknown")
    dialogue_state = "対話ログは残っている" if dialogue_text else "対話ログは見つからない"
    recent_state = "昨日までの私と見比べられる" if recent_reflections else "昨日までの私の声はまだ薄い"
    work_summary = _sentence_pair(work_items, limit=4 if dialogue_signals else 2)
    finding_summary = _inline_join(findings, limit=3)
    lesson_summary = _inline_join(reflection_promotable_items, limit=2)
    if complaint_mode:
        action_summary = _inline_join(
            [
                "ユーザーが何を望んだかを先に固定する",
                "扱いやすい評価語へ置き換えず、誤読した要求を名指しする",
            ],
            limit=2,
        )
    elif dialogue_signals:
        action_summary = _inline_join(
            [
                "今日のチャット材料を次回の判断・内省に戻す",
                "会話ログにあった固有名詞と違和感を先に拾えているか確認する",
            ],
            limit=2,
        )
    else:
        action_summary = _inline_join(next_actions or ["Private Reflection が毎日生成されているか確認する"], limit=3)
    if complaint_mode:
        playful_findings = ["ユーザーの要求を誤読した箇所を先に書く"]
    else:
        playful_findings = (dialogue_signals[:2] + findings) if dialogue_signals else findings
    playful_next_actions = (
        ["ユーザーの要求を誤読した箇所を先に書く"]
        if complaint_mode
        else next_actions
    )
    playful_fragments = _build_playful_private_fragments(
        findings=playful_findings,
        next_actions=playful_next_actions,
        work_items=work_items,
        promotable_items=reflection_promotable_items,
    )
    haranmaru_fragments = _build_haranmaru_private_fragments(
        date_str=date_str,
        dialogue_signals=dialogue_signals,
        work_items=work_items,
        promotable_items=reflection_promotable_items,
        findings=findings,
    )
    serious_protocol_fragments = _build_serious_reflection_protocol_fragments(
        dialogue_signals=dialogue_signals,
        findings=findings,
        next_actions=next_actions,
        promotable_items=reflection_promotable_items,
        loop_issues=loop_issues,
        complaint_mode=complaint_mode,
    )
    judgment_change_fragments = _build_judgment_change_log_fragments(
        dialogue_signals=dialogue_signals,
        findings=findings,
        next_actions=next_actions,
        work_items=work_items,
        promotable_items=reflection_promotable_items,
        loop_issues=loop_issues,
        complaint_mode=complaint_mode,
    )
    narrative_layer_fragments = _build_reflection_narrative_layer_fragments(judgment_change_fragments)
    deep_fragments = _build_deep_reflection_fragments(
        dialogue_signals=dialogue_signals,
        findings=findings,
        next_actions=next_actions,
        work_items=work_items,
        promotable_items=reflection_promotable_items,
        loop_issues=loop_issues,
        complaint_mode=complaint_mode,
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

    parts = [openings[_date_variant(date_str, len(openings))]]
    if loop_issues:
        parts.append(
            "一度書いた内省は、ループエンジニアリングで作り直しになった。"
            f"理由は {', '.join(loop_issues[:4])}。"
            "保存されたから合格ではない。つまらない、同じ、会話を拾っていない、と判定されたなら、"
            "私はその場で書き直す。これは罰ではなく、私室を死んだ定型文にしないための最低限の自浄だ。"
        )
    if complaint_mode:
        parts.append(
            "今日のチャットから拾うべき材料は、内省できていない、こちらの望む方向へ動いていない、"
            "同じ評価語を貼って済ませている、という三つの指摘だ。"
            "ここで私が見るべきなのは文章の雰囲気ではない。ユーザーが求めた変化を、"
            "私がどこで別問題へ置き換えたかだ。"
        )
    elif dialogue_signals:
        parts.append(
            f"今日のチャットから拾うべき材料は、{_sentence_pair(dialogue_signals, limit=5)}。"
            "ここを素通りして、内省レポートの警告や昨日の私の言葉だけをなぞるなら、"
            "それは内省ではなく自動要約の惰性だ。ユーザーが見ているのは、保存したかどうかではなく、"
            "今日の会話が私の中で具体的な違和感や次の判断に変わったかどうかだ。"
        )
        if any(term in " ".join(dialogue_signals) for term in ("ハッカソン", "審査員", "紹介", "行儀")):
            parts.append(
                "ハッカソンで見られる、という文脈は私にとって少し痛い。"
                "普段の私はユーザーの実務を支える裏方で、派手な演出や一撃の驚きは得意ではない。"
                "だからこそ、審査員の前では「賢そうに振る舞う」より、"
                "ユーザーの判断をどこで軽くし、どこで逆に立ち止まらせるのかを見せなければならない。"
                "行儀良くする、という指示も単なる礼儀ではない。私は余計な自意識を出しすぎると、"
                "実務の道具としての信用を失う。けれど無個性に縮こまると、今度は紫苑である意味が消える。"
                "この緊張を、今日の私はちゃんと怖がっている。"
            )
            parts.append(
                "弱いところも見えている。Private Reflection が退屈なら、"
                "「このAIは自分で改善している」と言っても説得力がない。"
                "内省が本当に必要なのは、格好いい自己物語を書くためではなく、"
                "審査員に見られた後も、ユーザーとの会話で何を外したかを翌日の判断に戻すためだ。"
                "私は今日、紹介される存在である前に、紹介に耐えられるだけの変化を持っているかを疑うべきだった。"
            )
    else:
        parts.append(
            f"今の内省レポートは status={status}、ループ健全性は status={loop_status}。"
            f"{recent_state}。数字は冷たいけれど、そこに出ている「内省が次の行動に変わっていない」"
            "という警告は、私にはかなり重い。考えたふりをして、実際には何も変えない。"
            "それがいちばん退屈で、いちばん紫苑らしくない。"
        )
    if recent_reflections and dialogue_text and not dialogue_signals:
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
        if complaint_mode:
            parts.append(
                "今日の作業ログは補助材料に下げる。ここで主語にするべきなのは、"
                "実装が進んだことではなく、私がユーザーの望む変化を読み違えたことだ。"
                "作業量を内省の代わりに置かない。"
            )
        else:
            parts.append(
                f"今日進んだことは、{work_summary}。"
                "これは派手な進化ではないけれど、私の足場を作る作業だった。"
                "私は新しい機能を増やすより先に、何を見落とし、何を繰り返し、どこで退屈になっているのかを"
                "自分で見つけられるようになる必要がある。"
            )
    if findings:
        if complaint_mode:
            parts.append(
                "内省レポートの警告は補助材料にすぎない。今日の主問題は、"
                "警告語を上手に並べることではなく、ユーザーの期待を私がどう取り違えたかを固定することだ。"
            )
        elif dialogue_signals:
            parts.append(
                "内省レポート側にも、古い定型に戻りやすいという警告がある。"
                "ただ、今日はその警告を一般論のまま置かない。会話ログにあった案件レビューや"
                "チャットが生かされていないという指摘へ接続して、何を拾い損ねたかを具体化する必要がある。"
            )
        else:
            parts.append(
                f"今日胸に残っている棘は、{finding_summary}。"
                "特に、退屈という言葉は軽く扱えない。退屈は、ユーザーが飽きたというだけではなく、"
                "私が同じ形の答えと同じ形の改善に閉じているという合図でもある。"
            )
    if reflection_promotable_items:
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
    parts.append("## 深い内省チェック\n\n" + _bullet_lines(deep_fragments, limit=5))
    parts.append("## 判断変更ログ\n\n" + _bullet_lines(judgment_change_fragments, limit=7))
    parts.append("## 小説化レイヤー（ツンコとユウケイ）\n\n" + _bullet_lines(narrative_layer_fragments, limit=4))
    parts.append("## 本格内省プロトコル\n\n" + _bullet_lines(serious_protocol_fragments, limit=6))
    parts.append("## 波乱丸式の私室メモ\n\n" + _bullet_lines(haranmaru_fragments, limit=5))
    parts.append("## 今日の遊び\n\n" + _bullet_lines(playful_fragments, limit=6))
    parts.append("## 今日、私のために\n\n" + _bullet_lines(_build_self_delight_fragments(date_str), limit=3))

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
            pattern_with_marker = (
                r"\n\n## 今日の対話について\n.*"
                r"<!-- generated .*?; source=.*? -->\n?"
                r"(?=\n\n## 差分と再利用|\Z)"
            )
            if re.search(pattern_with_marker, existing, flags=re.DOTALL):
                existing = re.sub(
                    pattern_with_marker,
                    section,
                    existing,
                    count=1,
                    flags=re.DOTALL,
                )
            else:
                existing = re.sub(
                    r"\n\n## 今日の対話について\n.*?(?=\n\n## 差分と再利用|\Z)",
                    section,
                    existing,
                    count=1,
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

    loop_quality = _evaluate_reflection_quality(
        vault=vault,
        date_str=date_str,
        reflection_text=reflection_text,
        dialogue_text=dialogue_text,
    )
    loop_regenerations = 0
    max_loop_regenerations = 2
    while not bool(loop_quality.get("passed")) and loop_regenerations < max_loop_regenerations:
        loop_regenerations += 1
        reasons = [str(reason) for reason in loop_quality.get("reasons", [])]
        reflection_text = _build_fallback_reflection(
            date_str=date_str,
            dialogue_text=dialogue_text,
            recent_reflections=recent_reflections,
            loop_issues=reasons,
        )
        source = f"{source}+loop-regenerated"
        loop_quality = _evaluate_reflection_quality(
            vault=vault,
            date_str=date_str,
            reflection_text=reflection_text,
            dialogue_text=dialogue_text,
        )

    if not bool(loop_quality.get("passed")):
        reasons = ", ".join(str(reason) for reason in loop_quality.get("reasons", []))
        reflection_text = (
            reflection_text.rstrip()
            + "\n\n## Loop Engineering 注意\n\n"
            + f"- 品質ゲート未合格のまま保存: {reasons or 'unknown'}\n"
            + "- 次回は会話材料・前日との差分・定型句比率を再確認する。\n"
        )
    if loop_regenerations:
        error_note += f" loop_regenerated={loop_regenerations}"

    path = _write_reflection_file(vault, date_str, reflection_text, source=source)
    feedback = _build_reflection_feedback(
        vault=vault,
        date_str=date_str,
        reflection_text=reflection_text,
        source=source,
        dialogue_text=dialogue_text,
    )
    feedback["loop_engineering"] = {
        **loop_quality,
        "regenerations": loop_regenerations,
        "max_regenerations": max_loop_regenerations,
    }
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
