"""チャットで教わった審査ノウハウを、回答前に決定的に保存・想起する。

2026-10 の調査で、対話室で教えたノウハウの大半が保存されていなかった。保存は回答後の
バックグラウンド処理と LLM 抽出任せで、雑談モードではツールも使わない。そのうえ紫苑は
保存していないのに「判断資産にします」と答えていた。ここでは次を LLM に頼らず行う。

* 判定: ``memory_promotion_policy.classify_lease_teaching``（1段）
* 保存: Obsidian ``Lease Intelligence/Knowledge/`` のノートと、判断資産候補（要確認）
* 想起: Knowledge ノートからの文字 n-gram 一致（回答前にプロンプトへ入れる）
* 正直さ: 保存の成否に合わない「判断資産にします」「永続化します」等を回答から外す。保存した回も
  保存先などの処理通知は本文に出さない（REV-540。結果は応答の teaching_save と指標ログで確かめる）
* 保存依頼: 「作ったテンプレートを保存して」は直前の紫苑の回答を保存する
* 指標: 教えた→保存した→想起した→回答で使った を ``data/shion_teaching_funnel.jsonl`` に残す

対話室と通常チャット（/api/chat）は ``prepare_teaching_turn`` → ``TeachingTurn.finalize`` で共通に使う。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.judgment_policy import (
    INSIGHT_CITATION_INSTRUCTION,
    POLICY,
    asset_citation,
    classify_knowledge_kind,
    format_policy_block,
    knowledge_kind_of,
)

from memory_promotion_policy import TEACHING_DOMAIN_TERMS
from runtime_paths import get_data_dir

FUNNEL_LOG_NAME = "shion_teaching_funnel.jsonl"
KNOWLEDGE_SOURCE_TYPE = "chat_teaching"
FUNNEL_EVENTS = ("taught", "saved", "recalled", "used")

# 「判断資産にしておいて」のように、指示語だけで中身が前の発言にある場合の閾値。
_ANAPHORIC_RESIDUAL_CHARS = 15
_EXPLICIT_PHRASES = re.compile(
    r"(判断資産(に|として|へ)?(して|しておいて|入れて|登録して|残して|覚えて|覚えといて)?|"
    r"覚えておいて|覚えといて|覚えて|記録しておいて|記録して|メモしておいて|メモして|登録して|入れて)"
)
# 保存・登録・永続化を「する／した」と言い切る表現。提案・一般論（「保存が必要です」「残すべきです」
# 「保存しておくと良い」）と問い返し（「保存しますか」）は含めない。2026-10-04 に「〇〇集として
# 永続化します」「いつでも呼び出せます」「今回の保存で…完了です」が素通りした。
_DONE = r"(いたします|いたしました|します|しました|しておきます|しておきました)"
SAVE_CLAIM_RE = re.compile(
    r"(判断資産(に|として)[^。\n]{0,12}(します|しました|登録|追加|記録|残し)|"
    r"覚え(ます|ました|ておきます|ておきました)|"
    rf"(保存|登録|記録|永続化|蓄積){_DONE}|"
    r"(保存|登録|記録|永続化)(は|が|を)?完了(しました|です|いたしました)|(保存|登録|記録)済みです|今回の(保存|登録|記録)|"
    r"残(します|しました|しておきます|しておきました)|"
    rf"(判断資産|集|データベース|ライブラリ|台帳|ナレッジ|Knowledge|テンプレート)(に|へ)[^。\n]{{0,8}}((追加|収録|格納){_DONE}|入れ(ます|ました|ておきます|ておきました))|"
    r"いつでも[^。\n]{0,20}(呼び出せ|取り出せ|引き出せ|参照でき)(ます|る))(?!か)"
)
_PROMISE_RE = re.compile(rf"[^。\n]*(?:{SAVE_CLAIM_RE.pattern})[^。\n]*[。]?")
# 保存処理の通知文（「Knowledgeノート `….md` に保存し、判断資産候補として反映した。」「保存先: …」）。
# 2026-10-09 に紫苑が実在しないノート名まで添えていた。保存した回でだけ外す。
_SAVE_NOTICE_RE = re.compile(
    r"[^。\n]*(?:保存先|保存したもの[:：]|Knowledge\s*ノート|判断資産候補|`[^`\n]*\.md`)[^。\n]*[。]?"
)
# 「保存して」「保存する必要があるな」「残しておいて」「判断資産に入れて」のような保存の依頼。
_SAVE_REQUEST_RE = re.compile(
    r"(保存|記録|登録|永続化)(して|しと|しよう|したい|する必要|が必要|お願い|頼む)|"
    r"残して|残しと|残したい|残す必要|(判断資産|ナレッジ|Knowledge)(に|へ)(入れ|し|登録)"
)
# 依頼の対象が紫苑の回答だと分かる語。指示語だけなら従来どおり直前のユーザー発言を優先する。
_ANSWER_REFERENCE_RE = re.compile(r"テンプレ|まとめ|判断軸|チェックリスト|回答|答え|作った|作って|紫苑の|さっきの|今の(案|内容)|この(案|内容)")
SHION_ANSWER_SOURCE = "shion_answer_saved_on_request"
# Knowledge ノートの題名づけ（記憶の保存なので ai_budget の MEMORY_FEATURES に入れて止めない）。
TITLE_FEATURE = "chat_teaching_title"


def _funnel_path() -> Path:
    override = os.environ.get("SHION_TEACHING_FUNNEL_PATH", "").strip()
    return Path(override) if override else get_data_dir() / FUNNEL_LOG_NAME


def _today() -> str:
    return _dt.date.today().isoformat()


def record_funnel_event(event: str, *, surface: str, **fields: Any) -> None:
    """指標用の1行を追記する。失敗しても対話は止めない。"""
    if event not in FUNNEL_EVENTS:
        return
    row = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "date": _today(),
        "event": event,
        "surface": surface,
        **fields,
    }
    try:
        path = _funnel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def resolve_teaching_claim(message: str, previous_user_message: str = "") -> str:
    """保存する本文を決める。指示だけの短い発言なら直前のユーザー発言を本文にする。"""
    text = " ".join(str(message or "").split())
    residual = _EXPLICIT_PHRASES.sub("", text).strip(" 　、。,.!！")
    previous = " ".join(str(previous_user_message or "").split())
    if len(residual) < _ANAPHORIC_RESIDUAL_CHARS and previous:
        return f"{previous}（{text}）"[:500]
    return text[:500]


def teaching_topic(claim: str, title_maker: Callable[[str], str] | None = None) -> str:
    """Knowledge ノートの見出し（ファイル名にもなる）。

    ``title_maker`` があれば内容を表す題名を作らせる。無いか失敗したら、相づち・感想の文
    （「友人の話、それは胸が痛むね」「そうだね」）を飛ばし、審査の語を含む最初の文を24字まで使う。
    2026-10-09 時点で Knowledge ノートの多くが会話の一文を題名にしていた。
    """
    from memory_promotion_policy import has_domain_keyword

    text = _EXPLICIT_PHRASES.sub("", str(claim or "")).strip(" 　、。,.!！")
    if title_maker is not None and text:
        try:
            title = _clean_title(title_maker(text))
        except Exception as exc:  # noqa: BLE001 - 題名づけの失敗で保存は止めない
            print(f"[TeachingCapture] 題名づけに失敗: {type(exc).__name__}: {exc}")
            title = ""
        if title:
            return title
    sentences = [s.strip(" 　*#>-") for s in re.split(r"[。！？!?\n]", text) if s.strip(" 　*#>-")]
    first = next((s for s in sentences if has_domain_keyword(s)), sentences[0] if sentences else text)
    return first[:24] or "対話で教わった審査ノウハウ"


def _clean_title(raw: str) -> str:
    title = re.sub(r"[*#`「」『』\"'【】]|^(題名|タイトル)[:：]\s*", "", str(raw or "").strip().splitlines()[0] if str(raw or "").strip() else "")
    title = title.strip(" 　。、.")
    return title if 2 <= len(title) <= 30 else ""


_TITLE_INSTRUCTION = (
    "次はリース審査の対話で保存するノートの本文。内容を表す題名を、20字以内の名詞句で1つだけ返す。"
    "あいさつ・相づち・感想・会話の言い回しは使わない。例: 「経営者の離婚とリース審査」「中古トラックの見積書の注意点」。"
    "題名だけを出力する。"
)


def gemini_knowledge_title(text: str) -> str:
    """Knowledge ノートの題名を Gemini（既定モデル）で作る。失敗したら空文字（呼び出し側が文から作る）。"""
    import requests

    from ai_runtime_client import tracked_ai_http_call
    from api.secret_access import get_gemini_api_key
    from config import get_gemini_model

    api_key = get_gemini_api_key()
    if not api_key:
        return ""
    model = get_gemini_model()
    payload = {
        "system_instruction": {"parts": [{"text": _TITLE_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": str(text)[:2000]}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
    }
    resp = tracked_ai_http_call(
        lambda: requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            json=payload,
            headers={"x-goog-api-key": api_key},
            timeout=8,
        ),
        provider="google",
        model=model,
        feature=TITLE_FEATURE,
    )
    parts = resp.json()["candidates"][0]["content"]["parts"]
    return "".join(str(p.get("text") or "") for p in parts if not p.get("thought"))


def _knowledge_dir(vault: Path) -> Path:
    from lease_intelligence_mind import mind_directory

    return mind_directory(Path(vault)) / "Knowledge"


def _normalized(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def _knowledge_already_has(vault: Path, claim: str) -> str:
    target = _normalized(claim)
    directory = _knowledge_dir(vault)
    if not target or not directory.exists():
        return ""
    for path in directory.glob("*.md"):
        try:
            if target in _normalized(path.read_text(encoding="utf-8", errors="ignore")):
                return str(path)
        except OSError:
            continue
    return ""


def _vault_relative(vault: Path, path: str) -> str:
    try:
        return str(Path(path).relative_to(Path(vault)))
    except ValueError:
        return Path(path).name


def save_request_target(message: str, previous_user_message: str = "", previous_assistant_message: str = "") -> str:
    """保存の依頼が直前の紫苑の回答（テンプレート・まとめ等）を指していれば、その回答を返す。違えば空。"""
    from memory_promotion_policy import classify_lease_teaching, has_domain_keyword

    text = " ".join(str(message or "").split())
    answer = str(previous_assistant_message or "").strip()
    if not answer or len(text) > 80 or not _SAVE_REQUEST_RE.search(text) or not has_domain_keyword(answer):
        return ""
    if _ANSWER_REFERENCE_RE.search(text):
        return answer
    # 「判断資産にしておいて」だけなら、直前に教えたユーザー発言の保存（従来の経路）を優先する。
    if classify_lease_teaching(previous_user_message)[0] or len(resolve_teaching_claim(text)) > _ANAPHORIC_RESIDUAL_CHARS + 20:
        return ""
    return answer


def _answer_body(answer: str) -> str:
    """保存する回答本文。正直さの注記・保存先の行・約束文・末尾の問い返しの段落を除く。"""
    text = re.sub(r"（この発言はまだ保存していません[^）]*）|^保存(先|したもの):.*$", "", answer, flags=re.MULTILINE)
    text = _PROMISE_RE.sub("", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    while paragraphs and paragraphs[-1].rstrip().endswith(("？", "?", "か。")):
        paragraphs.pop()
    return "\n\n".join(paragraphs)[:3000]


def answer_topic(answer: str, title_maker: Callable[[str], str] | None = None) -> str:
    """回答の見出し（最初の ### 行か【】）を保存名にする。無ければ内容から題名を作る。"""
    for line in str(answer or "").splitlines():
        heading = re.match(r"\s*(#{1,4}\s*(.+)|\**【(.+?)】)", line)
        if heading:
            name = re.sub(r"^(記録内容|保存内容)[:：]\s*|[*#`]", "", (heading.group(2) or heading.group(3) or "")).strip()
            if name:
                return name[:40]
    return teaching_topic(answer, title_maker)


def _write_knowledge_and_candidate(
    claim: str,
    *,
    topic: str,
    vault: Path | None,
    candidate_saver: Callable[..., dict[str, Any]],
    date_str: str,
    saver_kwargs: dict[str, Any] | None = None,
    source_type: str = KNOWLEDGE_SOURCE_TYPE,
) -> dict[str, Any]:
    knowledge_path = ""
    knowledge_duplicate = False
    knowledge_error = ""
    if vault is not None:
        try:
            existing = _knowledge_already_has(vault, claim)
            if existing:
                knowledge_path, knowledge_duplicate = existing, True
            else:
                from lease_intelligence_mind import record_lease_knowledge

                written = record_lease_knowledge(
                    vault,
                    topic,
                    claim,
                    date_str,
                    source_type=source_type,
                    confidence=0.6,
                    verification_status="user_taught_unverified",
                )
                knowledge_path = str(written.get("path") or "")
        except Exception as exc:  # noqa: BLE001 - 保存失敗は結果に残し、対話は続ける
            knowledge_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    candidate_id = ""
    candidate_duplicate = False
    candidate_error = ""
    try:
        captured = candidate_saver(claim, **(saver_kwargs or {})) or {}
        if captured.get("captured"):
            candidate_id = str((captured.get("candidate") or {}).get("id") or "")
            candidate_duplicate = bool(captured.get("duplicate"))
        else:
            candidate_error = str(captured.get("reason") or "not_captured")
    except Exception as exc:  # noqa: BLE001
        candidate_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    return {
        "saved": bool(knowledge_path or candidate_id),
        "claim": claim,
        "knowledge_path": _vault_relative(vault, knowledge_path) if (vault is not None and knowledge_path) else "",
        "knowledge_duplicate": knowledge_duplicate,
        "candidate_id": candidate_id,
        "candidate_duplicate": candidate_duplicate,
        "errors": [e for e in (knowledge_error, candidate_error) if e],
    }


def _record_saved(result: dict[str, Any], surface: str, **fields: Any) -> None:
    if result["saved"]:
        record_funnel_event(
            "saved",
            surface=surface,
            knowledge=bool(result["knowledge_path"]),
            candidate=bool(result["candidate_id"]),
            duplicate=result["knowledge_duplicate"] and (result["candidate_duplicate"] or not result["candidate_id"]),
            # 回答本文に保存先を出さないので、どこへ保存したかはここで確かめる。
            knowledge_path=result["knowledge_path"],
            candidate_id=result["candidate_id"],
            **fields,
        )


def save_lease_teaching(
    message: str,
    *,
    vault: Path | None,
    surface: str,
    candidate_saver: Callable[..., dict[str, Any]],
    previous_user_message: str = "",
    previous_assistant_message: str = "",
    date_str: str | None = None,
    title_maker: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """教示なら Knowledge と判断資産候補へ保存し、その結果を返す（回答前に同期で呼ぶ）。

    保存の依頼（「作ったテンプレートを保存して」）が直前の紫苑の回答を指すときは、その回答を保存する。
    ``candidate_saver(claim, **kw)`` は ``capture_chat_judgment_asset_if_needed`` 形式の結果を返す
    （回答の保存では ``user_requested=True`` を渡し、教示判定を飛ばして登録させる）。
    戻り値の ``saved`` が True のときだけ、紫苑は保存したと言ってよい。
    """
    from memory_promotion_policy import classify_lease_teaching

    date_str = date_str or _today()
    answer = save_request_target(message, previous_user_message, previous_assistant_message)
    if answer:
        body = _answer_body(answer)
        topic = answer_topic(body, title_maker)
        record_funnel_event("taught", surface=surface, reason="save_request_shion_answer")
        result = _write_knowledge_and_candidate(
            f"{topic}\n\n{body}",
            topic=topic,
            vault=vault,
            candidate_saver=candidate_saver,
            date_str=date_str,
            saver_kwargs={"user_requested": True},
            source_type=SHION_ANSWER_SOURCE,
        )
        _record_saved(result, surface, source="shion_answer")
        return {"is_teaching": True, "reason": "save_request_shion_answer", "source": "shion_answer", "topic": topic, **result}

    is_teaching, reason = classify_lease_teaching(message)
    if not is_teaching:
        return {"is_teaching": False, "saved": False, "reason": reason}
    claim = resolve_teaching_claim(message, previous_user_message)
    record_funnel_event("taught", surface=surface, reason=reason)
    result = _write_knowledge_and_candidate(
        claim, topic=teaching_topic(claim, title_maker), vault=vault, candidate_saver=candidate_saver, date_str=date_str
    )
    _record_saved(result, surface)
    return {"is_teaching": True, "reason": reason, **result}


def build_save_result_prompt_block(result: dict[str, Any]) -> str:
    """保存処理の実際の結果を紫苑へ渡す。紫苑はこの結果に反することを言わない。"""
    if not result.get("is_teaching"):
        return (
            "【今回の発言の保存結果】\n"
            "この発言は保存していない（審査ノウハウの教示とは判定しなかった）。"
            "「判断資産にします」「覚えます」「記録します」「永続化します」「いつでも呼び出せます」"
            "のように保存した・するとは言わないこと。"
            + _NO_INVENTED_PLACE
        )
    if not result.get("saved"):
        return (
            "【今回の発言の保存結果】\n"
            "審査ノウハウとして受け取ったが、保存に失敗した。保存したとは言わず、"
            "保存できなかったことを一言伝えること。"
        )
    places = []
    if result.get("knowledge_path"):
        places.append(f"Knowledgeノート `{result['knowledge_path']}`")
    if result.get("candidate_id"):
        places.append("判断資産候補（/judgment-review の要確認）")
    already = "（同じ内容が既に保存済み）" if result.get("knowledge_duplicate") else ""
    what = f"保存したもの: 直前の紫苑の回答「{result.get('topic')}」。" if result.get("source") == "shion_answer" else ""
    return (
        "【今回の発言の保存結果】\n"
        f"{what}保存済み{already}: {'・'.join(places)}。"
        "保存・反映したこと、保存先、判断資産候補のことは回答に書かない（処理結果はシステムが別に記録する。"
        "ユーザーは本文に処理の通知を出さないよう求めている）。内容にだけ普段どおり答えること。"
    )


_NO_INVENTED_PLACE = "保存先は Knowledge ノートと判断資産候補だけ。「〇〇集」「〇〇データベース」など無い機能名を作らないこと。"


def _drop_save_notices(text: str, knowledge_path: str = "") -> str:
    """保存の約束文・保存先の通知文を外す（保存した結果は ``teaching_save`` と指標ログに残る）。"""
    text = _PROMISE_RE.sub("", text)
    text = _SAVE_NOTICE_RE.sub("", text)
    if knowledge_path:
        text = re.sub(rf"[^。\n]*{re.escape(knowledge_path)}[^。\n]*[。]?", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def enforce_save_honesty(reply: str, result: dict[str, Any]) -> str:
    """保存の成否と食い違う約束を回答から外す。保存したときも保存の通知は本文に出さない。

    2026-10-09 ユーザー方針「判断資産候補（要確認）などは表示しなくていい、処理してくれれば」。
    以前は保存先を本文に添えていたが、保存結果は応答の ``teaching_save`` と指標ログ
    （``data/shion_teaching_funnel.jsonl`` の saved 行）でだけ確かめる。
    """
    text = str(reply or "")
    if not result.get("saved"):
        stripped = _PROMISE_RE.sub("", text).strip()
        if stripped != text.strip():
            note = (
                "（この発言はまだ保存していません。"
                "残したい場合は「判断資産に入れて」と送ってください。）"
            )
            return f"{stripped}\n\n{note}".strip()
        return text
    return _drop_save_notices(text, str(result.get("knowledge_path") or "")) or text.strip()


def _ngrams(text: str, n: int = 2) -> set[str]:
    compact = re.sub(r"[\s、。,.()（）「」『』:：/・*#>\-`]", "", str(text or ""))
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _note_body(text: str) -> str:
    parts = text.split("---", 2)
    body = parts[2] if text.startswith("---") and len(parts) == 3 else text
    lines = [
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.startswith(("#", ">", "- source_type", "- confidence", "- verification_status"))
    ]
    return " ".join(lines)


# どの審査の文にも出るため、想起の決め手にしない語。
_GENERIC_DOMAIN_TERMS = frozenset({"リース", "審査", "契約", "取引", "承認", "設備", "業界", "業種"})
# 言い方の違う同じ概念（想起だけで使う）。2026-10-04 に保存した多角化案件の審査コメントテンプレートが
# 「新規参入」「審査意見」のような問いで拾えなかった。
_RECALL_CONCEPTS = (
    ("多角化", re.compile(r"多角化|異業種|新規事業|新事業|新規参入|参入|進出|本業(以外|外|とは別|と別)")),
    ("審査コメント", re.compile(r"審査コメント|審査意見|稟議コメント|テンプレート|テンプレ")),
)


def _score_against(query: str, query_grams: set[str], body: str) -> float | None:
    """共有するリース審査語（汎用語を除く）を主、bigram 一致率を従にした一致度。"""
    grams = _ngrams(body)
    if not grams:
        return None
    # 文字 bigram だけだと「する」「車の」のような汎用断片で決まり、短いノートが拾えなかった。
    shared_terms = {
        term
        for term in TEACHING_DOMAIN_TERMS
        if term not in _GENERIC_DOMAIN_TERMS and term in query and term in body
    } | {name for name, pattern in _RECALL_CONCEPTS if pattern.search(query) and pattern.search(body)}
    overlap = len(query_grams & grams) / len(query_grams)
    if not shared_terms or (len(shared_terms) < 2 and overlap < 0.2):
        return None
    return len(shared_terms) + overlap


def _chat_taught_candidates() -> list[dict[str, Any]]:
    """チャットで教わった判断資産候補（要確認のまま）を読む。過去ログから救済した分を含む。"""
    path = get_data_dir() / "autoresearch_judgment_asset_candidates.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("research_topic") == "chat_judgment_teaching":
            rows.append(row)
    return rows


def recall_taught_knowledge(
    vault: Path | None,
    query: str,
    *,
    limit: int = 3,
    candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Knowledge ノートと、チャットで教わった判断資産候補から問いに近いものを返す（回答前の想起用）。"""
    query_grams = _ngrams(query)
    if len(query_grams) < 3:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    directory = _knowledge_dir(vault) if vault is not None else None
    if directory is not None and directory.exists():
        for path in directory.glob("*.md"):
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            body = _note_body(raw)
            score = _score_against(query, query_grams, body)
            if score is None:
                continue
            seen.add(_normalized(body)[:80])
            scored.append(
                (
                    score,
                    {
                        "path": _vault_relative(vault, str(path)),
                        "topic": path.stem,
                        "snippet": body[:220],
                        "score": round(score, 3),
                        "user_taught": KNOWLEDGE_SOURCE_TYPE in raw or "user_teaching" in raw,
                        "knowledge_kind": classify_knowledge_kind(body),
                        "citation": f"Knowledge ノート「{path.stem}」",
                    },
                )
            )
    for row in _chat_taught_candidates() if candidates is None else candidates:
        claim = str(row.get("edited_claim") or row.get("claim") or "")
        if not claim or _normalized(claim)[:80] in seen:
            continue
        score = _score_against(query, query_grams, claim)
        if score is None:
            continue
        seen.add(_normalized(claim)[:80])
        scored.append(
            (
                score,
                {
                    "path": f"judgment_candidate:{row.get('id')}",
                    "topic": f"{row.get('research_date') or ''}に教わった判断（要確認）",
                    "snippet": claim[:220],
                    "score": round(score, 3),
                    "user_taught": True,
                    "knowledge_kind": knowledge_kind_of(row, claim),
                    "citation": asset_citation(str(row.get("id") or ""), str(row.get("research_date") or ""), label="判断資産候補"),
                },
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in scored[:limit]]


def build_recall_prompt_block(items: list[dict[str, Any]], rag_hits: list[dict[str, Any]] | None = None) -> str:
    policies = [
        {"text": item["snippet"], "source": item.get("citation") or item["topic"]}
        for item in items
        if item.get("user_taught") and (item.get("knowledge_kind") or classify_knowledge_kind(item["snippet"])) == POLICY
    ]
    policy_texts = {p["text"] for p in policies}
    lines: list[str] = []
    for item in items:
        if item["snippet"] in policy_texts:
            continue
        cite = f"（出典: {item['citation']}）" if item.get("citation") else ""
        lines.append(f"- ユーザーが教えた知識（{item['topic']}）: {item['snippet']}{cite}")
    for hit in rag_hits or []:
        snippet = " ".join(str(hit.get("text") or "").split())[:200]
        if snippet:
            lines.append(f"- 参照ナレッジ（{hit.get('source') or hit.get('title') or 'RAG'}）: {snippet}")
    policy_block = format_policy_block(policies)
    if not lines:
        return policy_block
    block = (
        "【回答前に想起した知識】\n"
        "以下は回答前に検索した保存済み知識。関係があれば優先して使い、使った時は"
        "「以前教わった〇〇」のように出所を一言添える。関係がなければ無理に使わない。"
        + INSIGHT_CITATION_INSTRUCTION
        + "\n"
        + "\n".join(lines)
    )
    return f"{policy_block}\n\n{block}" if policy_block else block


def used_in_reply(item: dict[str, Any], reply: str) -> bool:
    """想起した知識が回答に反映されたかの粗い判定（指標用）。"""
    grams = _ngrams(item.get("snippet") or "", 3)
    if not grams:
        return False
    return len(grams & _ngrams(reply, 3)) / len(grams) >= 0.15


def previous_user_message(history: list[dict[str, Any]]) -> str:
    """会話履歴（今回の発言は未保存）から直前のユーザー発言を返す。指示語だけの教示の本文に使う。"""
    return next(
        (str(m.get("content") or "") for m in reversed(history or []) if str(m.get("role") or "") == "user"),
        "",
    )


def previous_assistant_message(history: list[dict[str, Any]]) -> str:
    """会話履歴から直前の紫苑の回答を返す。「作ったテンプレートを保存して」の保存対象に使う。"""
    return next(
        (str(m.get("content") or "") for m in reversed(history or []) if str(m.get("role") or "") == "assistant"),
        "",
    )


@dataclass
class TeachingTurn:
    """1回の発言についての 保存→想起 の結果。回答前に作り、回答後に ``finalize`` する。

    対話室（/api/lease-intelligence/dialogue）と通常チャット（/api/chat）で共通に使う。
    """

    surface: str
    save: dict[str, Any] = field(default_factory=lambda: {"is_teaching": False, "saved": False, "reason": "not_run"})
    recall_items: list[dict[str, Any]] = field(default_factory=list)
    rag_hits: list[dict[str, Any]] = field(default_factory=list)

    @property
    def save_context(self) -> str:
        return build_save_result_prompt_block(self.save)

    @property
    def recall_context(self) -> str:
        return build_recall_prompt_block(self.recall_items, self.rag_hits[:3])

    def finalize(self, reply: str) -> str:
        """保存の成否に合わせて回答を正し、想起した知識が使われたかを指標に残す。"""
        fixed = enforce_save_honesty(reply, self.save)
        for item in self.recall_items:
            if used_in_reply(item, fixed):
                record_funnel_event(
                    "used",
                    surface=self.surface,
                    path=item.get("path"),
                    user_taught=bool(item.get("user_taught")),
                )
        return fixed

    def response_extra(self) -> dict[str, Any]:
        return {
            "teaching_save": self.save,
            "pre_recall": [
                {"path": item.get("path"), "score": item.get("score"), "user_taught": item.get("user_taught")}
                for item in self.recall_items
            ],
        }


def vector_store_rag_search(surface: str) -> Callable[[str], list[dict[str, Any]]]:
    """``prepare_teaching_turn`` の ``rag_search`` に渡すローカル RAG 検索。"""

    def _search(query: str) -> list[dict[str, Any]]:
        from api.knowledge.vector_store import get_store

        return get_store().search(query, top_k=5, surface=surface)

    return _search


def prepare_teaching_turn(
    message: str,
    *,
    vault: Path | None,
    surface: str,
    candidate_saver: Callable[..., dict[str, Any]],
    previous_user_message: str = "",
    previous_assistant_message: str = "",
    rag_search: Callable[[str], list[dict[str, Any]]] | None = None,
    allow_save: bool = True,
    title_maker: Callable[[str], str] | None = None,
) -> TeachingTurn:
    """回答前に、教示なら保存し、審査の問いなら Knowledge・教わった候補・RAG を想起する。

    ``rag_search`` は RAG を別途引かない経路だけ渡す（通常チャットの RAG 分岐は自前で引く）。
    ``allow_save=False`` は想起だけ行う（紫苑レビューの依頼文は教示ではないので保存しない）。
    失敗しても対話は止めない（保存できなかった扱いにして、約束文は外れる）。
    """
    from memory_promotion_policy import has_domain_keyword

    turn = TeachingTurn(surface=surface)
    try:
        turn.save = {"is_teaching": False, "saved": False, "reason": "save_disabled"} if not allow_save else save_lease_teaching(
            message,
            vault=vault,
            surface=surface,
            candidate_saver=candidate_saver,
            previous_user_message=previous_user_message,
            previous_assistant_message=previous_assistant_message,
            title_maker=title_maker,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[TeachingCapture] 保存判定に失敗: {type(exc).__name__}: {exc}")
        return turn
    if not has_domain_keyword(message) or turn.save.get("is_teaching"):
        return turn
    try:
        turn.recall_items = recall_taught_knowledge(vault, message, limit=3)
    except Exception as exc:  # noqa: BLE001
        print(f"[TeachingCapture] 想起に失敗: {type(exc).__name__}: {exc}")
    if rag_search is not None:
        try:
            turn.rag_hits = list(rag_search(message) or [])
        except Exception as exc:  # noqa: BLE001
            print(f"[TeachingCapture] RAG検索に失敗: {exc}")
    for item in turn.recall_items:
        record_funnel_event(
            "recalled",
            surface=surface,
            path=item.get("path"),
            user_taught=bool(item.get("user_taught")),
        )
    return turn


# 朝報の経路別内訳。surface の接頭辞で束ねる（通常チャットは general/rag 分岐ごとに surface が違う）。
FUNNEL_SURFACE_LABELS = (
    ("lease_intelligence_dialogue", "対話室"),
    ("next_chat", "通常チャット"),
)


def _surface_label(surface: str) -> str:
    return next((label for prefix, label in FUNNEL_SURFACE_LABELS if surface.startswith(prefix)), "その他")


def funnel_summary(date_str: str | None = None, *, path: Path | None = None) -> dict[str, Any]:
    """指定日の 教えた→保存→想起→使用 の件数と累計を返す（朝報用）。"""
    target = path or _funnel_path()
    day: Counter[str] = Counter()
    total: Counter[str] = Counter()
    day_by_surface: dict[str, Counter[str]] = {}
    if target.exists():
        for line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = str(row.get("event") or "")
            if event not in FUNNEL_EVENTS:
                continue
            total[event] += 1
            if date_str and str(row.get("date") or "") == date_str:
                day[event] += 1
                label = _surface_label(str(row.get("surface") or ""))
                day_by_surface.setdefault(label, Counter())[event] += 1
    return {
        "date": date_str or "",
        "day": {event: day.get(event, 0) for event in FUNNEL_EVENTS},
        "total": {event: total.get(event, 0) for event in FUNNEL_EVENTS},
        "day_by_surface": {
            label: {event: counts.get(event, 0) for event in FUNNEL_EVENTS}
            for label, counts in day_by_surface.items()
        },
        "path": str(target),
    }
