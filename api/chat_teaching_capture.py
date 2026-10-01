"""チャットで教わった審査ノウハウを、回答前に決定的に保存・想起する。

2026-10 の調査で、対話室で教えたノウハウの大半が保存されていなかった。保存は回答後の
バックグラウンド処理と LLM 抽出任せで、雑談モードではツールも使わない。そのうえ紫苑は
保存していないのに「判断資産にします」と答えていた。ここでは次を LLM に頼らず行う。

* 判定: ``memory_promotion_policy.classify_lease_teaching``（1段）
* 保存: Obsidian ``Lease Intelligence/Knowledge/`` のノートと、判断資産候補（要確認）
* 想起: Knowledge ノートからの文字 n-gram 一致（回答前にプロンプトへ入れる）
* 正直さ: 保存の成否に合わない「判断資産にします」等を回答から外し、保存先を添える
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
_PROMISE_RE = re.compile(
    r"[^。\n]*(判断資産(に|として)[^。\n]{0,12}(します|しました|登録|追加|記録|残し)|"
    r"覚えます|覚えました|覚えておきます|記録します|記録しました|記録しておきます|"
    r"保存します|保存しました|登録します|登録しました|残しておきます)[^。\n]*[。]?"
)


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


def teaching_topic(claim: str) -> str:
    """Knowledge ノートの見出し。指示語を除いた最初の節を24字まで。"""
    text = _EXPLICIT_PHRASES.sub("", str(claim or "")).strip(" 　、。,.!！")
    first = re.split(r"[。\n]", text, maxsplit=1)[0].strip() or text
    return first[:24] or "対話で教わった審査ノウハウ"


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


def save_lease_teaching(
    message: str,
    *,
    vault: Path | None,
    surface: str,
    candidate_saver: Callable[[str], dict[str, Any]],
    previous_user_message: str = "",
    date_str: str | None = None,
) -> dict[str, Any]:
    """教示なら Knowledge と判断資産候補へ保存し、その結果を返す（回答前に同期で呼ぶ）。

    ``candidate_saver(claim)`` は ``capture_chat_judgment_asset_if_needed`` 形式の結果を返す。
    戻り値の ``saved`` が True のときだけ、紫苑は保存したと言ってよい。
    """
    from memory_promotion_policy import classify_lease_teaching

    is_teaching, reason = classify_lease_teaching(message)
    if not is_teaching:
        return {"is_teaching": False, "saved": False, "reason": reason}
    claim = resolve_teaching_claim(message, previous_user_message)
    record_funnel_event("taught", surface=surface, reason=reason)
    date_str = date_str or _today()

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
                    teaching_topic(claim),
                    claim,
                    date_str,
                    source_type=KNOWLEDGE_SOURCE_TYPE,
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
        captured = candidate_saver(claim) or {}
        if captured.get("captured"):
            candidate_id = str((captured.get("candidate") or {}).get("id") or "")
            candidate_duplicate = bool(captured.get("duplicate"))
        else:
            candidate_error = str(captured.get("reason") or "not_captured")
    except Exception as exc:  # noqa: BLE001
        candidate_error = f"{type(exc).__name__}: {str(exc)[:120]}"

    saved = bool(knowledge_path or candidate_id)
    result = {
        "is_teaching": True,
        "saved": saved,
        "reason": reason,
        "claim": claim,
        "knowledge_path": _vault_relative(vault, knowledge_path) if (vault is not None and knowledge_path) else "",
        "knowledge_duplicate": knowledge_duplicate,
        "candidate_id": candidate_id,
        "candidate_duplicate": candidate_duplicate,
        "errors": [e for e in (knowledge_error, candidate_error) if e],
    }
    if saved:
        record_funnel_event(
            "saved",
            surface=surface,
            knowledge=bool(knowledge_path),
            candidate=bool(candidate_id),
            duplicate=knowledge_duplicate and (candidate_duplicate or not candidate_id),
        )
    return result


def build_save_result_prompt_block(result: dict[str, Any]) -> str:
    """保存処理の実際の結果を紫苑へ渡す。紫苑はこの結果に反することを言わない。"""
    if not result.get("is_teaching"):
        return (
            "【今回の発言の保存結果】\n"
            "この発言は保存していない（審査ノウハウの教示とは判定しなかった）。"
            "「判断資産にします」「覚えます」「記録します」とは言わないこと。"
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
    return (
        "【今回の発言の保存結果】\n"
        f"保存済み{already}: {'・'.join(places)}。"
        "保存したことと保存先を一言だけ添えてよい。判断資産として正式採用されたとは言わない"
        "（人のレビュー待ちの候補である）。"
    )


def enforce_save_honesty(reply: str, result: dict[str, Any]) -> str:
    """保存の成否と食い違う約束を回答から外し、保存したときは保存先を添える。"""
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
    if "保存先" in text or (result.get("knowledge_path") and result["knowledge_path"] in text):
        return text
    places = []
    if result.get("knowledge_path"):
        places.append(f"`{result['knowledge_path']}`")
    if result.get("candidate_id"):
        places.append("判断資産候補（要確認）")
    return f"{text.rstrip()}\n\n保存先: {'・'.join(places)}"


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
    }
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
                },
            )
        )
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in scored[:limit]]


def build_recall_prompt_block(items: list[dict[str, Any]], rag_hits: list[dict[str, Any]] | None = None) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"- ユーザーが教えた知識（{item['topic']}）: {item['snippet']}")
    for hit in rag_hits or []:
        snippet = " ".join(str(hit.get("text") or "").split())[:200]
        if snippet:
            lines.append(f"- 参照ナレッジ（{hit.get('source') or hit.get('title') or 'RAG'}）: {snippet}")
    if not lines:
        return ""
    return (
        "【回答前に想起した知識】\n"
        "以下は回答前に検索した保存済み知識。関係があれば優先して使い、使った時は"
        "「以前教わった〇〇」のように出所を一言添える。関係がなければ無理に使わない。\n"
        + "\n".join(lines)
    )


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
    candidate_saver: Callable[[str], dict[str, Any]],
    previous_user_message: str = "",
    rag_search: Callable[[str], list[dict[str, Any]]] | None = None,
) -> TeachingTurn:
    """回答前に、教示なら保存し、審査の問いなら Knowledge・教わった候補・RAG を想起する。

    ``rag_search`` は RAG を別途引かない経路だけ渡す（通常チャットの RAG 分岐は自前で引く）。
    失敗しても対話は止めない（保存できなかった扱いにして、約束文は外れる）。
    """
    from memory_promotion_policy import has_domain_keyword

    turn = TeachingTurn(surface=surface)
    try:
        turn.save = save_lease_teaching(
            message,
            vault=vault,
            surface=surface,
            candidate_saver=candidate_saver,
            previous_user_message=previous_user_message,
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
