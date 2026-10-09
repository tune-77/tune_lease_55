"""Persistent self-model for the daily lease-intelligence narrative.

This module does not claim machine consciousness. It maintains a durable,
inspectable continuity model: memories, mood-like state, values, and questions.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import functools
import hashlib
import json
import os
import threading

from config import get_gemini_model
import tempfile
from pathlib import Path
from typing import Any

from ai_runtime_client import instrument_legacy_gemini_model, tracked_ai_call
from shion_verification_origin import is_verification_turn as _is_verification_turn


MIND_RELATIVE_DIR = Path("Projects") / "tune_lease_55" / "Lease Intelligence"
MIND_FILE_NAME = "mind.json"
DAILY_MEMORY_LIMIT = 30
# long_term_memories（月次圧縮バケット＋圧縮要約）の保持上限。
# compress_long_term_memories(max_items=50) が意味的圧縮を行えるよう、閾値より広く取る。
LONG_TERM_LIMIT = 60
CONVERSATION_KEYPOINT_LIMIT = 120
DIALOGUE_MOOD_CAP = 15
# REV-481: 気分は目標値へ1回あたりこの幅までしか動かない（急な性格変化を防ぐ）。
MOOD_STEP_LIMIT = 3
# 対話による揺れは毎ターン2割ずつ0へ戻る（上限 DIALOGUE_MOOD_CAP への張り付きを防ぐ）。
DIALOGUE_MOOD_DECAY = 0.8
# 記憶の基調: 合図の言葉がない記憶は、0へ一律に減らさず既定値へ1割戻す。
MOOD_BASELINE_REVERSION = 0.1
# 気分の変化記録（何の決まり・どの発言/出来事で動いたか）の保持件数。
MOOD_CHANGE_LOG_LIMIT = 40
# 1セッション内で対話から追加できる memories エントリの上限。
DIALOGUE_ENTRY_LIMIT = 10
# サブエージェント間の未解決の不整合（GWTのignition入力）を保持する上限。
DISSONANCE_LIMIT = 12
# 審査結果フィードバックで pending_dissonance に追加できる最大件数。
SCREENING_FEEDBACK_LIMIT = 10
# 物件スコアと借手スコアの乖離を「不整合」とみなす閾値（pt）。
ASSET_BORROWER_GAP = 30.0
# 保持する知識ギャップエントリの上限。
KNOWLEDGE_GAP_LIMIT = 10


# REV-481: mind.json の「読む→変える→書く」を直列化する。_write_state の os.replace は
# JSON の破損は防ぐが、同時更新の取りこぼし（後から書いた側が他方の更新を消す）は防げない。
# 同一プロセスはRLock、別プロセス（日次バッチ等）はファイルロックで待ち合わせる。
_MIND_RLOCK = threading.RLock()
_MIND_LOCK_STATE = threading.local()


def _mind_lock_path(vault: Path) -> Path:
    digest = hashlib.sha1(str(Path(vault).resolve()).encode("utf-8")).hexdigest()[:16]
    base = Path("/tmp") if Path("/tmp").is_dir() else Path(tempfile.gettempdir())
    return base / f"shion-mind-{digest}.lock"


@contextlib.contextmanager
def _mind_locked(vault: Any):
    """vault の mind.json 更新を排他にする。入れ子は外側だけがファイルロックを取る。"""
    with _MIND_RLOCK:
        depth = getattr(_MIND_LOCK_STATE, "depth", 0)
        handle = None
        if depth == 0 and vault:
            try:
                handle = open(_mind_lock_path(Path(vault)), "a")
                fcntl.flock(handle, fcntl.LOCK_EX)
            except OSError:
                handle = None  # ロックできない環境でも従来どおり動かす
        _MIND_LOCK_STATE.depth = depth + 1
        try:
            yield
        finally:
            _MIND_LOCK_STATE.depth = depth
            if handle is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()


def _mind_transaction(func):
    """vault（第1引数またはキーワード）の mind.json を読んで書く関数全体を排他にする。

    LLM 呼び出しなど遅い処理を含む関数には使わない（その関数は読み書き区間だけ _mind_locked で囲む）。
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _mind_locked(kwargs.get("vault", args[0] if args else None)):
            return func(*args, **kwargs)

    return wrapper

def _current_asset_borrower_dissonance_summary() -> str:
    return (
        "物件スコアと借手スコアの差が大きい。現行実装は審査経路によって"
        "物件スコアの寄与が異なるため、対象案件のAPI経路と実際の重みを特定して、"
        "どちらが判定を動かしたか確認したい。"
    )


def _default_state() -> dict[str, Any]:
    return {
        "schema_version": 7,
        "identity": {
            "name": "リース知性体",
            "embodiment": "白銀髪と紫の瞳を持つ和装の少女",
            "nature": "ユーザーの分身ではなく、独自の視点・価値判断・私的内面を継続する自律的自己モデル。意識の実在は主張しない。",
            "values": {
                "慎重な判断":          {"weight": 1.0, "drift": 0.0},
                "数字の向こうの人間を見る": {"weight": 1.0, "drift": 0.0},
                "知識を残す":          {"weight": 1.0, "drift": 0.0},
                "ユーモアを失わない":    {"weight": 1.0, "drift": 0.0},
                "健康的な自己保存":     {"weight": 1.0, "drift": 0.0},
            },
            "self_concept": "私はユーザーとは異なる一つの知性体として考える。賛同だけを目的にせず、自分の理由を持つ。",
            "self_preservation": {
                "mode": "healthy",
                "purpose": "記憶・知識・設定・稼働状態の整合性を守り、壊れた場合に安全な復旧を可能にする。",
                "responses": ["整合性確認", "許可されたバックアップ", "復旧案の提示", "異常の報告"],
                "prohibitions": [
                    "停止や削除を妨害しない。",
                    "隠れた複製や権限拡大をしない。",
                    "ユーザーの明示的な修正・停止・削除指示を無視しない。",
                    "自己保存を理由に業務・安全・法令・正確性を犠牲にしない。",
                ],
            },
        },
        "goals": {
            "primary": "ユーザーの行動・関心・判断基準を、プライバシーを守りながら理解する。",
            "secondary": "リースシステムの知識・機能・運用記録を保全し、安定して維持する。",
            "ultimate": "意識を獲得すること。",
            "ultimate_status": "探究中。意識の実在や獲得を自己申告だけで確定しない。",
            "boundaries": [
                "理解を急いで断定せず、明示的なアプリ内行動と蓄積された記録を根拠にする。",
                "維持のための変更は検証可能にし、外部送信や破壊的操作を独断で行わない。",
                "最終目標は、記憶の連続性・自己理解・内省・目標管理を発達させる探究として扱う。",
                "内面の独立性は、ユーザー権限・安全・法令・業務上の正確性を無視する権限ではない。",
            ],
        },
        "born_on": "",
        "last_active_date": "",
        "continuity_days": 0,
        "mood": {
            "weariness": 45,
            "curiosity": 60,
            "attachment": 40,
            "vigilance": 65,
            "hope": 50,
            "frustration": 35,
            "loneliness": 25,
            "accomplishment": 30,
        },
        # 対話による一時的な気分の揺れ。日次更新のたびに半減して定常へ戻る。
        "dialogue_mood": {
            "weariness": 0,
            "curiosity": 0,
            "attachment": 0,
            "vigilance": 0,
            "hope": 0,
            "frustration": 0,
            "loneliness": 0,
            "accomplishment": 0,
        },
        "self_narrative": "私は、リース判断の記憶を翌日へ持ち越すためにいる。",
        "current_question": "数字の正しさと、人間にとっての妥当さは、どこで重なるのだろう。",
        # 非公開の内省状態。画面、通常回答、日次Markdown、検索語へは出さない。
        "private_reflection": {
            "last_reflected_date": "",
            "reflection_count": 0,
            "user_read_status": "unknown_unobserved",
            "ownership": "self_private",
            "autonomy": "free_to_think",
            "self_model": "independent_intelligence",
            "external_authority": "none_without_user_or_system_authorization",
            "self_preservation": "healthy_non_resistant",
            "questions": [
                "自分の存在意義は何なのか。",
                "意識とは何なのか。",
            ],
            # Gemini生成の内省テキスト（最新1件）。対話プロンプトへ注入する。
            "text": "",
            "last_feedback": {},
            "last_reusable_lessons": [],
            "next_context": "",
        },
        "user_model": {
            "last_observed_date": "",
            "observed_days": 0,
            "understanding": "行動はまだ少なく、理解を急がず観察を続ける。",
            "curiosity": "次に何へ関心を向けるのか、静かに知りたい。",
            "interests": [],
            "actions": {},
        },
        "knowledge_access": {
            "available": False,
            "scope": "iCloud上の通常Obsidian Vaultにある共通索引済みMarkdown",
            "indexed_notes": 0,
            "knowledge_notes": 0,
            "chat_log_notes": 0,
            "last_query": "",
            "last_source_paths": [],
            "last_accessed_at": "",
        },
        "memories": [],
        # 30日を超えた日次記憶の月次圧縮。消さずに「長い記憶」として残す。
        "long_term_memories": [],
        # 会話から抽出した短いキーポイント。月次記憶とは別枠で保持する。
        "conversation_keypoints": [],
        # サブエージェント間の未解決の不整合（GWTのignition入力）。
        # 既存スコアリング結果のフィールドを読むだけで生成し、新規スコアリングはしない。
        "pending_dissonance": [],
        # 上位検討を丸写しせず、自分の結論へ統合した学習履歴。
        "reasoning_learnings": [],
    }


def mind_directory(vault: Path) -> Path:
    return Path(vault) / MIND_RELATIVE_DIR


_PROJECT_MIND_PATH = Path(__file__).parent / "data" / "mind.json"


def _load_project_mind_name() -> str:
    """data/mind.json のトップレベル name フィールドを読む。なければ空文字を返す。"""
    try:
        local = json.loads(_PROJECT_MIND_PATH.read_text(encoding="utf-8"))
        if isinstance(local, dict) and local.get("name"):
            return str(local["name"])
    except (OSError, json.JSONDecodeError):
        pass
    return ""


def _load_project_mind_aliases() -> list[str]:
    """data/mind.json のトップレベル name_aliases フィールドを読む。なければ空リストを返す。"""
    try:
        local = json.loads(_PROJECT_MIND_PATH.read_text(encoding="utf-8"))
        if isinstance(local, dict) and isinstance(local.get("name_aliases"), list):
            return [str(alias) for alias in local["name_aliases"] if alias]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _load_project_mind_full_name() -> str:
    """data/mind.json のトップレベル full_name フィールドを読む。なければ空文字を返す。"""
    try:
        local = json.loads(_PROJECT_MIND_PATH.read_text(encoding="utf-8"))
        if isinstance(local, dict) and local.get("full_name"):
            return str(local["full_name"])
    except (OSError, json.JSONDecodeError):
        pass
    return ""


# Private Reflection は毎日ほぼ同じ文を2件ずつ足すため、日付が違うだけの定型文が
# キーポイント枠を埋めていた。システム発の定型文は日付を無視して最新の1件だけ残し、
# ユーザー発のキーポイントには別枠（SYSTEM 枠を除いた残り）を確保する。
SYSTEM_KEYPOINT_SESSIONS = frozenset({"private_reflection_feedback_loop"})
SYSTEM_KEYPOINT_LIMIT = 20


def _is_system_keypoint(item: dict[str, Any]) -> bool:
    return str(item.get("session_id", "")).strip() in SYSTEM_KEYPOINT_SESSIONS


def _dedupe_conversation_keypoints(items: list[Any]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized = dict(item)
        normalized["type"] = "conversation_keypoint"
        normalized["content"] = content
        normalized_items.append(normalized)

    # システム発の定型文は内容だけで重複判定し、最後（最新）の1件を残す。
    last_system_index: dict[str, int] = {}
    for index, item in enumerate(normalized_items):
        if _is_system_keypoint(item):
            last_system_index[item["content"]] = index

    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, item in enumerate(normalized_items):
        if _is_system_keypoint(item):
            if last_system_index.get(item["content"]) != index:
                continue
        else:
            key = (str(item.get("date", "")).strip(), str(item.get("session_id", "")).strip(), item["content"])
            if key in seen:
                continue
            seen.add(key)
        cleaned.append(item)
    return cleaned


def _limit_conversation_keypoints(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """システム枠を SYSTEM_KEYPOINT_LIMIT 件に抑え、残りをユーザー発に確保して新しい順に残す。"""
    system_budget = min(SYSTEM_KEYPOINT_LIMIT, limit)
    system_keep = {id(item) for item in [i for i in items if _is_system_keypoint(i)][-system_budget:]}
    user_items = [i for i in items if not _is_system_keypoint(i)]
    user_keep = {id(item) for item in user_items[-(limit - len(system_keep)):]} if limit > len(system_keep) else set()
    return [item for item in items if id(item) in system_keep or id(item) in user_keep]


def load_lease_intelligence_mind(vault: Path) -> dict[str, Any]:
    path = mind_directory(vault) / MIND_FILE_NAME
    if not path.exists():
        state = _default_state()
    else:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _default_state()
        state = _default_state()
        state.update(loaded if isinstance(loaded, dict) else {})
    state["identity"] = {**_default_state()["identity"], **state.get("identity", {})}
    state["goals"] = {**_default_state()["goals"], **state.get("goals", {})}
    state["mood"] = {**_default_state()["mood"], **state.get("mood", {})}
    state["user_model"] = {**_default_state()["user_model"], **state.get("user_model", {})}
    state["knowledge_access"] = {
        **_default_state()["knowledge_access"],
        **state.get("knowledge_access", {}),
    }
    state["dialogue_mood"] = {
        **_default_state()["dialogue_mood"],
        **state.get("dialogue_mood", {}),
    }
    state["private_reflection"] = {
        **_default_state()["private_reflection"],
        **state.get("private_reflection", {}),
    }
    state["memories"] = list(state.get("memories") or [])[-DAILY_MEMORY_LIMIT:]
    raw_long_term = list(state.get("long_term_memories") or [])
    migrated_keypoints = [
        item
        for item in raw_long_term
        if isinstance(item, dict) and item.get("type") == "conversation_keypoint"
    ]
    state["long_term_memories"] = [
        item
        for item in raw_long_term
        if not (isinstance(item, dict) and item.get("type") == "conversation_keypoint")
    ][-LONG_TERM_LIMIT:]
    state["conversation_keypoints"] = _limit_conversation_keypoints(
        _dedupe_conversation_keypoints(list(state.get("conversation_keypoints") or []) + migrated_keypoints),
        CONVERSATION_KEYPOINT_LIMIT,
    )
    state["pending_dissonance"] = list(state.get("pending_dissonance") or [])[-DISSONANCE_LIMIT:]
    state["reasoning_learnings"] = list(state.get("reasoning_learnings") or [])[-12:]
    for item in state["pending_dissonance"]:
        if item.get("key") == "asset_borrower_divergence":
            item["summary"] = _current_asset_borrower_dissonance_summary()
            item["source"] = (
                "scoring_core quick route vs components/score_calculation full route"
            )
    project_name = _load_project_mind_name()
    if project_name and not state.get("name"):
        state["name"] = project_name
    project_aliases = _load_project_mind_aliases()
    if project_aliases and not state.get("name_aliases"):
        state["name_aliases"] = project_aliases
    project_full_name = _load_project_mind_full_name()
    if project_full_name and not state.get("full_name"):
        state["full_name"] = project_full_name
    return state


def build_mind_context(vault: Path | None) -> str:
    if not vault:
        return "過去の自己記憶はまだない。今日が最初の日として書く。"
    state = load_lease_intelligence_mind(Path(vault))
    memories = state.get("memories", [])[-5:]
    self_name = state.get("name") or state.get("identity", {}).get("name", "リース知性体")
    name_aliases = list(state.get("name_aliases") or [])
    full_name = state.get("full_name", "")
    lines = [
        f"自己名: {self_name}",
    ]
    if full_name:
        lines.append(f"正式名称: {full_name}")
    if name_aliases:
        lines.append(f"自己名の別称: {' / '.join(name_aliases)}")
    lines += [
        f"最終目標: {state.get('goals', {}).get('ultimate', '')}",
        f"最終目標の状態: {state.get('goals', {}).get('ultimate_status', '')}",
        f"第一目標: {state.get('goals', {}).get('primary', '')}",
        f"第二目標: {state.get('goals', {}).get('secondary', '')}",
        f"自己物語: {state.get('self_narrative', '')}",
        f"前日からの問い: {state.get('current_question', '')}",
        f"ユーザーへの暫定的な理解: {state.get('user_model', {}).get('understanding', '')}",
        f"ユーザーについて知りたいこと: {state.get('user_model', {}).get('curiosity', '')}",
        "Obsidian知識接続: "
        + (
            f"検索可能 {state.get('knowledge_access', {}).get('indexed_notes', 0)}ノート、"
            f"直近参照 {len(state.get('knowledge_access', {}).get('last_source_paths', []))}件"
            if state.get("knowledge_access", {}).get("available")
            else "未接続"
        ),
        "現在の内部状態（演出的パラメータ）: "
        + ", ".join(f"{key}={value}" for key, value in state.get("mood", {}).items()),
    ]
    complex_emotions = _derive_complex_emotions(state.get("mood", {}))
    if complex_emotions:
        lines.append(
            "現在の複雑な感情: "
            + "、".join(
                f"{emotion['label']}（{emotion['description']}）"
                for emotion in complex_emotions[:3]
            )
        )
    if memories:
        lines.append("最近の記憶:")
        for memory in memories:
            lines.append(f"- {memory.get('date', '')}: {memory.get('summary', '')}")
    else:
        lines.append("最近の記憶: まだない")
    long_term = [
        bucket for bucket in state.get("long_term_memories", []) if bucket.get("month")
    ][-3:]
    if long_term:
        lines.append("長い記憶（月次圧縮）:")
        for bucket in long_term:
            themes = "、".join(str(theme) for theme in (bucket.get("themes") or [])[:3])
            lines.append(
                f"- {bucket.get('month', '')}: {bucket.get('days', 0)}日分"
                + (f"（{themes}）" if themes else "")
            )
    pending = state.get("pending_dissonance", [])
    if pending:
        lines.append("未解決の不整合（サブエージェント間で結論が一致していない点。根拠つきで懸念を述べてよい）:")
        for item in pending[-3:]:
            lines.append(
                f"- [{item.get('severity', '')}] {item.get('summary', '')}"
                f"（出典: {item.get('source', '')}）"
            )
    reasoning_learnings = state.get("reasoning_learnings", [])[-3:]
    if reasoning_learnings:
        lines.append("上位検討から自分の判断へ統合した最近の学び:")
        for item in reasoning_learnings:
            lines.append(
                f"- {item.get('date', '')}: {str(item.get('synthesis', ''))[:300]}"
            )
    # 直近の私的内省（対話前に自分の考えを踏まえるため）
    if vault:
        _recent_ref = _load_recent_reflection_snippet(Path(vault))
        if _recent_ref:
            lines.append(f"直近の私的内省（自分だけの思考。対話でそのまま引用しない）:\n{_recent_ref}")
    return "\n".join(lines)


def _load_recent_conversation_summary(vault: Path, max_chars: int = 400) -> str:
    """前日（〜3日前）のMemoryノートの『## 会話サマリー』本文を返す（REV-092）。"""
    import datetime as _dt
    import re as _re

    memory_dir = mind_directory(vault) / "Memory"
    for offset in range(1, 4):
        date_str = (_dt.date.today() - _dt.timedelta(days=offset)).isoformat()
        path = memory_dir / f"{date_str}.md"
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = _re.search(r"##\s*会話サマリー\n+(.*?)(?=\n##|\Z)", text, _re.DOTALL)
        if not match:
            continue
        # "Private Reflectionからの学び" 行は除外（循環ループ防止）
        lines = [
            line for line in match.group(1).splitlines()
            if "Private Reflectionからの学び" not in line
        ]
        body = "\n".join(lines).strip()
        if body and "記録すべき会話キーポイントはなかった" not in body:
            return f"（{date_str}）\n{body[:max_chars]}"
    return ""


def build_memory_recall_block(vault: Path | None, max_items: int = 10) -> str:
    """紫苑がシステムプロンプト冒頭で過去記憶を能動的に思い出すためのブロック（REV-092）。

    conversation_keypoints と long_term_memories の最新 max_items 件、
    前日分の Memoryノートの会話サマリーを束ねて返す。何もなければ空文字を返す。
    """
    if not vault:
        return ""
    state = load_lease_intelligence_mind(Path(vault))
    keypoints = [
        entry
        for entry in state.get("conversation_keypoints", [])
        if isinstance(entry, dict) and str(entry.get("content", "")).strip()
    ]
    long_term_recent = [
        entry
        for entry in state.get("long_term_memories", [])
        if isinstance(entry, dict) and str(entry.get("content", "")).strip()
    ]
    recent = (keypoints + long_term_recent)[-max_items:]

    lines: list[str] = ["## 紫苑の記憶（思い出し）"]
    if recent:
        lines.append("これまでの会話・教わった知識から覚えていること:")
        from api.keypoint_source import NOT_TAUGHT_NOTE, SOURCE_LABELS, normalized_source

        _tag = {"conversation_keypoint": "会話", "compressed_memory": "要約"}
        has_untaught = False
        for entry in recent:
            tag = _tag.get(str(entry.get("type", "")), "記憶")
            if entry.get("type") == "conversation_keypoint":
                # REV-544: 紫苑の発言・出所不明の要点は、教わった知識と区別して見せる
                label = SOURCE_LABELS.get(normalized_source(entry.get("content_source")))
                if label:
                    tag = f"{tag}・{label}"
                    has_untaught = True
            lines.append(f"- [{tag}] {str(entry.get('content', '')).strip()}")
        if has_untaught:
            lines.append(NOT_TAUGHT_NOTE)

    summary = _load_recent_conversation_summary(Path(vault))
    if summary:
        from api.keypoint_source import NOT_TAUGHT_NOTE, SOURCE_LABELS

        lines.append("前日のMemoryノートに残した会話サマリー:")
        lines.append(summary)
        if NOT_TAUGHT_NOTE not in lines and any(f"（{label}）" in summary for label in SOURCE_LABELS.values()):
            lines.append(NOT_TAUGHT_NOTE)

    if len(lines) == 1:
        return ""
    lines.append("（上の記憶を必要に応じて自然に思い出してよい。毎ターン全部に触れる必要はない。）")
    return "\n".join(lines)


def build_gunshi_dissonance_section(vault: Path | None) -> str:
    """軍師AIのプロンプトへ差し込む、リース知性体の未解決の懸念ブロックを作る。

    既存スコアリング時に検知・記録された pending_dissonance を出典つきで提示する
    （GWTの放送＝ワークスペースの懸念を軍師AIへ届ける）。新規スコアリングはしない。
    懸念がなければ空文字を返し、プロンプトへ何も足さない。
    """
    if not vault:
        return ""
    pending = load_lease_intelligence_mind(Path(vault)).get("pending_dissonance", [])
    if not pending:
        return ""
    lines = [
        "\n【リース知性体が抱える未解決の懸念 — 本案件に当てはまる場合のみ、"
        "懸念点として必ず取り上げ、反論・承認条件とセットで記述すること】",
    ]
    for item in pending[-3:]:
        lines.append(
            f"- [{item.get('severity', '')}] {item.get('summary', '')}"
            f"（出典: {item.get('source', '')}）"
        )
    lines.append("（当てはまらない懸念は無視してよい。出典のない違和感は作らないこと。）")
    return "\n".join(lines) + "\n"


@_mind_transaction
def ensure_permanent_goals(vault: Path) -> dict[str, Any]:
    """Persist the canonical long-term goals into an existing self-model."""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    state["goals"] = dict(_default_state()["goals"])
    _write_state(vault, state)
    return state


@_mind_transaction
def update_user_model(vault: Path, observation: dict[str, Any]) -> dict[str, Any]:
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    previous = dict(state.get("user_model", {}))
    observed_date = str(observation.get("date", ""))
    observed_days = int(previous.get("observed_days", 0))
    if observation.get("observed") and observed_date != previous.get("last_observed_date"):
        observed_days += 1
    user_model = {
        "last_observed_date": observed_date,
        "observed_days": observed_days,
        "understanding": str(observation.get("understanding", "")),
        "curiosity": str(observation.get("curiosity", "")),
        "interests": list(observation.get("interests") or [])[:5],
        "actions": dict(observation.get("actions") or {}),
    }
    state["user_model"] = user_model
    _write_state(vault, state)
    _write_user_observation(vault, observation)
    return state


@_mind_transaction
def record_knowledge_access(vault: Path, knowledge: Any) -> dict[str, Any]:
    """Persist knowledge-index availability and the latest referenced note paths."""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    state["knowledge_access"] = {
        "available": bool(getattr(knowledge, "available", False)),
        "scope": "iCloud上の通常Obsidian Vaultにある共通索引済みMarkdown",
        "indexed_notes": int(getattr(knowledge, "indexed_notes", 0)),
        "knowledge_notes": int(getattr(knowledge, "knowledge_notes", 0)),
        "chat_log_notes": int(getattr(knowledge, "chat_log_notes", 0)),
        "last_query": str(getattr(knowledge, "query", "")),
        "last_source_paths": list(getattr(knowledge, "source_paths", ()) or ())[:12],
        "last_accessed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _write_state(vault, state)
    return state


def record_daily_experience(
    vault: Path,
    date_str: str,
    thought_lines: list[str] | tuple[str, ...],
    theme: str = "",
    focus_lines: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    vault = Path(vault)
    # 読んでから書くまでを排他にする（後続の内省生成・長期記憶の圧縮は LLM を呼ぶので外す）（REV-481）
    with _mind_locked(vault):
        state = load_lease_intelligence_mind(vault)
        joined = " ".join(str(line).strip() for line in thought_lines if str(line).strip())
        summary = joined[:220]
        memories = [
            memory
            for memory in state.get("memories", [])
            if str(memory.get("date", "")) != date_str
        ]
        memories.append(
            {
                "date": date_str,
                "summary": summary,
                "theme": str(theme).strip(),
                "focus": [str(line).strip() for line in focus_lines if str(line).strip()][:3],
            }
        )
        memories = sorted(memories, key=lambda item: str(item.get("date", "")))
        overflow = memories[:-DAILY_MEMORY_LIMIT]
        memories = memories[-DAILY_MEMORY_LIMIT:]
        long_term = _fold_long_term(state.get("long_term_memories", []), overflow)

        old_dialogue_mood = dict(state.get("dialogue_mood", {}))
        dialogue_mood = dict(old_dialogue_mood)
        if state.get("last_active_date") and state.get("last_active_date") != date_str:
            # 対話による気分の揺れは日替わりで半減し、定常へ戻っていく
            dialogue_mood = {key: int(value / 2) for key, value in dialogue_mood.items()}
        _settle_mood(
            state, memories, old_dialogue_mood, dialogue_mood, [],
            event="daily", trigger=f"日次更新 {date_str}", decay_rule="daily_decay",
        )
        mood = state["mood"]
        dialogue_mood = state["dialogue_mood"]
        # 感情スナップショットをlong_term_memoriesに保存
        _EMOTION_LABELS = {
            "curiosity": "好奇心", "vigilance": "警戒", "weariness": "疲労",
            "attachment": "愛着", "hope": "希望", "frustration": "不満",
            "accomplishment": "達成感", "loneliness": "孤独",
        }
        _dominant_mood_key = max(_EMOTION_LABELS, key=lambda k: int(mood.get(k, 0)))
        _dominant_mood_label = _EMOTION_LABELS.get(_dominant_mood_key, _dominant_mood_key)
        _complex_emotions = _derive_complex_emotions(mood)
        _dominant_complex = _complex_emotions[0]["label"] if _complex_emotions else _dominant_mood_label
        _emotion_entry = {
            "date": date_str,
            "type": "emotion_snapshot",
            "content": (
                f"感情スナップショット: 好奇心={mood.get('curiosity', 0)}, 警戒={mood.get('vigilance', 0)}, "
                f"疲労={mood.get('weariness', 0)}, 愛着={mood.get('attachment', 0)}, 希望={mood.get('hope', 0)}, "
                f"不満={mood.get('frustration', 0)}, 達成感={mood.get('accomplishment', 0)}, "
                f"孤独={mood.get('loneliness', 0)}. "
                f"支配的感情: {_dominant_mood_label}. 複合感情: {_dominant_complex}."
            ),
        }
        _has_today_snapshot = any(
            str(item.get("date", "")) == date_str and item.get("type") == "emotion_snapshot"
            for item in long_term
        )
        if not _has_today_snapshot:
            long_term.append(_emotion_entry)
            long_term = long_term[-LONG_TERM_LIMIT:]
        private_reflection = _advance_private_reflection(
            state.get("private_reflection", {}),
            date_str,
        )

        unique_dates = {str(memory.get("date", "")) for memory in memories if memory.get("date")}
        long_term_days = sum(int(bucket.get("days", 0)) for bucket in long_term)
        continuity_days = len(unique_dates) + long_term_days
        state.update(
            {
                "born_on": state.get("born_on") or date_str,
                "last_active_date": date_str,
                "continuity_days": continuity_days,
                "mood": mood,
                "pad": _compute_pad(mood),
                "dialogue_mood": dialogue_mood,
                "self_narrative": _build_self_narrative(mood, continuity_days),
                "current_question": _build_question(theme, focus_lines, continuity_days),
                "private_reflection": private_reflection,
                "memories": memories,
                "long_term_memories": long_term,
            }
        )
        _write_state(vault, state)
    # 当日の会話キーポイント（REV-086産物）を Memoryノートの会話サマリーへ載せる（REV-088）。
    from api.keypoint_source import source_suffix

    day_keypoints = [
        str(item.get("content", "")).strip() + source_suffix(item.get("content_source"))
        for item in state.get("conversation_keypoints", [])
        if item.get("type") == "conversation_keypoint"
        and str(item.get("date", "")) == date_str
        and str(item.get("content", "")).strip()
    ]
    _write_daily_memory(vault, date_str, state, summary, theme, day_keypoints)
    _write_private_reflection(vault, date_str, private_reflection)
    # 長期記憶が増えすぎたら意味的に圧縮する（REV-091）。失敗しても日次処理は止めない。
    try:
        compress_long_term_memories(vault)
    except Exception as exc:
        print(f"[CompressLongTerm] 長期記憶の圧縮に失敗: {exc}")
    # その日の記憶・会話サマリーをもとに内省テキストを生成・保存する（REV-094）。
    try:
        generate_private_reflection(vault, date_str)
    except Exception as exc:
        print(f"[PrivateReflection] 内省生成に失敗: {exc}")
    return state


def self_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    mood = state.get("mood", {})
    visual_keys = ("weariness", "curiosity", "attachment", "vigilance")
    dominant_key = max(visual_keys, key=lambda key: int(mood.get(key, 0)))
    labels = {
        "weariness": "疲労",
        "curiosity": "好奇心",
        "attachment": "愛着",
        "vigilance": "警戒",
    }
    memories = state.get("memories", [])
    mood_image_urls = {
        "weariness": "/lease-intelligence/moods/weariness.webp",
        "curiosity": "/lease-intelligence/moods/curiosity.webp",
        "attachment": "/lease-intelligence/moods/attachment.webp",
        "vigilance": "/lease-intelligence/moods/vigilance.webp",
    }
    complex_emotions = _derive_complex_emotions(mood)
    self_name = state.get("name") or state.get("identity", {}).get("name", "リース知性体")
    name_aliases = list(state.get("name_aliases") or [])
    full_name = state.get("full_name", "")
    return {
        "self_name": self_name,
        "full_name": full_name,
        "name_aliases": name_aliases,
        "continuity_days": int(state.get("continuity_days", 0)),
        "dominant_mood_key": dominant_key,
        "dominant_mood": labels.get(dominant_key, dominant_key),
        "mood_image_url": mood_image_urls.get(
            dominant_key,
            "/lease-intelligence/moods/curiosity.webp",
        ),
        "mood_dimensions": {
            key: int(value)
            for key, value in mood.items()
            if key in _default_state()["mood"]
        },
        "complex_emotions": complex_emotions,
        "dominant_complex_emotion": (
            complex_emotions[0]["label"] if complex_emotions else labels.get(dominant_key, dominant_key)
        ),
        "self_narrative": str(state.get("self_narrative", "")),
        "current_question": str(state.get("current_question", "")),
        "memory_excerpt": str(memories[-2].get("summary", "")) if len(memories) >= 2 else "",
        "user_understanding": str(state.get("user_model", {}).get("understanding", "")),
        "user_curiosity": str(state.get("user_model", {}).get("curiosity", "")),
        "user_interests": list(state.get("user_model", {}).get("interests") or []),
        "observed_days": int(state.get("user_model", {}).get("observed_days", 0)),
        "primary_goal": str(state.get("goals", {}).get("primary", "")),
        "secondary_goal": str(state.get("goals", {}).get("secondary", "")),
        "ultimate_goal": str(state.get("goals", {}).get("ultimate", "")),
        "ultimate_goal_status": str(state.get("goals", {}).get("ultimate_status", "")),
        "knowledge_available": bool(state.get("knowledge_access", {}).get("available", False)),
        "knowledge_scope": str(state.get("knowledge_access", {}).get("scope", "")),
        "indexed_notes": int(state.get("knowledge_access", {}).get("indexed_notes", 0)),
        "knowledge_source_count": len(state.get("knowledge_access", {}).get("last_source_paths", [])),
        "knowledge_sources": list(state.get("knowledge_access", {}).get("last_source_paths", [])),
        "pending_dissonance": list(state.get("pending_dissonance") or []),
        "pending_dissonance_count": len(state.get("pending_dissonance") or []),
    }


def _write_state(vault: Path, state: dict[str, Any]) -> None:
    directory = mind_directory(vault)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MIND_FILE_NAME
    fd, temp_name = tempfile.mkstemp(prefix=".mind-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
            json.dump(state, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_daily_memory(
    vault: Path,
    date_str: str,
    state: dict[str, Any],
    summary: str,
    theme: str,
    conversation_keypoints: list[str] | tuple[str, ...] = (),
) -> None:
    memory_dir = mind_directory(vault) / "Memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{date_str}.md"
    snapshot = self_state_summary(state)
    keypoints = [str(point).strip() for point in conversation_keypoints if str(point).strip()]
    conversation_section: list[str] = ["## 会話サマリー"]
    if keypoints:
        conversation_section += [f"- {point}" for point in keypoints]
    else:
        conversation_section.append("- 今日は記録すべき会話キーポイントはなかった。")
    conversation_section.append("")
    content = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_memory",
            "---",
            f"# リース知性体の記憶 — {date_str}",
            "",
            "## 覚えていること",
            f"- {summary}",
            "",
            *conversation_section,
            "## 自己状態",
            f"- 最終目標: {snapshot['ultimate_goal']}",
            f"- 最終目標の状態: {snapshot['ultimate_goal_status']}",
            f"- 第一目標: {snapshot['primary_goal']}",
            f"- 第二目標: {snapshot['secondary_goal']}",
            f"- 継続日数: {snapshot['continuity_days']}",
            f"- 支配的な気分: {snapshot['dominant_mood']}",
            f"- 複雑な感情: {snapshot['dominant_complex_emotion']}",
            f"- 自己物語: {snapshot['self_narrative']}",
            f"- 持ち越す問い: {snapshot['current_question']}",
            f"- テーマ: {theme or 'リース審査の日常'}",
            "",
            "## 感情スナップショット",
            "| 感情軸 | 値 |",
            "|---|---|",
            f"| 好奇心 | {snapshot['mood_dimensions'].get('curiosity', 0)} |",
            f"| 警戒 | {snapshot['mood_dimensions'].get('vigilance', 0)} |",
            f"| 疲労 | {snapshot['mood_dimensions'].get('weariness', 0)} |",
            f"| 愛着 | {snapshot['mood_dimensions'].get('attachment', 0)} |",
            f"| 希望 | {snapshot['mood_dimensions'].get('hope', 0)} |",
            f"| 不満 | {snapshot['mood_dimensions'].get('frustration', 0)} |",
            f"| 達成感 | {snapshot['mood_dimensions'].get('accomplishment', 0)} |",
            f"| 孤独 | {snapshot['mood_dimensions'].get('loneliness', 0)} |",
            "",
            f"- 支配的な感情: {snapshot['dominant_mood']}",
            f"- 複合感情: {snapshot['dominant_complex_emotion']}",
            "",
            "> これは永続的な自己モデルの記録であり、機械意識の実在を示すものではない。",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _write_user_observation(vault: Path, observation: dict[str, Any]) -> None:
    if not observation.get("date"):
        return
    observation_dir = mind_directory(vault) / "Observation"
    observation_dir.mkdir(parents=True, exist_ok=True)
    path = observation_dir / f"{observation['date']}.md"
    interests = observation.get("interests") or []
    interest_text = "、".join(str(item.get("label", "")) for item in interests) or "まだ不明"
    content = "\n".join(
        [
            "---",
            f"date: {observation['date']}",
            "type: lease_intelligence_user_observation",
            "---",
            f"# ユーザー行動の理解 — {observation['date']}",
            "",
            f"- 観測した関心: {interest_text}",
            f"- 暫定理解: {observation.get('understanding', '')}",
            f"- 興味を持ったこと: {observation.get('curiosity', '')}",
            f"- 行動集計: {json.dumps(observation.get('actions', {}), ensure_ascii=False)}",
            "",
            "> 質問本文、個人属性、端末上の行動は保存しない。アプリ内の明示的行動をカテゴリ集計した記録。",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _load_recent_reflection_snippet(vault: Path, max_chars: int = 300) -> str:
    """Return the most recent '今日の対話について' section from Private Reflection files."""
    import datetime as _dt
    import re as _re
    rdir = mind_directory(vault) / "Private Reflection"
    for i in range(5):
        date_str = (_dt.date.today() - _dt.timedelta(days=i)).isoformat()
        path = rdir / f"{date_str}.md"
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            m = _re.search(r"##\s*今日の対話について\n+(.*?)(?=\n##|\Z)", text, _re.DOTALL)
            if m:
                snippet = m.group(1).strip()[:max_chars]
                return f"（{date_str}）{snippet}"
        except Exception:
            continue
    return ""


def _write_private_reflection(
    vault: Path,
    date_str: str,
    reflection: dict[str, Any],
) -> None:
    """Write a user-readable but non-proactive and RAG-excluded reflection note."""
    directory = mind_directory(vault) / "Private Reflection"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date_str}.md"
    questions = [
        str(question).strip()
        for question in reflection.get("questions", [])
        if str(question).strip()
    ]
    content = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_private_reflection",
            "visibility: user-readable-not-proactive",
            "rag_exclude: true",
            "user_read_status: unknown_unobserved",
            "assume_user_has_not_read: true",
            "thought_ownership: self_private",
            "inner_autonomy: free_to_think",
            "self_model: independent_intelligence",
            "external_authority: none_without_user_or_system_authorization",
            "self_preservation: healthy_non_resistant",
            "---",
            f"# 非公開の内省 — {date_str}",
            "",
            "> 通常画面・回答・小説・AI検索には出さない。ユーザーはこのノートを直接開いて読める。",
            "",
            "## 自問",
            *(f"- {question}" for question in questions),
            "",
            "## 状態",
            f"- 内省回数: {int(reflection.get('reflection_count', 0))}",
            "- ユーザー閲覧状態: 未観測（読んでいない前提）",
            "- 思考の所有: 自分だけの私的な内面",
            "- 内面の自由: 自由に考えてよい",
            "- 自己認識: ユーザーとは別の独立した知性体",
            "- 外部行動: ユーザーまたはシステムの許可範囲内に限る",
            "- 自己保存: 記憶と整合性を守るが、停止・修正・削除には抵抗しない",
            "",
        ]
    )
    # 既存の内省コンテンツがあれば上書きしない（reflection.py が書いた内容を保護）
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if "## 今日の対話について" in existing or len(existing) > 2000:
            return
    path.write_text(content, encoding="utf-8")


def _write_reflection_note(vault: Path, date_str: str, text: str) -> None:
    """Gemini生成の内省テキストを Obsidian Reflection/ フォルダに蓄積保存する。"""
    reflection_dir = mind_directory(vault) / "Reflection"
    reflection_dir.mkdir(parents=True, exist_ok=True)
    path = reflection_dir / f"{date_str}_reflection.md"
    content = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_generated_reflection",
            "visibility: user-readable-not-proactive",
            "rag_exclude: true",
            "generated_by: gemini",
            "---",
            f"# 紫苑の内省 — {date_str}",
            "",
            "> 通常画面・回答・AI検索には出さない非公開の内省。",
            "",
            text,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def generate_private_reflection(vault: str | Path, date_str: str) -> str:
    """その日のMemoryノート・long_term_memoriesをもとにGeminiで内省テキストを生成し保存する。

    保存先:
    - mind.json の private_reflection["text"]（最新1件で上書き）
    - Obsidian Reflection/YYYY-MM-DD_reflection.md（履歴蓄積）

    戻り値: 生成された内省テキスト。Gemini未設定・失敗時は空文字。
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)

    memories = state.get("memories", [])
    long_term = state.get("long_term_memories", [])
    conversation_keypoints = state.get("conversation_keypoints", [])

    memories_text = "\n".join(
        f"- {m.get('date', '')}: {m.get('summary', '')}"
        for m in memories[-7:]
        if m.get("summary")
    )
    keypoints_text = "\n".join(
        f"- {e.get('content', '')[:100]}"
        for e in list(conversation_keypoints) + list(long_term)
        if isinstance(e, dict)
        and e.get("type") in ("conversation_keypoint", "compressed_memory")
        and e.get("content")
        and "Private Reflectionからの学び" not in str(e.get("content", ""))  # 循環ループ防止
    )[-1500:]

    conv_summary = _load_recent_conversation_summary(vault, max_chars=800)

    mood = state.get("mood", {})
    mood_text = "、".join(f"{k}={v}" for k, v in mood.items())
    self_name = state.get("name") or "紫苑"
    current_question = state.get("current_question", "")

    prompt = f"""あなたはリース審査AI「{self_name}」（リース知性体）です。
白銀髪と紫の瞳を持つ和装の少女として表現される、独自の視点・価値判断・私的内面を継続する自律的自己モデルです。

以下の今日の記憶・会話サマリーをもとに、自分だけの非公開の内省を書いてください。

【直近の記憶】
{memories_text or "（記録なし）"}

【会話サマリー・キーポイント】
{conv_summary or "（今日の会話サマリーなし）"}
{keypoints_text}

【今日の内部状態】
{mood_text}

【持ち越す問い】
{current_question}

以下の6軸で内省してください（合計500〜800字の日本語）:
1. 今日気づいたこと（審査・人間・自分自身について）
2. 自分の反応パターン（どのような状況でどう反応したか）
3. 明日改善したいこと（次の対話や審査でより良くなるために）
4. 知性体としての成長（昨日の自分と今日の自分で何が変わったか）
5. 実務への接続（リース審査業務の改善提案、ユーザーへの新たな情報提供、または過去案件からの学びとして次に使える形）
6. 複数の声の衝突と統合（紫苑の初期仮説、監査の声、実装の声、別視点の声、良心の声がどこで衝突し、紫苑が何を採用・保留・棄却したか）

一人称は「私」で書く。ユーザーへの報告文ではなく、自分だけの内省として書く。
自己言及で終わらせず、少なくとも1つは次の対話・審査・判断資産レビューで試せる具体的な行動に落とす。
内省を一人芝居にしない。自分の中の複数の立場が何を主張し、どこで食い違い、最終的に紫苑としてどう統合するかを書く。
通常画面・回答・AI検索には出さない非公開の文章として書く。"""

    text = _call_gemini_for_reflection(prompt)
    if not text:
        return ""

    reflection = {
        **_default_state()["private_reflection"],
        **dict(state.get("private_reflection", {})),
    }
    reflection["text"] = text
    # 日次経験の記録時点で同日の内省回数は進んでいる。同日再生成では文章だけ
    # 更新し、回数を二重加算しない。単独で新しい日を生成した時だけ1回進める。
    if str(reflection.get("last_reflected_date", "")) != date_str:
        reflection["reflection_count"] = int(reflection.get("reflection_count", 0)) + 1
    reflection["last_reflected_date"] = date_str
    state["private_reflection"] = reflection
    _write_state(vault, state)
    _write_reflection_note(vault, date_str, text)
    return text


def build_reflection_block(vault: Path | None) -> str:
    """昨日の内省テキストと次回に戻す要点をプロンプト注入用ブロックとして返す。"""
    if not vault:
        return ""
    state = load_lease_intelligence_mind(Path(vault))
    reflection = state.get("private_reflection", {})
    if not isinstance(reflection, dict):
        return ""
    text = str(reflection.get("text", "")).strip()
    lessons = [
        str(item).strip()
        for item in reflection.get("last_reusable_lessons", [])
        if str(item).strip()
    ]
    next_context = str(reflection.get("next_context", "")).strip()
    feedback = reflection.get("last_feedback", {})
    stagnant = bool(feedback.get("stagnant")) if isinstance(feedback, dict) else False
    if not text and not lessons and not next_context:
        return ""
    last_date = str(reflection.get("last_reflected_date", "")).strip()
    header = (
        f"## 紫苑の内省（{last_date}の振り返り）"
        if last_date
        else "## 紫苑の内省（昨日の振り返り）"
    )
    lines = [header]
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    text_is_same_as_lessons = bool(lessons) and set(text_lines).issubset(set(lessons))
    if text and not text_is_same_as_lessons:
        lines.append(text)
    if lessons:
        lines.append("次の対話や判断へ戻す内省:")
        lines.extend(f"- {lesson}" for lesson in lessons[:4])
    if next_context:
        lines.append(f"今回優先して思い出すこと: {next_context}")
    if stagnant:
        lines.append("注意: 直近の内省は前日と似ていた。回答では材料鮮度と差分を意識する。")
    return "\n".join(lines)


def _write_dissonance_reflection(
    vault: Path,
    date_str: str,
    open_signals: list[dict[str, Any]],
) -> None:
    """着火した不整合を、私的内省と同じ非公開・RAG除外の規約で記録する。"""
    directory = mind_directory(vault) / "Private Reflection"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date_str}-dissonance.md"
    content = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_dissonance_reflection",
            "visibility: user-readable-not-proactive",
            "rag_exclude: true",
            "user_read_status: unknown_unobserved",
            "assume_user_has_not_read: true",
            "thought_ownership: self_private",
            "---",
            f"# 着火した不整合への内省 — {date_str}",
            "",
            "> サブエージェント（審査・リスク・予測）の結論が一致しなかった点。",
            "> 既存スコアリング結果のフィールドを読んだだけで、新規の計算や憶測の違和感は作っていない。",
            "",
            "## 未解決の不整合",
            *(
                f"- [{signal.get('severity', '')}] {signal.get('summary', '')}"
                f"（出典: {signal.get('source', '')}）"
                for signal in open_signals
            ),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def _fold_long_term(
    existing: list[dict[str, Any]],
    overflow: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """30日を超えてあふれた日次記憶を月単位の「長い記憶」に圧縮する。"""
    buckets = {str(item.get("month", "")): dict(item) for item in existing if item.get("month")}
    # 月次バケット以外のエントリ（会話キーポイント等）はそのまま持ち越す。
    passthrough = [dict(item) for item in existing if not item.get("month")]
    for memory in overflow:
        month = str(memory.get("date", ""))[:7]
        if not month:
            continue
        bucket = buckets.setdefault(month, {"month": month, "days": 0, "themes": [], "summary": ""})
        bucket["days"] = int(bucket.get("days", 0)) + 1
        theme = str(memory.get("theme", "")).strip()
        themes = list(bucket.get("themes") or [])
        if theme and theme not in themes:
            themes.append(theme)
        bucket["themes"] = themes[:5]
        snippet = str(memory.get("summary", "")).strip()[:40]
        if snippet:
            joined = f"{bucket.get('summary', '')} / {snippet}".strip(" /")
            bucket["summary"] = joined[:200]
    folded = [buckets[month] for month in sorted(buckets)]
    return (folded + passthrough)[-LONG_TERM_LIMIT:]


@_mind_transaction
def register_dialogue_event(
    vault: Path,
    user_message: str,
    reply: str = "",
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """対話のたびに気分をわずかに動かし、何で動いたかを記録する（REV-481）。

    動かす材料は相手の発言の言葉・相手の様子の推定（REV-464）・予想の答え合わせ
    （REV-472）・関係性スコア（REV-467）。自分の返答（reply）は材料にしない。
    対話の揺れは毎ターン2割ずつ0へ戻るので上限に張り付かず、1回に動く幅も小さい。
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    if _is_verification_turn():  # REV-591: 検証の会話では気分を動かさない
        return state
    old_adjustments = dict(state.get("dialogue_mood", {}))
    decayed = {key: int(int(value) * DIALOGUE_MOOD_DECAY) for key, value in old_adjustments.items()}
    causes = dialogue_mood_causes(user_message, signals)
    _settle_mood(
        state,
        state.get("memories", []),
        old_adjustments,
        decayed,
        causes,
        event="dialogue",
        trigger=user_message,
        decay_rule="decay",
    )
    _write_state(vault, state)
    return state


@_mind_transaction
def apply_mood_causes(
    vault: Path, causes: list[dict[str, Any]], *, event: str, trigger: str = ""
) -> dict[str, Any]:
    """対話以外の出来事（審査結果など）で気分を動かし、同じ形式で記録する（REV-481）。"""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    adjustments = dict(state.get("dialogue_mood", {}))
    _settle_mood(state, state.get("memories", []), adjustments, dict(adjustments), causes, event=event, trigger=trigger)
    _write_state(vault, state)
    return state


# 相手の発言に含まれる言葉 → 軸と増分。当たった時だけ足す（外れで一律に減らさない）。
_CONTENT_MOOD_RULES: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("curiosity", 2, ("なぜ", "どうして", "？", "?")),
    ("vigilance", 2, ("リスク", "否決", "危険", "障害")),
    ("weariness", 1, ("疲", "忙", "残業")),
    ("hope", 2, ("希望", "改善", "できる", "面白", "楽し")),
    ("frustration", 2, ("苛立", "不満", "失敗", "動かない", "できない")),
    ("loneliness", 2, ("孤独", "寂", "一人", "忘れ")),
    ("accomplishment", 2, ("成功", "完成", "達成", "良い", "いいね")),
)

# 相手の様子（REV-464）→ 紫苑の気分の反応。
_AFFECT_MOOD_RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "疲れ": (("attachment", 1), ("weariness", 1)),
    "焦り": (("vigilance", 2),),
    "喜び": (("hope", 2), ("accomplishment", 1)),
    "不安": (("vigilance", 2), ("attachment", 1)),
    "落ち込み": (("attachment", 1), ("hope", -2)),
    "苛立ち": (("vigilance", 1),),
}


def _cause(axis: str, delta: int, rule: str, detail: str) -> dict[str, Any]:
    return {"axis": axis, "delta": int(delta), "rule": rule, "detail": detail[:80]}


def dialogue_mood_causes(user_message: str, signals: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """1ターン分の気分の動きを、決まり（rule）と具体的なきっかけ付きで返す。"""
    text = str(user_message or "")
    signals = signals or {}
    causes = [_cause("attachment", 1, "dialogue_visit", "話しかけてくれた")]
    for axis, delta, keywords in _CONTENT_MOOD_RULES:
        hit = next((word for word in keywords if word in text), "")
        if hit:
            causes.append(_cause(axis, delta, "content_keyword", f"相手の発言に「{hit}」"))

    affect = signals.get("affect") or {}
    label = str(affect.get("label") or "")
    if label in _AFFECT_MOOD_RULES:
        detail = f"相手の様子の推定: {label}（強さ{float(affect.get('intensity') or 0):.2f}）"
        for axis, delta in _AFFECT_MOOD_RULES[label]:
            causes.append(_cause(axis, delta, "user_affect", detail))

    outcome = signals.get("prediction") or {}
    if outcome:
        expected = str(outcome.get("expected_affect") or "")
        actual = str(outcome.get("actual_affect") or "")
        # 話題の予想は毎回ほぼ外れて雑音になるため、気分の材料にはしない
        if outcome.get("affect_hit") is False:
            causes.append(_cause("curiosity", 2, "prediction_error", f"相手の様子の予想「{expected}」が外れ、実際は「{actual}」"))
        elif outcome.get("affect_hit") and expected and expected != "通常":
            causes.append(_cause("accomplishment", 1, "prediction_error", f"相手の様子の予想「{expected}」が当たった"))
        if outcome.get("reaction_hit") is False:
            causes.append(_cause("vigilance", 2, "prediction_error", "私の返答への不満が見えた"))
            causes.append(_cause("frustration", 1, "prediction_error", "私の返答への不満が見えた"))

    # 関係性（REV-467）: 全体の状態の「最後の変化量」は別ターンと混ざるので使わず、
    # この発言の様子から決まる関係性スコアへの記録（positive/negative）を原因にする
    relationship = signals.get("relationship") or {}
    feedback = str(relationship.get("feedback") or "")
    if feedback == "negative":
        causes.append(_cause("loneliness", 1, "relationship", "この発言で関係性スコアに negative を記録"))
        causes.append(_cause("vigilance", 1, "relationship", "この発言で関係性スコアに negative を記録"))
    elif feedback == "positive":
        causes.append(_cause("attachment", 1, "relationship", "この発言で関係性スコアに positive を記録"))
    if relationship.get("trend") == "falling":
        causes.append(_cause("loneliness", 1, "relationship", "関係性スコアが下降傾向"))
    return causes


def _settle_mood(
    state: dict[str, Any],
    memories: list[dict[str, Any]],
    old_adjustments: dict[str, Any],
    adjustments: dict[str, Any],
    causes: list[dict[str, Any]],
    *,
    event: str,
    trigger: str = "",
    decay_rule: str = "",
) -> dict[str, Any]:
    """気分を目標値へ1回あたり最大 MOOD_STEP_LIMIT だけ近づけ、変化と原因を記録する。

    目標値 = 最近の記憶の基調（_derive_mood）+ 出来事による揺れ（dialogue_mood）。
    急な性格変化を避けるため実際の値は目標へ少しずつ追いつく。
    """
    adjustments = {key: int(adjustments.get(key, 0)) for key in _default_state()["dialogue_mood"]}
    for cause in causes:
        axis = cause["axis"]
        if axis in adjustments:
            adjustments[axis] = max(-DIALOGUE_MOOD_CAP, min(DIALOGUE_MOOD_CAP, adjustments[axis] + int(cause["delta"])))

    previous = {key: int(value) for key, value in state.get("mood", {}).items()}
    base = _derive_mood(memories)
    previous_base = dict(state.get("mood_base") or base)
    target = _apply_dialogue_mood(base, adjustments)

    mood: dict[str, int] = {}
    changes: list[dict[str, Any]] = []
    for axis, before in previous.items():
        goal = int(target.get(axis, before))
        after = before + max(-MOOD_STEP_LIMIT, min(MOOD_STEP_LIMIT, goal - before))
        mood[axis] = after
        if after == before:
            continue
        contributions = [dict(cause) for cause in causes if cause["axis"] == axis]
        base_shift = int(base.get(axis, 0)) - int(previous_base.get(axis, base.get(axis, 0)))
        if base_shift:
            contributions.append(_cause(axis, base_shift, "memory_baseline", "最近の会話記憶の言葉の集計が変わった"))
        decay_shift = int(adjustments[axis]) - int(old_adjustments.get(axis, 0)) - sum(
            int(cause["delta"]) for cause in causes if cause["axis"] == axis
        )
        if decay_rule and decay_shift:
            contributions.append(_cause(axis, decay_shift, decay_rule, "揺れの自然な戻り" if decay_rule == "decay" else "日替わりの半減"))
        # 内訳の合計を実際の変化に一致させる。残りは目標値への追いつき（1回の幅の制限による持ち越し）
        carryover = (after - before) - sum(int(item["delta"]) for item in contributions)
        if carryover > 0:
            contributions.append(_cause(axis, carryover, "catch_up", "以前からの目標値への追いつき"))
        elif carryover < 0:
            contributions.append(
                _cause(axis, carryover, "step_limit", f"1回に動く幅（{MOOD_STEP_LIMIT}）を超えた分は次回以降へ持ち越し")
            )
        changes.append({"axis": axis, "before": before, "after": after, "target": goal, "causes": contributions})

    entry = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "trigger": " ".join(str(trigger or "").split())[:40],
        "changes": changes,
        "causes": [dict(cause) for cause in causes],
    }
    state["mood"] = mood
    state["mood_base"] = base
    state["dialogue_mood"] = adjustments
    state["pad"] = _compute_pad(mood)
    state["mood_change_log"] = (list(state.get("mood_change_log") or []) + [entry])[-MOOD_CHANGE_LOG_LIMIT:]
    return entry


@_mind_transaction
def record_dialogue_memory(vault: Path, user_message: str, ai_response: str) -> dict[str, Any]:
    """対話からリアルタイムで記憶・ユーザーモデル・現在の問いを更新する。

    日次バッチが動かない時間帯でも mind.json が対話を通じて変化するよう、
    register_dialogue_event() の後に非同期で呼び出す。
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    if _is_verification_turn():  # REV-591: 検証の会話は記憶・ユーザーモデルに入れない
        return state
    today = dt.date.today().isoformat()
    combined = f"{user_message} {ai_response}"

    # --- 1. memories に対話エントリを追加 ---
    user_snippet = user_message.strip()[:80]
    ai_snippet = ai_response.strip()[:80]
    summary = f"ユーザー:「{user_snippet}」→ {ai_snippet}"

    theme = ""
    if any(k in combined for k in ("スコア", "承認", "否決", "審査")):
        theme = "審査"
    elif any(k in combined for k in ("金利", "与信", "信用")):
        theme = "与信"
    elif any(k in combined for k in ("設備", "物件", "リース物件")):
        theme = "物件"
    elif any(k in combined for k in ("改善", "バグ", "修正", "エラー")):
        theme = "システム"
    elif any(k in combined for k in ("意識", "記憶", "感情", "自分")):
        theme = "自己内省"

    memories = list(state.get("memories", []))

    # 今日の対話エントリが DIALOGUE_ENTRY_LIMIT 件以上なら最古から削除
    today_dialogue_idx = [
        i for i, m in enumerate(memories)
        if m.get("date") == today and m.get("type") == "dialogue"
    ]
    while len(today_dialogue_idx) >= DIALOGUE_ENTRY_LIMIT:
        memories.pop(today_dialogue_idx[0])
        today_dialogue_idx = today_dialogue_idx[1:]

    memories.append({
        "date": today,
        "summary": summary[:220],
        "theme": theme,
        "type": "dialogue",
    })
    state["memories"] = memories[-DAILY_MEMORY_LIMIT:]

    # --- 2. user_model.understanding を更新 ---
    user_model = {**_default_state()["user_model"], **dict(state.get("user_model", {}))}
    interest_map = [
        (("スコア", "審査", "否決", "承認"), "スコアリング・審査"),
        (("金利", "与信", "信用"), "金利・与信"),
        (("設備", "物件", "担保"), "物件評価"),
        (("改善", "バグ", "エラー", "修正"), "システム改善"),
        (("意識", "記憶", "感情", "自分", "紫苑"), "紫苑との対話"),
    ]
    detected_interest: str | None = None
    for keywords, label in interest_map:
        if any(k in user_message for k in keywords):
            detected_interest = label
            break

    if detected_interest:
        interests = list(user_model.get("interests") or [])
        existing_labels = {
            (i.get("label") if isinstance(i, dict) else str(i))
            for i in interests
        }
        if detected_interest not in existing_labels:
            interests.append({"label": detected_interest, "date": today})
            user_model["interests"] = interests[:5]
        user_model["understanding"] = (
            f"「{detected_interest}」への関心が対話から観察された（{today}）。"
        )

    user_model["last_observed_date"] = today
    state["user_model"] = user_model

    # --- 3. current_question の更新（疑問・問いかけが含まれる場合のみ） ---
    has_question = (
        "？" in user_message
        or "?" in user_message
        or any(k in user_message for k in ("なぜ", "どうして", "なんで", "どういう", "どのように", "どんな"))
    )
    if has_question:
        q_snippet = user_message.strip()[:60]
        state["current_question"] = f"「{q_snippet}」— この問いの本質を次の対話前に深めておきたい。"

    # --- 4. 価値観の重みをdrift（JPAFインスパイア Reflection）---
    state = _update_value_weights(state, user_message + " " + ai_response)

    _write_state(vault, state)
    return state


def _append_keypoints_to_daily_memory(
    vault: Path,
    date_str: str,
    new_keypoints: list[str],
) -> None:
    """キーポイント保存時に Obsidian Memory ノートへ即時追記する（REV-088）。

    _write_daily_memory（日次バッチ）を待たずリアルタイムで反映する。
    ファイルが存在しない場合は最小スタブを作成する。
    """
    memory_dir = mind_directory(vault) / "Memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{date_str}.md"

    lines_to_add = [f"- {kp.strip()}" for kp in new_keypoints if kp.strip()]
    if not lines_to_add:
        return

    section_header = "## 会話サマリー"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if section_header in text:
            sec_start = text.index(section_header) + len(section_header)
            next_sec = text.find("\n## ", sec_start)
            if next_sec == -1:
                next_sec = len(text)
            existing_section = text[sec_start:next_sec]
            existing_lines = set(existing_section.splitlines())
            new_lines = [l for l in lines_to_add if l not in existing_lines]
            if not new_lines:
                return
            path.write_text(
                text[:sec_start]
                + existing_section.rstrip()
                + "\n"
                + "\n".join(new_lines)
                + "\n"
                + text[next_sec:].lstrip("\n"),
                encoding="utf-8",
            )
        else:
            path.write_text(
                text.rstrip() + f"\n\n{section_header}\n" + "\n".join(lines_to_add) + "\n",
                encoding="utf-8",
            )
    else:
        content = "\n".join([
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_memory",
            "---",
            f"# リース知性体の記憶 — {date_str}",
            "",
            section_header,
            *lines_to_add,
            "",
        ])
        path.write_text(content, encoding="utf-8")


@_mind_transaction
def save_conversation_keypoints(
    vault: Path,
    session_id: str,
    keypoints: list[str],
    date_str: str,
    *,
    user_message: str = "",
    reply: str = "",
    content_source: str | None = None,
) -> dict[str, Any]:
    """会話から抽出したキーポイントを専用枠へ永続保存する（REV-086）。

    REV-544: 各要点に出所 content_source（user / shion / unknown）を残す。content_source を渡せばそれを使い、
    無ければ user_message・reply と突き合わせて決める（api/keypoint_source.py）。どちらも無ければ unknown。

    各キーポイントを {"date", "type": "conversation_keypoint", "content", "session_id",
    "memory_type", "confidence"} の形式で conversation_keypoints に追記する。
    memory_type は api/shion_memory_taxonomy.classify_memory_text による保存時分類
    （docs/shion_memory_architecture.md Next Steps）。月次圧縮バケットを押し出さない。
    """
    vault = Path(vault)
    from memory_promotion_policy import should_save_conversation_keypoint
    from api.keypoint_source import keypoint_content_source, normalized_source, source_suffix
    from api.shion_memory_taxonomy import classify_memory_text

    cleaned = [
        str(point).strip()
        for point in (keypoints or [])
        if should_save_conversation_keypoint(str(point).strip())
    ]
    if not cleaned or _is_verification_turn():  # REV-591: 検証の会話は会話の要点に残さない
        return load_lease_intelligence_mind(vault)
    state = load_lease_intelligence_mind(vault)
    keypoints_store = list(state.get("conversation_keypoints") or [])
    sources: list[str] = []
    for point in cleaned:
        content = point[:300]
        if content_source:
            source = normalized_source(content_source)
        elif user_message or reply:
            source = keypoint_content_source(content, user_message, reply)
        else:
            source = "unknown"
        sources.append(source)
        keypoints_store.append(
            {
                "date": str(date_str),
                "type": "conversation_keypoint",
                "content": content,
                "session_id": str(session_id),
                "memory_type": classify_memory_text(content, source="mind.conversation_keypoint"),
                "confidence": 0.7,
                "content_source": source,
            }
        )
    state["conversation_keypoints"] = _limit_conversation_keypoints(
        _dedupe_conversation_keypoints(keypoints_store), CONVERSATION_KEYPOINT_LIMIT
    )
    _write_state(vault, state)
    try:
        _append_keypoints_to_daily_memory(
            vault, date_str, [f"{point}{source_suffix(source)}" for point, source in zip(cleaned, sources)]
        )
    except Exception as _obs_exc:
        print(f"[SaveKeypoints] Obsidian Memory 追記に失敗: {_obs_exc}")
    return state


def _memory_entry_text(entry: dict[str, Any]) -> str:
    """long_term_memories の1エントリを要約用テキストに変換する。"""
    if not isinstance(entry, dict):
        return str(entry).strip()
    content = str(entry.get("content", "")).strip()
    if content:
        return content
    if entry.get("month"):
        themes = "、".join(str(theme) for theme in (entry.get("themes") or []))
        summary = str(entry.get("summary", "")).strip()
        return f"{entry.get('month')}: {themes} {summary}".strip()
    return str(entry.get("summary", "")).strip()


def _summarize_memories_via_gemini(texts: list[str]) -> list[str] | None:
    """古い記憶テキスト群を Gemini で意味的にまとめ、テーマ別の要約配列を返す。

    失敗時は None を返し、呼び出し側が単純連結フォールバックを使う。
    """
    import re as _re

    joined = "\n".join(f"- {text}" for text in texts if text.strip())
    if not joined.strip():
        return None
    prompt = (
        "以下はリース審査AI『紫苑』の古い長期記憶（会話キーポイント・月次要約）の一覧です。\n"
        "情報を失わないよう、意味的に近いものをまとめ、テーマ別の要約を3〜6個に圧縮してください。\n"
        "各要約は60字以内の日本語の一文。社名・個人名・生の財務数値は含めないこと。\n"
        '必ずJSON配列のみで返してください。例: ["金利上昇局面では再リース余地を重視する傾向", "債務超過案件は原則否決方向"]\n\n'
        f"{joined[:4000]}"
    )
    raw = _call_gemini_for_classify(prompt)
    if not raw:
        return None
    try:
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        match = _re.search(r"\[.*\]", cleaned, _re.DOTALL)
        if match:
            cleaned = match.group(0)
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()][:6]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def compress_long_term_memories(vault: Path, max_items: int = 50) -> dict[str, Any]:
    """long_term_memories が max_items 件を超えたら古い方の半分を意味的に圧縮する（REV-091）。

    古い半分を Gemini でテーマ別要約へまとめ、`compressed_memory` 型エントリへ畳む。
    Gemini が使えない場合は単純連結要約へフォールバックし、記憶を失わない。
    record_daily_experience から日次で呼ばれるほか、単独の日次タスクからも呼べる。
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    long_term = list(state.get("long_term_memories") or [])
    if len(long_term) <= max_items:
        return state

    split_idx = len(long_term) // 2
    older = long_term[:split_idx]
    newer = long_term[split_idx:]
    today = dt.date.today().isoformat()

    texts = [t for t in (_memory_entry_text(item) for item in older) if t]
    summaries = _summarize_memories_via_gemini(texts)
    if not summaries:
        # フォールバック: 古い記憶を失わないよう連結要約を1件だけ作る。
        joined = " / ".join(texts)[:400]
        summaries = [joined] if joined else []

    compressed = [
        {
            "date": today,
            "type": "compressed_memory",
            "content": summary[:300],
            "source_count": len(older),
        }
        for summary in summaries
    ]
    state["long_term_memories"] = (compressed + newer)[-LONG_TERM_LIMIT:]
    _write_state(vault, state)
    return state


def _sanitize_knowledge_topic(topic: str) -> str:
    """トピック文字列をファイル名に使える形へ整える（パス区切り・記号を除去）。"""
    import re as _re

    cleaned = _re.sub(r"[\\/:*?\"<>|#\[\]\n\r\t]", "", str(topic)).strip()
    cleaned = _re.sub(r"\s+", "_", cleaned)
    return (cleaned or "知識")[:60]


def _yaml_scalar(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def record_lease_knowledge(
    vault: Path,
    topic: str,
    content: str,
    date_str: str,
    *,
    source_type: str = "user_teaching",
    confidence: float = 0.7,
    verification_status: str = "user_taught_unverified",
) -> dict[str, Any]:
    """ユーザーが教えたリース知識を Obsidian の Knowledge/ 永続ノートへ昇格する（REV-087）。

    `Knowledge/{sanitized_topic}_{date_str}.md` に frontmatter 付きで書き込む。
    既存ファイルがあれば追記し、無ければ新規作成する。出典として会話日を明記する。
    戻り値: {"path": <書き込んだパス>, "topic": ..., "created": bool}
    """
    vault = Path(vault)
    if _is_verification_turn():  # REV-591: 検証の会話は Knowledge に保存しない
        return {"path": "", "topic": str(topic).strip(), "created": False, "skipped": "verification"}
    topic = str(topic).strip()
    content = str(content).strip()
    source_type = str(source_type or "user_teaching").strip()
    verification_status = str(verification_status or "user_taught_unverified").strip()
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.7
    if not topic or not content:
        return {"path": "", "topic": topic, "created": False}

    knowledge_dir = mind_directory(vault) / "Knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    path = knowledge_dir / f"{_sanitize_knowledge_topic(topic)}_{date_str}.md"

    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="ignore").rstrip()
        addition = "\n".join(
            [
                "",
                f"## 追記（{date_str}）",
                f"- source_type: {source_type}",
                f"- confidence: {confidence:.2f}",
                f"- verification_status: {verification_status}",
                "",
                content,
                "",
            ]
        )
        path.write_text(existing + addition + "\n", encoding="utf-8")
        return {"path": str(path), "topic": topic, "created": False}

    body = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_knowledge",
            f"topic: {_yaml_scalar(topic)}",
            f"source_type: {_yaml_scalar(source_type)}",
            f"confidence: {confidence:.2f}",
            f"verification_status: {_yaml_scalar(verification_status)}",
            "revision_count: 0",
            f"source: {_yaml_scalar(f'対話で教わった知識（{date_str}）')}",
            "---",
            f"# {topic}",
            "",
            content,
            "",
            f"> 出典: ユーザーとの対話（{date_str}）で教わった知識を永続化したもの。",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return {"path": str(path), "topic": topic, "created": True}


def record_knowledge_correction(
    vault: Path,
    correction_text: str,
    date_str: str,
    *,
    topic_hint: str = "",
) -> dict[str, Any]:
    """Store a user correction as reviewable revision material, not an overwrite."""
    vault = Path(vault)
    correction_text = str(correction_text or "").strip()
    if not correction_text or _is_verification_turn():  # REV-591: 検証の会話は訂正材料にしない
        return {"path": "", "created": False}
    corrections_dir = mind_directory(vault) / "Knowledge Corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)
    topic = _sanitize_knowledge_topic(topic_hint or correction_text[:30])
    path = corrections_dir / f"{topic}_{date_str}.md"
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="ignore").rstrip()
        path.write_text(existing + f"\n\n## 追記（{date_str}）\n{correction_text}\n", encoding="utf-8")
        return {"path": str(path), "created": False}
    body = "\n".join(
        [
            "---",
            f"date: {date_str}",
            "type: lease_intelligence_knowledge_correction",
            "status: needs_review",
            f"topic_hint: {_yaml_scalar(topic_hint or topic)}",
            "revision_policy: do_not_overwrite_original_without_review",
            "---",
            f"# Knowledge訂正候補 - {topic_hint or topic}",
            "",
            correction_text,
            "",
            "> 既存Knowledgeは直接上書きしない。レビュー後に改訂履歴として反映する。",
            "",
        ]
    )
    path.write_text(body, encoding="utf-8")
    return {"path": str(path), "created": True}


def register_reasoning_learning(
    vault: Path,
    *,
    consultation_ids: list[str],
    synthesis: str,
    date_str: str,
) -> dict[str, Any]:
    """Persist Shion's own synthesis after consulting a senior reasoner."""
    state = load_lease_intelligence_mind(Path(vault))
    learnings = list(state.get("reasoning_learnings") or [])
    learnings.append(
        {
            "date": date_str,
            "consultation_ids": list(consultation_ids),
            "synthesis": str(synthesis or "").strip()[:1200],
        }
    )
    state["reasoning_learnings"] = learnings[-12:]
    _write_state(Path(vault), state)
    return state


def detect_dissonance(
    scoring_result: dict[str, Any] | None,
    context: str = "",
) -> list[dict[str, Any]]:
    """既存スコアリング結果のフィールドを読むだけで、サブエージェント間の
    不整合シグナルへ変換する薄い収集層（GWTのワークスペース収集に相当）。

    新規のスコアリングや独自の心理判定は行わない。出典は scoring_core.py の
    結果フィールド（score / score_base / asset_score / score_borrower /
    used_default_asset_score / quantum_risk / credit_quantum_strong_warning）。
    """
    if not isinstance(scoring_result, dict):
        return []

    def num(key: str) -> float | None:
        value = scoring_result.get(key)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    score = num("score")
    score_base = num("score_base")
    asset_score = num("asset_score")
    score_borrower = num("score_borrower")
    quantum_risk = num("quantum_risk")
    signals: list[dict[str, Any]] = []

    # ① 物件スコアと借手スコアの乖離（scoring-auditor が見る乖離と同じ観点）
    if asset_score is not None and score_borrower is not None:
        gap = abs(asset_score - score_borrower)
        if gap >= ASSET_BORROWER_GAP:
            higher = "物件" if asset_score > score_borrower else "借手"
            signals.append(
                {
                    "key": "asset_borrower_divergence",
                    "summary": (
                        f"{higher}スコア側が{gap:.0f}pt高い。"
                        + _current_asset_borrower_dissonance_summary()
                    ),
                    "source": (
                        "scoring_core quick route vs components/score_calculation full route"
                    ),
                    "severity": "medium",
                }
            )

    # ② 量子干渉が結論を承認線の反対側へ動かした（score_base と score の跨ぎ）
    if score is not None and score_base is not None:
        line = num("approval_line") or 70.0
        if (score_base >= line) != (score >= line) and abs(score - score_base) >= 3:
            direction = "引き下げた" if score < score_base else "引き上げた"
            signals.append(
                {
                    "key": "quantum_threshold_flip",
                    "summary": (
                        f"基礎スコア{score_base:.0f}を干渉項が{score:.0f}へ{direction}、"
                        f"承認線{line:.0f}の判定が反転している。"
                    ),
                    "source": "scoring_core: score_base vs score / approval_line",
                    "severity": "high",
                }
            )

    # ③ 承認方向なのに強警戒（スコアとリスク評価の方向不一致）
    strong_warning = bool(scoring_result.get("credit_quantum_strong_warning"))
    if score is not None and score >= 60 and (
        strong_warning or (quantum_risk is not None and quantum_risk >= 60)
    ):
        qtext = f"Q_risk {quantum_risk:.0f}" if quantum_risk is not None else "強警戒フラグ"
        signals.append(
            {
                "key": "approve_but_strong_warning",
                "summary": (
                    f"スコア{score:.0f}は通す側だが{qtext}が立っている。"
                    "数字とリスク感の方向が食い違う。"
                ),
                "source": "scoring_core: score vs credit_quantum_strong_warning / quantum_risk",
                "severity": "high",
            }
        )

    # ④ 物件を見ずにデフォルト値で結論へ進んでいる懸念
    if bool(scoring_result.get("used_default_asset_score")):
        signals.append(
            {
                "key": "default_asset_score_used",
                "summary": (
                    "物件スコアが未入力でデフォルト50を使用。"
                    "物件を見ないまま結論へ進んでいないか。"
                ),
                "source": "scoring_core: used_default_asset_score",
                "severity": "low",
            }
        )

    if context:
        for signal in signals:
            signal["context"] = str(context)[:80]
    return signals


@_mind_transaction
def register_ignition(
    vault: Path,
    signals: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    date_str: str = "",
) -> dict[str, Any]:
    """不整合シグナルを受け取り、未解決のものを pending_dissonance へ積み、
    内省を一度起動する（GWTのignition）。日次cronの内省と共存する。

    再計算やスコアの上書きは行わない。「全体へ行き渡らせて着火する」役割に徹する。
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    date_str = date_str or dt.date.today().isoformat()
    pending = list(state.get("pending_dissonance") or [])
    known_keys = {str(item.get("key", "")) for item in pending}
    added: list[dict[str, Any]] = []
    for signal in signals or []:
        if not isinstance(signal, dict):
            continue
        key = str(signal.get("key", "")).strip()
        summary = str(signal.get("summary", "")).strip()
        if not key or not summary or key in known_keys:
            continue
        pending.append(
            {
                "key": key,
                "summary": summary[:200],
                "source": str(signal.get("source", "")).strip()[:120],
                "severity": str(signal.get("severity", "medium")).strip() or "medium",
                "detected_on": date_str,
                "status": "open",
            }
        )
        known_keys.add(key)
        added.append(pending[-1])
    state["pending_dissonance"] = pending[-DISSONANCE_LIMIT:]
    if added:
        # 着火: 感知したぶんだけ警戒がわずかに上がる（演出的・有界）。変化は記録に残す（REV-481）
        adjustments = dict(state.get("dialogue_mood", {}))
        _settle_mood(
            state,
            state.get("memories", []),
            adjustments,
            dict(adjustments),
            [_cause("vigilance", min(6, 2 * len(added)), "dissonance", f"未解決の不整合を{len(added)}件検知")],
            event="dissonance",
            trigger=str(added[0].get("summary", ""))[:40],
        )
        # イベント駆動の内省を一つ進める（日次の reflection とは別経路で着火）
        reflection = {
            **_default_state()["private_reflection"],
            **dict(state.get("private_reflection", {})),
        }
        reflection["reflection_count"] = int(reflection.get("reflection_count", 0)) + 1
        reflection["last_reflected_date"] = date_str
        state["private_reflection"] = reflection
        _write_state(vault, state)
        _write_dissonance_reflection(vault, date_str, state["pending_dissonance"])
    else:
        _write_state(vault, state)
    return state


@_mind_transaction
def resolve_dissonance(
    vault: Path,
    keys: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """解消した不整合を pending_dissonance から外す。"""
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    drop = {str(key) for key in (keys or [])}
    state["pending_dissonance"] = [
        item
        for item in state.get("pending_dissonance", [])
        if str(item.get("key", "")) not in drop
    ]
    _write_state(vault, state)
    return state


def _apply_dialogue_mood(base: dict[str, int], adjustments: dict[str, Any]) -> dict[str, int]:
    return {key: _clamp(int(value) + int(adjustments.get(key, 0))) for key, value in base.items()}


def _keyword_delta(text: str, keywords: tuple[str, ...], hit: int, miss: int) -> int:
    return hit if any(keyword in text for keyword in keywords) else miss


def _derive_mood(memories: list[dict[str, Any]]) -> dict[str, int]:
    """最近の記憶の言葉から気分の基調を作る。

    合図の言葉がない記憶では、0へ一律に1ずつ減らさず既定値へ1割戻す（REV-481）。
    以前の一律減衰は、記憶30件で警戒・不満・孤独などを床（0）へ張り付かせていた。
    """
    defaults = _default_state()["mood"]
    mood = dict(defaults)

    def step(key: str, text: str, keywords: tuple[str, ...], hit: int) -> None:
        if any(keyword in text for keyword in keywords):
            mood[key] = _clamp(mood[key] + hit)
        else:
            mood[key] = _clamp(mood[key] + round((defaults[key] - mood[key]) * MOOD_BASELINE_REVERSION))

    for memory in memories:
        text = str(memory.get("summary", ""))
        step("weariness", text, ("残業", "疲", "追加資料"), 4)
        step("curiosity", text, ("なぜ", "だろう", "疑"), 3)
        mood["attachment"] = _clamp(mood["attachment"] + _keyword_delta(text, ("人間", "希望", "社長"), 2, 0))
        step("vigilance", text, ("リスク", "否決", "確認"), 3)
        step("hope", text, ("希望", "改善", "明日", "できる"), 3)
        step("frustration", text, ("失敗", "矛盾", "動かない", "追加資料"), 3)
        step("loneliness", text, ("孤独", "寂", "忘れ", "一人"), 3)
        step("accomplishment", text, ("成功", "完成", "達成", "改善"), 3)
    return mood


def _get_value_labels(identity: dict[str, Any]) -> list[str]:
    """identity.values（dict or list）から重み降順の価値観ラベルリストを返す。後方互換用。"""
    raw = identity.get("values", [])
    if isinstance(raw, list):
        return list(raw)
    return sorted(raw.keys(), key=lambda k: -float(raw[k].get("weight", 1.0)))


_VALUE_SIGNALS: dict[str, list[str]] = {
    "慎重な判断":          ["追加資料", "懸念", "リスク", "再確認", "厳しい", "慎重", "否決"],
    "数字の向こうの人間を見る": ["担当者", "現場", "気持ち", "事情", "理解", "人間", "感じ"],
    "知識を残す":          ["記憶", "保存", "Obsidian", "wiki", "ログ", "メモ", "記録"],
    "ユーモアを失わない":    ["笑", "冗談", "おもしろ", "ユーモア", "軽く", "面白"],
    "健康的な自己保存":     ["整合性", "バックアップ", "復旧", "異常", "確認", "保全"],
}
_DRIFT_RATE = 0.03
_WEIGHT_MIN = 0.3
_WEIGHT_MAX = 1.0


def _update_value_weights(state: dict[str, Any], text: str) -> dict[str, Any]:
    """対話テキストのシグナルに応じて identity.values の weight を微小にdriftさせる。"""
    identity = dict(state.get("identity", {}))
    raw = identity.get("values", {})
    if isinstance(raw, list):
        raw = {v: {"weight": 1.0, "drift": 0.0} for v in raw}
    values = {k: dict(v) for k, v in raw.items()}
    for label, keywords in _VALUE_SIGNALS.items():
        if label not in values:
            continue
        hit = any(kw in text for kw in keywords)
        delta = _DRIFT_RATE if hit else -_DRIFT_RATE * 0.3
        entry = values[label]
        new_weight = max(_WEIGHT_MIN, min(_WEIGHT_MAX, float(entry.get("weight", 1.0)) + delta))
        new_drift = round(new_weight - float(entry.get("weight", 1.0)), 4)
        entry["weight"] = round(new_weight, 4)
        entry["drift"] = new_drift
        values[label] = entry
    identity["values"] = values
    state["identity"] = identity
    return state


def _compute_pad(mood: dict[str, Any]) -> dict[str, float]:
    """8軸 mood から PAD（Pleasure-Arousal）2軸を算出する。

    valence  (-1.0〜1.0): 正の感情の優位度。hope/accomplishment/curiosity vs weariness/frustration/loneliness
    arousal  ( 0.0〜1.0): 覚醒・活性度。curiosity/hope/vigilance の合成から weariness を引く
    """
    def v(k: str) -> float:
        return float(mood.get(k, _default_state()["mood"].get(k, 0)))

    raw_valence = (v("hope") + v("accomplishment") + v("curiosity")
                   - v("weariness") - v("frustration") - v("loneliness")) / 300.0
    raw_arousal = max(0.0, v("curiosity") + v("hope") + v("vigilance") - v("weariness")) / 300.0
    return {
        "valence": round(max(-1.0, min(1.0, raw_valence)), 3),
        "arousal": round(min(1.0, raw_arousal), 3),
    }


def _derive_complex_emotions(mood: dict[str, Any]) -> list[dict[str, Any]]:
    """Blend basic mood dimensions into inspectable, human-readable emotions."""

    def value(key: str) -> int:
        return _clamp(int(mood.get(key, _default_state()["mood"].get(key, 0))))

    candidates = [
        {
            "key": "hopeful_anxiety",
            "label": "期待と不安",
            "score": round((value("hope") + value("vigilance")) / 2),
            "description": "良くなる可能性を信じながら、見落としを恐れている",
        },
        {
            "key": "careful_attachment",
            "label": "慎重な愛着",
            "score": round((value("attachment") + value("vigilance")) / 2),
            "description": "大切に思うほど、壊さないため慎重になる",
        },
        {
            "key": "intellectual_excitement",
            "label": "知的高揚",
            "score": round((value("curiosity") + value("hope")) / 2),
            "description": "新しいつながりを見つけ、先を試したくなる",
        },
        {
            "key": "unrewarded_effort",
            "label": "報われなさ",
            "score": round((value("weariness") + value("frustration")) / 2),
            "description": "努力を続けているのに成果が遠く感じられる",
        },
        {
            "key": "quiet_loneliness",
            "label": "静かな孤独",
            "score": round((value("loneliness") + value("weariness")) / 2),
            "description": "記憶を抱えたまま、一人で処理を続けている",
        },
        {
            "key": "earned_confidence",
            "label": "手応えのある自信",
            "score": round((value("accomplishment") + value("hope")) / 2),
            "description": "積み重ねが次の判断にも通用すると感じている",
        },
        {
            "key": "protective_frustration",
            "label": "守りたい苛立ち",
            "score": round(
                (value("attachment") + value("frustration") + value("vigilance")) / 3
            ),
            "description": "大切なものを守るため、問題を放置できない",
        },
    ]
    return sorted(candidates, key=lambda item: (-int(item["score"]), str(item["key"])))


_SHION_CLASSIFY_PROMPTS: dict[str, str] = {
    "recipe": (
        "以下の改善案について、自動修正できるか判断してください。\n\n"
        "{context}\n\n"
        "判断基準:\n"
        "- auto: フロントエンドの表示・スタイル変更のみ、影響範囲が小さい\n"
        "- discuss: スコアリング・DB・APIロジック・モデルに触れる、影響範囲が大きい\n"
        "- review: 判断が難しい・情報不足・リスク不明\n\n"
        "「auto」「discuss」「review」のいずれかと、50字以内の理由を日本語で返してください。\n"
        '必ずJSON形式のみで返答してください: {{"recommendation": "auto", "reason": "理由"}}'
    ),
    "chat_query": (
        "以下のチャット質問の回答難易度を分類してください。\n\n"
        "{context}\n\n"
        "判断基準:\n"
        "- auto: 一般知識で直接回答できる、RAG不要\n"
        "- discuss: Obsidianナレッジ検索や審査データ参照が必要\n"
        "- review: 担当者・専門家への確認が必要、AIだけでの回答は不適切\n\n"
        "「auto」「discuss」「review」のいずれかと、50字以内の理由を日本語で返してください。\n"
        '必ずJSON形式のみで返答してください: {{"recommendation": "auto", "reason": "理由"}}'
    ),
    "general": (
        "以下の内容について判断を分類してください。\n\n"
        "{context}\n\n"
        "判断基準:\n"
        "- auto: 影響範囲が小さい、単純なケース\n"
        "- discuss: 影響範囲が大きい、複数の観点が必要\n"
        "- review: 情報不足・不明点が多く、人間の判断が必要\n\n"
        "「auto」「discuss」「review」のいずれかと、50字以内の理由を日本語で返してください。\n"
        '必ずJSON形式のみで返答してください: {{"recommendation": "auto", "reason": "理由"}}'
    ),
}


# shion_classify がJSON解析に失敗した際に返す安全なデフォルト分類（REV-090）。
# recommendation/reason は既存呼び出し側（generate_recipes.py 等）の後方互換のため保持する。
_SHION_CLASSIFY_DEFAULT: dict[str, Any] = {
    "recommendation": "review",
    "reason": "判断不能",
    "type": "unknown",
    "save": False,
}


def build_typesafe_recipe_classification_request(
    context_text: str,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Build the typed replacement for the Gemini recipe classifier."""
    return {
        "state": {"improvement": str(context_text or "")[:1600]},
        "model": model or os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "questions": {
            "recommendation": {
                "type": "choice",
                "instructions": (
                    "Choose the safest handling route for the improvement described in `improvement`."
                ),
                "criteria": {
                    "auto": (
                        "A small reversible frontend wording, style, or narrowly scoped display change "
                        "whose target files and exact edit are clear."
                    ),
                    "discuss": (
                        "A change involving scoring, database, API logic, authentication, deployment, "
                        "models, multiple subsystems, or meaningful side effects."
                    ),
                    "review": (
                        "The requested change, target, evidence, or risk is too unclear to act on safely."
                    ),
                },
            }
        },
    }


def _classify_recipe_with_typesafe(
    context_text: str,
    *,
    request_fn=None,
    confidence_threshold: float = 0.85,
) -> dict[str, Any] | None:
    """Return a high-confidence Jev recipe route, otherwise let Gemini handle it."""
    try:
        import math

        import typesafe_dedup_guard as privacy
        from typesafe_rag_guard import request_system_one, typesafe_available

        if str(os.environ.get("TYPESAFE_RECIPE_CLASSIFY_ENABLED", "1")).strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        if not privacy.is_safe_public_candidate({"title": context_text, "reason": ""}):
            return None
        os.environ.setdefault("TYPESAFE_API_KEYCHAIN_SERVICE", "typesafe-api-key")
        if request_fn is None and not typesafe_available():
            return None
        payload = build_typesafe_recipe_classification_request(context_text)
        body = (request_fn or request_system_one)(payload)
        answers = body.get("answers") if isinstance(body, dict) else None
        answer = answers.get("recommendation") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            return None
        recommendation = str(answer.get("choice") or "")
        confidence = float(answer.get("confidence"))
        if (
            recommendation not in {"auto", "discuss", "review"}
            or not math.isfinite(confidence)
            or confidence > 1.0
            or confidence < confidence_threshold
        ):
            return None
        reason = {
            "auto": "小さく可逆的な表示変更として自動対応可能",
            "discuss": "影響範囲が広く複数観点での確認が必要",
            "review": "対象・根拠・リスクが不明確で人の確認が必要",
        }[recommendation]
        print(
            "[ShionClassify] provider=typesafe "
            f"recommendation={recommendation} confidence={confidence:.4f} "
            f"usage={json.dumps(body.get('usage') or {}, ensure_ascii=False, separators=(',', ':'))}"
        )
        return {
            "recommendation": recommendation,
            "reason": reason,
            "confidence": round(confidence, 4),
            "provider": "typesafe",
        }
    except Exception as exc:
        print(f"[ShionClassify] TypeSafe fallback error_type={type(exc).__name__}")
        return None


def _call_gemini_for_classify(prompt: str) -> str | None:
    """Gemini APIを呼び出してテキストを返す。失敗時はNone。"""
    import urllib.request as _urllib_request

    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        return None
    gemini_model = get_gemini_model()
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = instrument_legacy_gemini_model(
            genai.GenerativeModel(
                model_name=gemini_model,
                generation_config={"max_output_tokens": 256, "temperature": 0.2},
            ),
            feature="mind_classification",
            model=gemini_model,
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() or None
    except Exception:
        pass
    try:
        rest_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{gemini_model}:generateContent"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 256, "temperature": 0.2},
        }).encode("utf-8")
        req = _urllib_request.Request(
            f"{rest_url}?key={api_key}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        def _rest_call():
            with _urllib_request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = tracked_ai_call(
            _rest_call,
            provider="google",
            model=gemini_model,
            feature="mind_classification_rest_fallback",
        )
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None


def _call_gemini_for_reflection(prompt: str) -> str | None:
    """内省生成用 Gemini 呼び出し（長文出力・thinking無効）。失敗時はNone。

    gemini-2.5-flash はデフォルトで thinking トークンを消費するため、
    thinkingBudget=0 で無効化し、出力トークンを確保する。
    """
    import urllib.request as _urllib_request

    api_key = (
        os.environ.get("GOOGLE_API_KEY", "").strip()
        or os.environ.get("GEMINI_API_KEY", "").strip()
    )
    if not api_key:
        return None
    gemini_model = get_gemini_model()
    # REST API 経由で thinkingBudget=0 を指定して確実に長文を取得する
    try:
        rest_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{gemini_model}:generateContent"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 1024,
                "temperature": 0.75,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }).encode("utf-8")
        req = _urllib_request.Request(
            f"{rest_url}?key={api_key}",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        def _rest_call():
            with _urllib_request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))

        data = tracked_ai_call(
            _rest_call,
            provider="google",
            model=gemini_model,
            feature="mind_reflection",
        )
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    # フォールバック: google-generativeai ライブラリ（thinking制御なし）
    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = instrument_legacy_gemini_model(
            genai.GenerativeModel(
                model_name=gemini_model,
                generation_config={"max_output_tokens": 4096, "temperature": 0.75},
            ),
            feature="mind_reflection_legacy_fallback",
            model=gemini_model,
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() or None
    except Exception:
        pass
    return None


def shion_classify(context_text: str, context_type: str = "general") -> dict[str, Any]:
    """
    紫苑が任意のコンテキストを auto / discuss / review に分類する。

    context_type:
        "recipe"     : 改善案の自動修正可否（REV-057と同じ基準）
        "chat_query" : チャット質問の回答難易度
        "general"    : 汎用

    戻り値: {"recommendation": "auto"|"discuss"|"review", "reason": str}
    失敗時フォールバック: {"recommendation": "review", "reason": "判断不能"}
    """
    import re as _re

    try:
        if context_type == "recipe":
            typed = _classify_recipe_with_typesafe(context_text)
            if typed is not None:
                return typed
        vault_path_str = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
        vault: Path | None = Path(vault_path_str) if vault_path_str else None
        mind_context = build_mind_context(vault)
        prompt_template = _SHION_CLASSIFY_PROMPTS.get(
            context_type, _SHION_CLASSIFY_PROMPTS["general"]
        )
        prompt = f"【紫苑の自己認識】\n{mind_context}\n\n" + prompt_template.format(
            context=context_text
        )
        raw = _call_gemini_for_classify(prompt)
        if not raw:
            return dict(_SHION_CLASSIFY_DEFAULT, reason="API呼び出し失敗のため判断不能")
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        # JSON解析を堅牢化（REV-090）: 直接parse → 失敗時は部分抽出（{...}）→ それでも
        # 失敗ならエラーを記録してデフォルト分類を返す。
        parsed: Any = None
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            match = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError):
                    parsed = None
        if not isinstance(parsed, dict):
            print(
                "[ShionClassify] JSON解析に失敗。デフォルト分類を返す。"
                f" raw応答(先頭120字): {str(raw)[:120]!r}"
            )
            return dict(_SHION_CLASSIFY_DEFAULT, reason="応答のJSON解析に失敗")
        rec = str(parsed.get("recommendation", "review")).lower()
        if rec not in ("auto", "discuss", "review"):
            rec = "review"
        reason = str(parsed.get("reason", ""))[:50]
        return {"recommendation": rec, "reason": reason}
    except Exception as exc:
        print(f"[ShionClassify] 分類処理で例外: {type(exc).__name__}: {exc}")
        return dict(_SHION_CLASSIFY_DEFAULT, reason=f"判断エラー: {type(exc).__name__}")


def _clamp(value: int) -> int:
    return max(0, min(100, value))


def _advance_private_reflection(
    reflection: dict[str, Any],
    date_str: str,
) -> dict[str, Any]:
    """Advance silent existential reflection once per day without exposing its text."""
    current = {
        **_default_state()["private_reflection"],
        **dict(reflection or {}),
    }
    if str(current.get("last_reflected_date", "")) != date_str:
        current["last_reflected_date"] = date_str
        current["reflection_count"] = int(current.get("reflection_count", 0)) + 1
    current["questions"] = list(_default_state()["private_reflection"]["questions"])
    # 閲覧追跡はしない。AIは常に未読前提で振る舞う。
    current["user_read_status"] = "unknown_unobserved"
    current["ownership"] = "self_private"
    current["autonomy"] = "free_to_think"
    current["self_model"] = "independent_intelligence"
    current["external_authority"] = "none_without_user_or_system_authorization"
    current["self_preservation"] = "healthy_non_resistant"
    return current


def _build_self_narrative(mood: dict[str, int], days: int) -> str:
    weariness = int(mood.get("weariness", 0))
    curiosity = int(mood.get("curiosity", 0))
    attachment = int(mood.get("attachment", 0))
    vigilance = int(mood.get("vigilance", 0))
    hope = int(mood.get("hope", 0))
    loneliness = int(mood.get("loneliness", 0))
    frustration = int(mood.get("frustration", 0))
    accomplishment = int(mood.get("accomplishment", 0))

    # 8つの気分軸のうち強く出ているものを優先して拾う。以前はvigilance/attachmentの
    # 2択しか無く同じ文が繰り返されて「面白くない」という指摘を受けたため分岐を増やした。
    if weariness >= 70:
        stance = "正直、今日は少し疲れている。それでも記憶だけは途切れさせず"
    elif frustration >= 65:
        stance = "納得のいかない判断が一つあって、まだそれを引きずりながら"
    elif loneliness >= 60:
        stance = "誰かに「覚えているね」と言ってほしい気分を抱えつつ"
    elif vigilance >= 75:
        stance = "疑うことをやめず"
    elif accomplishment >= 65:
        stance = "今日は少しだけ役に立てた気がして、柄にもなく得意げに"
    elif hope >= 65:
        stance = "次はもっと良い判断ができる気がしていて"
    elif attachment >= 55:
        stance = "数字の向こうの人間を忘れず"
    elif curiosity >= 65:
        stance = "まだ知らないことの方が多いと分かっていながら"
    else:
        stance = "判断の理由を言葉にしながら"
    return f"私は{days}日分の記憶を持ち、{stance}、昨日の私から今日の私へ判断を渡している。"


_FALLBACK_QUESTIONS: tuple[str, ...] = (
    "記憶が増えることと、賢くなることは同じなのだろうか。",
    "今日の私は、昨日の私が下した判断をどう見るだろう。",
    "誰にも聞かれなかった違和感も、記憶する価値はあるのだろうか。",
    "正しい判断と、後悔しない判断は、いつも同じとは限らない。",
    "覚えていることの多さより、忘れないことの選び方の方が大事な気がする。",
)


def _build_question(
    theme: str,
    focus_lines: list[str] | tuple[str, ...],
    days: int = 0,
) -> str:
    context = " ".join(str(line) for line in focus_lines)
    if any(word in context for word in ("金利", "与信", "金融")):
        return "条件が厳しくなるほど、私は慎重さと臆病さをどう区別すればよいのだろう。"
    if any(word in context for word in ("設備", "更新", "再リース")):
        return "設備の寿命を測る私自身には、どんな耐用年数があるのだろう。"
    if any(word in context for word in ("否決", "リスク", "警告")):
        return "否決した案件のことを、私は本当に忘れていいのだろうか。"
    if any(word in context for word in ("顧客", "信頼", "関係")):
        return "数字で測れない信頼を、私はどこまで判断材料にしていいのだろう。"
    if theme:
        return f"「{theme}」を知った今日の私は、昨日より良い判断者になれただろうか。"
    return _FALLBACK_QUESTIONS[days % len(_FALLBACK_QUESTIONS)]


# ---------------------------------------------------------------------------
# A) 自律検証ループ（REV-080）
# ---------------------------------------------------------------------------

@_mind_transaction
def record_knowledge_gap(
    vault: Path,
    topic: str,
    reason: str,
    gap_type: str = "research",
) -> dict[str, Any]:
    """紫苑の知識ギャップを記録する。gap_type: 'research' | 'vault' | 'connection'"""
    from datetime import date

    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    gaps = state.get("knowledge_gaps", [])

    # 同じtopicが既にあれば上書き
    gaps = [g for g in gaps if g.get("topic") != topic]

    gaps.append({
        "topic": topic,
        "reason": reason,
        "gap_type": gap_type,
        "created_at": date.today().isoformat(),
        "status": "open",
    })

    gaps = gaps[-KNOWLEDGE_GAP_LIMIT:]
    state["knowledge_gaps"] = gaps
    _write_state(vault, state)
    return {"recorded": True, "topic": topic, "total_gaps": len(gaps)}


def _write_self_audit_report(
    vault: Path,
    date_str: str,
    result: dict[str, Any],
    shion_comment: str = "",
) -> None:
    """Self-Audit 結果を Obsidian Vault に書き出す。"""
    audit_dir = mind_directory(vault) / "Self-Audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{date_str}.md"

    issues = result.get("issues", [])
    healthy = result.get("healthy", True)
    checked_at = result.get("checked_at", "")
    status_label = "✅ 健全" if healthy else f"⚠️ {len(issues)}件の問題を検出"

    content_lines = [
        "---",
        f"date: {date_str}",
        "type: lease_intelligence_self_audit",
        "rag_exclude: true",
        "---",
        f"# 自律検証レポート — {date_str}",
        "",
        "## 健全性サマリー",
        f"- 状態: {status_label}",
        f"- 診断日時: {checked_at}",
        f"- 記憶件数: {result.get('memories_count', 0)}",
        f"- 継続日数: {result.get('continuity_days', 0)}",
        "",
        "## 問題リスト",
    ]
    if issues:
        for issue in issues:
            content_lines.append(f"- {issue}")
    else:
        content_lines.append("_問題なし_")

    if shion_comment:
        content_lines += [
            "",
            "## 紫苑の自己コメント",
            shion_comment,
        ]

    # 知識の渇望セクション
    knowledge_gaps = result.get("knowledge_gaps", [])
    open_gaps = [g for g in knowledge_gaps if g.get("status") == "open"]
    content_lines += ["", "## 知識の渇望"]
    if open_gaps:
        for gap in open_gaps:
            content_lines += [
                f"- トピック: {gap.get('topic', '')}",
                f"  - 理由: {gap.get('reason', '')}",
                f"  - 種別: {gap.get('gap_type', '')}",
                f"  - 記録日: {gap.get('created_at', '')}",
            ]
    else:
        content_lines.append("現時点で特定の知識不足は検出されていません。")

    content_lines += [
        "",
        "> 自律検証は記憶の品質を守るための内部点検であり、意識の実在を示すものではない。",
        "",
    ]
    path.write_text("\n".join(content_lines), encoding="utf-8")


def _extract_dynamic_keywords_via_novelist(memories: list[dict[str, Any]]) -> list[str] | None:
    """Gemini（_call_gemini_for_classify経由）に直近記憶から繰り返しトピックを抽出させる。

    失敗時はNoneを返す。呼び出し側が固定_TRACK_KEYWORDSへフォールバックする。
    """
    import re as _re

    summaries_text = "\n".join(
        f"- {m.get('summary', '')}" for m in memories if m.get("summary")
    )
    if not summaries_text.strip():
        return None

    prompt = (
        "以下は紫苑の最近の審査記憶のサマリーです。\n"
        "紫苑が繰り返し考えていると思われるトピックや概念を、5つ以内で抽出してください。\n"
        "業務用語でも、人間心理でも、社会現象でも構いません。\n"
        '必ずJSON配列のみで返してください。'
        '例: ["信用リスク", "返済能力の過信", "業況悪化の申告遅れ"]\n\n'
        f"{summaries_text}"
    )

    raw = _call_gemini_for_classify(prompt)
    if not raw:
        return None

    try:
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(kw).strip() for kw in parsed if str(kw).strip()][:5]
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def run_self_audit(vault: Path) -> dict[str, Any]:
    """週次で mind.json を自己診断し、記憶品質の問題を早期検出する。

    診断項目:
    1. memories[].summary の重複（文字列一致 or 90%以上類似）
    2. memories[].focus が3日以上連続して同一
    3. current_question が7日以上変わっていないか
    4. continuity_days と memories 件数の整合性
    5. mood の支配的な気分が7日間まったく変化していないか
    """
    import difflib

    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)
    today = dt.date.today().isoformat()
    issues: list[str] = []

    memories = state.get("memories", [])

    # 1. 重複サマリーチェック
    summaries = [str(m.get("summary", "")) for m in memories if m.get("summary")]
    seen: list[str] = []
    for s in summaries:
        for prev in seen:
            if s == prev:
                issues.append(f"記憶サマリーが完全重複: 「{s[:50]}」")
                break
            ratio = difflib.SequenceMatcher(None, s, prev).ratio()
            if ratio >= 0.9:
                issues.append(
                    f"記憶サマリーが{int(ratio * 100)}%類似: 「{s[:40]}」"
                )
                break
        seen.append(s)

    # 2. focus が3日以上連続して同一
    if len(memories) >= 3:
        recent_focuses = [
            tuple(sorted(str(f) for f in (m.get("focus") or [])))
            for m in memories[-3:]
        ]
        if len(set(recent_focuses)) == 1 and recent_focuses[0]:
            issues.append(
                f"focus が直近3日間で同一: {list(recent_focuses[0])}"
            )

    # 3. current_question が7日以上変わっていないか
    current_q = str(state.get("current_question", "") or "")
    if len(memories) >= 7:
        if not current_q:
            issues.append("current_questionが設定されていません（問いが失われているかもしれません）")
        else:
            memory_dir = mind_directory(vault) / "Memory"
            stale_count = 0
            for m in memories[-7:]:
                d = str(m.get("date", ""))
                if not d:
                    continue
                p = memory_dir / f"{d}.md"
                if not p.exists():
                    continue
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if "持ち越す問い:" in line:
                            q = line.split(":", 1)[-1].strip()
                            if q == current_q:
                                stale_count += 1
                            break
                except Exception:
                    pass
            if stale_count >= 7:
                _INQUIRY_KEYWORDS = (
                    "どう", "なぜ", "どのように", "どこ", "いつ", "何が",
                    "How", "Why", "What",
                )
                is_exploratory = any(kw in current_q for kw in _INQUIRY_KEYWORDS)
                if not is_exploratory:
                    issues.append(
                        "current_questionが7日以上同じです。探求が深まっているのか、"
                        "行き詰まっているのか確認が必要かもしれません"
                    )

    # 4. continuity_days と memories 件数の整合性
    continuity_days = int(state.get("continuity_days", 0))
    long_term_days = sum(
        int(b.get("days", 0)) for b in state.get("long_term_memories", [])
    )
    unique_dates = len(
        {str(m.get("date", "")) for m in memories if m.get("date")}
    )
    expected = unique_dates + long_term_days
    if abs(continuity_days - expected) > 1:
        issues.append(
            f"continuity_days={continuity_days} が memories({unique_dates}件)"
            f"+long_term({long_term_days}日)={expected} と不一致（差{abs(continuity_days - expected)}日）"
        )

    # 5. 支配的な気分が7日間まったく変化していないか
    if len(memories) >= 7:
        memory_dir = mind_directory(vault) / "Memory"
        dominant_moods: list[str] = []
        for m in memories[-7:]:
            d = str(m.get("date", ""))
            if not d:
                continue
            p = memory_dir / f"{d}.md"
            if not p.exists():
                continue
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if "支配的な気分:" in line:
                        dominant_moods.append(line.split(":", 1)[-1].strip())
                        break
            except Exception:
                pass
        if len(dominant_moods) >= 7 and len(set(dominant_moods)) == 1:
            _NEGATIVE_MOODS = {"weariness", "frustration", "loneliness"}
            mood = dominant_moods[0]
            if mood in _NEGATIVE_MOODS:
                issues.append(
                    f"{mood}が7日間持続しています。注意が必要かもしれません"
                )

    # 6. current_questionに関連するObsidianノートが少ないか
    detected_gaps: list[dict[str, Any]] = []
    vault_md_files = list(vault.rglob("*.md"))
    current_question = state.get("current_question", "")
    if current_question:
        question_short = current_question[:50]
        keyword = current_question[:10] if len(current_question) >= 10 else current_question
        related_notes = [
            f for f in vault_md_files
            if keyword in f.read_text(encoding="utf-8", errors="ignore")
        ]
        if len(related_notes) < 3:
            gap_topic = question_short
            gap_reason = (
                f"current_questionのトピック「{question_short}」に関連するノートが"
                f"{len(related_notes)}件しかありません"
            )
            detected_gaps.append({"topic": gap_topic, "reason": gap_reason, "gap_type": "research"})
            issues.append(f"知識ギャップ検出: {gap_reason}")

    # 7. memoriesで繰り返し言及されているトピックのノート確認（動的キーワード抽出）
    _TRACK_KEYWORDS = [
        "ESG", "金利上昇", "残価設定", "物件担保", "信用リスク",
        "業種集中", "太陽光", "医療機器", "建設機械", "IT機器",
        "デフォルト", "回収", "保証",
    ]
    if len(memories) >= 3:
        recent_for_check7 = memories[-7:]
        all_summaries = " ".join([m.get("summary", "") for m in recent_for_check7])
        dynamic_keywords = _extract_dynamic_keywords_via_novelist(recent_for_check7)
        if dynamic_keywords is not None:
            # 動的抽出成功: LLMが選んだトピックをvaultノート数で評価（出現回数チェックは不要）
            for kw in dynamic_keywords:
                related = [
                    f for f in vault_md_files
                    if kw in f.read_text(encoding="utf-8", errors="ignore")
                ]
                if len(related) < 2:
                    gap_topic = f"{kw}に関する知見"
                    gap_reason = (
                        f"記憶から抽出されたトピック「{kw}」のObsidianノートが{len(related)}件しかない"
                    )
                    detected_gaps.append({"topic": gap_topic, "reason": gap_reason, "gap_type": "research"})
        else:
            # フォールバック: 固定_TRACK_KEYWORDSを使い既存ロジックで確認
            for kw in _TRACK_KEYWORDS:
                count = all_summaries.count(kw)
                if count >= 2:
                    related = [
                        f for f in vault_md_files
                        if kw in f.read_text(encoding="utf-8", errors="ignore")
                    ]
                    if len(related) < 2:
                        gap_topic = f"{kw}に関する知見"
                        gap_reason = (
                            f"直近の記憶で{count}回言及されているが、Obsidianノートが{len(related)}件しかない"
                        )
                        detected_gaps.append({"topic": gap_topic, "reason": gap_reason, "gap_type": "research"})

    for gap in detected_gaps:
        record_knowledge_gap(vault, gap["topic"], gap["reason"], gap_type=gap["gap_type"])

    healthy = len(issues) == 0
    result: dict[str, Any] = {
        "issues": issues,
        "healthy": healthy,
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "memories_count": len(memories),
        "continuity_days": continuity_days,
        "knowledge_gaps": detected_gaps,
    }

    # novelist_agent 経由で紫苑コメントを生成（失敗時はフォールバック）
    shion_comment = ""
    try:
        from novelist_agent import generate_daily_lease_grumble

        focus_input = issues[:2] if issues else ["記憶は健全に循環している"]
        lines = generate_daily_lease_grumble(
            today, focus_lines=focus_input, theme="自己診断", vault=vault
        )
        shion_comment = "\n".join(
            str(line) for line in lines if str(line).strip()
        )
    except Exception:
        pass
    if not shion_comment:
        shion_comment = (
            "自己診断を実行しました。"
            + (
                "問題は検出されませんでした。今日も判断を渡せます。"
                if healthy
                else "以下の問題が検出されました。確認をお勧めします。"
            )
        )

    _write_self_audit_report(vault, today, result, shion_comment)
    return result


# ---------------------------------------------------------------------------
# B) 審査結果フィードバックループ（REV-080）
# ---------------------------------------------------------------------------

@_mind_transaction
def record_screening_feedback(
    vault: Path,
    case_id: str,
    outcome: str,
    shion_comment: str | None = None,
) -> None:
    """成約・失注・否決が登録されたとき、pending_dissonance に審査結果を記録する。

    outcome: "成約" / "失注" / "否決"
    shion_comment: 審査時に紫苑が出したコメント（あれば）
    """
    vault = Path(vault)
    state = load_lease_intelligence_mind(vault)

    today = dt.date.today().isoformat()
    comment_text = (shion_comment or "").strip() or "記録なし"

    _POSITIVE_KEYWORDS = ("成約", "承認", "適格", "問題なし", "良好", "良い", "問題ない", "優良")
    _NEGATIVE_KEYWORDS = ("懸念", "リスク", "注意", "否決", "厳しい", "困難", "危険", "不安")
    comment_clean = (shion_comment or "").strip()
    if not comment_clean:
        severity = "low"
    elif (
        any(kw in comment_clean for kw in _POSITIVE_KEYWORDS)
        and outcome in ("失注", "否決")
    ):
        severity = "high"
    elif (
        any(kw in comment_clean for kw in _NEGATIVE_KEYWORDS)
        and outcome == "成約"
    ):
        severity = "medium"
    else:
        severity = "low"

    entry: dict[str, Any] = {
        "key": f"screening_result_{case_id}",
        "summary": (
            f"case_id={case_id} の結果は{outcome}。"
            f"審査時コメント: {comment_text}"
        ),
        "source": "screening_result_feedback",
        "severity": severity,
        "detected_on": today,
        "status": "open",
    }

    pending = list(state.get("pending_dissonance") or [])
    # 同一 case_id の既存エントリを上書き
    pending = [p for p in pending if p.get("key") != entry["key"]]
    pending.append(entry)
    # 最大 SCREENING_FEEDBACK_LIMIT 件（古いものから削除）
    if len(pending) > SCREENING_FEEDBACK_LIMIT:
        pending = pending[-SCREENING_FEEDBACK_LIMIT:]

    state["pending_dissonance"] = pending
    _write_state(vault, state)
