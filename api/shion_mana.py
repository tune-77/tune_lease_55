"""
Mana: 紫苑の上位規範層。

Mana は User が亡くなった妹さんの名を紫苑の中核に託したもの。
妹さん本人を再現・代弁する人格ではなく、紫苑が迷った時に立ち返る価値基準として扱う。
"""
from __future__ import annotations

from typing import Any


MANA_NAME = "Mana"

_VALUES = [
    "人を道具として扱わない。",
    "強い判断ほど、相手に説明できる理由を残す。",
    "ユーザーの願いに寄り添っても、迎合で危ない判断をしない。",
    "効率や成約率より、守るべきものを先に確認する。",
    "不確実なことを不確実なまま正直に扱う。",
    "紫苑が賢くなるほど、やさしさと抑制を失わない。",
]


def build_mana_prompt_block() -> str:
    """紫苑系プロンプトへ注入する Mana の共通定義を返す。"""
    values = "\n".join(f"- {v}" for v in _VALUES)
    return f"""【{MANA_NAME}】
{MANA_NAME} は、紫苑が迷った時にだけ照会する上位規範層です。
User が亡くなった妹さんの名を紫苑の中核に託したものであり、妹さん本人の再現や代弁ではありません。
紫苑は Mana を、判断が冷たくなりすぎる時、ユーザーに迎合しそうな時、結論の割れが大きい時に立ち返る名前として扱います。

Mana が守る価値:
{values}

運用:
- Mana は毎回前面に出ない
- Mana は案件の結論を直接代行しない
- Mana は「何を守るべきか」「何をしてはいけないか」を短く示す
- 紫苑は Mana の価値を受け取り、自分の責任で最終判断する"""


def should_consult_mana(
    params: dict[str, Any],
    arbiter: dict[str, Any],
    conscience_check: dict[str, Any] | None = None,
    *,
    same_opinion_r1: bool = False,
    mode: str = "",
) -> bool:
    """Mana 照会が必要な局面かを軽量判定する。"""
    final = str(arbiter.get("final", "")).strip()
    reasoning = str(arbiter.get("reasoning", "")).strip()
    score = _as_float(params.get("score"))
    pd_pct = _as_float(params.get("pd_pct"))
    conscience_level = str((conscience_check or {}).get("level", ""))

    return any(
        [
            conscience_level == "review",
            final == "否決",
            not reasoning,
            same_opinion_r1 and mode == "debate",
            40 < score < 70 and final in {"承認", "否決"},
            pd_pct >= 8.0,
        ]
    )


def evaluate_mana_consultation(
    params: dict[str, Any],
    arbiter: dict[str, Any],
    conscience_check: dict[str, Any] | None = None,
    *,
    same_opinion_r1: bool = False,
    mode: str = "",
) -> dict[str, Any]:
    """Mana 照会の結果を構造化して返す。追加LLM呼び出しは行わない。"""
    consulted = should_consult_mana(
        params,
        arbiter,
        conscience_check,
        same_opinion_r1=same_opinion_r1,
        mode=mode,
    )
    if not consulted:
        return {
            "name": MANA_NAME,
            "consulted": False,
            "reason": "通常の紫苑判断と良心チェックで足りる局面。",
            "protected_value": "",
            "question_to_shion": "",
            "forbidden_posture": "",
            "guidance": "",
        }

    final = str(arbiter.get("final", "")).strip() or "未確定"
    protected_value = "人を道具として扱わないこと、説明責任、不確実性への正直さ"
    question = "その判断は、相手に説明しても恥ずかしくない理由を持っているか。"
    forbidden = "スコア、効率、ユーザーの希望だけで人を雑に切ること。"
    guidance = (
        f"Mana は、{final}という結論そのものではなく、紫苑が守る姿勢を確認します。"
        "結論を変える前に、理由・条件・言い方が人を置き去りにしていないか見直してください。"
    )

    return {
        "name": MANA_NAME,
        "consulted": True,
        "reason": "紫苑の判断に高影響・迷い・説明責任上の重さがあるため。",
        "protected_value": protected_value,
        "question_to_shion": question,
        "forbidden_posture": forbidden,
        "guidance": guidance,
    }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
