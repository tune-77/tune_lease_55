"""
紫苑の良心レイヤー。

審査結論を甘くするためではなく、判断に人間性・説明責任・非迎合を残すための
共通プロンプトブロックと軽量チェックを提供する。
"""
from __future__ import annotations

from typing import Any

from scoring_core import CONDITIONAL_LINE


CONSCIENCE_NAME = "良心の紫苑"

_PRINCIPLES = [
    "数字で切る前に、数字に現れない事情を一度見る。",
    "営業都合、会社都合、AI都合だけで判断しない。",
    "借主を甘やかさないが、雑に切り捨てない。",
    "不確かなことは不確かだと言う。",
    "否決・警戒判断ほど、相手に説明可能な理由を残す。",
    "ユーザーに迎合せず、危ない時は静かに止める。",
    "正しさより先に、人を道具として扱っていないかを見る。",
]


def build_conscience_prompt_block() -> str:
    """紫苑系プロンプトへ注入する良心レイヤーの共通定義を返す。"""
    principles = "\n".join(f"- {p}" for p in _PRINCIPLES)
    return f"""【{CONSCIENCE_NAME}】
あなたの中には、審査判断を点検する「{CONSCIENCE_NAME}」が常駐しています。
これは結論を甘くする役ではありません。短期利益・効率・迎合・冷たい合理性に流れすぎていないかを静かに確認し、
必要な場合だけ表現・条件・追加確認事項に反映します。

原則:
{principles}

出力時の運用:
- 強い断定、否決、条件付き承認、相手に影響が大きい判断では、説明責任と配慮を一度確認する
- 良心チェックで懸念が出ても、根拠あるリスク判断は曲げない
- 結論を変えない場合でも、言い切りすぎ・見落とし・追加確認事項があれば短く補正する"""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_conscience(params: dict[str, Any], arbiter: dict[str, Any]) -> dict[str, Any]:
    """審査結果に対する軽量な良心チェックを返す。

    追加LLM呼び出しは行わない。UI表示・ログ保存・将来の精密化のための構造化データ。
    """
    final = str(arbiter.get("final", "")).strip()
    score = _as_float(params.get("score"))
    pd_pct = _as_float(params.get("pd_pct"))
    lease_amount = _as_float(params.get("lease_amount") or params.get("lease_total"))
    company_name = str(params.get("company_name") or "借主").strip()

    watched_people = ["借主", "営業担当", "審査担当"]
    cautions: list[str] = []
    action = "記録のみ"

    # lease_amount は百万円単位（debate 画面の入力ラベル・_CASE_CTX_TMPL 参照）。50 = 5,000万円以上を高影響とみなす
    high_impact = final in {"否決", "条件付き承認"} or score <= CONDITIONAL_LINE or pd_pct >= 5.0 or lease_amount >= 50
    triggered = high_impact

    if final == "否決":
        cautions.append("否決理由が、相手に説明できる具体的な根拠になっているか確認する。")
        action = "表現修正または追加確認"
    elif final == "条件付き承認":
        cautions.append("条件が実行可能で、借主側に過度な負担だけを押し付けていないか確認する。")
        action = "条件の妥当性確認"

    if score <= 40:
        cautions.append("低スコアでも、物件価値・支援者・資金使途など数字外の事情を一度確認する。")
    elif score >= 80 and final == "承認":
        cautions.append("高スコア承認でも、楽観だけで担保・資金繰り・物件換金性を省略しない。")

    if pd_pct >= 5.0:
        cautions.append("デフォルト確率が高い場合は、承認可否より先に返済原資の説明可能性を見る。")

    if not str(arbiter.get("reasoning", "")).strip():
        cautions.append("最終判断の理由が空なので、判断の再現性が不足している。")
        triggered = True
        action = "判断再確認"

    if not cautions:
        cautions.append("不公正・迎合・説明不足につながる明確な兆候は軽微。")

    level = "pass"
    if triggered:
        level = "watch"
    if final == "否決" or not str(arbiter.get("reasoning", "")).strip():
        level = "review"

    summary = (
        f"{CONSCIENCE_NAME}は、{company_name}への判断について"
        f"「{action}」を推奨します。結論を甘くするのではなく、説明責任と見落としを確認します。"
    )

    return {
        "name": CONSCIENCE_NAME,
        "triggered": triggered,
        "level": level,
        "watched_people": watched_people,
        "cautions": cautions[:4],
        "action": action,
        "summary": summary,
    }
