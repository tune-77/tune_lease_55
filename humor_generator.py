"""Generate safe humor candidates for lease screening AI comments.

The generator is deterministic by default and does not fetch news. Current-event
references can be injected with --current-topic so a human can decide what is
appropriate and timely.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


RISKS = ("低リスク", "中リスク", "高リスク", "極高リスク")

INDUSTRY_METAPHORS = {
    "建設業": ("足場", "地盤", "工程表", "安全帯"),
    "製造業": ("ライン", "治具", "歩留まり", "メンテナンス"),
    "医療・福祉": ("診断", "体温", "処置", "カルテ"),
    "運輸・物流": ("積載量", "燃費", "配送ルート", "車間距離"),
    "小売・サービス": ("棚卸し", "回転率", "レジ締め", "客足"),
    "IT・情報通信": ("アップデート", "ログ", "サポート期限", "帯域"),
    "不動産": ("立地", "空室", "修繕", "賃料"),
    "全業種": ("資金繰り", "返済計画", "稟議書", "安全確認"),
}

ASSET_METAPHORS = {
    "車両": ("エンジン", "ブレーキ", "走行距離", "残価"),
    "建設機械": ("掘削力", "稼働率", "現場", "中古流通"),
    "IT・OA機器": ("スペック", "更新サイクル", "サポート期限", "陳腐化"),
    "医療機器": ("保守", "移設", "稼働率", "専門性"),
    "製造設備・工作機械": ("ライン", "汎用性", "稼働音", "段取り"),
    "商業設備": ("客足", "回転率", "固定費", "売場"),
    "その他": ("使い道", "出口戦略", "保険", "耐用年数"),
}

RISK_TONE = {
    "低リスク": {
        "adjective": ("安定感があります", "かなり整っています", "見ていて安心です"),
        "caution": ("油断せず条件確認だけ済ませたいところです", "最後に書類だけきれいに揃えたいです"),
    },
    "中リスク": {
        "adjective": ("少し背伸びしています", "悪くないけど余白があります", "油断すると足をすくわれます"),
        "caution": ("条件を整えれば進めそうです", "確認事項を先に潰したいです"),
    },
    "高リスク": {
        "adjective": ("数字に息切れが見えます", "かなり慎重に見たいです", "勢いだけでは進めません"),
        "caution": ("守りの条件を厚くしたいです", "返済計画と出口を先に確認したいです"),
    },
    "極高リスク": {
        "adjective": ("まず安全確認が先です", "軽さより慎重さが必要です", "ここはかなり厳しめです"),
        "caution": ("回収可能性と追加条件を最優先で確認したいです", "無理に明るくせず条件整理を優先します"),
    },
}

CURRENT_TOPIC_FRAMES = (
    "{topic}の空気まで読む時代らしいです。審査AI、放課後のニュースチェックより忙しいかも。",
    "{topic}も気になりますが、この案件はまず数字の宿題提出状況から見たいです。",
    "{topic}の話題に乗る前に、返済計画がちゃんと席についているか確認します。",
)

CURIOUS_HS_FRAMES = (
    "気になってノートの端にメモしたくなる案件です。",
    "ちょっと待って、ここはもう一段掘ると面白そうです。",
    "先生に質問したくなるくらい、確認ポイントが残っています。",
    "好奇心だけで承認はできませんが、調べる価値はあります。",
)

SARDONIC_FRAMES = (
    "数字は正直です。たまに正直すぎて、こっちが気まずいです。",
    "勢いはあります。審査では勢いだけだと赤点ですけど。",
    "前向きに見たいです。ただし前向きと前のめりは別物です。",
    "資料が揃えば話は早いです。揃わないと私の放課後が消えます。",
)


@dataclass(frozen=True)
class HumorCandidate:
    risk: str
    industry: str
    asset: str
    tag: str
    comment: str
    current_topic: str = ""
    persona: str = "curious_sardonic_high_school_girl"


def _pick(options: tuple[str, ...], seed: str, offset: int = 0) -> str:
    digest = hashlib.sha256(f"{seed}:{offset}".encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _risk_or_default(risk: str) -> str:
    return risk if risk in RISKS else "中リスク"


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text.endswith(("。", "！", "？", ".", "!", "?")) else text + "。"


def generate_humor_candidates(
    *,
    industry: str = "全業種",
    asset: str = "その他",
    risk: str = "中リスク",
    current_topic: str = "",
    count: int = 12,
) -> list[HumorCandidate]:
    """Generate humor candidates matching the requested tone.

    The tone is mildly sardonic, curious, and school-girl-like without relying
    on sexualized or copyrighted character imitation.
    """
    risk = _risk_or_default(risk)
    industry_terms = INDUSTRY_METAPHORS.get(industry, INDUSTRY_METAPHORS["全業種"])
    asset_terms = ASSET_METAPHORS.get(asset, ASSET_METAPHORS["その他"])
    tone = RISK_TONE[risk]
    candidates: list[HumorCandidate] = []

    for i in range(max(0, count)):
        seed = f"{industry}|{asset}|{risk}|{current_topic}|{i}"
        industry_word = _pick(industry_terms, seed, 1)
        asset_word = _pick(asset_terms, seed, 2)
        adjective = _pick(tone["adjective"], seed, 3)
        caution = _pick(tone["caution"], seed, 4)
        curious = _pick(CURIOUS_HS_FRAMES, seed, 5)
        sardonic = _pick(SARDONIC_FRAMES, seed, 6)

        if current_topic and i % 3 == 0:
            opener = _pick(CURRENT_TOPIC_FRAMES, seed, 7).format(topic=current_topic)
        elif i % 3 == 1:
            opener = curious
        else:
            opener = sardonic

        opener = _sentence(opener)
        caution = _sentence(caution)
        adjective = adjective.rstrip("。")

        patterns = (
            f"{opener} {industry_word}と{asset_word}を見る限り、{adjective}。{caution}",
            f"{industry_word}は悪くありません。問題は{asset_word}と返済計画の相性で、{adjective}から、{caution}",
            f"{opener} {asset_word}だけに目を奪われず、{industry_word}の安定感も見たいです。{caution}",
            f"{industry_word}の話は面白いです。でも審査では{asset_word}と資金繰りが主役です。{caution}",
        )
        comment = _sentence(_pick(patterns, seed, 8))
        candidates.append(
            HumorCandidate(
                risk=risk,
                industry=industry,
                asset=asset,
                tag=f"{industry}:{asset}:{risk}",
                comment=comment,
                current_topic=current_topic,
            )
        )

    return candidates


def candidates_to_markdown(candidates: list[HumorCandidate], title: str = "ユーモア候補") -> str:
    today = _dt.date.today().isoformat()
    lines = [
        "---",
        f"created: {today}",
        "source: humor_generator.py",
        "project: tune_lease_55",
        "tags:",
        "  - humor",
        "  - generated",
        "---",
        "",
        f"# {title}",
        "",
        "審査本文は真面目に保ち、最後の1文だけ柔らかくするための候補。",
        "",
    ]
    for c in candidates:
        topic = f" / 時事: {c.current_topic}" if c.current_topic else ""
        lines.append(f"- **{c.risk} / {c.industry} / {c.asset}{topic}**: {c.comment}")
    lines.append("")
    return "\n".join(lines)


def candidates_to_json(candidates: list[HumorCandidate]) -> str:
    return json.dumps({"comments": [asdict(c) for c in candidates]}, ensure_ascii=False, indent=2)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate lease screening humor candidates.")
    parser.add_argument("--industry", default="全業種", help="業種。例: 建設業, 製造業, 医療・福祉")
    parser.add_argument("--asset", default="その他", help="物件。例: 車両, 建設機械, IT・OA機器")
    parser.add_argument("--risk", default="中リスク", choices=RISKS, help="リスク帯")
    parser.add_argument("--current-topic", default="", help="任意の時事ネタ。例: 金利上昇, 人手不足, EV補助金")
    parser.add_argument("--count", type=int, default=12, help="生成件数")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="出力形式")
    parser.add_argument("--out", default="", help="出力先ファイル。未指定なら標準出力")
    parser.add_argument("--title", default="ユーモア候補", help="Markdownタイトル")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates = generate_humor_candidates(
        industry=args.industry,
        asset=args.asset,
        risk=args.risk,
        current_topic=args.current_topic,
        count=args.count,
    )
    text = candidates_to_json(candidates) if args.format == "json" else candidates_to_markdown(candidates, args.title)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
