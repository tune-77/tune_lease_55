# -*- coding: utf-8 -*-
"""
slack_screening.py
==================
Slack フロントエンド版 リース審査入力フロー。

slack_bot.py から呼び出される。
チャンネルごとに状態を保持し、対話形式でデータ収集 → スコアリングを実行する。

使い方:
    ScreeningSession.get(channel_id)  # セッション取得/作成
    session.feed(text)                # ユーザー入力を処理
    → 返答テキストを返す
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ══════════════════════════════════════════════════════════════════════════════
# 業種マスター
# ══════════════════════════════════════════════════════════════════════════════

INDUSTRY_OPTIONS = [
    "製造業",
    "建設業",
    "サービス業",
    "卸売業",
    "小売業",
]

_INDUSTRY_ALIASES: dict[str, str] = {}
for _i, _name in enumerate(INDUSTRY_OPTIONS, 1):
    _INDUSTRY_ALIASES[str(_i)] = _name
    _INDUSTRY_ALIASES[_name] = _name
    # 部分一致
    _INDUSTRY_ALIASES[_name[:2]] = _name


def _parse_industry(text: str) -> Optional[str]:
    """業種テキスト/番号をモデル用業種に変換。"""
    t = text.strip()
    if t in _INDUSTRY_ALIASES:
        return _INDUSTRY_ALIASES[t]
    # 部分一致
    for k, v in _INDUSTRY_ALIASES.items():
        if k in t:
            return v
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 質問ステップ定義
# ══════════════════════════════════════════════════════════════════════════════

INDUSTRY_PROMPT = (
    "業種を選択してください（番号または名称）:\n"
    + "\n".join(f"{i}. {n}" for i, n in enumerate(INDUSTRY_OPTIONS, 1))
)

STEPS = [
    ("company_name",       "審査を開始します。\n\n*会社名* を入力してください。\n例: `株式会社○○`\n\n（途中で中止するには `キャンセル` と入力）"),
    ("industry",           INDUSTRY_PROMPT),
    ("revenue",            "💴 *売上高* を入力してください（単位: 万円）\n例: `50000`"),
    ("total_assets",       "📊 *総資産* を入力してください（万円）"),
    ("equity",             "🏦 *純資産* を入力してください（万円）"),
    ("operating_profit",   "📈 *営業利益* を入力してください（万円、赤字は `-5000` のように入力）"),
    ("net_income",         "💰 *当期純利益* を入力してください（万円）"),
    ("machinery_equipment","🔧 *機械設備* を入力してください（万円、不明なら `0`）"),
    ("other_fixed_assets", "🏗️ *その他固定資産* を入力してください（万円、不明なら `0`）"),
    ("depreciation",       "📉 *減価償却費* を入力してください（万円）"),
    ("rent_expense",       "🏢 *賃借料* を入力してください（万円）"),
    ("lease_amount",       "💳 *リース申込額* を入力してください（万円）\n例: `3000`"),
    ("lease_term",         "📅 *リース期間* を入力してください（ヶ月）\n例: `36`（3年）/ `60`（5年）"),
]

FIELD_NAMES = [s[0] for s in STEPS]
STEP_COUNT = len(STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# 数値パーサー
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(text: str) -> Optional[float]:
    """テキストから数値を抽出（カンマ・単位除去）。"""
    t = re.sub(r"[,，　 ]", "", text.strip())
    t = re.sub(r"万円.*|円.*|yen.*", "", t, flags=re.IGNORECASE)
    try:
        return float(t)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# セッション状態マシン
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreeningSession:
    channel_id: str
    step: int = 0            # 現在の質問インデックス
    data: dict = field(default_factory=dict)
    active: bool = True

    # チャンネルごとのセッション辞書（クラス変数）
    _sessions: dict[str, "ScreeningSession"] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        pass

    # ── クラスメソッド ──────────────────────────────────────────────────────

    _registry: dict[str, "ScreeningSession"] = {}

    @classmethod
    def get(cls, channel_id: str) -> Optional["ScreeningSession"]:
        return cls._registry.get(channel_id)

    @classmethod
    def start(cls, channel_id: str) -> "ScreeningSession":
        s = cls(channel_id=channel_id, step=0, data={}, active=True)
        cls._registry[channel_id] = s
        return s

    @classmethod
    def cancel(cls, channel_id: str) -> None:
        cls._registry.pop(channel_id, None)

    # ── インスタンスメソッド ──────────────────────────────────────────────

    def current_prompt(self) -> str:
        """現在のステップの質問文を返す。"""
        if self.step < STEP_COUNT:
            return STEPS[self.step][1]
        return ""

    def feed(self, text: str) -> str:
        """
        ユーザー入力を処理し、次の質問または結果を返す。
        Returns: Slackに送るテキスト
        """
        t = text.strip()

        # キャンセル
        if t in ("キャンセル", "cancel", "中止", "やめる", "quit"):
            ScreeningSession.cancel(self.channel_id)
            return "審査入力を中止しました。"

        # やり直し
        if t in ("やり直し", "restart", "最初から"):
            self.step = 0
            self.data = {}
            return "最初からやり直します。\n\n" + STEPS[0][1]

        field_name = FIELD_NAMES[self.step]

        # ── 業種 ──
        if field_name == "industry":
            industry = _parse_industry(t)
            if industry is None:
                return (
                    f"⚠️ 業種が認識できませんでした。番号（1〜5）または名称を入力してください。\n\n{INDUSTRY_PROMPT}"
                )
            self.data["industry"] = industry

        # ── 会社名 ──
        elif field_name == "company_name":
            if not t:
                return "会社名を入力してください。"
            self.data["company_name"] = t

        # ── リース期間（整数月）──
        elif field_name == "lease_term":
            n = _parse_number(t)
            if n is None or n <= 0:
                return "⚠️ 有効な月数を入力してください。例: `36`"
            self.data["lease_term"] = int(n)

        # ── その他数値 ──
        else:
            n = _parse_number(t)
            if n is None:
                return f"⚠️ 数値を入力してください（単位: 万円）。例: `5000`\n\n{STEPS[self.step][1]}"
            self.data[field_name] = n

        # 次のステップへ
        self.step += 1

        if self.step < STEP_COUNT:
            # 進捗バー
            pct = int(self.step / STEP_COUNT * 100)
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            progress = f"進捗: [{bar}] {pct}% ({self.step}/{STEP_COUNT})\n\n"
            return progress + STEPS[self.step][1]
        else:
            # 全データ収集完了 → スコアリング
            return self._run_scoring()

    def _run_scoring(self) -> str:
        """スコアリングを実行して結果テキストを返す。"""
        ScreeningSession.cancel(self.channel_id)  # セッション終了

        d = self.data
        company = d.get("company_name", "（不明）")
        industry = d.get("industry", "サービス業")

        # 万円 → 円単位に変換
        def _to_en(key: str) -> float:
            return (d.get(key) or 0.0) * 10000

        revenue             = _to_en("revenue")
        total_assets        = _to_en("total_assets")
        equity              = _to_en("equity")
        operating_profit    = _to_en("operating_profit")
        net_income          = _to_en("net_income")
        machinery_equipment = _to_en("machinery_equipment")
        other_fixed_assets  = _to_en("other_fixed_assets")
        depreciation        = _to_en("depreciation")
        rent_expense        = _to_en("rent_expense")
        lease_amount_man    = d.get("lease_amount", 0)  # 万円のまま表示用
        lease_term          = d.get("lease_term", 36)

        # スコアリング呼び出し
        result = None
        try:
            from scoring.predict_one import predict_one
            result = predict_one(
                revenue=revenue,
                total_assets=total_assets,
                equity=equity,
                operating_profit=operating_profit,
                net_income=net_income,
                machinery_equipment=machinery_equipment,
                other_fixed_assets=other_fixed_assets,
                depreciation=depreciation,
                rent_expense=rent_expense,
                industry=industry,
            )
        except Exception as e:
            result = None
            err_msg = str(e)

        # ── 結果フォーマット ────────────────────────────────────────────────
        lines = [
            "=" * 48,
            f"📋 *審査結果 — {company}*",
            "=" * 48,
            f"• 業種: {industry}",
            f"• リース申込額: {lease_amount_man:,.0f} 万円 / {lease_term} ヶ月",
            "",
        ]

        if result is None:
            lines += [
                "⚠️ AIモデルによる予測が実行できませんでした。",
                "（モデルファイル未配置 or 入力データ不足）",
                "",
                "*簡易財務チェック:*",
            ]
            # 簡易チェック（モデルなしでも出せる情報）
            if total_assets > 0:
                eq_ratio = equity / total_assets * 100
                lines.append(f"• 自己資本比率: {eq_ratio:.1f}%")
                if eq_ratio < 10:
                    lines.append("  → ⚠️ 自己資本比率が低い（10%未満）")
                elif eq_ratio >= 30:
                    lines.append("  → ✅ 自己資本比率は良好（30%以上）")
            if revenue > 0:
                op_ratio = operating_profit / revenue * 100
                lines.append(f"• 営業利益率: {op_ratio:.1f}%")
                if op_ratio < 0:
                    lines.append("  → ⚠️ 営業赤字")
                elif op_ratio >= 5:
                    lines.append("  → ✅ 営業利益率良好")
        else:
            hybrid = result["hybrid_prob"]
            ai_prob = result["ai_prob"]
            legacy  = result["legacy_prob"]
            decision = result["decision"]
            top5 = result.get("top5_reasons", [])

            # スコアを 0–100 に変換（hybrid_prob が低いほど優良）
            score = round((1 - hybrid) * 100, 1)

            decision_emoji = "✅" if decision == "承認" else "❌"
            risk_level = (
                "低リスク" if hybrid < 0.3 else
                "中リスク" if hybrid < 0.5 else
                "高リスク" if hybrid < 0.7 else
                "要注意"
            )

            lines += [
                f"*判定: {decision_emoji} {decision}*",
                f"スコア: *{score}点* / 100点",
                f"リスクレベル: {risk_level}",
                "",
                "*確率詳細:*",
                f"• AIモデル確率:   {ai_prob:.1%}",
                f"• 業種別ロジスティック: {legacy:.1%}",
                f"• ハイブリッド確率:    {hybrid:.1%}（低いほど良好）",
            ]

            if top5:
                lines += ["", "*主要判定要因（Top5）:*"]
                for reason in top5:
                    lines.append(f"• {reason}")

            # 財務サマリー
            lines += ["", "*財務サマリー:*"]
            if total_assets > 0:
                eq_ratio = equity / total_assets * 100
                lines.append(f"• 自己資本比率: {eq_ratio:.1f}%")
            if revenue > 0:
                op_ratio = operating_profit / revenue * 100
                lines.append(f"• 営業利益率: {op_ratio:.1f}%")
            if total_assets > 0 and revenue > 0:
                roa = net_income / total_assets * 100
                lines.append(f"• ROA: {roa:.1f}%")

        lines += [
            "",
            "再審査は `審査開始` と入力してください。",
            "=" * 48,
        ]

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 外部インターフェース
# ══════════════════════════════════════════════════════════════════════════════

def is_screening_active(channel_id: str) -> bool:
    """チャンネルで審査セッションが進行中かどうか。"""
    return channel_id in ScreeningSession._registry


def handle_screening_message(channel_id: str, text: str) -> Optional[str]:
    """
    審査フロー中のメッセージを処理する。
    セッションがなければ None を返す（他のコマンドへ委譲）。

    Returns:
        返答テキスト、またはセッションなしの場合 None
    """
    session = ScreeningSession.get(channel_id)
    if session is None:
        return None
    return session.feed(text)


def start_screening(channel_id: str) -> str:
    """審査セッションを開始し、最初の質問を返す。"""
    session = ScreeningSession.start(channel_id)
    return session.current_prompt()
