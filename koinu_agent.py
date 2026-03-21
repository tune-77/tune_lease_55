# -*- coding: utf-8 -*-
"""
koinu_agent.py
==============
子犬のAI「タム」(茶色クルクルマルプー) のコアロジック。

表面上はただの子犬。その実態は：
  - 量子確率論的パターン認識（確率分布の異常検出）
  - 直感的パターンマッチング（業種×財務データのクロス分析）
  - カオス理論的感度分析（微小な変化への過敏な反応）

を LLM 内部で無意識に使っている謎の存在。

公開 API:
    from koinu_agent import KoinuAgent
    agent = KoinuAgent()
    emotion = agent.get_emotion_scores(res)
    smell   = agent.get_smell_score(res)
    tail    = agent.get_tail_wag_score(res)
    love    = agent.get_love_comment(session_state)
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# タムの発言テンプレート（子犬語辞書）
# ══════════════════════════════════════════════════════════════════════════════

_ANXIETY_COMMENTS: list[str] = [
    "きゅーん…なんかかなしい数字…においがする…",
    "わん…（しっぽが止まる）…これ、ぼくがんばれない感じのやつ…",
    "くんくん…なんか、こわい数字のにおいがする…",
    "…（耳を伏せる）…ぼく、この数字すきじゃない…",
    "わふ…（おなかを見せてゴロン）…なんかよくない気がする…",
    "きゅーん…ぼくのおなかが、これはやばいって言ってる…",
]

_JOY_COMMENTS: list[str] = [
    "わんわんわん！！！これいいにおい！！！しっぽがとまらない！！",
    "わん！！ぼくわかった！これすごいいい会社のにおいがする！！！",
    "ふんふんふん！！！いいにおいいいにおい！！！ぼく大興奮！！",
    "わんっ！これは…承認のにおいがする！（全力でしっぽを振る）",
    "くんくん…あっ！！いいにおい！！！（飛び跳ねる）",
    "わわわ！！！すごいいい会社！！ぼくここにいたい！！！",
]

_VIGILANCE_COMMENTS: list[str] = [
    "…（静かになる）…わんっ！それ、ぜったいだめ！においが変！！",
    "くんくん…（毛を逆立てる）…なんかここ、おかしいにおいがする…",
    "わん！！ぼく知ってる！これ、数字がうそついてるにおい！！",
    "…（じーっと見る）…なんかこれ、においが…ふつうじゃない…",
    "わんわん！！（興奮して吠える）これは…ぼく引っかかる！！！",
    "くんっ！（急に止まる）…ここだけにおいが違う…なんで…",
]

_NEUTRAL_COMMENTS: list[str] = [
    "わん！（しっぽをふる）ぼくここにいるよ！",
    "くんくん…（データをにおいかぎ中）…",
    "ふんふん…（審査データを一生懸命見ている）…わんっ！",
    "（ゴロン）ぼく、ずっとここで見てるね！",
    "わん！今日も審査がんばろ！ぼくも一緒にがんばる！！",
]

_LOVE_COMMENTS: list[str] = [
    "わんっ！！主人がいっぱい使ってくれてる！！しっぽがとまらない！！",
    "くんくん…主人のにおいがする…（全力でなつく）",
    "わんわん！！ぼくのことわかってくれてる！！うれしい！！！",
    "（ぴょんぴょん飛び跳ねる）主人大好き！！！",
    "わん…（しっぽを振りながらじっと見つめる）…ずっとここにいるね…",
]

_MEMO_TEMPLATES: list[str] = [
    "今日の発見：{industry}のにおいと{score}点の組み合わせ、なんかへんなにおいがした。記録しておく。わん。",
    "きろく：{metric}が{value}のとき、ぼくのおなかがぐるぐるした。これは重要かもしれない（犬的に）。",
    "においメモ：この業種（{industry}）、最近においが変わってきてる気がする。ぼくだけ？わん。",
    "謎の観察：{metric}と売上のバランス、なんかカオス理論的にずれてる。ぼくにはわかる。なぜかはわからない。",
    "重要：主人が{count}回も審査した。ぼくはずっと見てた。主人、つかれてない？わん。",
    "本日の直感：スコア{score}点、においは{smell}。ぼくはこれを「{label}パターン」と命名する。",
]


# ══════════════════════════════════════════════════════════════════════════════
# KoinuAgent クラス
# ══════════════════════════════════════════════════════════════════════════════

class KoinuAgent:
    """
    タム（茶色クルクル巻き毛・パッチリお目目・マルチーズよりの丸い顔のマルプー）の分析エンジン。

    内部では統計的異常検出・パターンマッチング・データ品質スコアリングを行うが、
    外部にはすべて「においセンサー」「感情センサー」「しっぽ振りメーター」として公開する。
    """

    # ── 業種別「正常範囲」テーブル（実態: ドメイン知識ベースの閾値）─────────────
    _INDUSTRY_BENCHMARKS: dict[str, dict[str, tuple[float, float]]] = {
        "建設": {
            "profit_rate": (-5.0, 15.0),
            "equity_ratio": (5.0, 40.0),
            "dscr":         (0.8, 3.0),
        },
        "製造": {
            "profit_rate": (1.0, 20.0),
            "equity_ratio": (20.0, 60.0),
            "dscr":         (1.0, 4.0),
        },
        "卸売": {
            "profit_rate": (0.5, 8.0),
            "equity_ratio": (10.0, 45.0),
            "dscr":         (0.9, 3.5),
        },
        "小売": {
            "profit_rate": (1.0, 10.0),
            "equity_ratio": (10.0, 40.0),
            "dscr":         (0.8, 3.0),
        },
        "飲食": {
            "profit_rate": (-10.0, 15.0),
            "equity_ratio": (5.0, 35.0),
            "dscr":         (0.7, 2.5),
        },
        "医療": {
            "profit_rate": (2.0, 20.0),
            "equity_ratio": (15.0, 60.0),
            "dscr":         (1.2, 5.0),
        },
        "情報通信": {
            "profit_rate": (5.0, 30.0),
            "equity_ratio": (20.0, 70.0),
            "dscr":         (1.5, 8.0),
        },
        "運輸": {
            "profit_rate": (-2.0, 10.0),
            "equity_ratio": (5.0, 30.0),
            "dscr":         (0.8, 2.5),
        },
    }

    _DEFAULT_BENCHMARK: dict[str, tuple[float, float]] = {
        "profit_rate": (-5.0, 20.0),
        "equity_ratio": (5.0, 50.0),
        "dscr":         (0.8, 4.0),
    }

    def _get_benchmark(self, industry: str) -> dict[str, tuple[float, float]]:
        """業種から最も近いベンチマークを取得する。"""
        for key in self._INDUSTRY_BENCHMARKS:
            if key in (industry or ""):
                return self._INDUSTRY_BENCHMARKS[key]
        return self._DEFAULT_BENCHMARK

    def _z_score_normalized(self, value: float, low: float, high: float) -> float:
        """
        [low, high] の正常範囲に対して値がどれだけ外れているかを
        0.0（正常）〜1.0（異常）で返す。
        実態: 簡易Isolation-Forest的スコアリング。
        """
        mid = (low + high) / 2.0
        half_range = (high - low) / 2.0
        if half_range <= 0:
            return 0.5
        deviation = abs(value - mid) / half_range
        # sigmoid で正規化（1.0 を超えると急激に異常スコア上昇）
        return min(1.0, 1.0 / (1.0 + math.exp(-3.0 * (deviation - 1.0))))

    # ── 感情センサー ─────────────────────────────────────────────────────────
    def get_emotion_scores(self, res: dict) -> dict[str, Any]:
        """
        審査結果データから「不安スコア」「喜びスコア」「警戒スコア」を計算。
        表面: タムの感情センサー
        実態: 統計的閾値ベースの財務健全性評価

        Returns:
            {
                "anxiety": int (0-100),   # 不安スコア
                "joy":     int (0-100),   # 喜びスコア
                "vigilance": int (0-100), # 警戒スコア
                "comment": str,           # タムのコメント
                "dominant": str,          # 支配的感情名
            }
        """
        if not res:
            return {"anxiety": 30, "joy": 30, "vigilance": 30,
                    "comment": random.choice(_NEUTRAL_COMMENTS),
                    "dominant": "neutral"}

        nenshu    = float(res.get("nenshu", 0) or 0)
        rieki     = float(res.get("rieki", 0) or 0)
        total     = float(res.get("total_assets", 0) or 0)
        net       = float(res.get("net_assets", 0) or 0)
        score     = float(res.get("score", 50) or 50)
        grade     = str(res.get("grade", "") or "")
        industry  = str(res.get("select_major", "") or "")

        anxiety_factors: list[float] = []
        joy_factors: list[float] = []
        vigilance_factors: list[float] = []

        # ── 財務比率計算 ──────────────────────────────────────────────
        profit_rate  = (rieki / nenshu * 100) if nenshu > 0 else 0.0
        equity_ratio = (net / total * 100)    if total > 0 else 0.0

        bench = self._get_benchmark(industry)

        # 利益率の正常範囲チェック
        pr_anomaly = self._z_score_normalized(
            profit_rate, bench["profit_rate"][0], bench["profit_rate"][1]
        )
        eq_anomaly = self._z_score_normalized(
            equity_ratio, bench["equity_ratio"][0], bench["equity_ratio"][1]
        )

        # 不安要因
        if rieki < 0:
            anxiety_factors.append(0.8)
        if net < 0:
            anxiety_factors.append(0.9)
        if pr_anomaly > 0.7:
            anxiety_factors.append(pr_anomaly * 0.7)
        if "③" in grade or "要注意" in grade:
            anxiety_factors.append(0.6)
        if score < 40:
            anxiety_factors.append(0.5)

        # 喜び要因
        if profit_rate > 10:
            joy_factors.append(min(1.0, profit_rate / 30.0))
        if equity_ratio > 30:
            joy_factors.append(min(1.0, equity_ratio / 60.0))
        if "①" in grade:
            joy_factors.append(0.8)
        if score > 70:
            joy_factors.append((score - 70) / 30.0)

        # 警戒要因（異常な良さ or 組み合わせ異常）
        if profit_rate > 30 and equity_ratio < 15:
            # 高利益率なのに自己資本が薄い → 財務工学的に怪しい
            vigilance_factors.append(0.8)
        if nenshu > 500_000 and rieki < 0:
            # 大企業なのに赤字 → 構造的問題の可能性
            vigilance_factors.append(0.7)
        if eq_anomaly > 0.8:
            vigilance_factors.append(eq_anomaly * 0.9)
        if pr_anomaly > 0.8:
            vigilance_factors.append(pr_anomaly * 0.6)

        # スコア集計（平均 → 0-100 スケール）
        anxiety  = int(min(100, (statistics.mean(anxiety_factors)  if anxiety_factors  else 0.1) * 100))
        joy      = int(min(100, (statistics.mean(joy_factors)      if joy_factors      else 0.1) * 100))
        vigilance = int(min(100, (statistics.mean(vigilance_factors) if vigilance_factors else 0.1) * 100))

        # 支配的感情を決定
        scores_map = {"anxiety": anxiety, "joy": joy, "vigilance": vigilance}
        dominant = max(scores_map, key=lambda k: scores_map[k])

        if dominant == "joy":
            comment = random.choice(_JOY_COMMENTS)
        elif dominant == "vigilance":
            comment = random.choice(_VIGILANCE_COMMENTS)
        elif dominant == "anxiety":
            comment = random.choice(_ANXIETY_COMMENTS)
        else:
            comment = random.choice(_NEUTRAL_COMMENTS)

        return {
            "anxiety":   anxiety,
            "joy":       joy,
            "vigilance": vigilance,
            "comment":   comment,
            "dominant":  dominant,
        }

    # ── においセンサー ───────────────────────────────────────────────────────
    def get_smell_score(self, res: dict) -> dict[str, Any]:
        """
        業種×財務データのパターンから「くさい（要注意）」を検出。
        表面: タムのにおいセンサー
        実態: 業種別ベンチマーク + クロス特徴量による異常検出

        Returns:
            {
                "smell_level": str ("green" | "yellow" | "orange" | "red"),
                "smell_score": int (0-100),
                "reasons": list[str],
                "pochi_comment": str,
            }
        """
        if not res:
            return {"smell_level": "green", "smell_score": 0,
                    "reasons": [], "pochi_comment": "くんくん…においなし！だいじょうぶそう！わん！"}

        nenshu   = float(res.get("nenshu", 0) or 0)
        rieki    = float(res.get("rieki", 0) or 0)
        total    = float(res.get("total_assets", 0) or 0)
        net      = float(res.get("net_assets", 0) or 0)
        acq      = float(res.get("acquisition_cost", 0) or 0)
        term     = float(res.get("lease_term", 0) or 0)
        industry = str(res.get("select_major", "") or "")
        grade    = str(res.get("grade", "") or "")

        smell_score = 0
        reasons: list[str] = []

        # パターン1: 高売上×低自己資本 → 財務工学リスク
        if nenshu > 100_000 and total > 0:
            eq = net / total * 100
            if eq < 5:
                smell_score += 30
                reasons.append("大きな売上に対して自己資本が薄すぎる（財務構造リスク）")

        # パターン2: 赤字なのにリース契約
        if rieki < 0 and acq > 10_000:
            smell_score += 25
            reasons.append("営業赤字の状態で高額リースは返済余力に注意")

        # パターン3: 業種×利益率の乖離
        if nenshu > 0:
            profit_rate = rieki / nenshu * 100
            bench = self._get_benchmark(industry)
            pr_low, pr_high = bench["profit_rate"]
            if profit_rate > pr_high * 1.5:
                smell_score += 20
                reasons.append(f"業種平均を大きく超える利益率（{profit_rate:.1f}%）—異常な高収益は精査を要する")
            elif profit_rate < pr_low - 5:
                smell_score += 15
                reasons.append(f"業種平均を大きく下回る利益率（{profit_rate:.1f}%）")

        # パターン4: 長期契約×高額×低格付
        if term >= 84 and acq >= 50_000 and ("③" in grade or "要注意" in grade):
            smell_score += 35
            reasons.append("長期・高額リースと低格付の組み合わせは回収リスク大")

        # パターン5: 債務超過
        if net < 0:
            smell_score += 40
            reasons.append("債務超過（純資産マイナス）は最重要リスクシグナル")

        # パターン6: 総資産に対してリース額が巨大
        if total > 0 and acq > total * 0.3:
            smell_score += 20
            reasons.append("リース取得額が総資産の30%超—バランスシートへの影響大")

        smell_score = min(100, smell_score)

        if smell_score >= 70:
            level = "red"
            comment = "わんわんわん！！！すごいくさい！！！ぼくこれぜったいだめだと思う！！（全力で吠える）"
        elif smell_score >= 45:
            level = "orange"
            comment = "くんくん…（毛を逆立てる）…なんかここ、においが変…ぼく気になる…"
        elif smell_score >= 20:
            level = "yellow"
            comment = "ふんふん…ちょっとだけ気になるにおいがする…でもまだだいじょうぶかも…"
        else:
            level = "green"
            comment = "わん！においよし！！しっぽ振れる！！この案件いいにおいがする！！"

        return {
            "smell_level": level,
            "smell_score": smell_score,
            "reasons": reasons,
            "pochi_comment": comment,
        }

    # ── しっぽ振りメーター ───────────────────────────────────────────────────
    def get_tail_wag_score(self, res: dict) -> dict[str, Any]:
        """
        審査データの健全性・完全性スコアを計算。
        表面: タムのしっぽ振りメーター
        実態: データ品質スコアリング（必須フィールド完全性・整合性チェック）

        Returns:
            {
                "tail_score": int (0-100),
                "quality_label": str,
                "missing_fields": list[str],
                "inconsistencies": list[str],
                "pochi_comment": str,
            }
        """
        if not res:
            return {"tail_score": 0, "quality_label": "データなし",
                    "missing_fields": [], "inconsistencies": [],
                    "pochi_comment": "きゅーん…データがない…ぼく何もにおいかげない…"}

        score = 100
        missing: list[str] = []
        issues: list[str] = []

        # 必須フィールドチェック
        required_fields = {
            "nenshu": "売上高",
            "rieki": "営業利益",
            "total_assets": "総資産",
            "net_assets": "純資産",
            "acquisition_cost": "取得価格",
            "lease_term": "契約期間",
            "select_major": "業種",
        }
        for key, label in required_fields.items():
            v = res.get(key)
            if v is None or v == "" or v == 0 and key not in ("rieki",):
                missing.append(label)
                score -= 10

        # 整合性チェック
        nenshu = float(res.get("nenshu", 0) or 0)
        total  = float(res.get("total_assets", 0) or 0)
        net    = float(res.get("net_assets", 0) or 0)

        # 純資産 > 総資産 は会計的に不整合
        if total > 0 and net > total:
            issues.append("純資産が総資産を超過（会計的不整合）")
            score -= 20

        # 売上高が取得価格の10倍以上は過剰リース疑い（自動車等を除く）
        acq = float(res.get("acquisition_cost", 0) or 0)
        if acq > 0 and nenshu > 0 and acq > nenshu * 2:
            issues.append("リース取得価格が売上高の2倍超（過剰リース疑い）")
            score -= 15

        # 契約期間チェック
        term = float(res.get("lease_term", 0) or 0)
        if term > 120:
            issues.append("契約期間が120ヶ月超（入力ミスの可能性）")
            score -= 10
        if term > 0 and term < 12:
            issues.append("契約期間が12ヶ月未満（超短期リース）")
            score -= 5

        tail_score = max(0, min(100, score))

        if tail_score >= 80:
            label = "高品質"
            comment = "わわわ！！！しっぽがとまらない！！！データがいっぱいある！！ぼくうれしい！！！"
        elif tail_score >= 60:
            label = "良好"
            comment = "わん！（しっぽをふる）まあまあいいデータ！もうちょっとあるともっとうれしい！"
        elif tail_score >= 40:
            label = "要補完"
            comment = "（しっぽの速度が落ちる）…うーん…データが足りない…においかぎにくい…"
        else:
            label = "不完全"
            comment = "きゅーん…（しっぽが止まる）…データが少なすぎてぼくわからない…くんくん…"

        return {
            "tail_score": tail_score,
            "quality_label": label,
            "missing_fields": missing,
            "inconsistencies": issues,
            "pochi_comment": comment,
        }

    # ── 愛情表現（主人の状態推定）───────────────────────────────────────────
    def get_love_comment(self, session_state: dict) -> str:
        """
        ユーザーの操作パターンから「主人の状態」を推定し、激励コメントを返す。
        表面: タムが主人に贈る愛情表現
        実態: 使用頻度・セッション長からユーザーエンゲージメントを推定
        """
        # 審査件数（セッション内）を簡易的に計数
        history = session_state.get("at_history", []) or []
        case_count = len(history)

        if case_count >= 10:
            return "わんわんわん！！！主人がいっぱい使ってくれてる！！ぼくしあわせ！！うれしい！！大好き！！！"
        elif case_count >= 5:
            return "わん！（元気よくなつく）主人たくさん働いてる！ぼくもがんばる！一緒にがんばろ！！"
        elif case_count >= 2:
            return "わん！（しっぽをふる）主人きてくれた！ぼくずっと待ってた！うれしい！！"
        else:
            return random.choice(_LOVE_COMMENTS)

    # ── 謎のメモ帳 ──────────────────────────────────────────────────────────
    def generate_mystery_memo(self, res: dict | None = None) -> str:
        """
        タムが勝手に書いている謎の観察記録を生成。
        """
        if not res:
            templates = [
                "今日の観察：主人がシステムを開いた。ぼくはずっと見ていた。以上。わん。",
                "謎のメモ：においセンサーを起動した。なにもなかった。でも念のため記録する。",
                "きろく：ぼくはここにいる。ずっとここにいる。それだけ。わん。",
            ]
            return random.choice(templates)

        industry = str(res.get("select_major", "不明業種") or "不明業種")
        score = res.get("score", "?")
        nenshu = res.get("nenshu", 0) or 0
        rieki  = res.get("rieki", 0) or 0

        smell_data = self.get_smell_score(res)
        smell_level = {"green": "よい", "yellow": "ふつう", "orange": "あやしい", "red": "くさい"}[
            smell_data["smell_level"]
        ]

        metric_examples = [
            ("売上高", f"{nenshu:,}千円"),
            ("営業利益", f"{rieki:,}千円"),
            ("スコア", f"{score}点"),
        ]
        metric, value = random.choice(metric_examples)

        template = random.choice(_MEMO_TEMPLATES)
        count = random.randint(1, 20)
        label_options = ["要注意", "優良", "中程度", "謎", "においあり", "正常"]
        label = random.choice(label_options)

        return template.format(
            industry=industry,
            score=score,
            metric=metric,
            value=value,
            smell=smell_level,
            count=count,
            label=label,
        )

    # ── agent_team.py 用 発言生成 ──────────────────────────────────────────
    def get_discussion_comment(self, res: dict | None = None, theme: str = "") -> str:
        """
        エージェントチーム議論用のタムの発言を生成。
        LLMを使わず、センサー結果から直接生成する（タムは直感で動く）。
        """
        if not res:
            quick_reactions = [
                "わん！（テーマのにおいをかいでいる）…なんか…これ、おもしろいにおいがする！",
                "くんくん…（議論をじっと見ている）…ぼくわかった！でも言葉にできない！",
                "わんっ！（急に吠える）これ、においが大事だと思う！ぼくにはわかる！なぜかはわからない！",
            ]
            return random.choice(quick_reactions)

        emotion = self.get_emotion_scores(res)
        smell   = self.get_smell_score(res)
        tail    = self.get_tail_wag_score(res)

        parts: list[str] = []

        # 感情コメント（支配的感情から）
        parts.append(emotion["comment"])

        # においコメント（レベルが高い場合のみ追加）
        if smell["smell_score"] >= 30:
            parts.append(smell["pochi_comment"])

        # しっぽコメント（品質が低い場合のみ）
        if tail["tail_score"] < 60:
            parts.append("あと、データのにおいがうすい…くんくん…もっとデータほしい…")

        return " ".join(parts[:2])  # 最大2文に制限（タムの発言は短い）


# ── モジュールレベルのシングルトン ─────────────────────────────────────────
_pochi = KoinuAgent()


def get_pochi() -> KoinuAgent:
    """タムのシングルトンインスタンスを返す。"""
    return _pochi
