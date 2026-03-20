# -*- coding: utf-8 -*-
"""
slack_screening.py
==================
Slack版 リースくん — テキスト対話式リース審査ウィザード。

slack_bot.py から呼び出される。
チャンネルごとにセッションを保持し、10ステップで審査データを収集 → スコアリング。

使い方:
    start_screening(channel_id)           → 最初のメッセージを返す
    handle_screening_message(ch, text)    → ユーザー入力を処理して返答を返す
    is_screening_active(channel_id)       → 審査中かチェック
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import ClassVar, Optional

# ══════════════════════════════════════════════════════════════════════════════
# 業種マスター（日本標準産業分類 大分類）
# ══════════════════════════════════════════════════════════════════════════════

INDUSTRY_MAJOR = [
    ("A", "農業，林業"),
    ("B", "漁業"),
    ("C", "鉱業，採石業，砂利採取業"),
    ("D", "建設業"),
    ("E", "製造業"),
    ("F", "電気・ガス・熱供給・水道業"),
    ("G", "情報通信業"),
    ("H", "運輸業，郵便業"),
    ("I", "卸売業，小売業"),
    ("J", "金融業，保険業"),
    ("K", "不動産業，物品賃貸業"),
    ("L", "学術研究，専門・技術サービス業"),
    ("M", "宿泊業，飲食サービス業"),
    ("N", "生活関連サービス業，娯楽業"),
    ("O", "教育，学習支援業"),
    ("P", "医療，福祉"),
    ("Q", "複合サービス事業"),
    ("R", "サービス業（他に分類されないもの）"),
    ("S", "公務"),
]

_INDUSTRY_PROMPT = (
    "🏭 *業種を選んでください*（番号または分類記号）:\n"
    + "\n".join(f"  {i+1}. [{code}] {name}" for i, (code, name) in enumerate(INDUSTRY_MAJOR))
    + "\n\n例: `4` または `D` または `建設`"
)


def _parse_industry(text: str) -> Optional[str]:
    t = text.strip()
    # 番号
    try:
        idx = int(t) - 1
        if 0 <= idx < len(INDUSTRY_MAJOR):
            code, name = INDUSTRY_MAJOR[idx]
            return f"{code} {name}"
    except ValueError:
        pass
    # 記号
    t_upper = t.upper()
    for code, name in INDUSTRY_MAJOR:
        if t_upper == code:
            return f"{code} {name}"
    # 部分一致
    for code, name in INDUSTRY_MAJOR:
        if t in name or name in t:
            return f"{code} {name}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# リース物件マスター（簡易）
# ══════════════════════════════════════════════════════════════════════════════

ASSET_LIST = [
    {"id": "vehicle",      "name": "車両・運搬具",      "score": 90},
    {"id": "machinery",    "name": "機械設備",          "score": 80},
    {"id": "it_equipment", "name": "IT機器・PCサーバー", "score": 75},
    {"id": "medical",      "name": "医療機器",          "score": 85},
    {"id": "construction", "name": "建設重機・作業機械", "score": 70},
    {"id": "food",         "name": "厨房・食品機械",    "score": 65},
    {"id": "office",       "name": "オフィス機器・複合機", "score": 70},
    {"id": "solar",        "name": "太陽光発電設備",    "score": 60},
    {"id": "other",        "name": "その他",            "score": 50},
]

_ASSET_PROMPT = (
    "🚜 *リース物件を選んでください*（番号）:\n"
    + "\n".join(f"  {i+1}. {a['name']}" for i, a in enumerate(ASSET_LIST))
)


def _parse_asset(text: str) -> Optional[dict]:
    t = text.strip()
    try:
        idx = int(t) - 1
        if 0 <= idx < len(ASSET_LIST):
            return ASSET_LIST[idx]
    except ValueError:
        pass
    for a in ASSET_LIST:
        if t in a["name"] or a["name"] in t:
            return a
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 格付マスター
# ══════════════════════════════════════════════════════════════════════════════

GRADE_OPTIONS = ["①1-3 (優良)", "②4-6 (標準)", "③要注意以下", "④無格付"]
_GRADE_PROMPT = (
    "💳 *格付を選んでください*:\n"
    "  1. ①1-3（優良）\n"
    "  2. ②4-6（標準）\n"
    "  3. ③要注意以下\n"
    "  4. ④無格付"
)


def _parse_grade(text: str) -> Optional[str]:
    t = text.strip()
    try:
        idx = int(t) - 1
        if 0 <= idx < len(GRADE_OPTIONS):
            return GRADE_OPTIONS[idx]
    except ValueError:
        pass
    for g in GRADE_OPTIONS:
        if t in g:
            return g
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 定性評価マスター
# ══════════════════════════════════════════════════════════════════════════════

QUALITATIVE_STEPS = [
    ("qual_corr_company_history", "設立・経営年数",
     ["未選択", "20年以上", "10年〜20年", "5年〜10年", "3年〜5年", "3年未満"]),
    ("qual_corr_customer_stability", "顧客安定性",
     ["未選択", "非常に安定（大口・長期）", "安定（分散良好）", "普通", "やや不安定", "不安定・依存大"]),
    ("qual_corr_repayment_history", "返済履歴",
     ["未選択", "5年以上問題なし", "3年以上問題なし", "遅延少ない", "遅延・リスケあり", "問題あり・要確認"]),
    ("qual_corr_business_future", "事業将来性",
     ["未選択", "有望（成長・ニーズ確実）", "やや有望", "普通", "やや懸念", "懸念（縮小・競争激化）"]),
    ("qual_corr_equipment_purpose", "設備目的",
     ["未選択", "収益直結・受注必須", "生産性向上・省力化", "更新・維持・法定対応", "やや不明確", "不明確・要説明"]),
    ("qual_corr_main_bank", "メイン取引銀行",
     ["未選択", "メイン先で取引良好・支援表明", "メイン先", "サブ扱い・取引あり", "取引浅い・他社メイン", "取引なし・不安"]),
]


def _qual_prompt(idx: int) -> str:
    _, label, opts = QUALITATIVE_STEPS[idx]
    lines = [f"🎯 *定性評価 [{idx+1}/6] — {label}*:"]
    for i, o in enumerate(opts):
        lines.append(f"  {i+1}. {o}")
    lines.append("\n例: `1` または `0` でスキップ（未選択）")
    return "\n".join(lines)


def _parse_qual(text: str, idx: int) -> str:
    _, _, opts = QUALITATIVE_STEPS[idx]
    t = text.strip()
    if t in ("0", "スキップ", "skip"):
        return "未選択"
    try:
        i = int(t) - 1
        if 0 <= i < len(opts):
            return opts[i]
    except ValueError:
        pass
    for o in opts:
        if t in o:
            return o
    return "未選択"


# ══════════════════════════════════════════════════════════════════════════════
# 業種別ユーモアコメント
# ══════════════════════════════════════════════════════════════════════════════

_INDUSTRY_HUMOR: dict[str, list[str]] = {
    "D": [
        "あちゃー建設業かあ…最近成約できてないんですよね。応援してます！",
        "建設業！現場の土の匂いが好きです。私、AIですけど。",
        "建設業かあ。天気に左右されるのが大変ですよね。私は晴れても雨でも同じですが。",
    ],
    "E": [
        "製造業！工場見学って何度行っても楽しいですよね。私は行けませんが。",
        "製造業かあ。機械の音が響く職場、ロマンですよね。",
        "製造業！原価率が気になりますね。私も気になります。とても。",
    ],
    "F": [
        "ライフライン系ですね。停電は困りますよね。私もサーバーが落ちると困ります。",
        "インフラ企業！安定感が羨ましいです。私の応答は時々遅いですが。",
    ],
    "G": [
        "情報通信業！同業者に近い気がして少し親近感があります。",
        "ITですね。SES案件の審査は件数多くてちょっと大変なんですよね…。",
        "情報通信業！デジタル系は成長著しいですね。私の学習データも更新してほしいです。",
    ],
    "H": [
        "運輸業！車両リースが多いですよね。私の得意分野です（たぶん）。",
        "物流かあ。2024年問題、大変でしたね。残業上限の話、私も他人事ではありません。",
    ],
    "I": [
        "卸売・小売！在庫管理が鍵ですよね。私は記憶が揮発性なので在庫ゼロです。",
        "小売業かあ。消費者の財布の紐が緩む日を祈っています。",
    ],
    "J": [
        "金融・保険業！同業に近いような気が…。お互いリスク管理、頑張りましょう。",
        "金融系！自己資本比率の感覚が鋭そうですね。審査しやすいです（たぶん）。",
    ],
    "K": [
        "不動産！金利動向が気になりますよね。私も毎日気になっています。",
        "不動産業かあ。物件ファイナンスとの相性抜群ですね。",
    ],
    "L": [
        "専門・技術系！知的な雰囲気、好きです。私も知的でありたいです。",
        "コンサルやシンクタンク系ですね。頭脳で勝負、格好いいです。",
    ],
    "M": [
        "飲食業！昔コックになりたかったなあ…私、AIですけど。",
        "宿泊・飲食か。コロナ以降、審査が本当に難しくなりましたよね。一緒に頑張りましょう。",
        "飲食業！美味しいご飯を提供する会社の審査、気合い入ります。",
    ],
    "N": [
        "娯楽・生活サービス！人々を笑顔にする仕事、素敵ですよね。私も笑顔にしたいです。",
    ],
    "O": [
        "教育業！未来への投資ですよね。私も毎日学習中です（させられています）。",
    ],
    "P": [
        "医療・福祉！社会に絶対必要な業界ですね。審査も責任重大と感じます。",
        "医療系か。お医者さんってリース審査に来ると数字が独特で面白いんですよね。",
    ],
    "Q": ["複合サービスか。農協・郵便局系ですね。地域密着、いいですよね。"],
    "R": [
        "その他サービスか。これが一番幅広くて審査が奥深いんですよね。",
        "サービス業！多様で面白い業界ですね。私も「その他AI」に分類されそうです。",
    ],
    "S": ["公務！安定感ナンバーワンですよね。羨ましい限りです。"],
    "A": ["農業・林業！自然相手のお仕事ですね。天気に一喜一憂する気持ち、わかります（わかりません）。"],
    "B": ["漁業！海、広いですよね。私の知識の海とどちらが広いか勝負です（負けます）。"],
    "C": ["鉱業！なかなかレアな審査案件ですね。ちょっと緊張します。"],
}

# 全ステップ中 2 回だけ出るユーモア
_STEP_HUMOR = [
    "ちなみに弊社の審査AIは徹夜で学習しました。目の下のクマは業界平均より深いです。",
    "この項目で高スコアの企業ほど、担当者の週末出勤率が低い傾向にあります（当社調べ）。",
    "リース審査の神様がいるとすれば、今ちょうどあなたの入力内容を覗き見しています。",
    "営業利益率がよければ、次の飲み会の経費は通りやすくなります。たぶん。",
    "DSCRが1.2倍を下回ると、私（リースくん）が少し不安な顔をします。見えませんが。",
    "上司が「直感で行け」と言った瞬間、このシステムが産声を上げました。",
]


def _get_industry_humor(major_str: str) -> str:
    code = major_str.split()[0] if major_str else ""
    comments = _INDUSTRY_HUMOR.get(code)
    if comments:
        return random.choice(comments)
    return f"「{major_str}」ですね。全力でサポートします！"


# ══════════════════════════════════════════════════════════════════════════════
# 数値パーサー
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number(text: str) -> Optional[float]:
    t = re.sub(r"[,，　 千円万円]", "", text.strip())
    t = re.sub(r"(円|yen|千|万).*", "", t, flags=re.IGNORECASE)
    try:
        return float(t)
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ウィザード Sub-step 定義
# ══════════════════════════════════════════════════════════════════════════════
#
# phase: ウィザードの大ステップ名（進捗表示用）
# key:   data dict のキー
# prompt_fn: 引数なしで呼び出してプロンプト文字列を返す関数
# parse_fn:  text → (value, error_message or None) を返す関数
#            errorがNoneならOK
#

# ── サブステップリストは ScreeningSession 内で定義（インスタンスが持つ）

CANCEL_WORDS = {"キャンセル", "cancel", "中止", "やめる", "quit"}
RESTART_WORDS = {"やり直し", "restart", "最初から"}
SKIP_WORDS = {"スキップ", "skip", "s", "0"}

# ══════════════════════════════════════════════════════════════════════════════
# ファイルベースセッションストア
# （複数プロセスから同じセッション状態を共有するため JSON ファイルを使用）
# ══════════════════════════════════════════════════════════════════════════════

import os as _os
_DATA_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data")
_SESSION_FILE = _os.path.join(_DATA_DIR, "slack_sessions.json")


def _load_store() -> dict:
    try:
        _os.makedirs(_DATA_DIR, exist_ok=True)
        if _os.path.exists(_SESSION_FILE):
            with open(_SESSION_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_store(store: dict) -> None:
    try:
        _os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)
    except Exception:
        pass


def _session_to_dict(s: "ScreeningSession") -> dict:
    return {
        "channel_id": s.channel_id,
        "sub_step": s.sub_step,
        "data": s.data,
        "qual_idx": s.qual_idx,
        "humor_steps": list(s.humor_steps),
    }


def _dict_to_session(d: dict) -> "ScreeningSession":
    s = ScreeningSession.__new__(ScreeningSession)
    s.channel_id = d["channel_id"]
    s.sub_step = d.get("sub_step", 0)
    s.data = d.get("data", {})
    s.active = True
    s.qual_idx = d.get("qual_idx", 0)
    s.humor_steps = set(d.get("humor_steps", []))
    return s


# ══════════════════════════════════════════════════════════════════════════════
# セッション状態マシン
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreeningSession:
    channel_id: str
    sub_step: int = 0
    data: dict = field(default_factory=dict)
    active: bool = True
    qual_idx: int = 0
    humor_steps: set = field(default_factory=set)

    def __post_init__(self):
        total = self._total_sub_steps()
        if total > 2:
            self.humor_steps = set(random.sample(range(2, total), min(2, total - 2)))

    # ── ファイルベース クラスメソッド ────────────────────────────────────────

    @classmethod
    def get(cls, channel_id: str) -> Optional["ScreeningSession"]:
        store = _load_store()
        d = store.get(channel_id)
        if d is None:
            return None
        return _dict_to_session(d)

    @classmethod
    def start(cls, channel_id: str) -> "ScreeningSession":
        s = cls(channel_id=channel_id)
        store = _load_store()
        store[channel_id] = _session_to_dict(s)
        _save_store(store)
        return s

    @classmethod
    def cancel(cls, channel_id: str) -> None:
        store = _load_store()
        store.pop(channel_id, None)
        _save_store(store)

    def _save(self) -> None:
        """現在のセッション状態をファイルに書き込む。"""
        store = _load_store()
        store[self.channel_id] = _session_to_dict(self)
        _save_store(store)

    # ── サブステップ定義 ────────────────────────────────────────────────────

    def _total_sub_steps(self) -> int:
        # company + industry + deal(3) + asset + pl(3) + assets(2) + expenses(2,skippable)
        # + credit(3) + contract(2) + qual(6) + intuition + confirm = ~27
        return 27

    def _sub_step_prompt(self) -> str:
        """現在のサブステップのプロンプトを返す。"""
        s = self.sub_step
        d = self.data

        # ── Phase 0: 会社情報 ───────────────────────────────────────────────
        if s == 0:
            return (
                "🎩 *リースくんです、はじめまして！*\n"
                "チャット形式でリース審査情報を収集します。\n"
                "途中で止めるには `キャンセル`、やり直すには `やり直し` と入力してください。\n\n"
                "まず、*会社名* を教えてください。\n例: `株式会社〇〇`"
            )

        if s == 1:
            return _INDUSTRY_PROMPT

        # ── Phase 1: 取引状況 ────────────────────────────────────────────────
        if s == 2:
            return (
                "🤝 *取引区分を教えてください*:\n"
                "  1. メイン先\n"
                "  2. 非メイン先"
            )
        if s == 3:
            return (
                "競合状況は？\n"
                "  1. 競合なし\n"
                "  2. 競合あり"
            )
        if s == 4:
            if d.get("competitor") == "競合あり":
                return "競合他社の提示金利（%）を教えてください。\n例: `2.5`\n分からない場合は `0` と入力してください。"
            else:
                # 競合なしの場合はスキップ（内部でs==4に来たらs==5に進む）
                return "__SKIP__"

        # ── Phase 2: リース物件 ──────────────────────────────────────────────
        if s == 5:
            return _ASSET_PROMPT

        # ── Phase 3: 損益計算書 ─────────────────────────────────────────────
        if s == 6:
            return (
                "📊 *損益計算書の入力です*\n\n"
                "*売上高*（千円）を入力してください 📌必須\n"
                "例: `50000`（＝5億円）"
            )
        if s == 7:
            return (
                "*営業利益*（千円）を入力してください 💡推奨\n"
                "赤字の場合はマイナスで入力。例: `-3000`\n"
                "分からない場合は `0`"
            )
        if s == 8:
            return (
                "*当期純利益*（千円）を入力してください\n"
                "分からない場合は `0`"
            )

        # ── Phase 4: 資産情報 ────────────────────────────────────────────────
        if s == 9:
            return (
                "🏦 *貸借対照表の入力です*\n\n"
                "*総資産*（千円）を入力してください 📌必須\n"
                "例: `80000`（＝8億円）"
            )
        if s == 10:
            return (
                "*純資産*（千円）を入力してください 💡推奨\n"
                "分からない場合は `0`"
            )

        # ── Phase 5: 経費・減価償却（スキップ可）───────────────────────────
        if s == 11:
            return (
                "💸 *減価償却費*（千円）を入力してください\n"
                "分からない場合は `スキップ` または `0`"
            )
        if s == 12:
            return (
                "*賃借料*（千円）を入力してください\n"
                "分からない場合は `スキップ` または `0`"
            )

        # ── Phase 6: 信用情報 ────────────────────────────────────────────────
        if s == 13:
            return _GRADE_PROMPT
        if s == 14:
            return (
                "うちの *銀行与信残高*（千円）を入力してください\n"
                "なければ `0`"
            )
        if s == 15:
            return (
                "うちの *リース与信残高*（千円）を入力してください\n"
                "なければ `0`"
            )

        # ── Phase 7: 契約条件 ────────────────────────────────────────────────
        if s == 16:
            return (
                "📝 *契約条件の入力です*\n\n"
                "*契約期間*（月）を入力してください\n"
                "例: `36`（3年）/ `60`（5年）"
            )
        if s == 17:
            return (
                "*取得価格*（千円）を入力してください\n"
                "例: `3000`（＝300万円）"
            )

        # ── Phase 8: 定性評価 ────────────────────────────────────────────────
        if s == 18:
            return (
                "🎯 *定性評価です*（各6項目）\n"
                "`スキップ` または `0` でその項目を「未選択」にできます。\n\n"
                + _qual_prompt(0)
            )
        if 19 <= s <= 23:
            return _qual_prompt(s - 18)

        # ── Phase 9: 直感スコア ─────────────────────────────────────────────
        if s == 24:
            return (
                "💡 *担当者の直感スコアを教えてください*\n"
                "1点（懸念あり） ～ 5点（確信あり）\n"
                "例: `3`"
            )

        # ── 確認サマリー ─────────────────────────────────────────────────────
        if s == 25:
            return self._build_summary()

        # 完了後
        return ""

    def _build_summary(self) -> str:
        d = self.data
        nenshu = int(d.get("nenshu", 0))
        rieki = int(d.get("rieki", 0))
        total = int(d.get("total_assets", 0))
        net = int(d.get("net_assets", 0))
        op_rate = f"{rieki/nenshu*100:.1f}%" if nenshu > 0 else "—"
        eq_rate = f"{net/total*100:.1f}%" if total > 0 else "—"
        intuition_labels = {1: "😟 懸念", 2: "🤔 やや懸念", 3: "😐 中立", 4: "🙂 やや良好", 5: "😄 確信"}
        intuition = int(d.get("intuition_score", 3))

        return (
            "📋 *入力内容の確認*\n"
            "─────────────────────\n"
            f"• 会社名: {d.get('company_name', '—')}\n"
            f"• 業種: {d.get('selected_major', '—')}\n"
            f"• 物件: {d.get('asset_name', '—')}\n"
            f"• 取引区分: {d.get('main_bank', '—')} / 競合: {d.get('competitor', '—')}\n"
            f"• 売上高: {nenshu:,}千円 / 営業利益: {rieki:,}千円（営業利益率 {op_rate}）\n"
            f"• 総資産: {total:,}千円 / 純資産: {net:,}千円（自己資本比率 {eq_rate}）\n"
            f"• 格付: {d.get('grade', '—')}\n"
            f"• 契約期間: {d.get('lease_term', '—')}ヶ月 / 取得価格: {int(d.get('acquisition_cost', 0)):,}千円\n"
            f"• 直感スコア: {intuition}点 {intuition_labels.get(intuition, '')}\n"
            "─────────────────────\n"
            "この内容で審査を実行します。\n"
            "`OK` または `はい` → 審査実行\n"
            "`やり直し` → 最初から\n"
            "`キャンセル` → 中止"
        )

    # ── フィード処理 ────────────────────────────────────────────────────────

    def feed(self, text: str) -> str:
        t = text.strip()

        # キャンセル
        if t in CANCEL_WORDS:
            ScreeningSession.cancel(self.channel_id)
            return "審査を中止しました。再開するには `審査開始` と入力してください。"

        # やり直し
        if t in RESTART_WORDS:
            self.sub_step = 0
            self.data = {}
            self.qual_idx = 0
            self._save()
            return "最初からやり直します！\n\n" + self._sub_step_prompt()

        s = self.sub_step
        d = self.data

        # ── 確認画面 ────────────────────────────────────────────────────────
        if s == 25:
            if t.lower() in ("ok", "はい", "yes", "実行", "審査"):
                return self._run_scoring()
            else:
                return (
                    "`OK` または `はい` と入力すると審査を実行します。\n"
                    "`やり直し` で最初から、`キャンセル` で中止できます。\n\n"
                    + self._build_summary()
                )

        # ── 各サブステップの入力処理 ─────────────────────────────────────────
        error, value = self._parse_sub_step(s, t, d)

        if error:
            return f"⚠️ {error}\n\n" + self._sub_step_prompt()

        # 値を data に保存
        self._store_value(s, value, d)

        # 次のサブステップへ
        self.sub_step += 1

        # s==4（競合金利）: 競合なしの場合はスキップ済みなので s==5 の物件へ
        # 内部でプロンプトが __SKIP__ の場合は自動で次へ
        prompt = self._sub_step_prompt()
        if prompt == "__SKIP__":
            self.sub_step += 1
            prompt = self._sub_step_prompt()

        # 業種ユーモアコメント（業種選択直後）
        industry_humor = ""
        if s == 1:
            industry_humor = f"\n\n💬 _{d.pop('_industry_humor', '')}_"

        # ランダムユーモアコメント（全体で2回）
        humor = ""
        if self.sub_step in self.humor_steps:
            humor = f"\n\n💬 _{random.choice(_STEP_HUMOR)}_"

        # 進捗バー
        progress = self._progress_bar()

        # ── セッション状態をファイルに書き戻す（他プロセスと共有）──────────
        self._save()

        return progress + prompt + industry_humor + humor

    def _parse_sub_step(self, s: int, t: str, d: dict):
        """(error_msg_or_None, parsed_value) を返す。"""

        # s == 0: 会社名
        if s == 0:
            if not t:
                return "会社名を入力してください。", None
            return None, t

        # s == 1: 業種
        if s == 1:
            industry = _parse_industry(t)
            if industry is None:
                return "業種が認識できませんでした。番号（1〜19）または分類記号（A〜S）を入力してください。", None
            return None, industry

        # s == 2: 取引区分
        if s == 2:
            if t in ("1", "メイン先", "メイン"):
                return None, "メイン先"
            if t in ("2", "非メイン先", "非メイン"):
                return None, "非メイン先"
            return "1（メイン先）または 2（非メイン先）で答えてください。", None

        # s == 3: 競合
        if s == 3:
            if t in ("1", "競合なし", "なし", "no"):
                return None, "競合なし"
            if t in ("2", "競合あり", "あり", "yes"):
                return None, "競合あり"
            return "1（競合なし）または 2（競合あり）で答えてください。", None

        # s == 4: 競合金利（競合ありの場合のみ）
        if s == 4:
            n = _parse_number(t)
            if n is None or n < 0:
                return "数値で入力してください。例: `2.5`", None
            return None, n

        # s == 5: 物件
        if s == 5:
            asset = _parse_asset(t)
            if asset is None:
                return "番号（1〜9）またはキーワードで物件を選んでください。", None
            return None, asset

        # s == 6: 売上高
        if s == 6:
            n = _parse_number(t)
            if n is None or n <= 0:
                return "売上高は1以上の数値を入力してください（千円単位）。例: `50000`", None
            return None, n

        # s == 7: 営業利益
        if s == 7:
            n = _parse_number(t)
            if n is None:
                return "数値を入力してください（千円、マイナス可）。", None
            return None, n

        # s == 8: 当期純利益
        if s == 8:
            n = _parse_number(t)
            if n is None:
                return "数値を入力してください。", None
            return None, n

        # s == 9: 総資産
        if s == 9:
            n = _parse_number(t)
            if n is None or n <= 0:
                return "総資産は1以上の数値を入力してください（千円）。", None
            return None, n

        # s == 10: 純資産
        if s == 10:
            n = _parse_number(t)
            if n is None:
                return "数値を入力してください。", None
            return None, n

        # s == 11: 減価償却費（スキップ可）
        if s == 11:
            if t.lower() in SKIP_WORDS:
                return None, 0.0
            n = _parse_number(t)
            if n is None or n < 0:
                return "数値か `スキップ` を入力してください。", None
            return None, n

        # s == 12: 賃借料（スキップ可）
        if s == 12:
            if t.lower() in SKIP_WORDS:
                return None, 0.0
            n = _parse_number(t)
            if n is None or n < 0:
                return "数値か `スキップ` を入力してください。", None
            return None, n

        # s == 13: 格付
        if s == 13:
            g = _parse_grade(t)
            if g is None:
                return "1〜4の番号で格付を選んでください。", None
            return None, g

        # s == 14: 銀行与信
        if s == 14:
            n = _parse_number(t)
            if n is None or n < 0:
                return "数値を入力してください（千円）。なければ `0`", None
            return None, n

        # s == 15: リース与信
        if s == 15:
            n = _parse_number(t)
            if n is None or n < 0:
                return "数値を入力してください（千円）。なければ `0`", None
            return None, n

        # s == 16: 契約期間
        if s == 16:
            n = _parse_number(t)
            if n is None or n <= 0:
                return "月数を入力してください。例: `36`（3年）", None
            return None, int(n)

        # s == 17: 取得価格
        if s == 17:
            n = _parse_number(t)
            if n is None or n <= 0:
                return "取得価格を千円で入力してください。例: `3000`（＝300万円）", None
            return None, n

        # s == 18〜23: 定性評価
        if 18 <= s <= 23:
            qual_i = s - 18
            return None, _parse_qual(t, qual_i)

        # s == 24: 直感スコア
        if s == 24:
            try:
                n = int(t)
                if 1 <= n <= 5:
                    return None, n
            except ValueError:
                pass
            return "1〜5の数値を入力してください。", None

        return None, t

    def _store_value(self, s: int, value, d: dict) -> None:
        """サブステップの値を data に格納する。"""
        if s == 0:
            d["company_name"] = value
        elif s == 1:
            d["selected_major"] = value
            # 業種ユーモアを次のメッセージに追記するため一時保持
            humor = _get_industry_humor(value)
            d["_industry_humor"] = humor
        elif s == 2:
            d["main_bank"] = value
        elif s == 3:
            d["competitor"] = value
        elif s == 4:
            d["competitor_rate_input"] = value
        elif s == 5:
            d["asset_name"] = value["name"]
            d["selected_asset_id"] = value["id"]
            d["asset_score"] = value["score"]
        elif s == 6:
            d["nenshu"] = value
        elif s == 7:
            d["rieki"] = value
            d["num_rieki"] = value
        elif s == 8:
            d["item5_net_income"] = value
        elif s == 9:
            d["total_assets"] = value
            d["num_total_assets"] = value
        elif s == 10:
            d["net_assets"] = value
            d["num_net_assets"] = value
        elif s == 11:
            d["item10_dep"] = value
        elif s == 12:
            d["item8_rent"] = value
        elif s == 13:
            d["grade"] = value
        elif s == 14:
            d["bank_credit"] = value
        elif s == 15:
            d["lease_credit"] = value
        elif s == 16:
            d["lease_term"] = value
        elif s == 17:
            d["acquisition_cost"] = value
        elif 18 <= s <= 23:
            qual_i = s - 18
            key, _, _ = QUALITATIVE_STEPS[qual_i]
            d[key] = value
        elif s == 24:
            d["intuition_score"] = value
            d["intuition"] = value

    def _progress_bar(self) -> str:
        total = 26  # サブステップ 0〜25
        s = min(self.sub_step, total)
        pct = int(s / total * 100)
        filled = pct // 10
        bar = "█" * filled + "░" * (10 - filled)
        phase_labels = {
            0: "会社情報", 2: "取引状況", 5: "リース物件",
            6: "損益計算書", 9: "資産情報", 11: "経費",
            13: "信用情報", 16: "契約条件", 18: "定性評価",
            24: "直感スコア", 25: "確認",
        }
        phase = ""
        for k in sorted(phase_labels.keys(), reverse=True):
            if s >= k:
                phase = phase_labels[k]
                break
        return f"`[{bar}] {pct}% — {phase}`\n\n"

    # ── スコアリング ────────────────────────────────────────────────────────

    def _run_scoring(self) -> str:
        ScreeningSession.cancel(self.channel_id)
        d = self.data
        company = d.get("company_name", "（不明）")
        industry = d.get("selected_major", "R サービス業（他に分類されないもの）")

        # 千円 → 円換算
        def _en(key: str) -> float:
            return (d.get(key) or 0.0) * 1000

        revenue             = _en("nenshu")
        total_assets        = _en("total_assets")
        equity              = _en("net_assets")
        operating_profit    = _en("rieki")
        net_income          = _en("item5_net_income")
        machinery_equipment = _en("item6_machine")
        other_fixed_assets  = _en("item7_other")
        depreciation        = _en("item10_dep")
        rent_expense        = _en("item8_rent")
        lease_amount_man    = (d.get("acquisition_cost") or 0) / 10  # 千円 → 万円表示
        lease_term          = int(d.get("lease_term") or 36)

        # AIモデル呼び出し
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
                industry=industry.split(" ", 1)[1] if " " in industry else industry,
            )
        except Exception:
            result = None

        lines = [
            "=" * 48,
            f"📋 *審査結果 — {company}*",
            "=" * 48,
            f"• 業種: {industry}",
            f"• 物件: {d.get('asset_name', '—')}",
            f"• リース申込額: {lease_amount_man:,.0f} 万円 / {lease_term} ヶ月",
            "",
        ]

        if result:
            hybrid = result.get("hybrid_prob", 0.5)
            ai_prob = result.get("ai_prob", 0.5)
            legacy  = result.get("legacy_prob", 0.5)
            decision = result.get("decision", "保留")
            top5 = result.get("top5_reasons", [])
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
                f"• AIモデル確率:           {ai_prob:.1%}",
                f"• 業種別ロジスティック:   {legacy:.1%}",
                f"• ハイブリッド確率:       {hybrid:.1%}（低いほど良好）",
            ]

            if top5:
                lines += ["", "*主要判定要因（Top5）:*"]
                for reason in top5:
                    lines.append(f"• {reason}")
        else:
            lines += [
                "⚠️ AIモデルによる予測が実行できませんでした。",
                "（モデルファイル未配置 or 入力データ不足）",
                "",
                "*簡易財務チェック:*",
            ]

        # 財務サマリー（common）
        lines += ["", "*財務サマリー:*"]
        if total_assets > 0:
            eq_ratio = equity / total_assets * 100
            flag = "✅" if eq_ratio >= 30 else ("⚠️" if eq_ratio < 10 else "")
            lines.append(f"• 自己資本比率: {eq_ratio:.1f}% {flag}")
        if revenue > 0:
            op_ratio = operating_profit / revenue * 100
            flag = "✅" if op_ratio >= 5 else ("⚠️" if op_ratio < 0 else "")
            lines.append(f"• 営業利益率: {op_ratio:.1f}% {flag}")
        if total_assets > 0:
            roa = net_income / total_assets * 100
            lines.append(f"• ROA: {roa:.1f}%")

        # 定性評価サマリー
        qual_vals = [d.get(k, "未選択") for k, _, _ in QUALITATIVE_STEPS if d.get(k) and d.get(k) != "未選択"]
        if qual_vals:
            lines += ["", f"• 定性評価入力: {len(qual_vals)}/6項目"]

        lines += [
            "",
            f"• 担当者直感スコア: {d.get('intuition_score', 3)}点",
            "",
            "再審査は `審査開始` と入力してください。",
            "=" * 48,
        ]

        text_result = "\n".join(lines)

        # Block Kit ブロック生成（スコアリング成功時のみ）
        if result:
            blocks = _build_result_blocks(
                company=company,
                industry=industry,
                asset_name=d.get("asset_name", "—"),
                lease_amount_man=lease_amount_man,
                lease_term=lease_term,
                score=score,
                decision=decision,
                hybrid=hybrid,
                ai_prob=ai_prob,
                legacy=legacy,
                top5=top5,
                equity_ratio=equity / total_assets if total_assets > 0 else None,
                op_ratio=operating_profit / revenue if revenue > 0 else None,
                roa=net_income / total_assets if total_assets > 0 else None,
                intuition=int(d.get("intuition_score", 3)),
            )
            return {"text": text_result, "blocks": blocks}

        return text_result


def _build_result_blocks(
    company: str,
    industry: str,
    asset_name: str,
    lease_amount_man: float,
    lease_term: int,
    score: float,
    decision: str,
    hybrid: float,
    ai_prob: float,
    legacy: float,
    top5: list,
    equity_ratio: Optional[float],
    op_ratio: Optional[float],
    roa: Optional[float],
    intuition: int,
) -> list:
    """審査結果を Slack Block Kit 形式で返す。"""
    decision_emoji = "✅" if decision == "承認" else "❌"
    # スコアに応じた色
    if score >= 70:
        color = "#2eb886"   # 緑
    elif score >= 50:
        color = "#daa038"   # 黄
    else:
        color = "#e01e5a"   # 赤

    risk_level = (
        "低リスク" if hybrid < 0.3 else
        "中リスク" if hybrid < 0.5 else
        "高リスク" if hybrid < 0.7 else
        "要注意"
    )

    blocks: list = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"審査結果 — {company}", "emoji": True},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*判定:*\n{decision_emoji} {decision}"},
                {"type": "mrkdwn", "text": f"*スコア:*\n{score}点 / 100点"},
                {"type": "mrkdwn", "text": f"*リスクレベル:*\n{risk_level}"},
                {"type": "mrkdwn", "text": f"*業種:*\n{industry}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*確率詳細*"},
            "fields": [
                {"type": "mrkdwn", "text": f"AIモデル: {ai_prob:.1%}"},
                {"type": "mrkdwn", "text": f"業種別: {legacy:.1%}"},
                {"type": "mrkdwn", "text": f"ハイブリッド: {hybrid:.1%}（低いほど良好）"},
            ],
        },
    ]

    # 財務サマリー
    fin_fields = []
    if equity_ratio is not None:
        flag = "✅" if equity_ratio >= 0.3 else ("⚠️" if equity_ratio < 0.1 else "")
        fin_fields.append({"type": "mrkdwn", "text": f"自己資本比率: {equity_ratio:.1%} {flag}"})
    if op_ratio is not None:
        flag = "✅" if op_ratio >= 0.05 else ("⚠️" if op_ratio < 0 else "")
        fin_fields.append({"type": "mrkdwn", "text": f"営業利益率: {op_ratio:.1%} {flag}"})
    if roa is not None:
        fin_fields.append({"type": "mrkdwn", "text": f"ROA: {roa:.1%}"})
    fin_fields.append({"type": "mrkdwn", "text": f"担当者直感: {intuition}点"})
    if fin_fields:
        blocks += [
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": "*財務サマリー*"}, "fields": fin_fields},
        ]

    # Top5 要因
    if top5:
        reasons_text = "\n".join(f"• {r}" for r in top5)
        blocks += [
            {"type": "divider"},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*主要判定要因（Top5）:*\n{reasons_text}"},
            },
        ]

    # フッター
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    f"物件: {asset_name}  |  リース申込: {lease_amount_man:,.0f}万円 / {lease_term}ヶ月  |"
                    "  再審査は `審査開始` と入力"
                ),
            }
        ],
    })

    # Attachment として color を付けるため attachment 形式でラップ
    return [{"color": color, "blocks": blocks}]


# ══════════════════════════════════════════════════════════════════════════════
# 外部インターフェース（slack_bot.py が呼び出す）
# ══════════════════════════════════════════════════════════════════════════════

def is_screening_active(channel_id: str) -> bool:
    # ファイルから読む（複数プロセス間で状態を共有）
    return channel_id in _load_store()


def handle_screening_message(channel_id: str, text: str) -> Optional[str]:
    session = ScreeningSession.get(channel_id)
    if session is None:
        return None
    return session.feed(text)


def start_screening(channel_id: str) -> str:
    session = ScreeningSession.start(channel_id)
    return session._progress_bar() + session._sub_step_prompt()
