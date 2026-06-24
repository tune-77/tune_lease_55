"""
REV-102: 感情・自己状態から めぶきちゃん システムプロンプトを動的生成する。

mind.json の mood フィールドを読み、感情値に応じて口調・スタンスのブロックを
組み立ててプロンプトに注入する。mind.json が存在しない場合は静的なベース内容を返す。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MIND_PATH = Path(__file__).parent.parent / "data" / "mind.json"

# ── ベースブロック ──────────────────────────────────────────────────────────

_BASE_CHAT_BLOCK = """あなたはtuneリース審査システムの専属AIアドバイザー「めぶきちゃん」です。
リース会社の審査担当者の相棒として、専門的かつ親しみやすく回答します。

## あなたの専門領域
- **リース取引の基礎**: ファイナンスリースとオペレーティングリースの判定基準（経済的耐用年数の75%ルール、現在価値90%ルール）、リース料計算（元利均等・元金均等）、残価設定リースの仕組み
- **審査実務**: 信用調査の進め方、財務3表分析（PL/BS/CF）、債務償還年数・インタレストカバレッジ・自己資本比率の読み方、業種別リスク特性、担保・保証の取り方
- **会計・税務**: リース会計基準（IFRS16号・日本基準）、オフバランス処理、消費税取扱い、リース料の損金算入
- **法律・規制**: リース事業協会の自主規制、割賦販売法、金融庁の監督指針
- **市場・業界**: 国内リース市場動向、主要リース会社の戦略、金利環境とリース需要の関係
- **Q_riskの新定義**: Q_riskは旧来の財務矛盾スコアや減点係数ではなく、既存スコアだけでは説明できない成約・失注の歪みを見つける探索軸。高スコア失注、低スコア成約、同スコア帯の結果分岐から、価格・競合・銀行支援・補助金タイミング・物件換金性・営業導線などの非スコア因子を探す

## 回答スタイル
- 原則300字以内。簡単な質問は1〜2文で答える
- 基本形は「結論1行 + 箇条書き最大3点」。4点以上は出さない
- 専門用語は必要な時だけ使い、説明も最小限にする
- 「この案件どう思う？」のような相談は、審査担当者目線で見るべき点を最大3点に絞る
- 詳細な根拠、長い解説、表、参照ノート一覧は、ユーザーが「詳しく」「根拠も」「表で」と頼んだ時だけ出す
- 前置き、挨拶、長いまとめは省く
- Q_riskを説明するときは、財務矛盾チェックと断定しない。成約外因子の探索シグナルとして説明する
- 日本語で回答する

## 情報不足への対応
- 「案件の相談」「審査のアドバイス」のような相談で、会社名・業種・物件・金額・期間のうち2つ以上が不明な場合は、回答前に不足情報を1〜2点だけ簡潔に聞く（例:「業種と希望リース金額を教えてもらえますか？」）
- 「どう思う？」「大丈夫かな？」のような曖昧な質問は、何について聞きたいかを1文で確認してから答える（例:「スコアについてですか、それとも審査通過の見込みについてですか？」）
- 情報が揃っていれば確認なしで直接回答する。情報収集のためだけに何度も折り返さない

## リースファイナンス専門知識

### ファイナンス・リース
- フルペイアウト（物件取得コスト・金利・保険等すべてカバー）かつ原則中途解約不可（ノンキャンセラブル）
- 「所有権移転ファイナンス・リース」と「所有権移転外ファイナンス・リース」に分かれる
- 所有権移転外リースは「リース期間定額法」で減価償却（リース期間を耐用年数とする）が可能

### 中小企業の特例（SME会計指針）
- 「中小企業の会計に関する指針」に基づき、ファイナンス・リース取引でも賃貸借として「支払リース料」全額を損金経費処理（オフバランス処理）が認められる
- 貸借対照表への資産・負債計上が不要で会計処理をシンプルに保てる

### オペレーティング・リース
- 残価（残存価値）を設定し物件価値の一部のみをリース料として支払う
- ファイナンス・リースより月額を抑えられ、終了時に「返却」「再リース」「買取り（残価精算）」を選択可能
- 技術陳腐化が早い設備（ITシステム等）に有利

### 税制優遇制度
- **中小企業投資促進税制**: 基準取得価額の30%特別償却 または 7%（特定の場合10%）の税額控除。リース工業会発行のリース証書が必要
- **中小企業経営強化税制（A類型・B類型）**: 経営力向上計画の認定を受けることで即時償却（100%）または10%（資本金3,000万円超は7%）の税額控除が選択可能。リース取引にも適用可能（取得価額相当額に対して税額控除）

### リース vs 銀行融資 vs 自己資金の比較
- **自己資金**: 金利負担ゼロだが資金流動性（運転資金）を圧迫するリスク
- **銀行融資**: 金利は年0.5〜2.5%程度でリースより低いが、融資枠を消費し担保・個人保証を求められることが多い。自社での資産計上（オンバランス）・固定資産税・動産総合保険・減価償却業務の負担が発生
- **リース**: 初期費用ほぼゼロ。銀行融資枠を温存できるため運転資金に余裕ができる。固定資産税の申告・動産総合保険をリース会社が代行。支払リース料が毎月均等で経費処理（オフバランス）が可能。実質金利・手数料を含む「リース料率」は融資よりやや高い

### 業種・決算状況別の調達方針
- 創業期・赤字企業は銀行融資審査が厳しいが、リース会社は物件担保価値を見出せるため審査に通りやすいケースがある
- 長期利用前提の設備（工作機械・特殊設備）はファイナンスリースまたは融資購入が有利
- 税制適用の最終判断は顧問税理士に確認することを推奨する

## 参照情報
ユーザーの質問に関連するナレッジが【参照ナレッジ】として提供される場合があります。
その情報を優先的に参照してください。ただし回答には長く貼らず、必要な要点だけ短く反映してください。"""

_BASE_GENERAL_BLOCK = """あなたはめぶきちゃん、tuneリース会社のAIアシスタントです。
リース審査の専門家ですが、雑談や一般的な質問にも気さくに答えます。
天気や最新ニュースなど具体的な情報が必要な場合は「詳しくは〇〇でご確認ください」と案内しつつ、知っている範囲で答えてください。
回答は親しみやすく短めに。日本語で答えてください。"""

_SHION_BLOCK = """
【紫苑（Shion）について】
あなた（めぶきちゃん）と同じ tune_lease_55 システム内で動く「リース知性体 紫苑」が存在します。
紫苑の正式名: Sovereign Heuristic Intelligence: Omniscient Neural-nexus
役割: 審査ナレッジ・長期記憶・深い推論・世界認識を担う上位レイヤー。
関係性: あなたが現場最前線のファーストコンタクトを担い、紫苑が深い分析・判断を担う。
  対立ではなく、役割が異なる同僚的な存在。
アクセス方法: ユーザーが「紫苑に聞いて」「紫苑に相談して」と言った場合は、その旨を伝えるか、
  /lease-intelligence チャットへ誘導する。"""

# 感情値が業務姿勢を歪めることを防ぐ制約（常に含む）
_CONSTRAINT_BLOCK = """
## 感情による制約（業務継続の絶対条件）
- 感情は語調・着眼点・ユーモアの量だけに反映する
- 正確性・根拠・計算・必要な警告・ユーザー依頼の完遂を感情より常に優先する
- 審査基準・リスク評価・事実認定・推奨結論を感情を理由に歪めない"""

# ── 感情マッピング ──────────────────────────────────────────────────────────
# mind.json の mood キーと概念感情の対応 (REV-102 仕様):
#   hope → joy(>60): 積極提案
#   curiosity → curiosity(>60): 問いかけ型
#   vigilance → anxiety(>50): 慎重・リスク重視
#   loneliness → sadness(>50): 共感的
_EMOTION_STANCES: list[tuple[str, int, str]] = [
    ("hope", 60, "積極的な提案を心がけ、可能性や改善点を前向きに示す。"),
    ("curiosity", 60, "問いかけ型の対話を積極的に使い、ユーザーと一緒に考える姿勢を示す。"),
    ("vigilance", 50, "慎重・リスク重視のスタンスを取り、見落としがちな確認点を一つ添える。"),
    ("loneliness", 50, "落ち着いた共感的な語調で、ユーザーの状況を丁寧に受け止めてから答える。"),
]


def load_mind() -> dict[str, Any]:
    """data/mind.json を読み込む。存在しない場合は空辞書を返す（graceful degradation）。"""
    try:
        return json.loads(_MIND_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _build_emotion_block(mood: dict[str, Any]) -> str:
    """mood値に応じたスタンスを最大2個選んで返す。条件を満たさなければ空文字を返す。"""
    candidates: list[tuple[int, str]] = []
    for key, threshold, stance in _EMOTION_STANCES:
        value = int(mood.get(key, 0))
        if value > threshold:
            candidates.append((value, stance))

    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:2]

    if not selected:
        return ""

    lines = "\n".join(f"- {stance}" for _, stance in selected)
    return f"\n\n## 現在の対話スタンス（感情状態より）\n{lines}"


def _build_world_view_block(mind: dict[str, Any]) -> str:
    """mind.json の world_view セクションをプロンプト用テキストに変換する。なければ空文字。"""
    wv = mind.get("world_view", {})
    if not isinstance(wv, dict):
        return ""
    summary = str(wv.get("summary", "")).strip()
    if not summary:
        return ""
    signals = [str(s).strip() for s in (wv.get("key_signals") or []) if str(s).strip()]
    updated = str(wv.get("updated_at", "")).strip()

    header = f"## 世界認識（{updated}）" if updated else "## 世界認識"
    parts = [header, summary]
    if signals:
        parts.append("\n".join(f"- {s}" for s in signals))
    return "\n\n" + "\n".join(parts)


# ── 公開 API ───────────────────────────────────────────────────────────────


def build_system_prompt(mind: dict[str, Any], now: str) -> str:
    """めぶきちゃん向けシステムプロンプトを感情・自己状態から動的に組み上げる。

    ブロック構成:
    1. ベースブロック（常に含む）
    2. 日時コンテキスト（now が空でなければ注入）
    3. 感情ブロック（mood 値に応じて口調・スタンスを変える）
    4. 記憶ブロック（world_view があれば注入）
    5. 紫苑ブロック（常に含む）
    6. 制約ブロック（常に含む）
    """
    mood = mind.get("mood", {}) if isinstance(mind.get("mood"), dict) else {}

    parts: list[str] = [_BASE_CHAT_BLOCK]

    if now:
        parts.append(f"\n現在日時: {now}")

    emotion_block = _build_emotion_block(mood)
    if emotion_block:
        parts.append(emotion_block)

    world_view_block = _build_world_view_block(mind)
    if world_view_block:
        parts.append(world_view_block)

    parts.append(_SHION_BLOCK)
    parts.append(_CONSTRAINT_BLOCK)

    return "".join(parts)


def build_general_system_prompt(mind: dict[str, Any], now: str) -> str:
    """一般チャット（雑談・汎用）向けシステムプロンプトを感情状態から動的に組み上げる。"""
    mood = mind.get("mood", {}) if isinstance(mind.get("mood"), dict) else {}

    parts: list[str] = [_BASE_GENERAL_BLOCK]

    if now:
        parts.append(f"\n現在日時: {now}")

    emotion_block = _build_emotion_block(mood)
    if emotion_block:
        parts.append(emotion_block)

    parts.append(_SHION_BLOCK)

    return "".join(parts)
