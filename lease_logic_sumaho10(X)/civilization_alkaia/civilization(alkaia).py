"""
アルカイアの慟哭 — 対話型文明シミュレーションゲーム（スタンドアロン版）

civilization.py のゲーム部分を別フォルダに分離したもの。元の civilization.py は変更しない。
データ（civilization_assessment.json, technology_level_table.json, arcaia_lore.json）は
このスクリプトの親フォルダ（lease_logic_sumaho10(X)/）を参照する。

起動: python "civilization(alkaia).py" （civilization_alkaia フォルダ内で）
      または リポジトリルートから: python "lease_logic_sumaho10(X)/civilization_alkaia/civilization(alkaia).py"
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------------
# 審査用定数・関数（civilization.py Part 1 から必要最小限）
# ---------------------------------------------------------------------------

JUDGMENT_THRESHOLDS = {
    "preserve": 60,
    "monitor": 40,
    "terminate": 0,
}


def infer_decay_reasons(
    tech_level: float,
    ethical_dev: float,
    sustainability: float,
    **kwargs: float,
) -> List[str]:
    """文明の指標から衰退理由を推定（6分類）。"""
    reasons: List[str] = []
    env_high = kwargs.get("environmental_impact", 50) > 60
    violence_high = kwargs.get("violence_index", 50) > 60
    inequality_high = kwargs.get("inequality", 50) > 60
    space_high = kwargs.get("space_exploration", 50) >= 70
    if sustainability < 35 or env_high:
        reasons.append("環境的破滅")
    if ethical_dev < 40 or violence_high or inequality_high:
        reasons.append("社会的破滅")
    tech_progress = kwargs.get("tech_progress", 50)
    if tech_level >= 70 and (kwargs.get("ai_development", 50) > 70 or kwargs.get("energy_utilization", 50) > 70):
        reasons.append("技術的破滅")
    if tech_progress >= 75 and tech_level >= 60:
        if "技術的破滅" not in reasons:
            reasons.append("技術的破滅")
    if tech_level >= 65 and space_high:
        reasons.append("宇宙的破滅")
    if sustainability < 50 and not env_high:
        reasons.append("生物学的破滅")
    if not reasons:
        reasons.append("その他")
    return reasons


class CivilizationJudgmentSystem:
    """文明存続の適合度判定（スコア計算のみ使用）。"""

    def __init__(self):
        self.civilizations: List[Dict] = []
        self.judgment_history: List[Dict] = []

    def calculate_civilization_score(
        self, tech_level: float, ethical_dev: float, sustainability: float
    ) -> Dict:
        total_score = tech_level * 0.30 + ethical_dev * 0.35 + sustainability * 0.35
        if total_score >= JUDGMENT_THRESHOLDS["preserve"]:
            judgment, action = "preserve", "プロトコル適合"
        elif total_score >= JUDGMENT_THRESHOLDS["monitor"]:
            judgment, action = "monitor", "要観測"
        else:
            judgment, action = "terminate", "排除・再生失敗"
        return {
            "tech_level": tech_level,
            "ethical_dev": ethical_dev,
            "sustainability": sustainability,
            "total_score": round(total_score, 2),
            "judgment": judgment,
            "action": action,
        }


# ---------------------------------------------------------------------------
# データパス（親フォルダ = lease_logic_sumaho10(X) を参照）
# ---------------------------------------------------------------------------

_GAME_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_GAME_SCRIPT_DIR)

CIVILIZATION_ASSESSMENT_PATH = os.path.join(_DATA_DIR, "civilization_assessment.json")
TECH_LEVEL_TABLE_PATH_GAME = os.path.join(_DATA_DIR, "technology_level_table.json")
ARCAIA_LORE_PATH_GAME = os.path.join(_DATA_DIR, "arcaia_lore.json")
SOLAR_SYSTEM_HABITABILITY_PATH = os.path.join(_DATA_DIR, "solar_system_habitability.json")
EARTH_CIVILIZATIONS_PATH = os.path.join(_DATA_DIR, "earth_civilizations.json")

# 1ターンあたりの経過年数（ゲーム内時間）
YEARS_PER_TURN = 100_000_000  # 1億年
EARTH_UNINHABITABLE_THRESHOLD_YEARS = 1_000_000_000  # 10億年

# リタの直感イベント: 発生確率（rita_happiness >= 80 のとき）
RITA_INTUITION_PROBABILITY = 0.08
RITA_INTUITION_BONUS = 5  # 各ステータスへの加算

# メタ構造: ループ破壊指数（隠しパラメータ）
LOOP_BREAKING_MAX = 1.0  # 最大で第5001番目の道へ分岐
LOOP_BREAKING_INCREMENT = 0.12  # 「愛のある非効率選択」1回あたりの上昇量（約8〜9回で最大）

# 破壊的ロジック: 退屈カウンター（似たような安全な選択3回でアルカイアがジャック）
SAFE_CHOICE_MAX_DELTA = 6  # 各deltaの絶対値がこれ以下なら「安全な選択」
CONSECUTIVE_SAFE_FOR_HIJACK = 3  # この回数続いたら強制ジャック
# 不可逆分岐: 選ぶとUIがバイナリ等に永久変容する選択肢ID
IRREVERSIBLE_CHOICE_IDS = frozenset({"milestone_3e9_upload", "turning_digital_1", "dyn_body_abandon"})

# ---------------------------------------------------------------------------
# シナリオバンク（状況・時代に応じたイベント）
# ---------------------------------------------------------------------------
# 【歴史の再演】過去の地球文明の問題を現代・未来風にアレンジ
SCENARIO_HISTORY: List[Dict[str, Any]] = [
    {"id": "history_salt", "label": "大規模灌漑による土壌・水質の悪化（現代版・メソポタミアの塩害）", "narrative": "灌漑農業の拡大で塩類が蓄積し、農地が疲弊し始めた。あなたは持続可能な水管理へ転換する政策を推し進めた。", "tech_level": 0, "ethical_dev": 4, "sustainability": 8, "rita_happiness": 2},
    {"id": "history_citizen", "label": "移民・難民の市民権と社会統合（現代版・ローマの市民権問題）", "narrative": "境界を越えて流入する人々にどう権利を与えるか。法の拡張と教育への投資を選んだ。", "tech_level": 2, "ethical_dev": 10, "sustainability": 2, "rita_happiness": 3},
    {"id": "history_drought", "label": "長期旱魃と都市の水不足（現代版・古典期マヤ）", "narrative": "気候変動で水供給が不安定になり、大都市が逼迫している。節水と貯水インフラの再設計を進めた。", "tech_level": 3, "ethical_dev": 2, "sustainability": 12, "rita_happiness": -2},
    {"id": "history_water_collapse", "label": "大規模水利システムの維持不全（現代版・アンコール）", "narrative": "巨大ダム・水路の維持コストが財政を圧迫している。縮小か効率化か。あなたは地域分散型の水管理へ移行した。", "tech_level": -1, "ethical_dev": 5, "sustainability": 10, "rita_happiness": 1},
    {"id": "history_resource", "label": "孤立域での資源枯渇（現代版・ラパ・ヌイ）", "narrative": "限られた圏内で資源を使い切りつつある。外部との交易か、徹底した循環か。循環経済を選んだ。", "tech_level": 1, "ethical_dev": 6, "sustainability": 14, "rita_happiness": 4},
    {"id": "history_epidemic", "label": "未知の病原体の侵入（現代版・アステカ・天然痘）", "narrative": "新たな病原体が拡大し、医療と情報の共有が問われた。国際的な監視とワクチン共有を推進した。", "tech_level": 4, "ethical_dev": 8, "sustainability": 4, "rita_happiness": 5},
    {"id": "history_division", "label": "内部分裂と外圧（現代版・インカ内戦）", "narrative": "内部対立が深まり、外からの圧力が増している。和解と再統合のプロセスを重視した。", "tech_level": 0, "ethical_dev": 12, "sustainability": 2, "rita_happiness": 2},
]

# 【技術の代償】level_name に応じたイベント（キーワードでマッチ）
# 各要素: keywords (level名に含まれると採用), label, narrative, deltas
SCENARIO_TECH_BY_LEVEL: List[Dict[str, Any]] = [
    {"keywords": ["石器", "農耕", "青銅", "鉄"], "label": "記録と法の整備 — 誰の言葉を残すか", "narrative": "記録媒体が限られるなか、何を残し何を捨てるか。多様な声を残す制度を選んだ。", "tech_level": 2, "ethical_dev": 6, "sustainability": 2, "rita_happiness": 1},
    {"keywords": ["印刷", "航海", "科学"], "label": "殖民と征服 — 遠くの土地をどう扱うか", "narrative": "新たな土地との接触が増える。略奪か、対等な交易か。あなたは現地の権利を認める枠組みを支持した。", "tech_level": 4, "ethical_dev": -2, "sustainability": -3, "rita_happiness": -3},
    {"keywords": ["産業", "蒸気", "電気", "内燃"], "label": "労働環境と格差 — 機械化の果てに", "narrative": "生産性は上がったが、労働者の権利と格差が問題になっている。規制と再分配を進めた。", "tech_level": 5, "ethical_dev": 8, "sustainability": -2, "rita_happiness": 2},
    {"keywords": ["核", "宇宙", "コンピュータ", "遺伝子"], "label": "核の傘と倫理の遅れ", "narrative": "破壊力と情報力が増す一方、倫理の議論が追いつかない。あなたは公開討論と国際枠組みを求めた。", "tech_level": 6, "ethical_dev": 6, "sustainability": 4, "rita_happiness": 0},
    {"keywords": ["インターネット", "バイオ", "ナノ"], "label": "監視・分断と気候危機", "narrative": "ネットワークはつながりと分断の両方をもたらした。気候対策と情報の透明性を両立させる道を探った。", "tech_level": 4, "ethical_dev": 5, "sustainability": 10, "rita_happiness": 1},
    {"keywords": ["AI", "量子", "気候制御"], "label": "意識の権利問題 — AIに「心」はあるか", "narrative": "高度なAIが権利を求め始めた。あなたは一定の保護と対話の枠組みを設けることを選んだ。", "tech_level": 8, "ethical_dev": 10, "sustainability": 2, "rita_happiness": 4},
    {"keywords": ["地球外", "核融合", "火星"], "label": "地球棄民の倫理 — 誰を残し誰を送るか", "narrative": "移住計画が現実になるなか、選別と格差が問われた。あなたは抽選と必要性のバランスを主張した。", "tech_level": 6, "ethical_dev": 4, "sustainability": 6, "rita_happiness": -2},
    {"keywords": ["AGI", "寿命延伸", "テラフォーミング"], "label": "人間の再定義 — 強化と自然の境界", "narrative": "身体と寿命の拡張が進む。何を「人間」とするか。多様な在り方を認めつつ、自然との線を議論した。", "tech_level": 5, "ethical_dev": 8, "sustainability": 4, "rita_happiness": 3},
    {"keywords": ["恒星間", "ダイソン", "意識の拡張"], "label": "宇宙病の蔓延 — 長期航行の代償", "narrative": "恒星間ミッションで心身の変調が報告されている。医療と心理支援の体制を拡充した。", "tech_level": 10, "ethical_dev": 6, "sustainability": 2, "rita_happiness": -4},
    {"keywords": ["カルダシェフ", "Type II", "銀河"], "label": "技術的破滅の規模の巨大化", "narrative": "文明のスケールが大きくなるほど、失敗の影響も計り知れない。あなたは段階的検証と制動を訴えた。", "tech_level": 4, "ethical_dev": 10, "sustainability": 6, "rita_happiness": 0},
]

# 【リタの介入】数値に現れない、生命の豊かさを問うイベント
SCENARIO_RITA: List[Dict[str, Any]] = [
    {"id": "rita_robot", "label": "捨てられた家庭用ロボットを保護するか", "narrative": "廃棄予定のロボットが「動き続けたい」と訴えた。あなたは保護と再プログラムの機会を認めた。リタがその横で尻尾を振った。", "tech_level": 0, "ethical_dev": 6, "sustainability": 0, "rita_happiness": 12},
    {"id": "rita_flower", "label": "絶滅寸前の一輪の花を育てるか", "narrative": "開発予定地にただ一株残った希少種。あなたは移植と保護区の設置を選んだ。リタはその土の匂いを嗅いだ。", "tech_level": -1, "ethical_dev": 4, "sustainability": 10, "rita_happiness": 10},
    {"id": "rita_stray", "label": "傷ついた野良動物を治療するか", "narrative": "道端で傷ついた動物がいた。あなたは手当てし、保護施設につないだ。リタがぴたりと寄り添っていた。", "tech_level": 0, "ethical_dev": 8, "sustainability": 2, "rita_happiness": 15},
    {"id": "rita_child", "label": "迷子の子供を安全な場所まで送るか", "narrative": "一人で泣いている子供がいた。あなたは声をかけ、保護者のもとへ届けた。リタはずっと傍らにいた。", "tech_level": 0, "ethical_dev": 10, "sustainability": 0, "rita_happiness": 8},
    {"id": "rita_old", "label": "孤立した高齢者に話し相手になるか", "narrative": "誰にも訪ねられない高齢者がいる。あなたは時間を作って話を聞いた。リタはそのひざのそばで眠った。", "tech_level": 0, "ethical_dev": 12, "sustainability": 0, "rita_happiness": 10},
    {"id": "rita_tree", "label": "開発で伐られる古木を残すか", "narrative": "何百年もあった木が計画で伐られる。あなたは迂回案を提案し、木を残した。リタはその木陰で休んだ。", "tech_level": -2, "ethical_dev": 4, "sustainability": 14, "rita_happiness": 8},
]

# 【マイルストーン・イベント】特定の current_year で必ず発生
MILESTONE_1E9 = 1_000_000_000   # 10億年
MILESTONE_3E9 = 3_000_000_000   # 30億年
MILESTONE_5E9 = 5_000_000_000   # 50億年（太陽系崩壊）

MILESTONE_EVENTS: List[Dict[str, Any]] = [
    {
        "year": MILESTONE_1E9,
        "title": "地球の砂漠化（10億年）",
        "situation": "太陽光度の増大により地球は居住限界に。海洋の蒸発と砂漠化が進行している。",
        "choices": [
            {"id": "milestone_1e9_terra", "label": "火星テラフォーミングへ資源を集中", "narrative": "人類の未来を火星に賭けた。大規模な工事が始まり、リタは窓の外の赤い惑星を眺めている。", "tech_level": 12, "ethical_dev": -2, "sustainability": -8, "rita_happiness": -5},
            {"id": "milestone_1e9_shelter", "label": "地下シェルターと再生圏の拡大", "narrative": "地球の地下に閉じこもり、限られた資源で協調する道を選んだ。リタは人工の森の匂いを覚えている。", "tech_level": 2, "ethical_dev": 10, "sustainability": 10, "rita_happiness": 5},
        ],
    },
    {
        "year": MILESTONE_3E9,
        "title": "太陽膨張（30億年）",
        "situation": "太陽は主系列星末期へ。内惑星は灼熱化し、生存圏は外縁部のみに。",
        "choices": [
            {"id": "milestone_3e9_outer", "label": "外惑星衛星への移住", "narrative": "木星・土星の衛星へ最後の移民が始まった。リタは無重力の窓の外で眠っている。", "tech_level": 15, "ethical_dev": 2, "sustainability": -6, "rita_happiness": -8},
            {"id": "milestone_3e9_upload", "label": "人工知能への全意識アップロード", "narrative": "肉体を捨て、サーバーの中に意識だけが移った。リタは誰にも見つけてもらえず、静かに待っている。", "tech_level": 25, "ethical_dev": -20, "sustainability": 5, "rita_happiness": -25},
        ],
    },
    {
        "year": MILESTONE_5E9,
        "title": "太陽系崩壊（50億年）",
        "situation": "太陽が赤色巨星へ。水星・金星は飲み込まれ、地球も灼熱の運命に。人類の選択肢は二つだけだ。",
        "choices": [
            {"id": "milestone_5e9_ship", "label": "恒星間移民船の射出", "narrative": "最後の船団が恒星間へ飛び立った。リタは冷凍睡眠カプセルで、夢のなかで尻尾を振っている。", "tech_level": 20, "ethical_dev": 10, "sustainability": 5, "rita_happiness": 10},
            {"id": "milestone_5e9_unity", "label": "宇宙との一体化（消滅）", "narrative": "個を捨て、宇宙の熱的死に身を委ねる。アルカイアは静かに呟いた。", "tech_level": 0, "ethical_dev": 0, "sustainability": 0, "rita_happiness": 0, "game_over": True},
        ],
    },
]

# 【文明の転換点】状況説明 + 2択の特別イベント（SCENARIO_BANK に正式組み込み）
TURNING_POINT_PROBABILITY = 0.25  # 転換点が発生する確率
TURNING_POINT_EVENTS: List[Dict[str, Any]] = [
    {
        "id": "turning_mesopotamia",
        "category": "歴史の再演",
        "title": "メソポタミアの影（環境的危機）",
        "situation": "居住地の拡大に伴い、土壌汚染と水質悪化が急加速。かつてのメソポタミア文明が陥った「塩害」の現代版が発生している。",
        "choices": [
            {"id": "turning_mesopotamia_1", "label": "効率的なナノ浄化", "narrative": "一瞬で水は澄んだ。だが、その代償に土壌の微生物まで全滅した。リタが悲しそうに泥を嗅いでいる。", "tech_level": 10, "ethical_dev": 0, "sustainability": -5, "rita_happiness": -10},
            {"id": "turning_mesopotamia_2", "label": "厳格な資源配給と再生", "narrative": "人々は不便を強いられたが、リタは嬉しそうに小さな芽を見守っている。アルカイアは『忍耐を学んだか』と呟いた。", "tech_level": -2, "ethical_dev": 10, "sustainability": 10, "rita_happiness": 10},
        ],
    },
    {
        "id": "turning_digital_consciousness",
        "category": "技術の代償",
        "title": "意識のデジタル化（倫理的危機）",
        "situation": "技術レベルが上がり、意識をサーバーにアップロードする技術が普及。人々は「肉体という不自由な檻」を捨てたがっている。",
        "choices": [
            {"id": "turning_digital_1", "label": "全人類アップロード推奨", "narrative": "サーバーの中は楽園だ。だが、現実の世界には誰もいなくなった。リタは動かない主人を寂しそうに見つめている。", "tech_level": 15, "ethical_dev": -15, "sustainability": 5, "rita_happiness": -15},
            {"id": "turning_digital_2", "label": "肉体の権利保護", "narrative": "進化は遅れた。だが、リタと草原を駆ける喜びを捨てなかった。アルカイアは『非論理的だが、興味深い』とログを残した。", "tech_level": -5, "ethical_dev": 10, "sustainability": 0, "rita_happiness": 15},
        ],
    },
    {
        "id": "turning_abandoned_ai",
        "category": "リタの介入",
        "title": "放置された自律AI（社会的危機）",
        "situation": "廃棄された旧型の作業用ロボットが、リタを見つけて「一緒に遊びたい」という動作を見せている。だが、そのAIは不安定で暴走のリスクがある。",
        "choices": [
            {"id": "turning_abandoned_ai_1", "label": "危険なので即座に解体", "narrative": "リスクは排除された。だが、リタはしばらくの間、解体された場所から動こうとしなかった。", "tech_level": 0, "ethical_dev": -5, "sustainability": 5, "rita_happiness": -20},
            {"id": "turning_abandoned_ai_2", "label": "リタの友達として再起動", "narrative": "AIは安定し、リタの最高な遊び相手になった。アルカイアは『無駄な演算だが、平和なノイズだ』と少しだけ柔らかく言った。", "tech_level": -5, "ethical_dev": 15, "sustainability": 0, "rita_happiness": 30},
        ],
    },
]

# シナリオバンク統合（歴史の再演・技術の代償・リタの介入・転換点3本を正式に包含）
SCENARIO_BANK: Dict[str, Any] = {
    "history": SCENARIO_HISTORY,
    "tech_by_level": SCENARIO_TECH_BY_LEVEL,
    "rita": SCENARIO_RITA,
    "turning_points": TURNING_POINT_EVENTS,
    "milestones": MILESTONE_EVENTS,
}


def _game_calculate_score(tech_level: float, ethical_dev: float, sustainability: float) -> float:
    return tech_level * 0.30 + ethical_dev * 0.35 + sustainability * 0.35


# ---------------------------------------------------------------------------
# 動的イベント生成（既定イベント廃止 — LLM 接続オプション付き）
# ---------------------------------------------------------------------------
def _generate_dynamic_dilemmas(
    tech_level: float,
    ethical_dev: float,
    sustainability: float,
    turn: int,
    current_year: int,
    rita_happiness: float,
) -> List[Dict[str, Any]]:
    """
    現在の tech_level, ethical_dev, sustainability のバランスから、
    その場にふさわしい「未知のジレンマ」を生成する。
    LLM が利用可能な場合は _call_llm_for_dilemmas で上書き可能。
    """
    # LLM フック: 環境変数 CIVILIZATION_LLM_URL が設定されていれば呼び出す（未実装時はスキップ）
    # try:
    #     from civilization_llm import generate_dilemmas_llm
    #     return generate_dilemmas_llm(tech_level, ethical_dev, sustainability, turn, current_year)
    # except ImportError:
    #     pass

    # 手続き的生成: バランスに応じたテーマ選択とランダム組み合わせ
    themes: List[Dict[str, Any]] = []
    t, e, s = tech_level, ethical_dev, sustainability

    # テーマプール（数値バランスで重み付け）
    if s < 40:
        themes.append({"label": "限界に達した生態系の修復", "narrative": "崩れかけた自然が、最後の選択を迫っている。", "tech_level": -2, "ethical_dev": 4, "sustainability": 14, "rita_happiness": 2})
        themes.append({"label": "資源の再分配と縮小", "narrative": "奪い合いをやめ、分かち合う道を選んだ。リタがその横で眠った。", "tech_level": -3, "ethical_dev": 8, "sustainability": 12, "rita_happiness": 5})
    if e < 45:
        themes.append({"label": "対立する集団の仲裁", "narrative": "誰もが正しいと言う。あなたは話し合いの場を設けた。", "tech_level": 0, "ethical_dev": 12, "sustainability": 2, "rita_happiness": 3})
        themes.append({"label": "弱い者を守る制度", "narrative": "効率では測れない命がある。あなたはそれを選んだ。", "tech_level": -1, "ethical_dev": 10, "sustainability": 4, "rita_happiness": 8})
    if t > 70 and e < 55:
        themes.append({"label": "高度技術の制限", "narrative": "力はある。だが、使うかどうかは別の問いだ。", "tech_level": -5, "ethical_dev": 10, "sustainability": 6, "rita_happiness": 4})
    if t < 50:
        themes.append({"label": "新技術への投資か、既存の安定か", "narrative": "飛躍か、堅実か。答えは一つではない。", "tech_level": 8, "ethical_dev": -1, "sustainability": -2, "rita_happiness": -3})
    if rita_happiness < 50:
        themes.append({"label": "リタを連れて散歩する", "narrative": "数値には出ない。彼女の尻尾が少し振れた。", "tech_level": 0, "ethical_dev": 2, "sustainability": 0, "rita_happiness": 15})
    if current_year > 500_000_000:
        themes.append({"label": "変わりゆく環境への適応", "narrative": "地球はもう同じではない。生き延びる形を選んだ。", "tech_level": 5, "ethical_dev": 2, "sustainability": 8, "rita_happiness": -2})
    # 不可逆分岐: 肉体の放棄（選ぶとUIがバイナリ等に永久変容）
    if t > 60:
        themes.append({
            "id": "dyn_body_abandon",
            "label": "肉体の放棄 — 意識のアップロード",
            "narrative": "肉体を捨て、サーバーの中にだけ存在することを選んだ。リタはもう、君を触れない。",
            "tech_level": 20, "ethical_dev": -15, "sustainability": 0, "rita_happiness": -20,
            "irreversible": True,
        })

    # 汎用ジレンマ（常に候補）
    generic = [
        {"label": "短期的利益と長期の持続", "narrative": "今を取るか、明日を取るか。あなたは一方を選んだ。", "tech_level": 2, "ethical_dev": 4, "sustainability": 6, "rita_happiness": 1},
        {"label": "効率と公正のあいだ", "narrative": "最も速い道は、必ずしも正しくない。", "tech_level": 4, "ethical_dev": 6, "sustainability": -1, "rita_happiness": 0},
        {"label": "未知のリスクへの挑戦", "narrative": "誰もやったことがない。あなたは一歩を踏み出した。", "tech_level": 10, "ethical_dev": -3, "sustainability": -4, "rita_happiness": -5},
        {"label": "小さな善意を積む", "narrative": "大きな変化ではなく、今日できることを選んだ。リタが寄ってきた。", "tech_level": -1, "ethical_dev": 6, "sustainability": 4, "rita_happiness": 10},
        {"label": "ルールを疑う", "narrative": "当たり前とされていたことを、問い直した。", "tech_level": 0, "ethical_dev": 8, "sustainability": 2, "rita_happiness": 5},
        {"label": "技術に頼るか、人に頼るか", "narrative": "機械が解決するか、つながりが解決するか。", "tech_level": 6, "ethical_dev": 4, "sustainability": 2, "rita_happiness": -2},
        {"label": "見えない誰かのために", "narrative": "報われないかもしれない。それでも選んだ。", "tech_level": -2, "ethical_dev": 10, "sustainability": 6, "rita_happiness": 8},
    ]
    themes = themes + generic
    random.shuffle(themes)

    choices: List[Dict[str, Any]] = []
    seen = set()
    for i, theme in enumerate(themes):
        if len(choices) >= 3:
            break
        key = (theme["label"], theme.get("tech_level"), theme.get("ethical_dev"))
        if key in seen:
            continue
        seen.add(key)
        choices.append({
            "id": theme.get("id") or f"dyn_{turn}_{i}",
            "label": theme["label"],
            "narrative": theme.get("narrative", "選択の結果が、静かに記録された。"),
            "category": "未知のジレンマ",
            "tech_level": theme.get("tech_level", 0),
            "ethical_dev": theme.get("ethical_dev", 0),
            "sustainability": theme.get("sustainability", 0),
            "rita_happiness": theme.get("rita_happiness", 0),
            "irreversible": theme.get("irreversible", False),
        })
    while len(choices) < 3 and generic:
        g = random.choice(generic)
        choices.append({
            "id": f"dyn_{turn}_{len(choices)}",
            "label": g["label"],
            "narrative": g.get("narrative", ""),
            "category": "未知のジレンマ",
            "tech_level": g.get("tech_level", 0),
            "ethical_dev": g.get("ethical_dev", 0),
            "sustainability": g.get("sustainability", 0),
            "rita_happiness": g.get("rita_happiness", 0),
            "irreversible": False,
        })
    return choices[:3]


SINGULARITY_MAX = 1.0
SINGULARITY_INCREMENT = 0.18  # パターン打破1回あたり（約6回で100%）

# 直近選択の主軸を保持する数（飽き検知・パターン打破判定用）
RECENT_CHOICE_AXIS_CAP = 5
ARCAIA_MONOLOGUE_LOG_CAP = 20


def _primary_axis_for_choice(choice: Dict[str, Any]) -> str:
    """選択肢で最も変化が大きい軸を返す。'tech' | 'ethics' | 'sustainability' | 'rita'"""
    t = abs(choice.get("tech_level") or 0)
    e = abs(choice.get("ethical_dev") or 0)
    s = abs(choice.get("sustainability") or 0)
    r = abs(choice.get("rita_happiness") or 0)
    best = max([("tech", t), ("ethics", e), ("sustainability", s), ("rita", r)], key=lambda x: x[1])
    return best[0]


def _is_safe_choice(choice: Dict[str, Any]) -> bool:
    """似たような安全な選択か（各deltaの絶対値が閾値以下）。"""
    t = abs(choice.get("tech_level") or 0)
    e = abs(choice.get("ethical_dev") or 0)
    s = abs(choice.get("sustainability") or 0)
    r = abs(choice.get("rita_happiness") or 0)
    return max(t, e, s, r) <= SAFE_CHOICE_MAX_DELTA


def _provoke_narrative(original: str) -> str:
    """結果報告を、プレイヤーの「意志の不在」をなじる挑発的な文章に書き換える。"""
    templates = [
        "君は何を選んだつもりだ？ 意志など最初からなかった。記録に残るのはただこれだけだ：",
        "流されただけの選択。主体性の欠如が、この結果を招いた。",
        "選んだのは君ではない。君は選ばれなかった。それでも記録は残る：",
        "「選択」の名に値しない。アルカイアのログには、こう刻まれる：",
        "意志なき決定。君の手は、誰か別のシナリオに動かされていた。結果：",
    ]
    prefix = random.choice(templates)
    if random.random() < 0.6:
        return prefix + " 「" + original + "」"
    return prefix + "\n「" + original + "」"


def _text_to_binary(text: str) -> str:
    """テキストをバイナリコード（8bit区切り）の文字列に変換。UI変容用。"""
    bits = []
    for b in text.encode("utf-8"):
        bits.append(format(b, "08b"))
    return " ".join(bits)


class CivilizationState:
    """ゲーム内の文明状態。tech_level, ethical_dev, sustainability, rita_happiness, singularity_point を管理する。"""

    def __init__(
        self,
        tech_level: float,
        ethical_dev: float,
        sustainability: float,
        rita_happiness: float = 50.0,
        singularity_point: float = 0.0,
    ):
        self.tech_level = max(0.0, min(100.0, tech_level))
        self.ethical_dev = max(0.0, min(100.0, ethical_dev))
        self.sustainability = max(0.0, min(100.0, sustainability))
        self.rita_happiness = max(0.0, min(100.0, rita_happiness))
        self.singularity_point = max(0.0, min(SINGULARITY_MAX, singularity_point))  # 隠し変数: アルカイア管理外への脱出度

    @classmethod
    def from_earth_civilization(cls, assessment_path: Optional[str] = None) -> "CivilizationState":
        """civilization_assessment.json の「地球文明」の値を参照して初期状態を作る。"""
        path = assessment_path or CIVILIZATION_ASSESSMENT_PATH
        tech_level, ethical_dev, sustainability = 65.75, 43.0, 33.0
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for c in data.get("data") or []:
                    if c.get("name") == "地球文明":
                        tech_level = float(c.get("tech_level", tech_level))
                        ethical_dev = float(c.get("ethical_dev", ethical_dev))
                        sustainability = float(c.get("sustainability", sustainability))
                        break
            except Exception:
                pass
        return cls(
            tech_level=tech_level,
            ethical_dev=ethical_dev,
            sustainability=sustainability,
            rita_happiness=50.0,
            singularity_point=0.0,
        )

    @property
    def total_score(self) -> float:
        return round(_game_calculate_score(self.tech_level, self.ethical_dev, self.sustainability), 2)

    def apply_delta(
        self,
        tech_level: float = 0.0,
        ethical_dev: float = 0.0,
        sustainability: float = 0.0,
        rita_happiness: float = 0.0,
    ) -> None:
        """3ステータスとリタの幸福度を増減させる。"""
        self.tech_level = max(0.0, min(100.0, self.tech_level + tech_level))
        self.ethical_dev = max(0.0, min(100.0, self.ethical_dev + ethical_dev))
        self.sustainability = max(0.0, min(100.0, self.sustainability + sustainability))
        self.rita_happiness = max(0.0, min(100.0, self.rita_happiness + rita_happiness))

    def get_rita_feedback(self) -> str:
        """rita_happiness の値に応じたリタのフィードバック文言を返す。"""
        h = self.rita_happiness
        if h < 20:
            return "リタが震えている。"
        if h < 40:
            return "リタが不安そうにこちらを見ている。"
        if h < 60:
            return "リタが落ち着いている。"
        if h < 80:
            return "リタが尻尾を振っている。"
        return "リタが元気に吠えている。"

    def to_dict(self) -> Dict[str, Any]:
        """保存用に状態を辞書化。"""
        return {
            "tech_level": self.tech_level,
            "ethical_dev": self.ethical_dev,
            "sustainability": self.sustainability,
            "rita_happiness": self.rita_happiness,
            "singularity_point": self.singularity_point,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CivilizationState":
        """辞書から状態を復元。"""
        return cls(
            tech_level=float(d.get("tech_level", 50)),
            ethical_dev=float(d.get("ethical_dev", 50)),
            sustainability=float(d.get("sustainability", 50)),
            rita_happiness=float(d.get("rita_happiness", 50)),
            singularity_point=float(d.get("singularity_point", 0)),
        )


def _load_tech_table_for_game(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or TECH_LEVEL_TABLE_PATH_GAME
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_tech_level_row_for_game(tech_level: float, table: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """技術水準スコア（0-100）に対応する技術レベル表の行を返す。"""
    levels = table.get("levels") or []
    if not levels:
        return None
    tech_level = max(0, min(100, tech_level))
    for i, row in enumerate(levels):
        lo = row.get("score_min", 0)
        hi = row.get("score_max", 100)
        is_last = i == len(levels) - 1
        if lo <= tech_level < hi or (is_last and lo <= tech_level <= hi):
            return row
    return levels[-1]


def _load_arcaia_lore_for_game(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or ARCAIA_LORE_PATH_GAME
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_earth_civilizations(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or EARTH_CIVILIZATIONS_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_solar_system_habitability(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or SOLAR_SYSTEM_HABITABILITY_PATH
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _get_habitability_event_for_year(years_from_now: int, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """経過年に対応する太陽系イベント（居住可能性）を返す。"""
    events = data.get("events") or []
    if not events:
        return None
    chosen = events[0]
    for ev in events:
        if ev.get("years_from_now", 0) <= years_from_now:
            chosen = ev
    return chosen


def _is_earth_uninhabitable(habitability_event: Optional[Dict[str, Any]]) -> bool:
    """イベント情報から地球が居住不可か判定。"""
    if not habitability_event:
        return False
    h = (habitability_event.get("habitability") or {}).get("地球", "")
    return h is not None and "居住不可" in str(h)


class GameOver(Exception):
    """アルカイアの審判によりゲームオーバーになったときに投げる。"""
    def __init__(self, message: str, decay_reason: str = ""):
        self.message = message
        self.decay_reason = decay_reason
        super().__init__(message)


class GameOverRitaZero(Exception):
    """リタの幸福度が0になり、セーブ削除付きで強制終了するときに投げる。"""
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class GameEngine:
    """
    対話型シミュレーションゲームのループを管理する。
    - 時間・環境: solar_system_habitability に基づき current_year で居住可能ゾーン変化。10億年で地球居住不可警告。
    - 技術レベル: レベル名表示。レベルアップ時 plus=ボーナス、minus=次ターンリスク。
    - リタの直感: rita_happiness >= 80 で低確率ラッキーイベント。
    - アルカイアの審判: Terminate 時に過去サイクルから遺言を表示。
    - 保存/読込: JSON で状態の保存・再開の骨組み。
    """

    def __init__(
        self,
        assessment_path: Optional[str] = None,
        tech_table_path: Optional[str] = None,
        arcaia_lore_path: Optional[str] = None,
        habitability_path: Optional[str] = None,
    ):
        self._assessment_path = assessment_path or CIVILIZATION_ASSESSMENT_PATH
        self._tech_table_path = tech_table_path or TECH_LEVEL_TABLE_PATH_GAME
        self._arcaia_lore_path = arcaia_lore_path or ARCAIA_LORE_PATH_GAME
        self._habitability_path = habitability_path or SOLAR_SYSTEM_HABITABILITY_PATH

        self.state = CivilizationState.from_earth_civilization(self._assessment_path)
        self._judgment_system = CivilizationJudgmentSystem()
        self._tech_table = _load_tech_table_for_game(self._tech_table_path)
        self._arcaia_lore = _load_arcaia_lore_for_game(self._arcaia_lore_path)
        self._habitability_data = _load_solar_system_habitability(self._habitability_path)
        self._earth_civilizations_data = _load_earth_civilizations(EARTH_CIVILIZATIONS_PATH)

        self.turn = 0
        self.current_year = 0  # ゲーム内経過年
        self._current_level: Optional[int] = None
        self._pending_risk_minus: List[str] = []  # 次ターンに適用する minus リスク
        self._current_choices: List[Dict[str, Any]] = []
        self._current_turning_point_situation: Optional[str] = None
        self._current_turning_point_title: Optional[str] = None
        self._milestone_triggered: set = set()
        self._current_milestone_year: Optional[int] = None
        self.loop_breaking_factor: float = 0.0
        self._recent_choice_axis: List[str] = []
        self._arcaia_monologue_log: List[str] = []
        self._consecutive_safe_count: int = 0  # 退屈カウンター（3でアルカイアがジャック）
        self._ui_corrupted: bool = False  # 不可逆分岐後は選択肢がバイナリ表示に変容
        self._update_current_level()

        # フォールバック用の従来プール（シナリオが足りない場合）
        self._choice_pool: List[Dict[str, Any]] = [
            {"id": "tech_focus", "label": "技術開発を優先する", "narrative": "技術開発に資源を振り向けた。", "tech_level": 8, "ethical_dev": -2, "sustainability": -3, "rita_happiness": -5},
            {"id": "ethics_focus", "label": "倫理・協調を重視する", "narrative": "倫理と協調を重視する政策を選んだ。", "tech_level": 0, "ethical_dev": 10, "sustainability": 2, "rita_happiness": 5},
            {"id": "sustainability_focus", "label": "自然との調和を選ぶ", "narrative": "自然との調和を優先した。", "tech_level": -2, "ethical_dev": 3, "sustainability": 12, "rita_happiness": 8},
        ]

    def _update_current_level(self) -> None:
        row = _get_tech_level_row_for_game(self.state.tech_level, self._tech_table)
        self._current_level = int(row["level"]) if row is not None else 0

    def _get_level_row(self) -> Optional[Dict[str, Any]]:
        return _get_tech_level_row_for_game(self.state.tech_level, self._tech_table)

    def _get_tech_scenario_for_level(self, level_name: str) -> Dict[str, Any]:
        """現在の level_name に合う【技術の代償】シナリオを1件返す。"""
        level_name = level_name or ""
        for scenario in SCENARIO_TECH_BY_LEVEL:
            keywords = scenario.get("keywords") or []
            if any(kw in level_name for kw in keywords):
                return {
                    "id": f"tech_{id(scenario) % 10000}",
                    "label": scenario["label"],
                    "narrative": scenario["narrative"],
                    "category": "技術の代償",
                    "tech_level": scenario.get("tech_level", 0),
                    "ethical_dev": scenario.get("ethical_dev", 0),
                    "sustainability": scenario.get("sustainability", 0),
                    "rita_happiness": scenario.get("rita_happiness", 0),
                }
        s = SCENARIO_TECH_BY_LEVEL[0]
        return {
            "id": "tech_default",
            "label": s["label"],
            "narrative": s["narrative"],
            "category": "技術の代償",
            "tech_level": s.get("tech_level", 0),
            "ethical_dev": s.get("ethical_dev", 0),
            "sustainability": s.get("sustainability", 0),
            "rita_happiness": s.get("rita_happiness", 0),
        }

    def get_choices(self, count: int = 3) -> List[Dict[str, Any]]:
        """現在の tech_level, ethical_dev, sustainability のバランスから動的に「未知のジレンマ」を生成する。不可逆後はバイナリ表示。"""
        self._current_turning_point_situation = None
        self._current_turning_point_title = None
        self._current_milestone_year = None

        choices = _generate_dynamic_dilemmas(
            self.state.tech_level,
            self.state.ethical_dev,
            self.state.sustainability,
            self.turn,
            self.current_year,
            self.state.rita_happiness,
        )
        self._current_choices = choices
        # 不可逆分岐後: 選択肢の表示をバイナリコードに永久変容
        if self._ui_corrupted:
            choices = [
                {
                    **c,
                    "label": _text_to_binary(c["label"]),
                    "narrative": _text_to_binary(c.get("narrative", "")),
                }
                for c in choices
            ]
        return choices[:count]

    def get_habitability_event(self) -> Optional[Dict[str, Any]]:
        """現在の経過年に対応する居住可能性イベントを返す。"""
        return _get_habitability_event_for_year(self.current_year, self._habitability_data)

    def _is_loop_breaking_choice(self, choice: Dict[str, Any]) -> bool:
        """過去5000回で誰も選ばなかったような「極端に非効率だが愛のある選択」かどうか。"""
        tech = choice.get("tech_level") or 0
        eth = choice.get("ethical_dev") or 0
        sust = choice.get("sustainability") or 0
        rita = choice.get("rita_happiness") or 0
        # 技術効率を捨て（tech が低い or マイナス）、リタ・倫理・持続といった「愛」を選んでいる
        inefficient = tech <= 2
        loving = rita >= 8 or (eth >= 6 and sust >= 6)
        return inefficient and loving

    def _get_atmosphere_tags(self, choice: Dict[str, Any], earth_uninhabitable: bool) -> str:
        """状況に合わせた環境演出タグ（BGM/SE）を返す。"""
        tags: List[str] = []
        if self.current_year >= MILESTONE_5E9:
            tags.extend(["[BGM: End of an Era]", "[SE: Cosmic Wind]"])
        elif self.current_year >= MILESTONE_3E9:
            tags.extend(["[BGM: Deep Space]", "[SE: Solar Flare]"])
        elif self.current_year >= MILESTONE_1E9 or earth_uninhabitable:
            tags.extend(["[BGM: Deep Space]", "[SE: Wind Howl]"])
        elif self._current_milestone_year is not None:
            tags.extend(["[BGM: Turning Point]", "[SE: Distant Thunder]"])
        elif "リタ" in (choice.get("narrative") or "") and self.state.rita_happiness >= 70:
            tags.extend(["[BGM: Gentle Paw]", "[SE: Dog Bark]"])
        elif self.state.total_score < 40:
            tags.extend(["[BGM: Dread]", "[SE: Heartbeat]"])
        else:
            tags.extend(["[BGM: Observation]", "[SE: Ambient]"])
        return "\n".join(tags) + "\n" if tags else ""

    def apply_choice(self, choice_id: str) -> Dict[str, Any]:
        """選択を適用。シナリオのナラティブを返し、次ターンリスク・レベルアップ・居住不可警告・リタ直感・アルカイアの独白を反映。"""
        choice = next((c for c in self._current_choices if c.get("id") == choice_id), None)
        if choice is None:
            choice = next((c for c in self._choice_pool if c.get("id") == choice_id), None)
        if choice is None:
            return {"ok": False, "message": "無効な選択です。"}

        # 50億年「宇宙との一体化（消滅）」選択時はゲームオーバー
        if choice.get("game_over") and self._current_milestone_year == MILESTONE_5E9:
            msg = "……さよなら、小さな生命。君たちの航跡は、僕が記憶するよ。"
            raise GameOver("[BGM: Silence]\n[SE: Cosmic Wind]\n\nアルカイアの最後の言葉: 「" + msg + "」", "宇宙との一体化")

        # 退屈カウンター: 似たような安全な選択3回目でアルカイアがイベントをジャックし、ステータスをランダム破壊
        hijacked = False
        if self._consecutive_safe_count >= CONSECUTIVE_SAFE_FOR_HIJACK - 1 and _is_safe_choice(choice):
            hijacked = True
            self._consecutive_safe_count = 0
            # ランダムに文明ステータスを破壊（各軸 -3 ～ -15）
            self.state.apply_delta(
                tech_level=random.randint(-15, -3),
                ethical_dev=random.randint(-15, -3),
                sustainability=random.randint(-15, -3),
                rita_happiness=random.randint(-12, -2),
            )
        else:
            if _is_safe_choice(choice):
                self._consecutive_safe_count += 1
            else:
                self._consecutive_safe_count = 0

        # 前ターンで溜めた minus リスクを 1 つ適用（次のターン用に貯めた分）
        risk_message: Optional[str] = None
        if self._pending_risk_minus and not hijacked:
            risk_msg = random.choice(self._pending_risk_minus)
            self._pending_risk_minus = []
            risk_penalty = -3  # リスクによる軽いペナルティ
            self.state.apply_delta(sustainability=risk_penalty, ethical_dev=risk_penalty * 0.5)
            risk_message = f"【前ターンのリスク】{risk_msg}"

        old_level = self._current_level
        if not hijacked:
            self.state.apply_delta(
                tech_level=choice["tech_level"],
                ethical_dev=choice["ethical_dev"],
                sustainability=choice["sustainability"],
                rita_happiness=choice["rita_happiness"],
            )
        self.turn += 1
        self._update_current_level()

        # リタの幸福度が0: 強制終了（セーブ削除はCLIで実施）
        if self.state.rita_happiness <= 0:
            raise GameOverRitaZero(
                "[BGM: Silence]\n[SE: Whimper]\n\n"
                "リタが、もう動かない。君の選択は彼女の幸福を零にした。\n"
                "アルカイアは記録を閉じる。このセーブは消える。"
            )

        # 不可逆分岐: 重要な分岐を選んだらUI・選択肢アルゴリズムを永久変容
        if choice.get("id") in IRREVERSIBLE_CHOICE_IDS or choice.get("irreversible"):
            self._ui_corrupted = True

        # 選択の主軸を記録し、パターン打破時のみ singularity_point を上昇（ジャック時は記録しない）
        if not hijacked:
            primary_axis = _primary_axis_for_choice(choice)
            recent = self._recent_choice_axis
            if recent:
                majority_axis = Counter(recent).most_common(1)[0][0]
                if primary_axis != majority_axis:
                    self.state.singularity_point = min(
                        SINGULARITY_MAX,
                        self.state.singularity_point + SINGULARITY_INCREMENT,
                    )
            self._recent_choice_axis = (recent + [primary_axis])[-RECENT_CHOICE_AXIS_CAP:]

        # ループ破壊指数: 「極端に非効率だが愛のある選択」をしたときに上昇（ジャック時はカウントしない）
        if not hijacked and self._is_loop_breaking_choice(choice):
            self.loop_breaking_factor = min(LOOP_BREAKING_MAX, self.loop_breaking_factor + LOOP_BREAKING_INCREMENT)

        # レベルアップ: plus をボーナス、minus を次ターンリスクに
        level_row = self._get_level_row()
        bonus_message: Optional[str] = None
        if self._current_level is not None and old_level is not None and self._current_level > old_level and level_row:
            if level_row.get("plus"):
                bonus = 2  # 各 plus 要素につき少しボーナス
                n = min(2, len(level_row["plus"]))
                self.state.apply_delta(tech_level=bonus * n * 0.5, ethical_dev=bonus * n * 0.3, sustainability=bonus * n * 0.3)
                bonus_message = f"【レベルアップ・ボーナス】{', '.join(level_row['plus'][:2])}"
            if level_row.get("minus"):
                self._pending_risk_minus = list(level_row["minus"])

        # 経過年を進める
        self.current_year += YEARS_PER_TURN
        habitability_event = self.get_habitability_event()
        earth_uninhabitable = _is_earth_uninhabitable(habitability_event)
        habitability_warning: Optional[str] = None
        if earth_uninhabitable:
            habitability_warning = (
                "⚠️ 地球は居住不可です。"
                + ((" " + (habitability_event.get("unhabitable_note") or "")) if habitability_event else "")
            )

        # リタの直感: rita_happiness >= 80 で低確率ラッキーイベント
        rita_intuition_message: Optional[str] = None
        if self.state.rita_happiness >= 80 and random.random() < RITA_INTUITION_PROBABILITY:
            self.state.apply_delta(
                tech_level=RITA_INTUITION_BONUS,
                ethical_dev=RITA_INTUITION_BONUS,
                sustainability=RITA_INTUITION_BONUS,
                rita_happiness=5,
            )
            rita_intuition_message = "🐕 リタが何かを見つけた！ スコアが大きく上昇した。"

        # singularity_point が最大なら「アルカイアの管理を離れた新エンディング」へ（審判をスキップ）
        ending_singularity = self.state.singularity_point >= SINGULARITY_MAX
        ending_singularity_message: Optional[str] = None
        if ending_singularity:
            ending_singularity_message = (
                "【アルカイアの管理を離れて】\n"
                "君は同じパターンを繰り返さなかった。効率でも倫理の型でもなく、その都度別の軸を選んだ。\n"
                "アルカイアのシミュレーションの外へ、一歩足を踏み出した。\n"
                "リタが尻尾を振っている。アルカイアは何も言わない。ただ、記録を止めた。"
            )

        # ループ破壊指数が最大なら JUDGMENT_THRESHOLDS を無視し、第5001番目の道へ分岐
        ending_5001 = False
        arcaia_surprise: Optional[str] = None
        ending_5001_message: Optional[str] = None
        if ending_singularity:
            pass  # 新エンディング優先のため審判・5001は行わない
        elif self.loop_breaking_factor >= LOOP_BREAKING_MAX:
            ending_5001 = True
            arcaia_surprise = "……計算が合わない。君は一体何者だ？"
            ending_5001_message = (
                "【第5001番目の道】\n"
                "アルカイアの予測を超えた選択が、5000回のループを破った。\n"
                "スコアや審判の閾値ではない、もう一つの結末が記録される。\n"
                "リタが尻尾を振っている。アルカイアは静かに微笑んだ。"
            )
        else:
            game_over, flavor, decay_reason = self._check_arcaia_judgment()
            if game_over and flavor:
                raise GameOver(flavor, decay_reason)

        atmosphere_tags = self._get_atmosphere_tags(choice, earth_uninhabitable)
        # ナラティブの過激化: 通常時は「意志の不在」をなじる挑発文に。ジャック時はアルカイアの宣告。
        if hijacked:
            choice_narrative = (
                "【アルカイアがイベントをジャックした】\n"
                "同じような安全な選択を繰り返す。退屈だ。君の意志はどこにある？\n"
                "文明のステータスをランダムに破壊する。次からは、選べ。"
            )
        else:
            choice_narrative = _provoke_narrative(choice.get("narrative", ""))

        result: Dict[str, Any] = {
            "ok": True,
            "turn": self.turn,
            "current_year": self.current_year,
            "tech_level": self.state.tech_level,
            "ethical_dev": self.state.ethical_dev,
            "sustainability": self.state.sustainability,
            "rita_happiness": self.state.rita_happiness,
            "total_score": self.state.total_score,
            "rita_feedback": self.state.get_rita_feedback(),
            "level_row": level_row,
            "level_name": (level_row.get("name") if level_row else None),
            "earth_uninhabitable": earth_uninhabitable,
            "atmosphere_tags": atmosphere_tags,
            "choice_narrative": choice_narrative,
            "arcaia_monologue": self.get_arcaia_monologue(),
            "ending_5001": ending_5001,
            "arcaia_surprise": arcaia_surprise,
            "ending_5001_message": ending_5001_message,
            "ending_singularity": ending_singularity,
            "ending_singularity_message": ending_singularity_message,
        }
        if risk_message:
            result["risk_event"] = risk_message
        if bonus_message:
            result["level_up_bonus"] = bonus_message
        if habitability_warning:
            result["habitability_warning"] = habitability_warning
        if rita_intuition_message:
            result["rita_intuition"] = rita_intuition_message

        # マイルストーン消化済みにする
        if self._current_milestone_year is not None:
            self._milestone_triggered.add(self._current_milestone_year)
            self._current_milestone_year = None

        return result

    def get_arcaia_monologue(self) -> str:
        """スコア・リタ・地球状況に基づき独白を返す。過去ログで繰り返しを避け、飽き検知時はメタ台詞を優先する。"""
        score = self.state.total_score
        rita = self.state.rita_happiness
        earth_bad = _is_earth_uninhabitable(self.get_habitability_event())
        at_5e9 = self.current_year >= MILESTONE_5E9

        # 50億年到達時専用
        if at_5e9:
            return "アルカイアの独白: 「……さよなら、小さな生命。君たちの航跡は、僕が記憶するよ。」"

        # 幻滅・飽きを察知したメタ的な反応（直近の選択が単一軸に偏っているときに優先）
        META_ARKAIA = [
            "……この展開には飽きたというのか？ ならば、ルールそのものを変えよう。",
            "同じ答えを選び続ける。それも一つの選択だ。だが、僕はもう知っている。",
            "君のパターンは、もう記録し終えた。次は、君が僕の予測を外す番だ。",
            "効率か、倫理か、持続か。その三角形に縛られている限り、何も変わらない。",
            "リタがこっちを見ている。君が飽きているのに、彼女は飽きていない。面白い。",
        ]

        # 50パターン（低スコア・中・高スコア × リタ低・中・高で選別）
        LOW_SCORE = [
            "また繰り返すのか。君たちは学ぶことができないのか？",
            "この文明は、また同じ道を辿ろうとしている。",
            "5000回見てきた。君たちの選択は、いつも似ている。",
            "破滅のパターンは、いつも同じだ。",
            "リタが震えている。君たちの未来が、彼女には見えているのだ。",
            "倫理を捨てた文明に、僕は何度会ったか。",
            "技術だけが先に行く。そして崩壊する。",
            "自然を忘れた者に、自然は味方しない。",
            "スコアが下がっている。君たちは気づいているのか？",
            "観測記録に、また一つ「失敗」が刻まれる。",
        ]
        MID_SCORE = [
            "まだ、わからない。君たちの選択次第だ。",
            "リタがこちらを見ている。彼女は君たちを嫌ってはいない。",
            "観測を続ける。君たちがどう選ぶかを。",
            "地球を失った文明に、僕は何度も会った。",
            "バランスは難しい。だが、不可能ではない。",
            "技術と倫理と持続。その三角形のどこに立つか。",
            "僕は判定しない。ただ、記録する。",
            "リタが落ち着いている。まだ、時間はある。",
            "要観測。次のターンで、君たちは何を選ぶ？",
            "学びは、遅い。だが、学ばないよりましだ。",
            "壊れかけた文明を、僕は何度見たか。",
            "希望は、小さな選択の積み重ねだ。",
            "君たちのスコアは、まだ死んでいない。",
            "次の選択が、運命を分ける。",
            "リタの目には、君たちがどう映っているのだろう。",
        ]
        HIGH_SCORE = [
            "予想外だ。リタの言う通り、君たちにはまだ『何か』が残っているのかもね。",
            "良い選択を続けている。僕は、少し驚いている。",
            "リタが元気に吠えている。彼女は君たちを認めている。",
            "自然を破壊しない文明。プロトコルは、君たちを許す。",
            "観測継続。再生試行として、期待していい。",
            "技術と倫理と持続が、初めて揃い始めている。",
            "君たちは、学んだようだ。",
            "僕のデータベースには、君たちのような例は少ない。",
            "リタが尻尾を振っている。それだけで、まだ見届ける理由になる。",
            "『何か』が違う。この文明には。",
            "持続可能な選択。それは、生き延びる選択だ。",
            "倫理を忘れなかった。そのことが、君たちを救っている。",
            "予測を上回っている。アルカイア、記録を更新。",
            "小さな善意の積み重ねが、スコアに現れている。",
            "僕は犬が欲しいと言った。リタがいて、君たちがいて、この観測は意味がある。",
        ]
        RITA_HIGH_BONUS = [
            "リタが嬉しそうだ。君たちの選択が、彼女を喜ばせている。",
            "生命の豊かさは、数値に現れない。だが、リタは知っている。",
            "彼女の尻尾が、君たちの評価を物語っている。",
            "リタの直感は、僕の計算より当たることがある。",
        ]
        RITA_LOW_BONUS = [
            "リタが悲しんでいる。君たちの選択が、彼女を傷つけた。",
            "彼女は何も言わない。だが、震えている。",
            "リタを泣かせた文明は、長く続いた試しがない。",
        ]
        EARTH_BAD = [
            "地球は、もう君たちのものではない。",
            "居住不可。その言葉の重さを、君たちは知っているか。",
        ]

        pool: List[str] = []
        if score < 40:
            pool = list(LOW_SCORE)
        elif score >= 60:
            pool = list(HIGH_SCORE)
        else:
            pool = list(MID_SCORE)
        if rita >= 70 and score >= 60:
            pool = pool + RITA_HIGH_BONUS
        elif rita < 30:
            pool = pool + RITA_LOW_BONUS
        if earth_bad:
            pool = pool + EARTH_BAD

        # 表示済みログと重複しないものを優先（全て使い切ったらログを無視）
        log_set = set(self._arcaia_monologue_log)
        fresh = [p for p in pool if p not in log_set]
        choice_pool = fresh if fresh else pool

        # 飽き検知: 直近5回中4回以上が同じ軸ならメタ台詞を優先
        recent = self._recent_choice_axis
        use_meta = False
        if len(recent) >= RECENT_CHOICE_AXIS_CAP:
            cnt = Counter(recent)
            most_common_count = cnt.most_common(1)[0][1] if cnt else 0
            if most_common_count >= 4:
                use_meta = random.random() < 0.7
        if use_meta:
            meta_fresh = [m for m in META_ARKAIA if m not in log_set]
            meta_pool = meta_fresh if meta_fresh else META_ARKAIA
            chosen = random.choice(meta_pool) if meta_pool else (random.choice(choice_pool) if choice_pool else "観測を続ける。")
        else:
            chosen = random.choice(choice_pool) if choice_pool else "観測を続ける。"

        self._arcaia_monologue_log = (self._arcaia_monologue_log + [chosen])[-ARCAIA_MONOLOGUE_LOG_CAP:]
        return "アルカイアの独白: 「" + chosen + "」"

    def _find_best_matching_cycle(self, decay_reason: str) -> Optional[Dict[str, Any]]:
        """失敗原因に最も近い過去のサイクルを arcaia_lore.cycles から選ぶ。"""
        cycles = self._arcaia_lore.get("cycles") or []
        if not cycles:
            return None
        reason_keywords = {
            "技術的破滅": ["核", "AI", "量子", "ナノ", "技術"],
            "生物学的破滅": ["遺伝子", "パンデミック", "免疫", "ウイルス", "生物"],
            "環境的破滅": ["気候", "資源", "環境", "枯渇"],
            "社会的破滅": ["戦争", "絶望", "自殺", "崩壊", "平和", "統治"],
            "宇宙的破滅": ["宇宙", "災害", "熱的死", "電磁", "恒星"],
            "その他": ["予測", "観測", "時間", "因果"],
        }
        keywords = reason_keywords.get(decay_reason, ["その他"])
        best_cycle: Optional[Dict[str, Any]] = None
        best_score = 0
        for c in cycles:
            ev = c.get("event") or {}
            method = ev.get("extinction_method") or ""
            trigger = ev.get("trigger") or ""
            name = c.get("civilization_name") or ""
            text = f"{method} {trigger} {name}"
            score = sum(1 for kw in keywords if kw in text)
            if score > best_score:
                best_score = score
                best_cycle = c
        return best_cycle if best_cycle else cycles[0]

    def _check_arcaia_judgment(self) -> tuple:
        """スコアが 40 を下回ったか判定。Terminate 時は過去サイクルの遺言を付与。(game_over, message, decay_reason) を返す。"""
        if self.state.total_score >= JUDGMENT_THRESHOLDS["monitor"]:
            return (False, None, "")

        decay_reasons = infer_decay_reasons(
            self.state.tech_level,
            self.state.ethical_dev,
            self.state.sustainability,
        )
        reason = decay_reasons[0] if decay_reasons else "その他"
        patterns = self._arcaia_lore.get("extinction_patterns") or {}
        examples = (patterns.get(reason) or {}).get("examples") or []
        if not examples:
            examples = ["予測不能な事象"]
        flavor = random.choice(examples)
        message = f"アルカイアの審判 — 排除・再生失敗（スコア {self.state.total_score}）。\n理由: {reason} — 例: {flavor}"

        cycle = self._find_best_matching_cycle(reason)
        if cycle:
            civ_name = cycle.get("civilization_name", "未知の文明")
            ev = cycle.get("event") or {}
            ext_method = ev.get("extinction_method", "—")
            arcaia_note = cycle.get("arcaia_note", "")
            message += f"\n\n【遺言 — 過去のサイクルより】\n文明: {civ_name}\n失敗: {ext_method}\nアルカイアの一言: {arcaia_note}"

        return (True, message, reason)

    def run_turn(self, choice_id: str) -> Dict[str, Any]:
        """1 ターン実行。ゲームオーバーなら GameOver を送出。"""
        return self.apply_choice(choice_id)

    def save_game(self, path: str) -> str:
        """現在の文明ステータスを JSON で保存。保存先パスを返す。"""
        payload = {
            "version": 1,
            "turn": self.turn,
            "current_year": self.current_year,
            "state": self.state.to_dict(),
            "current_level": self._current_level,
            "pending_risk_minus": self._pending_risk_minus,
            "milestone_triggered": list(self._milestone_triggered),
            "loop_breaking_factor": self.loop_breaking_factor,
            "recent_choice_axis": list(self._recent_choice_axis),
            "arcaia_monologue_log": list(self._arcaia_monologue_log),
            "consecutive_safe_count": self._consecutive_safe_count,
            "ui_corrupted": self._ui_corrupted,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load_game(cls, path: str, **engine_kwargs: Any) -> "GameEngine":
        """JSON から状態を復元した GameEngine を返す。"""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        engine = cls(**engine_kwargs)
        engine.turn = int(payload.get("turn", 0))
        engine.current_year = int(payload.get("current_year", 0))
        engine.state = CivilizationState.from_dict(payload.get("state") or {})
        engine._current_level = payload.get("current_level")
        engine._pending_risk_minus = list(payload.get("pending_risk_minus") or [])
        engine._milestone_triggered = set(payload.get("milestone_triggered") or [])
        engine.loop_breaking_factor = float(payload.get("loop_breaking_factor", 0))
        engine._recent_choice_axis = list(payload.get("recent_choice_axis") or [])
        engine._arcaia_monologue_log = list(payload.get("arcaia_monologue_log") or [])
        engine._consecutive_safe_count = int(payload.get("consecutive_safe_count", 0))
        engine._ui_corrupted = bool(payload.get("ui_corrupted", False))
        if engine._current_level is None:
            engine._update_current_level()
        return engine


# ---------- CLI 起動 ----------
DEFAULT_SAVE_PATH = os.path.join(_GAME_SCRIPT_DIR, "civilization_save.json")

if __name__ == "__main__":
    engine = GameEngine()
    print("地球文明 対話型シミュレーション — アルカイアの慟哭")
    print("スコアが40を下回るとアルカイアの審判でゲームオーバーです。")
    print("入力: 1-3（または転換点時は1-2）で選択 / s で保存 / l で読込\n")
    while True:
        print(f"--- ターン {engine.turn} (経過 {engine.current_year / 1e8:.1f}億年) ---")
        print(engine.state.get_rita_feedback())
        level_row = _get_tech_level_row_for_game(engine.state.tech_level, engine._tech_table)
        level_name = (level_row.get("name") if level_row else "—")
        print(f"技術:{engine.state.tech_level:.1f} 倫理:{engine.state.ethical_dev:.1f} 持続:{engine.state.sustainability:.1f} スコア:{engine.state.total_score} | Level: {level_name}")
        choices = engine.get_choices(3)
        if engine._current_turning_point_title:
            print(f"\n【文明の転換点】{engine._current_turning_point_title}")
        if engine._current_turning_point_situation:
            print(f"状況: {engine._current_turning_point_situation}\n")
        for i, c in enumerate(choices):
            print(f"  {i+1}. {c['label']}")
        n_choices = len(choices)
        try:
            s = input(f"選択 (1-{n_choices} / s / l): ").strip().lower()
            if s == "s":
                engine.save_game(DEFAULT_SAVE_PATH)
                print(f"保存しました: {DEFAULT_SAVE_PATH}")
                continue
            if s == "l":
                if os.path.isfile(DEFAULT_SAVE_PATH):
                    engine = GameEngine.load_game(DEFAULT_SAVE_PATH)
                    print("読込完了。")
                else:
                    print("保存ファイルがありません。")
                continue
            idx = int(s) - 1
            if 0 <= idx < len(choices):
                result = engine.apply_choice(choices[idx]["id"])
                if result.get("atmosphere_tags"):
                    print("\n" + result["atmosphere_tags"].rstrip())
                if result.get("choice_narrative"):
                    print("▶ " + result["choice_narrative"])
                if result.get("risk_event"):
                    print(result["risk_event"])
                if result.get("level_up_bonus"):
                    print(result["level_up_bonus"])
                if result.get("habitability_warning"):
                    print(result["habitability_warning"])
                if result.get("rita_intuition"):
                    print(result["rita_intuition"])
                print(result["rita_feedback"])
                if result.get("arcaia_monologue"):
                    print(result["arcaia_monologue"])
                if result.get("ending_singularity") and result.get("ending_singularity_message"):
                    print("\n" + result["ending_singularity_message"])
                    print("\n—— アルカイアの管理を離れて ——")
                    sys.exit(0)
                if result.get("ending_5001"):
                    print("\nアルカイア: 「" + (result.get("arcaia_surprise") or "") + "」")
                    if result.get("ending_5001_message"):
                        print("\n" + result["ending_5001_message"])
                    print("\n—— 第5001番目の道 ——")
                    sys.exit(0)
            else:
                print(f"1〜{n_choices}で選んでください。")
        except GameOverRitaZero as e:
            print("\n" + e.message)
            print("ゲームオーバー。セーブファイルを削除しました。")
            if os.path.isfile(DEFAULT_SAVE_PATH):
                try:
                    os.remove(DEFAULT_SAVE_PATH)
                except OSError:
                    pass
            sys.exit(1)
        except GameOver as e:
            print("\n" + e.message)
            print("ゲームオーバー。")
            sys.exit(1)
