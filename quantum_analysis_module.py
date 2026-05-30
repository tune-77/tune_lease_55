"""
Quantum-Inspired Analysis Module
高スコア（>=80）案件の業種固有矛盾を干渉計算で検出する軽量モジュール。
NumPy/SciPy のみ。iMac 2019 (Intel) で動作する CPU-only 実装。
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = "data/quantum_model.joblib"
_CONFIG_PATH = "data/quantum_config.json"


def _load_config() -> dict:
    import json, os
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("quantum_config.json 読み込み失敗、デフォルト値を使用: %s", e)
    return {}


_CFG = _load_config()
_T = _CFG.get("thresholds", {})
_M = _CFG.get("model", {})
_S = _CFG.get("scoring", {})

THRESHOLD_SECONDARY_REVIEW: float     = float(_T.get("secondary_review", 35.0))
THRESHOLD_SECONDARY_REVIEW_MID: float = float(_T.get("secondary_review_mid", 45.0))
THRESHOLD_HIGH_RISK: float            = float(_T.get("high_risk", 60.0))
SCORE_TRIGGER: int                    = int(_S.get("trigger_score", 70))
SCORE_HIGH_THRESHOLD: int             = int(_S.get("high_score_threshold", 80))
_WEIGHT_BOOST_INTERFERENCE: float = float(_T.get("weight_boost_interference", 0.6))
_WEIGHT_BOOST_FACTOR: float       = float(_T.get("weight_boost_factor", 1.3))
_WEIGHT_BOOST_MAX: float          = float(_T.get("weight_boost_max", 3.0))
_ACTIVE_PAIR_MIN: float           = float(_T.get("active_pair_min_interference", 0.05))
_ENTANGLE_ALPHA: float            = float(_M.get("entangle_alpha", 50.0))
_ENTANGLE_RISK_FACTOR: float      = float(_M.get("entangle_risk_factor", 0.2))
_MIN_INDUSTRY_CASES: int          = int(_CFG.get("training", {}).get("min_industry_cases", 2))
_OOD_Z_THRESHOLD: float           = float(_T.get("ood_z_threshold", 2.0))
_DISC_MIN_CASES: int              = int(_CFG.get("training", {}).get("disc_min_cases", 5))
_DISC_WEIGHT_SCALE: float         = float(_T.get("disc_weight_scale", 2.0))
_DISC_TRIGGER_N: int              = int(_CFG.get("training", {}).get("disc_trigger_n", 10))

_GRADE_MAP: dict[str, float] = {
    "①A格": 9.0, "①a": 9.0, "A": 9.0,
    "②B格": 7.0, "②b": 7.0, "B": 7.0,
    "③C格": 5.0, "③c": 5.0, "C": 5.0,
    "④無格付": 3.0, "④": 3.0, "無格付": 3.0,
    "⑤D格": 1.0, "⑤d": 1.0, "D": 1.0,
}

_TREND_MAP: dict[str, float] = {
    "1-3": 2.0, "1": 1.0, "2": 2.0, "3": 3.0,
    "4-6": 5.0, "4": 4.0, "5": 5.0, "6": 6.0,
    "7-9": 8.0, "7": 7.0, "8": 8.0, "9": 9.0,
    "無格付": 5.0,
}

# 業種ペア重み (建設業=D で追加重み)
_BASE_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "depreciation",  1.0),
    ("op_profit",    "trend_val",     1.0),
    ("net_income",   "ord_profit",    0.15),  # 全件ほぼ整合→小重み
    ("revenue",      "op_profit",     0.6),   # 売上と利益の全業種共通チェック
    # net_assets_val ⇔ net_income: 入力データにnet_assetsがほぼゼロのため削除（discriminative sep=-0.085）
]

# 業種別追加ペア: (変数A, 変数B, 重み)
_CONSTRUCTION_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "machines",      2.0),
    ("op_profit",    "equip_total",   2.5),  # 設備なし高利益
    # qualit_score ⇔ op_profit: 成約企業が「定性良く高利益」という自然パターンを誤検知（discriminative sep=-0.142）
]

# H 運輸業（44 道路貨物など）: 車両・設備集約型
_TRANSPORT_PAIRS: list[tuple[str, str, float]] = [
    ("machines",     "op_profit",     2.2),  # 車両なし高利益は矛盾
    ("depreciation", "machines",      2.0),  # 設備あれば償却高いはず
    ("op_profit",    "equip_total",   1.8),
]

# P 医療・福祉（83 医療業, 85 社会保険・福祉）
_MEDICAL_PAIRS: list[tuple[str, str, float]] = [
    ("net_income",   "op_profit",     2.5),  # 保険収入安定のはず→乖離は異常
    ("ord_profit",   "op_profit",     2.0),  # 借入過多で経常利益急落
    ("depreciation", "machines",      1.5),  # 医療機器投資と償却
]

# E 製造業（09 食料品, 21 金属, 24 生産用機械）: 装置産業
_MANUFACTURING_PAIRS: list[tuple[str, str, float]] = [
    ("machines",     "op_profit",     2.0),  # 装置産業なのに設備薄い
    ("depreciation", "machines",      2.2),  # 設備と償却の整合
    ("op_profit",    "equip_total",   1.8),
]

# K 不動産・物品賃貸（70 リース・レンタル）
_RENTAL_PAIRS: list[tuple[str, str, float]] = [
    ("depreciation", "op_profit",     2.5),  # 賃貸資産の償却が必須
    ("machines",     "net_income",    1.8),  # 保有資産と収益性
]

# G 情報通信（39-43）: 知識集約型・設備軽量型
_IT_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "equip_total",   1.8),  # IT企業で設備過多×利益低迷
    ("net_income",   "ord_profit",    1.5),  # 特別損益チェック
    ("revenue",      "op_profit",     2.0),  # 売上はあるが利益が激減（成長倒れ）
]

# I 卸売・小売（50-61）: 薄利多売型
_WHOLESALE_PAIRS: list[tuple[str, str, float]] = [
    ("revenue",      "op_profit",     2.5),  # 薄利業態で利益乖離が大きい
    ("net_income",   "ord_profit",    1.5),  # 棚卸損失・特別損失
    ("op_profit",    "depreciation",  1.2),  # 倉庫・搬送設備と利益の整合
]

# J 金融・保険（62-69）: レバレッジ依存型
_FINANCE_PAIRS: list[tuple[str, str, float]] = [
    ("net_income",       "ord_profit",    2.5),  # 特別損益・金利負担の乖離
    ("op_profit",        "ord_profit",    2.0),  # 金利コストで経常利益急落
    ("equity_ratio_val", "net_income",    2.0),  # 自己資本比率と利益整合（レバレッジ過大）
]

# L 学術・専門サービス（71-79）: 知識・人的資本型
_PROFESSIONAL_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "machines",      2.0),  # 専門サービスで設備過多（外注倒れ）
    ("revenue",      "op_profit",     1.8),  # 受注はあるが粗利が出ない
    ("net_income",   "ord_profit",    1.2),
]

# M 宿泊・飲食（80-82）: 設備・労働集約型
_HOSPITALITY_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "machines",      1.8),  # 厨房・設備と収益性の乖離
    ("depreciation", "machines",      2.0),  # 設備と償却の整合（老朽化リスク）
    ("op_profit",    "equip_total",   1.5),
]

# 業種コード → 追加ペアリスト のマップ
_INDUSTRY_PAIR_MAP: dict[str, list[tuple[str, str, float]]] = {
    "D": _CONSTRUCTION_PAIRS,
    "E": _MANUFACTURING_PAIRS,
    "G": _IT_PAIRS,
    "H": _TRANSPORT_PAIRS,
    "I": _WHOLESALE_PAIRS,
    "J": _FINANCE_PAIRS,
    "K": _RENTAL_PAIRS,
    "L": _PROFESSIONAL_PAIRS,
    "M": _HOSPITALITY_PAIRS,
    "P": _MEDICAL_PAIRS,
}

# 業種コード → 日本語説明
INDUSTRY_RISK_DESCRIPTIONS: dict[str, dict] = {
    "D": {
        "name": "建設業",
        "risk_summary": "設備投資(機械・機器)なしで高利益を計上するパターンが典型的な矛盾。",
        "key_pairs": ["op_profit ⇔ machines", "op_profit ⇔ equip_total"],
    },
    "E": {
        "name": "製造業",
        "risk_summary": "装置産業のため設備投資が薄い場合は外注依存・設備老朽化リスク。償却と設備の乖離を監視。",
        "key_pairs": ["machines ⇔ op_profit", "depreciation ⇔ machines"],
    },
    "G": {
        "name": "情報通信業",
        "risk_summary": "知識集約型のため設備は軽量なはず。設備過多×利益低迷は不採算システム投資を示唆。売上急拡大でも利益が伴わない場合は構造的赤字リスク。",
        "key_pairs": ["revenue ⇔ op_profit", "op_profit ⇔ equip_total"],
    },
    "H": {
        "name": "運輸業",
        "risk_summary": "車両・設備が収益の源泉。設備薄く高利益は燃料費・整備費の過少計上を示唆。",
        "key_pairs": ["machines ⇔ op_profit", "depreciation ⇔ machines"],
    },
    "I": {
        "name": "卸売・小売業",
        "risk_summary": "薄利多売業態のため利益率乖離は棚卸損失・返品・不良在庫を示唆。売上規模と利益の乖離に注意。",
        "key_pairs": ["revenue ⇔ op_profit", "net_income ⇔ ord_profit"],
    },
    "J": {
        "name": "金融・保険業",
        "risk_summary": "レバレッジ依存型。自己資本比率と収益性の乖離は過大借入リスク。金利上昇で経常利益が急落するパターンを監視。",
        "key_pairs": ["net_income ⇔ ord_profit", "equity_ratio_val ⇔ net_income"],
    },
    "K": {
        "name": "不動産・物品賃貸",
        "risk_summary": "賃貸資産の減価償却が高いはず。償却費と利益の乖離は資産評価の歪みを示す。",
        "key_pairs": ["depreciation ⇔ op_profit", "machines ⇔ net_income"],
    },
    "L": {
        "name": "学術・専門サービス業",
        "risk_summary": "人的資本型のため設備過多は不採算。受注はあるが粗利が出ない場合はプロジェクト原価管理の失敗を示唆。",
        "key_pairs": ["op_profit ⇔ machines", "revenue ⇔ op_profit"],
    },
    "M": {
        "name": "宿泊・飲食業",
        "risk_summary": "設備・労働集約型。厨房・設備と収益性の乖離は稼働率低下・食材コスト高騰を示唆。償却と設備の整合が重要。",
        "key_pairs": ["op_profit ⇔ machines", "depreciation ⇔ machines"],
    },
    "P": {
        "name": "医療・福祉",
        "risk_summary": "保険収入で安定するはずが純利益と営業利益が大きく乖離する場合は借入過多・特別損失を疑う。",
        "key_pairs": ["net_income ⇔ op_profit", "ord_profit ⇔ op_profit"],
    },
}


# ── 量子状態マップ ─────────────────────────────────────────────────────────────

class QuantumFeatureMap:
    """連続値 → Bloch 球面の単一量子ビット状態 (theta, phi) にマップ"""

    def __init__(self, mu: dict[str, float] | None = None, sigma: dict[str, float] | None = None):
        self.mu: dict[str, float] = mu or {}
        self.sigma: dict[str, float] = sigma or {}

    def _zscore(self, x: float, key: str) -> float:
        s = self.sigma.get(key, 1.0) or 1.0
        return (x - self.mu.get(key, 0.0)) / s

    def to_state(self, x: float, key: str, var_type: str = "log") -> np.ndarray:
        """
        戻り値: 複素 2-vector |ψ⟩ = [cos(θ/2), e^{iφ}·sin(θ/2)]
        var_type: 'log' | 'ratio' | 'grade' | 'binary'
        """
        if var_type == "log":
            z = self._zscore(math.log1p(max(x, 0.0)), key)
            theta = math.pi * _sigmoid(z)
        elif var_type == "ratio":
            z = self._zscore(x, key)
            theta = math.pi * (0.5 + 0.5 * math.tanh(z))
        elif var_type == "grade":
            # 0..10 スケールを 0..π にマップ
            theta = math.pi * max(0.0, min(x, 10.0)) / 10.0
        else:  # binary
            theta = math.pi if x > 0.5 else 0.0

        phi = math.pi * math.copysign(abs(x) / (abs(x) + 1.0), x)
        return np.array([
            math.cos(theta / 2),
            math.sin(theta / 2) * complex(math.cos(phi), math.sin(phi)),
        ], dtype=complex)

    def fit(self, records: list[dict[str, float]]) -> None:
        """成約案件の各変数を z-score 正規化のため統計量を学習"""
        if not records:
            return
        keys = set().union(*[r.keys() for r in records])
        for k in keys:
            if k.startswith("_"):  # 非数値メタキーをスキップ
                continue
            vals = [r[k] for r in records if k in r and r[k] is not None]
            if not vals:
                continue
            try:
                arr = np.array(vals, dtype=float)
            except (ValueError, TypeError):
                continue
            self.mu[k] = float(np.mean(arr))
            self.sigma[k] = float(np.std(arr)) or 1.0


# ── 干渉計算 ──────────────────────────────────────────────────────────────────

class QuantumInterferenceAnalyzer:

    @staticmethod
    def inner_product(psi_a: np.ndarray, psi_b: np.ndarray) -> complex:
        return complex(np.vdot(psi_a, psi_b))

    @staticmethod
    def interference(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
        """乖離度 = 1 - |⟨ψA|ψB⟩|²  ∈ [0, 1]"""
        ip = QuantumInterferenceAnalyzer.inner_product(psi_a, psi_b)
        return 1.0 - float(abs(ip) ** 2)

    @staticmethod
    def fubini_study(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
        """Fubini-Study 距離 ∈ [0, √2]"""
        ip = QuantumInterferenceAnalyzer.inner_product(psi_a, psi_b)
        return math.sqrt(max(0.0, 2.0 - 2.0 * float(abs(ip))))

    @staticmethod
    def entanglement_entropy(qualit_flag: float, fin_state: np.ndarray) -> float:
        """
        qualitative リスクフラグ と財務状態の von Neumann エントロピー
        qualit_flag=0 (リスクなし) → 直積状態 = S=0
        qualit_flag>0 → CNOT で絡み合い → S>0
        S ∈ [0, 1]  (log2 正規化)
        """
        if qualit_flag <= 0.0:
            return 0.0
        # |q⟩ = sin(π/2 * qualit_flag)|0⟩ + cos(π/2 * qualit_flag)|1⟩
        theta_q = math.pi / 2.0 * min(qualit_flag, 1.0)
        alpha = math.cos(theta_q)
        beta = math.sin(theta_q)

        f0, f1 = float(fin_state[0].real), float(abs(fin_state[1]))
        # 2量子ビット振幅: CNOT(α|0⟩+β|1⟩)(f0|0⟩+f1|1⟩)
        #   |00⟩: α·f0, |01⟩: α·f1, |10⟩: β·f1, |11⟩: β·f0
        amps = np.array([alpha * f0, alpha * f1, beta * f1, beta * f0], dtype=float)
        probs = amps ** 2
        probs /= (probs.sum() + 1e-12)

        # 還元密度行列 ρ_fin の固有値 (partial trace over qubit q)
        # ρ_00 = |A00|²+|A10|², ρ_11 = |A01|²+|A11|², ρ_01 = A00·A01+A10·A11
        r00 = probs[0] + probs[2]
        r11 = probs[1] + probs[3]
        # エントロピー = -Σ λ log2(λ)
        eigs = np.array([r00, r11])
        eigs = eigs[eigs > 1e-12]
        return float(-np.sum(eigs * np.log2(eigs)))


# ── メインクラス ──────────────────────────────────────────────────────────────

class QuantumGate:
    """
    QuantumGate: fit / predict / save / load の四点セット。
    MahalanobisScorer と同じ疎結合パターンを踏襲。
    """

    def __init__(
        self,
        industry_pair_weights: dict[str, float] | None = None,
        entangle_alpha: float | None = None,
    ):
        self.feature_map = QuantumFeatureMap()
        self.analyzer = QuantumInterferenceAnalyzer()
        self.industry_pair_weights: dict[str, float] = industry_pair_weights or {}
        self.industry_weights: dict[str, dict[str, float]] = {}  # 業種別ペア重み (Q.5)
        self.entangle_alpha = entangle_alpha if entangle_alpha is not None else _ENTANGLE_ALPHA
        self._fitted = False

    # ── 学習 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _unwrap_inputs(case: dict[str, Any]) -> dict[str, Any]:
        """{"inputs": {...}} と flat dict の両形式を統一して inputs dict を返す"""
        inp = dict(case.get("inputs", case))
        # トップレベルの業種キーを inputs に補完（past_cases 形式）
        for k in ("industry_major", "industry_sub"):
            if k not in inp and k in case:
                inp[k] = case[k]
        return inp

    def fit(
        self,
        df_seiyaku: list[dict[str, Any]],
        df_lost: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        成約案件の特徴量分布から正規化統計量を学習。
        失注案件がある場合は業種別ペア重みを微調整。
        """
        records = [_extract_features(self._unwrap_inputs(c)) for c in df_seiyaku]
        records = [r for r in records if r is not None]
        self.feature_map.fit(records)  # type: ignore[arg-type]

        if df_lost:
            lost_records = [_extract_features(self._unwrap_inputs(c)) for c in df_lost]
            lost_records = [r for r in lost_records if r is not None]
            if len(records) >= _DISC_TRIGGER_N and len(lost_records) >= _DISC_TRIGGER_N:
                self._tune_weights_discriminative(records, lost_records)
            else:
                self._tune_weights(lost_records)

        self._fitted = True
        logger.info("QuantumGate fitted: seiyaku=%d lost=%d", len(records), len(df_lost or []))

    def _tune_weights(self, lost_records: list[dict[str, float]]) -> None:
        """グローバル + 業種別の両方で重みを調整する"""
        if not lost_records:
            return
        self._tune_weights_global(lost_records)
        self._tune_weights_by_industry(lost_records)

    def _tune_weights_global(self, lost_records: list[dict[str, float]]) -> None:
        """全業種合算でペア重みを調整（industry_pair_weights に反映）"""
        pair_scores: dict[str, list[float]] = {}
        for rec in lost_records:
            fs = self._feature_states(rec)
            for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                d = self.analyzer.interference(psi_a, psi_b)
                pair_scores.setdefault(name, []).append(d)
        for name, vals in pair_scores.items():
            mean_d = float(np.mean(vals))
            if mean_d >= _WEIGHT_BOOST_INTERFERENCE:
                base = self.industry_pair_weights.get(name, 1.0)
                self.industry_pair_weights[name] = min(base * _WEIGHT_BOOST_FACTOR, _WEIGHT_BOOST_MAX)

    def _tune_weights_discriminative(
        self,
        seiyaku_records: list[dict[str, float]],
        lost_records: list[dict[str, float]],
    ) -> None:
        """
        成約/失注の干渉度平均差（separation）でペア重みを設定する。
        separation = mean_lost - mean_seiyaku
          > 0: 失注で干渉度が高い = 予測的ペア → weight boost
          ≤ 0: 逆向きまたはノイズ → weight 引き下げ
        """
        from collections import defaultdict

        pair_won: dict[str, list[float]] = defaultdict(list)
        pair_lost_d: dict[str, list[float]] = defaultdict(list)

        for rec in seiyaku_records:
            fs = self._feature_states(rec)
            for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                pair_won[name].append(self.analyzer.interference(psi_a, psi_b))

        for rec in lost_records:
            fs = self._feature_states(rec)
            for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                pair_lost_d[name].append(self.analyzer.interference(psi_a, psi_b))

        stats: dict[str, dict] = {}
        boosted = 0
        for name in set(pair_won) | set(pair_lost_d):
            won_v = pair_won.get(name, [])
            lost_v = pair_lost_d.get(name, [])
            if len(won_v) < _DISC_MIN_CASES or len(lost_v) < _DISC_MIN_CASES:
                continue
            mean_won = float(np.mean(won_v))
            mean_lost = float(np.mean(lost_v))
            sep = mean_lost - mean_won
            stats[name] = {
                "mean_won": round(mean_won, 4),
                "mean_lost": round(mean_lost, 4),
                "separation": round(sep, 4),
                "n_won": len(won_v),
                "n_lost": len(lost_v),
            }
            if sep > 0:
                new_w = min(1.0 + sep * _DISC_WEIGHT_SCALE, _WEIGHT_BOOST_MAX)
                boosted += 1
            else:
                new_w = max(0.3, 1.0 + sep)
            self.industry_pair_weights[name] = round(new_w, 4)

        self.pair_discrimination_stats_: dict[str, dict] = stats
        logger.info(
            "discriminative weight tuning (global): %d/%d ペアが失注側で高干渉",
            boosted, len(stats),
        )

        self._tune_weights_discriminative_by_industry(seiyaku_records, lost_records)

    def _tune_weights_discriminative_by_industry(
        self,
        seiyaku_records: list[dict[str, float]],
        lost_records: list[dict[str, float]],
    ) -> None:
        """業種別に separation を計算してペア重みを設定する"""
        from collections import defaultdict

        by_ind_won: dict[str, list] = defaultdict(list)
        by_ind_lost: dict[str, list] = defaultdict(list)
        for rec in seiyaku_records:
            by_ind_won[rec.get("_major_str", "") or "__unknown__"].append(rec)
        for rec in lost_records:
            by_ind_lost[rec.get("_major_str", "") or "__unknown__"].append(rec)

        for major in set(by_ind_won) | set(by_ind_lost):
            won_recs = by_ind_won.get(major, [])
            lost_recs = by_ind_lost.get(major, [])
            if len(won_recs) < _DISC_MIN_CASES or len(lost_recs) < _DISC_MIN_CASES:
                logger.debug(
                    "業種 %s: won=%d lost=%d < %d件、業種別discriminative tuning スキップ",
                    major, len(won_recs), len(lost_recs), _DISC_MIN_CASES,
                )
                continue

            pair_won: dict[str, list] = defaultdict(list)
            pair_lost_d: dict[str, list] = defaultdict(list)
            for rec in won_recs:
                fs = self._feature_states(rec)
                for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                    pair_won[name].append(self.analyzer.interference(psi_a, psi_b))
            for rec in lost_recs:
                fs = self._feature_states(rec)
                for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                    pair_lost_d[name].append(self.analyzer.interference(psi_a, psi_b))

            ind_weights = dict(self.industry_weights.get(major, {}))
            updated = 0
            for name in set(pair_won) | set(pair_lost_d):
                won_v = pair_won.get(name, [])
                lost_v = pair_lost_d.get(name, [])
                if len(won_v) < _DISC_MIN_CASES or len(lost_v) < _DISC_MIN_CASES:
                    continue
                sep = float(np.mean(lost_v)) - float(np.mean(won_v))
                if sep > 0:
                    new_w = min(1.0 + sep * _DISC_WEIGHT_SCALE, _WEIGHT_BOOST_MAX)
                else:
                    new_w = max(0.3, 1.0 + sep)
                ind_weights[name] = round(new_w, 4)
                updated += 1

            if ind_weights:
                self.industry_weights[major] = ind_weights
                logger.info(
                    "業種 %s: discriminative tuning won=%d lost=%d %d ペア更新",
                    major, len(won_recs), len(lost_recs), updated,
                )

    def _tune_weights_by_industry(self, lost_records: list[dict[str, float]]) -> None:
        """業種別に分離してペア重みを調整（industry_weights に反映）"""
        by_industry: dict[str, list[dict]] = {}
        for rec in lost_records:
            major = rec.get("_major_str", "") or "__unknown__"
            by_industry.setdefault(major, []).append(rec)

        for major, records in by_industry.items():
            if len(records) < _MIN_INDUSTRY_CASES:
                logger.debug(
                    "業種 %s: サンプル %d件 < 最小 %d件、業種別重み学習スキップ",
                    major, len(records), _MIN_INDUSTRY_CASES,
                )
                continue
            pair_scores: dict[str, list[float]] = {}
            for rec in records:
                fs = self._feature_states(rec)
                for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                    d = self.analyzer.interference(psi_a, psi_b)
                    pair_scores.setdefault(name, []).append(d)
            ind_weights = dict(self.industry_weights.get(major, {}))
            boosted = 0
            for name, vals in pair_scores.items():
                mean_d = float(np.mean(vals))
                if mean_d >= _WEIGHT_BOOST_INTERFERENCE:
                    base = ind_weights.get(name, 1.0)
                    ind_weights[name] = min(base * _WEIGHT_BOOST_FACTOR, _WEIGHT_BOOST_MAX)
                    boosted += 1
            if ind_weights:
                self.industry_weights[major] = ind_weights
                logger.info(
                    "業種 %s: ペア重み学習完了 (%d件) → %d ペア調整",
                    major, len(records), boosted,
                )

    # ── 予測 ─────────────────────────────────────────────────────────────────

    def predict(self, case: dict[str, Any]) -> dict[str, Any]:
        """
        case: form_result dict（千円単位）または past_cases.data dict
        戻り値: quantum_risk, pair_anomalies, entangle_entropy,
                geo_distance_max, verdict, explanation
        """
        if not self._fitted:
            raise RuntimeError(
                "QuantumGate is not fitted. Run fit() or load a trained model first."
            )
        inputs: dict[str, Any] = dict(case.get("inputs", case))
        # トップレベルの業種キーを inputs に補完（past_cases 形式）
        for _k in ("industry_major", "industry_sub"):
            if _k not in inputs and _k in case:
                inputs[_k] = case[_k]
        rec = _extract_features(inputs)
        if rec is None:
            return _null_result()

        fs = self._feature_states(rec)
        pair_anomalies: dict[str, float] = {}
        geo_distances: dict[str, float] = {}

        # 業種別重みを優先、なければグローバル重みにフォールバック（旧モデル互換）
        major = rec.get("_major_str", "")
        _ind_w: dict[str, float] = getattr(self, "industry_weights", {}).get(major, {})
        _glob_w: dict[str, float] = self.industry_pair_weights

        def _boost(name: str) -> float:
            return _ind_w.get(name) or _glob_w.get(name, 1.0)

        for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
            d = self.analyzer.interference(psi_a, psi_b)
            pair_anomalies[name] = round(d, 4)
            geo_distances[name] = round(self.analyzer.fubini_study(psi_a, psi_b), 4)

        # Q_risk 集約 — 有意な乖離のペアのみで正規化（ペア追加によるスコア希釈防止）
        active_risk = 0.0
        active_w = 0.0
        for name, psi_a, psi_b, w in self._iter_pairs(rec, fs):
            boost = _boost(name)
            d = pair_anomalies.get(name, 0.0)
            if d > _ACTIVE_PAIR_MIN:
                active_risk += w * boost * d
                active_w += w * boost

        q_risk_normalized = (active_risk / active_w * 100.0) if active_w > 0 else 0.0

        # Entanglement
        qualit_flag = float(rec.get("qualit_binary", 0.0))
        fin_state = fs.get("op_profit")
        ent_entropy = 0.0
        if fin_state is not None:
            ent_entropy = self.analyzer.entanglement_entropy(qualit_flag, fin_state)

        q_risk = float(np.clip(q_risk_normalized + self.entangle_alpha * ent_entropy * _ENTANGLE_RISK_FACTOR, 0.0, 100.0))
        geo_max = max(geo_distances.values()) if geo_distances else 0.0

        verdict = _verdict(q_risk)
        explanation = _explain(pair_anomalies, rec)

        pair_contributions: dict[str, float] = {}
        for name, psi_a, psi_b, w in self._iter_pairs(rec, fs):
            boost = _boost(name)
            d = pair_anomalies.get(name, 0.0)
            if d > _ACTIVE_PAIR_MIN and active_w > 0:
                pair_contributions[name] = round((w * boost * d / active_w) * 100.0, 4)
            else:
                pair_contributions[name] = 0.0
        explained_risk = round(sum(pair_contributions.values()), 4)

        entropy_risk = round(float(self.entangle_alpha * ent_entropy * _ENTANGLE_RISK_FACTOR), 4)
        residual_signal = entropy_risk  # エントロピー由来の未説明リスク成分（SC.4 閾値比較用）

        ood_flags: dict[str, bool] = {}
        for key, val in rec.items():
            if key.startswith("_") or key not in self.feature_map.mu:
                continue
            try:
                z = self.feature_map._zscore(float(val), key)
                ood_flags[key] = abs(z) > _OOD_Z_THRESHOLD
            except (TypeError, ValueError):
                pass

        return {
            "quantum_risk": round(q_risk, 2),
            "pair_anomalies": pair_anomalies,
            "pair_contributions": pair_contributions,
            "explained_risk": explained_risk,
            "entropy_risk": entropy_risk,
            "residual_signal": residual_signal,
            "ood_flags": ood_flags,
            "entangle_entropy": round(ent_entropy, 4),
            "geo_distance_max": round(geo_max, 4),
            "verdict": verdict,
            "explanation": explanation,
        }

    def _feature_states(self, rec: dict[str, float]) -> dict[str, np.ndarray]:
        fm = self.feature_map
        states: dict[str, np.ndarray] = {}
        for key, vtype in [
            ("op_profit",        "log"),
            ("depreciation",     "log"),
            ("machines",         "log"),
            ("equip_total",      "log"),
            ("net_income",       "log"),
            ("ord_profit",       "log"),
            ("revenue",          "log"),
            ("net_assets_val",   "log"),
            ("trend_val",        "grade"),
            ("qualit_score",     "ratio"),
            ("equity_ratio_val", "ratio"),
        ]:
            v = rec.get(key, 0.0)
            states[key] = fm.to_state(v, key, vtype)
        return states

    def _iter_pairs(
        self,
        rec: dict[str, float],
        states: dict[str, np.ndarray],
    ):
        """(pair_name, psi_a, psi_b, base_weight) を yield"""
        pairs = list(_BASE_PAIRS)
        major = rec.get("_major_str", "")
        extra = _INDUSTRY_PAIR_MAP.get(major, [])
        pairs = pairs + list(extra)
        for a, b, w in pairs:
            psi_a = states.get(a)
            psi_b = states.get(b)
            if psi_a is None or psi_b is None:
                continue
            yield f"{a}_x_{b}", psi_a, psi_b, w

    # ── 永続化 ────────────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)
        logger.info("QuantumGate saved: %s", path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "QuantumGate":
        return joblib.load(path)

    @staticmethod
    @lru_cache(maxsize=1)
    def load_cached(path: str = MODEL_PATH) -> "QuantumGate":
        model = QuantumGate.load(path)
        if not model._fitted:
            raise RuntimeError(f"Loaded model at {path} has _fitted=False. Re-run train_quantum.py.")
        return model


# ── ユーティリティ ────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _grade_to_float(grade_str: str) -> float:
    """'④無格付', 'B', '①1-3 (優良)' 等 → 数値 (0..10)"""
    s = str(grade_str)
    for key, val in _GRADE_MAP.items():
        if key in s:
            return val
    # "①1-3 (優良)" 形式: _TREND_MAP キーを substring マッチ
    for key, val in _TREND_MAP.items():
        if key in s:
            return val
    return 5.0


def _trend_to_float(trend_str: str) -> float:
    """'1-3', '無格付' 等 → 数値"""
    s = str(trend_str).strip()
    return _TREND_MAP.get(s, 5.0)


def _infer_major_code(industry_sub: str) -> str:
    """
    industry_sub の先頭数値コードから JSIC 大分類コードを推定。
    例: "44 道路貨物運送業" → "H"
    """
    try:
        code = int(industry_sub.split(" ")[0].strip())
    except (ValueError, IndexError):
        return ""
    if 1 <= code <= 2:    return "A"  # 農業
    if 3 <= code <= 4:    return "B"  # 漁業
    if code == 5:         return "C"  # 鉱業
    if 6 <= code <= 11:   return "D"  # 建設業
    if 12 <= code <= 35:  return "E"  # 製造業
    if 36 <= code <= 38:  return "F"  # 電気・ガス・水道
    if 39 <= code <= 43:  return "G"  # 情報通信
    if 44 <= code <= 49:  return "H"  # 運輸・郵便
    if 50 <= code <= 55:  return "I"  # 卸売業
    if 56 <= code <= 61:  return "I"  # 小売業
    if 62 <= code <= 69:  return "J"  # 金融・保険
    if code == 70:        return "K"  # 不動産・物品賃貸
    if 71 <= code <= 79:  return "L"  # 学術・専門サービス
    if 80 <= code <= 82:  return "M"  # 宿泊・飲食
    if 83 <= code <= 89:  return "P"  # 医療・福祉
    if 90 <= code <= 99:  return "R"  # サービス業他
    return ""


def _extract_features(inputs: dict[str, Any]) -> dict[str, float] | None:
    """
    form_result または past_cases.inputs から特徴量 dict (float) に変換。
    千円単位 → 百万円 (/1000)。
    """
    try:
        def _f(key: str, default: float = 0.0) -> float:
            v = inputs.get(key, default)
            return float(v) if v is not None else default

        op = _f("op_profit") / 1000.0
        dep = _f("depreciation") / 1000.0
        mach = _f("machines", _f("machinery_equipment")) / 1000.0
        net = _f("net_income") / 1000.0
        ordi = _f("ord_profit") / 1000.0

        # grade / trend
        grade_raw = inputs.get("grade", inputs.get("trend_grade_t0", "④無格付"))
        trend_val = _grade_to_float(str(grade_raw))

        # qualitative
        q = inputs.get("qualitative", {})
        if isinstance(q, dict):
            tags = q.get("strength_tags", []) or q.get("tags", [])
            onehot = q.get("onehot", {})
            if isinstance(onehot, dict):
                qualit_score = float(sum(onehot.values()))
            else:
                qualit_score = float(len(onehot) if hasattr(onehot, "__len__") else 0)
            qualit_binary = 1.0 if (len(tags) > 0 or qualit_score > 0) else 0.0
        else:
            qualit_score = 0.0
            qualit_binary = 0.0

        industry_major = str(inputs.get("industry_major") or "")
        major_code = industry_major.split(" ")[0].strip() if industry_major else ""

        # industry_major が未設定の場合、industry_sub の数値コードから推定
        if not major_code:
            industry_sub = str(inputs.get("industry_sub") or "")
            major_code = _infer_major_code(industry_sub)

        revenue = _f("nenshu") / 1000.0
        net_assets_val = _f("net_assets") / 1000.0
        total_assets_val = max(_f("total_assets", 1.0) / 1000.0, 0.001)
        equity_ratio_val = net_assets_val / total_assets_val

        return {
            "op_profit":           op,
            "depreciation":        dep,
            "machines":            mach,
            "equip_total":         dep + mach,
            "net_income":          net,
            "ord_profit":          ordi,
            "revenue":             revenue,
            "net_assets_val":      net_assets_val,
            "equity_ratio_val":    equity_ratio_val,
            "trend_val":           trend_val,
            "qualit_score":        qualit_score,
            "qualit_binary":       qualit_binary,
            "industry_major_code": float(ord(major_code[0])) if major_code else 0.0,
            "_major_str":          major_code,  # _iter_pairs の文字列比較用
        }
    except Exception as exc:
        logger.debug("feature extraction failed: %s", exc)
        return None


def _verdict(q_risk: float) -> str:
    if q_risk >= THRESHOLD_HIGH_RISK:
        return "高リスク"
    if q_risk >= THRESHOLD_SECONDARY_REVIEW:
        return "要再審"
    return "正常"


def _explain(pair_anomalies: dict[str, float], rec: dict[str, float]) -> list[str]:
    top = sorted(pair_anomalies.items(), key=lambda x: x[1], reverse=True)[:3]
    msgs = []
    label_map = {
        "op_profit_x_depreciation":      "営業利益と減価償却の乖離（設備投資不整合）",
        "op_profit_x_trend_val":         "営業利益とトレンドグレードの矛盾（将来不確実）",
        "net_income_x_ord_profit":       "純利益と経常利益の乖離（特別損益異常）",
        "op_profit_x_machines":          "営業利益と機械装置の乖離（資本集約性不整合）",
        "op_profit_x_equip_total":       "高利益×低設備の矛盾（建設業典型リスク）",
        "revenue_x_op_profit":           "売上高と営業利益の乖離（利益率異常）",
        "equity_ratio_val_x_net_income": "自己資本比率と純利益の乖離（レバレッジ過大リスク）",
        "op_profit_x_ord_profit":        "営業利益と経常利益の乖離（金利負担・営業外費用異常）",
    }
    for name, val in top:
        label = label_map.get(name, name)
        msgs.append(f"{label}: 乖離度 {val:.2f}")
    return msgs


def compute_simple_q_risk(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    財務矛盾ルールによる q_risk 計算。
    QuantumGate.predict() と同一フォーマットを返す。
    学習済みモデルファイル不要。

    実績データ（成約1188/失注757）の discriminative analysis で有効と確認された
    2本のシグナルと補助ルールで構成:
      R1: 格付低×利益率高 (sep=+0.064 最強)
      R2: 売上規模対比利益率異常 (sep=+0.030)
      R3: 営業赤字
      R4: 特別損益乖離
      R5: 業種別設備矛盾（建設/運輸）
    """
    def _f(k: str, d: float = 0.0) -> float:
        v = inputs.get(k)
        return float(v) if v is not None else d

    nenshu_k = _f("nenshu")
    op_k     = _f("op_profit")
    ord_k    = _f("ord_profit")
    net_k    = _f("net_income")
    mach_k   = _f("machines", _f("machinery_equipment"))

    grade_raw = str(inputs.get("grade") or "④無格付")
    grade_val = _grade_to_float(grade_raw)  # 9=A格(最良) / 1=D格(最悪)

    op_margin = op_k / max(nenshu_k, 1.0) if nenshu_k > 0 else 0.0

    industry_major = str(inputs.get("industry_major") or "")
    major_code = industry_major.split(" ")[0].strip()
    if not major_code:
        major_code = _infer_major_code(str(inputs.get("industry_sub") or ""))

    score = 0.0
    flags: list[str] = []

    # R1: 格付低×利益率高（最強シグナル）
    # 低格付なのに利益率が高い = 粉飾・一過性利益の疑い
    if grade_val <= 5.0 and op_margin > 0.05:
        grade_factor = (5.0 - grade_val) / 5.0          # C格=0.0 … D格=0.8
        margin_factor = min((op_margin - 0.05) / 0.25, 1.0)
        contrib = 35.0 * max(grade_factor, 0.1) * (0.5 + 0.5 * margin_factor)
        score += contrib
        flags.append(f"格付({grade_raw})と利益率({op_margin:.0%})の矛盾")

    # R2: 売上規模対比利益率異常（売上はあるが利益が薄すぎる）
    if nenshu_k > 10_000 and op_margin < 0.005:
        severity = max(0.0, 0.005 - op_margin) / 0.005
        contrib = 20.0 * severity
        score += contrib
        flags.append(f"売上({nenshu_k/1000:.0f}百万)対比利益率異常({op_margin:.1%})")

    # R3: 営業赤字
    if op_k < 0:
        red_depth = min(abs(op_k) / max(nenshu_k, 1.0), 0.5) / 0.5
        contrib = 25.0 + 15.0 * red_depth
        score += contrib
        flags.append(f"営業赤字({op_k/1000:.1f}百万)")

    # R4: 特別損益乖離（経常→純利益で大幅減）
    if ord_k > 0 and net_k < ord_k * 0.5:
        gap = (ord_k - net_k) / ord_k
        contrib = min(15.0, gap * 20.0)
        score += contrib
        flags.append(f"特別損益乖離（経常→純利益 {gap:.0%}減）")

    # R5: 業種別設備矛盾
    if major_code == "D" and op_k > 0 and mach_k < op_k * 0.3:
        contrib = min(15.0, (1.0 - mach_k / max(op_k * 0.3, 1.0)) * 15.0)
        score += contrib
        flags.append(f"建設業: 設備薄({mach_k/1000:.1f}百万)×高利益({op_k/1000:.1f}百万)")
    elif major_code == "H" and op_k > 0 and mach_k < op_k:
        contrib = min(10.0, (1.0 - mach_k / max(op_k, 1.0)) * 10.0)
        score += contrib
        flags.append(f"運輸業: 車両薄({mach_k/1000:.1f}百万)×利益({op_k/1000:.1f}百万)")

    q_risk = float(np.clip(score, 0.0, 100.0))
    return {
        "quantum_risk":      round(q_risk, 2),
        "pair_anomalies":    {},
        "pair_contributions": {},
        "explained_risk":    round(q_risk, 2),
        "entropy_risk":      0.0,
        "residual_signal":   0.0,
        "ood_flags":         {},
        "entangle_entropy":  0.0,
        "geo_distance_max":  0.0,
        "verdict":           _verdict(q_risk),
        "explanation":       flags,
    }


def _null_result() -> dict[str, Any]:
    return {
        "quantum_risk": 0.0,
        "pair_anomalies": {},
        "pair_contributions": {},
        "explained_risk": 0.0,
        "entropy_risk": 0.0,
        "residual_signal": 0.0,
        "ood_flags": {},
        "entangle_entropy": 0.0,
        "geo_distance_max": 0.0,
        "verdict": "正常",
        "explanation": [],
    }
