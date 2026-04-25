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
]

# 業種別追加ペア: (変数A, 変数B, 重み)
_CONSTRUCTION_PAIRS: list[tuple[str, str, float]] = [
    ("op_profit",    "machines",      2.0),
    ("op_profit",    "equip_total",   2.5),  # 設備なし高利益
    ("qualit_score", "op_profit",     1.2),
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

# 業種コード → 追加ペアリスト のマップ
_INDUSTRY_PAIR_MAP: dict[str, list[tuple[str, str, float]]] = {
    "D": _CONSTRUCTION_PAIRS,
    "H": _TRANSPORT_PAIRS,
    "P": _MEDICAL_PAIRS,
    "E": _MANUFACTURING_PAIRS,
    "K": _RENTAL_PAIRS,
}

# 業種コード → 日本語説明
INDUSTRY_RISK_DESCRIPTIONS: dict[str, dict] = {
    "D": {
        "name": "建設業",
        "risk_summary": "設備投資(機械・機器)なしで高利益を計上するパターンが典型的な矛盾。",
        "key_pairs": ["op_profit ⇔ machines", "op_profit ⇔ equip_total"],
    },
    "H": {
        "name": "運輸業",
        "risk_summary": "車両・設備が収益の源泉。設備薄く高利益は燃料費・整備費の過少計上を示唆。",
        "key_pairs": ["machines ⇔ op_profit", "depreciation ⇔ machines"],
    },
    "P": {
        "name": "医療・福祉",
        "risk_summary": "保険収入で安定するはずが純利益と営業利益が大きく乖離する場合は借入過多・特別損失を疑う。",
        "key_pairs": ["net_income ⇔ op_profit", "ord_profit ⇔ op_profit"],
    },
    "E": {
        "name": "製造業",
        "risk_summary": "装置産業のため設備投資が薄い場合は外注依存・設備老朽化リスク。償却と設備の乖離を監視。",
        "key_pairs": ["machines ⇔ op_profit", "depreciation ⇔ machines"],
    },
    "K": {
        "name": "不動産・物品賃貸",
        "risk_summary": "賃貸資産の減価償却が高いはず。償却費と利益の乖離は資産評価の歪みを示す。",
        "key_pairs": ["depreciation ⇔ op_profit", "machines ⇔ net_income"],
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
        entangle_alpha: float = 50.0,
    ):
        self.feature_map = QuantumFeatureMap()
        self.analyzer = QuantumInterferenceAnalyzer()
        self.industry_pair_weights: dict[str, float] = industry_pair_weights or {}
        self.entangle_alpha = entangle_alpha  # S ∈ [0,1] × alpha = entangle bonus
        self._fitted = False

    # ── 学習 ─────────────────────────────────────────────────────────────────

    def fit(
        self,
        df_seiyaku: list[dict[str, Any]],
        df_lost: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        成約案件の特徴量分布から正規化統計量を学習。
        失注案件がある場合は業種別ペア重みを微調整。
        """
        records = [_extract_features(c) for c in df_seiyaku]
        records = [r for r in records if r is not None]
        self.feature_map.fit(records)  # type: ignore[arg-type]

        if df_lost:
            lost_records = [_extract_features(c) for c in df_lost]
            lost_records = [r for r in lost_records if r is not None]
            self._tune_weights(lost_records)

        self._fitted = True
        logger.info("QuantumGate fitted: seiyaku=%d lost=%d", len(records), len(df_lost or []))

    def _tune_weights(self, lost_records: list[dict[str, float]]) -> None:
        """失注案件で乖離度が高いペアの重みを増幅する（簡易 weight boosting）"""
        if not lost_records:
            return
        pair_scores: dict[str, list[float]] = {}
        for rec in lost_records:
            fs = self._feature_states(rec)
            for name, psi_a, psi_b, _ in self._iter_pairs(rec, fs):
                d = self.analyzer.interference(psi_a, psi_b)
                pair_scores.setdefault(name, []).append(d)
        # 平均乖離度が 0.6 超のペアを 1.3 倍まで増幅
        for name, vals in pair_scores.items():
            mean_d = float(np.mean(vals))
            if mean_d >= 0.6:
                base = self.industry_pair_weights.get(name, 1.0)
                self.industry_pair_weights[name] = min(base * 1.3, 3.0)

    # ── 予測 ─────────────────────────────────────────────────────────────────

    def predict(self, case: dict[str, Any]) -> dict[str, Any]:
        """
        case: form_result dict（千円単位）または past_cases.data dict
        戻り値: quantum_risk, pair_anomalies, entangle_entropy,
                geo_distance_max, verdict, explanation
        """
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

        for name, psi_a, psi_b, w in self._iter_pairs(rec, fs):
            boost = self.industry_pair_weights.get(name, 1.0)
            d = self.analyzer.interference(psi_a, psi_b)
            pair_anomalies[name] = round(d, 4)
            geo_distances[name] = round(self.analyzer.fubini_study(psi_a, psi_b), 4)
            _ = (w, boost)  # pair contribution computed in Q_risk below

        # Q_risk 集約 — 有意な乖離(>0.05)のペアのみで正規化
        # ペア追加で total_w が膨らみスコアが希釈されるのを防ぐ
        active_risk = 0.0
        active_w = 0.0
        for name, psi_a, psi_b, w in self._iter_pairs(rec, fs):
            boost = self.industry_pair_weights.get(name, 1.0)
            d = pair_anomalies.get(name, 0.0)
            if d > 0.05:
                active_risk += w * boost * d
                active_w += w * boost

        q_risk_normalized = (active_risk / active_w * 100.0) if active_w > 0 else 0.0

        # Entanglement
        qualit_flag = float(rec.get("qualit_binary", 0.0))
        fin_state = fs.get("op_profit")
        ent_entropy = 0.0
        if fin_state is not None:
            ent_entropy = self.analyzer.entanglement_entropy(qualit_flag, fin_state)

        q_risk = float(np.clip(q_risk_normalized + self.entangle_alpha * ent_entropy * 0.2, 0.0, 100.0))
        geo_max = max(geo_distances.values()) if geo_distances else 0.0

        verdict = _verdict(q_risk)
        explanation = _explain(pair_anomalies, rec)

        return {
            "quantum_risk": round(q_risk, 2),
            "pair_anomalies": pair_anomalies,
            "entangle_entropy": round(ent_entropy, 4),
            "geo_distance_max": round(geo_max, 4),
            "verdict": verdict,
            "explanation": explanation,
        }

    def _feature_states(self, rec: dict[str, float]) -> dict[str, np.ndarray]:
        fm = self.feature_map
        states: dict[str, np.ndarray] = {}
        for key, vtype in [
            ("op_profit",    "log"),
            ("depreciation", "log"),
            ("machines",     "log"),
            ("equip_total",  "log"),
            ("net_income",   "log"),
            ("ord_profit",   "log"),
            ("trend_val",    "grade"),
            ("qualit_score", "ratio"),
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
        return QuantumGate.load(path)


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

        return {
            "op_profit":           op,
            "depreciation":        dep,
            "machines":            mach,
            "equip_total":         dep + mach,
            "net_income":          net,
            "ord_profit":          ordi,
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
    if q_risk >= 60:
        return "高リスク"
    if q_risk >= 35:
        return "要再審"
    return "正常"


def _explain(pair_anomalies: dict[str, float], rec: dict[str, float]) -> list[str]:
    top = sorted(pair_anomalies.items(), key=lambda x: x[1], reverse=True)[:3]
    msgs = []
    label_map = {
        "op_profit_x_depreciation":  "営業利益と減価償却の乖離（設備投資不整合）",
        "op_profit_x_trend_val":     "営業利益とトレンドグレードの矛盾（将来不確実）",
        "net_income_x_ord_profit":   "純利益と経常利益の乖離（特別損益異常）",
        "op_profit_x_machines":      "営業利益と機械装置の乖離（資本集約性不整合）",
        "op_profit_x_equip_total":   "高利益×低設備の矛盾（建設業典型リスク）",
        "qualit_score_x_op_profit":  "定性リスクと財務利益の干渉",
    }
    for name, val in top:
        label = label_map.get(name, name)
        msgs.append(f"{label}: 乖離度 {val:.2f}")
    return msgs


def _null_result() -> dict[str, Any]:
    return {
        "quantum_risk": 0.0,
        "pair_anomalies": {},
        "entangle_entropy": 0.0,
        "geo_distance_max": 0.0,
        "verdict": "正常",
        "explanation": [],
    }
