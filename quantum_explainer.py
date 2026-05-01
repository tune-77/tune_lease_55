"""
quantum_explainer.py
====================
量子干渉スコアの説明可能性モジュール。

QuantumGate.predict() の戻り値を受け取り、
変数レベルの加法的寄与（Shapley 的分解）を返す。
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from quantum_analysis_module import INDUSTRY_RISK_DESCRIPTIONS, _OOD_Z_THRESHOLD

if TYPE_CHECKING:
    from quantum_analysis_module import QuantumGate

_VAR_LABELS: dict[str, str] = {
    "op_profit":     "営業利益",
    "depreciation":  "減価償却費",
    "machines":      "機械設備",
    "equip_total":   "設備合計",
    "net_income":    "当期純利益",
    "ord_profit":    "経常利益",
    "trend_val":     "格付トレンド",
    "qualit_score":  "定性評価スコア",
    "qualit_binary": "定性評価",
}


class QuantumExplainer:
    """量子干渉スコアの説明可能性クラス。

    Args:
        gate: 学習済み QuantumGate（OOD 検出などで使用、省略可）
    """

    def __init__(self, gate: "QuantumGate | None" = None) -> None:
        self._gate = gate

    def shapley_contributions(self, predict_result: dict[str, Any]) -> dict[str, float]:
        """pair_contributions から変数別の加法的寄与 dict を返す。

        各ペア(A, B)の寄与点数を A・B に等分して集計する。
        sum(返り値.values()) == explained_risk（±1e-3 以内）。

        Args:
            predict_result: QuantumGate.predict() の戻り値

        Returns:
            {変数名: 加法的寄与点数} の dict
        """
        pair_contribs: dict[str, float] = predict_result.get("pair_contributions", {})
        var_contribs: dict[str, float] = {}

        for pair_name, contrib in pair_contribs.items():
            if "_x_" not in pair_name:
                continue
            var_a, var_b = pair_name.split("_x_", 1)
            half = contrib / 2.0
            var_contribs[var_a] = var_contribs.get(var_a, 0.0) + half
            var_contribs[var_b] = var_contribs.get(var_b, 0.0) + half

        return {k: round(v, 4) for k, v in var_contribs.items()}

    _COUNTERFACTUAL_CAUTION = (
        "数値操作推奨ではない: この分析は仮説的シナリオです。"
        "実際の財務操作を推奨するものではありません。"
    )

    def counterfactual(
        self,
        case: dict[str, Any],
        var: str,
        target_val: float,
    ) -> dict[str, Any]:
        """1 変数を差し替えた反事実シナリオの Q_risk 差分を返す。

        Args:
            case: 元の case dict（predict() と同じ形式）
            var:  差し替える変数名（inputs キー名、千円単位）
            target_val: 差し替え後の値

        Returns:
            original_risk, counterfactual_risk, delta (cf - orig),
            var, target_val, caution（「数値操作推奨ではない」注記）

        Raises:
            ValueError: gate が設定されていない場合
        """
        if self._gate is None:
            raise ValueError(
                "QuantumExplainer に gate が設定されていません。"
                "QuantumExplainer(gate=fitted_gate) で生成してください"
            )

        original_risk = self._gate.predict(case)["quantum_risk"]

        modified = copy.deepcopy(case)
        if "inputs" in modified:
            modified["inputs"][var] = target_val
        else:
            modified[var] = target_val

        cf_risk = self._gate.predict(modified)["quantum_risk"]

        return {
            "original_risk": original_risk,
            "counterfactual_risk": cf_risk,
            "delta": round(cf_risk - original_risk, 4),
            "var": var,
            "target_val": target_val,
            "caution": self._COUNTERFACTUAL_CAUTION,
        }

    def build_narrative(
        self,
        predict_result: dict[str, Any],
        case: dict[str, Any] | None = None,
        industry_cases: list[dict[str, Any]] | None = None,
    ) -> str:
        """テンプレートベースの自然言語レポートを生成する（LLM 不要）。

        出力例:
          「減価償却費 が業種平均より 93%低い（n=7件、参考値）。建設業典型リスク。寄与: +8.3点」

        Args:
            predict_result: QuantumGate.predict() の戻り値
            case: 元の case dict（業種・入力値取得に使用）
            industry_cases: 同業種の case リスト（業種平均比較に使用、n<5 は無効）

        Returns:
            自然言語レポート文字列
        """
        shapley = self.shapley_contributions(predict_result)

        # 業種コード・ラベル取得
        major_code = ""
        case_inputs: dict[str, Any] = {}
        if case is not None:
            case_inputs = dict(case.get("inputs", case))
            industry_str = str(case_inputs.get("industry_major", ""))
            major_code = industry_str.split(" ")[0].strip() if industry_str else ""

        industry_info = INDUSTRY_RISK_DESCRIPTIONS.get(major_code, {})
        industry_label = industry_info.get("name", "")

        # 上位 3 変数のナレーション生成
        top_vars = sorted(shapley.items(), key=lambda kv: kv[1], reverse=True)
        sentences: list[str] = []

        for var, contrib in top_vars[:3]:
            if contrib <= 0:
                continue
            label = _VAR_LABELS.get(var, var)

            # 業種平均比較テキスト
            deviation_text = ""
            if case_inputs and industry_cases:
                raw_val = case_inputs.get(var)
                if raw_val is not None:
                    ind_vals: list[float] = []
                    for ic in industry_cases:
                        ic_inp = ic.get("inputs", ic)
                        v = ic_inp.get(var)
                        if v is not None:
                            try:
                                ind_vals.append(float(v))
                            except (TypeError, ValueError):
                                pass
                    if len(ind_vals) >= self._MIN_CASES_FOR_PERCENTILE:
                        mean_val = sum(ind_vals) / len(ind_vals)
                        n = len(ind_vals)
                        if mean_val > 0:
                            dev_pct = (float(raw_val) / mean_val - 1) * 100
                            direction = "高い" if dev_pct > 0 else "低い"
                            deviation_text = (
                                f" が業種平均より {abs(dev_pct):.0f}%{direction}"
                                f"（n={n}件、参考値）"
                            )

            # 業種典型リスクコンテキスト（最初の変数のみ付与）
            industry_ctx = f"{industry_label}典型リスク" if (industry_label and not sentences) else ""

            parts = [f"{label}{deviation_text}"]
            if industry_ctx:
                parts.append(industry_ctx)
            parts.append(f"寄与: +{contrib:.1f}点")
            sentences.append("。".join(parts))

        if not sentences:
            risk = predict_result.get("quantum_risk", 0.0)
            return f"量子リスク {risk:.1f}点（詳細情報なし）"

        return "\n".join(sentences)

    _MIN_CASES_FOR_PERCENTILE: int = 5

    def industry_percentile(
        self,
        var: str,
        value: float,
        cases: list[dict[str, Any]],
    ) -> float | None:
        """業種内での value の分位数を返す（n<5 は None）。

        Args:
            var:   変数名（inputs キー名）
            value: 分位数を求める値
            cases: 同業種の case dict リスト

        Returns:
            0〜100 の分位数（小数第2位まで）、または n<5 のとき None

        Note:
            n=XX件（参考値）として UI 側で件数を必ず明示すること。
        """
        values: list[float] = []
        for case in cases:
            inputs = case.get("inputs", case)
            raw = inputs.get(var)
            if raw is None:
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue

        n = len(values)
        if n < self._MIN_CASES_FOR_PERCENTILE:
            return None

        count_below = sum(1 for v in values if v < value)
        count_equal = sum(1 for v in values if v == value)
        pctile = (count_below + 0.5 * count_equal) / n * 100.0
        return round(pctile, 2)

    # ── Phase 4: 幾何学的説明 ─────────────────────────────────────────────────

    def distance_to_boundary(self, case: dict[str, Any]) -> dict[str, float]:
        """
        現在ケースから量子リスク判定境界までの距離を返す。

        量子リスクの境界は quantum_risk = 35 (要注意閾値)。
        境界への「ロジット距離」: 各ペアの干渉振幅変化量が
        境界突破に必要な最小値として求まる。

        Returns:
            {
                "quantum_risk": float,       # 現在の量子リスクスコア
                "threshold": float,          # 判定境界 (35)
                "gap": float,                # 現在値 − 閾値
                "normalized_gap": float,     # gap / threshold (相対距離)
                "margin": str,               # "safe" | "warning" | "danger"
            }
        """
        if self._gate is None:
            raise ValueError("gate が設定されていません。")
        pred = self._gate.predict(case)
        qrisk = pred.get("quantum_risk", 0.0)
        threshold = 35.0
        gap = float(qrisk) - threshold
        return {
            "quantum_risk": float(qrisk),
            "threshold": threshold,
            "gap": round(gap, 3),
            "normalized_gap": round(gap / threshold, 4),
            "margin": "danger" if gap >= 10 else ("warning" if gap >= 0 else "safe"),
        }

    def minimum_change_vector(
        self,
        case: dict[str, Any],
        target_risk: float = 34.9,
        step_size: float = 0.05,
        max_iters: int = 200,
    ) -> dict[str, Any]:
        """
        quantum_risk を target_risk 以下にするための最小変更ベクトルを求める。

        各変数を1単位（千円 or 1）変化させたときの quantum_risk 感度を計算し、
        感度の大きい変数から優先的に変更量を割り当てるGreedy法で
        最小コストの変更ベクトルを推定する。

        Returns:
            {
                "original_risk": float,
                "target_risk": float,
                "achievable": bool,          # 収束できたか
                "final_risk": float,
                "changes": [                 # 変更量が大きい順
                    {"var": str, "label": str, "unit_sensitivity": float,
                     "recommended_change": float, "unit": str}
                ],
                "total_change_magnitude": float,
            }
        """
        if self._gate is None:
            raise ValueError("gate が設定されていません。")

        pred = self._gate.predict(case)
        orig_risk = float(pred.get("quantum_risk", 0.0))

        if orig_risk <= target_risk:
            return {
                "original_risk": orig_risk,
                "target_risk": target_risk,
                "achievable": True,
                "final_risk": orig_risk,
                "changes": [],
                "total_change_magnitude": 0.0,
            }

        # 対象変数と単位設定（千円系は1000、率系は0.01）
        TARGET_VARS = [
            ("op_profit",    "営業利益",     1_000, "千円"),
            ("depreciation", "減価償却費",   1_000, "千円"),
            ("machines",     "機械設備",     1_000, "千円"),
            ("net_income",   "当期純利益",   1_000, "千円"),
            ("ord_profit",   "経常利益",     1_000, "千円"),
            ("equip_total",  "設備合計",     1_000, "千円"),
        ]

        def _perturb_risk(var: str, delta: float) -> float:
            modified = copy.deepcopy(case)
            cur = float(modified.get(var, 0))
            modified[var] = cur + delta
            return float(self._gate.predict(modified).get("quantum_risk", orig_risk))

        # 各変数の感度計算（1単位増加でリスクがどれだけ下がるか）
        sensitivities = []
        for var, label, unit_delta, unit_str in TARGET_VARS:
            r_plus  = _perturb_risk(var, +unit_delta)
            r_minus = _perturb_risk(var, -unit_delta)
            sensitivity = (r_minus - r_plus) / (2 * unit_delta)  # リスク低減感度
            sensitivities.append({
                "var": var, "label": label,
                "unit_sensitivity": round(sensitivity, 6),
                "unit_delta": unit_delta,
                "unit": unit_str,
            })

        # 感度の高い順にソート
        sensitivities.sort(key=lambda x: x["unit_sensitivity"], reverse=True)

        # Greedy 最小変更探索
        modified = copy.deepcopy(case)
        changes: dict[str, float] = {s["var"]: 0.0 for s in sensitivities}
        current_risk = orig_risk
        achievable = False

        for _ in range(max_iters):
            if current_risk <= target_risk:
                achievable = True
                break
            # 最も感度が高い変数を1ステップ変更
            best = None
            best_reduction = 0.0
            for s in sensitivities:
                if s["unit_sensitivity"] <= 0:
                    continue
                trial_risk = _perturb_risk(s["var"], +s["unit_delta"] * step_size)
                reduction = current_risk - trial_risk
                if reduction > best_reduction:
                    best_reduction = reduction
                    best = s
            if best is None:
                break
            cur = float(modified.get(best["var"], 0))
            modified[best["var"]] = cur + best["unit_delta"] * step_size
            changes[best["var"]] = changes[best["var"]] + best["unit_delta"] * step_size
            current_risk = float(self._gate.predict(modified).get("quantum_risk", current_risk))

        total_mag = sum(abs(v) for v in changes.values())
        change_list = [
            {
                "var": s["var"],
                "label": s["label"],
                "unit_sensitivity": s["unit_sensitivity"],
                "recommended_change": round(changes[s["var"]], 1),
                "unit": s["unit"],
            }
            for s in sensitivities
            if abs(changes.get(s["var"], 0)) > 0
        ]
        change_list.sort(key=lambda x: abs(x["recommended_change"]), reverse=True)

        return {
            "original_risk": orig_risk,
            "target_risk": target_risk,
            "achievable": achievable,
            "final_risk": round(current_risk, 3),
            "changes": change_list,
            "total_change_magnitude": round(total_mag, 1),
        }

    def geometric_summary(self, case: dict[str, Any]) -> dict[str, Any]:
        """
        distance_to_boundary と minimum_change_vector をまとめた
        幾何学的説明サマリーを返す。

        Returns:
            {
                "boundary": dict,       # distance_to_boundary の結果
                "min_change": dict,     # minimum_change_vector の結果
                "interpretation": str,  # 自然言語サマリー
            }
        """
        boundary = self.distance_to_boundary(case)
        if boundary["margin"] == "safe":
            min_change = {
                "original_risk": boundary["quantum_risk"],
                "target_risk": 35.0,
                "achievable": True,
                "final_risk": boundary["quantum_risk"],
                "changes": [],
                "total_change_magnitude": 0.0,
            }
        else:
            min_change = self.minimum_change_vector(case)

        qrisk = boundary["quantum_risk"]
        gap = boundary["gap"]
        margin = boundary["margin"]
        if margin == "safe":
            interp = (
                f"量子リスクは {qrisk:.1f} で境界(35)から {abs(gap):.1f}pt 安全域にあります。"
            )
        elif margin == "warning":
            top = min_change["changes"][:2] if min_change["changes"] else []
            vars_str = "、".join(f"{c['label']}+{c['recommended_change']:,.0f}{c['unit']}" for c in top)
            interp = (
                f"量子リスクは {qrisk:.1f} で境界超過({gap:+.1f}pt)。"
                f"最小変更候補: {vars_str or '変数感度が低く改善困難'}。"
            )
        else:
            top = min_change["changes"][:3] if min_change["changes"] else []
            vars_str = "、".join(f"{c['label']}+{c['recommended_change']:,.0f}{c['unit']}" for c in top)
            interp = (
                f"高リスク(量子スコア {qrisk:.1f})。境界まで {gap:+.1f}pt。"
                f"{'改善達成可能: ' + vars_str if min_change['achievable'] else '現在の変数範囲では改善が困難です。'}。"
            )

        return {
            "boundary": boundary,
            "min_change": min_change,
            "interpretation": interp,
        }

    def ood_check(self, rec: dict[str, Any]) -> bool:
        """入力が学習分布の外挿域かを判定（|z| > 2.0 で True）。

        学習統計量（mu/sigma）が存在しない変数はスキップする。
        fit([]) など学習データが空の場合は常に False を返す。

        Args:
            rec: _extract_features() の出力 dict（変数名 → 数値）

        Returns:
            外挿域の変数が 1 つでもあれば True、なければ False

        Raises:
            ValueError: gate が設定されていない場合
        """
        if self._gate is None:
            raise ValueError("QuantumExplainer に gate が設定されていません。QuantumExplainer(gate=fitted_gate) で生成してください")
        fm = self._gate.feature_map
        for key, val in rec.items():
            if key.startswith("_"):
                continue
            if key not in fm.mu:
                continue
            try:
                z = fm._zscore(float(val), key)
                if abs(z) > _OOD_Z_THRESHOLD:
                    return True
            except (TypeError, ValueError):
                continue
        return False
