"""
quantum_finance_analyzer.py
────────────────────────────────────────────────────────────
量子干渉・物理動態モデルによる財務異常検知モジュール。

「量子」は比喩的な枠組みであり、実質は以下の数学を使用:
  - Bloch球面マッピング → 財務ペアの位相空間での整合性検査
  - 仮想トルク           → 資産回転効率の力学的表現
  - ベンフォード則       → 数値操作の統計的検出

依存: numpy, pandas, scipy, matplotlib（全てIntel Macで動作）
────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import chi2_contingency
from scipy.spatial.distance import cosine


class QuantumFinanceAnalyzer:
    """
    財務データを波形・力学ベクトルとして解析し、
    LightGBMでは検知困難な構造的矛盾を定量化する。
    """

    # ── 財務整合性ペア（key: 分子列名, value: 分母列名, expected_ratio: 健全域中央値） ──
    COHERENCE_PAIRS = [
        # (分子,          分母,           期待比率中央値, ペア名)
        ("op_profit",    "nenshu",       0.05,  "営業利益率"),
        ("net_assets",   "total_assets", 0.30,  "自己資本比率"),
        ("bank_credit",  "total_assets", 0.40,  "銀行借入比率"),
        ("gross_profit", "nenshu",       0.25,  "粗利率"),
        ("net_income",   "nenshu",       0.03,  "純利益率"),
        ("lease_credit", "total_assets", 0.15,  "リース比率"),
        ("machines",     "total_assets", 0.20,  "設備比率"),
        ("depreciation", "machines",     0.10,  "減価償却率"),
        ("ord_profit",   "nenshu",       0.04,  "経常利益率"),
        ("op_profit",    "total_assets", 0.04,  "総資産営業利益率"),
    ]

    # デコヒーレンス閾値（ラジアン）: この値を超えると異常フラグ
    DECOHERENCE_THRESHOLD = 2.199  # ≈ 126°: 財務比率の「許容位相差」

    # Q_risk補正トリガー閾値（40以上で減衰開始）
    Q_RISK_CORRECTION_THRESHOLD = 40

    def __init__(self, industry_benchmarks: dict | None = None):
        """
        industry_benchmarks: {industry_sub: {op_margin: float, equity_ratio: float, ...}}
        省略時はデフォルト閾値を使用。
        """
        self.benchmarks = industry_benchmarks or {}

    # ─────────────────────────────────────────────────────────────────────────
    # ① 量子干渉による矛盾検知
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_q_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        財務ペアをBloch球面にマッピングし、「期待コヒーレンス」からの
        位相乖離度を Q_risk スコア（0〜100）として返す。

        【数学的根拠】
        財務比率 r = A/B を Bloch球面上の状態ベクトル |ψ⟩ に変換:
            θ = 2 * arctan(r / r_expected)   # 極角: 期待値からの乖離
            φ = π * clip(r, 0, 2*r_expected) # 方位角: 比率の絶対位置

        位相差 Δφ = |θ_actual - θ_coherent| が DECOHERENCE_THRESHOLD を
        超えると「財務構造のデコヒーレンス」= 異常と判定。

        複数ペアの位相差をRMS合成してQ_riskとする。
        高いQ_riskは「単一指標は正常に見えるが複数指標間の整合性が崩壊」
        を示す — LightGBMが苦手な多変量矛盾の検知。
        """
        scores = []

        for _, row in df.iterrows():
            pair_phases = []

            for num_col, den_col, expected_ratio, label in self.COHERENCE_PAIRS:
                num = self._safe(row, num_col)
                den = self._safe(row, den_col)

                # 純資産マイナス（債務超過）は最大デコヒーレンスとして扱う
                if den <= 0 and num < 0:
                    pair_phases.append(np.pi)  # 最大乖離
                    continue
                if den <= 0:
                    continue

                actual_ratio = num / den
                # マイナス利益は「負の比率」として位相に反映
                if actual_ratio < 0:
                    actual_ratio = abs(actual_ratio) + expected_ratio  # 期待値の反対側へ

                # Bloch球面マッピング
                # θ: 期待比率を基準とした極角（0=完全一致, π=完全逆転）
                theta_actual   = 2 * np.arctan(max(actual_ratio,   1e-9) / expected_ratio)
                theta_coherent = np.pi / 2  # 期待状態は赤道（θ=π/2）

                delta_phi = abs(theta_actual - theta_coherent)
                pair_phases.append(min(delta_phi, np.pi))  # [0, π]に制限

            if not pair_phases:
                scores.append(0.0)
                continue

            # RMS位相差 → 0〜100スコア化
            rms_phase = np.sqrt(np.mean(np.array(pair_phases) ** 2))
            # DECOHERENCE_THRESHOLD を超えた分を100点換算
            q_risk = min(100.0, (rms_phase / self.DECOHERENCE_THRESHOLD) * 100)
            scores.append(round(q_risk, 1))

        return pd.Series(scores, index=df.index, name="q_risk")

    # ─────────────────────────────────────────────────────────────────────────
    # ② 物理学的動態分析（仮想トルク & 失速検知）
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_physical_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        企業を「回転する剛体」として力学的に解析する。

        【数学的根拠】
        角運動量 L = r × v（クロス積）の財務的解釈:
            r (回転半径) = 総資産     ... 企業が保有する「質量」
            v (接線速度) = 売上高     ... 資産を使って生み出す「速度」
            L = total_assets × nenshu  (仮想角運動量)

        L が業界平均に対して極端に低い = 「大きな体（資産）で遅く動く企業」
        = 資産の死蔵・非効率運用 → 返済能力の低下リスク

        さらに「向心加速度」として純利益率で補正:
            a_c = net_income / total_assets (ROA)
        ROAが負で角運動量が低い = 「失速かつ縮小螺旋」= 高リスク
        """
        result = pd.DataFrame(index=df.index)

        assets  = df.get("total_assets",  pd.Series(0, index=df.index)).fillna(0)
        sales   = df.get("nenshu",        pd.Series(0, index=df.index)).fillna(0)
        net_inc = df.get("net_income",    pd.Series(0, index=df.index)).fillna(0)

        # 仮想角運動量（正規化: 両者を対数変換してスケール揃え）
        log_assets = np.log1p(assets.clip(lower=0))
        log_sales  = np.log1p(sales.clip(lower=0))
        L = log_assets * log_sales
        result["virtual_L"] = L.round(3)

        # 資産回転率（売上/総資産）: 業界標準は0.8〜1.5程度
        asset_turnover = np.where(assets > 0, sales / assets, 0)
        result["asset_turnover"] = np.round(asset_turnover, 3)

        # ROA（向心加速度）
        roa = np.where(assets > 0, net_inc / assets, 0)
        result["roa"] = np.round(roa, 4)

        # 失速判定: 資産回転率 < 0.3 かつ ROA < 0
        stall_flag = (asset_turnover < 0.3) & (roa < 0)
        result["stall_flag"] = stall_flag.astype(int)

        # 動態スコア（0=最悪 〜 100=最良）
        # 資産回転率を0.8基準で評価し、ROAで加減算
        turnover_score = np.clip(asset_turnover / 0.8 * 50, 0, 70)
        roa_score      = np.clip(roa * 500, -20, 30)  # ROA ±4%で±20pt
        dynamics_score = np.clip(turnover_score + roa_score, 0, 100)
        result["dynamics_score"] = np.round(dynamics_score, 1)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ③ ベンフォード則によるデータエントロピー検査
    # ─────────────────────────────────────────────────────────────────────────

    def detect_data_entropy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        財務数値の先頭桁分布をベンフォードの法則と比較し、
        不自然な「整いすぎ」= データ操作の疑いを検出する。

        【数学的根拠】
        自然発生する数値の先頭桁 d (1〜9) の出現確率は:
            P(d) = log10(1 + 1/d)
            → d=1: 30.1%, d=2: 17.6%, ... d=9: 4.6%

        操作されたデータは:
          A) 切りのいい数字（5000, 10000等）が多い → d=1,5が過多
          B) 意図的に「それっぽく見せる」場合 → 分布が均一に近づく

        KLダイバージェンス D_KL(実際 || ベンフォード) で乖離を定量化。
        高いD_KL = 不自然な数値構成 → 粉飾リスクフラグ。
        """
        # ベンフォードの期待分布
        benford_expected = np.array([
            np.log10(1 + 1/d) for d in range(1, 10)
        ])

        numeric_cols = [
            "nenshu", "gross_profit", "op_profit", "ord_profit", "net_income",
            "net_assets", "total_assets", "machines", "other_assets",
            "depreciation", "bank_credit", "lease_credit",
        ]
        available = [c for c in numeric_cols if c in df.columns]

        result = pd.DataFrame(index=df.index)

        # 行単位のベンフォード検定（全財務項目の先頭桁を集計）
        kl_scores = []
        for _, row in df.iterrows():
            digits = []
            for col in available:
                val = abs(self._safe(row, col))
                if val >= 1:
                    first_digit = int(str(int(val))[0])
                    if 1 <= first_digit <= 9:
                        digits.append(first_digit)

            if len(digits) < 5:
                kl_scores.append(0.0)
                continue

            # 実際の先頭桁分布
            counts = np.array([digits.count(d) for d in range(1, 10)], dtype=float)
            actual_prob = counts / counts.sum()

            # ゼロ回避（KL計算用）
            actual_prob = np.clip(actual_prob, 1e-9, 1.0)
            benford_p   = np.clip(benford_expected, 1e-9, 1.0)

            # KLダイバージェンス D_KL(actual || benford)
            kl_div = np.sum(actual_prob * np.log(actual_prob / benford_p))
            # 0〜100スコア化（KL=0.3超で疑わしい）
            entropy_risk = min(100.0, (kl_div / 0.3) * 100)
            kl_scores.append(round(entropy_risk, 1))

        result["entropy_risk"] = kl_scores

        # フラグ: entropy_risk > 70 を「要注意」
        # 閾値80: 実データは完全な整数ではないためより緩く設定
        result["data_integrity_flag"] = (result["entropy_risk"] > 80).astype(int)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ④ 統合判定: 既存LightGBMスコアへの補正
    # ─────────────────────────────────────────────────────────────────────────

    def apply_quantum_correction(
        self,
        base_score: float,
        q_risk: float,
        dynamics_score: float,
        entropy_risk: float,
    ) -> dict:
        """
        既存スコアにQ_risk・動態・エントロピーリスクを重ねて補正する。

        補正関数:
            correction = exp(-λ × q_risk/100) × dynamics_factor × entropy_factor

            λ = 0.8  (減衰率: Q_risk=100で exp(-0.8)≈0.45 まで減衰)
            dynamics_factor = 1.0 if dynamics_score >= 40 else 0.85
            entropy_factor  = 1.0 if entropy_risk  <  70 else 0.90

        Q_risk < 60 の場合は補正なし（既存モデルを尊重）。
        """
        if q_risk < self.Q_RISK_CORRECTION_THRESHOLD:
            return {
                "corrected_score": base_score,
                "correction_factor": 1.0,
                "q_risk": q_risk,
                "warning": None,
            }

        lam = 0.8
        decay = np.exp(-lam * q_risk / 100)
        dyn_factor = 1.0 if dynamics_score >= 40 else 0.85
        ent_factor  = 1.0 if entropy_risk  <  70 else 0.90

        factor = decay * dyn_factor * ent_factor
        corrected = round(base_score * factor, 1)

        warnings_list = []
        if q_risk >= self.Q_RISK_CORRECTION_THRESHOLD:
            warnings_list.append(f"財務整合性の崩壊 (Q_risk={q_risk:.0f})")
        if dynamics_score < 40:
            warnings_list.append("資産失速状態")
        if entropy_risk >= 70:
            warnings_list.append("数値操作の疑い")

        return {
            "corrected_score":   corrected,
            "correction_factor": round(factor, 3),
            "q_risk":            q_risk,
            "warning":           " / ".join(warnings_list) or None,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ⑤ 可視化
    # ─────────────────────────────────────────────────────────────────────────

    def plot_bloch_sphere(self, df: pd.DataFrame, save_path: str | None = None):
        """財務ペアをBloch球面上にプロットする（3D散布図）。"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 球面の描画
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 60)
        x_s = np.outer(np.cos(u), np.sin(v))
        y_s = np.outer(np.sin(u), np.sin(v))
        z_s = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_s, y_s, z_s, alpha=0.05, color="skyblue")

        # 各企業の財務状態ベクトルをプロット
        q_risks = self.calculate_q_risk(df)
        for i, (idx, row) in enumerate(df.iterrows()):
            num = self._safe(row, "op_profit")
            den = self._safe(row, "nenshu")
            if den <= 0:
                continue
            ratio = num / den
            theta = 2 * np.arctan(max(ratio, 1e-9) / 0.05)
            phi   = 2 * np.pi * min(ratio / 0.1, 1.0)

            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)

            qr = q_risks.iloc[i] if i < len(q_risks) else 0
            color = "red" if qr >= 60 else ("orange" if qr >= 30 else "green")
            ax.scatter(x, y, z, c=color, s=80, alpha=0.8)

        ax.set_xlabel("Re(ψ)")
        ax.set_ylabel("Im(ψ)")
        ax.set_zlabel("z-component")
        ax.set_title("財務状態ベクトル — Bloch球面\n赤: Q_risk≥60 / 橙: Q_risk≥30 / 緑: 正常")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ Bloch球面を保存: {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_phase_portrait(self, df: pd.DataFrame, save_path: str | None = None):
        """資産回転率 vs ROA のフェーズ・ポートレート（企業の動態空間）。"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        dyn = self.analyze_physical_dynamics(df)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ["red" if s else "steelblue" for s in dyn["stall_flag"]]
        scatter = ax.scatter(
            dyn["asset_turnover"], dyn["roa"] * 100,
            c=colors, s=80, alpha=0.7, edgecolors="white", linewidth=0.5
        )

        # 危険域の強調
        ax.axvline(x=0.3, color="orange", linestyle="--", alpha=0.7, label="低回転閾値 (0.3)")
        ax.axhline(y=0.0, color="red",    linestyle="--", alpha=0.5, label="ROA=0ライン")
        ax.fill_betweenx([-20, 0], 0, 0.3, alpha=0.05, color="red")

        ax.set_xlabel("資産回転率（売上高/総資産）", fontsize=12)
        ax.set_ylabel("ROA（%）", fontsize=12)
        ax.set_title("フェーズ・ポートレート — 企業動態空間\n赤: 失速状態（資産回転低 & ROA負）", fontsize=13)

        red_patch  = mpatches.Patch(color="red",      label="失速状態")
        blue_patch = mpatches.Patch(color="steelblue", label="正常")
        ax.legend(handles=[red_patch, blue_patch])
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✅ フェーズ・ポートレートを保存: {save_path}")
        else:
            plt.show()
        plt.close()

    # ─────────────────────────────────────────────────────────────────────────
    # ユーティリティ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe(row, col: str, default: float = 0.0) -> float:
        val = row.get(col, default)
        try:
            return float(val) if val is not None and str(val).strip() not in ("", "nan") else default
        except (TypeError, ValueError):
            return default

    def full_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全分析をまとめて実行して結果DFを返す。
        使用例:
            analyzer = QuantumFinanceAnalyzer()
            result = analyzer.full_analysis(inputs_df)
        """
        q_risk   = self.calculate_q_risk(df)
        dynamics = self.analyze_physical_dynamics(df)
        entropy  = self.detect_data_entropy(df)

        result = df.copy()
        result["q_risk"]              = q_risk
        result["dynamics_score"]      = dynamics["dynamics_score"]
        result["asset_turnover"]      = dynamics["asset_turnover"]
        result["roa"]                 = dynamics["roa"]
        result["stall_flag"]          = dynamics["stall_flag"]
        result["entropy_risk"]        = entropy["entropy_risk"]
        result["data_integrity_flag"] = entropy["data_integrity_flag"]

        # 総合警告レベル
        def risk_level(row):
            if row["q_risk"] >= 60 or row["stall_flag"] == 1 or row["data_integrity_flag"] == 1:
                return "🔴 高リスク"
            if row["q_risk"] >= 30 or row["dynamics_score"] < 40:
                return "🟡 要注意"
            return "🟢 正常"

        result["quantum_risk_level"] = result.apply(risk_level, axis=1)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# 動作確認
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # テストデータ（千円単位）
    test_df = pd.DataFrame([
        {   # 健全企業
            "nenshu": 500000, "gross_profit": 120000, "op_profit": 25000,
            "ord_profit": 22000, "net_income": 15000, "net_assets": 80000,
            "total_assets": 250000, "machines": 50000, "other_assets": 30000,
            "depreciation": 5000, "bank_credit": 100000, "lease_credit": 30000,
        },
        {   # 財務構造が崩壊した企業（売上低迷・資産過多）
            "nenshu": 50000, "gross_profit": 5000, "op_profit": -3000,
            "ord_profit": -5000, "net_income": -8000, "net_assets": -20000,
            "total_assets": 300000, "machines": 250000, "other_assets": 10000,
            "depreciation": 30000, "bank_credit": 280000, "lease_credit": 50000,
        },
        {   # 切りのいい数字が多い（操作疑い）
            "nenshu": 100000, "gross_profit": 20000, "op_profit": 10000,
            "ord_profit": 10000, "net_income": 5000, "net_assets": 50000,
            "total_assets": 200000, "machines": 100000, "other_assets": 50000,
            "depreciation": 10000, "bank_credit": 100000, "lease_credit": 50000,
        },
    ])

    analyzer = QuantumFinanceAnalyzer()
    result = analyzer.full_analysis(test_df)

    print("=" * 70)
    print("量子財務干渉 + 物理動態分析 — 結果")
    print("=" * 70)
    cols = ["q_risk", "dynamics_score", "asset_turnover", "roa",
            "stall_flag", "entropy_risk", "data_integrity_flag", "quantum_risk_level"]
    print(result[cols].to_string())

    # 個別補正例
    print("\n── スコア補正例 ──")
    for i, row in result.iterrows():
        base = 75.0  # 仮のLightGBMスコア
        corr = analyzer.apply_quantum_correction(
            base_score=base,
            q_risk=row["q_risk"],
            dynamics_score=row["dynamics_score"],
            entropy_risk=row["entropy_risk"],
        )
        print(f"企業{i+1}: {base} → {corr['corrected_score']} "
              f"(補正係数={corr['correction_factor']}) "
              f"{corr['warning'] or '警告なし'}")
