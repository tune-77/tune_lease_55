"""
quantum_finance_engine.py
────────────────────────────────────────────────────────────
⚛️ 業界標準リファレンス型・量子財務干渉（Q_risk）計算エンジン
複素ベクトル（フェーザ）を用いた財務構造の論理的矛盾検知モジュール。
────────────────────────────────────────────────────────────
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class QuantumFinanceEngine:
    def __init__(self):
        # 業界別標準比率（e-Stat等の簡易ベンチマーク）
        # 比率の定義: [売上高/総資産, 純資産/総資産, 営業利益/総資産, 総負債/総資産]
        self.benchmarks = {
            "D": {"sales_assets": 1.10, "equity_assets": 0.30, "profit_assets": 0.04, "debt_assets": 0.70}, # 建設業
            "E": {"sales_assets": 0.90, "equity_assets": 0.40, "profit_assets": 0.05, "debt_assets": 0.60}, # 製造業
            "H": {"sales_assets": 1.20, "equity_assets": 0.25, "profit_assets": 0.03, "debt_assets": 0.75}, # 運輸業
            "P": {"sales_assets": 0.80, "equity_assets": 0.45, "profit_assets": 0.06, "debt_assets": 0.55}, # 医療・福祉
            "I": {"sales_assets": 1.30, "equity_assets": 0.35, "profit_assets": 0.05, "debt_assets": 0.65}, # サービス業
            "DEFAULT": {"sales_assets": 1.00, "equity_assets": 0.30, "profit_assets": 0.05, "debt_assets": 0.70}
        }

    def _get_industry_key(self, industry_code: str) -> str:
        """業種コードの先頭文字からマスタのキーを解決する"""
        if not industry_code or not isinstance(industry_code, str):
            return "DEFAULT"
        
        code = industry_code.strip().upper()[0]
        if code in self.benchmarks:
            return code
        return "DEFAULT"

    def calculate_phasors(self, row: pd.Series, industry_code: str) -> dict:
        """
        個別財務データから各項目の複素ベクトル（Phasor）を算出する。
        numpy.exp(1j * theta) を用いて計算効率を最適化。
        """
        key = self._get_industry_key(industry_code)
        bench = self.benchmarks[key]

        # 財務数値の取得
        sales = float(row.get("nenshu", 0.0))
        assets = float(row.get("total_assets", 1.0))
        equity = float(row.get("net_assets", 0.0))
        profit = float(row.get("op_profit", 0.0))
        debt = max(0.0, assets - equity)

        if assets <= 0:
            assets = 1.0 # ゼロ除算防御

        # 実績比率
        r_sales = sales / assets
        r_equity = equity / assets
        r_profit = profit / assets
        r_debt = debt / assets

        # 位相変換 (Phase Mapping)
        # θ = arctan2(実績, 期待期待値) - π/4
        theta_sales = np.arctan2(r_sales, bench["sales_assets"]) - (np.pi / 4.0)
        theta_equity = np.arctan2(r_equity, bench["equity_assets"]) - (np.pi / 4.0)
        theta_profit = np.arctan2(r_profit, bench["profit_assets"]) - (np.pi / 4.0)
        theta_debt = np.arctan2(r_debt, bench["debt_assets"]) - (np.pi / 4.0)

        # 振幅（正規化絶対値）
        amp_sales = np.clip(sales / (sales + assets + 1.0), 0.1, 2.0)
        amp_equity = np.clip(abs(equity) / (abs(equity) + assets + 1.0), 0.1, 2.0)
        amp_profit = np.clip(abs(profit) / (abs(profit) + assets + 1.0), 0.1, 2.0)
        amp_debt = np.clip(debt / (debt + assets + 1.0), 0.1, 2.0)

        # 複素フェーザ表現: z = A * exp(j*theta)
        z_sales = amp_sales * np.exp(1j * theta_sales)
        z_equity = amp_equity * np.exp(1j * theta_equity)
        z_profit = amp_profit * np.exp(1j * theta_profit)
        z_debt = amp_debt * np.exp(1j * theta_debt)

        return {
            "z_sales": z_sales, "z_equity": z_equity, "z_profit": z_profit, "z_debt": z_debt,
            "theta_sales": theta_sales, "theta_equity": theta_equity, "theta_profit": theta_profit, "theta_debt": theta_debt,
            "amp_sales": amp_sales, "amp_equity": amp_equity, "amp_profit": amp_profit, "amp_debt": amp_debt
        }

    def analyze_interference(self, row: pd.Series, industry_code: str) -> dict:
        """複素ベクトルの相殺的干渉から Q_risk を算出する。"""
        phasors = self.calculate_phasors(row, industry_code)
        
        # 波の合成 (負債は差し引く方向で干渉させる)
        z_total = phasors["z_sales"] + phasors["z_equity"] + phasors["z_profit"] - phasors["z_debt"]

        # 強度の計算
        i_max = abs(phasors["z_sales"]) + abs(phasors["z_equity"]) + abs(phasors["z_profit"]) + abs(phasors["z_debt"])
        i_actual = abs(z_total)

        # Q_risk（虚無エネルギーによる論理的矛盾率）
        if i_max > 0:
            q_risk = ((i_max - i_actual) / i_max) * 100.0
        else:
            q_risk = 0.0

        penalty_factor = 1.0
        if q_risk > 35.0:
            penalty_factor = np.exp(-1.2 * (q_risk - 35.0) / 100.0)

        return {
            "q_risk": np.round(q_risk, 2),
            "i_max": np.round(i_max, 4),
            "i_actual": np.round(i_actual, 4),
            "penalty_factor": np.round(penalty_factor, 4),
            "phasors": phasors
        }

    def visualize_argand_diagram(self, row: pd.Series, industry_code: str):
        """アルガン図（複素平面）上に波の相殺状態をプロットして返却する。"""
        analysis = self.analyze_interference(row, industry_code)
        phasors = analysis["phasors"]
        
        z_sales = phasors["z_sales"]
        z_equity = phasors["z_equity"]
        z_profit = phasors["z_profit"]
        z_debt = phasors["z_debt"]
        
        # 負債は差し引く方向で合成
        z_total = z_sales + z_equity + z_profit - z_debt

        fig, ax = plt.subplots(figsize=(4, 4))

        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1A1C24")

        # 原点
        ax.axhline(0, color="#31363F", alpha=0.8)
        ax.axvline(0, color="#31363F", alpha=0.8)

        # ベクトルの連鎖描画（矢印）
        p1 = (z_sales.real, z_sales.imag)
        p2 = (p1[0] + z_equity.real, p1[1] + z_equity.imag)
        p3 = (p2[0] + z_profit.real, p2[1] + z_profit.imag)

        ax.quiver(0, 0, z_sales.real, z_sales.imag, angles='xy', scale_units='xy', scale=1, 
                  color="#00F5FF", width=0.015, label="売上")
        ax.quiver(p1[0], p1[1], z_equity.real, z_equity.imag, angles='xy', scale_units='xy', scale=1, 
                  color="#BD00FF", width=0.015, label="純資産")
        ax.quiver(p2[0], p2[1], z_profit.real, z_profit.imag, angles='xy', scale_units='xy', scale=1, 
                  color="#00FF66", width=0.015, label="利益")
        ax.quiver(p3[0], p3[1], -z_debt.real, -z_debt.imag, angles='xy', scale_units='xy', scale=1, 
                  color="#FFCC00", width=0.015, label="負債")

        # 合成結果のベクトル
        ax.quiver(0, 0, z_total.real, z_total.imag, angles='xy', scale_units='xy', scale=1, 
                  color="#FF007A", width=0.01, label="最終合成")

        # 理想最大（相殺なし）の円
        i_max = analysis["i_max"]
        circle = plt.Circle((0, 0), i_max, color="#00F5FF", fill=False, linestyle=":", alpha=0.3, label="最大理論強度")
        ax.add_patch(circle)

        # グラフ範囲調整
        r_max = max(i_max * 1.2, 0.5)
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)

        ax.set_xlabel("実部 (Real)", color="#FFFFFF", fontsize=9)
        ax.set_ylabel("虚部 (Imag)", color="#FFFFFF", fontsize=9)
        ax.set_title(f"量子財務干渉 アルガン図 (Q_risk: {analysis['q_risk']}%)", color="#FFFFFF", pad=15)
        ax.tick_params(colors="#888888", labelsize=8)
        ax.grid(True, color="#31363F", alpha=0.5)

        legend = ax.legend(facecolor="#0E1117", edgecolor="#31363F", loc="upper left", fontsize=8)
        plt.setp(legend.get_texts(), color="#FFFFFF")

        return fig

    def visualize_wave_interference(self, row: pd.Series, industry_code: str):
        """時間軸上で実際の正弦波がどう干渉（相殺・増幅）しているかを描画する。"""
        analysis = self.analyze_interference(row, industry_code)
        phasors = analysis["phasors"]

        amp_sales = phasors["amp_sales"]
        theta_sales = phasors["theta_sales"]
        amp_equity = phasors["amp_equity"]
        theta_equity = phasors["theta_equity"]
        amp_profit = phasors["amp_profit"]
        theta_profit = phasors["theta_profit"]
        amp_debt = phasors["amp_debt"]
        theta_debt = phasors["theta_debt"]

        # 時間軸 (1周期分)
        t = np.linspace(0, 2 * np.pi, 500)
        
        # 各波 (負債は差し引く方向で干渉)
        y_sales = amp_sales * np.sin(t + theta_sales)
        y_equity = amp_equity * np.sin(t + theta_equity)
        y_profit = amp_profit * np.sin(t + theta_profit)
        y_debt = amp_debt * np.sin(t + theta_debt)
        
        y_total = y_sales + y_equity + y_profit - y_debt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#1A1C24")

        # 描画
        ax.plot(t, y_sales, color="#00F5FF", linewidth=1.5, alpha=0.8, label="売上波動")
        ax.plot(t, y_equity, color="#BD00FF", linewidth=1.5, alpha=0.8, label="純資産波動")
        ax.plot(t, y_profit, color="#00FF66", linewidth=1.5, alpha=0.8, label="利益波動")
        ax.plot(t, y_debt, color="#FFCC00", linewidth=1.5, alpha=0.8, label="負債波動")
        ax.plot(t, y_total, color="#FF007A", linewidth=3, linestyle="-", label="最終合成波動")

        # ゼロライン
        ax.axhline(0, color="#31363F", alpha=0.8)

        ax.set_xlim(0, 2 * np.pi)
        ax.set_xlabel("位相 (Time/Phase)", color="#FFFFFF", fontsize=8)
        ax.set_ylabel("振幅 (Amplitude)", color="#FFFFFF", fontsize=8)
        ax.set_title(f"財務波動干渉シミュレーション (相殺率: {analysis['q_risk']}%)", color="#FFFFFF", pad=12, fontsize=10)
        
        ax.tick_params(colors="#888888", labelsize=8)
        ax.grid(True, color="#31363F", alpha=0.4)
        
        legend = ax.legend(facecolor="#0E1117", edgecolor="#31363F", loc="upper right", fontsize=8)
        plt.setp(legend.get_texts(), color="#FFFFFF")

        return fig

    def run_on_ibm_quantum(self, row: pd.Series, industry_code: str, api_token: str = None) -> dict:
        """
        Qiskit を用いて財務データを量子回路にエンコードし、量子干渉を観測する。
        APIトークンがある場合は実機への接続インターフェースを提供。
        """
        phasors = self.calculate_phasors(row, industry_code)
        theta_s = phasors["theta_sales"]
        theta_e = phasors["theta_equity"]
        amp_s = phasors["amp_sales"]
        amp_e = phasors["amp_equity"]

        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
        except ImportError:
            return {
                "error": "Qiskit がインストールされていません。\n`pip install qiskit` を実行してください。",
                "qrisk_sim": 0.0
            }

        # 4量子ビット回路
        # Qubit 0:売上, Qubit 1:純資産, Qubit 2:営業利益, Qubit 3:総負債
        qc = QuantumCircuit(4)
        
        metrics = [
            (phasors["amp_sales"], phasors["theta_sales"]),
            (phasors["amp_equity"], phasors["theta_equity"]),
            (phasors["amp_profit"], phasors["theta_profit"]),
            (phasors["amp_debt"], phasors["theta_debt"])
        ]
        
        for i, (amp, theta) in enumerate(metrics):
            qc.ry(2 * np.arctan2(amp, 1.0), i)
            qc.p(theta, i)
            
        # 4つの量子を絡み合わせる（干渉の生成）
        for i in range(4):
            qc.h(i)
        for i in range(3):
            qc.cx(i, i+1)
            
        # 状態ベクトルのシミュレーション
        state = Statevector.from_instruction(qc)
        probs = state.probabilities_dict()

        msg = "🤖 Qiskit ローカル量子シミュレータで動作中"
        if api_token and len(api_token.strip()) > 10:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
                
                # IBM Quantum サービスにログイン
                service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_token)
                
                # 稼働中で最も待ち時間が少ない実機を取得
                backend = service.least_busy(operational=True, simulator=False)
                
                # 測定ゲートの追加
                qc_real = qc.copy()
                qc_real.measure_all()
                
                # ⚠️ ISA (Instruction Set Architecture) トランスパイルの実行
                # 近年のIBM実機はハードウェアが直接理解できるゲートのみを受け付けます
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
                pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
                isa_circuit = pm.run(qc_real)
                
                # 実機への送信（非同期）
                sampler = SamplerV2(mode=backend)
                job = sampler.run([isa_circuit])
                
                msg = (
                    f"🚀 **IBM Quantum 実機 ({backend.name}) へジョブ送信成功！**\n\n"
                    f"- **Job ID**: `{job.job_id()}`\n"
                    f"- 現在のキューに登録されました。実機からの観測結果の返却には数分〜数十分かかります。\n"
                    f"- 進捗状況は、ご自身の [IBM Quantum Dashboard](https://quantum.ibm.com/jobs) からリアルタイムで追跡可能です。"
                )
            except Exception as e:
                msg = f"⚠️ **実機送信エラー**: `{str(e)}` \n\nAPIキーまたは `qiskit-ibm-runtime` ライブラリの不足です。ローカルで代替シミュレートしました。"


        # 状態 '1111' (4指標すべてが反転している最悪状態) の確率を量子版のQ_riskとする
        qrisk_quantum = np.round(probs.get('1111', 0.0) * 100.0, 2)

        return {
            "qrisk_quantum": qrisk_quantum,
            "probabilities": probs,
            "circuit_str": str(qc.draw(output='text')),
            "msg": msg
        }

    def retrieve_ibm_quantum_result(self, job_id: str, api_token: str) -> dict:
        """Job ID を指定して IBM Quantum から実行結果を取得する"""
        import numpy as np
        if not api_token or not job_id:
            return {"error": "APIトークンおよびJob IDが必要です。"}
            
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=api_token)
            job = service.job(job_id.strip())
            
            status = job.status()
            
            # 文字列に変換して比較
            status_str = str(status)
            
            if "DONE" in status_str:
                result = job.result()
                # SamplerV2 の結果パース
                pub_result = result[0]
                # meas または クラシカルレジスタ名
                data_keys = list(pub_result.data.keys())
                if not data_keys:
                    return {"error": "測定データが見つかりません。"}
                    
                counts = pub_result.data[data_keys[0]].get_counts()
                
                # 確率に変換
                total_shots = sum(counts.values())
                probs = {k: np.round(v / total_shots, 4) for k, v in counts.items()}
                
                qrisk = np.round(probs.get('1111', 0.0) * 100.0, 2)
                
                return {
                    "status": "DONE",
                    "probabilities": probs,
                    "qrisk_quantum": qrisk,
                    "msg": "✨ 実機からデータの取得に成功しました！"
                }
            else:
                return {
                    "status": status_str,
                    "msg": f"⏳ 現在のステータス: `{status_str}` (まだ実行中、またはキュー待ちです)"
                }
        except Exception as e:
            return {"error": f"結果の取得中にエラーが発生しました: {str(e)}"}



