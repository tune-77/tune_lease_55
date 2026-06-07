import sqlite3
import json
import os
import pandas as pd
import numpy as np
from contextlib import closing
from sklearn.metrics import roc_auc_score, classification_report
from runtime_paths import get_data_path

# パスの設定
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)
_DB_PATH = get_data_path("lease_data.db")

if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from components.asset_finance import AssetFinanceEngine

class AssetFinanceBacktester:
    """
    物件スコアリングの重み（財務点、BEP点、優先度点）が
    実態（成約/失注/デフォルト）に即しているかをバックテスト検証するシステム。
    """
    def __init__(self):
        self.engine = AssetFinanceEngine()

    def load_historical_data(self):
        """過去データをDBから抽出"""
        if not os.path.exists(_DB_PATH):
            return pd.DataFrame()
            
        try:
            with closing(sqlite3.connect(_DB_PATH)) as conn:
                query = """
                SELECT id, industry_sub, final_status, data 
                FROM past_cases 
                WHERE final_status IS NOT NULL AND final_status != ''
                """
                df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return pd.DataFrame()

    def run_backtest(self):
        """バックテストの実行"""
        df = self.load_historical_data()
        if df.empty:
            return "【エラー】検証に必要な過去データが不足しています。"

        results = []
        for _, row in df.iterrows():
            case_id = row["id"]
            status = row["final_status"]
            
            # JSONのパース
            data_raw = row["data"]
            if isinstance(data_raw, str):
                try:
                    data = json.loads(data_raw)
                except:
                    continue
            else:
                data = data_raw if data_raw else {}

            # AssetFinanceに必要なパラメータ補正（ない場合のデフォルト値）
            asset_type = data.get("asset_type") or data.get("asset_category")
            if not asset_type or asset_type not in self.engine.ASSET_PARAMS:
                # 類推またはデフォルト
                asset_name = str(data.get("asset_name", "")).lower()
                if "車" in asset_name or "トラック" in asset_name:
                    asset_type = "車両"
                elif "機械" in asset_name or "旋盤" in asset_name:
                    asset_type = "工作機械"
                elif "パソコン" in asset_name or "it" in asset_name:
                    asset_type = "PC/IT"
                else:
                    asset_type = "建機"  # 最多想定

            # スコア用データ辞書の構築
            eval_data = {
                'asset_type': asset_type,
                'term': int(data.get("lease_term") or data.get("lease_months") or 60),
                'down_payment': float(data.get("down_payment_rate") or data.get("down_payment") or 0.0),
                'financial_score': 'High' if float(data.get("score") or 50) > 70 else 'Medium' if float(data.get("score") or 50) > 40 else 'Low',
                'annual_km': float(data.get("annual_km") or 0.0),
                'has_maintenance_lease': bool(data.get("has_maintenance_lease") or False),
                'main_bank_support': bool(data.get("main_bank_support") or False),
                'bank_coordination': bool(data.get("bank_coordination") or False),
                'core_business': bool(data.get("core_business") or True),
                'related_assets': bool(data.get("related_assets") or False),
            }

            try:
                score, bep_month, bep_ratio, _, _, _, _ = self.engine.calculate_score(eval_data)
                decision, _ = self.engine.get_decision(score)
                
                # ラベル（成約=1, その他=0）
                is_success = 1 if "成約" in status or "承認" in status else 0
                
                results.append({
                    "case_id": case_id,
                    "actual_status": status,
                    "is_success": is_success,
                    "calculated_score": score,
                    "ai_decision": decision,
                    "bep_month": bep_month
                })
            except:
                continue

        if not results:
            return "【エラー】検証可能な有効データが見つかりませんでした。"

        res_df = pd.DataFrame(results)
        
        # 統計評価
        total_cases = len(res_df)
        success_rate_ai = (res_df["ai_decision"].isin(["承認", "条件付き承認"])).mean()
        
        # ROC-AUCの計算 (計算スコアが成約/失注をどれだけ分離できているか)
        try:
            auc = roc_auc_score(res_df["is_success"], res_df["calculated_score"])
        except:
            auc = 0.5

        report_str = f"""
=========================================
📊 物件スコアリング・バックテスト結果
=========================================
● 検証案件数: {total_cases}件
● 現行ロジックでの擬似承認率: {success_rate_ai*100:.1f}%

【判別性能 (ROC-AUC)】
● スコアの分離力: {auc:.3f}
  (目安: 0.7以上で良好, 0.8以上で優秀)

【合致度分析】
"""
        # 混同行列
        for dec in ["承認", "条件付き承認", "要審議（上位承認）", "否決"]:
            sub = res_df[res_df["ai_decision"] == dec]
            if len(sub) > 0:
                real_success_pct = sub["is_success"].mean() * 100
                report_str += f" ・ AI判定が「{dec}」の案件の、実際の成約率: {real_success_pct:.1f}% ({len(sub)}件)\n"

        report_str += "\n💡 【改善のアドバイス】\n"
        if auc < 0.65:
            report_str += "⚠️ スコアの分離力が低いです。LGD加点と財務加点のバランスを見直し、リスク案件をより厳しく評価する必要があります。\n"
        else:
            report_str += "✅ 現在の配分は過去実績とおおむね整合しています。\n"

        return report_str

def run_cli_backtest():
    tester = AssetFinanceBacktester()
    print(tester.run_backtest())

if __name__ == "__main__":
    run_cli_backtest()
