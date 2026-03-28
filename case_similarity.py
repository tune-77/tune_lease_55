import numpy as np
import json
from typing import List, Dict, Any

class CaseSimilarityEngine:
    """
    案件間の類似度を計算し、過去事例から類似案件を抽出するエンジン。
    """

    # 類似度計算に使用する財務・属性項目と重み
    FEATURES = {
        "sales_log": 0.20,       # 売上規模（対数）
        "op_profit_margin": 0.15, # 営業利益率
        "equity_ratio": 0.25,     # 自己資本比率
        "debt_to_sales": 0.15,    # 債務売上比率
        "industry_match": 0.25,   # 業種の一致度
    }

    def __init__(self, past_cases: List[Dict[str, Any]]):
        self.past_cases = past_cases

    def _extract_features(self, case_data: Dict[str, Any]) -> np.ndarray:
        """
        案件データから特徴量ベクトルを抽出。
        """
        # 財務数値の取得
        revenue = float(case_data.get("nenshu", 0) or 0)
        op_profit = float(case_data.get("op_profit", 0) or 0)
        equity_ratio = float(case_data.get("equity_ratio", 0) or case_data.get("user_eq", 0) or 0)
        total_debt = float(case_data.get("bank_credit", 0) or 0) + float(case_data.get("lease_credit", 0) or 0)

        # 指標の計算と正規化（簡易的なクリッピングとスケーリング）
        f_sales = np.log1p(max(revenue, 0)) / 25.0  # 100億で約0.9
        f_margin = np.clip(op_profit / (revenue + 1e-6), -0.2, 0.4) * 2.5 + 0.5 # -20%~40% -> 0~2
        f_equity = np.clip(equity_ratio, -0.1, 0.8) * 1.2 + 0.1 # -10%~80% -> 0~1
        f_debt = np.clip(total_debt / (revenue + 1e-6), 0, 2.0) / 2.0 # 0~200% -> 0~1

        return np.array([f_sales, f_margin, f_equity, f_debt])

    def _analyze_conditions(self, data: Dict[str, Any]) -> List[str]:
        """
        成約の「決め手」となった条件を特定する。
        成約登録時に手動で入力された loan_conditions を最優先する。
        """
        conditions = []
        
        # 1. 成約登録時に入力された確定条件を優先
        actual_conds = data.get("loan_conditions")
        if actual_conds and isinstance(actual_conds, list):
            conditions.extend(actual_conds)
            
        # 既に抽出された条件（重複）を避けるためのセット
        existing = set(conditions)

        # 2. 定性フラグ（BN項目）からの推論（補足）
        if (data.get("Parent_Guarantor") or data.get("bn_s_parent")) and "親会社等保証" not in existing:
            conditions.append("親会社保証")
        if (data.get("Main_Bank_Support") or data.get("bn_s_main_bank")) and "金融機関と協調" not in existing:
            conditions.append("メイン銀行支援")
        if (data.get("Related_Assets") or data.get("bn_s_rel_assets")) and "担保・保全あり" not in existing:
            conditions.append("関連資産の担保/保全")
        if (data.get("Co_Lease") or data.get("bn_s_co_lease")) and "金融機関と協調" not in existing:
            conditions.append("協調リース（共同与信）")
        
        # 期間短縮（例: 標準36ヶ月より短い）
        term = data.get("lease_term") or data.get("lease_months")
        if term and int(term) <= 24:
            conditions.append(f"期間短縮({term}ヶ月)")
            
        # 自己資金投入（頭金相当）
        if data.get("down_payment") or data.get("bn_s_down_payment"):
            conditions.append("頭金/自己資金")

        return conditions

    def find_similar(self, current_case: Dict[str, Any], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        現在の案件に類似する過去案件を抽出する。
        (承認・成約済みの案件のみを対象とする)
        """
        # 成功事例のみにフィルタリング
        success_cases = []
        for p in self.past_cases:
            data = p.get("data")
            if isinstance(data, str):
                try: data = json.loads(data)
                except: continue
            elif not data: data = p
            
            status = data.get("final_status", "")
            if "成約" in status or "承認" in status:
                success_cases.append(p)
        
        if not success_cases:
            return []

        curr_vec = self._extract_features(current_case)
        curr_ind = current_case.get("industry_sub", "")

        scored_cases = []
        for past in success_cases:
            # 過去データの取得
            data = past.get("data")
            if isinstance(data, str):
                try: data = json.loads(data)
                except: continue
            elif not data: data = past

            past_vec = self._extract_features(data)
            past_ind = data.get("industry_sub", "")

            # 財務ベクトルのコサイン類似度（簡易的にユークリッド距離の逆数を使用）
            dist = np.linalg.norm(curr_vec - past_vec)
            financial_sim = 1.0 / (1.0 + dist)

            # 業種一致ボーナス
            industry_sim = 1.0 if curr_ind == past_ind else 0.0

            # 総合スコア（重み付け）
            total_sim = (financial_sim * 0.7) + (industry_sim * 0.3)
            
            # 成約条件の分析
            status = data.get("final_status", "")
            is_success = "成約" in status or "承認" in status
            conditions = self._analyze_conditions(data) if is_success else []

            scored_cases.append({
                "case": data,
                "similarity": total_sim,
                "financial_sim": financial_sim,
                "industry_match": curr_ind == past_ind,
                "is_success": is_success,
                "conditions": conditions
            })

        # 類似度順にソート
        scored_cases.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_cases[:top_n]
