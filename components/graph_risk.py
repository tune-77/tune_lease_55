import json
import os
import random
import math
from typing import Dict, List, Tuple, Any

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_GRAPH_FILE = os.path.join(os.path.dirname(_BASE_DIR), "data", "industry_graph.json")
_TRENDS_FILE = os.path.join(os.path.dirname(_BASE_DIR), "data", "industry_trends_extended.json")

class GraphRiskEngine:
    def __init__(self):
        self.graph = self._load_json(_GRAPH_FILE)
        self.trends = self._load_json(_TRENDS_FILE)
        self.nodes = self.graph.get("industries", [])
        self.edges = self.graph.get("edges", [])

    def _load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def get_industry_base_risk(self, industry_name: str) -> float:
        """
        特定の業界のベースリスク（キーワードなどからの推定値）を返す。
        0.0 (安全) 〜 1.0 (極めて危険)
        """
        trend = self.trends.get(industry_name, {})
        text = trend.get("text", "").lower()
        if not text:
            return 0.1 # デフォルト

        # シンプルなネガティブキーワード判定
        risk_score = 0.2
        bad_words = ["倒産", "不足", "減少", "赤字", "課題", "懸念", "不透明", "問題", "物価高", "円安"]
        for word in bad_words:
            if word in text:
                risk_score += 0.08
        
        # 特定の緊急トピック
        if "2024年問題" in text or "2025年問題" in text:
            risk_score += 0.15
            
        return min(0.9, risk_score)

    def calculate_network_risk(self, target_industry: str) -> Dict:
        """
        グラフ理論に基づき、隣接業界からのリスク伝播を計算する。
        """
        if target_industry not in self.nodes:
            # 部分一致検索
            matches = [n for n in self.nodes if target_industry in n]
            if matches:
                 target_industry = matches[0]
            else:
                 return {"network_risk_score": 0.05, "reason": "No graph data"}

        # 1. 自身のベースリスク
        base_risk = self.get_industry_base_risk(target_industry)
        
        # 2. 隣接ノード（ソース）からの影響を計算
        # target_industry が source ではなく target になっているエッジを探す
        # (例: A -> B のとき、Aの不調はBに伝播する)
        impacts = []
        total_propagation = 0.0
        
        for edge in self.edges:
            if edge["target"] == target_industry:
                source = edge["source"]
                weight = edge.get("weight", 0.5)
                source_risk = self.get_industry_base_risk(source)
                
                # 重みが大きいほど、ソースのリスクがターゲットに強く伝播する
                propagation = source_risk * weight
                impacts.append({
                    "from": source,
                    "source_risk": source_risk,
                    "weight": weight,
                    "impact": propagation
                })
                total_propagation += propagation

        # 最終スコア。自身のベースと他者からの影響の加重平均的な何か
        # ここでは単純に max(base, propagated) or sum 的なロジック
        network_risk_score = min(0.95, base_risk + (total_propagation * 0.3))

        return {
            "target_industry": target_industry,
            "base_risk": base_risk,
            "network_risk_score": network_risk_score,
            "impacted_by": sorted(impacts, key=lambda x: x["impact"], reverse=True),
            "summary": f"{target_industry}のネットワークリスクは {network_risk_score:.2f} です。"
        }

    def calculate_centrality(self) -> Dict[str, float]:
        """
        固有ベクトル中心性（Eigenvector Centrality）を計算し、
        各業界の「システム的な重要度（リスクのハブ度）」を算出する。
        """
        # 初期の中心性を 1.0 で初期化
        centrality = {node: 1.0 for node in self.nodes}
        
        # べき乗法（Power Iteration）で最大固有値ベクトルを求める（簡易版 10回イテレーション）
        for _ in range(10):
            new_centrality = {node: 0.0 for node in self.nodes}
            for edge in self.edges:
                source = edge["source"]
                target = edge["target"]
                weight = edge.get("weight", 0.5)
                # 依存関係の向き（source がこけると target がこける）を考慮
                if source in centrality and target in centrality:
                    new_centrality[target] += centrality[source] * weight
            
            # 正規化 (L2ノルム)
            norm = math.sqrt(sum(v**2 for v in new_centrality.values()))
            if norm == 0: break
            centrality = {node: v / norm for node, v in new_centrality.items()}
            
        return centrality

    def run_scenario_simulation(self, target_industry: str, n_simulations: int = 500) -> Dict[str, Any]:
        """
        モンテカルロ法を用いて、不確実性下でのリスク波及をシミュレーションする。
        """
        if target_industry not in self.nodes:
            return {"error": "Industry not found"}

        results = []
        for _ in range(n_simulations):
            # 1. 各ノードのベースリスクにノイズ（ボラティリティ）を加える
            temp_risks = {}
            for node in self.nodes:
                base = self.get_industry_base_risk(node)
                # ±20% のランダムな変動を加える
                noise = random.uniform(0.8, 1.2)
                temp_risks[node] = min(0.99, max(0.01, base * noise))
            
            # 2. ターゲット業界への伝播を計算
            total_propagation = 0.0
            for edge in self.edges:
                if edge["target"] == target_industry:
                    source = edge["source"]
                    weight = edge.get("weight", 0.5)
                    # 伝播の不確実性（サプライチェーンの寸断確率など）
                    uncertainty = random.uniform(0.7, 1.3)
                    total_propagation += temp_risks[source] * weight * uncertainty
            
            sim_score = min(0.99, temp_risks[target_industry] + (total_propagation * 0.3))
            results.append(sim_score)
        
        results.sort()
        mean_risk = sum(results) / len(results)
        var_95 = results[int(len(results) * 0.95)] # 95% 信頼区間の最大値（VaR）
        
        return {
            "target_industry": target_industry,
            "mean_risk": mean_risk,
            "max_risk_95": var_95, # 最悪シナリオ期待値
            "min_risk": results[0],
            "distribution": results[::len(results)//10] if len(results) > 10 else results # 10分位数サンプル
        }

if __name__ == "__main__":
    engine = GraphRiskEngine()
    # テスト
    res = engine.calculate_network_risk("07 職別工事業")
    print(json.dumps(res, ensure_ascii=False, indent=2))
