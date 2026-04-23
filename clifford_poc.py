import numpy as np

class GeometricAlgebraCore:
    """
    2Dおよび高次元の基本的な幾何代数（Clifford Algebra）演算を提供するクラス。
    ここでは直感的な理解のため、各特徴量を d次元ベクトル として扱い、
    内積（スカラー）と外積（バイベクトル）を計算します。
    """
    @staticmethod
    def inner_product(u: np.ndarray, v: np.ndarray) -> float:
        """
        内積 (u · v):
        2つのベクトルの「整合性」「バランスの良さ」をスカラーで抽出。
        """
        return float(np.dot(u, v))
    
    @staticmethod
    def wedge_product_2d(u: np.ndarray, v: np.ndarray) -> float:
        """
        外積/ウェッジ積 (u ∧ v) の2次元特化版:
        2つのベクトルが張る平行四辺形の「面積（符号付き）」。
        指標間の「構造的ズレ（歪み）」をバイベクトルの大きさとして抽出。
        """
        assert len(u) == 2 and len(v) == 2, "2D vectors required for simple wedge"
        return float(u[0]*v[1] - u[1]*v[0])

    @staticmethod
    def geometric_product_2d(u: np.ndarray, v: np.ndarray) -> tuple[float, float]:
        """
        幾何積 (uv = u·v + u∧v):
        内積成分（スカラー）と外積成分（バイベクトル）を同時に返す。
        """
        inner = GeometricAlgebraCore.inner_product(u, v)
        wedge = GeometricAlgebraCore.wedge_product_2d(u, v)
        return inner, wedge

class CliffordSuccessPredictor:
    """
    CliffordNetの概念を用いた成約最適化予測エンジン。
    No-FFN (Feed Forward Networkなし) で、幾何積のみによる高い表現力を目指す。
    """
    def __init__(self, feature_dim=2):
        self.feature_dim = feature_dim
        # ランダムな重みで各スカラー指標をd次元ベクトルにマッピングする行列
        # 実際には学習されるパラメータ
        self.embedding_weights = {
            'sales_growth': np.array([0.8, 0.2]),
            'capital_ratio': np.array([0.5, 0.5]),
            'liquidity': np.array([0.2, 0.8]),
            'years_in_business': np.array([0.9, -0.1]),
            'operating_margin': np.array([0.6, 0.4]), # 新規追加：営業利益率
            'asset_turnover': np.array([0.4, 0.6])    # 新規追加：総資産回転率
        }
        
    def vectorize(self, features: dict) -> dict:
        """ 1. ベクトル化: スカラー指標をベクトル空間へマッピング """
        vectors = {}
        for key, value in features.items():
            if key in self.embedding_weights:
                vectors[key] = self.embedding_weights[key] * value
        return vectors
        
    def sparse_rolling_interaction(self, vectors: dict) -> list:
        """ 
        2. ローリング相互作用: 
        隣り合う（意味的に関連の強い）指標間の幾何積のみを計算し計算負荷を抑える。
        """
        keys = list(vectors.keys())
        interactions = []
        for i in range(len(keys) - 1):
            u = vectors[keys[i]]
            v = vectors[keys[i+1]]
            # 幾何積の計算
            inner, wedge = GeometricAlgebraCore.geometric_product_2d(u, v)
            interactions.append({
                'pair': f"{keys[i]} x {keys[i+1]}",
                'inner_scalar': inner,
                'wedge_bivector': wedge
            })
        return interactions
        
    def gated_geometric_residual(self, interactions: list) -> float:
        """
        3. Gated Geometric Residual (GGR):
        幾何的相互作用を非線形ゲート（Sigmoid等）に通し、成約スコアを算出。
        """
        score = 0.0
        for inter in interactions:
            # 内積（整合性）をベーススコアに
            base = inter['inner_scalar']
            # 外積（歪み）をゲートとして機能させる（ここでは簡略化したシグモイド）
            gate = 1 / (1 + np.exp(-inter['wedge_bivector']))
            # Residualを加味したスコアリング (No-FFN)
            score += base * gate
            
        # 最終確率に正規化 (0.0 ~ 1.0)
        prob = 1 / (1 + np.exp(-score * 0.1))
        return prob

    def predict(self, features: dict):
        vectors = self.vectorize(features)
        interactions = self.sparse_rolling_interaction(vectors)
        prob = self.gated_geometric_residual(interactions)
        
        return {
            'success_probability': prob,
            'vectors': vectors,
            'interactions': interactions
        }

class CliffordVisualizerLogic:
    """
    Flet等のCanvasで「成約の形」をリアルタイム描画するための座標計算ロジック。
    """
    def __init__(self, center_x=200, center_y=200, scale=100):
        self.cx = center_x
        self.cy = center_y
        self.scale = scale
        
    def calculate_golden_area(self) -> list:
        """ 成約の黄金面積 (基準となる幾何学的形状: 理想的な多角形) """
        # 例として理想的な指標値が全て1.0とした場合のベクトル先端座標
        ideal_vectors = [
            np.array([0.8, 0.2]),   # sales
            np.array([-0.5, 0.5]),  # capital (座標系で見やすくするため向きを調整)
            np.array([-0.2, -0.8]), # liquidity
            np.array([0.9, -0.1])   # years
        ]
        return self._vectors_to_canvas_points(ideal_vectors)
        
    def calculate_current_state(self, vectors_dict: dict) -> list:
        """ 現在の案件の形状 (動的歪み描画用) """
        # 各ベクトルを描画用に少し回転・配置させる
        points = []
        vec_list = list(vectors_dict.values())
        # 四象限に散らすための回転行列などを用いて視覚化を工夫
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        
        canvas_vectors = []
        for i, v in enumerate(vec_list):
            theta = angles[i % 4]
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)]
            ])
            rotated_v = np.dot(rot, v)
            canvas_vectors.append(rotated_v)
            
        return self._vectors_to_canvas_points(canvas_vectors)
        
    def calculate_guide_vectors(self, current_points: list, golden_points: list) -> list:
        """ 成約へのガイド (どの方向に指標を調整すべきかのベクトル) """
        guides = []
        for curr, gold in zip(current_points, golden_points):
            guides.append({
                'start': curr,
                'end': gold,
                'delta': (gold[0] - curr[0], gold[1] - curr[1])
            })
        return guides
        
    def _vectors_to_canvas_points(self, vectors: list) -> list:
        points = []
        for v in vectors:
            x = self.cx + v[0] * self.scale
            y = self.cy - v[1] * self.scale # Y軸反転
            points.append((x, y))
        return points

if __name__ == "__main__":
    # --- テスト実行 ---
    print("=== CliffordNet Success Predictor POC ===")
    predictor = CliffordSuccessPredictor()
    
    # ダミー財務データ (標準化済み想定)
    sample_deal = {
        'sales_growth': 1.2,
        'capital_ratio': 0.8,
        'liquidity': 0.5,
        'years_in_business': 1.5
    }
    
    result = predictor.predict(sample_deal)
    print(f"成約確率 (CliffordNet Score): {result['success_probability']:.2%}\n")
    
    print("[相互作用 (Geometric Products)]")
    for inter in result['interactions']:
        print(f"  {inter['pair']}:")
        print(f"    内積(整合性) = {inter['inner_scalar']:.3f}")
        print(f"    外積(歪み)   = {inter['wedge_bivector']:.3f}")

    print("\n[Canvas用描画座標計算]")
    vis_logic = CliffordVisualizerLogic()
    golden = vis_logic.calculate_golden_area()
    current = vis_logic.calculate_current_state(result['vectors'])
    guides = vis_logic.calculate_guide_vectors(current, golden)
    
    print(f"  黄金面積座標: {golden}")
    print(f"  現在の案件座標: {current}")
    print(f"  ガイドベクトル[0]の差分: {guides[0]['delta']}")
