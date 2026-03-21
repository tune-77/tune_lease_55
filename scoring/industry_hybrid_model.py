"""
業種別のハイブリッドモデル

各業種ごとに異なる回帰係数を持つ場合に対応
例: 製造業、建設業、サービス業でそれぞれ異なる審査基準
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
import joblib

class IndustrySpecificHybridModel:
    """業種別ハイブリッドモデル"""
    
    def __init__(self):
        self.industry_models = {}  # 業種ごとのモデル
        self.industry_coefficients = {}  # 業種ごとの係数
        self.default_model = None  # デフォルトモデル（業種不明用）
        self.ai_models = {}  # 業種ごとのAIモデル
        
    def load_industry_coefficients(
        self,
        industry_coefficients: Dict[str, Dict[str, float]],
        industry_intercepts: Dict[str, float] = None
    ):
        """
        業種別の回帰係数を読み込み
        
        Args:
            industry_coefficients: {
                '製造業': {'ROA': 0.15, 'equity_ratio': 0.08, ...},
                '建設業': {'ROA': 0.12, 'equity_ratio': 0.10, ...},
                ...
            }
            industry_intercepts: {
                '製造業': -2.5,
                '建設業': -2.8,
                ...
            }
        
        Examples:
            >>> coefficients = {
            ...     '製造業': {
            ...         'ROA': 0.15,
            ...         'equity_ratio': 0.08,
            ...         'machinery_ratio': -0.05
            ...     },
            ...     '建設業': {
            ...         'ROA': 0.12,
            ...         'equity_ratio': 0.10,
            ...         'machinery_ratio': -0.08
            ...     }
            ... }
            >>> model.load_industry_coefficients(coefficients)
        """
        self.industry_coefficients = industry_coefficients
        self.industry_intercepts = industry_intercepts or {}
        
        print(f"✅ 業種別係数を読み込み: {len(industry_coefficients)} 業種")
        for industry in industry_coefficients.keys():
            n_vars = len(industry_coefficients[industry])
            intercept = self.industry_intercepts.get(industry, 0.0)
            print(f"   - {industry}: {n_vars}変数, 切片={intercept:.3f}")
    
    def get_industry_importance(self, industry: str) -> pd.DataFrame:
        """
        特定業種の特徴量重要度を取得
        
        Args:
            industry: 業種名
        
        Returns:
            特徴量重要度のDataFrame
        """
        if industry not in self.industry_coefficients:
            raise ValueError(f"業種 '{industry}' の係数がありません")
        
        coeffs = self.industry_coefficients[industry]
        
        importance = pd.DataFrame({
            'feature': list(coeffs.keys()),
            'coefficient': list(coeffs.values()),
            'abs_coefficient': [abs(v) for v in coeffs.values()],
            'direction': ['positive' if v > 0 else 'negative' for v in coeffs.values()]
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance
    
    def compare_industries(self) -> pd.DataFrame:
        """
        業種間で係数を比較
        
        Returns:
            業種別の係数比較表
        """
        # 全業種で共通する変数を抽出
        all_features = set()
        for coeffs in self.industry_coefficients.values():
            all_features.update(coeffs.keys())
        
        comparison = []
        for feature in all_features:
            row = {'feature': feature}
            for industry, coeffs in self.industry_coefficients.items():
                row[industry] = coeffs.get(feature, np.nan)
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # 業種間の差異（標準偏差）を計算
        industry_cols = [col for col in df.columns if col != 'feature']
        df['std_across_industries'] = df[industry_cols].std(axis=1)
        
        return df.sort_values('std_across_industries', ascending=False)
    
    def predict_by_industry(
        self,
        X: pd.DataFrame,
        industry_column: str = 'industry'
    ) -> np.ndarray:
        """
        業種別に予測
        
        Args:
            X: 特徴量DataFrame（industry列を含む）
            industry_column: 業種カラムの名前
        
        Returns:
            予測確率
        """
        if industry_column not in X.columns:
            raise ValueError(f"カラム '{industry_column}' が見つかりません")
        
        predictions = np.zeros(len(X))
        
        for industry in X[industry_column].unique():
            mask = X[industry_column] == industry
            
            if industry in self.industry_coefficients:
                # 業種別の係数で予測
                coeffs = self.industry_coefficients[industry]
                intercept = self.industry_intercepts.get(industry, 0.0)
                
                # 利用可能な変数のみ使用
                available_features = [f for f in coeffs.keys() if f in X.columns]
                
                score = intercept
                for feature in available_features:
                    score += X.loc[mask, feature] * coeffs[feature]
                
                # ロジスティック変換
                predictions[mask] = 1 / (1 + np.exp(-score))
            else:
                print(f"⚠️  業種 '{industry}' の係数がありません（デフォルト使用）")
                # デフォルトモデルまたは平均値
                predictions[mask] = 0.5
        
        return predictions
    
    def train_industry_specific_ai(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        industry_column: str = 'industry',
        min_samples: int = 50
    ):
        """
        業種ごとにAIモデルを訓練
        
        Args:
            X: 特徴量DataFrame
            y: ターゲット
            industry_column: 業種カラム名
            min_samples: 業種別モデルを作る最小サンプル数
        """
        from src.model import CreditScoringModel
        
        print(f"\n=== 業種別AIモデルの学習（最小{min_samples}社）===")
        
        for industry in X[industry_column].unique():
            mask = X[industry_column] == industry
            X_industry = X[mask].drop(columns=[industry_column])
            y_industry = y[mask]
            
            n_samples = len(X_industry)
            
            if n_samples >= min_samples:
                print(f"\n{industry}: {n_samples}社")
                
                # 訓練・検証分割
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_industry, y_industry, 
                    test_size=0.2, 
                    stratify=y_industry if y_industry.sum() > 1 else None,
                    random_state=42
                )
                
                # AIモデル学習
                model = CreditScoringModel(model_type='lightgbm')
                model.build_model(n_estimators=100, max_depth=5, learning_rate=0.05)
                
                try:
                    model.train(X_train, y_train, X_val, y_val)
                    self.ai_models[industry] = model
                    print(f"  ✅ {industry}モデル構築完了")
                except Exception as e:
                    print(f"  ❌ エラー: {e}")
            else:
                print(f"\n{industry}: {n_samples}社（データ不足、全体モデルを使用）")
    
    def create_industry_ensemble(
        self,
        X: pd.DataFrame,
        industry_column: str = 'industry'
    ) -> np.ndarray:
        """
        業種別のハイブリッド予測
        
        Args:
            X: 特徴量DataFrame
            industry_column: 業種カラム名
        
        Returns:
            アンサンブル予測
        """
        predictions = np.zeros(len(X))
        
        for industry in X[industry_column].unique():
            mask = X[industry_column] == industry
            X_industry = X[mask].drop(columns=[industry_column])
            
            # 既存モデルの予測
            legacy_pred = self.predict_by_industry(
                X[mask], 
                industry_column
            )
            
            # AIモデルの予測
            if industry in self.ai_models:
                ai_pred = self.ai_models[industry].predict_proba(X_industry)
            else:
                # 業種別モデルがない場合はデフォルト
                ai_pred = np.full(mask.sum(), 0.5)
            
            # 加重平均（業種ごとに調整可能）
            predictions[mask] = 0.3 * legacy_pred + 0.7 * ai_pred
        
        return predictions
    
    def analyze_industry_differences(self) -> Dict:
        """
        業種間の違いを分析
        
        Returns:
            分析結果の辞書
        """
        comparison = self.compare_industries()
        
        # 業種間で差が大きい変数（重要な業種特性）
        high_variance = comparison.nlargest(10, 'std_across_industries')
        
        # 全業種で共通して重要な変数
        comparison_abs = comparison.copy()
        industry_cols = [col for col in comparison.columns 
                        if col not in ['feature', 'std_across_industries']]
        
        for col in industry_cols:
            comparison_abs[col] = comparison_abs[col].abs()
        
        comparison_abs['mean_abs'] = comparison_abs[industry_cols].mean(axis=1)
        universally_important = comparison_abs.nlargest(10, 'mean_abs')
        
        analysis = {
            'industry_specific': high_variance,
            'universally_important': universally_important,
            'comparison_table': comparison
        }
        
        print("\n=== 業種間の違い分析 ===")
        print("\n【業種特有の重要変数】（業種間で差が大きい）")
        print(high_variance[['feature', 'std_across_industries']].to_string(index=False))
        
        print("\n【全業種共通の重要変数】")
        print(universally_important[['feature', 'mean_abs']].to_string(index=False))
        
        return analysis
    
    def recommend_features_for_industry(self, industry: str, top_n: int = 10) -> List[str]:
        """
        特定業種で重要な特徴量を推奨
        
        Args:
            industry: 業種名
            top_n: 推奨する特徴量数
        
        Returns:
            推奨特徴量のリスト
        """
        importance = self.get_industry_importance(industry)
        return importance.head(top_n)['feature'].tolist()
    
    def save_models(self, base_path: str = "models/industry_specific"):
        """
        業種別モデルを保存
        
        Args:
            base_path: 保存先ディレクトリ
        """
        from pathlib import Path
        
        path = Path(base_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 係数を保存
        joblib.dump(self.industry_coefficients, path / "industry_coefficients.pkl")
        joblib.dump(self.industry_intercepts, path / "industry_intercepts.pkl")
        
        # 業種別AIモデルを保存
        for industry, model in self.ai_models.items():
            safe_name = industry.replace('/', '_').replace(' ', '_')
            model.save_model(path / f"{safe_name}_model.pkl")
        
        print(f"✅ 業種別モデルを保存: {base_path}")


def example_industry_coefficients():
    """
    業種別係数の入力例
    """
    print("="*70)
    print("業種別ハイブリッドモデル - 係数入力例")
    print("="*70)
    
    # 業種別の係数（例）
    industry_coefficients = {
        '製造業': {
            'ROA': 0.15,
            'equity_ratio': 0.08,
            'debt_ratio': -0.12,
            'machinery_ratio': -0.05,      # 製造業は機械設備比率が重要
            'asset_turnover': 0.06,
            'depreciation_rate': -0.08,    # 設備年齢が重要
        },
        '建設業': {
            'ROA': 0.12,
            'equity_ratio': 0.10,           # 建設業は自己資本比率がより重要
            'debt_ratio': -0.15,            # 負債への感度が高い
            'machinery_ratio': -0.08,
            'operating_margin': 0.12,
            'rent_to_revenue': -0.18,       # 建設機械のリース負担
        },
        'サービス業': {
            'ROA': 0.18,                    # サービス業はROAが最重要
            'equity_ratio': 0.06,
            'debt_ratio': -0.10,
            'operating_margin': 0.15,       # 利益率が重要
            'asset_turnover': 0.10,         # 資産効率が重要
            'rent_to_revenue': -0.12,       # 店舗賃料
        },
        '卸売業': {
            'ROA': 0.14,
            'equity_ratio': 0.07,
            'debt_ratio': -0.11,
            'asset_turnover': 0.12,         # 回転率が最重要
            'operating_margin': 0.08,
            'current_ratio': 0.05,
        },
        '小売業': {
            'ROA': 0.13,
            'equity_ratio': 0.09,
            'debt_ratio': -0.13,
            'asset_turnover': 0.11,
            'operating_margin': 0.10,
            'rent_to_revenue': -0.22,       # 店舗賃料が最重要
        }
    }
    
    # 切片
    industry_intercepts = {
        '製造業': -2.5,
        '建設業': -2.8,
        'サービス業': -2.3,
        '卸売業': -2.6,
        '小売業': -2.7,
    }
    
    # モデル初期化
    model = IndustrySpecificHybridModel()
    model.load_industry_coefficients(industry_coefficients, industry_intercepts)
    
    # 業種間の比較
    print("\n" + "="*70)
    model.analyze_industry_differences()
    
    # 各業種で重要な変数
    print("\n" + "="*70)
    print("業種別の重要変数（Top 3）")
    print("="*70)
    for industry in industry_coefficients.keys():
        print(f"\n{industry}:")
        importance = model.get_industry_importance(industry)
        for idx, row in importance.head(3).iterrows():
            direction = "↑" if row['direction'] == 'positive' else "↓"
            print(f"  {direction} {row['feature']:20s}: {row['coefficient']:7.3f}")
    
    return model


if __name__ == "__main__":
    example_industry_coefficients()
