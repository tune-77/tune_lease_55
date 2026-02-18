"""
実際に利用可能なデータ項目に特化した特徴量エンジニアリング

利用可能データ:
- 売上高
- 総資産
- 純資産（自己資本）
- 営業利益
- 当期純利益
- 機械設備（有形固定資産）
- その他固定資産
- 減価償却費
- 賃貸料（リース料）
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

class CustomFinancialFeatures:
    """実データに合わせた特徴量計算"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        利用可能なデータから全ての財務指標を計算
        
        Args:
            df: 以下のカラムを含むDataFrame
                - revenue (売上高)
                - total_assets (総資産)
                - equity (純資産/自己資本)
                - operating_profit (営業利益)
                - net_income (当期純利益)
                - machinery_equipment (機械設備)
                - other_fixed_assets (その他固定資産)
        
        Returns:
            財務指標を追加したDataFrame
        """
        result = df.copy()
        
        print("\n=== 財務指標の計算 ===")
        
        # ===== 1. 収益性指標 =====
        print("\n【収益性指標】")
        
        # ROA（総資産利益率）- 最重要指標
        result['ROA'] = (df['net_income'] / df['total_assets']) * 100
        print("✓ ROA（総資産利益率）")
        
        # ROE（自己資本利益率）
        result['ROE'] = (df['net_income'] / df['equity']) * 100
        print("✓ ROE（自己資本利益率）")
        
        # 売上高営業利益率
        result['operating_margin'] = (df['operating_profit'] / df['revenue']) * 100
        print("✓ 売上高営業利益率")
        
        # 売上高純利益率
        result['net_margin'] = (df['net_income'] / df['revenue']) * 100
        print("✓ 売上高純利益率")
        
        # 総資産営業利益率
        result['operating_ROA'] = (df['operating_profit'] / df['total_assets']) * 100
        print("✓ 総資産営業利益率")
        
        # ===== 2. 安全性指標 =====
        print("\n【安全性指標】")
        
        # 自己資本比率 - 最重要指標
        result['equity_ratio'] = (df['equity'] / df['total_assets']) * 100
        print("✓ 自己資本比率")
        
        # 負債比率（総資産から純資産を引く）
        total_debt = df['total_assets'] - df['equity']
        result['debt_ratio'] = (total_debt / df['total_assets']) * 100
        print("✓ 負債比率")
        
        # 負債自己資本比率（D/Eレシオ）
        result['debt_equity_ratio'] = (total_debt / df['equity']) * 100
        print("✓ 負債自己資本比率")
        
        # ===== 3. 効率性指標 =====
        print("\n【効率性指標】")
        
        # 総資産回転率
        result['asset_turnover'] = df['revenue'] / df['total_assets']
        print("✓ 総資産回転率")
        
        # 固定資産回転率
        fixed_assets = df['machinery_equipment'] + df['other_fixed_assets']
        result['fixed_asset_turnover'] = df['revenue'] / (fixed_assets + 1)  # ゼロ除算回避
        print("✓ 固定資産回転率")
        
        # 純資産回転率
        result['equity_turnover'] = df['revenue'] / df['equity']
        print("✓ 純資産回転率")
        
        # ===== 4. 固定資産関連指標（リース審査で重要）=====
        print("\n【固定資産関連指標】")
        
        # 固定資産比率
        result['fixed_asset_ratio'] = (fixed_assets / df['total_assets']) * 100
        print("✓ 固定資産比率")
        
        # 機械設備比率（リース審査で特に重要）
        result['machinery_ratio'] = (df['machinery_equipment'] / df['total_assets']) * 100
        print("✓ 機械設備比率")
        
        # 固定資産対純資産比率（固定長期適合率の代替）
        result['fixed_to_equity'] = (fixed_assets / df['equity']) * 100
        print("✓ 固定資産対純資産比率")
        
        # 機械設備の自己資本カバー率
        result['machinery_equity_coverage'] = (df['equity'] / df['machinery_equipment']) * 100
        print("✓ 機械設備の自己資本カバー率")
        
        # ===== 5. 減価償却関連指標（設備投資の積極性）=====
        if 'depreciation' in df.columns:
            print("\n【減価償却関連指標】")
            
            # 減価償却費率（売上高比）
            result['depreciation_to_revenue'] = (df['depreciation'] / df['revenue']) * 100
            print("✓ 減価償却費率（対売上高）")
            
            # 設備の償却進行度（減価償却費÷機械設備）
            # 高い = 古い設備、低い = 新しい設備
            result['depreciation_rate'] = (df['depreciation'] / (df['machinery_equipment'] + 1)) * 100
            print("✓ 設備償却進行度")
            
            # 設備投資余力（EBITDA的な指標）
            # 営業利益 + 減価償却費 = キャッシュ創出力
            result['EBITDA_proxy'] = df['operating_profit'] + df['depreciation']
            result['EBITDA_margin'] = (result['EBITDA_proxy'] / df['revenue']) * 100
            print("✓ EBITDA代替指標（営業CF創出力）")
            
            # 設備年齢推定（機械設備÷減価償却費 = 平均残存年数の逆数的指標）
            result['equipment_age_proxy'] = df['machinery_equipment'] / (df['depreciation'] + 1)
            print("✓ 設備年齢推定値")
            
            # 減価償却の純資産カバー率
            result['depreciation_equity_ratio'] = (df['depreciation'] / df['equity']) * 100
            print("✓ 減価償却費の純資産比率")
        
        # ===== 6. リース負担関連指標（リース審査で最重要）=====
        if 'rent_expense' in df.columns:
            print("\n【リース負担関連指標】")
            
            # リース料負担率（売上高比）- 重要指標
            result['rent_to_revenue'] = (df['rent_expense'] / df['revenue']) * 100
            print("✓ リース料負担率（対売上高）")
            
            # リース料の営業利益カバー率
            result['operating_profit_to_rent'] = (df['operating_profit'] / (df['rent_expense'] + 1)) * 100
            print("✓ 営業利益のリース料カバー率")
            
            # リース料の純利益カバー率
            result['net_income_to_rent'] = (df['net_income'] / (df['rent_expense'] + 1)) * 100
            print("✓ 純利益のリース料カバー率")
            
            # 既存リース負担度（純資産比）
            result['rent_to_equity'] = (df['rent_expense'] / df['equity']) * 100
            print("✓ リース料の純資産負担率")
            
            # リース依存度（リース料÷減価償却費）
            # 高い = 自己所有より賃借依存、低い = 自己所有型
            if 'depreciation' in df.columns:
                result['lease_dependency'] = (df['rent_expense'] / (df['depreciation'] + 1)) * 100
                print("✓ リース依存度（vs自己所有）")
            
            # 総固定費負担（リース料+減価償却費）
            if 'depreciation' in df.columns:
                result['total_fixed_cost'] = df['rent_expense'] + df['depreciation']
                result['total_fixed_cost_ratio'] = (result['total_fixed_cost'] / df['revenue']) * 100
                print("✓ 総固定費負担率")
                
                # キャッシュアウト型固定費（リース料）vs 非キャッシュ（減価償却）
                result['cash_fixed_cost_ratio'] = (df['rent_expense'] / result['total_fixed_cost']) * 100
                print("✓ キャッシュアウト固定費比率")
        
        # ===== 7. 規模指標 =====
        print("\n【規模指標】")
        
        # 対数変換（スケールの違いを吸収）
        result['log_revenue'] = np.log1p(df['revenue'])
        result['log_assets'] = np.log1p(df['total_assets'])
        print("✓ 売上高・総資産の対数変換")
        
        # ===== 6. 派生指標 =====
        print("\n【派生指標】")
        
        # 営業利益/純利益比率（営業外損益の影響度）
        result['operating_to_net_ratio'] = df['operating_profit'] / (df['net_income'] + 0.001)  # ゼロ除算回避
        print("✓ 営業利益/純利益比率")
        
        # 1円の資産で何円稼ぐか
        result['profit_per_asset'] = df['net_income'] / df['total_assets']
        print("✓ 資産あたり利益")
        
        # 1円の売上で何円残るか
        result['profit_per_revenue'] = df['net_income'] / df['revenue']
        print("✓ 売上あたり利益")
        
        print(f"\n✅ 合計 {len([c for c in result.columns if c not in df.columns])} 個の財務指標を計算")
        
        return result
    
    def calculate_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        リスクフラグ（危険信号）を計算
        
        Args:
            df: 財務指標を含むDataFrame
        
        Returns:
            リスクフラグを追加したDataFrame
        """
        result = df.copy()
        
        print("\n=== リスクフラグの設定 ===")
        
        # 赤字フラグ
        result['is_loss'] = (df['net_income'] < 0).astype(int)
        print(f"✓ 赤字企業: {result['is_loss'].sum()}社")
        
        # 営業赤字フラグ
        result['is_operating_loss'] = (df['operating_profit'] < 0).astype(int)
        print(f"✓ 営業赤字企業: {result['is_operating_loss'].sum()}社")
        
        # 債務超過フラグ
        result['is_negative_equity'] = (df['equity'] < 0).astype(int)
        print(f"✓ 債務超過企業: {result['is_negative_equity'].sum()}社")
        
        # 自己資本比率が低い
        if 'equity_ratio' in result.columns:
            result['low_equity_ratio'] = (result['equity_ratio'] < 20).astype(int)
            print(f"✓ 自己資本比率20%未満: {result['low_equity_ratio'].sum()}社")
        
        # ROAが低い
        if 'ROA' in result.columns:
            result['low_ROA'] = (result['ROA'] < 2).astype(int)
            print(f"✓ ROA 2%未満: {result['low_ROA'].sum()}社")
        
        # 固定資産が過大
        if 'fixed_to_equity' in result.columns:
            result['high_fixed_assets'] = (result['fixed_to_equity'] > 100).astype(int)
            print(f"✓ 固定資産>純資産: {result['high_fixed_assets'].sum()}社")
        
        # リース負担が重い（売上高の5%超）
        if 'rent_to_revenue' in result.columns:
            result['high_rent_burden'] = (result['rent_to_revenue'] > 5).astype(int)
            print(f"✓ リース料負担大（売上5%超）: {result['high_rent_burden'].sum()}社")
        
        # リース料が利益を圧迫（営業利益<リース料）
        if 'operating_profit_to_rent' in result.columns:
            result['rent_exceeds_profit'] = (result['operating_profit_to_rent'] < 100).astype(int)
            print(f"✓ リース料>営業利益: {result['rent_exceeds_profit'].sum()}社")
        
        # 設備が古い（償却進行度が高い）
        if 'depreciation_rate' in result.columns:
            result['old_equipment'] = (result['depreciation_rate'] > 15).astype(int)
            print(f"✓ 設備老朽化: {result['old_equipment'].sum()}社")
        
        return result
    
    def get_important_features_for_lease(self) -> List[str]:
        """
        リース審査で特に重要な特徴量リスト
        
        Returns:
            重要特徴量のリスト
        """
        return [
            # 収益性（最重要）
            'ROA',
            'ROE',
            'operating_margin',
            'net_margin',
            
            # 安全性（最重要）
            'equity_ratio',
            'debt_ratio',
            'debt_equity_ratio',
            
            # 固定資産関連（リース特有）
            'machinery_ratio',
            'fixed_asset_ratio',
            'fixed_to_equity',
            'machinery_equity_coverage',
            
            # リース負担関連（リース審査で最重要）
            'rent_to_revenue',
            'operating_profit_to_rent',
            'rent_to_equity',
            'lease_dependency',
            'total_fixed_cost_ratio',
            
            # 減価償却・設備関連
            'depreciation_to_revenue',
            'EBITDA_margin',
            'depreciation_rate',
            
            # 効率性
            'asset_turnover',
            'fixed_asset_turnover',
            
            # 規模
            'log_revenue',
            'log_assets',
            
            # リスクフラグ
            'is_loss',
            'is_operating_loss',
            'low_equity_ratio',
            'low_ROA',
            'high_rent_burden',
            'rent_exceeds_profit',
        ]
    
    def create_sample_data_format(self) -> pd.DataFrame:
        """
        入力データのサンプルフォーマットを生成
        
        Returns:
            サンプルDataFrame
        """
        sample = pd.DataFrame({
            'company_id': [1, 2, 3],
            'revenue': [500_000_000, 300_000_000, 150_000_000],  # 売上高（円）
            'total_assets': [800_000_000, 400_000_000, 200_000_000],  # 総資産
            'equity': [300_000_000, 100_000_000, 50_000_000],  # 純資産
            'operating_profit': [50_000_000, -5_000_000, 10_000_000],  # 営業利益
            'net_income': [30_000_000, -10_000_000, 5_000_000],  # 当期純利益
            'machinery_equipment': [200_000_000, 150_000_000, 80_000_000],  # 機械設備
            'other_fixed_assets': [100_000_000, 50_000_000, 30_000_000],  # その他固定資産
            'depreciation': [20_000_000, 15_000_000, 10_000_000],  # 減価償却費
            'rent_expense': [12_000_000, 18_000_000, 8_000_000],  # 賃貸料（リース料）
        })
        
        return sample
    
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        入力データの妥当性をチェック
        
        Args:
            df: チェックするDataFrame
        
        Returns:
            (妥当性, エラーメッセージのリスト)
        """
        errors = []
        
        required_cols = [
            'revenue', 'total_assets', 'equity', 
            'operating_profit', 'net_income',
            'machinery_equipment', 'other_fixed_assets'
        ]
        
        # 必須カラムのチェック
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"必須カラムが不足: {missing_cols}")
        
        # 数値型チェック
        for col in required_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"{col} が数値型ではありません")
        
        # 論理チェック
        if 'total_assets' in df.columns and 'equity' in df.columns:
            if (df['total_assets'] < df['equity']).any():
                errors.append("総資産が純資産より小さい企業があります")
        
        if 'revenue' in df.columns:
            if (df['revenue'] <= 0).any():
                errors.append("売上高が0以下の企業があります")
        
        return len(errors) == 0, errors


if __name__ == "__main__":
    # テスト
    print("="*70)
    print("カスタム特徴量エンジニアリング - テスト")
    print("="*70)
    
    engine = CustomFinancialFeatures()
    
    # サンプルデータ
    print("\n【サンプルデータ形式】")
    sample = engine.create_sample_data_format()
    print(sample.to_string(index=False))
    
    # 入力チェック
    print("\n【入力データ検証】")
    is_valid, errors = engine.validate_input_data(sample)
    if is_valid:
        print("✅ データ形式は正常です")
    else:
        print("❌ エラー:")
        for error in errors:
            print(f"  - {error}")
    
    # 特徴量計算
    result = engine.calculate_all_features(sample)
    result = engine.calculate_risk_flags(result)
    
    # 重要特徴量の表示
    print("\n【計算結果（重要指標）】")
    important_features = engine.get_important_features_for_lease()
    available_features = [f for f in important_features if f in result.columns]
    
    for feature in available_features[:10]:
        print(f"\n{feature}:")
        print(result[feature].to_string(index=False))
