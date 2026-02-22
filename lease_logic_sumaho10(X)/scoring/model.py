"""
審査モデルの構築と学習
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path

class CreditScoringModel:
    """信用審査モデルのクラス"""
    
    def __init__(self, model_type: str = 'lightgbm'):
        """
        Args:
            model_type: 'logistic', 'random_forest', 'lightgbm', 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.best_threshold = 0.5
        
    def build_model(self, **kwargs) -> Any:
        """モデルを構築"""
        
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                **kwargs
            )
        
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 20),
                min_samples_leaf=kwargs.get('min_samples_leaf', 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.05),
                num_leaves=kwargs.get('num_leaves', 31),
                min_child_samples=kwargs.get('min_child_samples', 20),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=kwargs.get('subsample', 0.8),
                colsample_bytree=kwargs.get('colsample_bytree', 0.8),
                scale_pos_weight=kwargs.get('scale_pos_weight', 1),
                random_state=42,
                n_jobs=-1
            )
        
        else:
            raise ValueError(f"未対応のモデルタイプ: {self.model_type}")
        
        return self.model
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """
        モデルを学習
        
        Args:
            X_train: 訓練データ
            y_train: 訓練ラベル
            X_val: 検証データ（オプション）
            y_val: 検証ラベル（オプション）
        """
        if self.model is None:
            self.build_model()
        
        # LightGBMとXGBoostは早期終了をサポート
        if self.model_type in ['lightgbm', 'xgboost'] and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc'
            )
        else:
            self.model.fit(X_train, y_train)
        
        # 特徴量重要度を保存
        self._extract_feature_importance(X_train.columns)
        
        print(f"✅ {self.model_type} モデルの学習完了")
    
    def _extract_feature_importance(self, feature_names):
        """特徴量重要度を抽出"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(self.model.coef_[0])
            }).sort_values('importance', ascending=False)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """デフォルト確率を予測"""
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        クラスを予測
        
        Args:
            X: 特徴量
            threshold: 分類閾値（デフォルトは0.5）
        """
        if threshold is None:
            threshold = self.best_threshold
        
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def optimize_threshold(
        self, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        metric: str = 'f1'
    ) -> float:
        """
        最適な分類閾値を探索
        
        Args:
            X_val: 検証データ
            y_val: 検証ラベル
            metric: 最適化する指標 ('f1', 'precision', 'recall')
        
        Returns:
            最適な閾値
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        y_proba = self.predict_proba(X_val)
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, y_pred)
            elif metric == 'precision':
                score = precision_score(y_val, y_pred)
            elif metric == 'recall':
                score = recall_score(y_val, y_pred)
            else:
                score = f1_score(y_val, y_pred)
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        self.best_threshold = thresholds[best_idx]
        
        print(f"最適閾値: {self.best_threshold:.2f} ({metric}={scores[best_idx]:.3f})")
        
        return self.best_threshold
    
    def evaluate(
        self, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        モデルを評価
        
        Args:
            X_test: テストデータ
            y_test: テストラベル
        
        Returns:
            評価指標の辞書
        """
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        # AUC-ROC
        auc_score = roc_auc_score(y_test, y_proba)
        
        # 混同行列
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 各種指標
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'auc_roc': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        # レポート表示
        print("\n" + "="*60)
        print(f"モデル評価結果 ({self.model_type})")
        print("="*60)
        print(f"AUC-ROC:    {auc_score:.4f}")
        print(f"Accuracy:   {accuracy:.4f}")
        print(f"Precision:  {precision:.4f}")
        print(f"Recall:     {recall:.4f}")
        print(f"F1-Score:   {f1:.4f}")
        print("\n混同行列:")
        print(f"  TN: {tn:5d}  |  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  |  TP: {tp:5d}")
        print("="*60 + "\n")
        
        return metrics
    
    def cross_validate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        cv: int = 5
    ) -> Dict[str, float]:
        """
        交差検証を実行
        
        Args:
            X: 特徴量
            y: ターゲット
            cv: 分割数
        
        Returns:
            評価指標の辞書
        """
        if self.model is None:
            self.build_model()
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=skf, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"\n交差検証 (CV={cv}):")
        print(f"  AUC-ROC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'mean_auc': scores.mean(),
            'std_auc': scores.std(),
            'scores': scores
        }
    
    def save_model(self, filepath: str):
        """モデルを保存"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"✅ モデルを保存: {filepath}")
    
    def load_model(self, filepath: str):
        """モデルを読み込み"""
        self.model = joblib.load(filepath)
        print(f"✅ モデルを読み込み: {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        特徴量重要度を取得
        
        Args:
            top_n: 上位N件を取得
        
        Returns:
            特徴量重要度のDataFrame
        """
        if self.feature_importance is None:
            return None
        
        return self.feature_importance.head(top_n)


if __name__ == "__main__":
    # テスト
    from data_loader import DataLoader
    from feature_engineering import FinancialFeatureEngine
    
    print("=== モデル学習のテスト ===\n")
    
    # ダミーデータを生成
    loader = DataLoader()
    df = loader.create_synthetic_company_data(1000)
    
    # 特徴量を作成
    engine = FinancialFeatureEngine()
    df = engine.calculate_financial_ratios(df)
    
    # ターゲット変数を生成（ダミー）
    # ROAが低いほど倒産リスクが高いと仮定
    y = (df['ROA'] < df['ROA'].quantile(0.2)).astype(int)
    
    # 特徴量を選択
    X = df.select_dtypes(include=[np.number]).drop(columns=['company_id'], errors='ignore')
    X = engine.handle_missing_values(X)
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"訓練データ: {X_train.shape}, テストデータ: {X_test.shape}\n")
    
    # モデルを学習
    for model_type in ['logistic', 'lightgbm']:
        print(f"\n--- {model_type.upper()} ---")
        model = CreditScoringModel(model_type=model_type)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        # 特徴量重要度
        importance = model.get_feature_importance(top_n=5)
        if importance is not None:
            print("\n特徴量重要度 (Top 5):")
            print(importance.to_string(index=False))
