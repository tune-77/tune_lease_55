import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import umap


class UMAPAnomalyScorer:
    """
    Isolation Forest (一クラス異常検知) + UMAP 2D可視化 の複合スコアラー。

    Isolation Forest: 成約案件のみで学習 → 新規案件が「過去の成約案件と似ているか」を非線形に評価
    UMAP: 成約＋失注全件で学習 → 2D散布図用の座標を提供
    """

    LOG_KEYS = ('nenshu', 'profit', 'income', 'machines', 'credit', 'expense')

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names
        self.log_features = [f for f in feature_names if any(k in f for k in self.LOG_KEYS)]
        self.scaler = StandardScaler()
        self.iso_forest = IsolationForest(
            n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1
        )
        self.umap_model = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, low_memory=False
        )
        # キャリブレーション用
        self._score_min: float = -0.5
        self._score_max: float = 0.0
        # UMAP埋め込み（全学習データ）
        self.umap_embeddings_: np.ndarray | None = None
        self.train_labels_: list[str] = []
        self.train_size: int = 0
        self.last_updated: str | None = None

    # ── 前処理 ──────────────────────────────────────────────

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for feat in self.log_features:
            if feat in self.feature_names:
                idx = self.feature_names.index(feat)
                X[:, idx] = np.log1p(np.maximum(0, X[:, idx]))
        return X

    # ── 学習 ────────────────────────────────────────────────

    def fit(
        self,
        df_won,
        df_labeled=None,
        labels: list[str] | None = None,
    ) -> None:
        """
        df_won      : 成約案件 DataFrame（Isolation Forest 学習用）
        df_labeled  : 成約＋失注全件 DataFrame（UMAP 学習用）。None なら df_won を使用
        labels      : df_labeled に対応するラベルリスト（'成約'/'失注'）
        """
        # --- Isolation Forest ---
        X_won = self._preprocess(df_won[self.feature_names].values)
        X_won_scaled = self.scaler.fit_transform(X_won)
        self.iso_forest.fit(X_won_scaled)

        # スコアのキャリブレーション（5〜95パーセンタイルを 0〜100 にマップ）
        raw = self.iso_forest.score_samples(X_won_scaled)
        self._score_min = float(np.percentile(raw, 5))
        self._score_max = float(np.percentile(raw, 95))

        # --- UMAP ---
        if df_labeled is not None and labels is not None:
            X_all = self._preprocess(df_labeled[self.feature_names].values)
            X_all_scaled = self.scaler.transform(X_all)
            self.umap_embeddings_ = self.umap_model.fit_transform(X_all_scaled)
            self.train_labels_ = list(labels)
        else:
            self.umap_embeddings_ = self.umap_model.fit_transform(X_won_scaled)
            self.train_labels_ = ['成約'] * len(df_won)

        self.train_size = len(self.train_labels_)
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── スコアリング ─────────────────────────────────────────

    def score(self, x_raw) -> tuple[float, float, float]:
        """(anomaly_score 0-100, umap_x, umap_y) を返す。"""
        import pandas as pd
        if isinstance(x_raw, pd.DataFrame):
            X = self._preprocess(x_raw[self.feature_names].values)
        else:
            X = self._preprocess(np.array(x_raw))
        X_scaled = self.scaler.transform(X)

        raw = float(self.iso_forest.score_samples(X_scaled)[0])
        span = max(self._score_max - self._score_min, 1e-6)
        score = float(np.clip((raw - self._score_min) / span * 100, 0, 100))

        xy = self.umap_model.transform(X_scaled)[0]
        return round(score, 1), float(xy[0]), float(xy[1])

    def find_similar(self, x_raw, top_k: int = 3) -> list[dict]:
        """UMAP空間で最近傍の成約案件を返す。"""
        _, ux, uy = self.score(x_raw)
        if self.umap_embeddings_ is None:
            return []
        labels = np.array(self.train_labels_)
        emb = self.umap_embeddings_
        dists = np.sqrt((emb[:, 0] - ux) ** 2 + (emb[:, 1] - uy) ** 2)
        # 成約のみ対象
        dists_won = np.where(labels == '成約', dists, np.inf)
        top_idx = np.argsort(dists_won)[:top_k]
        return [
            {"x": float(emb[i, 0]), "y": float(emb[i, 1]), "status": labels[i]}
            for i in top_idx
        ]

    def get_embeddings(self, max_points: int = 600) -> list[dict]:
        """フロントエンド散布図用の埋め込みデータ（ダウンサンプリング済み）を返す。"""
        if self.umap_embeddings_ is None:
            return []
        labels = np.array(self.train_labels_)
        emb = self.umap_embeddings_
        result = []
        for status in ('成約', '失注'):
            idx = np.where(labels == status)[0]
            if len(idx) > max_points // 2:
                rng = np.random.default_rng(42)
                idx = rng.choice(idx, max_points // 2, replace=False)
            for i in idx:
                result.append({
                    "x": round(float(emb[i, 0]), 3),
                    "y": round(float(emb[i, 1]), 3),
                    "s": status,
                })
        return result

    # ── 永続化 ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "UMAPAnomalyScorer":
        return joblib.load(path)
