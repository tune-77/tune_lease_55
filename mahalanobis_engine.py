import numpy as np
import joblib
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class MahalanobisScorer:
    def __init__(self, feature_names: list[str], score_scale: float = 5.0):
        self.feature_names = feature_names
        self.log_features = [f for f in feature_names if any(k in f for k in ('nenshu', 'profit', 'assets'))]
        self.scaler = StandardScaler()
        self.precision_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.score_scale = score_scale
        self.train_size = 0
        self.last_updated: str | None = None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        for feat in self.log_features:
            if feat in self.feature_names:
                idx = self.feature_names.index(feat)
                X[:, idx] = np.log1p(np.maximum(0, X[:, idx]))
        return X

    def _unlog(self, feat: str, log_val: float) -> float:
        if feat in self.log_features:
            return np.expm1(log_val)
        return log_val

    def fit(self, df_raw) -> None:
        X = self._preprocess(df_raw[self.feature_names].values)
        self.train_size = len(X)
        X_scaled = self.scaler.fit_transform(X)
        self.mu_ = np.mean(X_scaled, axis=0)

        if self.train_size >= 30:
            est = MinCovDet(support_fraction=0.9, random_state=42).fit(X_scaled)
        else:
            est = EmpiricalCovariance().fit(X_scaled)
        self.precision_ = est.get_precision()
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M")

    def get_analysis(self, x_raw: list | np.ndarray) -> tuple:
        x_proc = self._preprocess(x_raw)
        x_scaled = self.scaler.transform(x_proc)
        delta = (x_scaled - self.mu_).flatten()

        g_delta = self.precision_ @ delta
        d_sq = float(delta @ g_delta)
        d = np.sqrt(max(0.0, d_sq))
        score = float(np.clip(100 * np.exp(-d / self.score_scale), 0, 100))

        grad = 2.0 * g_delta
        if d_sq > 1e-12:
            contribs = (delta * g_delta) / d_sq
        else:
            contribs = np.zeros(len(delta))

        return score, d, grad, contribs

    def advise_improvement(self, x_raw: list | np.ndarray, top_k: int = 3) -> list[dict]:
        x_proc = self._preprocess(x_raw)
        x_scaled = self.scaler.transform(x_proc)
        delta = (x_scaled - self.mu_).flatten()

        g_delta = self.precision_ @ delta
        grad = 2.0 * g_delta
        prec_diag = np.diag(self.precision_)

        # 効率指標=単位リーマン長あたりD²減少量
        efficiency = grad ** 2 / np.maximum(prec_diag, 1e-12)
        top_idx = np.argsort(efficiency)[::-1][:top_k]

        results = []
        for idx in top_idx:
            feat = self.feature_names[idx]
            # 最急降下方向ステップ（スケール空間で0.5単位）
            step_scaled = -np.sign(grad[idx]) * 0.5
            # 元単位に逆変換
            current_scaled = float(x_scaled[0, idx])
            recommended_scaled = current_scaled + step_scaled

            current_orig = self._unlog(feat, float(x_proc[0, idx]))
            recommended_log = recommended_scaled * self.scaler.scale_[idx] + self.scaler.mean_[idx]
            recommended_orig = self._unlog(feat, recommended_log)

            delta_orig = recommended_orig - current_orig
            direction = "▲ 増やす" if delta_orig > 0 else "▼ 減らす"
            results.append({
                "feat": feat,
                "direction": direction,
                "current": current_orig,
                "recommended": recommended_orig,
                "delta": delta_orig,
            })
        return results

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "MahalanobisScorer":
        return joblib.load(path)
