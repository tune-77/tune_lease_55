import numpy as np
import joblib
from sklearn.covariance import MinCovDet, OAS
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
            est = OAS().fit(X_scaled)
        self.precision_ = est.get_precision()
        eigvals = np.linalg.eigvalsh(self.precision_)
        if np.any(eigvals <= 0):
            raise ValueError("precision_ must be positive definite")
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




def _gaussian_kl(mu_p: np.ndarray, cov_p: np.ndarray, mu_q: np.ndarray, cov_q: np.ndarray) -> float:
    p = cov_p.shape[0]
    cov_q_inv = np.linalg.pinv(cov_q)
    delta = mu_q - mu_p
    term1 = float(np.trace(cov_q_inv @ cov_p))
    term2 = float(delta.T @ cov_q_inv @ delta)
    sign_p, logdet_p = np.linalg.slogdet(cov_p)
    sign_q, logdet_q = np.linalg.slogdet(cov_q)
    if sign_p <= 0 or sign_q <= 0:
        raise ValueError("Covariance determinant sign invalid")
    return 0.5 * (term1 + term2 - p + (logdet_q - logdet_p))


def _matrix_sqrt_psd(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def compute_kl_drift(scorer: "MahalanobisScorer", recent_X: np.ndarray) -> float:
    """最近データ分布と学習分布の KL ダイバージェンス D_KL(recent || train) を返す。"""
    if scorer.precision_ is None or scorer.mu_ is None:
        raise ValueError("Scorer is not fitted")
    X_proc = scorer._preprocess(recent_X)
    X_scaled = scorer.scaler.transform(X_proc)
    mu_r = np.mean(X_scaled, axis=0)
    cov_r = np.cov(X_scaled, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]])
    p = cov_r.shape[0]
    cov_r += np.eye(p) * 1e-8

    prec_t = scorer.precision_
    cov_t = np.linalg.pinv(prec_t)
    delta = scorer.mu_ - mu_r
    term1 = float(np.trace(prec_t @ cov_r))
    term2 = float(delta.T @ prec_t @ delta)
    sign_t, logdet_t = np.linalg.slogdet(cov_t)
    sign_r, logdet_r = np.linalg.slogdet(cov_r)
    if sign_t <= 0 or sign_r <= 0:
        raise ValueError("Covariance determinant sign invalid")
    return 0.5 * (term1 + term2 - p + (logdet_t - logdet_r))



def compute_js_drift(scorer: "MahalanobisScorer", recent_X: np.ndarray) -> float:
    """ガウス近似で JS ダイバージェンスを返す。"""
    if scorer.precision_ is None or scorer.mu_ is None:
        raise ValueError("Scorer is not fitted")
    X_proc = scorer._preprocess(recent_X)
    X_scaled = scorer.scaler.transform(X_proc)
    mu_r = np.mean(X_scaled, axis=0)
    cov_r = np.cov(X_scaled, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]])
    p = cov_r.shape[0]
    cov_r += np.eye(p) * 1e-8

    cov_t = np.linalg.pinv(scorer.precision_)
    mu_t = scorer.mu_
    m_mu = 0.5 * (mu_r + mu_t)
    m_cov = 0.5 * (cov_r + cov_t) + np.eye(p) * 1e-8

    kl_r_m = _gaussian_kl(mu_r, cov_r, m_mu, m_cov)
    kl_t_m = _gaussian_kl(mu_t, cov_t, m_mu, m_cov)
    return 0.5 * (kl_r_m + kl_t_m)


def compute_wasserstein2_drift(scorer: "MahalanobisScorer", recent_X: np.ndarray) -> float:
    """ガウス近似Wasserstein-2距離を返す。"""
    if scorer.precision_ is None or scorer.mu_ is None:
        raise ValueError("Scorer is not fitted")
    X_proc = scorer._preprocess(recent_X)
    X_scaled = scorer.scaler.transform(X_proc)
    mu_r = np.mean(X_scaled, axis=0)
    cov_r = np.cov(X_scaled, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]])
    p = cov_r.shape[0]
    cov_r += np.eye(p) * 1e-8

    cov_t = np.linalg.pinv(scorer.precision_)
    mu_t = scorer.mu_

    mean_term = float(np.sum((mu_r - mu_t) ** 2))
    sqrt_cov_t = _matrix_sqrt_psd(cov_t)
    inner = sqrt_cov_t @ cov_r @ sqrt_cov_t
    cov_term = float(np.trace(cov_r + cov_t - 2.0 * _matrix_sqrt_psd(inner)))
    return max(0.0, mean_term + cov_term)


def compute_fisher_rao_proxy(scorer: "MahalanobisScorer", recent_X: np.ndarray) -> float:
    """Fisher-Rao の実装コストを抑えるための proxy (Bhattacharyya distance)。"""
    if scorer.precision_ is None or scorer.mu_ is None:
        raise ValueError("Scorer is not fitted")
    X_proc = scorer._preprocess(recent_X)
    X_scaled = scorer.scaler.transform(X_proc)
    mu_r = np.mean(X_scaled, axis=0)
    cov_r = np.cov(X_scaled, rowvar=False)
    if cov_r.ndim == 0:
        cov_r = np.array([[float(cov_r)]])
    p = cov_r.shape[0]
    cov_r += np.eye(p) * 1e-8

    cov_t = np.linalg.pinv(scorer.precision_)
    delta = (mu_r - scorer.mu_).reshape(-1, 1)
    sigma = 0.5 * (cov_r + cov_t) + np.eye(p) * 1e-8
    sigma_inv = np.linalg.pinv(sigma)

    term1 = float(0.125 * (delta.T @ sigma_inv @ delta).item())
    sign_s, logdet_s = np.linalg.slogdet(sigma)
    sign_r, logdet_r = np.linalg.slogdet(cov_r)
    sign_t, logdet_t = np.linalg.slogdet(cov_t)
    if sign_s <= 0 or sign_r <= 0 or sign_t <= 0:
        raise ValueError("Covariance determinant sign invalid")
    term2 = float(0.5 * (logdet_s - 0.5 * (logdet_r + logdet_t)))
    return term1 + term2


def summarize_distribution_drift(scorer: "MahalanobisScorer", recent_X: np.ndarray, thresholds: dict | None = None) -> dict:
    """複数指標でドリフト判定を返す。"""
    thresholds = thresholds or {
        "kl": 0.35,
        "js": 0.08,
        "wasserstein2": 1.2,
        "fisher_rao_proxy": 0.12,
    }
    kl = compute_kl_drift(scorer, recent_X)
    js = compute_js_drift(scorer, recent_X)
    w2 = compute_wasserstein2_drift(scorer, recent_X)
    fr = compute_fisher_rao_proxy(scorer, recent_X)
    flags = {
        "kl": kl >= thresholds["kl"],
        "js": js >= thresholds["js"],
        "wasserstein2": w2 >= thresholds["wasserstein2"],
        "fisher_rao_proxy": fr >= thresholds["fisher_rao_proxy"],
    }
    triggered = [k for k, v in flags.items() if v]
    return {
        "metrics": {"kl": kl, "js": js, "wasserstein2": w2, "fisher_rao_proxy": fr},
        "thresholds": thresholds,
        "flags": flags,
        "is_drift": len(triggered) >= 2,
        "triggered_metrics": triggered,
    }
