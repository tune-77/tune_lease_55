"""
Phase 2: 自然勾配最適化 A/B比較

Fisher情報行列を用いた自然勾配降下法(NGD)と
現行の勾配降下法(GD)を同条件で比較し、AUC/収束速度を返す。

希少業種（n<5）で自然勾配はRiemanianメトリクスが平坦化されるため
パラメータ空間の歪みを補正でき、小データでの収束が改善しやすい。
"""

from __future__ import annotations

import sys
import os
import numpy as np
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ── フィーチャー行列構築 ────────────────────────────────────────────────────────

def _build_feature_matrix(cases: list[dict], coeff_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    各 case から coeff_keys に対応する線形特徴量ベクトルを構築する。
    _calculate_z の処理を展開し、係数ごとの寄与量を特徴量として返す。
    戻り値: X (N x D), y (N,)
    """
    from scoring_core import _get_industry_flags
    from data_cases import get_score_weights

    w_b, w_a, _, _ = get_score_weights()

    WIN_STATUSES = {"成約", "検収完了", "検収"}
    VALID = WIN_STATUSES | {"失注"}
    cases = [c for c in cases if c.get("final_status") in VALID]

    rows, labels = [], []
    for c in cases:
        inp = c.get("inputs", {})
        res = c.get("result", {})
        flags = _get_industry_flags(c.get("industry_major", ""))

        feat: dict[str, float] = {}
        feat["intercept"] = 1.0

        for ind_key, active in flags.items():
            feat[ind_key] = 1.0 if active else 0.0

        nenshu = float(inp.get("nenshu") or 0)
        feat["sales_log"] = np.log1p(nenshu) if nenshu > 0 else 0.0

        bc = float(inp.get("bank_credit") or 0)
        feat["bank_credit_log"] = np.log1p(bc) if bc > 0 else 0.0

        lc = float(inp.get("lease_credit") or 0)
        feat["lease_credit_log"] = np.log1p(lc) if lc > 0 else 0.0

        for raw_key in ["op_profit", "ord_profit", "net_income", "machines",
                        "other_assets", "rent", "gross_profit",
                        "depreciation", "dep_expense", "rent_expense"]:
            feat[raw_key] = float(inp.get(raw_key) or 0) / 1000.0

        feat["contracts"] = float(inp.get("contracts") or 0)

        grade = inp.get("grade") or "1-3"
        feat["grade_4_6"] = 1.0 if "4-6" in grade else 0.0
        feat["grade_watch"] = 1.0 if "要注意" in grade else 0.0
        feat["grade_none"] = 1.0 if "無格付" in grade else 0.0

        sd = c.get("sales_dept", "未設定")
        feat["dept_utsunomiya"] = 1.0 if sd == "宇都宮営業部" else 0.0
        feat["dept_oyama"] = 1.0 if sd == "小山営業部" else 0.0
        feat["dept_ashikaga"] = 1.0 if sd == "足利営業部" else 0.0
        feat["dept_saitama"] = 1.0 if sd == "埼玉営業部" else 0.0

        rows.append([feat.get(k, 0.0) for k in coeff_keys])
        labels.append(1 if c["final_status"] in WIN_STATUSES else 0)

    return np.array(rows, dtype=float), np.array(labels, dtype=float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-9
    return -float(np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.5


# ── 勾配降下法 (GD) ──────────────────────────────────────────────────────────

def _run_gd(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    lr: float = 0.05,
    n_steps: int = 200,
    l2: float = 1e-3,
) -> tuple[np.ndarray, list[float], list[float]]:
    w = w_init.copy()
    loss_hist, auc_hist = [], []
    N = len(y)
    for _ in range(n_steps):
        p = _sigmoid(X @ w)
        grad = (X.T @ (p - y)) / N + l2 * w
        w -= lr * grad
        loss_hist.append(_binary_cross_entropy(y, _sigmoid(X @ w)))
        auc_hist.append(_auc(y, X @ w))
    return w, loss_hist, auc_hist


# ── 自然勾配降下法 (NGD) ─────────────────────────────────────────────────────

def _fisher_matrix(X: np.ndarray, p: np.ndarray, l2: float = 1e-3, damping: float = 0.1) -> np.ndarray:
    """
    Fisher情報行列 F = X^T diag(p(1-p)) X / N + (l2 + damping)*I
    damping はTrustRegion的な安定化項で発散を防ぐ。
    """
    weights = np.clip(p * (1 - p), 1e-8, 0.25)
    F = (X * weights[:, None]).T @ X / len(p)
    F += (l2 + damping) * np.eye(F.shape[0])
    return F


_MAX_GRAD_NORM = 5.0  # 自然勾配ステップの最大ノルム


def _run_ngd(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    lr: float = 0.5,
    n_steps: int = 200,
    l2: float = 1e-3,
    damping: float = 0.1,
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    自然勾配降下法: w ← w - lr * F^{-1} g
    - Fisher行列にdampingを加えて安定化
    - 自然勾配ノルムをクリップして発散防止
    """
    w = w_init.copy()
    loss_hist, auc_hist = [], []
    N = len(y)
    prev_loss = float("inf")
    for _ in range(n_steps):
        p = _sigmoid(X @ w)
        grad = (X.T @ (p - y)) / N + l2 * w
        F = _fisher_matrix(X, p, l2, damping)
        try:
            nat_grad = np.linalg.solve(F, grad)
        except np.linalg.LinAlgError:
            nat_grad = grad
        # ノルムクリップ（発散防止）
        ng_norm = np.linalg.norm(nat_grad)
        if ng_norm > _MAX_GRAD_NORM:
            nat_grad = nat_grad * (_MAX_GRAD_NORM / ng_norm)
        w_new = w - lr * nat_grad
        new_loss = _binary_cross_entropy(y, _sigmoid(X @ w_new))
        # 損失が増加したら学習率を半減してリトライ（1回のみ）
        if new_loss > prev_loss * 1.05:
            w_new = w - (lr * 0.5) * nat_grad
            new_loss = _binary_cross_entropy(y, _sigmoid(X @ w_new))
        w = w_new
        prev_loss = new_loss
        loss_hist.append(new_loss)
        auc_hist.append(_auc(y, X @ w))
    return w, loss_hist, auc_hist


# ── A/B 比較インターフェース ──────────────────────────────────────────────────

def run_ab_comparison(
    coeff_key: str = "全体_既存先",
    n_steps: int = 200,
    gd_lr: float = 0.05,
    ngd_lr: float = 0.5,
    l2: float = 1e-3,
    by_industry: bool = False,
) -> dict[str, Any]:
    """
    GD と NGD を同条件で比較し、収束履歴・最終AUC・係数差を返す。

    戻り値:
        {
            "n_cases": int,
            "n_features": int,
            "coeff_keys": list[str],
            "gd": {"final_auc": float, "final_loss": float, "auc_hist": list, "loss_hist": list, "w": ndarray},
            "ngd": {...同上...},
            "winner": "gd" | "ngd" | "tie",
            "auc_delta": float,  # ngd - gd
            "by_industry": dict | None,
        }
    """
    from data_cases import load_all_cases, get_effective_coeffs

    cases = load_all_cases()
    coeffs = get_effective_coeffs(coeff_key)
    coeff_keys = [k for k in coeffs.keys() if not k.startswith("_")]

    X, y = _build_feature_matrix(cases, coeff_keys)
    if len(X) < 5 or y.sum() < 2 or (1 - y).sum() < 2:
        return {"error": "有効データ不足（成約/失注が各2件以上必要）"}

    w_init = np.array([coeffs.get(k, 0.0) for k in coeff_keys])

    w_gd, gd_loss, gd_auc = _run_gd(X, y, w_init, lr=gd_lr, n_steps=n_steps, l2=l2)
    w_ngd, ngd_loss, ngd_auc = _run_ngd(X, y, w_init, lr=ngd_lr, n_steps=n_steps, l2=l2, damping=0.1)

    auc_delta = ngd_auc[-1] - gd_auc[-1]
    winner = "ngd" if auc_delta > 0.005 else ("gd" if auc_delta < -0.005 else "tie")

    result: dict[str, Any] = {
        "n_cases": len(X),
        "n_features": len(coeff_keys),
        "coeff_keys": coeff_keys,
        "gd": {
            "final_auc": gd_auc[-1],
            "final_loss": gd_loss[-1],
            "auc_hist": gd_auc,
            "loss_hist": gd_loss,
            "w": w_gd,
        },
        "ngd": {
            "final_auc": ngd_auc[-1],
            "final_loss": ngd_loss[-1],
            "auc_hist": ngd_auc,
            "loss_hist": ngd_loss,
            "w": w_ngd,
        },
        "winner": winner,
        "auc_delta": auc_delta,
        "coeff_keys": coeff_keys,
        "by_industry": None,
    }

    if by_industry:
        from data_cases import load_all_cases as _lac
        all_cases = _lac()
        industries = sorted({c.get("industry_major", "不明") for c in all_cases})
        ind_results = {}
        for ind in industries:
            ind_cases = [c for c in all_cases if c.get("industry_major", "不明") == ind]
            Xi, yi = _build_feature_matrix(ind_cases, coeff_keys)
            if len(Xi) < 4 or yi.sum() < 1 or (1 - yi).sum() < 1:
                ind_results[ind] = {"skip": True, "n": len(Xi)}
                continue
            _, _, gd_a = _run_gd(Xi, yi, w_init, lr=gd_lr, n_steps=n_steps, l2=l2)
            _, _, ng_a = _run_ngd(Xi, yi, w_init, lr=ngd_lr, n_steps=n_steps, l2=l2, damping=0.1)
            ind_results[ind] = {
                "n": len(Xi),
                "gd_auc": gd_a[-1],
                "ngd_auc": ng_a[-1],
                "delta": ng_a[-1] - gd_a[-1],
            }
        result["by_industry"] = ind_results

    return result
