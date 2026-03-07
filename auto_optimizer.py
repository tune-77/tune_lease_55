"""
自動係数最適化モジュール。
成約/失注登録データが 初回50件、以降20件ごとに係数最適化を自動トリガーする。
"""
import json
import os
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_META_FILE   = os.path.join(_SCRIPT_DIR, "data", "training_meta.json")

MIN_START        = 50   # 初回学習に必要な最低件数
RETRAIN_INTERVAL = 20   # 以降の更新間隔（件数）

_DEFAULT_META: dict = {
    "last_trained_count": 0,
    "last_trained_at":    None,
    "last_auc":           None,
    "total_runs":         0,
}


# ─────────────────────────────────────────────────────────────────────────────
# メタ情報の読み書き
# ─────────────────────────────────────────────────────────────────────────────

def load_training_meta() -> dict:
    try:
        with open(_META_FILE, "r", encoding="utf-8") as f:
            return {**_DEFAULT_META, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_META)


def save_training_meta(meta: dict) -> None:
    os.makedirs(os.path.dirname(_META_FILE), exist_ok=True)
    with open(_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# 登録件数取得
# ─────────────────────────────────────────────────────────────────────────────

def get_registered_count() -> int:
    """成約 + 失注の登録済み件数を返す。"""
    from data_cases import load_all_cases
    cases = load_all_cases()
    return sum(1 for c in cases if c.get("final_status") in ["成約", "失注"])


# ─────────────────────────────────────────────────────────────────────────────
# ステータス取得（UI 表示用）
# ─────────────────────────────────────────────────────────────────────────────

def get_training_status() -> dict:
    """
    現在のトレーニング状態を返す。

    Returns:
        {
            "count":              int,          # 現在の登録件数
            "last_trained_count": int,          # 前回学習時の件数
            "last_trained_at":    str | None,   # 前回学習日時
            "last_auc":           float | None, # 前回のAUC
            "total_runs":         int,          # 累計学習回数
            "should_retrain":     bool,         # 今すぐ学習すべきか
            "next_trigger":       int,          # 次回トリガーまであと何件
            "phase":              str,          # "waiting" | "ready" | "active"
        }
    """
    meta  = load_training_meta()
    count = get_registered_count()
    last  = meta["last_trained_count"]

    if count < MIN_START:
        phase        = "waiting"
        next_trigger = MIN_START - count
        should       = False
    elif last == 0:
        # 50件到達済みで未学習
        phase        = "ready"
        next_trigger = 0
        should       = True
    else:
        gap          = count - last
        next_trigger = max(0, RETRAIN_INTERVAL - gap)
        should       = gap >= RETRAIN_INTERVAL
        phase        = "ready" if should else "active"

    return {
        "count":              count,
        "last_trained_count": last,
        "last_trained_at":    meta.get("last_trained_at"),
        "last_auc":           meta.get("last_auc"),
        "total_runs":         meta.get("total_runs", 0),
        "should_retrain":     should,
        "next_trigger":       next_trigger,
        "phase":              phase,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 自動最適化実行
# ─────────────────────────────────────────────────────────────────────────────

def run_auto_optimization(force: bool = False) -> dict | None:
    """
    条件を満たしていれば optimize_score_weights_from_regression() を実行し、
    推奨重みを coeff_overrides.json に保存する。

    Args:
        force: True の場合は件数条件を無視して強制実行。

    Returns:
        最適化結果 dict、またはスキップ時は None。
    """
    status = get_training_status()
    if not force and not status["should_retrain"]:
        return None

    from analysis_regression import optimize_score_weights_from_regression
    from data_cases import load_auto_coeffs, save_auto_coeffs

    result = optimize_score_weights_from_regression()
    if result is None:
        return None

    # 推奨重みを coeff_auto.json（自動専用）に保存
    # → 手動設定の coeff_overrides.json は上書きしない
    auto = load_auto_coeffs()
    auto["_auto_weight_borrower"] = result["recommended_borrower_pct"]
    auto["_auto_weight_asset"]    = result["recommended_asset_pct"]
    if "recommended_quant_pct" in result:
        auto["_auto_weight_quant"] = result["recommended_quant_pct"]
        auto["_auto_weight_qual"]  = result["recommended_qual_pct"]
    save_auto_coeffs(auto)

    # メタ情報を更新
    meta = load_training_meta()
    meta["last_trained_count"] = status["count"]
    meta["last_trained_at"]    = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta["last_auc"]           = result.get("auc_borrower_asset")
    meta["total_runs"]         = meta.get("total_runs", 0) + 1
    save_training_meta(meta)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# サイドバー用ステータスウィジェット（Streamlit）
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar_training_status() -> None:
    """サイドバーに学習ステータスをコンパクト表示する。"""
    import streamlit as st

    try:
        s = get_training_status()
    except Exception:
        return

    with st.sidebar.expander("🧠 係数自動学習ステータス", expanded=False):
        count = s["count"]
        phase = s["phase"]

        if phase == "waiting":
            remaining = s["next_trigger"]
            st.progress(count / MIN_START, text=f"登録件数: {count} / {MIN_START}件")
            st.caption(f"初回学習まであと **{remaining}件**")

        elif phase == "ready":
            st.success(f"✅ 学習可能（{count}件）")
            st.caption("「係数分析・更新」モードから実行 or 次回結果登録時に自動実行")

        else:  # active
            gap       = count - s["last_trained_count"]
            remaining = s["next_trigger"]
            st.progress(gap / RETRAIN_INTERVAL, text=f"+{gap} / {RETRAIN_INTERVAL}件")
            st.caption(f"次回更新まであと **{remaining}件**")

        if s["last_trained_at"]:
            auc_str = f"　AUC: {s['last_auc']:.3f}" if s["last_auc"] else ""
            st.caption(f"前回学習: {s['last_trained_at']}{auc_str}　（累計{s['total_runs']}回）")
