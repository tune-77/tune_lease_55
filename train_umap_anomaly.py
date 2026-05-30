import sys
import os
import json
import numpy as np
import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from data_cases import load_all_cases
from train_mahalanobis import FEATURES, _extract_val
from umap_anomaly_engine import UMAPAnomalyScorer

MODEL_PATH = os.path.join(_DIR, "data", "umap_anomaly_model.joblib")
EMBED_PATH = os.path.join(_DIR, "data", "umap_embeddings.json")


def _to_df(cases: list[dict]) -> pd.DataFrame:
    rows = [{f: _extract_val(c, f) for f in FEATURES} for c in cases]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=FEATURES)


def train() -> None:
    print("データ読込中...")
    all_cases = load_all_cases()
    won   = [c for c in all_cases if c.get("final_status") == "成約"]
    lost  = [c for c in all_cases if c.get("final_status") == "失注"]
    print(f"成約: {len(won)}件  失注: {len(lost)}件")

    df_won     = _to_df(won)
    df_labeled = pd.concat([_to_df(won), _to_df(lost)], ignore_index=True)
    labels     = ["成約"] * len(won) + ["失注"] * len(lost)

    print(f"学習開始（Isolation Forest + UMAP 2D）…")
    scorer = UMAPAnomalyScorer(FEATURES)
    scorer.fit(df_won, df_labeled, labels)

    os.makedirs("data", exist_ok=True)
    scorer.save(MODEL_PATH)
    print(f"モデル保存: {MODEL_PATH}")

    # 散布図用埋め込み JSON を保存（フロントが /api/umap/embeddings で取得）
    embeddings = scorer.get_embeddings(max_points=600)
    with open(EMBED_PATH, "w", encoding="utf-8") as f:
        json.dump({"points": embeddings, "updated": scorer.last_updated}, f)
    print(f"埋め込み保存: {EMBED_PATH}  ({len(embeddings)}点)")

    _backtest(scorer, won, lost)


def _backtest(scorer: UMAPAnomalyScorer, won: list, lost: list) -> None:
    print("\n=== バックテスト ===")

    def get_scores(cases, n=400):
        out = []
        for c in cases[:n]:
            row = {f: _extract_val(c, f) for f in FEATURES}
            s, _, _ = scorer.score(pd.DataFrame([row]))
            out.append(s)
        return np.array(out)

    w_scores = get_scores(won)
    l_scores = get_scores(lost)
    print(f"成約  mean={w_scores.mean():.1f}  median={np.median(w_scores):.1f}")
    print(f"失注  mean={l_scores.mean():.1f}  median={np.median(l_scores):.1f}")
    print(f"差分  {l_scores.mean() - w_scores.mean():+.1f}  (負=失注が低い=正常)")

    # Mahalanobis との独立性確認
    try:
        from mahalanobis_engine import MahalanobisScorer
        maha = MahalanobisScorer.load(os.path.join(_DIR, "data", "mahalanobis_model.joblib"))
        all_cases_sub = won[:300] + lost[:300]
        umap_sc, maha_sc = [], []
        for c in all_cases_sub:
            row = {f: _extract_val(c, f) for f in FEATURES}
            df_row = pd.DataFrame([row])
            us, _, _ = scorer.score(df_row)
            ms, *_ = maha.get_analysis(df_row)
            umap_sc.append(us)
            maha_sc.append(float(ms))
        r = float(np.corrcoef(umap_sc, maha_sc)[0, 1])
        print(f"\nUMAP異常スコア vs マハラノビス相関: r={r:.3f}")
        if abs(r) < 0.70:
            print("  → |r|<0.70: 独立したシグナルとして採用可能")
        else:
            print("  → |r|>=0.70: 重複シグナルの可能性。散布図のみ提供を検討")
    except Exception as e:
        print(f"マハラノビス比較スキップ: {e}")


if __name__ == "__main__":
    train()
