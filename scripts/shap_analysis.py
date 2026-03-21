"""
SHAP（SHapley Additive exPlanations）分析スクリプト
====================================================
past_cases.jsonl の成約/失注データでXGBoostモデルを学習し、
SHAPによる判定根拠の可視化を行う。

生成するグラフ:
  1. summary_bar.png   - 全変数の平均|SHAP|ランキング（変数重要度）
  2. summary_beeswarm.png - 各変数の影響方向と大きさ（蜂の巣グラフ）
  3. waterfall_*.png   - 個別案件の判定根拠（なぜ成約/失注したか）

使い方:
    python scripts/shap_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

import shap
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# ==============================
# 日本語フォント設定
# ==============================
FONT_PATH = Path(__file__).parent.parent / "NotoSansCJKjp-Regular.otf"
if FONT_PATH.exists():
    fm.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    plt.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "DejaVu Sans"]

OUT_DIR = Path(__file__).parent.parent / "dashboard_images" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# データ読み込み
# ==============================
DATA_PATH = Path(__file__).parent.parent.parent / "past_cases.jsonl"
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent.parent / "past_cases.jsonl"

cases = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
cases = [c for c in cases if c.get("final_status") in ("成約", "失注")]
print(f"分析対象: {len(cases)}件  成約:{sum(1 for c in cases if c['final_status']=='成約')}  失注:{sum(1 for c in cases if c['final_status']=='失注')}")

# ==============================
# 特徴量エンジニアリング
# ==============================
CAT_COLS = ["industry_major", "customer_type", "main_bank",
            "competitor", "grade", "deal_source", "lease_asset"]
NUM_COLS = ["nenshu", "op_profit", "ord_profit", "bank_credit", "lease_credit",
            "acquisition_cost", "contracts", "lease_asset_score",
            "score", "score_borrower", "pd_percent", "qual_score"]

# 日本語ラベル（グラフ表示用）
LABEL_JA = {
    "industry_major":   "業種",
    "customer_type":    "顧客区分",
    "main_bank":        "メイン銀行",
    "competitor":       "競合有無",
    "grade":            "等級",
    "deal_source":      "案件ルート",
    "lease_asset":      "リース物件",
    "nenshu":           "年商",
    "op_profit":        "営業利益",
    "ord_profit":       "経常利益",
    "bank_credit":      "銀行与信",
    "lease_credit":     "リース与信",
    "acquisition_cost": "取得価額",
    "contracts":        "契約件数",
    "lease_asset_score":"物件スコア",
    "score":            "総合スコア",
    "score_borrower":   "借手スコア",
    "pd_percent":       "デフォルト確率(%)",
    "qual_score":       "定性スコア",
}


def build_features(cases):
    rows = []
    for c in cases:
        inp = c.get("inputs", {})
        res = c.get("result", {})
        qs  = inp.get("qualitative_scoring", {})
        rows.append({
            "target":           1 if c["final_status"] == "成約" else 0,
            "industry_major":   c.get("industry_major", "")[:3],
            "customer_type":    c.get("customer_type", ""),
            "main_bank":        c.get("main_bank", ""),
            "competitor":       c.get("competitor", ""),
            "grade":            inp.get("grade", ""),
            "deal_source":      inp.get("deal_source", ""),
            "lease_asset":      inp.get("lease_asset_name", ""),
            "nenshu":           inp.get("nenshu", 0),
            "op_profit":        inp.get("op_profit", 0),
            "ord_profit":       inp.get("ord_profit", 0),
            "bank_credit":      inp.get("bank_credit", 0),
            "lease_credit":     inp.get("lease_credit", 0),
            "acquisition_cost": inp.get("acquisition_cost", 0),
            "contracts":        inp.get("contracts", 0),
            "lease_asset_score":inp.get("lease_asset_score", 0),
            "score":            res.get("score", 0),
            "score_borrower":   res.get("score_borrower", 0),
            "pd_percent":       res.get("pd_percent", 0),
            "qual_score":       qs.get("weighted_score", 0),
        })
    return pd.DataFrame(rows)


df = build_features(cases)

# カテゴリ変数をラベルエンコード
encoders = {}
df_enc = df.copy()
for col in CAT_COLS:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

feature_cols = CAT_COLS + NUM_COLS
X = df_enc[feature_cols].values
y = df_enc["target"].values

# 日本語列名リスト（グラフ用）
feature_names_ja = [LABEL_JA.get(c, c) for c in feature_cols]

# ==============================
# XGBoostモデル学習
# ==============================
model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X, y)

# 交差検証スコア（小データなので参考値）
cv_scores = cross_val_score(model, X, y, cv=min(5, len(cases)//2), scoring="accuracy")
print(f"交差検証 Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  (参考値・件数少)")

# ==============================
# SHAP値計算
# ==============================
explainer   = shap.TreeExplainer(model)
shap_values = explainer(pd.DataFrame(X, columns=feature_names_ja))

# ==============================
# グラフ1: Summary Bar（変数重要度）
# ==============================
fig, ax = plt.subplots(figsize=(9, 6))
shap.plots.bar(shap_values, max_display=15, show=False, ax=ax)
ax.set_title("変数重要度（平均|SHAP値|）\n成約/失注への影響が大きい変数ランキング", fontsize=13, pad=12)
ax.set_xlabel("平均 |SHAP値|（成約確率への影響度）", fontsize=10)
plt.tight_layout()
out = OUT_DIR / "summary_bar.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"保存: {out}")

# ==============================
# グラフ2: Beeswarm（方向+大きさ）
# ==============================
fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("各変数の影響方向と大きさ\n赤=高値・青=低値、右=成約方向・左=失注方向", fontsize=12, pad=12)
plt.tight_layout()
out = OUT_DIR / "summary_beeswarm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"保存: {out}")

# ==============================
# グラフ3: Waterfall（個別案件）
# ==============================
# 成約・失注それぞれ最初の2件を可視化
for status, label in [("成約", "contract"), ("失注", "lost")]:
    idx_list = [i for i, c in enumerate(cases) if c["final_status"] == status]
    for rank, idx in enumerate(idx_list[:2]):
        c    = cases[idx]
        pred = model.predict_proba(X[idx:idx+1])[0][1]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
        industry = c.get("industry_major", "")
        title = (
            f"【{status}】案件{rank+1}: {industry}\n"
            f"成約確率予測: {pred*100:.1f}%  実績: {c['final_status']}"
        )
        plt.title(title, fontsize=11, pad=12)
        plt.tight_layout()
        out = OUT_DIR / f"waterfall_{label}_{rank+1}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"保存: {out}")

# ==============================
# テキストサマリ出力
# ==============================
print("\n" + "="*55)
print("  SHAP 変数重要度サマリ（上位10変数）")
print("="*55)
mean_abs = np.abs(shap_values.values).mean(axis=0)
ranking  = np.argsort(mean_abs)[::-1]
print(f"{'順位':>4} {'変数':<20} {'平均|SHAP|':>12}")
print("-"*40)
for rank, i in enumerate(ranking[:10]):
    print(f"{rank+1:>4} {feature_names_ja[i]:<20} {mean_abs[i]:>12.4f}")

print(f"\n生成画像: {OUT_DIR}/")
print("  summary_bar.png         変数重要度バーグラフ")
print("  summary_beeswarm.png    影響方向ビースウォーム")
print("  waterfall_contract_*.png 成約案件の個別根拠")
print("  waterfall_lost_*.png    失注案件の個別根拠")
