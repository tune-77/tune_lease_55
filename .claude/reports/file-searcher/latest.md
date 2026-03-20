---
agent: file-searcher
task: scoring/ディレクトリ・data_cases.py SQLite初期化・score_calculation.py加点処理・lease_logic_sumaho12.pyのimport・lease_data.dbテーブル一覧
timestamp: 2026-03-20 14:00
status: success
reads_from: []
---

## サマリー

scoring/ ディレクトリ（5ファイル）の関数シグネチャ・data_cases.py の SQLite 初期化経路（migrate_to_sqlite.py に委譲）・components/score_calculation.py の加点処理箇所・lease_logic_sumaho12.py での scoring/ の動的 import・lease_data.db の全4テーブルを確認した。

---

## 1. scoring/ ディレクトリ全ファイル概要

### `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/__init__.py`
- `predict_one` と `map_industry_major_to_scoring` を公開するパッケージ init。
- `__all__ = ["predict_one", "map_industry_major_to_scoring"]`

### `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/predict_one.py`
主要関数シグネチャ:

```python
def map_industry_major_to_scoring(selected_major: str) -> str:
    """sumaho の業種文字列 → 学習モデル用業種ラベル（"製造業"/"建設業"/"サービス業"/"卸売業"/"小売業"）"""

def predict_one(
    revenue: float,
    total_assets: float,
    equity: float,
    operating_profit: float,
    net_income: float,
    machinery_equipment: float,
    other_fixed_assets: float,
    depreciation: float,
    rent_expense: float,
    industry: str,
    base_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    1件分の財務データ（円単位）で学習モデルを実行。
    戻り値: {"legacy_prob", "ai_prob", "hybrid_prob", "decision"("承認"|"否決"), "top5_reasons"}
    hybrid_prob = 0.3 * legacy_prob + 0.7 * ai_prob
    """
```

### `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/model.py`
主要クラス・シグネチャ:

```python
class CreditScoringModel:
    def __init__(self, model_type: str = 'lightgbm')
    def build_model(self, **kwargs) -> Any
    def train(self, X_train, y_train, X_val=None, y_val=None)
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray   # デフォルト確率を返す（[:,1]）
    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray
    def optimize_threshold(self, X_val, y_val, metric='f1') -> float
    def evaluate(self, X_test, y_test) -> Dict[str, float]
    def cross_validate(self, X, y, cv=5) -> Dict[str, float]
    def save_model(self, filepath: str)
    def load_model(self, filepath: str)
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame
```
対応 model_type: `'logistic'`, `'random_forest'`, `'lightgbm'`, `'xgboost'`

### `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/feature_engineering_custom.py`
主要クラス・シグネチャ:

```python
class CustomFinancialFeatures:
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame
        # 収益性・安全性・効率性・固定資産・減価償却・リース負担・規模・派生の各指標を計算（約30変数）
    def calculate_risk_flags(self, df: pd.DataFrame) -> pd.DataFrame
        # 赤字・営業赤字・債務超過・低自己資本・低ROA・固定資産過大・高リース負担等のフラグ（0/1）
    def get_important_features_for_lease(self) -> List[str]
        # リース審査で重要な特徴量名リスト（収益性・安全性・固定資産・リース負担・リスクフラグ 計30件）
    def validate_input_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]
    def create_sample_data_format(self) -> pd.DataFrame
```

### `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/industry_hybrid_model.py`
主要クラス・シグネチャ:

```python
class IndustrySpecificHybridModel:
    def load_industry_coefficients(self, industry_coefficients: Dict, industry_intercepts: Dict = None)
    def predict_by_industry(self, X: pd.DataFrame, industry_column: str = 'industry') -> np.ndarray
        # 業種ごとの線形スコア → ロジスティック変換で確率を返す
    def get_industry_importance(self, industry: str) -> pd.DataFrame
    def compare_industries(self) -> pd.DataFrame
    def train_industry_specific_ai(self, X, y, industry_column='industry', min_samples=50)
    def create_industry_ensemble(self, X, industry_column='industry') -> np.ndarray
    def analyze_industry_differences(self) -> Dict
    def recommend_features_for_industry(self, industry: str, top_n: int = 10) -> List[str]
    def save_models(self, base_path: str = "models/industry_specific")
```

---

## 2. SQLite 初期化部分（テーブル定義）

**CREATE TABLE 文は `data_cases.py` には存在しない。**
`data_cases.py` の `save_case_log()` が DB ファイル不在時に `migrate_to_sqlite.init_db()` を呼び出す設計。

### `migrate_to_sqlite.py` — `init_db()` の CREATE TABLE

```sql
CREATE TABLE IF NOT EXISTS past_cases (
    id           TEXT PRIMARY KEY,
    timestamp    TEXT,
    industry_sub TEXT,
    score        REAL,
    user_eq      REAL,
    final_status TEXT,
    data         TEXT        -- 審査ログ全体のJSON文字列
)
```

### `components/shinsa_gunshi.py` — `init_db()` の CREATE TABLE（同じ DB に書かれる）

```sql
CREATE TABLE IF NOT EXISTS gunshi_cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    industry    TEXT    NOT NULL,
    score       REAL    NOT NULL,
    pd_pct      REAL    NOT NULL,
    resale      TEXT    NOT NULL,
    repeat_cnt  INTEGER NOT NULL DEFAULT 0,
    subsidy     INTEGER NOT NULL DEFAULT 0,
    bank        INTEGER NOT NULL DEFAULT 0,
    intuition   INTEGER NOT NULL DEFAULT 3,
    prior_prob  REAL    NOT NULL,
    posterior   REAL    NOT NULL,
    result      TEXT    NOT NULL DEFAULT '未登録',
    notes       TEXT    NOT NULL DEFAULT ''
    -- マイグレーションで lease_case_id カラムが後から追加される
);

CREATE TABLE IF NOT EXISTS phrase_weights (
    phrase_id  TEXT    PRIMARY KEY,
    industry   TEXT    NOT NULL,
    wins       INTEGER NOT NULL DEFAULT 0,
    total      INTEGER NOT NULL DEFAULT 0
);
```

---

## 3. `data/lease_data.db` 既存テーブル一覧

```
past_cases       -- 審査ログ（migrate_to_sqlite.py で定義）
gunshi_cases     -- 審査軍師の案件履歴（shinsa_gunshi.py で定義）
sqlite_sequence  -- AUTOINCREMENT管理（SQLite内部）
phrase_weights   -- 軍師フレーズ重み（shinsa_gunshi.py で定義）
```

---

## 4. `components/score_calculation.py` の加点処理

`run_scoring()` 関数内で `contract_prob`（成約可能性スコア）への加点・減点処理が複数ある。主な箇所:

| 処理内容 | 効果 |
|---|---|
| `main_bank == "メイン先"` のとき `main_bank_eff`（デフォルト5）を加算 | +5程度 |
| 競合なしのとき `comp_none_eff`（デフォルト5）を加算、競合ありは `comp_present_eff`（負値）を加算 | 可変（±） |
| 定性タグスコア × `tag_coef`（デフォルト2.0/pt、最大10pt）＋熱意テキストありで `passion_coef` を加算 | 最大+20程度 |
| 自己資本比率 × `equity_coef`（回帰更新値。デフォルト0）を加算 | 回帰更新後のみ有効 |
| リース負担率が業種平均の0.5倍以下で **+2**、2倍以上は -3、3倍以上は -6 | ±6 |
| BN（ベイジアンネット）エンジン結果が session_state に格納されていれば `_bn_effect` を加算 | 可変 |
| 競合金利差（自社が有利なほどプラス） | 可変 |
| カスタムルール（`business_rules.json` 定義） score_delta を加減算 | 可変 |

加点後は `contract_prob = max(0, min(100, contract_prob))` でクランプ。

---

## 5. `lease_logic_sumaho12.py` での `scoring/` の import 方法

`lease_logic_sumaho12.py` では `scoring/` を直接 import しておらず、`components/score_calculation.py` の `run_scoring()` を呼び出す。

```python
# lease_logic_sumaho12.py L1040
from components.score_calculation import run_scoring
run_scoring(...)
```

`run_scoring()` 内で学習モデルが**実行時動的 import** される:

```python
# components/score_calculation.py L820-821
from scoring.predict_one import predict_one, map_industry_major_to_scoring
```

入力値は千円単位 → `* 1000` で円単位に変換して `predict_one()` に渡す。

---

## コアファイル（直接関連）

- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/predict_one.py` — メイン推論関数
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/model.py` — CreditScoringModel クラス
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/feature_engineering_custom.py` — 特徴量エンジニアリング
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/industry_hybrid_model.py` — 業種別ハイブリッドモデル
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/__init__.py` — パッケージ init
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/components/score_calculation.py` — スコア計算・加点処理
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/data_cases.py` — SQLite読み書き（init_db は migrate_to_sqlite に委譲）
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/migrate_to_sqlite.py` — DBスキーマ定義（init_db / past_cases テーブル）
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/components/shinsa_gunshi.py` — gunshi_cases / phrase_weights テーブル定義

## 関連ファイル（間接的）

- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/lease_logic_sumaho12.py` — Streamlit エントリポイント（score_calculation を呼び出す）
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/coeff_definitions.py` — COEFFS・BAYESIAN_PRIOR_EXTRA 等の係数定義
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/bayesian_engine.py` — BN エンジン（THRESHOLD_APPROVAL 等）
- `/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/scoring/models/industry_specific/` — 学習済みモデルファイル群（.pkl）

---

## 課題・リスク

- `data_cases.py` に CREATE TABLE がなく `migrate_to_sqlite.py` に分離されているため、テーブル追加マイグレーションは手動対応が必要。
- `predict_one` はエラーを全 `try/except` で握り潰して `None` を返す。モデルファイル不備でも無警告で素通りする。
- `score_calculation.py` の加点処理は `contract_prob` に対して複数箇所で直接加算しており、デバッグが困難。

## 後続エージェントへの申し送り

- **change-impact-analyzer**: `score_calculation.py` の加点処理（L539-686, L892-894）は `data_cases.py` の係数・ルール管理と密結合。変更時は影響範囲の確認を推奨。
- **code-reviewer**: `predict_one` のエラーサイレント設計と `contract_prob` の複数箇所クランプ処理をレビュー推奨。
- **security-checker**: `data/lease_data.db` は `.gitignore` 対象だが、審査データが JSON 文字列丸ごと `data` カラムに格納されているため個人情報・財務情報の取り扱いを確認推奨。
