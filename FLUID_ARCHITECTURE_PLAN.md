# tuneLease55 流体化アーキテクチャ計画書

> 作成: 2026-05-24  
> 対象: tune_lease_55（Tune式リース審査AIシステム）

---

## 現状診断：「孤立した池」問題

コードベースを精査した結果、流動化のための**部品はほぼ揃っている**。不足しているのは「接続」だ。

| 既存ファイル | 機能 | 状態 |
|---|---|---|
| `auto_optimizer.py` | 係数自動最適化（20件ごと） | 稼働中・孤立 |
| `retraining_pipeline.py` | LightGBM/RF再学習パイプライン | 稼働中・孤立 |
| `macro_drift_monitor.py` | コンセプトドリフト検知 | 稼働中・孤立 |
| `llm_pdca_reflection.py` | LLMによる月次PDCA反省 | 稼働中・出力が未活用 |
| `bayesian_engine.py` | ベイジアンネットワーク（pgmpy） | 実装済み・スコアに未接続 |
| `model_review_hooks.py` | AUC監視フック | 稼働中・孤立 |
| `natural_gradient_optimizer.py` | 自然勾配法最適化 | 実装済み・未統合 |
| `fetch_estat_annual.py` | e-Stat業種データ取得 | スクリプト存在・未スケジュール |
| `scoring/industry_hybrid_model.py` | 業種別ハイブリッドモデル | クラス定義済み・未接続 |

**核心的問題**: これらは全て孤立したスクリプトとして動いており、互いに通信していない。  
ドリフトを検知しても再学習は起動しない。PDCA反省ルールが生成されても軍師プロンプトには届かない。  
実際の成約/失注データはあるが、デフォルト判定フィードバックが存在しない（`delinquent`フィールドが未入力）。

---

## 全体アーキテクチャ図：「流体方程式システム」

```
╔══════════════════════════════════════════════════════════════════════╗
║                    外部環境入力ストリーム                              ║
║  [日銀金利API] [e-Stat業種統計] [RSSニュース] [EDINET決算]             ║
╚══════════════╦═══════════════════════════════════════════════════════╝
               ↓
╔══════════════╩═══════════════════════════════╗
║     FluidPipeline（中央イベントバス）         ║  ← 新規作成: fluid_pipeline.py
║  ┌─────────────────────────────────────┐    ║
║  │  EventBus: DRIFT / NEW_CASES /      │    ║
║  │  MODEL_UPDATED / KNOWLEDGE_UPDATED  │    ║
║  └──────────────┬──────────────────────┘    ║
╚═════════════════╪════════════════════════════╝
         ┌────────┴────────────────────────┐
         ↓                                 ↓
╔════════╩══════════╗            ╔═════════╩═════════╗
║  学習エンジン層    ║            ║  知識エンジン層     ║
║                   ║            ║                    ║
║ [macro_drift_     ║  DRIFT     ║ [obsidian RAG]     ║
║  monitor.py]      ║──event──→  ║ [fetch_estat]      ║
║       ↓           ║            ║ [lease_news_digest]║
║ [retraining_      ║            ║       ↓            ║
║  pipeline.py]     ║            ║ [KnowledgeFeeder]  ║
║       ↓           ║            ║  (新規)            ║
║ [auto_optimizer]  ║            ╚════════════════════╝
║       ↓           ║                     ↓
║ [bayesian_engine] ║            KNOWLEDGE_UPDATED event
╚═════════╤═════════╝                     ↓
          ↓                    ╔═══════════╩══════════╗
╔═════════╩═════════════╗      ║  重み動的調整エンジン  ║
║  モデルアンサンブル層   ║      ║                      ║
║                       ║      ║ [IndustryWeightMatrix]║
║ LightGBM (既存) +     ║      ║  業種×時期 → 重み     ║
║ BayesianNet (既存) +  ║      ║  (新規)               ║
║ IndustryHybrid (既存) ║      ╚═══════════╤══════════╝
║       ↓               ║                  ↓
║ スコア合成             ║←←←←←←←←←←←←←←←
╚══════════╤════════════╝
           ↓
╔══════════╩════════════════════╗
║    審査・出力層                ║
║                               ║
║ [scoring_core.py]             ║
║ [shinsa_gunshi_logic.py]      ║  ← プロンプト動的更新（PDCA連携）
║ [quantum_analysis_module.py]  ║
╚══════════╤════════════════════╝
           ↓
╔══════════╩════════════════════╗
║    フィードバック収集層         ║
║                               ║
║ [ScreeningOutcomes DB]  ←新規 ║  成約/失注/遅延/デフォルト
║ [DelayedFeedbackHandler] ←新規║  タイムラグ付き帰結追跡
║ [llm_pdca_reflection.py]      ║  月次→週次化
╚══════════╤════════════════════╝
           ↓（フィードバックループ）
           ↑←←←←←←←← FluidPipeline（NEW_CASES event）
```

---

## フェーズ別実装ロードマップ

### Phase 0：基盤整備（1〜2週間）★最優先

**目的**: 全ての流体化の前提となるフィードバックデータを揃える。  
部品はあっても「実際のデフォルト/延滞データ」がなければ何も学習できない。

#### 0-A：成果フィードバックテーブル整備

```sql
-- data/lease_data.db に追加
CREATE TABLE IF NOT EXISTS screening_outcomes (
    case_id         TEXT    PRIMARY KEY,
    contract_date   TEXT,           -- 成約日
    scheduled_end   TEXT,           -- リース満了予定日
    actual_status   TEXT,           -- 'normal' | 'late_30' | 'late_90' | 'default'
    status_updated_at TEXT,
    loss_given_default REAL,        -- 実損額（円）
    notes           TEXT
);
```

**着手ファイル**: `migrate_to_sqlite.py` にマイグレーション追加  
**UI**: `components/` に `outcome_recorder.py` を新規作成（既存案件に後から結果入力できる画面）

#### 0-B：`delinquent`フラグの実データ接続

`retraining_pipeline.py` の `FEATURE_COLS` は正しいが、ターゲット変数 `delinquent` が  
どこからも入力されていない（常に 0 かNULL）。

```python
# retraining_pipeline.py の _load_training_data() を修正
# screening_outcomes テーブルを JOIN して actual_status から delinquent を計算する
def _load_training_data(conn):
    query = """
    SELECT sr.*, 
           CASE WHEN so.actual_status IN ('late_90','default') THEN 1 ELSE 0 END as delinquent
    FROM screening_records sr
    LEFT JOIN screening_outcomes so ON sr.case_id = so.case_id
    WHERE so.actual_status IS NOT NULL  -- 結果が判明した案件のみ
    """
```

**工数**: small（2〜3h）、**インパクト**: 全学習系を実データで動かす前提条件

---

### Phase 1：パイプラインオーケストレーター（2〜3週間）★高優先

**目的**: 孤立した部品を「接続された川」にする。

#### 1-A：`fluid_pipeline.py` 新規作成

既存の各モジュールを呼び出す軽量なイベントバス。重い処理はサブプロセスで起動。

```python
# fluid_pipeline.py（新規）
class FluidPipelineEvent(Enum):
    NEW_CASES = "new_cases"         # 案件が蓄積された
    DRIFT_DETECTED = "drift"        # ドリフト検知
    MODEL_UPDATED = "model_updated" # モデル更新完了
    KNOWLEDGE_UPDATED = "knowledge" # 知識ベース更新

class FluidPipeline:
    """既存モジュールを繋ぐイベントバス。新規ロジックは書かない。"""
    
    def on_case_registered(self, case_count: int):
        """案件登録のたびに呼ばれる（screening_recorder.py から）"""
        # ドリフト検知（既存 macro_drift_monitor.py）
        result = check_concept_drift()
        if result["is_drift"]:
            self.emit(FluidPipelineEvent.DRIFT_DETECTED, result)
        
        # 再学習判定（既存 auto_optimizer.py）
        if should_retrain(case_count):
            trigger_retraining()  # retraining_pipeline.py
            self.emit(FluidPipelineEvent.MODEL_UPDATED, {})
    
    def on_model_updated(self, metrics: dict):
        """モデル更新後：PDCA反省 & 軍師プロンプト更新"""
        # 既存 llm_pdca_reflection.py 呼び出し
        run_monthly_pdca_reflection(force=True)
        # PDCA結果を軍師プロンプトに反映（後述 Phase 2）
        update_gunshi_system_prompt_from_pdca()
```

**接続ポイント**: `screening_recorder.py` の `record_screening()` 末尾に  
`FluidPipeline().on_case_registered()` を1行追加するだけ。

#### 1-B：定期外部データ取得のスケジュール化

```python
# scripts/daily_knowledge_feed.py（新規・cronまたはlaunchdで実行）
def run_daily_feed():
    # 既存スクリプトを順番に呼ぶだけ
    fetch_estat_annual()      # 既存: fetch_estat_annual.py
    fetch_base_rate()         # 新規: 日銀API（短期金利・長期金利）
    fetch_industry_news_rss() # 新規: リース/設備投資関連RSSフィード
    reindex_obsidian()        # 既存: obsidianのベクトルDB更新
```

```bash
# launchd/com.tunelease.daily_feed.plist（既存 launchd/ ディレクトリに追加）
# 毎朝6時に実行
```

---

### Phase 2：評価基準の流動化（3〜4週間）★中優先

**目的**: 業種・時期によってスコア重みが変わる仕組みを作る。

#### 2-A：`IndustryWeightMatrix` の実装

`scoring/industry_hybrid_model.py` には `IndustrySpecificHybridModel` クラスが既にあるが  
係数読み込みロジックが未完成。これを `data/industry_capex_lease.json`（fetch_estat が生成）  
と接続する。

```python
# industry_weight_matrix.py（新規）
class IndustryWeightMatrix:
    """
    業種コード × 期間 → (borrower_weight, asset_weight, quant_weight, qual_weight) を返す。
    e-Stat データ + 過去審査結果から自動更新。
    """
    
    def get_weights(self, industry_major: str, as_of: date) -> dict:
        """
        data_cases.py の get_score_weights() の代替として呼ばれる。
        デフォルトは既存の (0.85, 0.15, 0.6, 0.4) を返すので後方互換。
        """
        key = self._lookup_key(industry_major, as_of)
        return self._weight_cache.get(key, DEFAULT_WEIGHTS)
    
    def recompute(self, conn: sqlite3.Connection):
        """
        screening_outcomes と業種データから各業種の最適重みを再計算。
        Phase 1 の MODEL_UPDATED イベントで呼ばれる。
        """
        # 業種別デフォルト率 × e-Stat 業種特性 → 重み更新
```

`data_cases.py` の `get_score_weights()` を差し替えるのではなく、  
`scoring_core.py` の呼び出し部分だけを `IndustryWeightMatrix.get_weights()` に切り替える。

#### 2-B：軍師プロンプトのPDCA自動更新

現状: `llm_pdca_reflection.py` は `data/pdca_ai_rules.json` を生成するが、  
`shinsa_gunshi_logic.py` はこのファイルを読んでいない。

```python
# shinsa_gunshi_logic.py の SYSTEM_PROMPT 生成部分に追加
def _build_system_prompt(base_prompt: str) -> str:
    """PDCA反省結果を動的にシステムプロンプトへ注入"""
    pdca_rules = load_pdca_rules()  # 既存 llm_pdca_reflection.py
    
    if pdca_rules.get("generated_at"):
        recent_lessons = pdca_rules.get("lessons_learned", [])[-3:]  # 直近3件
        injection = "\n\n【最近の審査精度フィードバック】\n"
        injection += "\n".join(f"- {l}" for l in recent_lessons)
        return base_prompt + injection
    return base_prompt
```

---

### Phase 3：知識の流動化（4〜6週間）★中優先

**目的**: 外部環境が自動で知識ベースに流れ込む仕組みを作る。

#### 3-A：`KnowledgeFeeder` 新規作成

```python
# knowledge_feeder.py（新規）
class KnowledgeFeeder:
    """
    外部データを信頼度付きでObsidian/ChromaDBに投入する。
    古い知識を自動でdeprecateする。
    """
    
    SOURCES = {
        "boj_rate": {"url": "https://www.boj.or.jp/...", "trust": 0.95, "ttl_days": 30},
        "nikkei_rss": {"url": "https://rss.nikkei.com/...", "trust": 0.75, "ttl_days": 7},
        "estat_industry": {"trust": 0.90, "ttl_days": 365},
    }
    
    def ingest(self, source_key: str, content: str, metadata: dict):
        """信頼度と有効期限付きでベクトルDBに投入"""
        doc = {
            "content": content,
            "source": source_key,
            "trust_score": self.SOURCES[source_key]["trust"],
            "expires_at": (datetime.now() + timedelta(
                days=self.SOURCES[source_key]["ttl_days"])).isoformat(),
            **metadata
        }
        self._upsert_to_chroma(doc)
    
    def deprecate_expired(self):
        """有効期限切れのドキュメントをベクトルDBから削除または低優先化"""
        pass
```

#### 3-B：取得する外部データソース

| ソース | 取得方法 | 既存ファイル | 優先度 |
|---|---|---|---|
| 日銀政策金利 | JSON API | なし（新規） | 高 |
| 法人企業統計（年次） | e-Stat API | `fetch_estat_annual.py` ✓ | 高 |
| 産業別倒産率 | TSR/帝国DB | `industry_benchmarks.json` ✓ | 中 |
| リース業界ニュース | RSS | `lease_news_digest.py` ✓（手動） | 中 |
| EDINET有価証券報告書 | EDINET API | `edinet_collector.py` ✓ | 低 |

---

### Phase 4：遅延フィードバック処理（6〜8週間）★低優先

**目的**: リースの性質（3〜7年）に合わせた時間軸でのフィードバック設計。

#### 4-A：`DelayedFeedbackHandler`

```python
# delayed_feedback_handler.py（新規）
class DelayedFeedbackHandler:
    """
    リース審査の特性：契約から判定まで数年かかる。
    中間シグナル（30日延滞、90日延滞）を早期警戒として活用し、
    最終判定（デフォルト/正常完了）まで待たずに学習を進める。
    """
    
    # 重み付け：最終デフォルトを1.0として
    SIGNAL_WEIGHTS = {
        "normal_12months":  0.3,  # 12ヶ月延滞なし → 予測精度向上に寄与
        "late_30":          0.5,  # 30日延滞 → 要注意シグナル
        "late_90":          0.8,  # 90日延滞 → 準デフォルトとして扱う
        "full_default":     1.0,  # 確定デフォルト
        "normal_complete":  1.0,  # 正常完了
    }
    
    def collect_intermediate_signals(self):
        """毎月実行: 既存案件の支払状況をチェックしてDBに記録"""
        pass
    
    def compute_weighted_labels(self, conn) -> pd.DataFrame:
        """確定していない案件の暫定ラベルを計算して学習に使用"""
        pass
```

---

## 最初に着手すべき「最小実装」

Phase 0 の中でも**今週着手できる最小セット**を示す。  
この3つが揃えば「流体化の最低限の骨格」が完成する。

### STEP 1：フィードバックDBテーブル（1日）

```bash
# migrate_to_sqlite.py に以下を追記して実行
python3 migrate_to_sqlite.py --add-outcomes-table
```

`screening_outcomes` テーブルを作成し、既存の `past_cases.jsonl` の  
`final_status`（成約/失注）を自動マイグレーションする。

### STEP 2：`FluidPipeline` の最小実装（2日）

```python
# fluid_pipeline.py（最初は30行で十分）
def on_case_registered():
    """screening_recorder.py の末尾から呼ぶ。これだけ。"""
    from macro_drift_monitor import check_concept_drift
    from auto_optimizer import get_registered_count, should_retrain
    
    drift = check_concept_drift()
    if drift["is_drift"]:
        print(f"[FluidPipeline] ドリフト検知: {drift['message']}")
        # Slack通知（既存 slack_notify.py）
        
    if should_retrain(get_registered_count()):
        from retraining_pipeline import run_retraining
        run_retraining(triggered_by="fluid_pipeline_auto")
```

### STEP 3：軍師プロンプトのPDCA注入（半日）

`shinsa_gunshi_logic.py` の `_SYSTEM_PROMPT` 定数を関数に変更し、  
`data/pdca_ai_rules.json` の内容を末尾に注入する。  
既存の `llm_pdca_reflection.py` はそのまま使える。

---

## リスクと技術的課題

### リスク1：フィードバックデータ不足（最大リスク）

**問題**: 実際のデフォルト案件が少ない（優良案件が多い）と、  
モデルの正例クラスが極端に少なくなる（クラス不均衡）。

**対策**: 
- `retraining_pipeline.py` に `scale_pos_weight`（LightGBM）の自動計算を追加
- 30日延滞・90日延滞を「弱い正例」として使う（`DelayedFeedbackHandler`）
- 最低50件の結果判明案件が揃うまでは既存モデルを維持

### リスク2：自動再学習による精度劣化

**問題**: 少数データで再学習するとモデルが悪化する可能性がある。

**対策**: 既存の `retraining_pipeline.py` には AUC 比較ロールバック機構が実装済み。  
`AUC_MIN_IMPROVEMENT = -0.02` の設定を `-0.01` に締める（劣化許容度を下げる）。  
`model_review_hooks.py` の監視も流体パイプラインに接続する。

### リスク3：外部データ品質

**問題**: RSSフィード・APIの品質がバラバラ。誤情報が知識ベースに入る。

**対策**: `KnowledgeFeeder` の `trust_score` による重み付け検索。  
一次ソース（日銀・e-Stat・EDINET）は trust=0.9+、ニュースは trust=0.7 として  
RAGの検索結果に信頼度をメタデータとして付与し、軍師プロンプトに渡す。

### リスク4：Streamlit セッション状態との競合

**問題**: 自動再学習が Streamlit の st.session_state を壊す可能性がある。

**対策**: 再学習はサブプロセスで実行（`subprocess.Popen`）。  
モデルファイルのスワップは `FileLock` を使う（`retraining_pipeline.py` で既に実装済み）。  
Streamlit 側は起動時にのみモデルをロードし、`st.cache_resource` でキャッシュする。

### リスク5：係数最適化の暴走

**問題**: `auto_optimizer.py` と `retraining_pipeline.py` が独立して係数を変えると矛盾が生じる。

**対策**: `FluidPipeline` が**唯一の再学習トリガー**になり、両者が同時に走らないよう  
`FileLock`（既存 `.retraining.lock`）を共有する。  
`coeff_history.jsonl`（既存）への記録を再学習ごとに必ず行う。

---

## 実装優先度マトリクス

| タスク | 影響度 | 工数 | 着手順 |
|---|---|---|---|
| `screening_outcomes` テーブル作成 | ★★★★★ | S(1日) | **1番目** |
| `delinquent` フィールドを実データ接続 | ★★★★★ | S(半日) | **2番目** |
| `FluidPipeline` 最小実装（接続のみ） | ★★★★☆ | S(2日) | **3番目** |
| 軍師プロンプトへのPDCA注入 | ★★★☆☆ | S(半日) | **4番目** |
| `daily_knowledge_feed.py` + launchd | ★★★☆☆ | M(3日) | **5番目** |
| `IndustryWeightMatrix` 実装 | ★★★☆☆ | M(4日) | **6番目** |
| `KnowledgeFeeder` + 信頼度管理 | ★★☆☆☆ | L(1週) | **7番目** |
| `DelayedFeedbackHandler` | ★★☆☆☆ | L(1週) | **8番目** |

---

## 「流体方程式」としての完成形イメージ

```
時刻 t=0: 案件Aが登録される
     ↓
     FluidPipeline が起動
     ↓
     ドリフト？→ No → 通常処理
                Yes → retraining_pipeline.py が非同期で起動
                       ↓
                       AUC改善 → モデルスワップ → llm_pdca_reflection.py 実行
                                                   ↓
                                                   pdca_ai_rules.json 更新
                                                   ↓
                                                   次の案件から軍師プロンプトが変わる
     ↓
     毎朝6時: daily_knowledge_feed.py 実行
     ↓
     日銀金利変動 → KnowledgeFeeder に投入
     ↓
     翌朝の軍師AIは「金利上昇局面」を織り込んだ状態で討論する

時刻 t+180日: 案件Aの「6ヶ月支払状況」確認
     ↓
     DelayedFeedbackHandler が 0.3 重みの暫定ラベルを付与
     ↓
     次の再学習でこの案件も学習データとして活用される
```

全ての流れが「データが入るたびに場全体が更新される」という  
流体方程式の性質を持つようになる。

---

*このドキュメントは実装進捗に応じて更新すること。*  
*Phase 0 完了後、Plans.md に Phase 5 として追記することを推奨。*
