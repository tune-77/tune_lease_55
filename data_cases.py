"""
案件・係数・重みの読み書きモジュール（lease_logic_sumaho10）
load_all_cases, save_all_cases, save_case_log, load_coeff_overrides, save_coeff_overrides,
get_score_weights, get_model_blend_weights, get_effective_coeffs, load_consultation_memory, append_consultation_memory,
load_case_news, append_case_news, find_similar_past_cases を提供。
st は使わず、保存失敗時は False/None を返す。呼び元で st.error 等を表示すること。
"""
import os
import sys
import json
import datetime
from typing import Optional
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from coeff_definitions import COEFFS
from charts import _equity_ratio_display

# ファイルパス（絶対パス固定）
_DATA_DIR = "/Users/kobayashiisaoryou/clawd/lease_logic_sumaho12/data"
CASES_FILE = os.path.join(os.path.dirname(_DATA_DIR), "past_cases.jsonl") # obsolete
DB_PATH = os.path.join(_DATA_DIR, "lease_data.db")
COEFF_OVERRIDES_FILE = os.path.join(_DATA_DIR, "coeff_overrides.json")
COEFF_AUTO_FILE      = os.path.join(_DATA_DIR, "coeff_auto.json")
COEFF_HISTORY_FILE   = os.path.join(_DATA_DIR, "coeff_history.jsonl")
CONSULTATION_MEMORY_FILE = os.path.join(_DATA_DIR, "consultation_memory.jsonl")
CASE_NEWS_FILE = os.path.join(_DATA_DIR, "case_news.jsonl")

# スコア重みのデフォルト（借手/物件、総合/定性）。回帰最適化で上書き可能。
DEFAULT_WEIGHT_BORROWER = 0.85
DEFAULT_WEIGHT_ASSET = 0.15
DEFAULT_WEIGHT_QUANT = 0.6
DEFAULT_WEIGHT_QUAL = 0.4


def load_consultation_memory(max_entries=20):
    """AI審査オフィサー相談のメモを読み込む。直近 max_entries 件を返す。"""
    if not os.path.exists(CONSULTATION_MEMORY_FILE):
        return []
    entries = []
    try:
        with open(CONSULTATION_MEMORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, TypeError):
                    continue
    except (OSError, IOError, PermissionError):
        return []
    return entries[-max_entries:] if len(entries) > max_entries else entries


def append_consultation_memory(user_text: str, assistant_text: str):
    """相談1往復をメモに追記。失敗時は静かに無視。"""
    try:
        with open(CONSULTATION_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "user": (user_text or "")[:5000],
                "assistant": (assistant_text or "")[:5000],
                "ts": datetime.datetime.now().isoformat(),
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_all_cases():
    """過去案件を全件読み込み（SQLiteから）。

    past_cases（詳細JSON）と screening_records（サマリー）をマージして返す。
    同一IDが両方にある場合は past_cases 側を優先（final_status等の更新情報があるため）。
    """
    import sqlite3
    from contextlib import closing

    # ── 1. past_cases から読み込み（詳細JSON） ──────────────────────
    past_map = {}  # id -> data dict
    if os.path.exists(DB_PATH):
        try:
            with closing(sqlite3.connect(DB_PATH)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT data FROM past_cases ORDER BY timestamp ASC")
                for row in cursor.fetchall():
                    try:
                        d = json.loads(row[0])
                        if d.get("id"):
                            past_map[str(d["id"])] = d
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[Error in load_all_cases/past_cases]: {e}", file=sys.stderr)

    # ── 2. screening_records から読み込み（サマリー列） ────────────
    _SCREENING_DB = os.path.join(_DATA_DIR, "screening_db.sqlite")
    screening_extra = []
    if os.path.exists(_SCREENING_DB):
        try:
            with closing(sqlite3.connect(_SCREENING_DB)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM screening_records ORDER BY created_at ASC"
                )
                for row in cursor.fetchall():
                    row_id = str(row["id"])
                    if row_id in past_map:
                        continue  # past_cases 側を優先
                    # サマリー列から最低限のデータ構造を組み立てる
                    result_json = {}
                    try:
                        if row["memo"]:
                            result_json = json.loads(row["memo"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    # memo JSON から会社名・詳細情報を取り出す
                    company_name = result_json.get("company_name", "")
                    company_no   = result_json.get("company_no", "")
                    entry = {
                        "id": row_id,
                        "timestamp": row["created_at"],
                        "industry_sub": row["industry_sub"] or result_json.get("industry_sub", ""),
                        "industry_major": row["industry_major"] or result_json.get("industry_major", ""),
                        "score": row["score"] or 0,
                        "hantei": row["judgment"] or "—",
                        "final_status": result_json.get("final_status", "未登録"),
                        "company_name": company_name,
                        "company_no": company_no,
                        "inputs": result_json.get("inputs", {
                            "nenshu": row["revenue_m"] or 0,
                            "op_profit": row["op_profit_m"] or 0,
                        }),
                        "result": result_json.get("result", {
                            "score": row["score"] or 0,
                            "hantei": row["judgment"] or "—",
                            "contract_prob": row["contract_prob"] or 0,
                        }),
                        "pricing": result_json.get("pricing", {}),
                        "_from_screening_records": True,
                        "_memo_raw": row["memo"] or "",
                    }
                    screening_extra.append(entry)
        except Exception as e:
            print(f"[Error in load_all_cases/screening_records]: {e}", file=sys.stderr)

    # ── 3. マージして timestamp 昇順でソート ───────────────────────
    all_cases = list(past_map.values()) + screening_extra
    all_cases.sort(key=lambda c: c.get("timestamp", ""), reverse=False)
    return all_cases


def load_past_cases():
    """save_case_log で保存された過去の審査ログをすべて読み込む。"""
    return load_all_cases()


try:
    import streamlit as _st

    @_st.cache_data(ttl=60)
    def load_all_cases_cached() -> list:
        """load_all_cases の 1 分キャッシュ版（Streamlit 環境専用）。"""
        return load_all_cases()

except Exception:
    def load_all_cases_cached() -> list:  # type: ignore[misc]
        return load_all_cases()


def find_similar_past_cases(current_case_data: dict, max_count: int = 3):
    """
    現在の案件データに基づき、財務・属性の近い過去案件を高度な手法で検索する。
    """
    all_past = load_all_cases()
    if not all_past:
        return []

    from case_similarity import CaseSimilarityEngine
    engine = CaseSimilarityEngine(all_past)
    
    similar_results = engine.find_similar(current_case_data, top_n=max_count)
    
    # UI表示用に必要な情報を抽出して返す
    output = []
    for item in similar_results:
        case = item["case"]
        output.append({
            "id": case.get("id"),
            "name": case.get("borrower_name", "匿名企業"),
            "industry": case.get("industry_sub", ""),
            "score": item["case"].get("result", {}).get("score", 0),
            "status": case.get("final_status", "未登録"),
            "similarity": round(item["similarity"] * 100, 1),
            "equity": round(float(case.get("equity_ratio", 0) or case.get("user_eq", 0) or 0) * 100, 1),
            "revenue": case.get("nenshu", 0),
            "result": case.get("result", {}),
            "data": case
        })
    return output


def analyze_lost_cases(industry_sub=None):
    """
    失注案件の統計を分析する。
    """
    all_cases = load_all_cases()
    # ケースの data プロパティまたはルートからステータスを確認
    lost_cases = []
    for c in all_cases:
        status = c.get("final_status")
        if status == "失注":
            lost_cases.append(c)
    
    if industry_sub:
        lost_cases = [c for c in lost_cases if c.get("industry_sub") == industry_sub]
        
    reasons = {}
    competitors = {}
    comp_rates = []
    
    for c in lost_cases:
        r = c.get("lost_reason", "不明")
        if not r: r = "不明"
        reasons[r] = reasons.get(r, 0) + 1
        
        comp = c.get("competitor_name", "不明")
        if comp:
            competitors[comp] = competitors.get(comp, 0) + 1
        
        rate = c.get("competitor_rate")
        if rate and isinstance(rate, (int, float)):
            comp_rates.append(rate)
            
    return {
        "total": len(lost_cases),
        "reasons": reasons,
        "competitors": competitors,
        "avg_competitor_rate": sum(comp_rates) / len(comp_rates) if comp_rates else None,
        "cases": lost_cases
    }

def delete_case(case_id: str) -> bool:
    """指定IDの案件を1件削除する。全件置き換えを使わない安全な単体削除。"""
    if not os.path.exists(DB_PATH):
        return False
    try:
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.execute("DELETE FROM past_cases WHERE id = ?", (case_id,))
            conn.commit()
        return True
    except Exception:
        return False


def update_case(case_id: str, updates: dict) -> bool:
    """指定IDの案件の data フィールドを更新する。全件置き換えを使わない安全な単体更新。"""
    if not os.path.exists(DB_PATH):
        return False
    try:
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT data FROM past_cases WHERE id = ?", (case_id,)
            ).fetchone()
            if row is None:
                return False
            data = json.loads(row[0])
            data.update(updates)
            final_status = data.get("final_status", "")
            json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
            conn.execute(
                "UPDATE past_cases SET data = ?, final_status = ? WHERE id = ?",
                (json_str, final_status, case_id),
            )
            conn.commit()
        return True
    except Exception:
        return False


def save_all_cases(cases):
    """案件一覧をUPSERT保存。既存データは上書き、新規データは追加。既存レコードは削除しない。

    ⚠️ 注意: 後方互換のため残しているが、新規コードでは
    delete_case() / update_case() / save_case_log() を使うこと。
    """
    if not os.path.exists(DB_PATH):
        return False

    try:
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("BEGIN")
                # DELETEはせず、UPSERT（INSERT OR REPLACE）で処理するよう修正
                for data in cases:
                    case_id = data.get("id")
                    timestamp = data.get("timestamp", "")
                    industry_sub = data.get("industry_sub", "")
                    final_status = data.get("final_status", "")

                    score, user_eq = None, None
                    res = data.get("result", {})
                    if isinstance(res, dict):
                        score = res.get("score")
                        user_eq = res.get("user_eq")

                    try:
                        score_val = float(score) if score is not None else None
                    except (TypeError, ValueError):
                        score_val = None

                    try:
                        user_eq_val = float(user_eq) if user_eq is not None else None
                    except (TypeError, ValueError):
                        user_eq_val = None

                    json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
                    # INSERT OR REPLACE: 同一IDが存在すれば上書き、なければ追加
                    cursor.execute("""
                        INSERT OR REPLACE INTO past_cases
                        (id, timestamp, industry_sub, score, user_eq, final_status, data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (case_id, timestamp, industry_sub, score_val, user_eq_val, final_status, json_str))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return True
    except Exception:
        return False


def load_coeff_overrides():
    """保存済みの係数オーバーライド（手動設定）を読み込む。無ければ None。"""
    if not os.path.exists(COEFF_OVERRIDES_FILE):
        return None
    try:
        with open(COEFF_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_coeff_overrides(overrides_dict, comment: str = ""):
    """係数オーバーライド（手動設定）を JSON で保存。変更履歴も記録。失敗時は False。"""
    dirpath = os.path.dirname(COEFF_OVERRIDES_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        # 変更前の値を取得して差分を記録
        before = load_coeff_overrides() or {}
        with open(COEFF_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides_dict, f, ensure_ascii=False, indent=2)
        _append_coeff_history(
            change_type="manual",
            before=before,
            after=overrides_dict,
            comment=comment,
        )
        _save_governance_snapshot(overrides=overrides_dict, comment=comment)
        return True
    except Exception:
        return False


def _save_governance_snapshot(overrides: dict, comment: str = "") -> None:
    """係数オーバーライドのスナップショットを governance_snapshots.json に追記する（最大50件）。"""
    snap_path = os.path.join(_DATA_DIR, "governance_snapshots.json")
    try:
        if os.path.exists(snap_path):
            with open(snap_path, "r", encoding="utf-8") as f:
                snaps = json.load(f)
        else:
            snaps = []
        snaps.append({
            "id": f"snap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "comment": comment,
            "overrides": overrides,
        })
        snaps = snaps[-50:]
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snaps, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_governance_snapshots() -> list:
    """ガバナンス・スナップショット一覧を返す（新しい順）。無ければ空リスト。"""
    snap_path = os.path.join(_DATA_DIR, "governance_snapshots.json")
    try:
        if not os.path.exists(snap_path):
            return []
        with open(snap_path, "r", encoding="utf-8") as f:
            snaps = json.load(f)
        return list(reversed(snaps))
    except Exception:
        return []


def load_auto_coeffs() -> dict:
    """自動最適化で生成された推奨重みを読み込む。無ければ空 dict。"""
    if not os.path.exists(COEFF_AUTO_FILE):
        return {}
    try:
        with open(COEFF_AUTO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_auto_coeffs(auto_dict: dict, comment: str = "") -> bool:
    """自動最適化の推奨重みを専用ファイルに保存。変更履歴も記録。失敗時は False。"""
    dirpath = os.path.dirname(COEFF_AUTO_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        before = load_auto_coeffs() or {}
        with open(COEFF_AUTO_FILE, "w", encoding="utf-8") as f:
            json.dump(auto_dict, f, ensure_ascii=False, indent=2)
        _append_coeff_history(
            change_type="auto",
            before=before,
            after=auto_dict,
            comment=comment or "自動最適化による更新",
        )
        return True
    except Exception:
        return False


def _append_coeff_history(change_type: str, before: dict, after: dict, comment: str = "") -> None:
    """係数変更履歴を JSONL に1行追記する。"""
    try:
        os.makedirs(os.path.dirname(COEFF_HISTORY_FILE), exist_ok=True)
        # 変更されたキーだけ抽出
        all_keys = set(before.keys()) | set(after.keys())
        changed = {
            k: {"before": before.get(k), "after": after.get(k)}
            for k in all_keys
            if before.get(k) != after.get(k)
        }
        record = {
            "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
            "change_type": change_type,   # "manual" or "auto"
            "comment":     comment,
            "changed_keys": changed,
            "snapshot_after": after,
        }
        with open(COEFF_HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 履歴書き込み失敗は本処理に影響させない


def load_coeff_history() -> list:
    """係数変更履歴を新しい順で返す。"""
    if not os.path.exists(COEFF_HISTORY_FILE):
        return []
    try:
        records = [json.loads(l) for l in open(COEFF_HISTORY_FILE, encoding="utf-8") if l.strip()]
        return list(reversed(records))
    except Exception:
        return []


def get_score_weights():
    """
    借手/物件・総合/定性の重みを返す。(w_borrower, w_asset, w_quant, w_qual)。
    優先順位: 手動設定 (coeff_overrides.json) > 自動最適化 (coeff_auto.json) > デフォルト値
    """
    auto     = load_auto_coeffs()
    manual   = load_coeff_overrides() or {}
    # score_weights キー（手動）vs _auto_weight_* キー（自動）を統合
    sw = manual.get("score_weights") or {}
    # 借手/物件重み: 手動 > 自動 > デフォルト
    w_b  = sw.get("borrower") or auto.get("_auto_weight_borrower")
    w_a  = sw.get("asset")    or auto.get("_auto_weight_asset")
    w_q  = sw.get("quant")    or auto.get("_auto_weight_quant")
    w_q2 = sw.get("qual")     or auto.get("_auto_weight_qual")
    if w_b is not None and w_a is not None and (w_b + w_a) > 0:
        s_ba = w_b + w_a
        w_borrower, w_asset = w_b / s_ba, w_a / s_ba
    else:
        w_borrower, w_asset = DEFAULT_WEIGHT_BORROWER, DEFAULT_WEIGHT_ASSET
    if w_q is not None and w_q2 is not None and (w_q + w_q2) > 0:
        s_qq = w_q + w_q2
        w_quant, w_qual = w_q / s_qq, w_q2 / s_qq
    else:
        w_quant, w_qual = DEFAULT_WEIGHT_QUANT, DEFAULT_WEIGHT_QUAL
    return (w_borrower, w_asset, w_quant, w_qual)


def get_model_blend_weights():
    """
    ① 全体モデル / ② 指標モデル / ③ 業種別モデル の混合重みを返す。
    優先順位: 手動設定 (coeff_overrides.json) > 自動最適化 (coeff_auto.json) > デフォルト (0.5/0.3/0.2)
    戻り値: (w_main, w_bench, w_ind) — 合計 1.0
    """
    _DEFAULT_MAIN  = 0.5
    _DEFAULT_BENCH = 0.3
    _DEFAULT_IND   = 0.2
    auto   = load_auto_coeffs()
    manual = load_coeff_overrides() or {}
    mw = manual.get("model_blend_weights") or {}
    w_m  = mw.get("main")  or auto.get("_auto_blend_w_main")
    w_b  = mw.get("bench") or auto.get("_auto_blend_w_bench")
    w_i  = mw.get("ind")   or auto.get("_auto_blend_w_ind")
    if w_m is not None and w_b is not None and w_i is not None:
        total = float(w_m) + float(w_b) + float(w_i)
        if total > 0:
            return float(w_m) / total, float(w_b) / total, float(w_i) / total
    return _DEFAULT_MAIN, _DEFAULT_BENCH, _DEFAULT_IND


def get_effective_coeffs(key=None):
    """指定キーの係数セットを返す。オーバーライドがあればマージ。"""
    if key is None:
        key = "全体_既存先"
    overrides = load_coeff_overrides() or {}
    base_key = key
    if base_key not in COEFFS:
        base_key = key.replace("_既存先", "").replace("_新規先", "")
    base = dict(COEFFS.get(base_key, COEFFS["全体_既存先"]))
    if overrides.get(base_key):
        base.update(overrides[base_key])
    if overrides.get(key):
        base.update(overrides[key])
    return base


def append_case_news(record: dict):
    """案件ごとのニュースを1件追記。失敗時は False。"""
    if not record:
        return True
    try:
        data = dict(record)
        data.setdefault("saved_at", datetime.datetime.now().isoformat())
        with open(CASE_NEWS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_case_news(case_id: Optional[str] = None):
    """保存済みニュースを読み込む。case_id を指定するとその案件分だけ。"""
    if not os.path.exists(CASE_NEWS_FILE):
        return []
    records = []
    try:
        with open(CASE_NEWS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if case_id is not None and rec.get("case_id") != case_id:
                    continue
                records.append(rec)
    except Exception:
        return []
    return records


class CustomJSONEncoder(json.JSONEncoder):
    """
    Numpyの各種数値型やPandasの欠損値等をPython標準型に変換して
    JSONシリアライズ可能にするエンコーダ。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):  # np.nan や pd.NA の対応
            return None
        try:
            return super().default(obj)
        except Exception:
            return str(obj)

def save_case_log(data):
    """審査1件分のログをSQLiteに追記し、生成した案件IDを返す。失敗時は None。"""
    import sqlite3
    case_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    data["id"] = case_id
    data["timestamp"] = datetime.datetime.now().isoformat()
    data["final_status"] = "未登録"
    
    industry_sub = data.get("industry_sub", "")
    score, user_eq = None, None
    res = data.get("result", {})
    if isinstance(res, dict):
        score, user_eq = res.get("score"), res.get("user_eq")
        
    try:
        score_val = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_val = None

    try:
        user_eq_val = float(user_eq) if user_eq is not None else None
    except (TypeError, ValueError):
        user_eq_val = None
        
    try:
        if not os.path.exists(DB_PATH):
            from migrate_to_sqlite import init_db
            init_db()
            
        json_str = json.dumps(data, ensure_ascii=False, cls=CustomJSONEncoder)
        from contextlib import closing
        with closing(sqlite3.connect(DB_PATH)) as conn:
            # テーブルが存在しない場合は作成（DBファイルが存在してもテーブルが欠けているケースに対応）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS past_cases (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    industry_sub TEXT,
                    score REAL,
                    user_eq REAL,
                    final_status TEXT,
                    data TEXT
                )
            """)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO past_cases
                (id, timestamp, industry_sub, score, user_eq, final_status, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (case_id, data["timestamp"], industry_sub, score_val, user_eq_val, data["final_status"], json_str))
            conn.commit()
        return case_id
    except Exception as e:
        import traceback
        print(f"[Error in save_case_log (SQLite)]: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def update_case_field(case_id: str, key: str, value: object) -> bool:
    """指定された case_id のレコードに対して、[key] = value を追加・更新する（SQLite版）。"""
    import sqlite3
    if not case_id or not os.path.exists(DB_PATH):
        return False
        
    try:
        from contextlib import closing
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            # 該当レコードのdata(JSON文字列)を取得
            cursor.execute("SELECT data FROM past_cases WHERE id = ?", (case_id,))
            row = cursor.fetchone()
            if not row:
                return False

            data_json = row[0]
            try:
                case_data = json.loads(data_json)
            except (json.JSONDecodeError, ValueError):
                case_data = {}

            # JSON側の更新
            case_data[key] = value
            new_json_str = json.dumps(case_data, ensure_ascii=False, cls=CustomJSONEncoder)

            # もし特定のキー（ステータス等）が単独カラムにもあれば一緒に更新する
            update_cols = "data = ?"
            update_args = [new_json_str]

            if key == "final_status":
                update_cols += ", final_status = ?"
                update_args.append(str(value))
            elif key == "industry_sub":
                update_cols += ", industry_sub = ?"
                update_args.append(str(value))

            update_args.append(case_id)

            cursor.execute(f"UPDATE past_cases SET {update_cols} WHERE id = ?", tuple(update_args))
            conn.commit()
        return True
    except Exception as e:
        import traceback
        print(f"[Error in update_case_field (SQLite)]: {e}", file=sys.stderr)
        return False
