"""
案件・係数・重みの読み書きモジュール（lease_logic_sumaho10）
load_all_cases, save_all_cases, save_case_log, load_coeff_overrides, save_coeff_overrides,
get_score_weights, get_effective_coeffs, load_consultation_memory, append_consultation_memory,
load_case_news, append_case_news, find_similar_past_cases を提供。
st は使わず、保存失敗時は False/None を返す。呼び元で st.error 等を表示すること。
"""
import os
import sys
import json
import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from coeff_definitions import COEFFS
from charts import _equity_ratio_display

# ファイルパス（リポジトリルート基準）
CASES_FILE = os.path.join(_REPO_ROOT, "past_cases.jsonl")
COEFF_OVERRIDES_FILE = os.path.join(_REPO_ROOT, "data", "coeff_overrides.json")
CONSULTATION_MEMORY_FILE = os.path.join(_REPO_ROOT, "consultation_memory.jsonl")
CASE_NEWS_FILE = os.path.join(_REPO_ROOT, "case_news.jsonl")

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
    """過去案件を全件読み込み。"""
    if not os.path.exists(CASES_FILE):
        return []
    cases = []
    try:
        with open(CASES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return cases


def load_past_cases():
    """save_case_log で保存された過去の審査ログをすべて読み込む。"""
    return load_all_cases()


def find_similar_past_cases(selected_sub: str, user_equity_ratio: float, max_count: int = 3):
    """業界が同じで自己資本比率が近い過去案件を最大 max_count 件返す。"""
    cases = load_past_cases()
    candidates = []
    for c in cases:
        if c.get("industry_sub") != selected_sub:
            continue
        res = c.get("result") or {}
        eq = res.get("user_eq")
        if eq is None:
            continue
        try:
            eq_val = float(_equity_ratio_display(eq) or 0)
        except (TypeError, ValueError):
            continue
        diff = abs(eq_val - user_equity_ratio)
        status = c.get("final_status", "未登録")
        score = res.get("score")
        candidates.append({"diff": diff, "case": c, "equity": eq_val, "status": status, "score": score})
    candidates.sort(key=lambda x: x["diff"])
    return [x["case"] for x in candidates[:max_count]]


def save_all_cases(cases):
    """案件一覧を上書き保存。失敗時は False。"""
    try:
        with open(CASES_FILE, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def load_coeff_overrides():
    """保存済みの係数オーバーライドを読み込む。無ければ None。"""
    if not os.path.exists(COEFF_OVERRIDES_FILE):
        return None
    try:
        with open(COEFF_OVERRIDES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_coeff_overrides(overrides_dict):
    """係数オーバーライドを JSON で保存。失敗時は False。"""
    dirpath = os.path.dirname(COEFF_OVERRIDES_FILE)
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    try:
        with open(COEFF_OVERRIDES_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides_dict, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def get_score_weights():
    """借手/物件・総合/定性の重みを返す。(w_borrower, w_asset, w_quant, w_qual)。"""
    overrides = load_coeff_overrides() or {}
    sw = overrides.get("score_weights") or {}
    w_b = sw.get("borrower")
    w_a = sw.get("asset")
    w_q = sw.get("quant")
    w_q2 = sw.get("qual")
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


def load_case_news(case_id: str | None = None):
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


def save_case_log(data):
    """審査1件分のログを追記し、生成した案件IDを返す。失敗時は None。"""
    case_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    data["id"] = case_id
    data["timestamp"] = datetime.datetime.now().isoformat()
    data["final_status"] = "未登録"
    try:
        with open(CASES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        return case_id
    except Exception:
        return None
