import os
import sys
import json
import datetime
import traceback
from typing import Optional, Dict, Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _SCRIPT_DIR
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_cases import load_all_cases
from ai_chat import chat_with_retry, get_ollama_model
from app_logger import log_warning, log_info, log_error

PDCA_RULES_FILE = os.path.join(_SCRIPT_DIR, "data", "pdca_ai_rules.json")

def load_pdca_rules() -> dict:
    """保存されたPDCA反映ルールを読み込む"""
    if not os.path.exists(PDCA_RULES_FILE):
        return {}
    try:
        with open(PDCA_RULES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"Failed to load PDCA rules: {e}", context="load_pdca_rules")
        return {}

def save_pdca_rules(data: dict) -> bool:
    """PDCA反映ルールを保存する"""
    try:
        os.makedirs(os.path.dirname(PDCA_RULES_FILE), exist_ok=True)
        with open(PDCA_RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        log_error(f"Failed to save PDCA rules: {e}", context="save_pdca_rules")
        return False

def format_cases_for_llm(cases: list) -> str:
    """LLMに読ませるための案件サマリー文字列を生成する"""
    lines = []
    for i, c in enumerate(cases):
        status = c.get("final_status", "不明")
        inputs = c.get("inputs", {})
        res = c.get("result", {})
        
        industry = c.get("industry_major", "") or c.get("industry_sub", "")
        eq_ratio = res.get("user_eq", inputs.get("user_eq_ratio", 0))
        score = res.get("score", 0)
        hantei = res.get("hantei", "")
        nenshu = inputs.get("nenshu", 0)
        
        lines.append(f"案件{i+1}: 【結果】{status} | 業種:{industry} | AI判定:{hantei}({score:.1f}点) | 自己資本比率:{eq_ratio}% | 年商:{nenshu}千円")
    
    return "\n".join(lines)

def run_monthly_pdca_reflection(force: bool = False, max_cases: int = 20) -> Optional[Dict[str, Any]]:
    """
    直近の案件データをLLMに渡し、月次の審査傾向の振り返りを実行する。
    抽出されたルールは pdca_ai_rules.json に保存される。
    """
    all_cases = load_all_cases()
    # 未登録以外の案件を抽出（成約、失注、デフォルトなど確実に結果が出ているもの）
    resolved_cases = [c for c in all_cases if c.get("final_status") and c.get("final_status") != "未登録"]
    
    # タイムスタンプで降順（新しい順）にソート
    resolved_cases.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    target_cases = resolved_cases[:max_cases]
    
    if len(target_cases) < 5 and not force:
        log_info("PDCA Reflection skipped: Not enough resolved cases (minimum 5 required).")
        return {"status": "skipped", "reason": "Not enough cases"}

    cases_text = format_cases_for_llm(target_cases)
    
    prompt = f"""あなたはリース審査システムの「AI審査マネージャー」です。
システムが自動的に学習（PDCA）を行うため、直近のリース案件の結果一覧を分析してください。

【直近{len(target_cases)}件の審査案件データ】
{cases_text}

【指示】
上記データを分析し、「失注」や「デフォルト（もしあれば）」、「スコアが低いのに成約した」などの傾向を見つけてください。
その分析結果を踏まえ、明日からのAI審査員（軍師AIなど）のプロンプトに動的に追加すべき「具体的な審査上の注意書きルール」を1〜3つ作成してください。

【出力フォーマット】
以下のJSON構造のみを出力してください。Markdownのコードブロック(```json)は含めず、純粋なJSONテキストのみを出力してください。

{{
  "reflection_summary": "今月の傾向に関する定性的な分析コメント（200文字程度）",
  "ai_prompt_addons": [
    "追加ルール1（例: 今月は建設業での失注が多いため、建設業の案件は資金繰り指標を通常より厳しく評価すること）",
    "追加ルール2（もしあれば設定）"
  ]
}}
"""
    
    try:
        model = get_ollama_model()
        # ai_chat.chat_with_retry は st.session_state を参照するため、一部環境（直接実行）でエラーになる可能性あり
        # そのため、このスクリプトは Streamlit 経由 (components 内部など) で呼ばれる想定。
        # 単独実行時はモックの session_state が必要。
        import streamlit as st
        # dummy wrapper if not in context
        if not hasattr(st, "session_state") or not st.session_state:
             st.session_state = {"ai_engine": "gemini", "gemini_model": "gemini-2.0-flash"}

        ans = chat_with_retry(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            retries=2,
            timeout_seconds=90,
        )
        
        content = ((ans.get("message") or {}).get("content") or "").strip()
        if not content:
            raise ValueError("LLM returned empty content")
            
        # JSON部分だけを抽出する（LLMが余計な文字を入れた場合の防御）
        import re
        json_match = re.search(r'\\{.*\\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
            
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # Markdownバッククォートが含まれている場合は除去
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)
            
        if "reflection_summary" not in parsed or "ai_prompt_addons" not in parsed:
            raise KeyError("JSON missing required fields")
            
        # 保存する
        save_data = {
            "last_run": datetime.datetime.now().isoformat(timespec="seconds"),
            "analyzed_count": len(target_cases),
            "reflection_summary": parsed["reflection_summary"],
            "ai_prompt_addons": parsed["ai_prompt_addons"]
        }
        
        save_pdca_rules(save_data)
        log_info(f"PDCA Reflection completed successfully. {len(parsed['ai_prompt_addons'])} rules generated.")
        return {"status": "success", "data": save_data}
        
    except Exception as e:
        error_msg = f"PDCA Reflection failed: {e}\n{traceback.format_exc()}"
        log_error(error_msg, context="run_monthly_pdca_reflection")
        return {"status": "error", "reason": str(e)}

if __name__ == "__main__":
    # テスト用
    import streamlit as st
    if not hasattr(st, "session_state"):
        st.session_state = {}
    print("Running PDCA reflection test...")
    res = run_monthly_pdca_reflection(force=True, max_cases=10)
    print(json.dumps(res, ensure_ascii=False, indent=2))
