import os
import json

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_FILE_PATH = os.path.join(_SCRIPT_DIR, "data", "business_rules.json")

def load_business_rules() -> dict:
    """
    data/business_rules.json を読み込み辞書として返す。
    ファイルが存在しない場合はデフォルトの辞書を返す。
    """
    if not os.path.exists(RULES_FILE_PATH):
        # デフォルトのルール構造
        return {
            "thresholds": {
                "approval": 0.70,
                "review": 0.40
            },
            "score_modifiers": {
                "learning_model_reject_penalty_multiplier": 0.5,
                "capital_deficiency_penalty": -5.0
            },
            "industry_rules": {
                "default": {
                    "require_review_if_deficit": False
                },
                "construction": {
                    "require_review_if_deficit": True
                }
            },
            "custom_rules": []
        }
    
    try:
        with open(RULES_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading business rules: {e}")
        return {}

def save_business_rules(rules: dict) -> bool:
    """
    辞書を data/business_rules.json に書き込んで保存する。
    成功した場合はTrue、失敗した場合はFalseを返す。
    """
    os.makedirs(os.path.dirname(RULES_FILE_PATH), exist_ok=True)
    try:
        with open(RULES_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"Error saving business rules: {e}")
        return False

def evaluate_condition(value, op: str, threshold) -> bool:
    """ 数値と演算子から条件の真偽を判定する """
    if value is None or threshold is None:
        return False
    try:
        v = float(value)
        t = float(threshold)
        if op == "=" or op == "==": return v == t
        if op == ">": return v > t
        if op == "<": return v < t
        if op == ">=": return v >= t
        if op == "<=": return v <= t
    except (ValueError, TypeError):
        # 文字列比較等へのフォールバック（将来用）
        pass
    return False

def evaluate_custom_rules(custom_rules: list, context: dict) -> dict:
    """
    ユーザー定義のカスタムルールリストを評価し、適用結果を返す。
    context (dict): 現在の審査対象となる指標や業種のデータ
                   (例: {"industry": "D 建設業", "op_profit": -5000, "user_eq_ratio": 15.5, ...})
    戻り値: {
      "score_delta": -15,               # 加減点の合計
      "forced_status": "review",        # 強制変更ステータス("review", "reject", None)
      "applied_reasons": ["適用理由1"]  # 適用されたルールの説明文リスト
    }
    """
    result = {
        "score_delta": 0.0,
        "forced_status": None,
        "applied_reasons": []
    }
    
    if not custom_rules:
        return result
        
    for rule in custom_rules:
        # 1. 業種チェック
        target_obj_ind = rule.get("industry", "ALL")
        if target_obj_ind != "ALL":
            # context["industry"] の前方一致（"D" や "建設業" などを許容）などの簡易マッチ
            curr_ind = context.get("industry", "")
            if target_obj_ind not in curr_ind and curr_ind not in target_obj_ind:
                continue
                
        # 2. 条件（複数AND）チェック
        # 後方互換のため旧フォーマットからの変換もサポート
        conditions = rule.get("conditions", [])
        if not conditions:
            legacy_target = rule.get("condition_target")
            if legacy_target:
                conditions = [{
                    "target": legacy_target,
                    "op": rule.get("condition_op"),
                    "value": rule.get("condition_value")
                }]
            else:
                continue

        all_met = True
        reasons_parts = []
        for cond in conditions:
            var = cond.get("target")
            op = cond.get("op")
            val = cond.get("value")
            
            if var not in context:
                all_met = False
                break
                
            actual_value = context.get(var)
            if not evaluate_condition(actual_value, op, val):
                all_met = False
                break
                
            reasons_parts.append(f"{var} が {val} {op}")
        
        if all_met:
            # 3. アクション適用
            action = rule.get("action_type")
            val = rule.get("action_value")
            
            # 理由文の作成
            ind_str = "全業種" if target_obj_ind == "ALL" else target_obj_ind
            action_str = f"スコアを {val}点引く" if action == "deduct_score" else f"ステータスを {val} に強制"
            cond_str = " かつ ".join(reasons_parts)
            reason = f"【カスタムルール適用】{ind_str} で {cond_str} のため、{action_str}。"
            
            if action == "deduct_score":
                try:
                    result["score_delta"] -= abs(float(val)) # 常に入力値を減点として扱う
                except (ValueError, TypeError):
                    pass
            elif action == "force_status":
                # reject > review の優先順位等があれば制御するが、一旦上書き
                if str(val) in ["review", "要審議"]:
                    result["forced_status"] = "要審議"
                elif str(val) in ["reject", "否決"]:
                    result["forced_status"] = "否決"
                    
            result["applied_reasons"].append(reason)
            
    return result

def simulate_rules_on_past_cases(cases: list, rules: dict) -> dict:
    """
    過去の案件データ全件に対して、現在のルール（基本閾値＋カスタムルール）を適用し、
    判定ステータス（承認圏内/要審議/否決）がどのように変化するか集計する。
    """
    thresholds = rules.get("thresholds", {})
    approval_line = thresholds.get("approval", 0.70) * 100
    review_line = thresholds.get("review", 0.40) * 100
    custom_rules = rules.get("custom_rules", [])
    
    # 変化マトリクス [old_status][new_status] = count
    matrix = {
        "承認圏内": {"承認圏内": 0, "要審議": 0, "否決": 0},
        "要審議": {"承認圏内": 0, "要審議": 0, "否決": 0},
        "否決": {"承認圏内": 0, "要審議": 0, "否決": 0},
        "不明": {"承認圏内": 0, "要審議": 0, "否決": 0}
    }
    details = []
    
    for c in cases:
        inputs = c.get("inputs", {})
        res = c.get("result", {})
        if not isinstance(inputs, dict) or not isinstance(res, dict):
            continue
            
        old_status = c.get("final_status", "")
        if old_status not in matrix:
            old_status = "不明"
            
        # context 用データの構築（実際に審査時に構築しているものに極力近づける）
        context_data = {
            "industry": c.get("industry_major", ""),
            "nenshu": inputs.get("nenshu", 0),
            "op_profit": inputs.get("op_profit", 0),
            "ord_profit": inputs.get("ord_profit", 0),
            "net_income": inputs.get("net_income", 0),
            "net_assets": inputs.get("net_assets", 0),
            "total_assets": inputs.get("assets", 0),
            "user_eq_ratio": res.get("raw", {}).get("user_equity_ratio", 0),
            # 以下は inputs に依存するが簡易的にデフォルト値を置く
            "term": 60,
            "cost": 10000,
            "bank_credit": inputs.get("bank_credit", 0),
            "lease_credit": inputs.get("lease_credit", 0)
        }
        
        # オリジナルのスコア（ペナルティなどを除いた純粋なベーススコアと仮定するか、ログのスコアを使うか）
        # ここではログに残っている最終スコアからカスタムルールの影響を（もしあれば）取り除いたものは不明なため、
        # 簡易的に raw スコア等から再計算するか、記録されている score をベースとして扱う
        base_score = res.get("score", 0)
        
        cr_result = evaluate_custom_rules(custom_rules, context_data)
        new_score = base_score + cr_result["score_delta"]
        new_score = max(0, min(100, new_score))
        
        # 新ステータスの判定
        if cr_result.get("forced_status"):
            new_status = cr_result.get("forced_status")
        elif new_score < review_line:
            new_status = "否決"
        elif new_score >= approval_line:
            new_status = "承認圏内"
        else:
            new_status = "要審議"
            
        # ゆらぎ吸収
        if new_status not in ["承認圏内", "要審議", "否決"]:
            new_status = "要審議"
            
        matrix[old_status][new_status] += 1
        
        if old_status != new_status and old_status != "不明":
            details.append({
                "id": c.get("id", "Unknown"),
                "industry": context_data["industry"],
                "old_score": base_score,
                "new_score": new_score,
                "old_status": old_status,
                "new_status": new_status,
                "reasons": cr_result["applied_reasons"]
            })
            
    return {
        "matrix": matrix,
        "changed_cases": details,
        "total": len(cases)
    }

