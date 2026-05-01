import json
import os
import sqlite3
from datetime import datetime

# ── パス設定 ──────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
_DB_PATH = os.path.join(_REPO_ROOT, "data", "lease_data.db")

def run_tracking():
    print(f"[{datetime.now()}] 物件市場相場データ収集バッチ 開始")
    
    if not os.path.exists(_DB_PATH):
        print(f"❌ DBファイルが存在しません: {_DB_PATH}")
        return

    # APIキーの取得 (環境変数)
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        print("⚠️ GEMINI_API_KEY が設定されていません。統計モデルで実行します。")

    # 相場取得エンジンのインポート
    import sys
    if _REPO_ROOT not in sys.path:
        sys.path.append(_REPO_ROOT)
    
    try:
        from components.asset_score_detail import _search_scores
    except ImportError as e:
        print(f"❌ モジュールインポートエラー: {e}")
        return

    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    # 成約済み案件の抽出
    rows = cursor.execute(
        "SELECT id, data FROM past_cases WHERE final_status = '成約'"
    ).fetchall()

    print(f"🔍 追跡対象（成約済み）: {len(rows)} 件")
    success_count = 0

    for case_id, data_json in rows:
        try:
            data = json.loads(data_json or "{}")
        except Exception:
            continue

        inputs = data.get("inputs", {})
        if not inputs:
            continue

        # パラメータの解析
        asset_category = inputs.get("lease_asset_id", "other")
        asset_name = inputs.get("asset_name", inputs.get("asset_type", ""))
        lease_term = int(inputs.get("lease_term", 60))
        acquisition_cost = int(inputs.get("acquisition_cost", 0))

        # 詳細型番等がある場合は付与
        model_name = inputs.get("asd_model_name", "")
        search_query = f"{asset_name} {model_name}".strip() if model_name else asset_name

        print(f"👉 調査中: {case_id} | {search_query} ...")

        # 判定: 検索クエリが具体的か（2文字以下、または単一の一般名詞の場合は統計モデルへフォールバック）
        generic_keywords = {"車両", "自動車", "車", "営業車", "トラック", "パソコン", "pc", "機械", "ドローン", "建機", "重機"}
        is_generic = len(search_query) <= 2 or search_query in generic_keywords

        market_price_sen = 0
        
        if is_generic:
            print(f"   ℹ️ クエリが一般的すぎるため、統計的減価償却ロジックへフォールバックします。")
            # 簡易耐用年数マッピング (月次減価率)
            depreciation_rates = {
                "vehicle": 0.015, # 車両: 6年
                "medical": 0.012, # 医療: 7年
                "machinery": 0.010, # 工作機械: 10年
                "construction": 0.011, # 建機: 8年
                "pc": 0.020, # PC/IT: 4年
                "other": 0.015
            }
            rate = depreciation_rates.get(asset_category.lower(), 0.015)
            
            # 契約開始日からの経過月数を簡易計算 (デフォルト24ヶ月経過と仮定、実稼働時はタイムスタンプから算出)
            elapsed_months = 24 
            market_price_sen = int(acquisition_cost * ((1.0 - rate) ** elapsed_months))
        else:
            # Gemini Search Grounding 呼び出し
            result = _search_scores(
                category=asset_category,
                asset_name=search_query,
                lease_term_here=lease_term,
                acost_here=acquisition_cost,
                gemini_key=gemini_key
            )

            if result["error"]:
                print(f"   ❌ 調査失敗: {result['error']} (統計データで代替します)")
                rate = 0.015
                market_price_sen = int(acquisition_cost * ((1.0 - rate) ** 24))
            else:
                # 単位変換: 万円 → 千円
                market_price_man = result.get("current_price", 0)
                market_price_sen = market_price_man * 10  # 1万円 = 10千円

        # 履歴の保存 (ステップ2: データ収集のみ)
        cursor.execute("""
            INSERT INTO asset_price_history 
            (contract_id, inspected_at, current_market_price, residual_debt, profit_margin, is_alert_triggered)
            VALUES (?, ?, ?, NULL, NULL, 0)
        """, (
            case_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            market_price_sen
        ))
        success_count += 1

    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] バッチ完了。計 {success_count} 件のデータを収集・保存しました。")

if __name__ == "__main__":
    run_tracking()
