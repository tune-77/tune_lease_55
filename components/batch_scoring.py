"""
バッチ審査コンポーネント。
CSV アップロード → 一括スコアリング → 結果テーブル表示・ダウンロード。
"""
import io
import streamlit as st
import pandas as pd
import numpy as np

from data_cases import get_effective_coeffs, get_score_weights
from constants import APPROVAL_LINE, REVIEW_LINE


# ─────────────────────────────────────────────────────────────────────────────
# CSVテンプレート
# ─────────────────────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    # ── 基本情報 ─────────────────────────────────────────────
    "取引先ID",
    "審査日",              # YYYY-MM-DD 形式 (例: 2024-03-15)。空欄可（空欄=登録日時で代替）
    "企業名",
    "業種大分類",
    "業種小分類",
    "取引区分",           # "既存先" or "新規先"
    "営業担当部署",
    "紹介元",             # "銀行紹介", "メーカー紹介", "ディーラー紹介", "その他"
    "検収時期(年)",
    # ── 物件情報 ─────────────────────────────────────────────
    "リース期間(月)",
    "取得価格(千円)",
    "物件ID（任意）",     # vehicle / it_equipment / medical / manufacturing 等。空欄可
    "物件名（任意）",
    "物件スコア（任意）",
    # ── 競合情報 ─────────────────────────────────────────────
    "競合状況",           # "競合なし", "競合あり"
    "競合提示金利(%)",
    "競合他社名",
    "契約種別",           # "一般", "メンテ付" 等
    # ── 財務数値（千円単位）──────────────────────────────────
    "売上高(千円)",
    "売上総利益(千円)",
    "営業利益(千円)",
    "経常利益(千円)",
    "当期純利益(千円)",
    "純資産(千円)",
    "総資産(千円)",
    "機械装置(千円)",
    "その他資産(千円)",
    "減価償却費(千円)",
    "減価償却累計(千円)",
    "支払リース料(千円)",
    "地代家賃(千円)",
    "銀行借入(千円)",
    "リース残高(千円)",
    "契約件数",
    "格付",               # "1-3", "4-6", "7-8", "9(要注意)", "10(破綻懸念)", "無格付"
    # ── 定性スコアリング（constants.py の QUALITATIVE_SCORING_CORRECTION_ITEMS と完全一致）
    # ※ 0〜4 の整数で入力。空欄可（未入力扱い）
    "定性_設立経営年数",   # 0:3年未満 / 1:3〜5年 / 2:5〜10年 / 3:10〜20年 / 4:20年以上         [重み10%]
    "定性_顧客安定性",     # 0:不安定・依存大 / 1:やや不安定 / 2:普通 / 3:安定 / 4:非常に安定   [重み20%]
    "定性_返済履歴",       # 0:問題あり / 1:遅延・リスケあり / 2:遅延少ない / 3:3年以上○ / 4:5年以上○ [重み25%]
    "定性_事業将来性",     # 0:懸念 / 1:やや懸念 / 2:普通 / 3:やや有望 / 4:有望               [重み15%]
    "定性_設備目的",       # 0:不明確 / 1:やや不明確 / 2:更新・維持 / 3:生産性向上 / 4:収益直結 [重み15%]
    "定性_メイン取引銀行", # 0:取引なし / 1:取引浅い / 2:サブ扱い / 3:メイン先 / 4:メイン先・支援表明 [重み15%]
    # ── 定性タグ・担当者コメント ─────────────────────────────
    "強みタグ",            # カンマ区切り: 技術力, 業界人脈, 特許, 立地, 後継者あり, 関係者資産あり, 取引行と付き合い長い, 既存返済懸念ない
    "担当者直感スコア(1-5)", # 空欄可（3=中立、5=非常に良い印象）
    "特記事項",            # 自由記述（熱意・補足説明など）
    # ── 結果（★モデル学習に最重要）──────────────────────────
    "最終結果",            # "成約" or "失注" ← これがあると係数が自動再学習される
]

# サンプルデータ（各行の値数は _CSV_COLUMNS の列数と一致させること）
_CSV_SAMPLE = pd.DataFrame([
    # 列順: 取引先ID, 審査日, 企業名, 業種大, 業種小, 取引区分, 部署, 紹介元, 検収年, 物件5, 競合4, 財務17, 定性6, タグ+直感+特記+結果4
    ["10001", "2024-06-10", "A建設株式会社", "D 建設業", "06 総合工事業", "既存先", "宇都宮営業部", "銀行紹介", 2026,
     60, 30000, "construction_machine", "油圧ショベル", 80,
     "競合なし", 0, "", "一般",
     500000, 100000, 20000, 15000, 10000, 80000, 300000, 10000, 5000, 5000, 20000, 3000, 2000, 50000, 10000, 3, "4-6",
     3, 3, 4, 3, 3, 3,
     "取引行と付き合い長い", 3, "順調な推移", "成約"],
    ["10002", "2024-09-22", "B工業株式会社", "E 製造業", "13 輸送用機械器具製造業", "新規先", "小山営業部", "メーカー紹介", 2026,
     48, 80000, "vehicle", "大型トラック", 70,
     "競合あり", 1.5, "XX運輸機器", "一般",
     1200000, 300000, 50000, 30000, 20000, 200000, 800000, 50000, 10000, 20000, 100000, 5000, 8000, 100000, 20000, 1, "1-3",
     2, 3, 3, 3, 3, 1,
     "", 4, "新規大口受注あり", "失注"],
    ["10003", "2025-01-15", "Cシステムズ", "G 情報通信業", "75 情報サービス業", "既存先", "足利営業部", "ディーラー紹介", 2026,
     36, 20000, "it_equipment", "サーバー", 60,
     "競合なし", 0, "", "一般",
     800000, 400000, 30000, 20000, 15000, 150000, 400000, 5000, 2000, 2000, 5000, 1000, 10000, 50000, 0, 5, "1-3",
     4, 4, 4, 3, 3, 4,
     "技術力,業界人脈", 5, "社長の技術力が高い", "成約"],
], columns=_CSV_COLUMNS)

def _get_csv_template() -> bytes:
    """テンプレートCSVをバイト列で返す。"""
    return _CSV_SAMPLE.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# 1件分のバッチスコア計算
# ─────────────────────────────────────────────────────────────────────────────

def _score_one(row: dict) -> dict:
    """
    1案件分のスコアを計算して返す。
    scoring_core の run_quick_scoring を呼び出して、個別審査と同一の判定結果を得る。
    さらにDB保存用に完全なJSON構造を生成して返す。
    """
    try:
        from scoring_core import run_quick_scoring
        
        inputs = {
            "nenshu": float(row.get("売上高(千円)") or 0),
            "gross_profit": float(row.get("売上総利益(千円)") or 0),
            "op_profit": float(row.get("営業利益(千円)") or 0),
            "ord_profit": float(row.get("経常利益(千円)") or 0),
            "net_income": float(row.get("当期純利益(千円)") or 0),
            "net_assets": float(row.get("純資産(千円)") or 0),
            "total_assets": float(row.get("総資産(千円)") or 0),
            "machines": float(row.get("機械装置(千円)") or 0),
            "other_assets": float(row.get("その他資産(千円)") or 0),
            "dep_expense": float(row.get("減価償却費(千円)") or 0),
            "depreciation": float(row.get("減価償却累計(千円)") or 0),
            "rent_expense": float(row.get("支払リース料(千円)") or 0),
            "rent": float(row.get("地代家賃(千円)") or 0),
            "bank_credit": float(row.get("銀行借入(千円)") or 0),
            "lease_credit": float(row.get("リース残高(千円)") or 0),
            "contracts": int(row.get("契約件数") or 0),
            "grade": str(row.get("格付") or "1-3"),
            "customer_type": str(row.get("取引区分") or "既存先"),
            "industry_major": str(row.get("業種大分類") or "D 建設業"),
            "industry_sub": str(row.get("業種小分類") or "06 総合工事業"),
            "sales_dept": str(row.get("営業担当部署") or "未設定"),
            "main_bank": str(row.get("メイン取引銀行") or "なし"),
            "competitor": str(row.get("競合状況") or "競合なし"),
            "competitor_rate": float(row.get("競合提示金利(%)") or 0) / 100.0,
            "contract_type": str(row.get("契約種別") or "一般"),
            "deal_source": str(row.get("紹介元") or "その他"),
            "lease_term": int(row.get("リース期間(月)") or 60),
            "acceptance_year": int(row.get("検収時期(年)") or 2026),
            "acquisition_cost": float(row.get("取得価格(千円)") or 0),
            "lease_asset_id": str(row.get("物件ID（任意）") or "").strip(),
            "lease_asset_name": str(row.get("物件名（任意）") or "").strip(),
            "intuition_score": float(row.get("担当者直感スコア(1-5)") or 0),
        }
        
        passion_text = str(row.get("特記事項") or "")
        inputs["passion_text"] = passion_text
        
        acq_cost = inputs["acquisition_cost"]
        lease_term = inputs["lease_term"]
        asset_id_raw = inputs["lease_asset_id"]

        # ── 物件スコア ───────────────────────────────────────────
        asset_category = None
        asset_grade_label = "—"
        scoring_method = "簡易"
        asset_score = float(row.get("物件スコア（任意）") or 0)
        
        if asset_score <= 0:
            asset_score = 50.0

        if asset_id_raw and asset_id_raw.lower() not in ("", "nan", "none"):
            try:
                from category_config import ASSET_ID_TO_CATEGORY
                from asset_scorer import calc_asset_score

                asset_category = ASSET_ID_TO_CATEGORY.get(asset_id_raw)
                if asset_category:
                    contract = {"lease_months": lease_term}
                    asset_result = calc_asset_score(asset_category, {}, contract)
                    asset_score  = asset_result["total_score"]
                    asset_grade_label = asset_result["grade"]
                    scoring_method = "標準"
                else:
                    raise ValueError(f"未登録のasset_id: {asset_id_raw}")
            except Exception:
                asset_category = None
                asset_id_raw   = ""

        if not asset_category and asset_score == 50.0:
            term_ok     = 1.0 if 36 <= lease_term <= 72 else 0.6
            cost_ok     = 1.0 if 500 < acq_cost < 50000 else 0.7
            asset_score = (term_ok + cost_ok) / 2.0 * 100

        inputs["asset_score"] = asset_score
        inputs["lease_asset_score"] = asset_score
        
        # run_quick_scoring 呼び出し
        res = run_quick_scoring(inputs)

        # PD概算 (格付からマッピング)
        grade_to_pd = {
            "1-3": 0.5, "4-6": 2.0, "7-8": 6.0, "9(要注意)": 15.0, "10(破綻懸念)": 40.0, "無格付": 5.0
        }
        pd_pct = grade_to_pd.get(inputs["grade"], 5.0)
        
        # DB保存用のJSON構成
        final_status_raw = str(row.get("最終結果") or "").strip()
        final_status = final_status_raw if final_status_raw in ("成約", "失注") else "未登録"

        # 審査日 → ISO形式に変換（空欄は save_case_log が登録日時で補完）
        _shinsa_date_raw = str(row.get("審査日") or "").strip()
        _shinsa_timestamp = None
        if _shinsa_date_raw:
            import re as _re
            # YYYY-MM-DD / YYYY/MM/DD / YYYYMMDD など柔軟に対応
            _d = _re.sub(r"[/．。]", "-", _shinsa_date_raw).replace("年", "-").replace("月", "-").replace("日", "")
            try:
                from datetime import datetime as _dt
                _parsed = _dt.strptime(_d.strip()[:10], "%Y-%m-%d")
                _shinsa_timestamp = _parsed.isoformat()
            except ValueError:
                _shinsa_timestamp = None  # パース失敗時は登録日時で代替

        db_data = {
            # ── 個別審査の log_payload キーと合わせること（save_case_log が依存）──
            "company_name":   str(row.get("企業名") or ""),    # 案件一覧表示用
            "company_no":     str(row.get("取引先ID") or ""),   # 取引先ID
            "borrower_name":  str(row.get("企業名") or ""),    # 後方互換
            "industry_major": inputs["industry_major"],
            "industry_sub":   inputs["industry_sub"],
            "customer_type":  inputs["customer_type"],
            "main_bank":      inputs["main_bank"],
            "competitor":     inputs["competitor"],
            "competitor_rate": inputs["competitor_rate"],
            "sales_dept":     inputs["sales_dept"],
            "final_status":   final_status,  # 成約/失注は上書きされずにDBへ
            "timestamp":      _shinsa_timestamp,  # 審査日があればISO形式で事前セット
            "inputs": inputs,
            "result": {
                **res,
                # save_case_log が res.get("user_eq") で自己資本比率を取得するため追加
                "user_eq": res.get("user_equity_ratio"),
            },
            "financials": {
                "nenshu":       inputs["nenshu"],
                "gross_profit": inputs["gross_profit"],
                "op_profit":    inputs["op_profit"],
                "ord_profit":   inputs["ord_profit"],
                "net_income":   inputs["net_income"],
                "net_assets":   inputs["net_assets"],
                "assets":       inputs["total_assets"],
                "machines":     inputs["machines"],
                "other_assets": inputs["other_assets"],
                "bank_credit":  inputs["bank_credit"],
                "lease_credit": inputs["lease_credit"],
                "depreciation": inputs["depreciation"],
            }
        }

        # ── 定性スコアリング構造を生成して db_data に追加 ──────────────
        # constants.py の QUALITATIVE_SCORING_CORRECTION_ITEMS と完全に一致した形式で格納
        _QUAL_MAP = [
            ("company_history",    "定性_設立経営年数",   10),
            ("customer_stability", "定性_顧客安定性",     20),
            ("repayment_history",  "定性_返済履歴",       25),
            ("business_future",    "定性_事業将来性",     15),
            ("equipment_purpose",  "定性_設備目的",       15),
            ("main_bank",          "定性_メイン取引銀行", 15),
        ]
        _LEVEL_LABELS = {
            "company_history":    ["3年未満", "3〜5年", "5〜10年", "10〜20年", "20年以上"],
            "customer_stability": ["不安定・依存大", "やや不安定", "普通", "安定", "非常に安定"],
            "repayment_history":  ["問題あり", "遅延・リスケあり", "遅延少ない", "3年以上○", "5年以上○"],
            "business_future":    ["懸念", "やや懸念", "普通", "やや有望", "有望"],
            "equipment_purpose":  ["不明確", "やや不明確", "更新・維持", "生産性向上", "収益直結"],
            "main_bank":          ["取引なし", "取引浅い", "サブ扱い", "メイン先", "メイン先・支援表明"],
        }
        qual_items = {}
        qual_weight_sum = 0
        qual_weighted_total = 0.0
        for qid, col, weight in _QUAL_MAP:
            raw = row.get(col, "")
            try:
                val = int(float(raw)) if str(raw).strip() not in ("", "nan") else None
            except (ValueError, TypeError):
                val = None
            if val is not None and 0 <= val <= 4:
                labels = _LEVEL_LABELS.get(qid, [])
                level_label = labels[val] if val < len(labels) else str(val)
                qual_items[qid] = {"value": val, "label": col.replace("定性_", ""), "weight": weight, "level_label": level_label}
                qual_weight_sum += weight
                qual_weighted_total += (val / 4.0) * 100 * (weight / 100.0)
            else:
                qual_items[qid] = {"value": None, "label": col.replace("定性_", ""), "weight": weight, "level_label": None}

        qual_weighted_score = round((qual_weighted_total / qual_weight_sum * 100) if qual_weight_sum > 0 else 0)
        qual_weighted_score = min(100, max(0, qual_weighted_score))

        # 強みタグ
        strength_tags_raw = str(row.get("強みタグ") or "")
        strength_tags = [t.strip() for t in strength_tags_raw.split(",") if t.strip()]
        passion_text_val = str(row.get("特記事項") or "")

        if qual_weight_sum > 0:
            db_data["result"]["qualitative_scoring_correction"] = {
                "items": qual_items,
                "weighted_score": qual_weighted_score,
            }
            db_data["inputs"]["qualitative_scoring"] = {
                "items": qual_items,
                "weighted_score": qual_weighted_score,
            }
        db_data["inputs"]["qualitative"] = {
            "strength_tags": strength_tags,
            "passion_text": passion_text_val,
        }

        return {
            "UI表示用": {
                "取引先ID":     db_data["company_no"],
                "企業名":       db_data["borrower_name"],
                "借手スコア":     res["score_borrower"],
                "物件スコア":     round(asset_score, 1),
                "物件グレード":   asset_grade_label,
                "物件カテゴリ":   asset_category or "—",
                "スコアリング":   scoring_method,
                "総合スコア":     res["score"],
                "PD概算(%)":      pd_pct,
                "自己資本比率(%)": round(res["user_equity_ratio"], 1),
                "営業利益率(%)":   round(res["user_op_margin"], 1),
                "判定":           res["hantei"],
                "エラー":         "",
            },
            "DB保存用": db_data
        }
    except Exception as e:
        import traceback
        return {
            "UI表示用": {
                "取引先ID": row.get("取引先ID", ""), "企業名": row.get("企業名", ""),
                "借手スコア": None, "物件スコア": None, "物件グレード": None,
                "物件カテゴリ": None, "スコアリング": None,
                "総合スコア": None, "PD概算(%)": None,
                "自己資本比率(%)": None, "営業利益率(%)": None,
                "判定": "エラー", "エラー": f"{e}",
            },
            "DB保存用": None
        }


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def render_batch_scoring():
    """バッチ審査 UI を描画する。"""
    st.title("⚡ バッチ審査（CSV一括判定）")
    st.info(
        "複数案件を CSV でアップロードして一括スコアリングします。\n\n"
        "**「物件ID（任意）」列を入力すると個別審査と同じ `calc_asset_score()` ロジックが適用されます**（標準モード）。"
        "空欄の場合は取得価格・リース期間ベースの簡易スコアを使用します（簡易モード）。\n\n"
        "対応物件ID: `vehicle` / `it_equipment` / `medical` / `manufacturing` / "
        "`construction_machine` / `office_furniture` / `restaurant` / `renewable` / `other` 等"
    )

    # テンプレートダウンロード
    st.download_button(
        "📄 CSV テンプレートをダウンロード",
        data=_get_csv_template(),
        file_name="batch_shinsa_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("CSV をアップロード", type=["csv"], key="batch_csv_upload")
    if uploaded is None:
        st.caption("上記テンプレートを参考に CSV を作成してアップロードしてください。")
        return

    try:
        df_in = pd.read_csv(uploaded, encoding="utf-8-sig")
    except Exception:
        try:
            uploaded.seek(0)
            df_in = pd.read_csv(uploaded, encoding="shift_jis")
        except Exception as e:
            st.error(f"CSV 読み込みエラー: {e}")
            return
            
    # OCR等の空欄（NaN）を空文字に変換（falsy判定でデフォルト値へ安全にフォールバックさせるため）
    df_in = df_in.fillna("")

    # 必須列チェック
    missing_cols = [c for c in ["売上高(千円)", "総資産(千円)"] if c not in df_in.columns]
    if missing_cols:
        st.error(f"必須列が不足しています: {missing_cols}")
        return

    st.success(f"{len(df_in)} 件を読み込みました。")
    st.dataframe(df_in.head(5), width='stretch')
    
    save_to_db = st.checkbox("💾 判定後に結果をデータベース（過去案件）へ保存・蓄積する", value=False)

    if save_to_db:
        try:
            from backup_manager import get_last_backup_time
            last_ts = get_last_backup_time()
            if last_ts:
                st.info(f"🛡️ 最終バックアップ: **{last_ts}** — DB保存前に自動でバックアップを取ります")
            else:
                st.warning("⚠️ バックアップ未実施 — DB保存前に自動でバックアップを取ります")
        except Exception:
            pass

    if st.button("🚀 一括スコアリング実行", type="primary"):
        ui_results = []
        db_results = []
        prog = st.progress(0, text="スコアリング中...")
        for i, (_, row) in enumerate(df_in.iterrows()):
            out = _score_one(row.to_dict())
            ui_results.append(out["UI表示用"])
            if out.get("DB保存用"):
                db_results.append(out["DB保存用"])
            prog.progress((i + 1) / len(df_in), text=f"{i+1}/{len(df_in)} 件処理中...")
        prog.empty()

        df_out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(ui_results)], axis=1)

        # データベース保存処理
        if save_to_db and db_results:
            # ── DB保存前に必ずバックアップを取得 ──────────────────────────
            try:
                from backup_manager import run_backup
                bk = run_backup(force=True)  # force=True で即時バックアップ
                bk_files = [b["file"] for b in bk.get("backed_up", [])]
                if bk_files:
                    st.success(f"🛡️ バックアップ完了: {', '.join(bk_files)}")
                else:
                    st.info("🛡️ バックアップ: 最新版が既に存在するためスキップ")
            except Exception as bk_err:
                st.warning(f"⚠️ バックアップに失敗しました（保存は続行）: {bk_err}")
            # ─────────────────────────────────────────────────────────────

            from data_cases import save_case_log
            saved_count = 0
            with_result = 0
            for db_data in db_results:
                if save_case_log(db_data):
                    saved_count += 1
                    if db_data.get("final_status") in ("成約", "失注"):
                        with_result += 1
            if saved_count > 0:
                st.success(f"✅ {saved_count} 件をデータベースに保存しました（うち成約/失注あり: {with_result} 件）")
            
            # 成約/失注データが十分あれば係数自動再学習を実行
            if with_result > 0:
                try:
                    from auto_optimizer import run_auto_optimization, get_training_status
                    status = get_training_status()
                    if status["should_retrain"]:
                        with st.spinner(f"🧠 {status['count']} 件の実績データで係数を再学習中..."):
                            opt_result = run_auto_optimization(force=False)
                        if opt_result:
                            ab = opt_result.get("ab_test_result", {})
                            if ab.get("passed"):
                                st.success(f"🎉 係数自動更新完了！ {ab.get('reason', '')}")
                            else:
                                st.info(f"係数更新見送り（精度改善なし）: {ab.get('reason', '')}")
                    else:
                        st.info(f"📊 成約/失注データ蓄積中... 次回学習まであと {status['next_trigger']} 件")
                except Exception as e:
                    st.caption(f"自動学習スキップ: {e}")

        # サマリー表示
        col1, col2, col3 = st.columns(3)
        total = len(df_out)
        col1.metric("良決", f"{(df_out['判定'] == '良決').sum() if '良決' in df_out['判定'].values else (df_out['判定'] == '承認圏内').sum()}件 / {total}件")
        col2.metric("ボーダー", f"{(df_out['判定'] == 'ボーダー').sum() if 'ボーダー' in df_out['判定'].values else (df_out['判定'] == '要審議').sum()}件 / {total}件")
        col3.metric("否決", f"{(df_out['判定'] == '否決').sum()}件 / {total}件")

        # 結果テーブル
        st.subheader("判定結果")

        # スコアリングモード別件数を表示
        n_standard = (df_out.get("スコアリング") == "標準").sum() if "スコアリング" in df_out else 0
        n_simple   = len(df_out) - n_standard
        st.caption(
            f"🟢 標準モード（calc_asset_score 使用）: {n_standard}件　"
            f"⬜ 簡易モード（フォールバック）: {n_simple}件"
        )

        def _color_hantei(val):
            if val in ["良決", "承認圏内"]:
                return "background-color: #d4edda; color: #155724"
            elif val in ["ボーダー", "要審議"]:
                return "background-color: #fff3cd; color: #856404"
            elif val == "否決":
                return "background-color: #f8d7da; color: #721c24"
            return ""

        def _color_scoring(val):
            if val == "標準":
                return "background-color: #e8f5e9; color: #2e7d32"
            return "color: #9e9e9e"

        styled = (
            df_out.style
            .applymap(_color_hantei, subset=["判定"])
            .applymap(_color_scoring, subset=["スコアリング"])
        )
        st.dataframe(styled, width='stretch', hide_index=True)

        # ダウンロード
        csv_out = df_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "📥 結果 CSV をダウンロード",
            data=csv_out,
            file_name="batch_shinsa_result.csv",
            mime="text/csv",
        )
