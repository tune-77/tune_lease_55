import os
import json
import datetime
from tenacity import retry, wait_exponential, stop_after_attempt

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sumaho12 ではなくリポジトリルート (clawd 内) を参照
REPO_DIR = os.path.dirname(BASE_DIR)
JSIC_FILE = os.path.join(REPO_DIR, "industry_trends_jsic.json")
EXTENDED_TRENDS_FILE = os.path.join(REPO_DIR, "industry_trends_extended.json")
OUTPUT_FILE = os.path.join(REPO_DIR, "industry_reports_a4.json")

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_report_for_industry(industry_sub, basic_trend, web_trend):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Generating report for: {industry_sub}")
    
    prompt = f"""
あなたは中堅・中小企業向けのリース審査プロセスを支援する、プロフェッショナルなリサーチャー兼アナリストです。
以下の情報を基に、対象業種に関する「2025年 業界動向・課題・見通し」をまとめたA4サイズ（約1000〜1200文字程度）のレポートを作成してください。

【対象業種】
{industry_sub}

【基本動向（社内ナレッジ）】
{basic_trend}

【直近のウェブ検索結果（ニューススニペット等）】
{web_trend}

【出力要件】
1. レポートは最終的にPDFとしてそのまま添付される想定の、フォーマルかつ構造化されたビジネス文書としてください。
2. 以下の構成を見出しとして必ず含めること：
   - 「1. 2025年の業界見通しと総括」
   - 「2. 主要なビジネス課題・リスク要因」
   - 「3. 設備投資・リース需要の動向」
   - 「4. 審査・与信判断における着眼点」
3. 各セクションは具体的に記述し、無駄な引き延ばしは避けてください。文字数は全体で約1000〜1200字程度を目安とします。
4. Markdown見出し（# など）を見出しに過度に使用せず、シンプルに「■ 1. 2025年の業界見通しと総括」のような表記としてください（PDF変換時にプレーンテキストとして流し込める形式）。
5. 回答はレポート本文のみとし、前置きや後置きの挨拶は不要です。
"""

    model_name = os.environ.get("OLLAMA_MODEL", "lease-anna")
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling Ollama API for {industry_sub}: {e}")
        raise e
    
    return ""

def main():
    print("=== Start Generating Industry Reports ===")
    
    jsic_data = load_json(JSIC_FILE)
    extended_data = load_json(EXTENDED_TRENDS_FILE)
    
    # 既存の出力ファイルがあればロード
    output_data = load_json(OUTPUT_FILE)
    
    total_generated = 0
    
    for major_name, major_info in jsic_data.items():
        subs = major_info.get("sub", {})
        for sub_name, basic_trend in subs.items():
            # すでに今日のレポートが生成済みならスキップするロジック（オプション）
            if sub_name in output_data:
                cached = output_data[sub_name]
                if cached.get("generated_at", "").startswith(str(datetime.date.today())):
                    print(f"Skip (Already generated today): {sub_name}")
                    continue
            
            web_trend = extended_data.get(sub_name, {}).get("text", "")
            
            try:
                report_text = generate_report_for_industry(sub_name, basic_trend, web_trend)
                
                output_data[sub_name] = {
                    "generated_at": datetime.datetime.now().isoformat(),
                    "report_text": report_text
                }
                
                # 1件ごとに保存
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                    
                total_generated += 1
            except Exception as e:
                print(f"[ERROR] Failed for {sub_name}: {e}")
                
    print(f"=== Done. Total newly generated: {total_generated} ===")

if __name__ == "__main__":
    main()
