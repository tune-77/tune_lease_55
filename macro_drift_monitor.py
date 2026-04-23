import os
import sqlite3
import pandas as pd
import datetime

DB_PATH = "/Users/kobayashiisaoryou/clawd/tune_lease_55/data/lease_data.db"

def check_concept_drift(recent_days=7, threshold_points=3.0, min_recent_cases=3):
    """
    過去の全データと直近のデータの平均スコアを比較し、
    コンセプトドリフト（審査基準の陳腐化・マクロ環境の悪化）を検知する。
    """
    if not os.path.exists(DB_PATH):
        return {"is_drift": False, "message": "DB未作成"}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        # 過去の審査スコアとタイムスタンプを取得
        query = "SELECT timestamp, score FROM past_cases WHERE score IS NOT NULL"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < 10:
            return {"is_drift": False, "message": "データ蓄積待ち"}
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=recent_days)
        
        recent_df = df[df['timestamp'] >= cutoff_date]
        past_df = df[df['timestamp'] < cutoff_date]
        
        if len(recent_df) < min_recent_cases or len(past_df) < 5:
            return {"is_drift": False, "message": "直近のサンプル不足"}
            
        long_avg = past_df['score'].mean()
        short_avg = recent_df['score'].mean()
        diff = short_avg - long_avg
        
        is_drift = diff <= -threshold_points
        
        msg = f"異常なし (過去平均: {long_avg:.1f}点 / 直近{recent_days}日: {short_avg:.1f}点)"
        if is_drift:
            msg = f"直近{recent_days}日で平均スコアが大幅に悪化しています (過去平均より {diff:.1f}点 下落)。マクロ環境の変化（コンセプトドリフト）の可能性があります。"
            
        return {
            "is_drift": is_drift,
            "long_avg": long_avg,
            "short_avg": short_avg,
            "diff": diff,
            "message": msg
        }
        
    except Exception as e:
        return {"is_drift": False, "message": f"エラー: {str(e)}"}
