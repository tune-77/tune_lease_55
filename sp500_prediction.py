import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

def predict_sp500():
    print("データを取得中...")
    
    # 1. S&P500 (^GSPC) と ドル円 (JPY=X) のデータを取得
    # "Close" カラムだけを取り出してSeriesにする (ilocやsqueezeを使う)
    # yfinanceの仕様変更でMultiIndexになることがあるため、確実にSeriesにする
    
    sp500_df = yf.download("^GSPC", period="2y", interval="1d")
    usdjpy_df = yf.download("JPY=X", period="2y", interval="1d")
    
    # 'Close'列を取得し、もしMultiIndexならレベルを下げる
    if isinstance(sp500_df.columns, pd.MultiIndex):
        sp500 = sp500_df['Close'].iloc[:, 0]
    else:
        sp500 = sp500_df['Close']
        
    if isinstance(usdjpy_df.columns, pd.MultiIndex):
        usdjpy = usdjpy_df['Close'].iloc[:, 0]
    else:
        usdjpy = usdjpy_df['Close']

    # インデックス（日付）を揃えて結合
    data = pd.DataFrame({'SP500_USD': sp500, 'USDJPY': usdjpy}).dropna()
    
    # 2. 円建て価格を計算 (S&P500 * ドル円レート)
    data['SP500_JPY'] = data['SP500_USD'] * data['USDJPY']
    
    print("データ取得・計算完了。予測モデルを作成中...")

    data = data.reset_index()
    data['Date_Ordinal'] = data['Date'].map(datetime.datetime.toordinal)

    # 3. 円建て価格の予測モデル作成
    # シミュレーション: 為替が130円になった場合
    simulated_rate = 130.0
    
    # 直近のS&P500価格を取得
    current_sp500 = data['SP500_USD'].iloc[-1]
    current_jpy_value = data['SP500_JPY'].iloc[-1]
    
    # 130円になった場合の価格
    simulated_value = current_sp500 * simulated_rate
    
    # 下落率
    drop_rate = (simulated_value - current_jpy_value) / current_jpy_value * 100

    print(f"\n--- 為替シミュレーション (1ドル = {simulated_rate}円) ---")
    print(f"現在の円建て価格: {int(current_jpy_value):,} 円")
    print(f"130円になった場合: {int(simulated_value):,} 円")
    print(f"影響: {drop_rate:.1f}% の下落になります。")

    X = data[['Date_Ordinal']]
    y = data['SP500_JPY']

    model = LinearRegression()
    model.fit(X, y)

    # 未来30日分の日付
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 31)]
    future_ordinals = [[d.toordinal()] for d in future_dates]

    predictions = model.predict(future_ordinals)

    print("予測完了。グラフを描画中...")

    # 4. グラフ描画（円建て）
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['SP500_JPY'], label='Actual (JPY)', color='green')
    plt.plot(future_dates, predictions, label='Forecast (JPY)', color='red', linestyle='--')
    
    plt.title('S&P 500 Price Prediction in JPY (Calculated with USD/JPY)')
    plt.xlabel('Date')
    plt.ylabel('Price (JPY)')
    
    # 金額を見やすく（カンマ区切り）
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    
    plt.legend()
    plt.grid(True)
    
    save_path = "/Users/kobayashiisaoryou/clawd/tune_lease_55/sp500_forecast_jpy.png"
    plt.savefig(save_path)
    print(f"グラフを保存しました: {save_path}")
    
    # 予測値の表示
    pred_start = predictions[0][0] if isinstance(predictions[0], (list, np.ndarray)) else predictions[0]
    pred_end = predictions[-1][0] if isinstance(predictions[-1], (list, np.ndarray)) else predictions[-1]

    print("\n--- 円建てS&P500 予測値（参考） ---")
    print(f"明日: {int(pred_start):,} 円")
    print(f"30日後: {int(pred_end):,} 円")

    plt.show()

if __name__ == "__main__":
    predict_sp500()
