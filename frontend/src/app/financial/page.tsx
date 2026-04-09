"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Layout, TrendingUp, DollarSign, Activity } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ComposedChart } from 'recharts';

const INDUSTRY_OPTIONS = [
  "建設業", "小売業", "製造業", "卸売業", "医療・福祉", "飲食・宿泊業", "サービス業", "不動産業", "情報通信業", "運輸・物流",
];

export default function FinancialPage() {
  const [sales, setSales] = useState([500000, 520000, 550000]);
  const [profit, setProfit] = useState([30000, 35000, 38000]);
  const [netAssets, setNetAssets] = useState([120000, 145000, 170000]);
  const [industry, setIndustry] = useState("サービス業");
  
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState<any>(null);

  useEffect(() => {
    triggerMebuki('guide', '3期財務分析ですね！\n過去の決算を入力するとAIが12ヶ月後まで予測します！');
  }, []);

  const handleUpdate = (type: 'sales' | 'profit' | 'netAssets', index: number, val: string) => {
    const num = parseInt(val) || 0;
    if (type === 'sales') {
        const newSales = [...sales]; newSales[index] = num; setSales(newSales);
    } else if (type === 'profit') {
        const newProfit = [...profit]; newProfit[index] = num; setProfit(newProfit);
    } else {
        const newNet = [...netAssets]; newNet[index] = num; setNetAssets(newNet);
    }
  };

  const runForecast = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/api/forecast", {
        sales, profit, net_assets: netAssets, industry
      });
      setForecastData(res.data);
      triggerMebuki('approve', `予測完了しました！\n今回は${res.data.timesfm_available ? 'TimesFM' : '推計モデル'}を使用しています！`);
    } catch (err) {
      console.error(err);
      triggerMebuki('error', '予測エンジンの通信に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  // グラフ用データ整形
  const formatChartData = (histLabels: string[], histVals: number[], foreLabels: string[], foreVals: number[]) => {
      const data = [];
      for(let i=0; i<histLabels.length; i++) {
          data.push({ label: histLabels[i], count: i, history: histVals[i], forecast: null, isPast: true });
      }
      // 连接部分
      const lastHistVal = histVals[histVals.length - 1];
      const beginForeLabel = histLabels[histLabels.length - 1];
      // Note: we can map the next months for forecast
      for(let i=0; i<foreLabels.length; i++) {
          data.push({ label: foreLabels[i], count: histLabels.length + i, history: null, forecast: foreVals[i], isPast: false });
      }
      // To connect the lines, we can add the last history to the forecast
      data.find(d => d.label === beginForeLabel)!.forecast = lastHistVal;
      return data;
  };

  const renderChart = (title: string, histKey: string, foreKey: string, color: string) => {
      if(!forecastData) return null;
      const chartData = formatChartData(
          forecastData.months_history, forecastData[histKey], 
          forecastData.months_forecast, forecastData[foreKey]
      );

      return (
          <div className="bg-white border border-slate-200 rounded-2xl shadow-sm p-6 mb-6">
              <h3 className="text-xl font-black text-slate-700 flex items-center gap-2 mb-4">
                  <TrendingUp className="w-5 h-5" style={{ color }} />
                  {title} 推移予測
              </h3>
              <div className="w-full" style={{ height: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="label" scale="point" padding={{ left: 10, right: 10 }} tick={{fontSize: 10}} minTickGap={30} />
                          <YAxis tickFormatter={(val) => Math.round(val / 1000) + 'm'} width={60} />
                          <Tooltip 
                            formatter={(value: any) => new Intl.NumberFormat('ja-JP').format(value) + ' 千円'}
                            contentStyle={{ borderRadius: '10px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                          />
                          <Legend />
                          <Line type="monotone" dataKey="history" stroke={color} strokeWidth={2} dot={false} name="実績" />
                          <Line type="monotone" dataKey="forecast" stroke={color} strokeWidth={2} strokeDasharray="5 5" dot={false} name="AI予測帯" />
                          <Area type="monotone" dataKey="forecast" fill={color} fillOpacity={0.1} stroke="none" />
                      </ComposedChart>
                  </ResponsiveContainer>
              </div>
          </div>
      );
  };

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <Layout className="w-8 h-8 text-emerald-500" />
          3期財務AI予測
        </h1>
        <p className="text-slate-500 font-bold mt-2">過去3期分のデータを元に、TimesFMを利用して月次の未来予測を生成します。（単位：千円）</p>
      </div>

      <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200 shadow-inner mb-8">
          <div className="grid grid-cols-4 gap-4 mb-4 items-end font-bold text-slate-500 text-sm">
             <div>科目</div>
             <div>3期前</div>
             <div>2期前</div>
             <div>直近期</div>
          </div>
          
          <div className="grid grid-cols-4 gap-4 mb-3 items-center">
             <div className="font-bold text-slate-700">売上高</div>
             {[0, 1, 2].map(i => <input key={'s'+i} type="number" className="border border-slate-300 p-3 rounded-lg w-full font-mono font-bold" value={sales[i]} onChange={e => handleUpdate('sales', i, e.target.value)} />)}
          </div>
          <div className="grid grid-cols-4 gap-4 mb-3 items-center">
             <div className="font-bold text-slate-700">営業利益</div>
             {[0, 1, 2].map(i => <input key={'p'+i} type="number" className="border border-slate-300 p-3 rounded-lg w-full font-mono font-bold" value={profit[i]} onChange={e => handleUpdate('profit', i, e.target.value)} />)}
          </div>
          <div className="grid grid-cols-4 gap-4 mb-6 items-center">
             <div className="font-bold text-slate-700">純資産</div>
             {[0, 1, 2].map(i => <input key={'n'+i} type="number" className="border border-slate-300 p-3 rounded-lg w-full font-mono font-bold" value={netAssets[i]} onChange={e => handleUpdate('netAssets', i, e.target.value)} />)}
          </div>

          <div className="flex gap-4 items-end">
              <div className="flex-1">
                  <label className="block text-sm font-bold text-slate-500 mb-2">業種季節性プリセット</label>
                  <select className="border border-slate-300 p-3 rounded-lg w-full bg-white font-bold" value={industry} onChange={(e) => setIndustry(e.target.value)}>
                      {INDUSTRY_OPTIONS.map(opt => <option key={opt}>{opt}</option>)}
                  </select>
              </div>
              <button 
                  onClick={runForecast}
                  disabled={loading}
                  className="bg-emerald-600 hover:bg-emerald-500 text-white font-black py-3 px-8 rounded-xl shadow-lg transition-all flex items-center justify-center gap-2"
              >
                  {loading ? <Activity className="w-6 h-6 animate-spin" /> : <DollarSign className="w-6 h-6" />}
                  {loading ? 'AI予測計算中...' : '未来グラフを生成'}
              </button>
          </div>
      </div>

      {forecastData && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
             {renderChart('売上高', 'sales_history', 'sales_forecast', '#3b82f6')}
             {renderChart('営業利益', 'profit_history', 'profit_forecast', '#ef4444')}
             {renderChart('純資産', 'net_assets_history', 'net_assets_forecast', '#10b981')}
          </div>
      )}
    </div>
  );
}
