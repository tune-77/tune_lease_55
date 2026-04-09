"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Activity, Clock } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ComposedChart } from 'recharts';

export default function TimesFMPage() {
  const [activeTab, setActiveTab] = useState<'company'|'industry'|'rate'|'compare'>('industry');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [target, setTarget] = useState('建設業');
  
  useEffect(() => {
    triggerMebuki('guide', 'TimesFM時系列予測です！\n業種のトレンドや金利動向を先読みします！');
    fetchData('industry');
  }, []);

  const fetchData = async (tab: string) => {
      setLoading(true);
      setData(null);
      try {
          if(tab === 'industry') {
              const res = await axios.post("http://localhost:8000/api/timesfm/industry_trend", { industry: target, horizon_months: 24 });
              setData(res.data);
          } else if(tab === 'company') {
              // Stub parameter since we need a company name
              const res = await axios.post("http://localhost:8000/api/timesfm/company_score", { company_name: target || '株式会社テスト', horizon_months: 12 });
              setData(res.data);
          } else if(tab === 'rate') {
              const res = await axios.post("http://localhost:8000/api/timesfm/final_rate", { industry: target, horizon_months: 6 });
              setData(res.data);
          } else if(tab === 'compare') {
              const res = await axios.post("http://localhost:8000/api/timesfm/financial_paths", { company_name: target || '株式会社テスト', n_periods: 12 });
              setData(res.data);
          }
      } catch (err) {
          console.error(err);
      } finally {
          setLoading(false);
      }
  };

  const handleTabChange = (t: any) => {
      setActiveTab(t);
      if(t === 'industry' || t === 'rate') setTarget('建設業');
      else setTarget('株式会社ABC'); // demo generic name
      fetchData(t);
  };

  const executeSearch = () => {
      fetchData(activeTab);
  };

  const renderIndustryChart = () => {
      if(!data || !data.months_history) return null;
      const chartData = [];
      const histLen = data.months_history.length;
      const foreLen = data.months_forecast.length;
      
      for(let i=0; i<histLen; i++) {
         chartData.push({ month: data.months_history[i], history: data.avg_score_hist[i], forecast: null, low: null, high: null });
      }
      
      const lastHist = data.avg_score_hist[histLen-1];
      const lastLabel = data.months_history[histLen-1];
      
      for(let i=0; i<foreLen; i++) {
         const score = data.avg_score_fore[i];
         chartData.push({ month: data.months_forecast[i], history: null, forecast: score, low: score-3, high: score+3, range: [score-3, score+3] });
      }
      
      chartData.find(d => d.month === lastLabel)!.forecast = lastHist;
      
      return (
          <div className="w-full" style={{ height: '400px' }}>
              <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#334155" />
                      <XAxis dataKey="month" stroke="#e2e8f0" tick={{fill: '#94a3b8', fontSize: 10}} minTickGap={20} />
                      <YAxis stroke="#e2e8f0" tick={{fill: '#94a3b8'}} domain={[0, 100]} />
                      <Tooltip contentStyle={{backgroundColor: '#1e293b', border: 'none', color: '#fff'}} />
                      <Area type="monotone" dataKey="range" fill="#3b82f6" fillOpacity={0.2} stroke="none" name="不確実性帯" />
                      <Line type="monotone" dataKey="history" stroke="#4ade80" strokeWidth={2} dot={false} name="実績スコア" />
                      <Line type="monotone" dataKey="forecast" stroke="#60a5fa" strokeWidth={2} strokeDasharray="5 5" dot={false} name="予測トレンド" />
                  </ComposedChart>
              </ResponsiveContainer>
          </div>
      );
  };

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] bg-slate-900 border border-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500 rounded-3xl mt-4 mx-4">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-white flex items-center gap-3">
          <Clock className="w-8 h-8 text-fuchsia-500" />
          Foundation Model 時系列洞察
        </h1>
        <p className="text-slate-400 font-bold mt-2">TimesFM（Google Research）のゼロショット推論を用いた将来予測・シミュレーション基盤。</p>
      </div>

      <div className="flex gap-2 mb-6 border-b border-slate-800 pb-4">
          <button onClick={()=>handleTabChange('industry')} className={`px-6 py-2 rounded-xl font-bold transition-all ${activeTab==='industry'?'bg-fuchsia-600 text-white':'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>業種トレンド予測</button>
          <button onClick={()=>handleTabChange('company')} className={`px-6 py-2 rounded-xl font-bold transition-all ${activeTab==='company'?'bg-fuchsia-600 text-white':'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>個別スコア予測</button>
          <button onClick={()=>handleTabChange('rate')} className={`px-6 py-2 rounded-xl font-bold transition-all ${activeTab==='rate'?'bg-fuchsia-600 text-white':'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>成約金利推移</button>
          <button onClick={()=>handleTabChange('compare')} className={`px-6 py-2 rounded-xl font-bold transition-all ${activeTab==='compare'?'bg-fuchsia-600 text-white':'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}>GBM vs TimesFM</button>
      </div>

      <div className="flex gap-4 mb-8">
          <input 
              type="text" 
              className="px-4 py-3 bg-slate-800 border-none text-white rounded-xl w-64 focus:ring-2 focus:ring-fuchsia-500 outline-none font-bold"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              placeholder="ターゲットを入力..."
          />
          <button onClick={executeSearch} className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl flex items-center gap-2">
             {loading ? <Activity className="w-5 h-5 animate-spin" /> : '予測を実行'}
          </button>
      </div>

      <div className="bg-slate-950 p-6 rounded-2xl border border-slate-800/60 shadow-2xl">
          {loading ? (
             <div className="h-64 flex flex-col items-center justify-center text-slate-500"><Activity className="w-10 h-10 animate-spin mb-4 text-fuchsia-500" /> Computing Deep Representation...</div>
          ) : data?.error ? (
             <div className="h-64 flex flex-col items-center justify-center text-rose-500 font-bold">{data.error}</div>
          ) : activeTab === 'industry' ? (
             renderIndustryChart()
          ) : (
             <div className="p-8 text-center text-slate-400">
               <h3 className="text-xl text-white font-bold mb-4">データロード成功！ [ {data?.method || 'Method'} ]</h3>
               <p>（このタブの高度なグラフ描画処理は順次Reactコンポーネント化されます。右側のWalkthroughをご覧ください！）</p>
               <pre className="mt-8 text-left bg-slate-900 p-4 rounded-xl overflow-auto h-64 text-xs font-mono text-emerald-400 border border-slate-800">{JSON.stringify(data, null, 2)}</pre>
             </div>
          )}
      </div>
    </div>
  );
}
