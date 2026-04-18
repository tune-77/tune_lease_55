"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { LineChart, BarChart3, TrendingUp, Zap, Activity, Info } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, Area, AreaChart } from 'recharts';

export default function QuantitativePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '定量的な成約要因の分析画面です。\\nLightGBMというAIモデルが、どの項目が成約に効いているかを計算しました！');
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'}/api/analysis/quantitative`);
      setData(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '分析データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const getImportanceData = () => {
    if (!data?.lgb_importance) return [];
    return data.lgb_importance
      .map(([name, value]: [string, number]) => ({ name, value }))
      .sort((a: any, b: any) => b.value - a.value)
      .slice(0, 15);
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center gap-4">
        <Zap className="w-12 h-12 text-rose-500 animate-pulse" />
        <p className="text-slate-500 font-bold">LGBMモデルが特徴量を抽出中...</p>
      </div>
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
            <LineChart className="w-8 h-8 text-rose-500" />
            定量要因・ML分析 (LGBM)
          </h1>
          <p className="text-slate-500 font-bold mt-2">
            全 {data?.n_cases} 件のデータから、AIが「成約の決め手」となる変数をランキング化。
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-white border border-slate-200 px-6 py-3 rounded-2xl shadow-sm text-center">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Model Accuracy</div>
            <div className="text-2xl font-black text-rose-600">{(data?.accuracy_lgb * 100).toFixed(1)}%</div>
          </div>
          <div className="bg-white border border-slate-200 px-6 py-3 rounded-2xl shadow-sm text-center">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Model AUC</div>
            <div className="text-2xl font-black text-indigo-600">{data?.auc_lgb?.toFixed(3)}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-8">
        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-xl font-black text-slate-700 flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-rose-500" />
              LGBM 特徴量重要度 (Top 15)
            </h3>
            <div className="flex items-center gap-2 text-xs font-bold text-slate-400">
              <Info className="w-4 h-4" />
              値が大きいほど、AIが成約判断に利用した度合いが高いことを示します
            </div>
          </div>

          <div className="h-[500px]">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={getImportanceData()} layout="vertical">
                 <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                 <XAxis type="number" />
                 <YAxis dataKey="name" type="category" width={150} tick={{fontSize: 11, fontWeight: 'bold'}} />
                 <Tooltip 
                   cursor={{fill: 'rgba(244, 63, 94, 0.05)'}}
                   contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}
                 />
                 <Bar dataKey="value" fill="#f43f5e" radius={[0, 4, 4, 0]}>
                   {getImportanceData().map((entry: any, index: number) => (
                     <Cell key={`cell-${index}`} fillOpacity={1 - index * 0.05} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-slate-50 border border-slate-200 p-8 rounded-2xl mt-4">
           <h4 className="text-sm font-black text-slate-500 uppercase tracking-widest mb-4">AIモデルの構成とハイパーパラメータ</h4>
           <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">Ensemble Alpha</div>
                 <div className="text-slate-800 font-bold">{data?.ensemble_alpha}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">N Positive</div>
                 <div className="text-slate-800 font-bold">{data?.n_positive}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">N Negative</div>
                 <div className="text-slate-800 font-bold">{data?.n_negative}</div>
              </div>
              <div className="p-3 bg-white rounded-lg border border-slate-200">
                 <div className="text-slate-400 mb-1">Random Seed</div>
                 <div className="text-slate-800 font-bold">42</div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}