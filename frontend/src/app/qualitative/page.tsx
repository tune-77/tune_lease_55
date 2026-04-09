"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { Target, MessageSquare, Award, BarChart3, Activity } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';

export default function QualitativePage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '定性的な要因の分析です。\\n強みタグや、メイン取引かどうかが成約にどう響いているかを見れます。');
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:8000/api/analysis/qualitative");
      setData(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('error', '分析データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  const getCoefData = () => {
    if (!data?.lr_coef) return [];
    return data.lr_coef
      .map(([name, value]: [string, number]) => ({ name, value }))
      .sort((a: any, b: any) => Math.abs(b.value) - Math.abs(a.value));
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center gap-4">
        <MessageSquare className="w-12 h-12 text-pink-500 animate-bounce" />
        <p className="text-slate-500 font-bold">定性データをAIが分類計数中...</p>
      </div>
    </div>
  );

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8 flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
            <Target className="w-8 h-8 text-pink-500" />
            定性要因・成約寄与分析
          </h1>
          <p className="text-slate-500 font-bold mt-2">
            強みタグ・取引背景・社長の想いなど、数値化しにくい因子の成約寄与度。
          </p>
        </div>
        <div className="bg-white border border-slate-200 px-6 py-3 rounded-2xl shadow-sm text-center">
            <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Analysis Cases</div>
            <div className="text-2xl font-black text-pink-600">{data?.n_cases} 件</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
          <h3 className="text-xl font-black text-slate-700 mb-8 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-pink-500" />
            定性因子の回帰係数 (寄与度)
          </h3>
          <div className="h-[500px]">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={getCoefData()} layout="vertical">
                 <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                 <XAxis type="number" />
                 <YAxis dataKey="name" type="category" width={150} tick={{fontSize: 11, fontWeight: 'bold'}} />
                 <Tooltip 
                   cursor={{fill: 'rgba(236, 72, 153, 0.05)'}}
                   contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}}
                 />
                 <Bar dataKey="value" name="回帰係数" radius={[0, 4, 4, 0]}>
                   {getCoefData().map((entry: any, index: number) => (
                     <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ec4899' : '#94a3b8'} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-6">
           <div className="bg-pink-50 border border-pink-100 p-8 rounded-2xl">
              <h4 className="text-lg font-black text-pink-800 mb-4 flex items-center gap-2">
                 <Award className="w-6 h-6" />
                 AIの洞察
              </h4>
              <p className="text-pink-900 leading-relaxed font-bold">
                 「{getCoefData()[0]?.name}」が成約に対して最も強い{getCoefData()[0]?.value > 0 ? '正' : '負'}の影響を与えています。
                 {getCoefData()[0]?.value > 0 
                   ? 'この項目を商談で強調することが成約への近道です。' 
                   : 'この項目が懸念される場合は、早めに対策を打つ必要があります。'}
              </p>
           </div>
           
           <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
              <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-4">Regression Accuracy</div>
              <div className="flex items-end justify-between">
                 <div className="text-3xl font-black text-slate-800">{(data?.accuracy_lr * 100).toFixed(1)}%</div>
                 <div className="text-sm font-bold text-slate-500">{(data?.auc_lr * 100).toFixed(1)} AUD-ROC</div>
              </div>
           </div>
           
           <div className="bg-slate-900 p-8 rounded-2xl shadow-2xl overflow-hidden relative group">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-30 transition-opacity">
                 <Activity className="w-24 h-24 text-white" />
              </div>
              <h4 className="text-white font-black mb-2">アンサンブル最適化</h4>
              <p className="text-slate-400 text-sm mb-4">LRとLGBMを {data?.ensemble_alpha} : { (1-data?.ensemble_alpha).toFixed(1) } の比率で混合した際の精度</p>
              <div className="text-2xl font-black text-emerald-400">{(data?.accuracy_ensemble * 100).toFixed(1)}%</div>
           </div>
        </div>
      </div>
    </div>
  );
}