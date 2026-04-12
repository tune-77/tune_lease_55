"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { PieChart, TrendingUp, Users, Target, Activity, DollarSign, Award } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from 'recharts';

export default function HistoryDashPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    triggerMebuki('guide', '過去の成約実績をAIが深掘り分析しました！\\nどんな案件が通りやすいか、一目でわかりますよ。');
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await axios.get("http://localhost:8000/api/analysis/contract_drivers");
      setData(res.data);
    } catch (err) {
      console.error(err);
      triggerMebuki('reject', '分析データの取得に失敗しました。');
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <div className="p-8 flex items-center justify-center min-h-screen">
      <div className="flex flex-col items-center gap-4">
        <Activity className="w-12 h-12 text-blue-500 animate-spin" />
        <p className="text-slate-500 font-bold">AIが成約要因を集計中...</p>
      </div>
    </div>
  );

  const radarData = data?.avg_financials ? Object.entries(data.avg_financials).map(([name, value]) => ({
    subject: name,
    A: value,
    fullMark: 1000000,
  })) : [];

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-8">
        <h1 className="text-3xl font-black text-slate-800 flex items-center gap-3">
          <PieChart className="w-8 h-8 text-sky-500" />
          履歴分析・成約ダッシュボード
        </h1>
        <p className="text-slate-500 font-bold mt-2">成約した {data?.closed_count} 件のデータを元に、AIが共通因子を特定しました。</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {data?.top3_drivers?.map((driver: any, i: number) => (
          <div key={i} className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
            <div className={`absolute top-0 left-0 w-1 h-full ${driver.direction === 'プラス' ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
            <div className="flex items-center gap-4 mb-4">
              <div className={`p-3 rounded-xl ${driver.direction === 'プラス' ? 'bg-emerald-50' : 'bg-rose-50'}`}>
                <Target className={`w-6 h-6 ${driver.direction === 'プラス' ? 'text-emerald-600' : 'text-rose-600'}`} />
              </div>
              <div>
                <div className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Top Driver #{i+1}</div>
                <div className="text-lg font-black text-slate-700">{driver.label}</div>
              </div>
            </div>
            <div className="flex items-end justify-between">
              <div className={`text-sm font-bold ${driver.direction === 'プラス' ? 'text-emerald-600' : 'text-rose-600'}`}>
                {driver.direction}の寄与
              </div>
              <div className="text-2xl font-black text-slate-800">{Math.abs(driver.coef).toFixed(3)}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
          <h3 className="text-xl font-black text-slate-700 mb-6 flex items-center gap-2">
            <DollarSign className="w-5 h-5 text-emerald-500" />
            成約案件の平均財務モデル
          </h3>
          <div className="space-y-4">
            {data?.avg_financials && Object.entries(data.avg_financials).map(([name, value]: [any, any]) => (
              <div key={name} className="flex items-center justify-between p-3 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-sm font-bold text-slate-600">{name}</span>
                <span className="text-lg font-black text-slate-800">
                  {typeof value === 'number' ? (name.includes('%') ? value.toFixed(1) + '%' : value.toLocaleString() + 'k') : value}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm">
          <h3 className="text-xl font-black text-slate-700 mb-6 flex items-center gap-2">
            <Award className="w-5 h-5 text-amber-500" />
            強みタグの出現頻度 (集計)
          </h3>
          <div className="h-64">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={data?.tag_ranking?.slice(0, 8)} layout="vertical">
                 <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                 <XAxis type="number" hide />
                 <YAxis dataKey="0" type="category" width={100} tick={{fontSize: 10, fontWeight: 'bold'}} />
                 <Tooltip cursor={{fill: 'transparent'}} contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)'}} />
                 <Bar dataKey="1" fill="#3b82f6" radius={[0, 4, 4, 0]}>
                   {data?.tag_ranking?.map((entry: any, index: number) => (
                     <Cell key={`cell-${index}`} fill={index < 3 ? '#3b82f6' : '#94a3b8'} />
                   ))}
                 </Bar>
               </BarChart>
             </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
