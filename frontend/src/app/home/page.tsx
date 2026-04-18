"use client";
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { triggerMebuki } from '../../components/layout/FloatingMebuki';
import { PieChart, BarChart3, TrendingUp, Users, Target, Activity, CheckCircle, XCircle } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function HomeDashboard() {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // 画面マウント時にめぶきちゃんを更新
    triggerMebuki('guide', 'ホーム画面ですね！\n全社的な審査・成約の直近データを分析しました！');

    const fetchStats = async () => {
      try {
        const res = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/dashboard/stats`);
        setStats(res.data);
      } catch (err) {
        console.error("Failed to load dashboard stats", err);
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[calc(100vh-2rem)]">
        <div className="flex flex-col items-center">
          <Activity className="w-12 h-12 text-blue-500 animate-spin mb-4" />
          <h2 className="text-xl font-bold text-slate-500">データを集計中...</h2>
        </div>
      </div>
    );
  }

  const analysis = stats?.analysis;
  const recentCases = stats?.recent_cases || [];

  const avgScoreBorrower = (() => {
    const scores = (analysis?.closed_cases || [])
      .map((c: any) => c?.result?.score_borrower)
      .filter((v: any) => typeof v === 'number');
    return scores.length > 0 ? scores.reduce((a: number, b: number) => a + b, 0) / scores.length : null;
  })();

  return (
    <div className="p-8 min-h-[calc(100vh-2rem)] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="mb-12 relative overflow-hidden bg-gradient-to-r from-blue-600 to-indigo-700 rounded-[2.5rem] p-10 text-white shadow-2xl shadow-blue-500/20 group">
        <div className="relative z-10 max-w-2xl">
          <h1 className="text-4xl font-black mb-4 flex items-center gap-4">
            <span className="bg-white/20 p-3 rounded-2xl backdrop-blur-md">
              <PieChart className="w-8 h-8 text-white" />
            </span>
            リース審査アシスタント：めぶき
          </h1>
          <p className="text-blue-100 text-lg font-bold leading-relaxed mb-8 opacity-90">
            お疲れ様です！本日の審査状況と、蓄積された成約データをAIが徹底分析しました。<br/>
            最適な審査判断のためのインサイトをお届けします。
          </p>
          <div className="flex gap-4">
            <div className="bg-white/10 backdrop-blur-md border border-white/20 px-6 py-3 rounded-2xl">
              <div className="text-[10px] font-black uppercase tracking-widest text-blue-200">System Status</div>
              <div className="text-sm font-black flex items-center gap-2">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                ONLINE & SYNCED
              </div>
            </div>
            <div className="bg-white/10 backdrop-blur-md border border-white/20 px-6 py-3 rounded-2xl">
              <div className="text-[10px] font-black uppercase tracking-widest text-blue-200">AI Intelligence</div>
              <div className="text-sm font-black">TimesFM v1.2 Active</div>
            </div>
          </div>
        </div>

        {/* めぶきちゃんの画像 */}
        <div className="absolute right-0 bottom-0 h-full w-[40%] hidden lg:block select-none pointer-events-none">
          <div className="relative h-full w-full">
            <img 
              src="/mebuki.png" 
              alt="Mebuki" 
              className="absolute bottom-0 right-10 h-[110%] object-contain drop-shadow-[0_20px_50px_rgba(0,0,0,0.3)] group-hover:scale-105 transition-transform duration-700 ease-out"
            />
          </div>
        </div>
        
        {/* 装飾用背景 */}
        <div className="absolute top-[-20%] right-[-10%] w-96 h-96 bg-white/10 rounded-full blur-3xl"></div>
        <div className="absolute bottom-[-20%] left-[-10%] w-64 h-64 bg-blue-400/20 rounded-full blur-3xl"></div>
      </div>

      {!analysis ? (
        <div className="bg-amber-50 border border-amber-200 p-6 rounded-2xl flex items-start gap-4">
          <TrendingUp className="w-8 h-8 text-amber-500 shrink-0" />
          <div>
            <h3 className="font-bold text-amber-800 text-lg">成約データが不足しています</h3>
            <p className="text-amber-700 mt-1">成約データが5件以上貯まると、AIによる成約要因分析・実績集計がこの画面に表示されます。「結果登録」画面から最終ステータスを登録してください。</p>
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {/* KPIカード群 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                 <div>
                   <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">総成約数 (分析対象)</p>
                   <h3 className="text-4xl font-black text-slate-800 mt-2">{analysis.closed_count} <span className="text-lg font-bold text-slate-400">件</span></h3>
                 </div>
                 <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-full flex items-center justify-center">
                   <Target className="w-6 h-6" />
                 </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                 <div>
                   <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">成約平均 定性スコア</p>
                   <h3 className="text-4xl font-black text-indigo-700 mt-2">
                     {analysis.qualitative_summary?.avg_weighted?.toFixed(1) || "-"} 
                     <span className="text-lg font-bold text-slate-400"> / 100</span>
                   </h3>
                 </div>
                 <div className="w-12 h-12 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center">
                   <Users className="w-6 h-6" />
                 </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <div className="flex justify-between items-start">
                 <div>
                   <p className="text-sm font-bold text-slate-500 uppercase tracking-wider">成約平均 信用スコア</p>
                   <h3 className="text-4xl font-black text-emerald-600 mt-2">
                     {avgScoreBorrower !== null ? avgScoreBorrower.toFixed(1) : "-"}
                     <span className="text-lg font-bold text-slate-400"> %</span>
                   </h3>
                 </div>
                 <div className="w-12 h-12 bg-emerald-50 text-emerald-600 rounded-full flex items-center justify-center">
                   <BarChart3 className="w-6 h-6" />
                 </div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* 成約要因トップ3 */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
              <h3 className="font-bold text-slate-800 text-lg mb-4 flex items-center gap-2">
                <TrendingUp className="text-rose-500 w-5 h-5" />
                成約要因トップ3ドライバー
              </h3>
              <p className="text-xs text-slate-500 font-bold mb-6">成約に最も寄与している因子（回帰分析結果）</p>
              
              <div className="space-y-4">
                {analysis.top3_drivers?.map((d: any, index: number) => (
                  <div key={index} className="flex items-center gap-4 bg-slate-50 p-4 rounded-xl border border-slate-100">
                    <div className="w-8 h-8 shrink-0 bg-slate-800 text-white rounded-full flex items-center justify-center font-black">
                      {index + 1}
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-slate-700">{d.label}</div>
                      <div className="text-xs text-slate-500 mt-0.5">回帰係数: {(d.coef || 0).toFixed(4)}</div>
                    </div>
                    <div className={`text-sm font-bold px-3 py-1 rounded border ${
                      d.direction === 'プラス' ? 'bg-green-50 text-green-700 border-green-200' : 'bg-rose-50 text-rose-700 border-rose-200'
                    }`}>
                      {d.direction}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 成約案件 平均財務数値 */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200 h-96 flex flex-col">
              <h3 className="font-bold text-slate-800 text-lg mb-6 flex items-center gap-2">
                <BarChart3 className="text-blue-500 w-5 h-5" />
                成約案件の平均財務主要指標
              </h3>
              
              <div className="flex-1 overflow-auto border rounded-xl border-slate-200 bg-slate-50">
                <table className="w-full text-sm text-left">
                  <thead className="bg-slate-100/80 sticky top-0 font-bold text-slate-600">
                    <tr>
                      <th className="px-4 py-3 border-b">指標名</th>
                      <th className="px-4 py-3 border-b text-right">成約平均値</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.avg_financials && Object.entries(analysis.avg_financials)
                      .map(([k, v]: [string, any], i) => (
                      <tr key={k} className="border-b last:border-b-0 hover:bg-white transition-colors bg-white">
                        <td className="px-4 py-3 font-semibold text-slate-700">{k}</td>
                        <td className="px-4 py-3 text-right font-black text-indigo-700">
                          {typeof v === 'number' ? v.toLocaleString('ja-JP', { maximumFractionDigits: 1 }) : v}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 最近の案件一覧 */}
      <h3 className="font-black text-2xl text-slate-800 mt-12 mb-6 border-l-4 border-blue-600 pl-3">📋 最新の案件履歴</h3>
      
      {recentCases.length === 0 ? (
        <p className="text-slate-500 font-bold">まだ案件履歴がありません。審査画面からデータを入力・実行してください。</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {recentCases.slice(0, 10).map((c: any, i: number) => {
            const isClosed = c.final_status === "成約";
            const isLost = c.final_status === "失注";
            return (
              <div key={i} className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200 hover:shadow-md transition-shadow relative overflow-hidden group">
                <div className={`absolute top-0 left-0 w-2 h-full ${isClosed ? 'bg-green-500' : isLost ? 'bg-rose-500' : 'bg-slate-300'}`}></div>
                
                <div className="flex justify-between items-start mb-3 ml-2">
                  <div className="flex items-center gap-2">
                    {isClosed ? <CheckCircle className="text-green-500 w-5 h-5" /> : isLost ? <XCircle className="text-rose-500 w-5 h-5" /> : <Activity className="text-slate-400 w-5 h-5" />}
                    <span className="text-xs font-bold text-slate-400">{c.timestamp?.slice(0, 16)}</span>
                  </div>
                  <div className="text-xl font-black text-slate-800">
                    <span className="text-xs font-bold text-slate-400 mr-2">審査スコア</span>
                    {(c.result?.score ?? 0).toFixed(0)} <span className="text-sm">点</span>
                  </div>
                </div>
                
                <h4 className="font-bold text-lg text-slate-700 ml-2 mb-1">{c.industry_major || "不明な業種"} <span className="text-sm text-slate-500">- {c.industry_sub}</span></h4>
                <div className="flex items-center gap-3 ml-2 mt-3">
                  <span className={`text-[10px] uppercase font-black px-2 py-1 rounded bg-slate-100 ${isClosed ? 'text-green-700' : isLost ? 'text-rose-700' : 'text-slate-500'}`}>
                    {c.final_status || "未登録"}
                  </span>
                  <span className="text-xs font-bold text-slate-500 bg-slate-50 px-2 py-1 rounded border">
                    判定: {c.result?.hantei || "不明"}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
